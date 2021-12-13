/// Symbolic flat memory
///
/// Note: the API exposed for this backing is actually concolic. That is,
/// if we attempt to read a value that does not exist in symbolic memory,
/// we will redirect the read to a concrete backing.
///
/// It should be trivial to switch the backing to a pure symbolic memory
/// by returning unconstrained symbolic variables for undefined memory.
use std::collections::BTreeMap;
use std::mem::{transmute, MaybeUninit};
use std::ops::{Index, IndexMut};
use std::sync::Arc;

use either::Either;

use fugue::bv::BitVec;
use fugue::bytes::{ByteCast, Endian, Order};
use fugue::ir::il::pcode::{Operand, Register};
use fugue::ir::{Address, Translator};

use fuguex::state::paged::PagedState;
use fuguex::state::pcode::{Error as PCodeError, PCodeState};
use fuguex::state::{IntoStateValues, State, StateOps};

use itertools::Itertools;

use disjoint_interval_tree::Interval;
use disjoint_interval_tree::interval_tree::IntervalSet;

use thiserror::Error;

use crate::expr::{Expr, IVar, SymExpr};
use crate::solver::SolverContext;
use crate::value::Value;

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    State(#[from] PCodeError),
    #[error("unsatisfiable symbolic address: {0}")]
    UnsatAddress(SymExpr),
    #[error("unsatisfiable symbolic variable: {0}")]
    UnsatVariable(SymExpr),
}

impl Error {
    fn state(e: PCodeError) -> Error {
        Error::State(e)
    }
}

const PAGE_SIZE: usize = 4096;

#[derive(Debug, Clone)]
#[repr(transparent)]
struct Page {
    expressions: Box<[Option<SymExpr>; PAGE_SIZE]>,
}

impl Default for Page {
    fn default() -> Self {
        let mut expressions = Box::<[MaybeUninit<Option<SymExpr>>; PAGE_SIZE]>::new(unsafe {
            MaybeUninit::uninit().assume_init()
        });

        for expr in &mut expressions[..] {
            expr.write(None);
        }

        Self {
            expressions: unsafe { transmute(expressions) },
        }
    }
}

impl From<Page> for Box<[Option<SymExpr>]> {
    fn from(page: Page) -> Self {
        page.expressions
    }
}

impl Index<usize> for Page {
    type Output = Option<SymExpr>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.expressions[index]
    }
}

impl IndexMut<usize> for Page {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.expressions[index]
    }
}

impl Page {
    fn new() -> Self {
        Self::default()
    }

    fn is_symbolic(&self, start: usize, length: usize) -> bool {
        self.expressions[start..start + length]
            .iter()
            .any(Option::is_some)
    }

    fn size_of_concrete_region(&self, start: usize, length: usize) -> usize {
        self.expressions[start..start + length]
            .iter()
            .take_while(|e| e.is_none())
            .count()
    }

    fn merge_concrete(lc: &[u8], rc: &[u8], constraints: &SymExpr) -> Option<Self> {
        let mut page = Self::default();
        let mut did_change = false;
        for (p, (l, r)) in page
            .expressions
            .iter_mut()
            .zip(lc.iter().copied().zip(rc.iter().copied()))
        {
            if l != r {
                *p = Some(SymExpr::ite(
                    constraints.clone(),
                    BitVec::from(l).into(),
                    BitVec::from(r).into(),
                ));
                did_change = true
            }
        }

        if did_change {
            Some(page)
        } else {
            None
        }
    }

    fn merge_side(mut self, lc: &[u8], rc: &[u8], constraints: &SymExpr) -> Self {
        for (i, l) in self.expressions.iter_mut().enumerate().take(lc.len()) {
            *l = match l.take() {
                Some(s) => Some(SymExpr::ite(constraints.clone(), s, BitVec::from(rc[i]).into())),
                None => {
                    if lc[i] != rc[i] {
                        Some(SymExpr::ite(
                            constraints.clone(),
                            BitVec::from(lc[i]).into(),
                            BitVec::from(rc[i]).into(),
                        ))
                    } else {
                        None
                    }
                }
            }
        }
        self
    }

    fn merge_both(mut self, mut rs: Self, lc: &[u8], rc: &[u8], constraints: &SymExpr) -> Self {
        for (i, (l, r)) in self
            .expressions
            .iter_mut()
            .zip(rs.expressions.iter_mut())
            .enumerate()
            .take(lc.len())
        {
            match (l.take(), r.take()) {
                (Some(s1), Some(s2)) => *l = Some(SymExpr::ite(constraints.clone(), s1, s2)),
                (Some(s1), None) => {
                    *l = Some(SymExpr::ite(constraints.clone(), s1, BitVec::from(rc[i]).into()))
                }
                (None, Some(s2)) => {
                    *l = Some(SymExpr::ite(constraints.clone(), BitVec::from(lc[i]).into(), s2))
                }
                (None, None) => {
                    if lc[i] != rc[i] {
                        *l = Some(SymExpr::ite(
                            constraints.clone(),
                            BitVec::from(lc[i]).into(),
                            BitVec::from(rc[i]).into(),
                        ))
                    } else {
                        *l = None
                    }
                }
            }
        }
        self
    }

    pub fn view(&self, start: usize, length: usize) -> &[Option<SymExpr>] {
        &self.expressions[start..start + length]
    }

    pub fn view_full(&self) -> &[Option<SymExpr>] {
        &self.expressions[..]
    }

    pub fn view_mut(&mut self, start: usize, length: usize) -> &mut [Option<SymExpr>] {
        &mut self.expressions[start..start + length]
    }

    pub fn clear(&mut self, start: usize, length: usize) {
        for exp in self.view_mut(start, length) {
            *exp = None;
        }
    }
}

#[derive(Clone)]
pub struct ConcolicState<O: Order> {
    pub(crate) solver: SolverContext,
    pages: BTreeMap<u64, Page>,
    registers: BTreeMap<u64, Page>,
    temporaries: BTreeMap<u64, Page>,
    pub(crate) concrete: PCodeState<u8, O>,
    pub(crate) constraints: Vec<SymExpr>,
    symbolic_registers: bool,
    symbolic_temporaries: bool,
    symbolic_memory_regions: IntervalSet<u64>,
}

impl<O: Order> State for ConcolicState<O> {
    type Error = Error;

    fn fork(&self) -> Self {
        Self {
            solver: self.solver.clone(),
            pages: self.pages.clone(),
            registers: self.registers.clone(),
            temporaries: self.temporaries.clone(),
            concrete: self.concrete.fork(),
            constraints: self.constraints.clone(),
            symbolic_registers: self.symbolic_registers,
            symbolic_temporaries: self.symbolic_temporaries,
            symbolic_memory_regions: self.symbolic_memory_regions.clone(),
        }
    }

    fn restore(&mut self, other: &Self) {
        *self = other.fork()
    }
}

impl<O: Order> ConcolicState<O> {
    pub fn new(translator: Arc<Translator>, concrete: PCodeState<u8, O>) -> Self {
        Self {
            solver: SolverContext::new(translator),
            pages: BTreeMap::default(),
            registers: BTreeMap::default(),
            temporaries: BTreeMap::default(),
            concrete,
            constraints: Vec::new(),
            symbolic_registers: false,
            symbolic_temporaries: false,
            symbolic_memory_regions: IntervalSet::new(),
        }
    }

    pub fn symbolic_registers(&mut self, enabled: bool) {
        self.symbolic_registers = enabled;
    }

    pub fn symbolic_temporaries(&mut self, enabled: bool) {
        self.symbolic_temporaries = enabled;
    }

    pub fn symbolic_memory_region<I: Into<Interval<Address>>>(&mut self, range: I) {
        let range = range.into();
        self.symbolic_memory_regions
            .insert(u64::from(*range.start())..=u64::from(*range.end()));
    }

    fn merge_memory_pages(
        s1: &mut BTreeMap<u64, Page>,
        s2: BTreeMap<u64, Page>,
        c1: &PagedState<u8>,
        c2: &PagedState<u8>,
        lcons: &SymExpr,
        rcons: &SymExpr,
    ) -> Result<(), <PagedState<u8> as State>::Error> {
        use itertools::EitherOrBoth::{Both, Left, Right};
        use std::mem::take;

        let segment_pages = c1
            .segments()
            .iter()
            .map(|(iv, _)| {
                let start = u64::from(*iv.start()) / PAGE_SIZE as u64;
                let end = u64::from(*iv.end()) / PAGE_SIZE as u64;
                (start..=end).into_iter()
            })
            .flatten();

        *s1 = take(s1)
            .into_iter()
            .merge_join_by(s2.into_iter(), |(i, _), (j, _)| i.cmp(&j))
            .map(|kv| match kv {
                Left((p, c)) => {
                    let poff = p as usize * PAGE_SIZE;

                    let c1page = &c1.view_values_from(poff).unwrap();
                    let c2page = &c2.view_values_from(poff).unwrap();

                    (p, c.merge_side(c1page, c2page, &lcons))
                }
                Right((p, c)) => {
                    let poff = p as usize * PAGE_SIZE;

                    let c1page = &c1.view_values_from(poff).unwrap();
                    let c2page = &c2.view_values_from(poff).unwrap();

                    (p, c.merge_side(c2page, c1page, &rcons))
                }
                Both((p, lc), (_, rc)) => {
                    let poff = p as usize * PAGE_SIZE;

                    let c1page = &c1.view_values_from(poff).unwrap();
                    let c2page = &c2.view_values_from(poff).unwrap();

                    (p, lc.merge_both(rc, c1page, c2page, &lcons))
                }
            })
            .merge_join_by(segment_pages, |(i, _), j| i.cmp(j))
            .filter_map(|kv| match kv {
                Left((p, c)) | Both((p, c), _) => Some((p, c)),
                Right(p) => {
                    let poff = p as usize * PAGE_SIZE;

                    let c1page = &c1.view_values_from(poff).unwrap();
                    let c2page = &c2.view_values_from(poff).unwrap();

                    Page::merge_concrete(c1page, c2page, &lcons).map(|c| (p, c))
                }
            })
            .collect();

        Ok(())
    }

    fn merge_pages<S: StateOps<Value = u8>>(
        s1: &mut BTreeMap<u64, Page>,
        s2: BTreeMap<u64, Page>,
        c1: &S,
        c2: &S,
        lcons: &SymExpr,
        rcons: &SymExpr,
    ) -> Result<(), <S as State>::Error> {
        use itertools::EitherOrBoth::{Both, Left, Right};
        use std::mem::take;

        let size = c1.len();
        let s1c = c1.view_values(0u64, size)?;
        let s2c = c2.view_values(0u64, size)?;

        *s1 = take(s1)
            .into_iter()
            .merge_join_by(s2.into_iter(), |(i, _), (j, _)| i.cmp(&j))
            .map(|kv| match kv {
                Left((p, c)) => {
                    let poff = p as usize * PAGE_SIZE;

                    let c1page = &s1c[poff..];
                    let c2page = &s2c[poff..];

                    (p, c.merge_side(c1page, c2page, &lcons))
                }
                Right((p, c)) => {
                    let poff = p as usize * PAGE_SIZE;

                    let c1page = &s1c[poff..];
                    let c2page = &s2c[poff..];

                    (p, c.merge_side(c2page, c1page, &rcons))
                }
                Both((p, lc), (_, rc)) => {
                    let poff = p as usize * PAGE_SIZE;

                    let c1page = &s1c[poff..];
                    let c2page = &s2c[poff..];

                    (p, lc.merge_both(rc, c1page, c2page, &lcons))
                }
            })
            .merge_join_by((0..(size / PAGE_SIZE) as u64).into_iter(), |(i, _), j| {
                i.cmp(j)
            })
            .filter_map(|kv| match kv {
                Left((p, c)) | Both((p, c), _) => Some((p, c)),
                Right(p) => {
                    let poff = p as usize * PAGE_SIZE;

                    let c1page = &s1c[poff..];
                    let c2page = &s2c[poff..];

                    Page::merge_concrete(c1page, c2page, &lcons).map(|c| (p, c))
                }
            })
            .collect();

        Ok(())
    }

    pub fn merge(&mut self, other: Self) -> Result<(), Error> {
        use std::mem::take;

        let lcons = take(&mut self.constraints)
            .into_iter()
            .reduce(SymExpr::bool_and)
            .unwrap();
        let rcons = other.constraints.into_iter().reduce(SymExpr::bool_and).unwrap();

        let constraints = SymExpr::bool_or(lcons.clone(), rcons.clone());

        Self::merge_memory_pages(
            &mut self.pages,
            other.pages,
            self.concrete.memory(),
            other.concrete.memory(),
            &lcons,
            &rcons,
        )
        .map_err(PCodeError::Memory)?;

        Self::merge_pages(
            &mut self.registers,
            other.registers,
            self.concrete.registers(),
            other.concrete.registers(),
            &lcons,
            &rcons,
        )
        .map_err(PCodeError::Register)?;

        Self::merge_pages(
            &mut self.temporaries,
            other.temporaries,
            self.concrete.temporaries(),
            other.concrete.temporaries(),
            &lcons,
            &rcons,
        )
        .map_err(PCodeError::Temporary)?;

        self.constraints = vec![constraints];

        Ok(())
    }

    pub fn concrete_state(&self) -> &PCodeState<u8, O> {
        &self.concrete
    }

    pub fn concrete_state_mut(&mut self) -> &mut PCodeState<u8, O> {
        &mut self.concrete
    }

    pub fn push_constraint(&mut self, constraint: SymExpr) {
        self.constraints.push(constraint.simplify());
    }

    pub fn solve(&mut self, expr: SymExpr) -> Option<BitVec> {
        expr.solve(&mut self.solver, &self.constraints)
    }

    pub fn is_sat(&mut self) -> bool {
        self.solver.is_sat(&self.constraints)
    }

    pub fn is_symbolic_register(&self, register: &Register) -> bool {
        self.symbolic_registers || {
            let start = register.offset();
            let length = register.size();
            Self::is_symbolic(&self.registers, start, length)
        }
    }

    pub fn is_symbolic_memory(&self, start: Address, length: usize) -> bool {
        self.symbolic_memory_regions.overlaps(u64::from(start)..=u64::from(start)+(length as u64 - 1)) || {
            let start = u64::from(start);
            Self::is_symbolic(&self.pages, start, length)
        }
    }

    pub fn size_of_concrete_memory_region(&self, start: Address, length: usize) -> usize {
        let start = u64::from(start);
        Self::size_of_concrete_region(&self.pages, start, length)
    }

    pub fn is_symbolic_temporary(&self, start: u64, length: usize) -> bool {
        self.symbolic_temporaries || {
            Self::is_symbolic(&self.temporaries, start, length)
        }
    }

    pub fn is_symbolic_operand(&self, operand: &Operand) -> bool {
        match operand {
            Operand::Address { value, size } => {
                Self::is_symbolic(&self.pages, value.offset(), *size)
            }
            Operand::Register { offset, size, .. } => {
                Self::is_symbolic(&self.registers, *offset, *size)
            }
            Operand::Variable { offset, size, .. } => {
                Self::is_symbolic(&self.temporaries, *offset, *size)
            }
            _ => false,
        }
    }

    pub fn is_forced_symbolic_operand(&self, operand: &Operand) -> bool {
        match operand {
            Operand::Address { value, size } => {
                self.symbolic_memory_regions
                    .overlaps(u64::from(value)..=u64::from(value)+(*size as u64 - 1))
            }
            Operand::Register { .. } => {
                self.symbolic_registers
            }
            Operand::Variable { .. } => {
                self.symbolic_temporaries
            }
            _ => false,
        }
    }

    fn is_symbolic(pages: &BTreeMap<u64, Page>, start: u64, length: usize) -> bool {
        let page_size = PAGE_SIZE as u64;

        let aligned_start = start / page_size;
        let aligned_end = (start + length as u64) / page_size;

        let offset_start = start % page_size;
        let offset_end = (start + length as u64) % page_size;

        let range = (aligned_start..=aligned_end).step_by(PAGE_SIZE);
        let spos = 0;
        let epos = (aligned_end - aligned_start) as usize;

        for (idx, page) in range.enumerate() {
            let offset = if idx == spos {
                offset_start as usize
            } else {
                0
            };
            let length = if idx == epos {
                offset_end as usize - offset
            } else {
                PAGE_SIZE
            };

            if let Some(ref p) = pages.get(&page) {
                if p.is_symbolic(offset, length) {
                    return true;
                }
            }
        }

        false
    }

    fn size_of_concrete_region(pages: &BTreeMap<u64, Page>, start: u64, length: usize) -> usize {
        let page_size = PAGE_SIZE as u64;

        let aligned_start = start / page_size;
        let aligned_end = (start + length as u64) / page_size;

        let offset_start = start % page_size;
        let offset_end = (start + length as u64) % page_size;

        let range = (aligned_start..=aligned_end).step_by(PAGE_SIZE);
        let spos = 0;
        let epos = (aligned_end - aligned_start) as usize;

        let mut concrete_region = 0;

        for (idx, page) in range.enumerate() {
            let offset = if idx == spos {
                offset_start as usize
            } else {
                0
            };
            let length = if idx == epos {
                offset_end as usize - offset
            } else {
                PAGE_SIZE
            };

            if let Some(ref p) = pages.get(&page) {
                let sz = p.size_of_concrete_region(offset, length);
                concrete_region += sz;
                if sz <= length {
                    break;
                }
            }
        }

        concrete_region
    }

    pub fn concretise_register(&mut self, register: &Register) {
        let start = register.offset();
        let length = register.size();
        Self::concretise(&mut self.registers, start, length)
    }

    pub fn concretise_memory(&mut self, start: Address, length: usize) {
        let start = u64::from(start);
        Self::concretise(&mut self.pages, start, length)
    }

    pub fn concretise_temporary(&mut self, offset: u64, length: usize) {
        Self::concretise(&mut self.temporaries, offset, length)
    }

    pub fn force_concretise_operand(&mut self, operand: &Operand) {
        match operand {
            Operand::Address { value, size } => {
                Self::concretise(&mut self.pages, value.offset(), *size)
            }
            Operand::Register { offset, size, .. } => {
                Self::concretise(&mut self.registers, *offset, *size)
            }
            Operand::Variable { offset, size, .. } => {
                Self::concretise(&mut self.temporaries, *offset, *size)
            }
            _ => panic!("cannot concretise Operand::Constant"),
        }
    }

    pub fn concretise_operand(&mut self, operand: &Operand) {
        if self.is_forced_symbolic_operand(operand) {
            return
        }
        match operand {
            Operand::Address { value, size } => {
                Self::concretise(&mut self.pages, value.offset(), *size)
            }
            Operand::Register { offset, size, .. } => {
                Self::concretise(&mut self.registers, *offset, *size)
            }
            Operand::Variable { offset, size, .. } => {
                Self::concretise(&mut self.temporaries, *offset, *size)
            }
            _ => panic!("cannot concretise Operand::Constant"),
        }
    }

    pub fn concretise_operand_with<T: IntoStateValues<u8>>(
        &mut self,
        operand: &Operand,
        value: T,
    ) -> Result<(), Error> {
        if self.is_forced_symbolic_operand(operand) {
            // ensure write-back

            let mut buf = [0u8; 64]; // for now we hard-code it...
            let size = operand.size();
            value.into_values::<O>(&mut buf[..size]);

            self.write_operand_bytes(
                operand, buf[..size].iter().map(|v| SymExpr::val(*v)));
        } else {
            self.concrete
                .set_operand(operand, value)
                .map_err(Error::state)?;

            match operand {
                Operand::Address { value, size } => {
                    Self::concretise(&mut self.pages, value.offset(), *size)
                }
                Operand::Register { offset, size, .. } => {
                    Self::concretise(&mut self.registers, *offset, *size)
                }
                Operand::Variable { offset, size, .. } => {
                    Self::concretise(&mut self.temporaries, *offset, *size)
                }
                _ => panic!("cannot concretise Operand::Constant"),
            }
        }
        Ok(())
    }

    pub fn clear_temporaries(&mut self) {
        self.temporaries.clear()
    }

    fn concretise(pages: &mut BTreeMap<u64, Page>, start: u64, length: usize) {
        let page_size = PAGE_SIZE as u64;

        let aligned_start = start / page_size;
        let aligned_end = (start + length as u64) / page_size;

        let offset_start = start % page_size;
        let offset_end = (start + length as u64) % page_size;

        let range = (aligned_start..=aligned_end).step_by(PAGE_SIZE);
        let spos = 0;
        let epos = (aligned_end - aligned_start) as usize;

        for (idx, page) in range.enumerate() {
            let offset = if idx == spos {
                offset_start as usize
            } else {
                0
            };
            let length = if idx == epos {
                offset_end as usize - offset
            } else {
                PAGE_SIZE
            };

            if let Some(ref mut p) = pages.get_mut(&page) {
                p.clear(offset, length);
            }
        }
    }

    pub fn read_register_buffer(&self, register: &Register) -> Result<Option<SymExpr>, Error> {
        let start = register.offset();
        let length = register.size();
        Self::read_buffer(&self.registers, self.concrete.registers(), start, length)
            .map_err(PCodeError::Register)
            .map_err(Error::state)
    }

    pub fn read_memory_buffer(&self, start: Address, length: usize) -> Result<Option<SymExpr>, Error> {
        let start = u64::from(start);
        Self::read_buffer(&self.pages, self.concrete.memory(), start, length)
            .map_err(PCodeError::Memory)
            .map_err(Error::state)
    }

    // Treat byte range as a contiguous buffer
    fn read_buffer<T: StateOps<Value = u8>>(
        pages: &BTreeMap<u64, Page>,
        concs: &T,
        start: u64,
        length: usize,
    ) -> Result<Option<SymExpr>, T::Error> {
        let page_size = PAGE_SIZE as u64;

        let aligned_start = start / page_size;
        let aligned_end = (start + length as u64) / page_size;

        let offset_start = start % page_size;
        let offset_end = (start + length as u64) % page_size;

        let range = (aligned_start..=aligned_end).step_by(PAGE_SIZE);
        let spos = 0;
        let epos = (aligned_end - aligned_start) as usize / PAGE_SIZE;

        let mut expr = None;
        let concrete = concs.view_values(start, length)?;

        for (idx, page) in range.enumerate() {
            let offset = if idx == spos {
                offset_start as usize
            } else {
                0
            };
            let length = if idx == epos {
                offset_end as usize - offset
            } else {
                PAGE_SIZE
            };

            expr = Some(if let Some(ref p) = pages.get(&page) {
                let coff = idx * PAGE_SIZE;
                let mut it = concrete[coff..coff + length]
                    .iter()
                    .zip(p.view(offset, length));
                let init = expr.unwrap_or_else(|| {
                    let (b, e) = it.next().unwrap();
                    if let Some(ref e) = e {
                        e.clone()
                    } else {
                        BitVec::from(*b).into()
                    }
                });

                it.fold(init, |acc, (b, e)| match e {
                    None => {
                        let c = BitVec::from(*b);
                        SymExpr::concat(acc, c.into())
                    }
                    Some(e) => SymExpr::concat(acc, e.clone()),
                })
            } else {
                // call concrete
                let coff = idx * PAGE_SIZE;
                let mut it = concrete[coff..coff + length].iter();
                let init = expr.unwrap_or_else(|| BitVec::from(*it.next().unwrap()).into());
                it.fold(init, |acc, b| {
                    let c = BitVec::from(*b);
                    SymExpr::concat(acc, c.into())
                })
            })
        }

        Ok(expr)
    }

    pub fn read_register_bytes(&self, register: &Register) -> Result<Vec<SymExpr>, Error> {
        let start = register.offset();
        let length = register.size();
        Self::read_bytes(&self.registers, self.concrete.registers(), start, length)
            .map_err(PCodeError::Register)
            .map_err(Error::state)
    }

    pub fn read_memory_bytes(&self, start: Address, length: usize) -> Result<Vec<SymExpr>, Error> {
        let start = u64::from(start);
        Self::read_bytes(&self.pages, self.concrete.memory(), start, length)
            .map_err(PCodeError::Memory)
            .map_err(Error::state)
    }

    pub fn read_temporary_bytes(&self, start: u64, length: usize) -> Result<Vec<SymExpr>, Error> {
        Self::read_bytes(
            &self.temporaries,
            self.concrete.temporaries(),
            start,
            length,
        )
        .map_err(PCodeError::Temporary)
        .map_err(Error::state)
    }

    pub fn read_operand_bytes(&self, operand: &Operand) -> Result<Vec<SymExpr>, Error> {
        match operand {
            Operand::Address { value, size } => self.read_memory_bytes(*value, *size),
            Operand::Constant { size, value, .. } => {
                let value = *value;
                let mut buf = [0u8; 8];

                value.into_bytes::<O>(&mut buf);

                let bufr = if O::ENDIAN.is_big() {
                    &buf[8 - size..]
                } else {
                    &buf[..*size]
                };

                Ok(bufr
                    .iter()
                    .map(|b| BitVec::from(*b).into())
                    .collect::<Vec<_>>())
            }
            Operand::Register { .. } => {
                let register = operand.register().unwrap();
                self.read_register_bytes(&register)
            }
            Operand::Variable { offset, size, .. } => self.read_temporary_bytes(*offset, *size),
        }
    }

    pub fn read_symbolic_operand_bytes(&mut self, operand: &Operand) -> Vec<SymExpr> {
        match operand {
            Operand::Address { value, size } => {
                Self::read_symbolic_bytes(&mut self.pages, value.offset(), *size)
            },
            Operand::Constant { size, value, .. } => {
                let value = *value;
                let mut buf = [0u8; 8];

                value.into_bytes::<O>(&mut buf);

                let bufr = if O::ENDIAN.is_big() {
                    &buf[8 - size..]
                } else {
                    &buf[..*size]
                };

                bufr
                    .iter()
                    .map(|b| BitVec::from(*b).into())
                    .collect::<Vec<_>>()
            },
            Operand::Register { offset, size, .. } => {
                Self::read_symbolic_bytes(&mut self.registers, *offset, *size)
            },
            Operand::Variable { offset, size, .. } => {
                Self::read_symbolic_bytes(&mut self.temporaries, *offset, *size)
            },
        }
    }

    fn read_symbolic_bytes(
        pages: &mut BTreeMap<u64, Page>,
        start: u64,
        length: usize,
    ) -> Vec<SymExpr> {
        let page_size = PAGE_SIZE as u64;

        let aligned_start = start / page_size;
        let aligned_end = (start + length as u64) / page_size;

        let offset_start = start % page_size;
        let offset_end = (start + length as u64) % page_size;

        let range = (aligned_start..=aligned_end).step_by(PAGE_SIZE);
        let spos = 0;
        let epos = (aligned_end - aligned_start) as usize / PAGE_SIZE;

        let mut exprs = Vec::new();

        for (idx, page) in range.enumerate() {
            let offset = if idx == spos {
                offset_start as usize
            } else {
                0
            };
            let length = if idx == epos {
                offset_end as usize - offset
            } else {
                PAGE_SIZE
            };

            if let Some(p) = pages.get_mut(&page) {
                exprs.extend(p.view_mut(offset, length).iter_mut().map(|e| match e {
                    None => {
                        let v = SymExpr::from(IVar::new(8));
                        *e = Some(v.clone());
                        v
                    },
                    Some(e) => e.clone(),
                }));
            } else {
                let mut p = Page::new();

                exprs.extend(p.view_mut(offset, length).iter_mut().map(|pv| {
                    let v = SymExpr::from(IVar::new(8));
                    *pv = Some(v.clone());
                    v
                }));

                pages.insert(page, p);
            }
        }

        exprs
    }

    // Treat each byte in the range as a separate entity
    fn read_bytes<T: StateOps<Value = u8>>(
        pages: &BTreeMap<u64, Page>,
        concs: &T,
        start: u64,
        length: usize,
    ) -> Result<Vec<SymExpr>, T::Error> {
        let page_size = PAGE_SIZE as u64;

        let aligned_start = start / page_size;
        let aligned_end = (start + length as u64) / page_size;

        let offset_start = start % page_size;
        let offset_end = (start + length as u64) % page_size;

        let range = (aligned_start..=aligned_end).step_by(PAGE_SIZE);
        let spos = 0;
        let epos = (aligned_end - aligned_start) as usize / PAGE_SIZE;

        let mut exprs = Vec::new();
        let concrete = concs.view_values(start, length)?;

        for (idx, page) in range.enumerate() {
            let offset = if idx == spos {
                offset_start as usize
            } else {
                0
            };
            let length = if idx == epos {
                offset_end as usize - offset
            } else {
                PAGE_SIZE
            };

            if let Some(ref p) = pages.get(&page) {
                let coff = idx * PAGE_SIZE;
                exprs.extend(
                    concrete[coff..coff + length]
                        .iter()
                        .zip(p.view(offset, length))
                        .map(|(b, e)| match e {
                            None => BitVec::from(*b).into(),
                            Some(e) => e.clone(),
                        }),
                );
            } else {
                // call concrete
                let coff = idx * PAGE_SIZE;
                exprs.extend(
                    concrete[coff..coff + length]
                        .iter()
                        .map(|b| BitVec::from(*b).into()),
                );
            }
        }

        Ok(exprs)
    }

    fn copy_mixed_byte_page<T: StateOps<Value = u8>>(
        pages: &mut BTreeMap<u64, Page>,
        concs: &mut T,
        from_sym: &[Option<SymExpr>],
        from_con: &[u8],
        to: u64,
    ) -> Result<(), T::Error> {
        let page_size = PAGE_SIZE as u64;

        let mut aligned = to / page_size;
        let aligned_off = (to % page_size) as usize;

        let length = from_sym.len();
        let mut written = 0;

        // first page is at offset
        if aligned_off != 0 {
            let this_length = (PAGE_SIZE - aligned_off).min(length);
            let sym_range = &from_sym[..this_length];
            let con_range = &from_con[..this_length];

            if sym_range.iter().any(Option::is_some) {
                // range contains symbolic
                let page = pages.entry(aligned).or_insert_with(Page::new);
                let cview = concs.view_values_mut(to, this_length)?;
                let sview = page.view_mut(aligned_off, this_length);
                for (i, (s, c)) in sym_range.iter().zip(con_range).enumerate() {
                    if s.is_some() {
                        sview[i] = s.clone();
                    } else {
                        sview[i] = None;
                        cview[i] = *c;
                    }
                }
            } else {
                if let Some(page) = pages.get_mut(&aligned) {
                    // we clear the range to concretise it
                    page.clear(aligned_off, this_length)
                }
                concs.set_values(to, con_range)?;
            }

            aligned += page_size;
            written += this_length;
        }

        // aligned first page or second page spill-over
        if written < length {
            let this_length = length - written;
            let sym_range = &from_sym[written..];
            let con_range = &from_con[written..];

            if sym_range.iter().any(Option::is_some) {
                // range contains symbolic
                let page = pages.entry(aligned).or_insert_with(Page::new);
                let cview = concs.view_values_mut(to + written as u64, this_length)?;
                let sview = page.view_mut(0, this_length);
                for (i, (s, c)) in sym_range.iter().zip(con_range).enumerate() {
                    if s.is_some() {
                        sview[i] = s.clone();
                    } else {
                        sview[i] = None;
                        cview[i] = *c;
                    }
                }
            } else {
                if let Some(page) = pages.get_mut(&aligned) {
                    // we clear the range to concretise it
                    page.clear(0, this_length)
                }
                concs.set_values(to + written as u64, con_range)?;
            }
        }

        Ok(())
    }

    fn copy_mixed_bytes_forward<T: StateOps<Value = u8>>(
        pages: &mut BTreeMap<u64, Page>,
        concs: &mut T,
        from: u64,
        to: u64,
        length: usize,
    ) -> Result<(), T::Error> {
        let mut symbolic_page: Box<[Option<SymExpr>]> = Page::default().into();
        let mut concrete_page = [0u8; PAGE_SIZE];

        let page_size = PAGE_SIZE as u64;

        // from symbolic pages
        let aligned_start = from / page_size;
        let aligned_end = (from + length as u64) / page_size;

        let offset_start = from % page_size;
        let offset_end = (from + length as u64) % page_size;

        let range = (aligned_start..=aligned_end).step_by(PAGE_SIZE);
        let spos = 0;
        let epos = (aligned_end - aligned_start) as usize / PAGE_SIZE;

        let mut written = 0;

        for (idx, page) in range.enumerate() {
            let offset = if idx == spos {
                offset_start as usize
            } else {
                0
            };
            let length = if idx == epos {
                offset_end as usize - offset
            } else {
                PAGE_SIZE
            };

            if let Some(ref p) = pages.get(&page) {
                // set symbolic range
                symbolic_page.clone_from_slice(p.view_full());
                let sym_range = &mut symbolic_page[offset..offset + length];

                // get concrete range
                let con_range = &mut concrete_page[offset..offset + length];
                con_range.copy_from_slice(concs.view_values(page + offset as u64, length)?);

                Self::copy_mixed_byte_page(pages, concs, sym_range, con_range, to + written)?;
            } else {
                // ensure symbolic is all zeroed
                let sym_range = &mut symbolic_page[offset..offset + length];
                for exp in sym_range.iter_mut() {
                    *exp = None;
                }

                // get concrete range
                let con_range = &mut concrete_page[offset..offset + length];
                con_range.copy_from_slice(concs.view_values(page + offset as u64, length)?);

                Self::copy_mixed_byte_page(pages, concs, sym_range, con_range, to + written)?;
            }
            written += length as u64;
        }

        Ok(())
    }

    fn copy_mixed_bytes_backward<T: StateOps<Value = u8>>(
        pages: &mut BTreeMap<u64, Page>,
        concs: &mut T,
        from: u64,
        to: u64,
        length: usize,
    ) -> Result<(), T::Error> {
        let mut symbolic_page: Box<[Option<SymExpr>]> = Page::default().into();
        let mut concrete_page = [0u8; PAGE_SIZE];

        let page_size = PAGE_SIZE as u64;

        // from symbolic pages
        let aligned_start = from / page_size;
        let aligned_end = (from + length as u64) / page_size;

        let offset_start = from % page_size;
        let offset_end = (from + length as u64) % page_size;

        let range = (aligned_start..=aligned_end).rev().step_by(PAGE_SIZE);
        let spos = (aligned_end - aligned_start) as usize / PAGE_SIZE;
        let epos = 0;

        let to_end = to + length as u64;
        let mut written = 0;

        for (idx, page) in range.enumerate() {
            let offset = if idx == spos {
                offset_start as usize
            } else {
                0
            };
            let length = if idx == epos {
                offset_end as usize - offset
            } else {
                PAGE_SIZE
            };

            if let Some(ref p) = pages.get(&page) {
                // set symbolic range
                symbolic_page.clone_from_slice(p.view_full());
                let sym_range = &mut symbolic_page[offset..offset + length];

                // get concrete range
                let con_range = &mut concrete_page[offset..offset + length];
                con_range.copy_from_slice(concs.view_values(page + offset as u64, length)?);

                Self::copy_mixed_byte_page(
                    pages,
                    concs,
                    sym_range,
                    con_range,
                    to_end - written - length as u64,
                )?;
            } else {
                // ensure symbolic is all zeroed
                let sym_range = &mut symbolic_page[offset..length];
                for exp in sym_range.iter_mut() {
                    *exp = None;
                }

                // get concrete range
                let con_range = &mut concrete_page[offset..offset + length];
                con_range.copy_from_slice(concs.view_values(page + offset as u64, length)?);

                Self::copy_mixed_byte_page(
                    pages,
                    concs,
                    sym_range,
                    con_range,
                    to_end - written - length as u64,
                )?;
            }
            written += length as u64;
        }

        Ok(())
    }

    pub fn copy_mixed_memory_bytes(
        &mut self,
        from: u64,
        to: u64,
        length: usize,
    ) -> Result<(), Error> {
        let page_size = PAGE_SIZE as u64;

        let faligned_start = from / page_size;
        let faligned_end = (from + length as u64) / page_size;

        let taligned_start = to / page_size;

        if taligned_start >= faligned_start && taligned_start <= faligned_end {
            Self::copy_mixed_bytes_backward(
                &mut self.pages,
                self.concrete.memory_mut(),
                from,
                to,
                length,
            )
        } else {
            Self::copy_mixed_bytes_forward(
                &mut self.pages,
                self.concrete.memory_mut(),
                from,
                to,
                length,
            )
        }
        .map_err(PCodeError::Memory)
        .map_err(Error::state)
    }

    pub fn write_register_bytes<I: IntoIterator<Item = SymExpr>>(
        &mut self,
        register: &Register,
        it: I,
    ) {
        let start = register.offset();
        Self::write_bytes(&mut self.registers, start, it)
    }

    pub fn write_memory_bytes<I: IntoIterator<Item = SymExpr>>(&mut self, at: Address, it: I) {
        let start = u64::from(at);
        Self::write_bytes(&mut self.pages, start, it)
    }

    pub fn write_temporary_bytes<I: IntoIterator<Item = SymExpr>>(&mut self, at: u64, it: I) {
        Self::write_bytes(&mut self.temporaries, at, it)
    }

    pub fn write_operand_bytes<I: IntoIterator<Item = SymExpr>>(&mut self, operand: &Operand, it: I) {
        match operand {
            Operand::Address { value, .. } => self.write_memory_bytes(*value, it),
            Operand::Register { .. } => {
                let register = operand.register().unwrap();
                self.write_register_bytes(&register, it)
            }
            Operand::Variable { offset, .. } => self.write_temporary_bytes(*offset, it),
            _ => {
                panic!("write to constant: {}", operand)
            }
        }
    }

    fn write_bytes<I: IntoIterator<Item = SymExpr>>(pages: &mut BTreeMap<u64, Page>, at: u64, it: I) {
        let page_size = PAGE_SIZE as u64;

        let mut it = it.into_iter();
        let mut aligned = at / page_size;
        let aligned_off = (at % page_size) as usize;

        // first page is at offset
        if aligned_off != 0 {
            let page = pages.entry(aligned).or_insert_with(Page::new);
            for (d, s) in page
                .view_mut(aligned_off, PAGE_SIZE - aligned_off)
                .iter_mut()
                .zip(&mut it)
            {
                *d = Some(s);
            }

            aligned += page_size;
        }

        // write the rest page by page
        for chunk in it.chunks(PAGE_SIZE).into_iter() {
            let page = pages.entry(aligned).or_insert_with(Page::new);
            for (d, s) in page.view_mut(aligned_off, PAGE_SIZE).iter_mut().zip(chunk) {
                *d = Some(s);
            }
            aligned += page_size;
        }
    }

    pub fn read_register_primitive<T: ByteCast>(
        &self,
        register: &Register,
    ) -> Result<Option<SymExpr>, Error> {
        let start = register.offset();
        Self::read_primitive::<T, _>(&self.registers, self.concrete.registers(), start)
            .map_err(PCodeError::Register)
            .map_err(Error::state)
    }

    pub fn read_memory_primitive<T: ByteCast>(&self, at: Address) -> Result<Option<SymExpr>, Error> {
        let start = u64::from(at);
        Self::read_primitive::<T, _>(&self.pages, self.concrete.memory(), start)
            .map_err(PCodeError::Memory)
            .map_err(Error::state)
    }

    pub fn read_temporary_primitive<T: ByteCast>(&self, at: u64) -> Result<Option<SymExpr>, Error> {
        Self::read_primitive::<T, _>(&self.temporaries, self.concrete.temporaries(), at)
            .map_err(PCodeError::Temporary)
            .map_err(Error::state)
    }

    // Read a primitive using a given endianness
    fn read_primitive<T: ByteCast, S: StateOps<Value = u8>>(
        pages: &BTreeMap<u64, Page>,
        concs: &S,
        at: u64,
    ) -> Result<Option<SymExpr>, S::Error> {
        let buffer = Self::read_bytes(pages, concs, at, T::SIZEOF)?;
        Ok(if O::ENDIAN == Endian::Big {
            buffer.into_iter().reduce(|acc, v| SymExpr::concat(acc, v))
        } else {
            buffer.into_iter().reduce(|acc, v| SymExpr::concat(v, acc))
        })
    }

    pub fn write_register_primitive<T: ByteCast>(&mut self, register: &Register, expr: SymExpr) {
        let start = register.offset();
        Self::write_primitive::<T>(&mut self.registers, start, expr)
    }

    pub fn write_memory_primitive<T: ByteCast>(&mut self, at: Address, expr: SymExpr) {
        let start = u64::from(at);
        Self::write_primitive::<T>(&mut self.pages, start, expr)
    }

    pub fn write_temporary_primitive<T: ByteCast>(&mut self, at: u64, expr: SymExpr) {
        Self::write_primitive::<T>(&mut self.temporaries, at, expr)
    }

    fn write_primitive<T: ByteCast>(pages: &mut BTreeMap<u64, Page>, at: u64, expr: SymExpr) {
        let size = expr.bits();
        let bits = (T::SIZEOF * 8) as u32;

        let expr = if bits == size {
            expr
        } else if bits < size {
            if T::SIGNED {
                SymExpr::sign_extend(expr, size)
            } else {
                SymExpr::zero_extend(expr, size)
            }
        } else {
            SymExpr::extract(expr, 0, size)
        };

        Self::write_expr(pages, at, expr)
    }

    pub fn write_register_expr(&mut self, register: &Register, expr: SymExpr) {
        let start = register.offset();
        Self::write_expr(&mut self.registers, start, expr)
    }

    pub fn write_memory_expr(&mut self, at: Address, expr: SymExpr) {
        let start = u64::from(at);
        Self::write_expr(&mut self.pages, start, expr)
    }

    pub fn write_temporary_expr(&mut self, at: u64, expr: SymExpr) {
        Self::write_expr(&mut self.temporaries, at, expr)
    }

    pub fn write_operand_expr(&mut self, operand: &Operand, expr: SymExpr) {
        assert_eq!(expr.bits(), operand.size() as u32 * 8);
        match operand {
            Operand::Address { value, .. } => {
                Self::write_expr(&mut self.pages, value.offset(), expr)
            }
            Operand::Register { offset, .. } => {
                Self::write_expr(&mut self.registers, *offset, expr)
            }
            Operand::Variable { offset, .. } => {
                Self::write_expr(&mut self.temporaries, *offset, expr)
            }
            _ => panic!("cannot write to Operand::Constant"),
        }
    }

    fn write_expr(pages: &mut BTreeMap<u64, Page>, at: u64, expr: SymExpr) {
        debug_assert_eq!(expr.bits() % 8, 0);
        let at = u64::from(at);
        let page_size = PAGE_SIZE as u64;
        let bits = expr.bits();
        let bytes = bits / 8;

        let mut it = (0..bytes).map(|i| {
            if O::ENDIAN == Endian::Big {
                let lsb = bits - (i * 8) - 8;
                SymExpr::extract(expr.clone(), lsb, lsb + 8)
            } else {
                let lsb = i * 8;
                SymExpr::extract(expr.clone(), lsb, lsb + 8)
            }
        });

        let aligned = at / page_size;
        let aligned_off = (at % page_size) as usize;

        let overlap = PAGE_SIZE - aligned_off;

        let page1 = overlap.min(bytes as usize);
        let page2 = bytes as usize - page1;

        let page = pages.entry(aligned).or_insert_with(Page::new);
        let view = page.view_mut(aligned_off, page1);

        for (d, s) in view.iter_mut().zip(&mut it) {
            *d = Some(s);
        }

        // Handle the case where the primitive is split over two pages
        if page2 != 0 {
            let page = pages.entry(aligned + page_size).or_insert_with(Page::new);
            let view = page.view_mut(0, page2);
            for (d, s) in view.iter_mut().zip(&mut it) {
                *d = Some(s);
            }
        }
    }

    pub fn write_operand_value(
        &mut self,
        operand: &Operand,
        value: Either<BitVec, SymExpr>,
    ) -> Result<(), Error> {
        match value {
            Either::Left(bv) => if !self.is_forced_symbolic_operand(operand) {
                self.concretise_operand_with(operand, bv)?;
            } else {
                self.write_operand_expr(operand, bv.into())
            },
            Either::Right(expr) => {
                let val = expr.simplify();
                if !self.is_forced_symbolic_operand(operand) {
                    if let Expr::Val(bv) = &*val {
                        self.concretise_operand_with(operand, bv)?;
                    } else {
                        self.write_operand_expr(operand, val);
                    }
                } else {
                    self.write_operand_expr(operand, val)
                }
            }
        }
        Ok(())
    }

    pub fn read_operand_value(&mut self, operand: &Operand) -> Result<Either<BitVec, SymExpr>, Error> {
        if self.is_forced_symbolic_operand(operand) {
            let expr = self.read_symbolic_operand_expr(operand)?.simplify();
            if let Expr::Val(bv) = &*expr {
                Ok(Either::Left(bv.clone()))
            } else {
                Ok(Either::Right(expr))
            }
        } else {
            if self.is_symbolic_operand(operand) {
                let expr = self.read_operand_expr(operand)?.simplify();
                if let Expr::Val(bv) = &*expr {
                    self.concretise_operand_with(operand, bv)?;
                    Ok(Either::Left(bv.clone()))
                } else {
                    Ok(Either::Right(expr))
                }
            } else {
                let bv = self
                    .concrete
                    .get_operand::<BitVec>(operand)
                    .map_err(Error::state)?;
                Ok(Either::Left(bv))
            }
        }
    }

    pub fn read_symbolic_operand_expr(&mut self, operand: &Operand) -> Result<SymExpr, Error> {
        let vals = self.read_symbolic_operand_bytes(operand);
        let expr = if O::ENDIAN.is_big() {
            vals.into_iter()
                .reduce(|acc, v| SymExpr::concat(acc, v))
                .unwrap()
        } else {
            vals.into_iter()
                .rev()
                .reduce(|acc, v| SymExpr::concat(acc, v))
                .unwrap()
        };
        Ok(expr)
    }

    pub fn read_operand_expr(&self, operand: &Operand) -> Result<SymExpr, Error> {
        let vals = self.read_operand_bytes(operand)?;
        let expr = if O::ENDIAN.is_big() {
            vals.into_iter()
                .reduce(|acc, v| SymExpr::concat(acc, v))
                .unwrap()
        } else {
            vals.into_iter()
                .rev()
                .reduce(|acc, v| SymExpr::concat(acc, v))
                .unwrap()
        };
        Ok(expr)
    }

    pub fn read_memory_expr(&self, addr: Address, bits: usize) -> Result<SymExpr, Error> {
        let vals = self.read_memory_bytes(addr, bits / 8)?;
        let expr = if O::ENDIAN.is_big() {
            vals.into_iter()
                .reduce(|acc, v| SymExpr::concat(acc, v))
                .unwrap()
        } else {
            vals.into_iter()
                .rev()
                .reduce(|acc, v| SymExpr::concat(acc, v))
                .unwrap()
        };
        Ok(expr)
    }

    pub fn read_memory_symbolic(&mut self, addr: &SymExpr, bits: usize) -> Result<SymExpr, Error> {
        let addr = addr.simplify();

        if let Expr::Val(bv) = &*addr {
            let naddr = bv.to_u64().expect("64-bit address limit");
            self.read_memory_expr(self.solver.translator().address(naddr).into(), bits)
        } else {
            let addrs = addr.solve_many(&mut self.solver, &self.constraints);
            if addrs.is_empty() {
                return Err(Error::UnsatAddress(addr));
            }

            let mut init = SymExpr::from(BitVec::zero(bits)); // FIXME: we should not default to 0!
            for caddr in addrs {
                let addrv = caddr.to_u64().expect("64-bit address limit");
                let cval =
                    self.read_memory_expr(self.solver.translator().address(addrv).into(), bits)?;
                let cond = SymExpr::eq(addr.clone(), caddr.into());
                init = SymExpr::ite(cond, cval, init);
            }

            Ok(init)
        }
    }

    pub fn read_memory_symbolic_bytes(&mut self, addr: &SymExpr, count: usize) -> Result<Vec<SymExpr>, Error> {
        let addr = addr.simplify();

        if let Expr::Val(bv) = &*addr {
            let naddr = bv.to_u64().expect("64-bit address limit");
            self.read_memory_bytes(self.solver.translator().address(naddr).into(), count)
        } else {
            let addrs = addr.solve_many(&mut self.solver, &self.constraints);
            if addrs.is_empty() {
                return Err(Error::UnsatAddress(addr));
            }

            let mut bytes = Vec::new();

            for i in 0..count {
                let abits = addr.bits() as usize;
                let cidx = BitVec::from_usize(i, abits);
                let sidx = SymExpr::from(cidx.clone());

                // FIXME: we should not default to 0!
                let mut init = SymExpr::from(BitVec::from(0u8));

                for caddr in addrs.iter().cloned() {
                    let addrv = caddr.to_u64().expect("64-bit address limit") + (i as u64);
                    let cval =
                        self.read_memory_expr(self.solver.translator().address(addrv).into(), 8)?;
                    let cond = SymExpr::eq(&addr + &sidx, (&caddr + &cidx).into());
                    init = SymExpr::ite(cond, cval, init);
                }

                bytes.push(init);
            }

            Ok(bytes)
        }
    }

    fn write_memory_symbolic_aux(
        &mut self,
        aexpr: SymExpr,
        addr: Address,
        texpr: SymExpr,
    ) -> Result<(), Error> {
        let abits = aexpr.bits() as usize;
        let fexpr = self.read_memory_expr(addr, texpr.bits() as usize)?;

        let cond = SymExpr::eq(aexpr, BitVec::from_u64(addr.into(), abits).into());
        let current = SymExpr::ite(cond, texpr, fexpr);

        self.write_memory_expr(addr, current);

        Ok(())
    }

    pub fn write_memory_symbolic(&mut self, addr: &SymExpr, expr: SymExpr) -> Result<(), Error> {
        let addr = addr.simplify();

        if let Expr::Val(bv) = &*addr {
            let naddr = bv.to_u64().expect("64-bit address limit");
            self.write_memory_expr(self.solver.translator().address(naddr).into(), expr.clone());
        } else {
            for caddr in addr.solve_many(&mut self.solver, &self.constraints) {
                let addrv = caddr.to_u64().expect("64-bit address limit");
                self.write_memory_symbolic_aux(
                    addr.clone(),
                    self.solver.translator().address(addrv).into(),
                    expr.clone(),
                )?;
            }
        }

        Ok(())
    }

    pub fn search_memory_symbolic(&mut self, addr: &SymExpr, needle: &SymExpr, bytes: &SymExpr) -> Result<SymExpr, Error> {
        let buffer_len = bytes.maximise(&mut self.solver, &self.constraints)
            .ok_or_else(|| Error::UnsatVariable(bytes.clone()))?
            .to_usize()
            .expect("buffer maximum length exceeds usize");

        let needle_bits = needle.bits() as usize;
        let needle_bytes = needle_bits / 8;
        let address_bits = addr.bits() as usize;

        let mut cond = SymExpr::val(true);
        let mut retn = SymExpr::val(false);

        for i in 0..(buffer_len - needle_bytes) {
            let pos = addr + &SymExpr::val(BitVec::from_usize(i, address_bits));
            let val = self.read_memory_symbolic(&pos, needle_bits)?;

            let pcond = val.clone().lt(addr + &bytes) & !val.clone().lt(addr.clone());
            let ncond = val.clone().eq(needle.clone()) & pcond & !&cond;

            retn = SymExpr::ite(ncond, val.clone(), retn);
            cond = val.clone().eq(needle.clone()) | cond;

            if retn.is_true() || val == *needle {
                break
            }
        }

        Ok(retn)
    }

    pub fn compare_memory_symbolic(&mut self, addr1: &SymExpr, addr2: &SymExpr, bytes: &SymExpr) -> Result<SymExpr, Error> {
        let buffer_len = bytes.maximise(&mut self.solver, &self.constraints)
            .ok_or_else(|| Error::UnsatVariable(bytes.clone()))?
            .to_usize()
            .expect("buffer maximum length exceeds usize");

        let buffer1 = self.read_memory_symbolic_bytes(addr1, buffer_len)?;
        let buffer2 = self.read_memory_symbolic_bytes(addr2, buffer_len)?;

        let length_bits = bytes.bits() as usize;

        let mut cond = SymExpr::val(true);
        let mut retn = SymExpr::val(false);

        for i in 0..buffer_len {
            let b1 = &buffer1[i];
            let b2 = &buffer2[i];

            let idx = SymExpr::from(BitVec::from_usize(i, length_bits));

            let is_eq = b1.clone().eq(b2.clone());
            let is_lt = b1.clone().lt(b2.clone());

            let is_gt = !&is_lt & !&is_eq;
            let is_lt = is_lt & !is_eq;

            let len_cond = idx.lt(bytes.clone());

            let lt_val = SymExpr::ite(
                is_lt & cond.clone() & len_cond.clone(),
                SymExpr::from(BitVec::from(0xffu8)),
                retn,
            );

            retn = SymExpr::ite(
                is_gt & cond.clone() & len_cond,
                SymExpr::from(BitVec::from(0x1u8)),
                lt_val,
            );

            cond = cond & retn.clone().eq(SymExpr::from(BitVec::from(0u8)));

            if cond.is_true() {
                break
            }
        }

        Ok(retn)
    }
}
