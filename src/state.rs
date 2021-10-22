/// Symbolic flat memory
///
/// Note: the API exposed for this backing is actually concolic. That is,
/// if we attempt to read a value that does not exist in symbolic memory,
/// we will redirect the read to a concrete backing.
///
/// It should be trivial to switch the backing to a pure symbolic memory
/// by returning unconstrained symbolic variables for undefined memory.
use std::mem::{MaybeUninit, transmute};
use std::sync::Arc;

use either::Either;

use fugue::bytes::{ByteCast, Endian, Order};
use fugue::bv::BitVec;
use fugue::ir::{Address, Translator};
use fugue::ir::il::ecode::Expr;
use fugue::ir::il::pcode::{Operand, Register};

use fuguex::state::{IntoStateValues, State, StateOps};
use fuguex::state::pcode::{PCodeState, Error as PCodeError};

use fxhash::FxHashMap as HashMap;

use itertools::Itertools;

use thiserror::Error;

use crate::solver::SolverContext;
use crate::value::Value;

#[derive(Debug, Error)]
pub enum Error {
    #[error(transparent)]
    State(PCodeError),
    #[error("unsatisfiable symbolic address: {0}")]
    UnsatAddress(Expr),
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
    expressions: [Option<Expr>; PAGE_SIZE],
}

impl Default for Page {
    fn default() -> Self {
        let mut expressions: [MaybeUninit<Option<Expr>>; PAGE_SIZE] = unsafe {
            MaybeUninit::uninit().assume_init()
        };

        for expr in &mut expressions[..] {
            expr.write(None);
        }

        Self { expressions: unsafe { transmute(expressions) } }
    }
}

impl From<Page> for [Option<Expr>; PAGE_SIZE] {
    fn from(page: Page) -> Self {
        page.expressions
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

    pub fn view(&self, start: usize, length: usize) -> &[Option<Expr>] {
        &self.expressions[start..start + length]
    }

    pub fn view_full(&self) -> &[Option<Expr>] {
        &self.expressions[..]
    }

    pub fn view_mut(&mut self, start: usize, length: usize) -> &mut [Option<Expr>] {
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
    pages: HashMap<u64, Page>,
    registers: HashMap<u64, Page>,
    temporaries: HashMap<u64, Page>,
    pub(crate) concrete: PCodeState<u8, O>,
    pub(crate) constraints: Vec<Expr>,
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
            pages: HashMap::default(),
            registers: HashMap::default(),
            temporaries: HashMap::default(),
            concrete,
            constraints: Vec::new(),
        }
    }

    pub fn concrete_state(&self) -> &PCodeState<u8, O> {
        &self.concrete
    }

    pub fn concrete_state_mut(&mut self) -> &mut PCodeState<u8, O> {
        &mut self.concrete
    }

    pub fn push_constraint(&mut self, constraint: Expr) {
        self.constraints.push(constraint.simplify(&self.solver));
    }

    pub fn solve(&mut self, expr: Expr) -> Option<BitVec> {
        expr.solve(&mut self.solver, &self.constraints)
    }

    pub fn is_symbolic_register(&self, register: &Register) -> bool {
        let start = register.offset();
        let length = register.size();
        Self::is_symbolic(&self.registers, start, length)
    }

    pub fn is_symbolic_memory(&self, start: Address, length: usize) -> bool {
        let start = u64::from(start);
        Self::is_symbolic(&self.pages, start, length)
    }

    pub fn size_of_concrete_memory_region(&self, start: Address, length: usize) -> usize {
        let start = u64::from(start);
        Self::size_of_concrete_region(&self.pages, start, length)
    }

    pub fn is_symbolic_temporary(&self, start: u64, length: usize) -> bool {
        Self::is_symbolic(&self.temporaries, start, length)
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

    fn is_symbolic(pages: &HashMap<u64, Page>, start: u64, length: usize) -> bool {
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

    fn size_of_concrete_region(pages: &HashMap<u64, Page>, start: u64, length: usize) -> usize {
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

    pub fn concretise_operand(&mut self, operand: &Operand) {
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

    pub fn concretise_operand_with<T: IntoStateValues<u8>>(&mut self, operand: &Operand, value: T) -> Result<(), Error> {
        self.concrete.set_operand(operand, value).map_err(Error::state)?;
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
        Ok(())
    }

    pub fn clear_temporaries(&mut self) {
        self.temporaries.clear()
    }

    fn concretise(pages: &mut HashMap<u64, Page>, start: u64, length: usize) {
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

    pub fn read_register_buffer(
        &self,
        register: &Register,
    ) -> Result<Option<Expr>, Error> {
        let start = register.offset();
        let length = register.size();
        Self::read_buffer(&self.registers, self.concrete.registers(), start, length)
            .map_err(PCodeError::Register)
            .map_err(Error::state)
    }

    pub fn read_memory_buffer(
        &self,
        start: Address,
        length: usize,
    ) -> Result<Option<Expr>, Error> {
        let start = u64::from(start);
        Self::read_buffer(&self.pages, self.concrete.memory(), start, length)
            .map_err(PCodeError::Memory)
            .map_err(Error::state)
    }

    // Treat byte range as a contiguous buffer
    fn read_buffer<T: StateOps<Value=u8>>(
        pages: &HashMap<u64, Page>,
        concs: &T,
        start: u64,
        length: usize,
    ) -> Result<Option<Expr>, T::Error> {
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
                        Expr::concat(acc, c)
                    }
                    Some(e) => Expr::concat(acc, e.clone()),
                })
            } else {
                // call concrete
                let coff = idx * PAGE_SIZE;
                let mut it = concrete[coff..coff + length].iter();
                let init = expr.unwrap_or_else(|| BitVec::from(*it.next().unwrap()).into());
                it.fold(init, |acc, b| {
                    let c = BitVec::from(*b);
                    Expr::concat(acc, c)
                })
            })
        }

        Ok(expr)
    }

    pub fn read_register_bytes(
        &self,
        register: &Register,
    ) -> Result<Vec<Expr>, Error> {
        let start = register.offset();
        let length = register.size();
        Self::read_bytes(&self.registers, self.concrete.registers(), start, length)
            .map_err(PCodeError::Register)
            .map_err(Error::state)
    }

    pub fn read_memory_bytes(
        &self,
        start: Address,
        length: usize,
    ) -> Result<Vec<Expr>, Error> {
        let start = u64::from(start);
        Self::read_bytes(&self.pages, self.concrete.memory(), start, length)
            .map_err(PCodeError::Memory)
            .map_err(Error::state)
    }

    pub fn read_temporary_bytes(
        &self,
        start: u64,
        length: usize,
    ) -> Result<Vec<Expr>, Error> {
        Self::read_bytes(&self.temporaries, self.concrete.temporaries(), start, length)
            .map_err(PCodeError::Temporary)
            .map_err(Error::state)
    }

    pub fn read_operand_bytes(
        &self,
        operand: &Operand,
    ) -> Result<Vec<Expr>, Error> {
        match operand {
            Operand::Address { value, size } => {
                self.read_memory_bytes(value.into(), *size)
            },
            Operand::Constant { size, value, .. } => {
                let value = *value;
                let mut buf = [0u8; 8];

                value.into_bytes::<O>(&mut buf);

                let bufr = if O::ENDIAN.is_big() {
                    &buf[8-size..]
                } else {
                    &buf[..*size]
                };

                Ok(bufr.iter().map(|b| BitVec::from(*b).into()).collect::<Vec<_>>())
            },
            Operand::Register { .. } => {
                let register = operand.register().unwrap();
                self.read_register_bytes(&register)
            },
            Operand::Variable { offset, size, .. } => {
                self.read_temporary_bytes(*offset, *size)
            },
        }
    }

    // Treat each byte in the range as a separate entity
    fn read_bytes<T: StateOps<Value=u8>>(
        pages: &HashMap<u64, Page>,
        concs: &T,
        start: u64,
        length: usize,
    ) -> Result<Vec<Expr>, T::Error> {
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

    fn copy_mixed_byte_page<T: StateOps<Value=u8>>(
        pages: &mut HashMap<u64, Page>,
        concs: &mut T,
        from_sym: &[Option<Expr>],
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
                let cview = concs
                    .view_values_mut(to, this_length)?;
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
                let cview = concs
                    .view_values_mut(to + written as u64, this_length)?;
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
                concs
                    .set_values(to + written as u64, con_range)?;
            }
        }

        Ok(())
    }

    fn copy_mixed_bytes_forward<T: StateOps<Value=u8>>(
        pages: &mut HashMap<u64, Page>,
        concs: &mut T,
        from: u64,
        to: u64,
        length: usize,
    ) -> Result<(), T::Error> {
        let mut symbolic_page: [Option<Expr>; PAGE_SIZE] = Page::default().into();
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
                con_range.copy_from_slice(
                    concs
                        .view_values(page + offset as u64, length)?
                );

                Self::copy_mixed_byte_page(pages, concs, sym_range, con_range, to + written)?;
            } else {
                // ensure symbolic is all zeroed
                let sym_range = &mut symbolic_page[offset..offset + length];
                for exp in sym_range.iter_mut() {
                    *exp = None;
                }

                // get concrete range
                let con_range = &mut concrete_page[offset..offset + length];
                con_range.copy_from_slice(
                    concs
                        .view_values(page + offset as u64, length)?
                );

                Self::copy_mixed_byte_page(pages, concs, sym_range, con_range, to + written)?;
            }
            written += length as u64;
        }

        Ok(())
    }

    fn copy_mixed_bytes_backward<T: StateOps<Value=u8>>(
        pages: &mut HashMap<u64, Page>,
        concs: &mut T,
        from: u64,
        to: u64,
        length: usize,
    ) -> Result<(), T::Error> {
        let mut symbolic_page: [Option<Expr>; PAGE_SIZE] = Page::default().into();
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
                con_range.copy_from_slice(
                    concs
                        .view_values(page + offset as u64, length)?
                );

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
                con_range.copy_from_slice(
                    concs
                        .view_values(page + offset as u64, length)?
                );

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
            Self::copy_mixed_bytes_backward(&mut self.pages, self.concrete.memory_mut(), from, to, length)
        } else {
            Self::copy_mixed_bytes_forward(&mut self.pages, self.concrete.memory_mut(), from, to, length)
        }
        .map_err(PCodeError::Memory)
        .map_err(Error::state)
    }

    pub fn write_register_bytes<I: IntoIterator<Item = Expr>>(
        &mut self,
        register: &Register,
        it: I,
    ) {
        let start = register.offset();
        Self::write_bytes(&mut self.registers, start, it)
    }

    pub fn write_memory_bytes<I: IntoIterator<Item = Expr>>(&mut self, at: Address, it: I) {
        let start = u64::from(at);
        Self::write_bytes(&mut self.pages, start, it)
    }

    pub fn write_temporary_bytes<I: IntoIterator<Item = Expr>>(&mut self, at: u64, it: I) {
        Self::write_bytes(&mut self.temporaries, at, it)
    }

    pub fn write_operand_bytes<I: IntoIterator<Item = Expr>>(&mut self, operand: &Operand, it: I) {
        match operand {
            Operand::Address { value, .. } => {
                self.write_memory_bytes(value.into(), it)
            },
            Operand::Register { .. } => {
                let register = operand.register().unwrap();
                self.write_register_bytes(&register, it)
            },
            Operand::Variable { offset, .. } => {
                self.write_temporary_bytes(*offset, it)
            },
            _ => {
                panic!("write to constant: {}", operand)
            }
        }
    }

    fn write_bytes<I: IntoIterator<Item = Expr>>(
        pages: &mut HashMap<u64, Page>,
        at: u64,
        it: I,
    ) {
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
    ) -> Result<Option<Expr>, Error> {
        let start = register.offset();
        Self::read_primitive::<T, _>(&self.registers, self.concrete.registers(), start)
            .map_err(PCodeError::Register)
            .map_err(Error::state)
    }

    pub fn read_memory_primitive<T: ByteCast>(
        &self,
        at: Address,
    ) -> Result<Option<Expr>, Error> {
        let start = u64::from(at);
        Self::read_primitive::<T, _>(&self.pages, self.concrete.memory(), start)
            .map_err(PCodeError::Memory)
            .map_err(Error::state)
    }

    pub fn read_temporary_primitive<T: ByteCast>(
        &self,
        at: u64,
    ) -> Result<Option<Expr>, Error> {
        Self::read_primitive::<T, _>(&self.temporaries, self.concrete.temporaries(), at)
            .map_err(PCodeError::Temporary)
            .map_err(Error::state)
    }

    // Read a primitive using a given endianness
    fn read_primitive<T: ByteCast, S: StateOps<Value=u8>>(
        pages: &HashMap<u64, Page>,
        concs: &S,
        at: u64,
    ) -> Result<Option<Expr>, S::Error> {
        let buffer = Self::read_bytes(pages, concs, at, T::SIZEOF)?;
        Ok(if O::ENDIAN == Endian::Big {
            buffer.into_iter().fold1(|acc, v| Expr::concat(acc, v))
        } else {
            buffer.into_iter().fold1(|acc, v| Expr::concat(v, acc))
        })
    }

    pub fn write_register_primitive<T: ByteCast>(
        &mut self,
        register: &Register,
        expr: Expr,
    ) {
        let start = register.offset();
        Self::write_primitive::<T>(&mut self.registers, start, expr)
    }

    pub fn write_memory_primitive<T: ByteCast>(
        &mut self,
        at: Address,
        expr: Expr,
    ) {
        let start = u64::from(at);
        Self::write_primitive::<T>(&mut self.pages, start, expr)
    }

    pub fn write_temporary_primitive<T: ByteCast>(
        &mut self,
        at: u64,
        expr: Expr,
    ) {
        Self::write_primitive::<T>(&mut self.temporaries, at, expr)
    }

    fn write_primitive<T: ByteCast>(
        pages: &mut HashMap<u64, Page>,
        at: u64,
        expr: Expr,
    ) {
        let size = expr.bits();
        let bits = T::SIZEOF * 8;

        let expr = if bits == size {
            expr
        } else if bits < size {
            if T::SIGNED {
                Expr::cast_signed(expr, size)
            } else {
                Expr::cast_unsigned(expr, size)
            }
        } else {
            Expr::extract(expr, 0, size)
        };

        Self::write_expr(pages, at, expr)
    }

    pub fn write_register_expr(
        &mut self,
        register: &Register,
        expr: Expr,
    ) {
        let start = register.offset();
        Self::write_expr(&mut self.registers, start, expr)
    }

    pub fn write_memory_expr(
        &mut self,
        at: Address,
        expr: Expr,
    ) {
        let start = u64::from(at);
        Self::write_expr(&mut self.pages, start, expr)
    }

    pub fn write_temporary_expr(
        &mut self,
        at: u64,
        expr: Expr,
    ) {
        Self::write_expr(&mut self.temporaries, at, expr)
    }

    pub fn write_operand_expr(
        &mut self,
        operand: &Operand,
        expr: Expr,
    ) {
        assert_eq!(expr.bits(), operand.size() * 8);
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

    fn write_expr(
        pages: &mut HashMap<u64, Page>,
        at: u64,
        expr: Expr,
    ) {
        debug_assert_eq!(expr.bits() % 8, 0);
        let at = u64::from(at);
        let page_size = PAGE_SIZE as u64;
        let bits = expr.bits();
        let bytes = bits / 8;

        let mut it = (0..bytes).map(|i| {
            if O::ENDIAN == Endian::Big {
                let lsb = bits - (i * 8) - 8;
                Expr::extract(expr.clone(), lsb, lsb + 8)
            } else {
                let lsb = i * 8;
                Expr::extract(expr.clone(), lsb, lsb + 8)
            }
        });

        let aligned = at / page_size;
        let aligned_off = (at % page_size) as usize;

        let overlap = PAGE_SIZE - aligned_off;

        let page1 = overlap.min(bytes);
        let page2 = bytes - page1;

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

    pub fn write_operand_value(&mut self, operand: &Operand, value: Either<BitVec, Expr>) -> Result<(), Error> {
        match value {
            Either::Left(bv) => self.concretise_operand_with(operand, bv),
            Either::Right(expr) => {
                let val = expr.simplify(&self.solver);
                if let Expr::Val(bv) = val {
                    self.concretise_operand_with(operand, bv)?;
                } else {
                    self.write_operand_expr(operand, expr);
                }
                Ok(())
            },
        }
    }

    pub fn read_operand_value(&mut self, operand: &Operand) -> Result<Either<BitVec, Expr>, Error> {
        if self.is_symbolic_operand(operand) {
            let expr = self.read_operand_expr(operand)?.simplify(&self.solver);
            if let Expr::Val(bv) = expr {
                self.concretise_operand_with(operand, &bv)?;
                Ok(Either::Left(bv))
            } else {
                Ok(Either::Right(expr))
            }
        } else {
            let bv = self.concrete.get_operand::<BitVec>(operand).map_err(Error::state)?;
            Ok(Either::Left(bv))
        }
    }

    pub fn read_operand_expr(&self, operand: &Operand) -> Result<Expr, Error> {
        let vals = self.read_operand_bytes(operand)?;
        let expr = if O::ENDIAN.is_big() {
            vals.into_iter().fold1(|acc, v| Expr::concat(acc, v)).unwrap()
        } else {
            vals.into_iter().rev().fold1(|acc, v| Expr::concat(acc, v)).unwrap()
        };
        Ok(expr)
    }

    pub fn read_memory_expr(&self, addr: Address, bits: usize) -> Result<Expr, Error> {
        let vals = self.read_memory_bytes(addr, bits / 8)?;
        let expr = if O::ENDIAN.is_big() {
            vals.into_iter().fold1(|acc, v| Expr::concat(acc, v)).unwrap()
        } else {
            vals.into_iter().rev().fold1(|acc, v| Expr::concat(acc, v)).unwrap()
        };
        Ok(expr)
    }

    pub fn read_memory_symbolic(&mut self, addr: &Expr, bits: usize) -> Result<Expr, Error> {
        let addr = addr.simplify(&self.solver);

        if let Expr::Val(bv) = addr {
            let naddr = bv.to_u64().expect("64-bit address limit");
            self.read_memory_expr(self.solver.translator().address(naddr).into(), bits)
        } else {
            let addrs = addr.solve_many(&mut self.solver, &self.constraints);
            if addrs.is_empty() {
                return Err(Error::UnsatAddress(addr))
            }

            let mut init = Expr::from(BitVec::zero(bits)); // FIXME: we should not default to 0!
            for caddr in addrs {
                let addrv = caddr.to_u64().expect("64-bit address limit");
                let cval = self.read_memory_expr(self.solver.translator().address(addrv).into(), bits)?;
                let cond = Expr::int_eq(addr.clone(), caddr);
                init = Expr::ite(cond, cval, init);
            }

            Ok(init)
        }
    }

    fn write_memory_symbolic_aux(&mut self, aexpr: Expr, addr: Address, texpr: Expr) -> Result<(), Error> {
        let abits = aexpr.bits();
        let fexpr = self.read_memory_expr(addr, texpr.bits())?;

        let cond = Expr::int_eq(aexpr, BitVec::from_u64(addr.into(), abits));
        let current = Expr::ite(cond, texpr, fexpr);

        self.write_memory_expr(addr, current);

        Ok(())
    }

    pub fn write_memory_symbolic(&mut self, addr: &Expr, expr: Expr) -> Result<(), Error> {
        let addr = addr.simplify(&self.solver);

        if let Expr::Val(bv) = addr {
            let naddr = bv.to_u64().expect("64-bit address limit");
            self.write_memory_expr(self.solver.translator().address(naddr).into(), expr.clone());
        } else {
            for caddr in addr.solve_many(&mut self.solver, &self.constraints) {
                let addrv = caddr.to_u64().expect("64-bit address limit");
                self.write_memory_symbolic_aux(
                    addr.clone(),
                    self.solver.translator().address(addrv).into(),
                    expr.clone())?;
            }
        }

        Ok(())
    }
}
