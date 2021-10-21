use std::sync::Arc;

use either::Either;

use fugue::bv::BitVec;
use fugue::bytes::Order;
use fugue::fp::{self, float_format_from_size, Float, FloatFormat, FloatFormatOps};
use fugue::ir::{Address, AddressSpace, AddressValue, IntoAddress, Translator};
use fugue::ir::disassembly::ContextDatabase;
use fugue::ir::il::ecode::Expr;
use fugue::ir::il::pcode::Operand;

use fnv::FnvHashMap as HashMap;

use fuguex::intrinsics::{IntrinsicAction, IntrinsicHandler};
use fuguex::machine::{Branch, Interpreter, Outcome, StepState};
use fuguex::state::State;
use fuguex::state::pcode::PCodeState;

use parking_lot::RwLock;

use thiserror::Error;

use crate::state::{ConcolicState, Error as StateError};
use crate::value::Value;

#[derive(Debug, Error)]
pub enum Error {
    #[error("division by zero")]
    DivisionByZero,
    #[error(transparent)]
    Intrinsic(fuguex::intrinsics::Error<StateError>),
    #[error("error lifting instruction at {0}: {1}")]
    Lift(Address, #[source] fugue::ir::error::Error),
    #[error(transparent)]
    State(#[from] StateError),
    #[error("lifted instruction region {0}-{0}+{1} is symbolic")]
    SymbolicInstructions(Address, usize),
    #[error("program counter value is unsatisfiable")]
    UnsatisfiablePC,
    #[error("unsupported address size of {} bits", .0 * 8)]
    UnsupportedAddressSize(usize),
    #[error("unsupported branch destination in `{}` space", .0.name())]
    UnsupportedBranchDestination(Arc<AddressSpace>),
    #[error(transparent)]
    UnsupportedFloatFormat(#[from] fp::Error),
    #[error("unsupported operand size of {0} bytes; maximum supported is {1} bytes")]
    UnsupportedOperandSize(usize, usize),
}

pub enum NextLocation<O: Order> {
    Symbolic(Vec<(Branch, ConcolicState<O>)>),
    Concrete(Branch),
}

impl<O: Order> NextLocation<O> {
    fn join(self, tstate: ConcolicState<O>, fbranch: Branch, fstate: ConcolicState<O>) -> Self {
        match self {
            Self::Symbolic(mut states) => {
                states.push((fbranch, fstate));
                Self::Symbolic(states)
            },
            Self::Concrete(tbranch) => {
                Self::Symbolic(vec![(tbranch, tstate), (fbranch, fstate)])
            }
        }
    }

    fn unwrap_symbolic(self) -> Vec<(Branch, ConcolicState<O>)> {
        if let Self::Symbolic(states) = self {
            states
        } else {
            panic!("expected NextLocation::Symbolic")
        }
    }
}

pub type Outcomes<O> = Vec<(Branch, ConcolicState<O>)>;

#[derive(Clone)]
pub struct ConcolicContext<O: Order, const OPERAND_SIZE: usize> {
    program_counter: Arc<Operand>,
    state: ConcolicState<O>,
    translator: Arc<Translator>,
    translator_context: ContextDatabase,
    translator_cache: Arc<RwLock<HashMap<Address, StepState>>>,
    intrinsics: IntrinsicHandler<Outcomes<O>, ConcolicState<O>>,
}

trait ToSignedBytes {
    fn expand_as<O: Order, const OPERAND_SIZE: usize>(
        self,
        ctxt: &mut ConcolicContext<O, { OPERAND_SIZE }>,
        dest: &Operand,
        signed: bool,
    ) -> Result<(), Error>;
}

impl ToSignedBytes for bool {
    fn expand_as<O: Order, const OPERAND_SIZE: usize>(
        self,
        ctxt: &mut ConcolicContext<O, { OPERAND_SIZE }>,
        dest: &Operand,
        _signed: bool,
    ) -> Result<(), Error> {
        ctxt.state
            .concretise_operand_with(dest, self)
            .map_err(Error::State)?;

        Ok(())
    }
}

impl ToSignedBytes for BitVec {
    fn expand_as<O: Order, const OPERAND_SIZE: usize>(
        self,
        ctxt: &mut ConcolicContext<O, { OPERAND_SIZE }>,
        dest: &Operand,
        signed: bool,
    ) -> Result<(), Error> {
        let size = dest.size();

        if size > OPERAND_SIZE {
            return Err(Error::UnsupportedOperandSize(size, OPERAND_SIZE));
        }

        let dbits = size << 3;
        let target = if signed { self.signed() } else { self };
        let target = if target.bits() != dbits {
            target.cast(dbits)
        } else {
            target
        };

        ctxt.state
            .concretise_operand_with(dest, target)
            .map_err(Error::State)?;

        Ok(())
    }
}

impl ToSignedBytes for Expr {
    fn expand_as<O: Order, const OPERAND_SIZE: usize>(
        self,
        ctxt: &mut ConcolicContext<O, { OPERAND_SIZE }>,
        dest: &Operand,
        signed: bool,
    ) -> Result<(), Error> {
        let size = dest.size();

        if size > OPERAND_SIZE {
            return Err(Error::UnsupportedOperandSize(size, OPERAND_SIZE));
        }

        let dbits = size << 3;
        let target = if signed {
            Expr::cast_signed(self, dbits)
        } else if self.bits() != dbits {
            Expr::cast_unsigned(self, dbits)
        } else {
            self
        };

        ctxt.state.write_operand_value(dest, Either::Right(target))?;

        Ok(())
    }
}

impl<L, R> ToSignedBytes for Either<L, R>
where
    L: ToSignedBytes,
    R: ToSignedBytes,
{
    fn expand_as<O: Order, const OPERAND_SIZE: usize>(
        self,
        ctxt: &mut ConcolicContext<O, { OPERAND_SIZE }>,
        dest: &Operand,
        signed: bool,
    ) -> Result<(), Error> {
        match self {
            Either::Left(l) => l.expand_as(ctxt, dest, signed),
            Either::Right(r) => r.expand_as(ctxt, dest, signed),
        }
    }
}

impl<O: Order, const OPERAND_SIZE: usize> ConcolicContext<O, { OPERAND_SIZE }> {
    pub fn new(translator: Arc<Translator>, state: PCodeState<u8, O>) -> Self {
        Self {
            intrinsics: IntrinsicHandler::new(),
            program_counter: Arc::new(state.registers().program_counter().clone()),
            state: ConcolicState::new(translator.clone(), state),
            translator_cache: Arc::new(RwLock::new(HashMap::default())),
            translator_context: translator.context_database(),
            translator,
        }
    }

    pub fn push_constraint(&mut self, constraint: Expr) {
        self.state.push_constraint(constraint);
    }

    pub fn branch_expr(state: &mut ConcolicState<O>, expr: Expr) -> Result<NextLocation<O>, Error> {
        let mut states = Vec::new();
        let value = expr.simplify(&state.solver);

        let solns = value.solve_many(&mut state.solver, &state.constraints);

        for val in solns {
            let address = (&val).into_address_value(state.concrete.memory_space());
            let location = Branch::Global(address);

            let mut fork = state.fork();
            fork.push_constraint(Expr::int_eq(val, value.clone()));
            states.push((location, fork));
        }

        if states.is_empty() {
            Err(Error::UnsatisfiablePC)
        } else {
            Ok(NextLocation::Symbolic(states))
        }
    }

    pub fn branch_on(state: &mut ConcolicState<O>, pc: &Operand) -> Result<NextLocation<O>, Error> {
        match state.read_operand_value(pc)? {
            Either::Left(bv) => {
                let memory_space = state.concrete.memory_space();
                let location = Branch::Global(bv.into_address_value(memory_space));
                Ok(NextLocation::Concrete(location))
            }
            Either::Right(expr) => Self::branch_expr(state, expr),
        }
    }

    fn lift_bool1<COC, COS, COCO, COSO>(
        &mut self,
        opc: COC,
        ops: COS,
        dest: &Operand,
        rhs: &Operand,
    ) -> Result<Outcome<Outcomes<O>>, Error>
    where
        COC: FnOnce(bool) -> Result<COCO, Error>,
        COS: FnOnce(Expr) -> Result<COSO, Error>,
        COCO: ToSignedBytes,
        COSO: ToSignedBytes,
    {
        let rsize = rhs.size();
        if rsize > OPERAND_SIZE {
            return Err(Error::UnsupportedOperandSize(rsize, OPERAND_SIZE));
        }

        let value = match self.state.read_operand_value(rhs)? {
            Either::Left(bv) => Either::Left(opc(!bv.is_zero())?),
            Either::Right(expr) => Either::Right(ops(Expr::cast_bool(expr))?),
        };

        value.expand_as(self, dest, false)?;

        Ok(Outcome::Branch(Branch::Next))
    }

    fn lift_bool2<COC, COS, COCO, COSO>(
        &mut self,
        opc: COC,
        ops: COS,
        dest: &Operand,
        lhs: &Operand,
        rhs: &Operand,
    ) -> Result<Outcome<Outcomes<O>>, Error>
    where
        COC: FnOnce(bool, bool) -> Result<COCO, Error>,
        COS: FnOnce(Expr, Expr) -> Result<COSO, Error>,
        COCO: ToSignedBytes,
        COSO: ToSignedBytes,
    {
        let lsize = lhs.size();
        let rsize = rhs.size();

        if lsize > OPERAND_SIZE {
            return Err(Error::UnsupportedOperandSize(lsize, OPERAND_SIZE));
        }

        if rsize > OPERAND_SIZE {
            return Err(Error::UnsupportedOperandSize(rsize, OPERAND_SIZE));
        }

        let lhs_val = self
            .state
            .read_operand_value(lhs)?
            .map_right(|expr| Expr::cast_bool(expr));

        let rhs_val = self
            .state
            .read_operand_value(rhs)?
            .map_right(|expr| Expr::cast_bool(expr));

        let value = match (lhs_val, rhs_val) {
            (Either::Left(bv1), Either::Left(bv2)) => {
                Either::Left(opc(!bv1.is_zero(), !bv2.is_zero())?)
            }
            (Either::Left(bv), Either::Right(se)) => Either::Right(ops(bv.into(), se)?),
            (Either::Right(se), Either::Left(bv)) => Either::Right(ops(se, bv.into())?),
            (Either::Right(se1), Either::Right(se2)) => Either::Right(ops(se1, se2)?),
        };

        value.expand_as(self, dest, false)?;

        Ok(Outcome::Branch(Branch::Next))
    }

    fn lift_int1<COC, COS, COCO, COSO>(
        &mut self,
        opc: COC,
        ops: COS,
        dest: &Operand,
        rhs: &Operand,
        signed: bool,
    ) -> Result<Outcome<Outcomes<O>>, Error>
    where
        COC: FnOnce(BitVec) -> Result<COCO, Error>,
        COS: FnOnce(Expr) -> Result<COSO, Error>,
        COCO: ToSignedBytes,
        COSO: ToSignedBytes,
    {
        let rsize = rhs.size();
        if rsize > OPERAND_SIZE {
            return Err(Error::UnsupportedOperandSize(rsize, OPERAND_SIZE));
        }

        let value = match self.state.read_operand_value(rhs)? {
            Either::Left(bv) => Either::Left(opc(if signed { bv.signed() } else { bv })?),
            Either::Right(expr) => {
                let bits = expr.bits();
                Either::Right(ops(if signed {
                    Expr::cast_signed(expr, bits)
                } else {
                    expr
                })?)
            }
        };

        value.expand_as(self, dest, signed)?;

        Ok(Outcome::Branch(Branch::Next))
    }

    fn lift_int2<COC, COS, COCO, COSO>(
        &mut self,
        opc: COC,
        ops: COS,
        dest: &Operand,
        lhs: &Operand,
        rhs: &Operand,
        signed: bool,
    ) -> Result<Outcome<Outcomes<O>>, Error>
    where
        COC: FnOnce(BitVec, BitVec) -> Result<COCO, Error>,
        COS: FnOnce(Expr, Expr) -> Result<COSO, Error>,
        COCO: ToSignedBytes,
        COSO: ToSignedBytes,
    {
        let lsize = lhs.size();
        let lbits = lsize * 8;

        let rsize = rhs.size();

        if lsize > OPERAND_SIZE {
            return Err(Error::UnsupportedOperandSize(lsize, OPERAND_SIZE));
        }

        if rsize > OPERAND_SIZE {
            return Err(Error::UnsupportedOperandSize(rsize, OPERAND_SIZE));
        }

        let lhs_val = self
            .state
            .read_operand_value(lhs)?
            .map_left(|bv| bv.signed())
            .map_right(|expr| {
                let bits = expr.bits();
                Expr::cast_signed(expr, bits)
            });

        let rhs_val = self
            .state
            .read_operand_value(rhs)?
            .map_left(|bv| {
                if signed {
                    bv.signed_cast(lsize * 8)
                } else {
                    bv
                }
            })
            .map_right(|expr| {
                if signed {
                    Expr::cast_signed(expr, lbits)
                } else {
                    Expr::cast_unsigned(expr, lbits)
                }
            });

        let value = match (lhs_val, rhs_val) {
            (Either::Left(bv1), Either::Left(bv2)) => Either::Left(opc(bv1, bv2)?),
            (Either::Left(bv), Either::Right(se)) => Either::Right(ops(bv.into(), se)?),
            (Either::Right(se), Either::Left(bv)) => Either::Right(ops(se, bv.into())?),
            (Either::Right(se1), Either::Right(se2)) => Either::Right(ops(se1, se2)?),
        };

        value.expand_as(self, dest, signed)?;

        Ok(Outcome::Branch(Branch::Next))
    }

    fn lift_float1<COC, COS, COCO, COSO>(
        &mut self,
        opc: COC,
        ops: COS,
        dest: &Operand,
        rhs: &Operand,
    ) -> Result<Outcome<Outcomes<O>>, Error>
    where
        COC: FnOnce(Float, &FloatFormat) -> Result<COCO, Error>,
        COS: FnOnce(Expr, &HashMap<usize, Arc<FloatFormat>>) -> Result<COSO, Error>,
        COCO: ToSignedBytes,
        COSO: ToSignedBytes,
    {
        let rsize = rhs.size();
        if rsize > OPERAND_SIZE {
            return Err(Error::UnsupportedOperandSize(rsize, OPERAND_SIZE));
        }

        let format = float_format_from_size(rsize)?;

        let value = match self.state.read_operand_value(rhs)? {
            Either::Left(bv) => Either::Left(opc(format.from_bitvec(&bv), &format)?),
            Either::Right(expr) => {
                let formats = self.state.solver.translator.float_formats();
                Either::Right(ops(Expr::cast_float(expr, Arc::new(format)), formats)?)
            }
        };

        value.expand_as(self, dest, true)?;

        Ok(Outcome::Branch(Branch::Next))
    }

    fn lift_float2<COC, COS, COCO, COSO>(
        &mut self,
        opc: COC,
        ops: COS,
        dest: &Operand,
        lhs: &Operand,
        rhs: &Operand,
    ) -> Result<Outcome<Outcomes<O>>, Error>
    where
        COC: FnOnce(Float, Float, &FloatFormat) -> Result<COCO, Error>,
        COS: FnOnce(Expr, Expr, &HashMap<usize, Arc<FloatFormat>>) -> Result<COSO, Error>,
        COCO: ToSignedBytes,
        COSO: ToSignedBytes,
    {
        let lsize = lhs.size();
        let rsize = rhs.size();

        if lsize > OPERAND_SIZE {
            return Err(Error::UnsupportedOperandSize(lsize, OPERAND_SIZE));
        }

        if rsize > OPERAND_SIZE {
            return Err(Error::UnsupportedOperandSize(rsize, OPERAND_SIZE));
        }

        assert_eq!(lsize, rsize);

        let format = Arc::new(float_format_from_size(rsize)?);

        let lhs_val = self
            .state
            .read_operand_value(lhs)?
            .map_right(|expr| Expr::cast_float(expr, format.clone()));

        let rhs_val = self
            .state
            .read_operand_value(rhs)?
            .map_right(|expr| Expr::cast_float(expr, format.clone()));

        let formats = self.state.solver.translator.float_formats();
        let value = match (lhs_val, rhs_val) {
            (Either::Left(bv1), Either::Left(bv2)) => Either::Left(opc(
                format.from_bitvec(&bv1),
                format.from_bitvec(&bv2),
                &format,
            )?),
            (Either::Left(bv), Either::Right(se)) => {
                Either::Right(ops(Expr::cast_float(bv, format.clone()), se, formats)?)
            }
            (Either::Right(se), Either::Left(bv)) => {
                Either::Right(ops(se, Expr::cast_float(bv, format.clone()), formats)?)
            }
            (Either::Right(se1), Either::Right(se2)) => Either::Right(ops(se1, se2, formats)?),
        };

        value.expand_as(self, dest, true)?;

        Ok(Outcome::Branch(Branch::Next))
    }

    pub fn restore_state(&mut self, state: ConcolicState<O>) {
        self.state = state;
    }
}

impl<O: Order, const OPERAND_SIZE: usize> Interpreter for ConcolicContext<O, { OPERAND_SIZE }> {
    type State = ConcolicState<O>;
    type Error = Error;
    type Outcome = Vec<(Branch, Self::State)>;

    fn fork(&self) -> Self {
        Self {
            program_counter: self.program_counter.clone(),
            state: self.state.fork(),
            translator: self.translator.clone(),
            translator_context: self.translator_context.clone(),
            translator_cache: self.translator_cache.clone(),
            intrinsics: self.intrinsics.clone(),
        }
    }

    fn restore(&mut self, other: &Self) {
        self.state.restore(&other.state);
    }

    fn branch(&mut self, destination: &Operand) -> Result<Outcome<Self::Outcome>, Self::Error> {
        match destination {
            Operand::Constant { value, .. } => {
                let action = Branch::Local(*value as isize);
                Ok(Outcome::Branch(action))
            }
            Operand::Address { value, .. } => {
                let action = Branch::Global(value.clone());
                Ok(Outcome::Branch(action))
            }
            Operand::Register { space, .. } | Operand::Variable { space, .. } => {
                return Err(Error::UnsupportedBranchDestination(space.clone()))
            }
        }
    }

    fn ibranch(&mut self, destination: &Operand) -> Result<Outcome<Self::Outcome>, Self::Error> {
        if destination == self.state.concrete.registers().program_counter() {
            return self.icall(destination);
        }

        match Self::branch_on(&mut self.state, destination)? {
            NextLocation::Concrete(location) => Ok(Outcome::Branch(location)),
            NextLocation::Symbolic(states) => Ok(Outcome::Halt(states)),
        }
    }

    fn cbranch(
        &mut self,
        destination: &Operand,
        condition: &Operand,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        match self.state.read_operand_value(condition)? {
            Either::Left(bv) => {
                if bv.is_zero() {
                    Ok(Outcome::Branch(Branch::Next))
                } else {
                    self.branch(destination)
                }
            }
            Either::Right(expr) => {
                let mut tstate = self.state.fork();
                let mut fstate = self.state.fork();

                tstate.push_constraint(Expr::bool_eq(expr.clone(), BitVec::from_u8(1, 8)));

                let states = Self::branch_on(&mut tstate, destination)?;

                fstate.push_constraint(Expr::bool_eq(expr, BitVec::from_u8(0, 8)));

                Ok(Outcome::Halt(states.join(tstate, Branch::Next, fstate).unwrap_symbolic()))
            }
        }
    }

    fn call(&mut self, destination: &Operand) -> Result<Outcome<Self::Outcome>, Self::Error> {
        match destination {
            Operand::Address { value, .. } => Ok(Outcome::Branch(Branch::Global(value.clone()))),
            Operand::Constant { space, .. }
            | Operand::Register { space, .. }
            | Operand::Variable { space, .. } => {
                Err(Error::UnsupportedBranchDestination(space.clone()))
            }
        }
    }

    fn icall(&mut self, destination: &Operand) -> Result<Outcome<Self::Outcome>, Self::Error> {
        match Self::branch_on(&mut self.state, destination)? {
            NextLocation::Concrete(location) => Ok(Outcome::Branch(location)),
            NextLocation::Symbolic(states) => Ok(Outcome::Halt(states)),
        }
    }

    fn return_(&mut self, destination: &Operand) -> Result<Outcome<Self::Outcome>, Self::Error> {
        self.ibranch(destination)
    }

    fn copy(
        &mut self,
        source: &Operand,
        destination: &Operand,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        let size = source.size();
        if size > OPERAND_SIZE {
            return Err(Error::UnsupportedOperandSize(size, OPERAND_SIZE));
        }

        let sval = self.state.read_operand_value(source)?;
        self.state.write_operand_value(destination, sval)?;

        Ok(Outcome::Branch(Branch::Next))
    }

    fn load(
        &mut self,
        source: &Operand,
        destination: &Operand,
        space: Arc<AddressSpace>,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        let space_size = space.address_size();
        let space_word_size = space.word_size() as u64;

        assert_eq!(space_size, source.size());

        match self.state.read_operand_value(source)? {
            Either::Left(bv) => {
                let offset = bv
                    .to_u64()
                    .ok_or_else(|| Error::UnsupportedAddressSize(bv.bits() * 8))?;
                let addr_val = offset.wrapping_mul(space_word_size)
                    & 1u64
                        .checked_shl(space_size.checked_shl(3).unwrap_or(0) as u32)
                        .unwrap_or(0)
                        .wrapping_sub(1);

                let address = Operand::Address {
                    value: AddressValue::new(space, addr_val),
                    size: destination.size(),
                };

                self.copy(&address, destination)
            }
            Either::Right(expr) => {
                let bits = expr.bits();
                let addr = {
                    let off = Expr::int_mul(expr, BitVec::from_u64(space_word_size, bits));
                    let msk = BitVec::from_u64(
                        1u64.checked_shl(space_size.checked_shl(3).unwrap_or(0) as u32)
                            .unwrap_or(0)
                            .wrapping_sub(1),
                        bits,
                    );
                    Expr::int_and(off, msk)
                };

                let value = self
                    .state
                    .read_memory_symbolic(&addr, destination.size() * 8)?;
                self.state.write_operand_expr(destination, value);

                Ok(Outcome::Branch(Branch::Next))
            }
        }
    }

    fn store(
        &mut self,
        source: &Operand,
        destination: &Operand,
        space: Arc<AddressSpace>,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        let space_size = space.address_size();
        let space_word_size = space.word_size() as u64;

        assert_eq!(space_size, destination.size());

        match self.state.read_operand_value(destination)? {
            Either::Left(bv) => {
                let offset = bv
                    .to_u64()
                    .ok_or_else(|| Error::UnsupportedAddressSize(bv.bits() * 8))?;
                let addr_val = offset.wrapping_mul(space_word_size)
                    & 1u64
                        .checked_shl(space_size.checked_shl(3).unwrap_or(0) as u32)
                        .unwrap_or(0)
                        .wrapping_sub(1);

                let address = Operand::Address {
                    value: AddressValue::new(space, addr_val),
                    size: source.size(),
                };

                self.copy(source, &address)
            }
            Either::Right(expr) => {
                let bits = expr.bits();
                let addr = {
                    let off = Expr::int_mul(expr, BitVec::from_u64(space_word_size, bits));
                    let msk = BitVec::from_u64(
                        1u64.checked_shl(space_size.checked_shl(3).unwrap_or(0) as u32)
                            .unwrap_or(0)
                            .wrapping_sub(1),
                        bits,
                    );
                    Expr::int_and(off, msk)
                };

                let value = self
                    .state
                    .read_operand_value(source)?
                    .either(|bv| Expr::from(bv), |expr| expr);

                self.state.write_memory_symbolic(&addr, value)?;

                Ok(Outcome::Branch(Branch::Next))
            }
        }
    }

    fn int_eq(
        &mut self,
        destination: &Operand,
        operand1: &Operand,
        operand2: &Operand,
    ) -> Result<Outcome<Self::Outcome>, Error> {
        self.lift_int2(
            |u, v| Ok(u == v),
            |u, v| Ok(Expr::int_eq(u, v)),
            destination,
            operand1,
            operand2,
            false,
        )
    }

    fn int_not_eq(
        &mut self,
        destination: &Operand,
        operand1: &Operand,
        operand2: &Operand,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        self.lift_int2(
            |u, v| Ok(u != v),
            |u, v| Ok(Expr::int_neq(u, v)),
            destination,
            operand1,
            operand2,
            false,
        )
    }

    fn int_less(
        &mut self,
        destination: &Operand,
        operand1: &Operand,
        operand2: &Operand,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        self.lift_int2(
            |u, v| Ok(u < v),
            |u, v| Ok(Expr::int_lt(u, v)),
            destination,
            operand1,
            operand2,
            false,
        )
    }

    fn int_less_eq(
        &mut self,
        destination: &Operand,
        operand1: &Operand,
        operand2: &Operand,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        self.lift_int2(
            |u, v| Ok(u <= v),
            |u, v| Ok(Expr::int_le(u, v)),
            destination,
            operand1,
            operand2,
            false,
        )
    }

    fn int_sless(
        &mut self,
        destination: &Operand,
        operand1: &Operand,
        operand2: &Operand,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        self.lift_int2(
            |u, v| Ok(u < v),
            |u, v| Ok(Expr::int_slt(u, v)),
            destination,
            operand1,
            operand2,
            true,
        )
    }

    fn int_sless_eq(
        &mut self,
        destination: &Operand,
        operand1: &Operand,
        operand2: &Operand,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        self.lift_int2(
            |u, v| Ok(u <= v),
            |u, v| Ok(Expr::int_sle(u, v)),
            destination,
            operand1,
            operand2,
            true,
        )
    }

    fn int_zext(
        &mut self,
        destination: &Operand,
        operand: &Operand,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        self.lift_int1(|u| Ok(u), |u| Ok(u), destination, operand, false)
    }

    fn int_sext(
        &mut self,
        destination: &Operand,
        operand: &Operand,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        self.lift_int1(|u| Ok(u), |u| Ok(u), destination, operand, true)
    }

    fn int_add(
        &mut self,
        destination: &Operand,
        operand1: &Operand,
        operand2: &Operand,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        self.lift_int2(
            |u, v| Ok(u + v),
            |u, v| Ok(Expr::int_add(u, v)),
            destination,
            operand1,
            operand2,
            false,
        )
    }

    fn int_sub(
        &mut self,
        destination: &Operand,
        operand1: &Operand,
        operand2: &Operand,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        self.lift_int2(
            |u, v| Ok(u - v),
            |u, v| Ok(Expr::int_sub(u, v)),
            destination,
            operand1,
            operand2,
            false,
        )
    }

    fn int_carry(
        &mut self,
        destination: &Operand,
        operand1: &Operand,
        operand2: &Operand,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        self.lift_int2(
            |u, v| Ok(u.carry(&v)),
            |u, v| Ok(Expr::int_carry(u, v)),
            destination,
            operand1,
            operand2,
            false,
        )
    }

    fn int_scarry(
        &mut self,
        destination: &Operand,
        operand1: &Operand,
        operand2: &Operand,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        self.lift_int2(
            |u, v| Ok(u.signed_carry(&v)),
            |u, v| Ok(Expr::int_scarry(u, v)),
            destination,
            operand1,
            operand2,
            true,
        )
    }

    fn int_sborrow(
        &mut self,
        destination: &Operand,
        operand1: &Operand,
        operand2: &Operand,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        self.lift_int2(
            |u, v| Ok(u.signed_borrow(&v)),
            |u, v| Ok(Expr::int_sborrow(u, v)),
            destination,
            operand1,
            operand2,
            true,
        )
    }

    fn int_neg(
        &mut self,
        destination: &Operand,
        operand: &Operand,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        self.lift_int1(
            |u| Ok(-u),
            |u| Ok(Expr::int_neg(u)),
            destination,
            operand,
            true,
        )
    }

    fn int_not(
        &mut self,
        destination: &Operand,
        operand: &Operand,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        self.lift_int1(
            |u| Ok(!u),
            |u| Ok(Expr::int_not(u)),
            destination,
            operand,
            false,
        )
    }

    fn int_xor(
        &mut self,
        destination: &Operand,
        operand1: &Operand,
        operand2: &Operand,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        self.lift_int2(
            |u, v| Ok(u ^ v),
            |u, v| Ok(Expr::int_xor(u, v)),
            destination,
            operand1,
            operand2,
            false,
        )
    }

    fn int_and(
        &mut self,
        destination: &Operand,
        operand1: &Operand,
        operand2: &Operand,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        self.lift_int2(
            |u, v| Ok(u & v),
            |u, v| Ok(Expr::int_and(u, v)),
            destination,
            operand1,
            operand2,
            false,
        )
    }

    fn int_or(
        &mut self,
        destination: &Operand,
        operand1: &Operand,
        operand2: &Operand,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        self.lift_int2(
            |u, v| Ok(u | v),
            |u, v| Ok(Expr::int_or(u, v)),
            destination,
            operand1,
            operand2,
            false,
        )
    }

    fn int_left_shift(
        &mut self,
        destination: &Operand,
        operand1: &Operand,
        operand2: &Operand,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        self.lift_int2(
            |u, v| Ok(u << v),
            |u, v| Ok(Expr::int_shl(u, v)),
            destination,
            operand1,
            operand2,
            false,
        )
    }

    fn int_right_shift(
        &mut self,
        destination: &Operand,
        operand1: &Operand,
        operand2: &Operand,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        self.lift_int2(
            |u, v| Ok(u >> v),
            |u, v| Ok(Expr::int_shr(u, v)),
            destination,
            operand1,
            operand2,
            false,
        )
    }

    fn int_sright_shift(
        &mut self,
        destination: &Operand,
        operand1: &Operand,
        operand2: &Operand,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        self.lift_int2(
            |u, v| Ok(u >> v),
            |u, v| Ok(Expr::int_sar(u, v)),
            destination,
            operand1,
            operand2,
            true,
        )
    }

    fn int_mul(
        &mut self,
        destination: &Operand,
        operand1: &Operand,
        operand2: &Operand,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        self.lift_int2(
            |u, v| Ok(u * v),
            |u, v| Ok(Expr::int_mul(u, v)),
            destination,
            operand1,
            operand2,
            false,
        )
    }

    fn int_div(
        &mut self,
        destination: &Operand,
        operand1: &Operand,
        operand2: &Operand,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        self.lift_int2(
            |u, v| {
                if v.is_zero() {
                    Err(Error::DivisionByZero)
                } else {
                    Ok(u / v)
                }
            },
            |u, v| Ok(Expr::int_div(u, v)),
            destination,
            operand1,
            operand2,
            false,
        )
    }

    fn int_sdiv(
        &mut self,
        destination: &Operand,
        operand1: &Operand,
        operand2: &Operand,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        self.lift_int2(
            |u, v| {
                if v.is_zero() {
                    Err(Error::DivisionByZero)
                } else {
                    Ok(u / v)
                }
            },
            |u, v| Ok(Expr::int_sdiv(u, v)),
            destination,
            operand1,
            operand2,
            true,
        )
    }

    fn int_rem(
        &mut self,
        destination: &Operand,
        operand1: &Operand,
        operand2: &Operand,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        self.lift_int2(
            |u, v| {
                if v.is_zero() {
                    Err(Error::DivisionByZero)
                } else {
                    Ok(u % v)
                }
            },
            |u, v| Ok(Expr::int_rem(u, v)),
            destination,
            operand1,
            operand2,
            false,
        )
    }

    fn int_srem(
        &mut self,
        destination: &Operand,
        operand1: &Operand,
        operand2: &Operand,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        self.lift_int2(
            |u, v| {
                if v.is_zero() {
                    Err(Error::DivisionByZero)
                } else {
                    Ok(u % v)
                }
            },
            |u, v| Ok(Expr::int_srem(u, v)),
            destination,
            operand1,
            operand2,
            true,
        )
    }

    fn bool_not(
        &mut self,
        destination: &Operand,
        operand: &Operand,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        self.lift_bool1(|u| Ok(!u), |u| Ok(Expr::bool_not(u)), destination, operand)
    }

    fn bool_xor(
        &mut self,
        destination: &Operand,
        operand1: &Operand,
        operand2: &Operand,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        self.lift_bool2(
            |u, v| Ok(u ^ v),
            |u, v| Ok(Expr::bool_xor(u, v)),
            destination,
            operand1,
            operand2,
        )
    }

    fn bool_and(
        &mut self,
        destination: &Operand,
        operand1: &Operand,
        operand2: &Operand,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        self.lift_bool2(
            |u, v| Ok(u & v),
            |u, v| Ok(Expr::bool_and(u, v)),
            destination,
            operand1,
            operand2,
        )
    }

    fn bool_or(
        &mut self,
        destination: &Operand,
        operand1: &Operand,
        operand2: &Operand,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        self.lift_bool2(
            |u, v| Ok(u | v),
            |u, v| Ok(Expr::bool_or(u, v)),
            destination,
            operand1,
            operand2,
        )
    }

    fn float_eq(
        &mut self,
        destination: &Operand,
        operand1: &Operand,
        operand2: &Operand,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        self.lift_float2(
            |u, v, _fmt| Ok(u == v),
            |u, v, fmts| Ok(Expr::float_eq(u, v, fmts)),
            destination,
            operand1,
            operand2,
        )
    }

    fn float_not_eq(
        &mut self,
        destination: &Operand,
        operand1: &Operand,
        operand2: &Operand,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        self.lift_float2(
            |u, v, _fmt| Ok(u != v),
            |u, v, fmts| Ok(Expr::float_neq(u, v, fmts)),
            destination,
            operand1,
            operand2,
        )
    }

    fn float_less(
        &mut self,
        destination: &Operand,
        operand1: &Operand,
        operand2: &Operand,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        self.lift_float2(
            |u, v, _fmt| Ok(u < v),
            |u, v, fmts| Ok(Expr::float_le(u, v, fmts)),
            destination,
            operand1,
            operand2,
        )
    }

    fn float_less_eq(
        &mut self,
        destination: &Operand,
        operand1: &Operand,
        operand2: &Operand,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        self.lift_float2(
            |u, v, _fmt| Ok(u <= v),
            |u, v, fmts| Ok(Expr::float_le(u, v, fmts)),
            destination,
            operand1,
            operand2,
        )
    }

    fn float_is_nan(
        &mut self,
        destination: &Operand,
        operand: &Operand,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        self.lift_float1(
            |u, _fmt| Ok(u.is_nan()),
            |u, fmts| Ok(Expr::float_nan(u, fmts)),
            destination,
            operand,
        )
    }

    fn float_add(
        &mut self,
        destination: &Operand,
        operand1: &Operand,
        operand2: &Operand,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        self.lift_float2(
            |u, v, fmt| Ok(fmt.into_bitvec(u + v, fmt.bits())),
            |u, v, fmts| Ok(Expr::float_add(u, v, fmts)),
            destination,
            operand1,
            operand2,
        )
    }

    fn float_div(
        &mut self,
        destination: &Operand,
        operand1: &Operand,
        operand2: &Operand,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        self.lift_float2(
            |u, v, fmt| Ok(fmt.into_bitvec(u / v, fmt.bits())),
            |u, v, fmts| Ok(Expr::float_div(u, v, fmts)),
            destination,
            operand1,
            operand2,
        )
    }

    fn float_mul(
        &mut self,
        destination: &Operand,
        operand1: &Operand,
        operand2: &Operand,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        self.lift_float2(
            |u, v, fmt| Ok(fmt.into_bitvec(u * v, fmt.bits())),
            |u, v, fmts| Ok(Expr::float_mul(u, v, fmts)),
            destination,
            operand1,
            operand2,
        )
    }

    fn float_sub(
        &mut self,
        destination: &Operand,
        operand1: &Operand,
        operand2: &Operand,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        self.lift_float2(
            |u, v, fmt| Ok(fmt.into_bitvec(u - v, fmt.bits())),
            |u, v, fmts| Ok(Expr::float_sub(u, v, fmts)),
            destination,
            operand1,
            operand2,
        )
    }

    fn float_neg(
        &mut self,
        destination: &Operand,
        operand: &Operand,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        self.lift_float1(
            |u, fmt| Ok(fmt.into_bitvec(-u, fmt.bits())),
            |u, fmts| Ok(Expr::float_neg(u, fmts)),
            destination,
            operand,
        )
    }

    fn float_abs(
        &mut self,
        destination: &Operand,
        operand: &Operand,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        self.lift_float1(
            |u, fmt| Ok(fmt.into_bitvec(u.abs(), fmt.bits())),
            |u, fmts| Ok(Expr::float_abs(u, fmts)),
            destination,
            operand,
        )
    }

    fn float_sqrt(
        &mut self,
        destination: &Operand,
        operand: &Operand,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        self.lift_float1(
            |u, fmt| Ok(fmt.into_bitvec(u.sqrt(), fmt.bits())),
            |u, fmts| Ok(Expr::float_sqrt(u, fmts)),
            destination,
            operand,
        )
    }

    fn float_of_int(
        &mut self,
        destination: &Operand,
        operand: &Operand,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        let fmt = Arc::new(float_format_from_size(destination.size())?);
        self.lift_int1(
            |u| {
                let ival = u.as_bigint().into_owned();
                let fval = Float::from_bigint(fmt.frac_size, fmt.exp_size, ival);
                Ok(fmt.into_bitvec(fval, fmt.bits()))
            },
            |u| Ok(Expr::cast_float(u, fmt.clone())),
            destination,
            operand,
            true,
        )
    }

    fn float_of_float(
        &mut self,
        destination: &Operand,
        operand: &Operand,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        let fmt = float_format_from_size(destination.size())?;
        self.lift_float1(
            |u, fmt| Ok(fmt.into_bitvec(u, fmt.bits())),
            |u, _fmts| Ok(Expr::cast_float(u, Arc::new(fmt))),
            destination,
            operand,
        )
    }

    fn float_truncate(
        &mut self,
        destination: &Operand,
        operand: &Operand,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        self.lift_float1(
            |u, _fmts| Ok(u.trunc_into_bitvec(destination.size() * 8)),
            |u, _fmts| Ok(Expr::cast_signed(u, destination.size() * 8)),
            destination,
            operand,
        )
    }

    fn float_ceiling(
        &mut self,
        destination: &Operand,
        operand: &Operand,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        self.lift_float1(
            |u, fmt| Ok(fmt.into_bitvec(u.ceil(), fmt.bits())),
            |u, fmts| Ok(Expr::float_ceiling(u, fmts)),
            destination,
            operand,
        )
    }

    fn float_floor(
        &mut self,
        destination: &Operand,
        operand: &Operand,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        self.lift_float1(
            |u, fmt| Ok(fmt.into_bitvec(u.floor(), fmt.bits())),
            |u, fmts| Ok(Expr::float_floor(u, fmts)),
            destination,
            operand,
        )
    }

    fn float_round(
        &mut self,
        destination: &Operand,
        operand: &Operand,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        self.lift_float1(
            |u, fmt| Ok(fmt.into_bitvec(u.round(), fmt.bits())),
            |u, fmts| Ok(Expr::float_round(u, fmts)),
            destination,
            operand,
        )
    }

    fn pop_count(
        &mut self,
        destination: &Operand,
        operand: &Operand,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        self.lift_int1(
            |u| Ok(BitVec::from(u.count_ones())),
            |u| Ok(Expr::count_ones(u)),
            destination,
            operand,
            false,
        )
    }

    fn subpiece(&mut self, destination: &Operand, operand: &Operand, amount: &Operand) -> Result<Outcome<Self::Outcome>, Self::Error> {
        let amount_size = amount.size();
        if amount_size > OPERAND_SIZE {
            return Err(Error::UnsupportedOperandSize(amount_size, OPERAND_SIZE));
        }

        let input_size = operand.size();
        if input_size > OPERAND_SIZE {
            return Err(Error::UnsupportedOperandSize(input_size, OPERAND_SIZE));
        }

        let destination_size = destination.size();
        if destination_size > OPERAND_SIZE {
            return Err(Error::UnsupportedOperandSize(
                destination_size,
                OPERAND_SIZE,
            ));
        }

        // NOTE: amount is always constant
        let value = match self.state.read_operand_value(operand)? {
            Either::Left(bv) => {
                let amount = amount.offset() as usize;

                let mut input_buf = [0u8; OPERAND_SIZE];
                let input_view = &mut input_buf[..input_size];

                bv.into_bytes::<O>(input_view);

                let mut output_buf = [0u8; OPERAND_SIZE];
                let output_view = &mut output_buf[..destination_size];

                O::subpiece(output_view, input_view, amount);

                Either::Left(BitVec::from_bytes::<O>(&output_view, false))
            },
            Either::Right(expr) => {
                let src_size = expr.bits();
                let out_size = destination.size() * 8;

                let loff = amount.offset() as usize * 8;
                let trun_size = src_size.checked_sub(loff).unwrap_or(0);

                let trun = if out_size > trun_size {
                    // extract high + expand
                    let source_htrun = Expr::extract_high(expr, trun_size);
                    Expr::cast_unsigned(source_htrun, out_size)
                } else {
                    // extract
                    let hoff = loff + out_size;
                    Expr::extract(expr, loff, hoff)
                };

                Either::Right(trun)
            },
        };

        self.state.write_operand_value(destination, value)?;

        Ok(Outcome::Branch(Branch::Next))
    }

    fn intrinsic(&mut self, name: &str, operands: &[Operand], result: Option<&Operand>) -> Result<Outcome<Self::Outcome>, Self::Error> {
        let outcome = self.intrinsics.handle(name, &mut self.state, operands, result)
            .map_err(Error::Intrinsic)?;

        Ok(match outcome {
            IntrinsicAction::Pass => Outcome::Branch(Branch::Next),
            IntrinsicAction::Branch(address) => Outcome::Branch(Branch::Global(address)),
            IntrinsicAction::Halt(reason) => Outcome::Halt(reason),
        })
    }

    fn lift<A>(&mut self, address: A) -> Result<StepState, Self::Error>
    where A: IntoAddress {
        let address_value = address.into_address_value(self.interpreter_space());
        let address = Address::from(&address_value);

        // begin read lock region
        let rlock = self.translator_cache.read();

        let cached = rlock.get(&address)
            .map(|step_state| step_state.clone());

        drop(rlock);
        // end read lock region

        let step_state = if let Some(step_state) = cached {
            let diff = (step_state.fallthrough().offset() - step_state.address().offset()) as usize;
            if self.state.is_symbolic_memory(address, diff) {
                return Err(Error::SymbolicInstructions(address, diff))
            } else {
                step_state.clone()
            }
        } else {
            let view = self
                .state
                .concrete
                .view_values_from(&address)
                .map_err(StateError::State)?;

            let step_state = StepState::from(
                self.translator
                    .lift_pcode(&mut self.translator_context, address_value, view)
                    .map_err(|e| Error::Lift(address, e))?
            );

            let diff = (step_state.fallthrough().offset() - step_state.address().offset()) as usize;
            if self.state.is_symbolic_memory(address, diff) {
                return Err(Error::SymbolicInstructions(address, diff))
            }

            self.translator_cache
                .write()
                .insert(address, step_state.clone());

            step_state
        };

        self.state
            .concrete
            .set_address(&self.program_counter, address)
            .map_err(StateError::State)?;

        self.state.concretise_operand(&self.program_counter);

        Ok(step_state)
    }

    fn interpreter_space(&self) -> Arc<AddressSpace> {
        self.state.concrete.memory_space()
    }
}
