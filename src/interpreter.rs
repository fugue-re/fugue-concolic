use std::marker::PhantomData;
use std::sync::Arc;

use either::Either;

use fugue::bv::BitVec;
use fugue::bytes::Order;
use fugue::fp::{self, float_format_from_size, Float, FloatFormat, FloatFormatOps};
use fugue::ir::{Address, AddressSpace, AddressSpaceId, AddressValue, IntoAddress, Translator};
use fugue::ir::disassembly::ContextDatabase;
use fugue::ir::il::pcode::Operand;

use fnv::FnvHashMap as HashMap;

use fuguex::hooks::types::HookCallAction;
use fuguex::intrinsics::{IntrinsicAction, IntrinsicHandler};
use fuguex::machine::{Branch, Interpreter, Outcome, StepState};
use fuguex::state::State;
use fuguex::state::pcode::PCodeState;
use fuguex::state::register::ReturnLocation;

use parking_lot::RwLock;

use thiserror::Error;

use crate::backend::ValueSolver;
use crate::expr::SymExpr;
use crate::hooks::ClonableHookConcolic;
use crate::pointer::SymbolicPointerStrategy;
use crate::state::{ConcolicState, Error as StateError};

#[derive(Debug, Error)]
pub enum Error {
    #[error("division by zero")]
    DivisionByZero,
    #[error(transparent)]
    Hook(fuguex::hooks::types::Error<StateError>),
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
    #[error("unsupported branch destination")]
    UnsupportedBranchDestination,
    #[error(transparent)]
    UnsupportedFloatFormat(#[from] fp::Error),
    #[error("unsupported operand size of {0} bytes; maximum supported is {1} bytes")]
    UnsupportedOperandSize(usize, usize),
}

pub enum NextLocation<'ctx, O: Order, VS: ValueSolver<'ctx>> {
    Symbolic(Vec<(Branch, ConcolicState<'ctx, O, VS>)>),
    Concrete(Branch, PhantomData<&'ctx VS>),
}

impl<'ctx, O: Order, VS: ValueSolver<'ctx>> NextLocation<'ctx, O, VS> {
    fn join(self, tstate: ConcolicState<'ctx, O, VS>, fbranch: Branch, fstate: ConcolicState<'ctx, O, VS>) -> Self {
        match self {
            Self::Symbolic(mut states) => {
                states.push((fbranch, fstate));
                Self::Symbolic(states)
            },
            Self::Concrete(tbranch, _) => {
                Self::Symbolic(vec![(tbranch, tstate), (fbranch, fstate)])
            }
        }
    }

    fn unwrap_concrete_address(self) -> AddressValue {
        if let Self::Concrete(Branch::Global(address), _) = self {
            address
        } else {
            panic!("expected NextLocation::Concrete(Branch::Global(_))")
        }
    }

    fn unwrap_symbolic(self) -> Vec<(Branch, ConcolicState<'ctx, O, VS>)> {
        if let Self::Symbolic(states) = self {
            states
        } else {
            panic!("expected NextLocation::Symbolic")
        }
    }
}

pub type Outcomes<'ctx, O, VS> = Vec<(Branch, ConcolicState<'ctx, O, VS>)>;

#[derive(Clone)]
pub struct ConcolicContext<'ctx, O: Order, VS: ValueSolver<'ctx>, P: SymbolicPointerStrategy<'ctx, O, VS>, const OPERAND_SIZE: usize> {
    hooks: Vec<
        Box<dyn ClonableHookConcolic<State = ConcolicState<'ctx, O, VS>, Error = StateError, Outcome = Outcomes<'ctx, O, VS>> + 'ctx>,
    >,
    program_counter: Arc<Operand>,
    state: ConcolicState<'ctx, O, VS>,
    pointer_strategy: P,
    translator: Arc<Translator>,
    translator_context: ContextDatabase,
    translator_cache: Arc<RwLock<HashMap<Address, StepState>>>,
    intrinsics: IntrinsicHandler<Outcomes<'ctx, O, VS>, ConcolicState<'ctx, O, VS>>,
}

trait ToSignedBytes {
    fn expand_as<'ctx, O: Order, VS: ValueSolver<'ctx>, P: SymbolicPointerStrategy<'ctx, O, VS>, const OPERAND_SIZE: usize>(
        self,
        ctxt: &mut ConcolicContext<'ctx, O, VS, P, { OPERAND_SIZE }>,
        dest: &Operand,
        signed: bool,
    ) -> Result<(), Error>;
}

impl ToSignedBytes for bool {
    fn expand_as<'ctx, O: Order, VS: ValueSolver<'ctx>, P: SymbolicPointerStrategy<'ctx, O, VS>, const OPERAND_SIZE: usize>(
        self,
        ctxt: &mut ConcolicContext<'ctx, O, VS, P, { OPERAND_SIZE }>,
        dest: &Operand,
        _signed: bool,
    ) -> Result<(), Error> {
        ctxt.state
            .concretise_operand_with(dest, self)
            .map_err(Error::State)?;

        let bits = dest.size() * 8;
        let bv = Either::Left(BitVec::from_u32(if self { 1 } else { 0 }, bits));

        for hook in ctxt.hooks.iter_mut() {
            hook.hook_operand_write(&mut ctxt.state, dest, &bv)
                .map_err(Error::Hook)?;
        }

        Ok(())
    }
}

impl ToSignedBytes for BitVec {
    fn expand_as<'ctx, O: Order, VS: ValueSolver<'ctx>, P: SymbolicPointerStrategy<'ctx, O, VS>, const OPERAND_SIZE: usize>(
        self,
        ctxt: &mut ConcolicContext<'ctx, O, VS, P, { OPERAND_SIZE }>,
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

        let bv = Either::Left(target.clone());

        ctxt.state
            .concretise_operand_with(dest, target)
            .map_err(Error::State)?;

        for hook in ctxt.hooks.iter_mut() {
            hook.hook_operand_write(&mut ctxt.state, dest, &bv)
                .map_err(Error::Hook)?;
        }

        Ok(())
    }
}

impl ToSignedBytes for SymExpr {
    fn expand_as<'ctx, O: Order, VS: ValueSolver<'ctx>, P: SymbolicPointerStrategy<'ctx, O, VS>, const OPERAND_SIZE: usize>(
        self,
        ctxt: &mut ConcolicContext<'ctx, O, VS, P, { OPERAND_SIZE }>,
        dest: &Operand,
        signed: bool,
    ) -> Result<(), Error> {
        let size = dest.size();

        if size > OPERAND_SIZE {
            return Err(Error::UnsupportedOperandSize(size, OPERAND_SIZE));
        }

        let dbits = (size as u32) << 3;
        let target = if signed {
            SymExpr::sign_extend(self, dbits)
        } else if self.bits() != dbits {
            SymExpr::zero_extend(self, dbits)
        } else {
            self
        };

        let expr = Either::<BitVec, _>::Right(target);

        ctxt.state.write_operand_value(dest, expr.clone())?;

        for hook in ctxt.hooks.iter_mut() {
            hook.hook_operand_write(&mut ctxt.state, dest, &expr)
                .map_err(Error::Hook)?;
        }

        Ok(())
    }
}

impl<L, R> ToSignedBytes for Either<L, R>
where
    L: ToSignedBytes,
    R: ToSignedBytes,
{
    fn expand_as<'ctx, O: Order, VS: ValueSolver<'ctx>, P: SymbolicPointerStrategy<'ctx, O, VS>, const OPERAND_SIZE: usize>(
        self,
        ctxt: &mut ConcolicContext<'ctx, O, VS, P, { OPERAND_SIZE }>,
        dest: &Operand,
        signed: bool,
    ) -> Result<(), Error> {
        match self {
            Either::Left(l) => l.expand_as(ctxt, dest, signed),
            Either::Right(r) => r.expand_as(ctxt, dest, signed),
        }
    }
}

impl<'ctx, O: Order, VS: ValueSolver<'ctx>, P: SymbolicPointerStrategy<'ctx, O, VS>, const OPERAND_SIZE: usize> ConcolicContext<'ctx, O, VS, P, { OPERAND_SIZE }> {
    pub fn new(solver: VS, translator: Arc<Translator>, state: PCodeState<u8, O>, pointer_strategy: P) -> Self {
        Self {
            hooks: Vec::new(),
            intrinsics: IntrinsicHandler::new(),
            program_counter: state.registers().program_counter(),
            state: ConcolicState::new(solver, translator.clone(), state),
            pointer_strategy,
            translator_cache: Arc::new(RwLock::new(HashMap::default())),
            translator_context: translator.context_database(),
            translator,
        }
    }

    pub fn add_hook<H>(&mut self, hook: H)
    where
        H: ClonableHookConcolic<State = ConcolicState<'ctx, O, VS>, Error = StateError, Outcome = Outcomes<'ctx, O, VS>> + 'ctx,
    {
        self.hooks.push(Box::new(hook));
    }

    pub fn push_constraint(&mut self, constraint: SymExpr) {
        self.state.push_constraint(constraint);
    }

    pub fn state(&self) -> &ConcolicState<'ctx, O, VS> {
        &self.state
    }

    pub fn state_mut(&mut self) -> &mut ConcolicState<'ctx, O, VS> {
        &mut self.state
    }

    pub fn branch_expr(state: &mut ConcolicState<'ctx, O, VS>, expr: SymExpr) -> Result<NextLocation<'ctx, O, VS>, Error> {
        let mut states = Vec::new();
        let value = expr.simplify();

        let solns = value.solve_many(&mut state.solver, &state.constraints);

        for val in solns {
            let address = (&val).into_address_value(state.concrete.memory_space_ref());
            let location = Branch::Global(address);

            let mut fork = state.fork();
            fork.push_constraint(SymExpr::eq(val.into(), value.clone()));

            if fork.is_sat() {
                states.push((location, fork));
            }
        }

        if states.is_empty() {
            Err(Error::UnsatisfiablePC)
        } else {
            Ok(NextLocation::Symbolic(states))
        }
    }

    pub fn ibranch_on(state: &mut ConcolicState<'ctx, O, VS>, pc: &Operand) -> Result<NextLocation<'ctx, O, VS>, Error> {
        match state.read_operand_value(pc)? {
            Either::Left(bv) => {
                let memory_space = state.concrete.memory_space_ref();
                let location = Branch::Global(bv.into_address_value(memory_space));
                Ok(NextLocation::Concrete(location, PhantomData))
            }
            Either::Right(expr) => Self::branch_expr(state, expr),
        }
    }

    pub fn branch_on(state: &mut ConcolicState<'ctx, O, VS>, pc: &Operand) -> Result<NextLocation<'ctx, O, VS>, Error> {
        if let Operand::Address { value, .. } = pc {
            let memory_space = state.concrete.memory_space_ref();
            let address = value.into_address_value(memory_space);
            Ok(NextLocation::Concrete(Branch::Global(address), PhantomData))
        } else {
            match state.read_operand_value(pc)? {
                Either::Left(bv) => {
                    let memory_space = state.concrete.memory_space_ref();
                    let location = Branch::Global(bv.into_address_value(memory_space));
                    Ok(NextLocation::Concrete(location, PhantomData))
                }
                Either::Right(expr) => Self::branch_expr(state, expr),
            }
        }
    }

    fn get_address(&mut self, source: &Operand) -> Result<Either<AddressValue, SymExpr>, Error> {
        match self.state.read_operand_value(source)? {
            Either::Left(bv) => {
                let offset = bv
                    .to_u64()
                    .ok_or_else(|| Error::UnsupportedAddressSize(bv.bits() * 8))?;
                Ok(Either::Left(AddressValue::new(self.interpreter_space(), offset)))
            }
            Either::Right(expr) => {
                Ok(Either::Right(expr))
            }
        }
    }

    fn with_return_location<U, F>(&mut self, f: F) -> Result<U, Error>
    where
        F: FnOnce(&mut ConcolicState<'ctx, O, VS>, Either<Operand, SymExpr>) -> Result<U, Error>,
    {
        let rloc = (*self.state.concrete.registers().return_location()).clone();
        match rloc {
            ReturnLocation::Register(operand) => {
                f(&mut self.state, Either::Left(operand))
            },
            ReturnLocation::Relative(operand, offset) => {
                match self.get_address(&operand)? {
                    Either::Left(address) => {
                        let operand = Operand::Address {
                            value: address.into(),
                            size: self.interpreter_space().address_size(),
                        };
                        f(&mut self.state, Either::Left(operand))
                    },
                    Either::Right(expr) => {
                        let offset = BitVec::from_u64(offset, self.interpreter_space().address_size());
                        let aexpr = SymExpr::add(expr, offset.into());
                        f(&mut self.state, Either::Right(aexpr))
                    }
                }
            }
        }
    }

    fn skip_return(&mut self) -> Result<NextLocation<'ctx, O, VS>, Error> {
        // NOTE: for x86, etc. we need to clean-up the stack
        // arguments; currently, this is the responsibility of
        // hooks that issue a `HookCallAction::Skip`.
        let address = self.with_return_location(|state, operand| match operand {
            Either::Left(address) => Self::ibranch_on(state, &address),
            Either::Right(expr) => Self::branch_expr(state, expr),
        })?;

        // Next we pop the return address (if needed)
        let extra_pop = self.state.concrete.convention().default_prototype().extra_pop();

        if extra_pop > 0 {
            let stack_pointer = self.state.concrete.registers().stack_pointer().clone();
            let bits = stack_pointer.size() * 8;

            match self.state.read_operand_value(&stack_pointer)? {
                Either::Left(bv) => {
                    let address = bv + BitVec::from_u64(extra_pop, bits);
                    self.state.concretise_operand_with(&stack_pointer, address)?;
                },
                Either::Right(expr) => {
                    let address = SymExpr::add(expr, BitVec::from_u64(extra_pop, bits).into());
                    self.state.write_operand_value(&stack_pointer, Either::Right(address))?;
                },
            }
        }

        Ok(address)
    }

    fn lift_bool1<COC, COS, COCO, COSO>(
        &mut self,
        opc: COC,
        ops: COS,
        dest: &Operand,
        rhs: &Operand,
    ) -> Result<Outcome<Outcomes<'ctx, O, VS>>, Error>
    where
        COC: FnOnce(bool) -> Result<COCO, Error>,
        COS: FnOnce(SymExpr) -> Result<COSO, Error>,
        COCO: ToSignedBytes,
        COSO: ToSignedBytes,
    {
        let rsize = rhs.size();
        if rsize > OPERAND_SIZE {
            return Err(Error::UnsupportedOperandSize(rsize, OPERAND_SIZE));
        }

        for hook in self.hooks.iter_mut() {
            hook.hook_operand_read(&mut self.state, rhs)
                .map_err(Error::Hook)?;
        }

        let value = match self.state.read_operand_value(rhs)? {
            Either::Left(bv) => Either::Left(opc(!bv.is_zero())?),
            Either::Right(expr) => Either::Right(ops(SymExpr::cast_bool(expr))?),
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
    ) -> Result<Outcome<Outcomes<'ctx, O, VS>>, Error>
    where
        COC: FnOnce(bool, bool) -> Result<COCO, Error>,
        COS: FnOnce(SymExpr, SymExpr) -> Result<COSO, Error>,
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

        for hook in self.hooks.iter_mut() {
            hook.hook_operand_read(&mut self.state, lhs)
                .map_err(Error::Hook)?;
        }

        let lhs_val = self
            .state
            .read_operand_value(lhs)?
            .map_right(|expr| SymExpr::cast_bool(expr));

        for hook in self.hooks.iter_mut() {
            hook.hook_operand_read(&mut self.state, rhs)
                .map_err(Error::Hook)?;
        }

        let rhs_val = self
            .state
            .read_operand_value(rhs)?
            .map_right(|expr| SymExpr::cast_bool(expr));

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
    ) -> Result<Outcome<Outcomes<'ctx, O, VS>>, Error>
    where
        COC: FnOnce(BitVec) -> Result<COCO, Error>,
        COS: FnOnce(SymExpr) -> Result<COSO, Error>,
        COCO: ToSignedBytes,
        COSO: ToSignedBytes,
    {
        let rsize = rhs.size();
        if rsize > OPERAND_SIZE {
            return Err(Error::UnsupportedOperandSize(rsize, OPERAND_SIZE));
        }

        for hook in self.hooks.iter_mut() {
            hook.hook_operand_read(&mut self.state, rhs)
                .map_err(Error::Hook)?;
        }

        let value = match self.state.read_operand_value(rhs)? {
            Either::Left(bv) => Either::Left(opc(if signed { bv.signed() } else { bv })?),
            Either::Right(expr) => {
                let bits = expr.bits();
                Either::Right(ops(if signed {
                    SymExpr::sign_extend(expr, bits)
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
    ) -> Result<Outcome<Outcomes<'ctx, O, VS>>, Error>
    where
        COC: FnOnce(BitVec, BitVec) -> Result<COCO, Error>,
        COS: FnOnce(SymExpr, SymExpr) -> Result<COSO, Error>,
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

        for hook in self.hooks.iter_mut() {
            hook.hook_operand_read(&mut self.state, lhs)
                .map_err(Error::Hook)?;
        }

        let lhs_val = self
            .state
            .read_operand_value(lhs)?
            .map_left(|bv| bv.signed())
            .map_right(|expr| {
                let bits = expr.bits();
                SymExpr::sign_extend(expr, bits)
            });

        for hook in self.hooks.iter_mut() {
            hook.hook_operand_read(&mut self.state, rhs)
                .map_err(Error::Hook)?;
        }

        let rhs_val = self
            .state
            .read_operand_value(rhs)?
            .map_left(|bv| {
                if signed {
                    bv.signed().cast(lbits)
                } else {
                    bv.unsigned().cast(lbits)
                }
            })
            .map_right(|expr| {
                if signed {
                    SymExpr::sign_extend(expr, lbits as u32)
                } else {
                    SymExpr::zero_extend(expr, lbits as u32)
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
    ) -> Result<Outcome<Outcomes<'ctx, O, VS>>, Error>
    where
        COC: FnOnce(Float, &FloatFormat) -> Result<COCO, Error>,
        COS: FnOnce(SymExpr, &HashMap<usize, Arc<FloatFormat>>) -> Result<COSO, Error>,
        COCO: ToSignedBytes,
        COSO: ToSignedBytes,
    {
        let rsize = rhs.size();
        if rsize > OPERAND_SIZE {
            return Err(Error::UnsupportedOperandSize(rsize, OPERAND_SIZE));
        }

        for hook in self.hooks.iter_mut() {
            hook.hook_operand_read(&mut self.state, rhs)
                .map_err(Error::Hook)?;
        }

        let format = float_format_from_size(rsize)?;

        let value = match self.state.read_operand_value(rhs)? {
            Either::Left(bv) => Either::Left(opc(format.from_bitvec(&bv), &format)?),
            Either::Right(expr) => {
                let formats = self.translator.float_formats();
                Either::Right(ops(SymExpr::cast_float(expr, Arc::new(format)), formats)?)
                // let formats = self.state.solver.translator().float_formats().clone();
                // Either::Right(ops(SymExpr::cast_float(expr, Arc::new(format)), &formats)?)
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
    ) -> Result<Outcome<Outcomes<'ctx, O, VS>>, Error>
    where
        COC: FnOnce(Float, Float, &FloatFormat) -> Result<COCO, Error>,
        COS: FnOnce(SymExpr, SymExpr, &HashMap<usize, Arc<FloatFormat>>) -> Result<COSO, Error>,
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

        for hook in self.hooks.iter_mut() {
            hook.hook_operand_read(&mut self.state, lhs)
                .map_err(Error::Hook)?;
        }

        let format = Arc::new(float_format_from_size(rsize)?);

        let lhs_val = self
            .state
            .read_operand_value(lhs)?
            .map_right(|expr| SymExpr::cast_float(expr, format.clone()));

        for hook in self.hooks.iter_mut() {
            hook.hook_operand_read(&mut self.state, rhs)
                .map_err(Error::Hook)?;
        }

        let rhs_val = self
            .state
            .read_operand_value(rhs)?
            .map_right(|expr| SymExpr::cast_float(expr, format.clone()));

        let formats = self.translator.float_formats();
        // let formats = self.state.solver.translator().float_formats().clone();
        let value = match (lhs_val, rhs_val) {
            (Either::Left(bv1), Either::Left(bv2)) => Either::Left(opc(
                format.from_bitvec(&bv1),
                format.from_bitvec(&bv2),
                &format,
            )?),
            (Either::Left(bv), Either::Right(se)) => {
                Either::Right(ops(SymExpr::cast_float(bv.into(), format.clone()), se, &formats)?)
            }
            (Either::Right(se), Either::Left(bv)) => {
                Either::Right(ops(se, SymExpr::cast_float(bv.into(), format.clone()), &formats)?)
            }
            (Either::Right(se1), Either::Right(se2)) => Either::Right(ops(se1, se2, &formats)?),
        };

        value.expand_as(self, dest, true)?;

        Ok(Outcome::Branch(Branch::Next))
    }

    pub fn restore_state(&mut self, state: ConcolicState<'ctx, O, VS>) {
        self.state = state;
    }
}

impl<'ctx, O: Order, VS: ValueSolver<'ctx>, P: SymbolicPointerStrategy<'ctx, O, VS>, const OPERAND_SIZE: usize> Interpreter for ConcolicContext<'ctx, O, VS, P, { OPERAND_SIZE }> {
    type State = ConcolicState<'ctx, O, VS>;
    type Error = Error;
    type Outcome = Vec<(Branch, Self::State)>;

    fn fork(&self) -> Self {
        Self {
            hooks: self.hooks.clone(),
            program_counter: self.program_counter.clone(),
            state: self.state.fork(),
            pointer_strategy: self.pointer_strategy.clone(),
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
                let memory_space = self.state.concrete.memory_space_ref();
                let address = value.into_address_value(memory_space);
                let action = Branch::Global(address);
                Ok(Outcome::Branch(action))
            }
            Operand::Register { .. } | Operand::Variable { .. } => {
                return Err(Error::UnsupportedBranchDestination)
            }
        }
    }

    fn ibranch(&mut self, destination: &Operand) -> Result<Outcome<Self::Outcome>, Self::Error> {
        if destination == &*self.state.concrete.registers().program_counter() {
            return self.icall(destination);
        }

        match Self::ibranch_on(&mut self.state, destination)? {
            NextLocation::Concrete(location, _) => Ok(Outcome::Branch(location)),
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

                tstate.push_constraint(SymExpr::bool_eq(expr.clone(), BitVec::one(8).into()));
                let tsat = tstate.is_sat();

                fstate.push_constraint(SymExpr::bool_eq(expr, BitVec::zero(8).into()));
                let fsat = fstate.is_sat();

                let states = match (tsat, fsat) {
                    (true, true) => {
                        let states = Self::branch_on(&mut tstate, destination);
                        if let Err(Error::UnsatisfiablePC) = states {
                            NextLocation::Symbolic(vec![(Branch::Next, fstate)])
                        } else {
                            states?.join(tstate, Branch::Next, fstate)
                        }
                    },
                    (true, false) => {
                        Self::branch_on(&mut tstate, destination)?
                    },
                    (false, true) => {
                        NextLocation::Symbolic(vec![(Branch::Next, fstate)])
                    },
                    (false, false) => {
                        return Err(Error::UnsatisfiablePC)
                    },
                };

                Ok(Outcome::Halt(states.unwrap_symbolic()))
            }
        }
    }

    fn call(&mut self, destination: &Operand) -> Result<Outcome<Self::Outcome>, Self::Error> {
        match destination {
            Operand::Address { value: address, .. } => {
                let mut skip = false;

                for hook in self.hooks.iter_mut() {
                    match hook
                        .hook_call(&mut self.state, address)
                        .map_err(Error::Hook)?
                        .action
                    {
                        HookCallAction::Pass => (),
                        HookCallAction::Skip => {
                            skip = true;
                        }
                        HookCallAction::Halt(r) => return Ok(Outcome::Halt(r)),
                    }
                }

                if skip {
                    match self.skip_return()? {
                        NextLocation::Concrete(location, _) => Ok(Outcome::Branch(location)),
                        NextLocation::Symbolic(states) => Ok(Outcome::Halt(states)),
                    }
                } else {
                    let memory_space = self.state.concrete.memory_space_ref();
                    let address = address.into_address_value(memory_space);
                    Ok(Outcome::Branch(Branch::Global(address)))
                }
            },
            Operand::Constant { .. }
            | Operand::Register { .. }
            | Operand::Variable { .. } => {
                Err(Error::UnsupportedBranchDestination)
            }
        }
    }

    fn icall(&mut self, destination: &Operand) -> Result<Outcome<Self::Outcome>, Self::Error> {
        match Self::ibranch_on(&mut self.state, destination)? {
            location@NextLocation::Concrete(_, _) => {
                let mut skip = false;
                let address_value = location.unwrap_concrete_address();
                let address = (&address_value).into();

                for hook in self.hooks.iter_mut() {
                    match hook
                        .hook_call(&mut self.state, &address)
                        .map_err(Error::Hook)?
                        .action
                    {
                        HookCallAction::Pass => (),
                        HookCallAction::Skip => {
                            skip = true;
                        }
                        HookCallAction::Halt(r) => return Ok(Outcome::Halt(r)),
                    }
                }

                if skip {
                    match self.skip_return()? {
                        NextLocation::Concrete(location, _) => Ok(Outcome::Branch(location)),
                        NextLocation::Symbolic(states) => Ok(Outcome::Halt(states)),
                    }
                } else {
                    Ok(Outcome::Branch(Branch::Global(address_value)))
                }
            },
            NextLocation::Symbolic(states) => {
                // TODO: we need to trigger a call hook
                Ok(Outcome::Halt(states))
            },
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

        for hook in self.hooks.iter_mut() {
            hook.hook_operand_read(&mut self.state, source)
                .map_err(Error::Hook)?;
        }

        let sval = self.state.read_operand_value(source)?;

        self.state.write_operand_value(destination, sval.clone())?;

        for hook in self.hooks.iter_mut() {
            hook.hook_operand_write(&mut self.state, destination, &sval)
                .map_err(Error::Hook)?;
        }

        Ok(Outcome::Branch(Branch::Next))
    }

    fn load(
        &mut self,
        source: &Operand,
        destination: &Operand,
        space: AddressSpaceId,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        let space = self.translator.manager().space_by_id(space);
        let space_size = space.address_size();
        let space_word_size = space.word_size() as u64;

        assert_eq!(space_size, source.size());

        for hook in self.hooks.iter_mut() {
            hook.hook_operand_read(&mut self.state, source)
                .map_err(Error::Hook)?;
        }

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
                    value: Address::new(space, addr_val),
                    size: destination.size(),
                };

                self.copy(&address, destination)
            }
            Either::Right(expr) => {
                let bits = expr.bits();
                let addr = {
                    let off = SymExpr::mul(expr, BitVec::from_u64(space_word_size, bits as usize).into());
                    let msk = BitVec::from_u64(
                        1u64.checked_shl(space_size.checked_shl(3).unwrap_or(0) as u32)
                            .unwrap_or(0)
                            .wrapping_sub(1),
                        bits as usize,
                    );
                    SymExpr::and(off, msk.into())
                };

                let size = destination.size();

                for hook in self.hooks.iter_mut() {
                    hook.hook_symbolic_memory_read(&mut self.state, &addr, size)
                        .map_err(Error::Hook)?;
                }

                let value = self
                    .pointer_strategy
                    .read_symbolic_memory(&mut self.state, &addr, size * 8)
                    .map_err(|e| e.into())?;

                self.state.write_operand_expr(destination, value.clone());

                let bv = Either::Right(value);

                for hook in self.hooks.iter_mut() {
                    hook.hook_operand_write(&mut self.state, destination, &bv)
                        .map_err(Error::Hook)?;
                }

                Ok(Outcome::Branch(Branch::Next))
            }
        }
    }

    fn store(
        &mut self,
        source: &Operand,
        destination: &Operand,
        space: AddressSpaceId,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        let space = self.translator.manager().space_by_id(space);
        let space_size = space.address_size();
        let space_word_size = space.word_size() as u64;

        assert_eq!(space_size, destination.size());

        for hook in self.hooks.iter_mut() {
            hook.hook_operand_read(&mut self.state, destination)
                .map_err(Error::Hook)?;
        }

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
                    value: Address::new(space, addr_val),
                    size: source.size(),
                };

                self.copy(source, &address)
            }
            Either::Right(expr) => {
                let bits = expr.bits();
                let addr = {
                    let off = SymExpr::mul(expr, BitVec::from_u64(space_word_size, bits as usize).into());
                    let msk = BitVec::from_u64(
                        1u64.checked_shl(space_size.checked_shl(3).unwrap_or(0) as u32)
                            .unwrap_or(0)
                            .wrapping_sub(1),
                        bits as usize,
                    );
                    SymExpr::and(off, msk.into())
                };

                let value = self
                    .state
                    .read_operand_value(source)?;

                self.pointer_strategy
                    .write_symbolic_memory(&mut self.state, &addr, value.clone())
                    .map_err(|e| e.into())?;

                for hook in self.hooks.iter_mut() {
                    hook.hook_symbolic_memory_write(&mut self.state, &addr, &value)
                        .map_err(Error::Hook)?;
                }

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
            |u, v| Ok(SymExpr::eq(u, v)),
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
            |u, v| Ok(SymExpr::ne(u, v)),
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
            |u, v| Ok(SymExpr::lt(u, v)),
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
            |u, v| Ok(SymExpr::le(u, v)),
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
            |u, v| Ok(SymExpr::slt(u, v)),
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
            |u, v| Ok(SymExpr::sle(u, v)),
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
            |u, v| Ok(SymExpr::add(u, v)),
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
            |u, v| Ok(SymExpr::sub(u, v)),
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
            |u, v| Ok(SymExpr::carry(u, v)),
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
            |u, v| Ok(SymExpr::signed_carry(u, v)),
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
            |u, v| Ok(SymExpr::signed_borrow(u, v)),
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
            |u| Ok(SymExpr::neg(u)),
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
            |u| Ok(SymExpr::not(u)),
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
            |u, v| Ok(SymExpr::xor(u, v)),
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
            |u, v| Ok(SymExpr::and(u, v)),
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
            |u, v| Ok(SymExpr::or(u, v)),
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
            |u, v| Ok(SymExpr::shl(u, v)),
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
            |u, v| Ok(SymExpr::shr(u, v)),
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
            |u, v| Ok(SymExpr::signed_shr(u, v)),
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
            |u, v| Ok(SymExpr::mul(u, v)),
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
            |u, v| Ok(SymExpr::div(u, v)),
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
            |u, v| Ok(SymExpr::signed_div(u, v)),
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
            |u, v| Ok(SymExpr::rem(u, v)),
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
            |u, v| Ok(SymExpr::signed_rem(u, v)),
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
        self.lift_bool1(|u| Ok(!u), |u| Ok(SymExpr::bool_not(u)), destination, operand)
    }

    fn bool_xor(
        &mut self,
        destination: &Operand,
        operand1: &Operand,
        operand2: &Operand,
    ) -> Result<Outcome<Self::Outcome>, Self::Error> {
        self.lift_bool2(
            |u, v| Ok(u ^ v),
            |u, v| Ok(SymExpr::bool_xor(u, v)),
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
            |u, v| Ok(SymExpr::bool_and(u, v)),
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
            |u, v| Ok(SymExpr::bool_or(u, v)),
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
            |u, v, fmts| Ok(SymExpr::float_eq(u, v, fmts)),
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
            |u, v, fmts| Ok(SymExpr::float_ne(u, v, fmts)),
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
            |u, v, fmts| Ok(SymExpr::float_le(u, v, fmts)),
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
            |u, v, fmts| Ok(SymExpr::float_le(u, v, fmts)),
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
            |u, fmts| Ok(SymExpr::float_nan(u, fmts)),
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
            |u, v, fmts| Ok(SymExpr::float_add(u, v, fmts)),
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
            |u, v, fmts| Ok(SymExpr::float_div(u, v, fmts)),
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
            |u, v, fmts| Ok(SymExpr::float_mul(u, v, fmts)),
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
            |u, v, fmts| Ok(SymExpr::float_sub(u, v, fmts)),
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
            |u, fmts| Ok(SymExpr::float_neg(u, fmts)),
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
            |u, fmts| Ok(SymExpr::float_abs(u, fmts)),
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
            |u, fmts| Ok(SymExpr::float_sqrt(u, fmts)),
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
            |u| Ok(SymExpr::cast_float(u, fmt.clone())),
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
            |u, _fmts| Ok(SymExpr::cast_float(u, Arc::new(fmt))),
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
            |u, _fmts| Ok(SymExpr::sign_extend(u, destination.size() as u32 * 8)),
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
            |u, fmts| Ok(SymExpr::float_ceiling(u, fmts)),
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
            |u, fmts| Ok(SymExpr::float_floor(u, fmts)),
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
            |u, fmts| Ok(SymExpr::float_round(u, fmts)),
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
            |u| Ok(SymExpr::count_ones(u)),
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

        for hook in self.hooks.iter_mut() {
            hook.hook_operand_read(&mut self.state, operand)
                .map_err(Error::Hook)?;
        }

        for hook in self.hooks.iter_mut() {
            hook.hook_operand_read(&mut self.state, amount)
                .map_err(Error::Hook)?;
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
                let out_size = destination.size() as u32 * 8;

                let loff = amount.offset() as u32 * 8;
                let trun_size = src_size.checked_sub(loff).unwrap_or(0);

                let trun = if out_size > trun_size {
                    // extract high + expand
                    let source_htrun = SymExpr::extract_high(expr, trun_size);
                    SymExpr::zero_extend(source_htrun, out_size)
                } else {
                    // extract
                    let hoff = loff + out_size;
                    SymExpr::extract(expr, loff, hoff)
                };

                Either::Right(trun)
            },
        };

        self.state.write_operand_value(destination, value.clone())?;

        for hook in self.hooks.iter_mut() {
            hook.hook_operand_write(&mut self.state, destination, &value)
                .map_err(Error::Hook)?;
        }

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
        let memory_space = self.state.concrete.memory_space_ref();
        let address_value = address.into_address_value(memory_space);
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

            let pstep_state = //StepState::from(
                self.translator
                    .lift_pcode(&mut self.translator_context, address_value, &view[..8])
                    .map_err(|e| Error::Lift(address, e))?
            //);
            ;

            let step_state = StepState::from(pstep_state);

            let diff = (step_state.fallthrough().offset() - step_state.address().offset()) as usize;
            if self.state.is_symbolic_memory(address, diff) {
                return Err(Error::SymbolicInstructions(address, diff))
            }

            self.translator_cache
                .write()
                .insert(address, step_state.clone());

            step_state
        };

        /*
        self.state
            .concrete
            .set_address(&self.program_counter, address)
            .map_err(StateError::State)?;

        self.state.concretise_operand(&self.program_counter);
        */

        let address = BitVec::from_u64(
            u64::from(address),
            self.program_counter.size() * 8,
        );

        self.state.concretise_operand_with(&self.program_counter, address)?;

        Ok(step_state)
    }

    fn interpreter_space(&self) -> Arc<AddressSpace> {
        self.state.concrete.memory_space()
    }
}
