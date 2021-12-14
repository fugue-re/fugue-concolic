use either::Either;

use fugue::bv::BitVec;
use fugue::bytes::Order;

use std::marker::PhantomData;

use crate::backend::ValueSolver;
use crate::expr::SymExpr;
use crate::state::{ConcolicState, Error as StateError};

pub trait SymbolicPointerStrategy<'ctx, O: Order, VS: ValueSolver<'ctx>>: Clone {
    type Error: std::error::Error + Into<StateError>;

    fn read_symbolic_memory(
        &mut self,
        state: &mut ConcolicState<'ctx, O, VS>,
        address: &SymExpr,
        bits: usize,
    ) -> Result<SymExpr, Self::Error>;

    fn write_symbolic_memory(
        &mut self,
        state: &mut ConcolicState<'ctx, O, VS>,
        address: &SymExpr,
        value: Either<BitVec, SymExpr>,
    ) -> Result<(), Self::Error>;
}

#[derive(Clone)]
pub struct DefaultPointerStrategy<'ctx, O: Order, VS: ValueSolver<'ctx>>(PhantomData<&'ctx (O, VS)>);

impl<'ctx, O: Order, VS: ValueSolver<'ctx>> Default for DefaultPointerStrategy<'ctx, O, VS> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<'ctx, O: Order, VS: ValueSolver<'ctx>> SymbolicPointerStrategy<'ctx, O, VS> for DefaultPointerStrategy<'ctx, O, VS> {
    type Error = StateError;

    fn read_symbolic_memory(&mut self, state: &mut ConcolicState<'ctx, O, VS>, address: &SymExpr, bits: usize) -> Result<SymExpr, Self::Error> {
        state.read_memory_symbolic(address, bits)
    }

    fn write_symbolic_memory(&mut self, state: &mut ConcolicState<'ctx, O, VS>, address: &SymExpr, value: Either<BitVec, SymExpr>) -> Result<(), Self::Error> {
        state.write_memory_symbolic(
            address,
            value.either(|bv| SymExpr::from(bv), |expr| expr),
        )
    }
}
