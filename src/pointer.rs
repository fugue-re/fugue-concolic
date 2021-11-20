use either::Either;

use fugue::bv::BitVec;
use fugue::bytes::Order;

use std::marker::PhantomData;

use crate::expr::SymExpr;
use crate::state::{ConcolicState, Error as StateError};

pub trait SymbolicPointerStrategy<O: Order>: Clone {
    type Error: std::error::Error + Into<StateError>;

    fn read_symbolic_memory(
        &mut self,
        state: &mut ConcolicState<O>,
        address: &SymExpr,
        bits: usize,
    ) -> Result<SymExpr, Self::Error>;

    fn write_symbolic_memory(
        &mut self,
        state: &mut ConcolicState<O>,
        address: &SymExpr,
        value: Either<BitVec, SymExpr>,
    ) -> Result<(), Self::Error>;
}

#[derive(Clone)]
pub struct DefaultPointerStrategy<O: Order>(PhantomData<O>);

impl<O: Order> Default for DefaultPointerStrategy<O> {
    fn default() -> Self {
        Self(PhantomData)
    }
}

impl<O: Order> SymbolicPointerStrategy<O> for DefaultPointerStrategy<O> {
    type Error = StateError;

    fn read_symbolic_memory(&mut self, state: &mut ConcolicState<O>, address: &SymExpr, bits: usize) -> Result<SymExpr, Self::Error> {
        state.read_memory_symbolic(address, bits)
    }

    fn write_symbolic_memory(&mut self, state: &mut ConcolicState<O>, address: &SymExpr, value: Either<BitVec, SymExpr>) -> Result<(), Self::Error> {
        state.write_memory_symbolic(
            address,
            value.either(|bv| SymExpr::from(bv), |expr| expr),
        )
    }
}
