use either::Either;

use fugue::bv::BitVec;
use fugue::ir::Address;
use fugue::ir::il::pcode::{Operand, Register};

use fuguex::state::State;

use dyn_clone::{DynClone, clone_trait_object};

use fuguex::hooks::types::{Error, HookAction, HookCallAction, HookOutcome};

use crate::expr::SymExpr;

#[allow(unused)]
pub trait HookConcolic {
    type State: State;
    type Error: std::error::Error + Send + Sync + 'static;
    type Outcome;

    fn hook_memory_read(
        &mut self,
        state: &mut Self::State,
        address: &Address,
        bytes: usize,
    ) -> Result<HookOutcome<HookAction<Self::Outcome>>, Error<Self::Error>> {
        Ok(HookAction::Pass.into())
    }

    fn hook_symbolic_memory_read(
        &mut self,
        state: &mut Self::State,
        address: &SymExpr,
        bytes: usize,
    ) -> Result<HookOutcome<HookAction<Self::Outcome>>, Error<Self::Error>> {
        Ok(HookAction::Pass.into())
    }

    fn hook_memory_write(
        &mut self,
        state: &mut Self::State,
        address: &Address,
        value: &Either<BitVec, SymExpr>,
    ) -> Result<HookOutcome<HookAction<Self::Outcome>>, Error<Self::Error>> {
        Ok(HookAction::Pass.into())
    }

    fn hook_symbolic_memory_write(
        &mut self,
        state: &mut Self::State,
        address: &SymExpr,
        value: &Either<BitVec, SymExpr>,
    ) -> Result<HookOutcome<HookAction<Self::Outcome>>, Error<Self::Error>> {
        Ok(HookAction::Pass.into())
    }

    fn hook_register_read(
        &mut self,
        state: &mut Self::State,
        register: &Register,
    ) -> Result<HookOutcome<HookAction<Self::Outcome>>, Error<Self::Error>> {
        Ok(HookAction::Pass.into())
    }

    fn hook_register_write(
        &mut self,
        state: &mut Self::State,
        register: &Register,
        value: &Either<BitVec, SymExpr>,
    ) -> Result<HookOutcome<HookAction<Self::Outcome>>, Error<Self::Error>> {
        Ok(HookAction::Pass.into())
    }

    fn hook_operand_read(
        &mut self,
        state: &mut Self::State,
        operand: &Operand,
    ) -> Result<HookOutcome<HookAction<Self::Outcome>>, Error<Self::Error>> {
        match operand {
            Operand::Address { value: address, size } => {
                self.hook_memory_read(state, address, *size)
            },
            Operand::Register { .. } => {
                self.hook_register_read(state, &operand.register().unwrap())
            },
            _ => Ok(HookAction::Pass.into())
        }
    }

    fn hook_operand_write(
        &mut self,
        state: &mut Self::State,
        operand: &Operand,
        value: &Either<BitVec, SymExpr>,
    ) -> Result<HookOutcome<HookAction<Self::Outcome>>, Error<Self::Error>> {
        match operand {
            Operand::Address { value: address, .. } => {
                self.hook_memory_write(state, address, value)
            },
            Operand::Register { .. } => {
                self.hook_register_write(state, &operand.register().unwrap(), value)
            },
            _ => Ok(HookAction::Pass.into())
        }
    }

    fn hook_call(
        &mut self,
        state: &mut Self::State,
        destination: &Address,
    ) -> Result<HookOutcome<HookCallAction<Self::Outcome>>, Error<Self::Error>> {
        Ok(HookCallAction::Pass.into())
    }
}

pub trait ClonableHookConcolic: DynClone + HookConcolic { }
clone_trait_object!(
    <State, Error, Outcome> ClonableHookConcolic<State=State, Error=Error, Outcome=Outcome>
    where State: fuguex::state::State,
          Error: std::error::Error + Send + Sync + 'static,
);
