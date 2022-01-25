use fugue::bytes::Order;
use fugue::bv::BitVec;

use crate::backend::ValueSolver;
use crate::expr::SymExpr;
use crate::state::{ConcolicState, Error};

pub trait StringStateOps {
    fn symbolic_strnlen(&mut self, addr: &SymExpr, n: &SymExpr) -> Result<SymExpr, Error>;
    fn symbolic_strncmp(&mut self, addr1: &SymExpr, addr2: &SymExpr, n: &SymExpr) -> Result<SymExpr, Error>;
}

impl<'ctx, O: Order, VS: ValueSolver<'ctx>> StringStateOps for ConcolicState<'ctx, O, VS> {
    fn symbolic_strnlen(&mut self, addr: &SymExpr, n: &SymExpr) -> Result<SymExpr, Error> {
        let last = self.search_memory_symbolic(
            addr,
            &BitVec::zero(8).into(),
            n,
        )?;

        let last_len = &last - addr;

        Ok(SymExpr::ite(
            last.eq(BitVec::zero(8).into()),
            n.clone(),
            last_len,
        ))
    }

    fn symbolic_strncmp(&mut self, addr1: &SymExpr, addr2: &SymExpr, n: &SymExpr) -> Result<SymExpr, Error> {
        let l1 = self.symbolic_strnlen(addr1, n)?;
        let l2 = self.symbolic_strnlen(addr2, n)?;

        let one = SymExpr::from(BitVec::one(n.bits() as usize));
        let len = SymExpr::ite(l1.clone().lt(l2.clone()), l1, l2) + one;

        self.compare_memory_symbolic(addr1, addr2, &len)
    }
}
