use fugue::bv::BitVec;
use crate::expr::SymExpr;

pub trait ValueSolver<'ctx>: Clone + 'ctx {
    type Value: 'ctx;

    fn ast(&mut self, expr: &SymExpr) -> Self::Value;

    fn is_sat(&mut self, constraints: &[SymExpr]) -> bool;

    fn solve(&mut self, expr: &SymExpr, constraints: &[SymExpr]) -> Option<BitVec>;
    fn solve_many(&mut self, expr: &SymExpr, constraints: &[SymExpr]) -> Vec<BitVec>;

    fn minimise(&mut self, expr: &SymExpr, constraints: &[SymExpr]) -> Option<BitVec>;
    fn maximise(&mut self, expr: &SymExpr, constraints: &[SymExpr]) -> Option<BitVec>;
}
