use z3::{Context, SatResult, Solver};
use z3::ast::{Ast, BV};

use fugue::ir::Translator;
use fugue::ir::il::ecode::Var;
use fugue::ir::il::traits::*;

use fxhash::FxHashMap as HashMap;

use std::sync::Arc;

use crate::expr::{IVar, SymExpr};
use crate::value::Value;

const SOLVER_LIMIT: usize = 100;

#[derive(Clone)]
pub struct SolverContext<'ctx> {
    pub(crate) context: &'ctx Context,
    pub(crate) solver: Solver<'ctx>,
    pub(crate) vars: HashMap<Var, BV<'ctx>>,
    pub(crate) ivars: HashMap<IVar, BV<'ctx>>,
    pub(crate) translator: Arc<Translator>,
    pub(crate) limit: usize,
}

impl<'ctx> SolverContext<'ctx> {
    pub fn new(context: &'ctx Context, translator: Arc<Translator>) -> Self {
        Self::new_with(translator, SOLVER_LIMIT)
    }

    fn context(&self) -> &'ctx Context {
        self.context
    }

    pub fn new_with(context: &'ctx Context, translator: Arc<Translator>, limit: usize) -> Self {
        Self {
            context,
            solver: Solver::new(context),
            vars: HashMap::default(),
            ivars: HashMap::default(),
            translator,
            limit,
        }
    }

    pub fn var(&mut self, var: &Var) -> BV<'ctx> {
        if let Some(bv) = self.vars.get(var).cloned() {
            bv
        } else {
            let bv = BV::fresh_const(self.context, "var", var.bits() as u32);
            self.vars.insert(var.to_owned(), bv.clone());
            bv
        }
    }

    pub fn ivar(&mut self, var: &IVar) -> BV<'ctx> {
        if let Some(bv) = self.ivars.get(var).cloned() {
            bv
        } else {
            let bv = BV::fresh_const(self.context, "ivar", var.bits() as u32);
            self.ivars.insert(var.to_owned(), bv.clone());
            bv
        }
    }

    pub fn solver(&self) -> &Solver {
        self.borrow_solver()
    }

    pub fn translator(&self) -> Arc<Translator> {
        self.translator.clone()
    }

    pub fn is_sat(&mut self, constraints: &[SymExpr]) -> bool {
        constraints.is_empty() || {
            self.solver.push();

            for constraint in constraints.iter() {
                self.solver.assert(&constraint.ast(self).extract(0, 0)._eq(&BV::from_u64(self.context, 1, 1)));
            }

            let is_sat = self.solver.check() == SatResult::Sat;

            self.solver.pop(1);

            is_sat
        }
    }
}

