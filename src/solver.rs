use boolector::{BV, Btor, SolverResult};
use boolector::option::{BtorOption, ModelGen};

use fugue::ir::Translator;
use fugue::ir::il::ecode::{Expr, Var};

use fxhash::FxHashMap as HashMap;
use std::sync::Arc;

use crate::value::Value;

const SOLVER_LIMIT: usize = 100;

#[derive(Clone)]
pub struct SolverContext {
    pub(crate) solver: Arc<Btor>,
    pub(crate) vars: HashMap<Var, BV<Arc<Btor>>>,
    pub(crate) translator: Arc<Translator>,
    pub(crate) limit: usize,
}

impl SolverContext {
    pub fn new(translator: Arc<Translator>) -> Self {
        Self::new_with(translator, SOLVER_LIMIT)
    }

    pub fn new_with(translator: Arc<Translator>, limit: usize) -> Self {
        let solver = Btor::new();
        solver.set_opt(BtorOption::ModelGen(ModelGen::Asserted));
        solver.set_opt(BtorOption::Incremental(true));
        Self {
            solver: Arc::new(solver),
            vars: HashMap::default(),
            translator,
            limit,
        }
    }

    pub fn var(&mut self, var: &Var) -> BV<Arc<Btor>> {
        if let Some(bv) = self.vars.get(var).cloned() {
            bv
        } else {
            let bv = BV::new(self.solver(), var.bits() as u32, None);
            self.vars.insert(var.to_owned(), bv.clone());
            bv
        }
    }

    pub fn solver(&self) -> Arc<Btor> {
        self.solver.clone()
    }

    pub fn translator(&self) -> Arc<Translator> {
        self.translator.clone()
    }

    pub fn is_sat(&mut self, constraints: &[Expr]) -> bool {
        constraints.is_empty() || {
            self.solver.push(1);

            for constraint in constraints.iter() {
                constraint.ast(self).slice(0, 0).assert();
            }

            let is_sat = self.solver.sat() == SolverResult::Sat;

            self.solver.pop(1);

            is_sat
        }
    }
}
