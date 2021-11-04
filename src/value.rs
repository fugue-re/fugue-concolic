use boolector::{BV, BVSolution, Btor, SolverResult};

use fugue::bv::BitVec;
use fugue::ir::il::ecode::{BinOp, BinRel, Cast, UnOp, UnRel, Var};

use std::sync::Arc;

use crate::expr::{Expr, SymExpr, IVar, VisitRef};
use crate::solver::SolverContext;

struct ToAst<'c> {
    value: Option<BV<Arc<Btor>>>,
    ctxt: &'c mut SolverContext,
}

impl<'c> ToAst<'c> {
    fn new(ctxt: &'c mut SolverContext) -> Self {
        Self {
            value: None,
            ctxt,
        }
    }

    fn var(&mut self, var: &Var) -> BV<Arc<Btor>> {
        self.ctxt.var(var)
    }

    fn ivar(&mut self, ivar: &IVar) -> BV<Arc<Btor>> {
        self.ctxt.ivar(ivar)
    }

    fn solver(&self) -> Arc<Btor> {
        self.ctxt.solver()
    }

    fn value(&mut self) -> BV<Arc<Btor>> {
        self.value.take().unwrap()
    }
}

impl<'c, 'ecode> VisitRef<'ecode> for ToAst<'c> {
    fn visit_val_ref(&mut self, bv: &'ecode BitVec) {
        self.value = Some(if bv.bits() <= 64 {
            BV::from_u64(self.solver(), bv.to_u64().unwrap(), bv.bits() as u32)
        } else { // less than ideal...
            let s = format!("{:x}", bv);
            let s1 = s.strip_prefix("0x")
                .unwrap_or(&s);
            let s2 = s1.rsplit_once(':')
                .map(|(s1, _)| s1)
                .unwrap_or(&s1);

            BV::from_hex_str(self.solver(), s2, bv.bits() as u32)
        });
    }

    fn visit_var_ref(&mut self, var: &'ecode Var) {
        self.value = Some(self.var(var));
    }

    fn visit_ivar_ref(&mut self, ivar: &'ecode IVar) {
        self.value = Some(self.ivar(ivar));
    }

    fn visit_unrel_ref(&mut self, op: UnRel, _expr: &'ecode SymExpr) {
        panic!("unsupported operator: {:?}", op)
    }

    fn visit_unop_ref(&mut self, op: UnOp, expr: &'ecode SymExpr) {
        self.visit_expr_ref(expr);

        let value = self.value();
        self.value = Some(match op {
            UnOp::NOT => if expr.is_bool() {
                value.slice(0, 0).not().uext(7)
            } else {
                value.not()
            },
            UnOp::NEG => value.neg(),
            UnOp::ABS => {
                let c = value.sgte(&BV::zero(self.solver(), value.get_width()));
                c.cond_bv(&value, &value.neg())
            },
            UnOp::POPCOUNT => {
                let bv1 = BV::one(self.solver(), value.get_width());
                let mut cnt = BV::zero(self.solver(), value.get_width());

                // This is horrible.
                // TODO(slt): rewrite... Hacker's Delight
                for i in 0..value.get_width() {
                    let m = value.and(&bv1.sll(&BV::from_u32(self.solver(), i, value.get_width())));
                    cnt = m.cond_bv(&cnt.inc(), &cnt);
                }

                cnt
            },
            UnOp::SQRT |
            UnOp::FLOOR |
            UnOp::ROUND |
            UnOp::CEILING => panic!("unsupported operator: {:?}", op)
        })
    }

    fn visit_binrel_ref(&mut self, op: BinRel, lexpr: &'ecode SymExpr, rexpr: &'ecode SymExpr) {
        self.visit_expr_ref(lexpr);
        let lvalue = self.value();

        self.visit_expr_ref(rexpr);
        let rvalue = self.value();

        // NOTE: we uext by 7 to get byte-sized bools
        self.value = Some(match op {
            BinRel::EQ => lvalue._eq(&rvalue).uext(7),
            BinRel::NEQ => lvalue._ne(&rvalue).uext(7),
            BinRel::LT => lvalue.ult(&rvalue).uext(7),
            BinRel::SLT => lvalue.slt(&rvalue).uext(7),
            BinRel::LE => lvalue.ulte(&rvalue).uext(7),
            BinRel::SLE => lvalue.slte(&rvalue).uext(7),
            BinRel::CARRY => lvalue.uaddo(&rvalue).uext(7),
            BinRel::SCARRY => lvalue.saddo(&rvalue).uext(7),
            BinRel::SBORROW => lvalue.ssubo(&rvalue).uext(7),
        })
    }

    fn visit_binop_ref(&mut self, op: BinOp, lexpr: &'ecode SymExpr, rexpr: &'ecode SymExpr) {
        self.visit_expr_ref(lexpr);
        let lvalue = self.value();

        self.visit_expr_ref(rexpr);
        let rvalue = self.value();

        let is_bool = lexpr.is_bool() || rexpr.is_bool();

        self.value = Some(match op {
            BinOp::ADD => lvalue.add(&rvalue),
            BinOp::SUB => lvalue.sub(&rvalue),
            BinOp::DIV => lvalue.udiv(&rvalue),
            BinOp::SDIV => lvalue.sdiv(&rvalue),
            BinOp::MUL => lvalue.mul(&rvalue),
            BinOp::REM => lvalue.urem(&rvalue),
            BinOp::SREM => lvalue.srem(&rvalue),
            BinOp::AND => if is_bool { lvalue.slice(0, 0).and(&rvalue.slice(0, 0)) } else { lvalue.and(&rvalue) },
            BinOp::OR => if is_bool { lvalue.slice(0, 0).or(&rvalue.slice(0, 0)) } else { lvalue.or(&rvalue) },
            BinOp::XOR => if is_bool { lvalue.slice(0, 0).xor(&rvalue.slice(0, 0)) } else { lvalue.xor(&rvalue) },
            BinOp::SHL => lvalue.sll(&rvalue),
            BinOp::SHR => lvalue.srl(&rvalue),
            BinOp::SAR => lvalue.sra(&rvalue),
        })
    }

    fn visit_ite_ref(&mut self, cond: &'ecode SymExpr, texpr: &'ecode SymExpr, fexpr: &'ecode SymExpr) {
        self.visit_expr_ref(cond);
        let cvalue = self.value();

        self.visit_expr_ref(texpr);
        let lvalue = self.value();

        self.visit_expr_ref(fexpr);
        let rvalue = self.value();

        self.value = Some(cvalue.slice(0, 0).cond_bv(&lvalue, &rvalue))
    }

    fn visit_concat_ref(&mut self, lexpr: &'ecode SymExpr, rexpr: &'ecode SymExpr) {
        self.visit_expr_ref(lexpr);
        let lvalue = self.value();

        self.visit_expr_ref(rexpr);
        let rvalue = self.value();

        self.value = Some(lvalue.concat(&rvalue))
    }

    fn visit_extract_ref(&mut self, expr: &'ecode SymExpr, lsb: u32, msb: u32) {
        self.visit_expr_ref(&expr);
        let value = self.value();

        self.value = Some(value.slice(msb - 1, lsb))
    }

    fn visit_cast_ref(&mut self, expr: &'ecode SymExpr, cast: &'ecode Cast) {
        self.visit_expr_ref(&expr);
        let value = self.value();

        self.value = Some(match cast {
            Cast::Bool => value.slice(0, 0).uext(7),
            Cast::Low(bits) => value.slice(*bits as u32 - 1, 0),
            Cast::High(bits) => {
                let hbit = expr.bits() as u32;
                value.slice(hbit - 1, hbit - *bits as u32)
            },
            Cast::Signed(bits) => if expr.bits() < *bits as u32 {
                value.sext((*bits as u32 - expr.bits()) as u32)
            } else if expr.bits() > *bits as u32 {
                value.slice(*bits as u32 - 1, 0)
            } else {
                value
            },
            Cast::Unsigned(bits) => if expr.bits() < *bits as u32 {
                value.uext((*bits as u32 - expr.bits()) as u32)
            } else if expr.bits() > *bits as u32 {
                value.slice(*bits as u32 - 1, 0)
            } else {
                value
            },
            Cast::Float(_) => panic!("unsupported cast: {:?}", cast)
        })
    }
}

pub trait Value {
    fn ast(&self, ctxt: &mut SolverContext) -> BV<Arc<Btor>>;
    fn simplify(&self) -> SymExpr;
    fn solve(&self, ctxt: &mut SolverContext, constraints: &[SymExpr]) -> Option<BitVec>;
    fn solve_many(&self, ctxt: &mut SolverContext, constraints: &[SymExpr]) -> Vec<BitVec>;
}

impl Value for SymExpr {
    fn ast(&self, ctxt: &mut SolverContext) -> BV<Arc<Btor>> {
        let mut visitor = ToAst::new(ctxt);
        visitor.visit_expr_ref(self);
        visitor.value()
    }

    fn simplify(&self) -> SymExpr {
        self.to_owned().simplify()
    }

    fn solve(&self, ctxt: &mut SolverContext, constraints: &[SymExpr]) -> Option<BitVec> {
        let nx = self.simplify();
        if let Expr::Val(nx) = &*nx {
            Some(nx.clone())
        } else {
            ctxt.solver.push(1);

            for constraint in constraints.iter() {
                constraint.ast(ctxt).slice(0, 0).assert();
            }

            let ast = nx.ast(ctxt);
            let var = BV::new(ctxt.solver(), ast.get_width(), None);

            var._eq(&ast).assert();

            let nx = if ctxt.solver().sat() == SolverResult::Sat {
                Some(bv_from_solution(&var.get_a_solution().disambiguate(), nx.bits()))
            } else {
                None
            };

            ctxt.solver().pop(1);

            nx
        }
    }

    fn solve_many(&self, ctxt: &mut SolverContext, constraints: &[SymExpr]) -> Vec<BitVec> {
        let nx = self.simplify();
        if let Expr::Val(nx) = &*nx {
            vec![nx.clone()]
        } else {
            ctxt.solver().push(1);

            for constraint in constraints.iter() {
                constraint.ast(ctxt).slice(0, 0).assert();
            }

            let ast = nx.ast(ctxt);
            let var = BV::new(ctxt.solver(), ast.get_width(), None);

            var._eq(&ast).assert();

            let mut acc = Vec::with_capacity(ctxt.limit);
            while ctxt.solver().sat() == SolverResult::Sat && acc.len() < ctxt.limit {
                let soln = var.get_a_solution().disambiguate();
                acc.push(bv_from_solution(&soln, nx.bits()));
                var._ne(&BV::from_binary_str(ctxt.solver(), soln.as_01x_str())).assert();
            }

            ctxt.solver().pop(1);
            acc
        }
    }
}

fn bv_from_solution(soln: &BVSolution, bits: u32) -> BitVec {
    let bits = bits as usize;
    let x01 = soln.as_01x_str();
    if let Some(v) = soln.as_u64() {
        BitVec::from_u64(v, bits)
    } else {
        let mut bv = BitVec::zero(bits);
        for (i, c) in x01.bytes().rev().take(bits).enumerate() {
            if c != b'0' {
                bv.set_bit(i as u32)
            }
        }
        bv
    }
}
