use std::fmt;
use std::ops::Deref;
use std::ops::{Add, BitAnd, BitOr, BitXor, Div, Mul, Neg, Not, Rem, Shl, Shr, Sub};
use std::sync::Arc;
use std::sync::atomic::AtomicU64;
use std::sync::atomic::Ordering::SeqCst;

use hashconsing::{consign, HConsed, HashConsign};

use fnv::FnvHashMap as HashMap;

use fugue::bv::BitVec;
use fugue::fp::FloatFormat;
use fugue::ir::il::ecode::{BinOp, BinRel, Cast, UnOp, UnRel, Var};
use fugue::ir::il::traits::*;
use fugue::ir::space::AddressSpaceId;

use smallvec::SmallVec;

consign! {
    let EXPR = consign(100 * 1024 /* = capacity */) for Expr;
}

static IVAR_FACTORY: AtomicU64 = AtomicU64::new(0);

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct IVar(u64, u32);

impl fmt::Display for IVar {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "ivar{:x}:{}", self.id(), self.bits())
    }
}

impl IVar {
    pub fn new(bits: u32) -> Self {
        Self(IVAR_FACTORY.fetch_add(1, SeqCst), bits)
    }

    pub fn bits(&self) -> u32 {
        self.1
    }

    pub fn id(&self) -> u64 {
        self.0
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
#[repr(transparent)]
pub struct SymExpr(HConsed<Expr>);

impl fmt::Display for SymExpr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        (**self).fmt(f)
    }
}

impl Deref for SymExpr {
    type Target = Expr;

    fn deref(&self) -> &Self::Target {
        &*self.0
    }
}

impl From<BitVec> for SymExpr {
    fn from(e: BitVec) -> Self {
        SymExpr::val(e)
    }
}

impl From<Var> for SymExpr {
    fn from(v: Var) -> Self {
        SymExpr::var(v)
    }
}

impl From<IVar> for SymExpr {
    fn from(v: IVar) -> Self {
        SymExpr::ivar(v)
    }
}

impl From<Expr> for SymExpr {
    fn from(e: Expr) -> Self {
        Self(EXPR.mk(e))
    }
}

impl From<HConsed<Expr>> for SymExpr {
    fn from(e: HConsed<Expr>) -> Self {
        Self(e)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Expr {
    UnRel(UnRel, SymExpr),            // T -> bool
    BinRel(BinRel, SymExpr, SymExpr), // T * T -> bool

    UnOp(UnOp, SymExpr),            // T -> T
    BinOp(BinOp, SymExpr, SymExpr), // T * T -> T

    Cast(SymExpr, Cast), // T -> Cast::T
    Load(SymExpr, usize, AddressSpaceId), // SPACE[T]:SIZE -> T

    ExtractLow(SymExpr, u32),
    ExtractHigh(SymExpr, u32),
    Extract(SymExpr, u32, u32), // T T[LSB..MSB) -> T

    Concat(SymExpr, SymExpr),   // T * T -> T

    IfElse(SymExpr, SymExpr, SymExpr),

    Intrinsic(Arc<str>, SmallVec<[SymExpr; 4]>, usize),

    Val(BitVec), // BitVec -> T

    IVar(IVar), // pure symbolic variable
    Var(Var),
}

impl Expr {
    fn fmt_l1(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Val(v) => write!(f, "{:#x}", v),
            Expr::Var(v) => write!(f, "{}", v),
            Expr::IVar(v) => write!(f, "{}", v),


            Expr::Cast(expr, Cast::Bool) => { expr.fmt_l1(f)?; write!(f, " as bool") },
            Expr::Cast(expr, Cast::Signed(bits)) => { expr.fmt_l1(f)?;  write!(f, " as i{}", bits) },
            Expr::Cast(expr, Cast::Unsigned(bits)) => { expr.fmt_l1(f)?; write!(f, " as u{}", bits) },
            Expr::Cast(expr, Cast::Float(format)) => { expr.fmt_l1(f)?; write!(f, " as f{}", format.bits()) },

            Expr::ExtractLow(expr, bits) => write!(f, "extract-low({}, bits={})", expr, bits),
            Expr::ExtractHigh(expr, bits) => write!(f, "extract-high({}, bits={})", expr, bits),
            Expr::Extract(expr, lsb, msb) => write!(f, "extract({}, from={}, to={})", expr, lsb, msb),
            Expr::UnOp(UnOp::ABS, expr) => write!(f, "abs({})", expr),
            Expr::UnOp(UnOp::SQRT, expr) => write!(f, "sqrt({})", expr),
            Expr::UnOp(UnOp::ROUND, expr) => write!(f, "round({})", expr),
            Expr::UnOp(UnOp::CEILING, expr) => write!(f, "ceiling({})", expr),
            Expr::UnOp(UnOp::FLOOR, expr) => write!(f, "floor({})", expr),
            Expr::UnOp(UnOp::POPCOUNT, expr) => write!(f, "popcount({})", expr),

            Expr::UnRel(UnRel::NAN, expr) => write!(f, "is-nan({})", expr),

            Expr::BinRel(BinRel::CARRY, e1, e2) => write!(f, "carry({}, {})", e1, e2),
            Expr::BinRel(BinRel::SCARRY, e1, e2) => write!(f, "scarry({}, {})", e1, e2),
            Expr::BinRel(BinRel::SBORROW, e1, e2) => write!(f, "sborrow({}, {})", e1, e2),

            expr => write!(f, "({})", expr),
        }
    }

    fn fmt_l2(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::UnOp(UnOp::NEG, expr) => { write!(f, "-")?; expr.fmt_l1(f) },
            Expr::UnOp(UnOp::NOT, expr) => { write!(f, "!")?; expr.fmt_l1(f) },
            expr => expr.fmt_l1(f)
        }
    }

    fn fmt_l3(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::BinOp(BinOp::MUL, e1, e2) => { e1.fmt_l3(f)?; write!(f, " * ")?; e2.fmt_l2(f) }
            Expr::BinOp(BinOp::DIV, e1, e2) => { e1.fmt_l3(f)?; write!(f, " / ")?; e2.fmt_l2(f) }
            Expr::BinOp(BinOp::SDIV, e1, e2) => { e1.fmt_l3(f)?; write!(f, " s/ ")?; e2.fmt_l2(f) }
            Expr::BinOp(BinOp::REM, e1, e2) => { e1.fmt_l3(f)?; write!(f, " % ")?; e2.fmt_l2(f) }
            Expr::BinOp(BinOp::SREM, e1, e2) => { e1.fmt_l3(f)?; write!(f, " s% ")?; e2.fmt_l2(f) }
            expr => expr.fmt_l2(f)
        }
    }

    fn fmt_l4(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::BinOp(BinOp::ADD, e1, e2) => { e1.fmt_l4(f)?; write!(f, " + ")?; e2.fmt_l3(f) },
            Expr::BinOp(BinOp::SUB, e1, e2) => { e1.fmt_l4(f)?; write!(f, " - ")?; e2.fmt_l3(f) },
            expr => expr.fmt_l3(f)
        }
    }

    fn fmt_l5(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::BinOp(BinOp::SHL, e1, e2) => { e1.fmt_l5(f)?; write!(f, " << ")?; e2.fmt_l4(f) },
            Expr::BinOp(BinOp::SHR, e1, e2) => { e1.fmt_l5(f)?; write!(f, " >> ")?; e2.fmt_l4(f) },
            Expr::BinOp(BinOp::SAR, e1, e2) => { e1.fmt_l5(f)?; write!(f, " s>> ")?; e2.fmt_l4(f) },
            expr => expr.fmt_l4(f)
        }
    }

    fn fmt_l6(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::BinRel(BinRel::LT, e1, e2) => { e1.fmt_l6(f)?; write!(f, " < ")?; e2.fmt_l5(f) },
            Expr::BinRel(BinRel::LE, e1, e2) => { e1.fmt_l6(f)?; write!(f, " <= ")?; e2.fmt_l5(f) },
            Expr::BinRel(BinRel::SLT, e1, e2) => { e1.fmt_l6(f)?; write!(f, " s< ")?; e2.fmt_l5(f) },
            Expr::BinRel(BinRel::SLE, e1, e2) => { e1.fmt_l6(f)?; write!(f, " s<= ")?; e2.fmt_l5(f) },
            expr => expr.fmt_l5(f)
        }
    }

    fn fmt_l7(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::BinRel(BinRel::EQ, e1, e2) => { e1.fmt_l7(f)?; write!(f, " == ")?; e2.fmt_l6(f) },
            Expr::BinRel(BinRel::NEQ, e1, e2) => { e1.fmt_l7(f)?; write!(f, " != ")?; e2.fmt_l6(f) },
            expr => expr.fmt_l6(f)
        }
    }

    fn fmt_l8(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Expr::BinOp(BinOp::AND, e1, e2) = self {
            e1.fmt_l8(f)?;
            write!(f, " & ")?;
            e2.fmt_l7(f)
        } else {
            self.fmt_l7(f)
        }
    }

    fn fmt_l9(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Expr::BinOp(BinOp::XOR, e1, e2) = self {
            e1.fmt_l9(f)?;
            write!(f, " ^ ")?;
            e2.fmt_l8(f)
        } else {
            self.fmt_l8(f)
        }
    }

    fn fmt_l10(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Expr::BinOp(BinOp::OR, e1, e2) = self {
            e1.fmt_l10(f)?;
            write!(f, " | ")?;
            e2.fmt_l9(f)
        } else {
            self.fmt_l9(f)
        }
    }

    fn fmt_l11(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Expr::Concat(e1, e2) = self {
            e1.fmt_l11(f)?;
            write!(f, " ++ ")?;
            e2.fmt_l10(f)
        } else {
            self.fmt_l10(f)
        }
    }

    fn fmt_l12(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if let Expr::IfElse(c, e1, e2) = self {
            write!(f, "if ")?;
            c.fmt_l12(f)?;
            write!(f, " then ")?;
            e1.fmt_l12(f)?;
            write!(f, " else ")?;
            e2.fmt_l12(f)
        } else {
            self.fmt_l11(f)
        }
    }
}

impl fmt::Display for Expr {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        self.fmt_l12(f)
    }
}

macro_rules! impl_from_for {
    ($t:ident, true) => {
        impl From<$t> for Expr {
            fn from(t: $t) -> Expr {
                Expr::Val(BitVec::from(t).signed())
            }
        }
    };
    ($t:ident, false) => {
        impl From<$t> for Expr {
            fn from(t: $t) -> Expr {
                Expr::Val(BitVec::from(t).unsigned())
            }
        }
    };
}

macro_rules! impls_from_for {
    ($($tname:ident),*; $signed:ident) => {
        $(
            impl_from_for!($tname, $signed);
        )*
    };
}

impl From<bool> for Expr {
    fn from(t: bool) -> Expr {
        let b = EXPR.mk(Expr::Val(BitVec::from_u8(if t { 1 } else { 0 }, 8)));
        Expr::Cast(b.into(), Cast::Bool)
    }
}

impl From<BitVec> for Expr {
    fn from(t: BitVec) -> Expr {
        Expr::Val(t)
    }
}

impls_from_for! { i8, i16, i32, i64, i128, isize; true }
impls_from_for! { u8, u16, u32, u64, u128, usize; false }

impl SymExpr {
    pub fn val<T: Into<Expr>>(t: T) -> SymExpr {
        EXPR.mk(t.into()).into()
    }

    pub fn var(var: Var) -> SymExpr {
        EXPR.mk(Expr::Var(var)).into()
    }

    pub fn ivar(ivar: IVar) -> SymExpr {
        EXPR.mk(Expr::IVar(ivar)).into()
    }

    pub fn lift_unop(op: UnOp, v: SymExpr) -> SymExpr {
        EXPR.mk(Expr::UnOp(op, v)).into()
    }

    pub fn neg(v: SymExpr) -> SymExpr {
        if let Expr::Val(ref v) = &*v {
            Self::val(-v)
        } else if let Expr::UnOp(UnOp::NEG, vv) = &*v {
            vv.clone()
        } else {
            Self::lift_unop(UnOp::NEG, v)
        }
    }

    pub fn bool_not(v: SymExpr) -> SymExpr {
        v.cast_bool().not()
    }

    pub fn not(v: SymExpr) -> SymExpr {
        if let Expr::Val(ref v) = &*v {
            Self::val(!v)
        } else if let Expr::UnOp(UnOp::NOT, vv) = &*v {
            vv.clone()
        } else {
            Self::lift_unop(UnOp::NOT, v)
        }
    }

    pub fn abs(self) -> SymExpr {
        Self::lift_unop(UnOp::ABS, self)
    }

    pub fn sqrt(self) -> SymExpr {
        Self::lift_unop(UnOp::SQRT, self)
    }

    pub fn round(self) -> SymExpr {
        Self::lift_unop(UnOp::ROUND, self)
    }

    pub fn ceiling(self) -> SymExpr {
        Self::lift_unop(UnOp::CEILING, self)
    }

    pub fn floor(self) -> SymExpr {
        Self::lift_unop(UnOp::FLOOR, self)
    }

    #[deprecated(since = "0.0", note="use `SymExpr::count_ones`")]
    pub fn popcount(self) -> SymExpr {
        Self::lift_unop(UnOp::POPCOUNT, self)
    }

    pub fn count_ones(self) -> SymExpr {
        Self::lift_unop(UnOp::POPCOUNT, self)
    }


    pub fn lift_binop(op: BinOp, l: SymExpr, r: SymExpr) -> SymExpr {
        assert_eq!(l.bits(), r.bits());

        EXPR.mk(Expr::BinOp(op, l, r)).into()
    }

    pub fn bool_and(l: SymExpr, r: SymExpr) -> SymExpr {
        Self::and(l.cast_bool(), r.cast_bool())
    }

    pub fn bool_or(l: SymExpr, r: SymExpr) -> SymExpr {
        Self::or(l.cast_bool(), r.cast_bool())
    }

    pub fn bool_xor(l: SymExpr, r: SymExpr) -> SymExpr {
        Self::xor(l.cast_bool(), r.cast_bool())
    }

    pub fn and(l: SymExpr, r: SymExpr) -> SymExpr {
        if l.is_zero() || r.is_ones() {
            l
        } else if r.is_zero() || l.is_ones() {
            r
        } else if let (Expr::Val(ref lv), Expr::Val(ref rv)) = (&*l, &*r) {
            Self::val(lv & rv)
        } else {
            Self::lift_binop(BinOp::AND, l, r)
        }
    }

    pub fn or(l: SymExpr, r: SymExpr) -> SymExpr {
        if l.is_zero() || r.is_ones() {
            r
        } else if r.is_zero() || l.is_ones() {
            l
        } else if let (Expr::Val(ref lv), Expr::Val(ref rv)) = (&*l, &*r) {
            Self::val(lv | rv)
        } else {
            Self::lift_binop(BinOp::OR, l, r)
        }
    }

    pub fn xor(l: SymExpr, r: SymExpr) -> SymExpr {
        assert_eq!(l.bits(), r.bits());

        if l == r {
            Self::val(BitVec::zero(l.bits() as usize))
        } else if let (Expr::Val(ref lv), Expr::Val(ref rv)) = (&*l, &*r) {
            Self::val(lv ^ rv)
        } else if r.is_ones() {
            Self::not(l)
        } else if l.is_ones() {
            Self::not(r)
        } else if r.is_zero() {
            l
        } else if l.is_zero() {
            r
        } else {
            Self::lift_binop(BinOp::XOR, l, r)
        }
    }

    pub fn add(l: SymExpr, r: SymExpr) -> SymExpr {
        if l.is_zero() {
            r
        } else if r.is_zero() {
            l
        } else if let (Expr::Val(ref lv), Expr::Val(ref rv)) = (&*l, &*r) {
            Self::val(lv + rv)
        } else {
            Self::lift_binop(BinOp::ADD, l, r)
        }
    }

    pub fn mul(l: SymExpr, r: SymExpr) -> SymExpr {
        if l.is_zero() || r.is_zero() {
            Self::val(BitVec::zero(l.bits() as usize))
        } else if r.is_one() {
            l
        } else if l.is_one() {
            r
        } else if let (Expr::Val(ref lv), Expr::Val(ref rv)) = (&*l, &*r) {
            Self::val(lv * rv)
        } else {
            Self::lift_binop(BinOp::MUL, l, r)
        }
    }

    pub fn sub(l: SymExpr, r: SymExpr) -> SymExpr {
        if l.is_zero() {
            Self::neg(r)
        } else if r.is_zero() {
            l
        } else if let (Expr::Val(ref lv), Expr::Val(ref rv)) = (&*l, &*r) {
            Self::val(lv - rv)
        } else {
            Self::lift_binop(BinOp::SUB, l, r)
        }
    }

    pub fn div(l: SymExpr, r: SymExpr) -> SymExpr {
        if l.is_zero() || r.is_one() {
            l
        } else if let (Expr::Val(ref lv), Expr::Val(ref rv)) = (&*l, &*r) {
            Self::val(lv / rv)
        } else {
            Self::lift_binop(BinOp::DIV, l, r)
        }
    }

    pub fn signed_div(self, r: SymExpr) -> SymExpr {
        if self.is_zero() || r.is_one() {
            self.cast_signed()
        } else if let (Expr::Val(ref lv), Expr::Val(ref rv)) = (&*self, &*r) {
            Self::val(lv.signed_div(rv))
        } else {
            Self::lift_binop(BinOp::SDIV, self, r)
        }
    }

    pub fn rem(l: SymExpr, r: SymExpr) -> SymExpr {
        if let (Expr::Val(ref lv), Expr::Val(ref rv)) = (&*l, &*r) {
            Self::val(lv % rv)
        } else {
            Self::lift_binop(BinOp::REM, l, r)
        }
    }

    pub fn signed_rem(self, r: SymExpr) -> SymExpr {
        if let (Expr::Val(ref lv), Expr::Val(ref rv)) = (&*self, &*r) {
            Self::val(lv.signed_rem(rv))
        } else {
            Self::lift_binop(BinOp::REM, self, r)
        }
    }

    pub fn shl(l: SymExpr, r: SymExpr) -> SymExpr {
        if r.is_zero() || l.is_zero() {
            l
        } else if let (Expr::Val(ref lv), Expr::Val(ref rv)) = (&*l, &*r) {
            Self::val(lv << rv)
        } else {
            Self::lift_binop(BinOp::SHL, l, r)
        }
    }

    pub fn shr(l: SymExpr, r: SymExpr) -> SymExpr {
        if r.is_zero() || l.is_zero() {
            l
        } else if let (Expr::Val(ref lv), Expr::Val(ref rv)) = (&*l, &*r) {
            Self::val(lv >> rv)
        } else {
            Self::lift_binop(BinOp::SHR, l, r)
        }
    }

    pub fn signed_shr(self, r: SymExpr) -> SymExpr {
        if r.is_zero() || self.is_zero() {
           self
        } else if let (Expr::Val(ref lv), Expr::Val(ref rv)) = (&*self, &*r) {
            Self::val(lv.signed_shr(rv))
        } else {
            Self::lift_binop(BinOp::SAR, self, r)
        }
    }

    pub fn concat(self, r: SymExpr) -> SymExpr {
        let l = self;
        if let (Expr::Val(ref lv), Expr::Val(ref rv)) = (&*l, &*r) {
            let sz = lv.bits() + rv.bits();
            let h = lv.unsigned_cast(sz) << (rv.bits() as u32);
            let l = rv.unsigned_cast(sz);
            Self::val(h | l)
        } else {
            EXPR.mk(Expr::Concat(l, r)).into()
        }
    }

    pub fn extract_high(self, bits: u32) -> SymExpr {
        assert!(self.bits() >= bits);

        if bits == self.bits() {
            self
        } else if let Expr::Val(ref bv) = &*self {
            Self::val(bv.clone().unsigned() >> (bv.bits() as u32 - bits))
        } else {
            EXPR.mk(Expr::ExtractHigh(self, bits)).into()
        }
    }

    pub fn extract_low(self, bits: u32) -> SymExpr {
        assert!(self.bits() >= bits);

        if bits == self.bits() {
            self
        } else if let Expr::Val(ref bv) = &*self {
            Self::val(bv.unsigned_cast(bits as usize))
        } else {
            EXPR.mk(Expr::ExtractLow(self, bits)).into()
        }
    }

    pub fn extract(self, lsb: u32, msb: u32) -> SymExpr {
        assert!(msb > lsb && msb - lsb <= self.bits());

        if (msb - lsb) == self.bits() {
            self
        } else if let Expr::Val(ref bv) = &*self {
            Self::val(if lsb > 0 {
                (bv >> lsb).unsigned_cast((msb - lsb) as usize)
            } else {
                bv.unsigned_cast((msb - lsb) as usize)
            })
        } else if let Expr::Extract(ref ibv, lsbi, _msbi) = &*self {
            let loff = lsb + lsbi;
            let moff = (msb - lsb) + loff;
            EXPR.mk(Expr::Extract(ibv.clone(), loff, moff)).into()
        } else {
            EXPR.mk(Expr::Extract(self, lsb, msb)).into()
        }
    }

    pub fn cast_signed(self) -> SymExpr {
        if self.is_signed() {
            self
        } else if let Expr::Val(ref bv) = &*self {
            EXPR.mk(bv.clone().signed().into()).into()
        } else {
            let bits = self.bits();
            Self::cast(self, Cast::Signed(bits as usize))
        }
    }

    pub fn cast_unsigned(self) -> SymExpr {
        if self.is_unsigned() {
            self
        } else if let Expr::Val(ref bv) = &*self {
            EXPR.mk(bv.clone().unsigned().into()).into()
        } else {
            let bits = self.bits();
            Self::cast(self, Cast::Unsigned(bits as usize))
        }
    }

    pub fn cast_bool(self) -> SymExpr {
        if self.is_bool() {
            self
        } else if let Expr::Val(ref bv) = &*self {
            Self::val(!bv.is_zero())
        } else {
            Self::cast(self, Cast::Bool)
        }
    }

    pub fn cast_float(self, format: Arc<FloatFormat>) -> SymExpr {
        if self.is_float() {
            self
        } else {
            Self::cast(self, Cast::Float(format))
        }
    }

    // sign extend *to* bits
    pub fn sign_extend(self, bits: u32) -> SymExpr {
        if self.is_signed() && bits == self.bits() {
            self
        } else if let Expr::Val(ref bv) = &*self {
            Self::val(bv.signed_cast(bits as usize))
        } else {
            Self::cast(self, Cast::Signed(bits as usize))
        }
    }

    // zero extend *to* bits
    pub fn zero_extend(self, bits: u32) -> SymExpr {
        if self.is_unsigned() && bits == self.bits() {
            self
        } else if let Expr::Val(ref bv) = &*self {
            Self::val(bv.unsigned_cast(bits as usize))
        } else {
            Self::cast(self, Cast::Unsigned(bits as usize))
        }
    }

    pub fn cast(e: SymExpr, c: Cast) -> SymExpr {
        EXPR.mk(Expr::Cast(e, c)).into()
    }

    pub fn lift_unrel(op: UnRel, e: SymExpr) -> SymExpr {
        EXPR.mk(Expr::UnRel(op, e)).into()
    }

    pub fn is_nan(self) -> SymExpr {
        Self::lift_unrel(UnRel::NAN, self)
    }

    pub fn ite(self, texpr: SymExpr, fexpr: SymExpr) -> SymExpr {
        assert!(self.is_bool());
        assert_eq!(texpr.bits(), fexpr.bits());

        EXPR.mk(Expr::IfElse(self, texpr, fexpr)).into()
    }

    pub fn lift_binrel(op: BinRel, l: SymExpr, r: SymExpr) -> SymExpr {
        assert_eq!(l.bits(), r.bits());

        EXPR.mk(Expr::BinRel(op, l, r)).into()
    }

    pub fn carry(self, r: SymExpr) -> SymExpr {
        let l = self;
        if let (Expr::Val(ref lv), Expr::Val(ref rv)) = (&*l, &*r) {
            Self::val(lv.carry(rv))
        } else {
            Self::lift_binrel(BinRel::CARRY, l, r)
        }
    }

    pub fn signed_carry(self, r: SymExpr) -> SymExpr {
        let l = self.cast_signed();
        let r = r.cast_signed();
        if let (Expr::Val(ref lv), Expr::Val(ref rv)) = (&*l, &*r) {
            Self::val(lv.signed_carry(rv))
        } else {
            Self::lift_binrel(BinRel::SCARRY, l, r)
        }
    }

    pub fn signed_borrow(self, r: SymExpr) -> SymExpr {
        let l = self.cast_signed();
        let r = r.cast_signed();
        if let (Expr::Val(ref lv), Expr::Val(ref rv)) = (&*l, &*r) {
            Self::val(lv.signed_borrow(rv))
        } else {
            Self::lift_binrel(BinRel::SBORROW, l, r)
        }
    }

    pub fn bool_eq(self, r: SymExpr) -> SymExpr {
        self.cast_bool().eq(r.cast_bool())
    }

    pub fn bool_ne(self, r: SymExpr) -> SymExpr {
        self.cast_bool().ne(r.cast_bool())
    }

    pub fn float_nan(self, fmts: &HashMap<usize, Arc<FloatFormat>>) -> SymExpr {
        let fmt = fmts[&(self.bits() as usize)].clone();
        self.cast_float(fmt.clone()).is_nan()
    }

    pub fn float_neg(self, fmts: &HashMap<usize, Arc<FloatFormat>>) -> SymExpr {
        let fmt = fmts[&(self.bits() as usize)].clone();
        self.cast_float(fmt.clone()).neg()
    }

    pub fn float_abs(self, fmts: &HashMap<usize, Arc<FloatFormat>>) -> SymExpr {
        let fmt = fmts[&(self.bits() as usize)].clone();
        self.cast_float(fmt.clone()).abs()
    }

    pub fn float_sqrt(self, fmts: &HashMap<usize, Arc<FloatFormat>>) -> SymExpr {
        let fmt = fmts[&(self.bits() as usize)].clone();
        self.cast_float(fmt.clone()).sqrt()
    }

    pub fn float_ceiling(self, fmts: &HashMap<usize, Arc<FloatFormat>>) -> SymExpr {
        let fmt = fmts[&(self.bits() as usize)].clone();
        self.cast_float(fmt.clone()).ceiling()
    }

    pub fn float_floor(self, fmts: &HashMap<usize, Arc<FloatFormat>>) -> SymExpr {
        let fmt = fmts[&(self.bits() as usize)].clone();
        self.cast_float(fmt.clone()).floor()
    }

    pub fn float_round(self, fmts: &HashMap<usize, Arc<FloatFormat>>) -> SymExpr {
        let fmt = fmts[&(self.bits() as usize)].clone();
        self.cast_float(fmt.clone()).round()
    }

    pub fn float_eq(self, r: SymExpr, fmts: &HashMap<usize, Arc<FloatFormat>>) -> SymExpr {
        let fmt = fmts[&(self.bits() as usize)].clone();
        self.cast_float(fmt.clone())
            .eq(r.cast_float(fmt))
    }

    pub fn float_ne(self, r: SymExpr, fmts: &HashMap<usize, Arc<FloatFormat>>) -> SymExpr {
        let fmt = fmts[&(self.bits() as usize)].clone();
        self.cast_float(fmt.clone())
            .ne(r.cast_float(fmt))
    }

    pub fn float_lt(self, r: SymExpr, fmts: &HashMap<usize, Arc<FloatFormat>>) -> SymExpr {
        let fmt = fmts[&(self.bits() as usize)].clone();
        self.cast_float(fmt.clone())
            .lt(r.cast_float(fmt))
    }

    pub fn float_le(self, r: SymExpr, fmts: &HashMap<usize, Arc<FloatFormat>>) -> SymExpr {
        let fmt = fmts[&(self.bits() as usize)].clone();
        self.cast_float(fmt.clone())
            .le(r.cast_float(fmt))
    }

    pub fn float_add(self, r: SymExpr, fmts: &HashMap<usize, Arc<FloatFormat>>) -> SymExpr {
        let fmt = fmts[&(self.bits() as usize)].clone();
        self.cast_float(fmt.clone())
            .add(r.cast_float(fmt))
    }

    pub fn float_div(self, r: SymExpr, fmts: &HashMap<usize, Arc<FloatFormat>>) -> SymExpr {
        let fmt = fmts[&(self.bits() as usize)].clone();
        self.cast_float(fmt.clone())
            .div(r.cast_float(fmt))
    }

    pub fn float_mul(self, r: SymExpr, fmts: &HashMap<usize, Arc<FloatFormat>>) -> SymExpr {
        let fmt = fmts[&(self.bits() as usize)].clone();
        self.cast_float(fmt.clone())
            .mul(r.cast_float(fmt))
    }

    pub fn float_sub(self, r: SymExpr, fmts: &HashMap<usize, Arc<FloatFormat>>) -> SymExpr {
        let fmt = fmts[&(self.bits() as usize)].clone();
        self.cast_float(fmt.clone())
            .sub(r.cast_float(fmt))
    }

    pub fn eq(self, r: SymExpr) -> SymExpr {
        let l = self;
        if l == r {
            // trivial
            Self::val(true)
        } else if let (Expr::Val(ref lv), Expr::Val(ref rv)) = (&*l, &*r) {
            Self::val(lv == rv)
        } else {
            Self::lift_binrel(BinRel::EQ, l, r)
        }
    }

    pub fn ne(self, r: SymExpr) -> SymExpr {
        let l = self;
        if l == r {
            // trivial
            Self::val(false)
        } else if let (Expr::Val(ref lv), Expr::Val(ref rv)) = (&*l, &*r) {
            Self::val(lv != rv)
        } else {
            Self::lift_binrel(BinRel::NEQ, l, r)
        }
    }

    pub fn lt(self, r: SymExpr) -> SymExpr {
        let l = self;
        if let (Expr::Val(ref lv), Expr::Val(ref rv)) = (&*l, &*r) {
            assert!(lv.is_unsigned() && rv.is_unsigned());
            Self::val(lv.clone().unsigned() < rv.clone().unsigned())
        } else {
            Self::lift_binrel(BinRel::LT, l, r)
        }
    }

    pub fn slt(self, r: SymExpr) -> SymExpr {
        let l = self;
        if let (Expr::Val(ref lv), Expr::Val(ref rv)) = (&*l, &*r) {
            Self::val(lv.clone().signed() < rv.clone().signed())
        } else {
            Self::lift_binrel(BinRel::SLT, l, r)
        }
    }

    pub fn le(self, r: SymExpr) -> SymExpr {
        let l = self;
        if let (Expr::Val(ref lv), Expr::Val(ref rv)) = (&*l, &*r) {
            assert!(lv.is_unsigned() && rv.is_unsigned());
            Self::val(lv.clone().unsigned() <= rv.clone().unsigned())
        } else {
            Self::lift_binrel(BinRel::LE, l, r)
        }
    }

    pub fn sle(self, r: SymExpr) -> SymExpr {
        let l = self;
        if let (Expr::Val(ref lv), Expr::Val(ref rv)) = (&*l, &*r) {
            assert!(lv.is_signed() && rv.is_signed());
            Self::val(lv.clone().signed() <= rv.clone().signed())
        } else {
            Self::lift_binrel(BinRel::SLE, l, r)
        }
    }

    pub fn load(expr: SymExpr, bits: usize, space: AddressSpaceId) -> SymExpr {
        Self(EXPR.mk(Expr::Load(expr, bits, space)))
    }

    pub fn intrinsic<S: AsRef<str>, I: IntoIterator<Item=SymExpr>>(name: S, args: I, bits: usize) -> SymExpr {
        Self(EXPR.mk(Expr::Intrinsic(Arc::from(name.as_ref()), args.into_iter().collect(), bits)))
    }

    pub fn bits(&self) -> u32 {
        match &**self {
            Expr::Val(ref v) => v.bits() as u32,
            Expr::Var(ref v) => v.bits() as u32,
            Expr::IVar(ref v) => v.bits(),
            Expr::UnOp(_, ref v) | Expr::BinOp(_, ref v, _) => v.bits(),
            Expr::UnRel(_, _) | Expr::BinRel(_, _, _) => 8, // bool
            Expr::Cast(_, Cast::Bool) => 8, // bool
            Expr::Cast(_, c) => c.bits() as u32,
            Expr::IfElse(_, ref l, _) => l.bits(),
            Expr::Intrinsic(_, _, bits) => *bits as u32,
            Expr::Concat(ref l, ref r) => l.bits() + r.bits(),
            Expr::ExtractLow(_, bits) |
            Expr::ExtractHigh(_, bits) => *bits,
            Expr::Extract(_, lsb, msb) => msb - lsb,
            Expr::Load(_, bits, _) => *bits as u32,
        }
    }

    pub fn is_signed(&self) -> bool {
        match &**self {
            Expr::Val(ref v) => v.is_signed(),
            Expr::Cast(_, Cast::Signed(_))
            | Expr::BinOp(BinOp::SDIV | BinOp::SREM | BinOp::SAR, _, _)
            | Expr::UnOp(UnOp::NEG, _) => true,
            _ => false,
        }
    }

    pub fn is_unsigned(&self) -> bool {
        !self.is_signed()
    }

    pub fn is_int(&self) -> bool {
        !self.is_bool() && !self.is_float()
    }

    pub fn is_bool(&self) -> bool {
        match &**self {
            Expr::Cast(_, Cast::Bool) | Expr::UnRel(_, _) | Expr::BinRel(_, _, _) => true,
            Expr::UnOp(UnOp::NOT, ref v) => v.is_bool(),
            Expr::BinOp(BinOp::AND | BinOp::OR | BinOp::XOR, ref v1, ref v2) => v1.is_bool() || v2.is_bool(),
            _ => false,
        }
    }

    pub fn is_float(&self) -> bool {
        match &**self {
            Expr::UnOp(UnOp::ABS | UnOp::SQRT | UnOp::CEILING | UnOp::FLOOR | UnOp::ROUND, _) | Expr::Cast(_, Cast::Float(_)) => true,
            Expr::UnOp(UnOp::NEG, ref v) => v.is_float(),
            Expr::BinOp(BinOp::ADD | BinOp::SUB | BinOp::MUL | BinOp::DIV, ref v1, ref v2) => v1.is_float() || v2.is_float(),
            _ => false,
        }
    }

    pub fn is_one(&self) -> bool {
        matches!(&**self, Expr::Val(ref v) if v.is_one())
    }

    pub fn is_ones(&self) -> bool {
        matches!(&**self, Expr::Val(ref v) if (v.is_unsigned() && *v == v.max_value()) || (v.is_signed() && *v == BitVec::from_i8(-1, v.bits() as usize).signed()))
    }

    pub fn is_zero(&self) -> bool {
        matches!(&**self, Expr::Val(ref v) if v.is_zero())
    }

    pub fn is_true(&self) -> bool {
        matches!(&**self, Expr::Cast(ref v, Cast::Bool) if !v.is_zero())
    }

    pub fn is_false(&self) -> bool {
        matches!(&**self, Expr::Cast(ref v, Cast::Bool) if v.is_zero())
    }

    pub fn simplify(self) -> SymExpr {
        // default: rebuild + apply simplifications in ctors
        struct Simplify;
        impl<'expr> VisitMap<'expr> for Simplify {}

        Simplify.visit_expr(&self)
    }
}

impl Add for &'_ SymExpr {
    type Output = SymExpr;

    fn add(self, rhs: Self) -> Self::Output {
        SymExpr::add(self.clone(), rhs.clone())
    }
}

impl Add for SymExpr {
    type Output = SymExpr;

    fn add(self, rhs: Self) -> Self::Output {
        SymExpr::add(self, rhs)
    }
}

impl BitAnd for &'_ SymExpr {
    type Output = SymExpr;

    fn bitand(self, rhs: Self) -> Self::Output {
        SymExpr::and(self.clone(), rhs.clone())
    }
}

impl BitAnd for SymExpr {
    type Output = SymExpr;

    fn bitand(self, rhs: Self) -> Self::Output {
        SymExpr::and(self, rhs)
    }
}

impl BitOr for &'_ SymExpr {
    type Output = SymExpr;

    fn bitor(self, rhs: Self) -> Self::Output {
        SymExpr::or(self.clone(), rhs.clone())
    }
}

impl BitOr for SymExpr {
    type Output = SymExpr;

    fn bitor(self, rhs: Self) -> Self::Output {
        SymExpr::or(self, rhs)
    }
}

impl BitXor for &'_ SymExpr {
    type Output = SymExpr;

    fn bitxor(self, rhs: Self) -> Self::Output {
        SymExpr::xor(self.clone(), rhs.clone())
    }
}

impl BitXor for SymExpr {
    type Output = SymExpr;

    fn bitxor(self, rhs: Self) -> Self::Output {
        SymExpr::xor(self, rhs)
    }
}

impl Div for &'_ SymExpr {
    type Output = SymExpr;

    fn div(self, rhs: Self) -> Self::Output {
        if self.is_signed() || rhs.is_signed() {
            SymExpr::signed_div(self.clone(), rhs.clone())
        } else {
            SymExpr::div(self.clone(), rhs.clone())
        }
    }
}

impl Div for SymExpr {
    type Output = SymExpr;

    fn div(self, rhs: Self) -> Self::Output {
        if self.is_signed() || rhs.is_signed() {
            SymExpr::signed_div(self, rhs)
        } else {
            SymExpr::div(self, rhs)
        }
    }
}

impl Mul for &'_ SymExpr {
    type Output = SymExpr;

    fn mul(self, rhs: Self) -> Self::Output {
        SymExpr::mul(self.clone(), rhs.clone())
    }
}

impl Mul for SymExpr {
    type Output = SymExpr;

    fn mul(self, rhs: Self) -> Self::Output {
        SymExpr::mul(self, rhs)
    }
}

impl Neg for &'_ SymExpr {
    type Output = SymExpr;

    fn neg(self) -> Self::Output {
        SymExpr::neg(self.clone())
    }
}

impl Neg for SymExpr {
    type Output = SymExpr;

    fn neg(self) -> Self::Output {
        SymExpr::neg(self)
    }
}

impl Not for &'_ SymExpr {
    type Output = SymExpr;

    fn not(self) -> Self::Output {
        SymExpr::not(self.clone())
    }
}

impl Not for SymExpr {
    type Output = SymExpr;

    fn not(self) -> Self::Output {
        SymExpr::not(self)
    }
}

impl Rem for &'_ SymExpr {
    type Output = SymExpr;

    fn rem(self, rhs: Self) -> Self::Output {
        if self.is_signed() || rhs.is_signed() {
            SymExpr::signed_rem(self.clone(), rhs.clone())
        } else {
            SymExpr::rem(self.clone(), rhs.clone())
        }
    }
}

impl Rem for SymExpr {
    type Output = SymExpr;

    fn rem(self, rhs: Self) -> Self::Output {
        if self.is_signed() || rhs.is_signed() {
            SymExpr::signed_rem(self, rhs)
        } else {
            SymExpr::rem(self, rhs)
        }
    }
}

impl Shl for &'_ SymExpr {
    type Output = SymExpr;

    fn shl(self, rhs: Self) -> Self::Output {
        SymExpr::shl(self.clone(), rhs.clone())
    }
}

impl Shl for SymExpr {
    type Output = SymExpr;

    fn shl(self, rhs: Self) -> Self::Output {
        SymExpr::shl(self, rhs)
    }
}

impl Shr for &'_ SymExpr {
    type Output = SymExpr;

    fn shr(self, rhs: Self) -> Self::Output {
        if self.is_signed() || rhs.is_signed() {
            SymExpr::signed_shr(self.clone(), rhs.clone())
        } else {
            SymExpr::shr(self.clone(), rhs.clone())
        }
    }
}

impl Shr for SymExpr {
    type Output = SymExpr;

    fn shr(self, rhs: Self) -> Self::Output {
        if self.is_signed() || rhs.is_signed() {
            SymExpr::signed_shr(self, rhs)
        } else {
            SymExpr::shr(self, rhs)
        }
    }
}

impl Sub for &'_ SymExpr {
    type Output = SymExpr;

    fn sub(self, rhs: &SymExpr) -> Self::Output {
        SymExpr::sub(self.clone(), rhs.clone())
    }
}

impl Sub for SymExpr {
    type Output = SymExpr;

    fn sub(self, rhs: Self) -> Self::Output {
        SymExpr::sub(self, rhs)
    }
}


pub trait VisitRef<'expr> {
    fn visit_val_ref(&mut self, bv: &'expr BitVec);
    fn visit_var_ref(&mut self, var: &'expr Var);
    fn visit_ivar_ref(&mut self, ivar: &'expr IVar);

    #[allow(unused_variables)]
    fn visit_unop_ref(&mut self, op: UnOp, expr: &'expr SymExpr) {
        self.visit_expr_ref(expr);
    }

    #[allow(unused_variables)]
    fn visit_binop_ref(&mut self, op: BinOp, lexpr: &'expr SymExpr, rexpr: &'expr SymExpr) {
        self.visit_expr_ref(lexpr);
        self.visit_expr_ref(rexpr);
    }

    #[allow(unused_variables)]
    fn visit_unrel_ref(&mut self, op: UnRel, expr: &'expr SymExpr) {
        self.visit_expr_ref(expr);
    }

    #[allow(unused_variables)]
    fn visit_binrel_ref(&mut self, op: BinRel, lexpr: &'expr SymExpr, rexpr: &'expr SymExpr) {
        self.visit_expr_ref(lexpr);
        self.visit_expr_ref(rexpr);
    }

    #[allow(unused_variables)]
    fn visit_cast_ref(&mut self, expr: &'expr SymExpr, cast: &'expr Cast) {
        self.visit_expr_ref(expr);
    }

    #[allow(unused_variables)]
    fn visit_extract_ref(&mut self, expr: &'expr SymExpr, lsb: u32, msb: u32) {
        self.visit_expr_ref(expr);
    }

    #[allow(unused_variables)]
    fn visit_extract_low_ref(&mut self, expr: &'expr SymExpr, bits: u32) {
        self.visit_expr_ref(expr);
    }

    #[allow(unused_variables)]
    fn visit_extract_high_ref(&mut self, expr: &'expr SymExpr, bits: u32) {
        self.visit_expr_ref(expr);
    }

    fn visit_concat_ref(&mut self, lexpr: &'expr SymExpr, rexpr: &'expr SymExpr) {
        self.visit_expr_ref(lexpr);
        self.visit_expr_ref(rexpr);
    }

    fn visit_ite_ref(&mut self, cond: &'expr SymExpr, lexpr: &'expr SymExpr, rexpr: &'expr SymExpr) {
        self.visit_expr_ref(cond);
        self.visit_expr_ref(lexpr);
        self.visit_expr_ref(rexpr);
    }

    #[allow(unused_variables)]
    fn visit_load_ref(&mut self, expr: &'expr SymExpr, bits: usize, space: AddressSpaceId) {
        self.visit_expr_ref(expr);
    }

    #[allow(unused_variables)]
    fn visit_intrinsic_ref(&mut self, name: &str, args: &'expr [SymExpr], bits: usize) {
        for arg in args {
            self.visit_expr_ref(arg);
        }
    }

    fn visit_expr_ref(&mut self, expr: &'expr SymExpr) {
        match **expr {
            Expr::Val(ref v) => self.visit_val_ref(v),
            Expr::Var(ref v) => self.visit_var_ref(v),
            Expr::IVar(ref v) => self.visit_ivar_ref(v),
            Expr::UnOp(op, ref e) => self.visit_unop_ref(op, e),
            Expr::BinOp(op, ref l, ref r) => self.visit_binop_ref(op, l, r),
            Expr::UnRel(op, ref e) => self.visit_unrel_ref(op, e),
            Expr::BinRel(op, ref l, ref r) => self.visit_binrel_ref(op, l, r),
            Expr::ExtractLow(ref e, bits) => self.visit_extract_low_ref(e, bits),
            Expr::ExtractHigh(ref e, bits) => self.visit_extract_high_ref(e, bits),
            Expr::Extract(ref e, lsb, msb) => self.visit_extract_ref(e, lsb, msb),
            Expr::Concat(ref l, ref r) => self.visit_concat_ref(l, r),
            Expr::IfElse(ref c, ref l, ref r) => self.visit_ite_ref(c, l, r),
            Expr::Cast(ref e, ref c) => self.visit_cast_ref(e, c),
            Expr::Load(ref e, sz, spc) => self.visit_load_ref(e, sz, spc),
            Expr::Intrinsic(ref name, ref args, sz) => self.visit_intrinsic_ref(name, args, sz),
        }
    }
}

pub trait VisitMap<'expr> {
    fn visit_val(&mut self, bv: &'expr BitVec) -> SymExpr {
        EXPR.mk(Expr::Val(bv.clone())).into()
    }

    fn visit_var(&mut self, var: &'expr Var) -> SymExpr {
        EXPR.mk(Expr::Var(var.clone())).into()
    }

    fn visit_ivar(&mut self, ivar: &'expr IVar) -> SymExpr {
        EXPR.mk(Expr::IVar(ivar.clone())).into()
    }

    fn visit_unop(&mut self, op: UnOp, expr: &'expr SymExpr) -> SymExpr {
        SymExpr::lift_unop(op, self.visit_expr(expr))
    }

    fn visit_binop(&mut self, op: BinOp, lexpr: &'expr SymExpr, rexpr: &'expr SymExpr) -> SymExpr {
        SymExpr::lift_binop(op, self.visit_expr(lexpr), self.visit_expr(rexpr))
    }

    fn visit_unrel(&mut self, op: UnRel, expr: &'expr SymExpr) -> SymExpr {
        SymExpr::lift_unrel(op, self.visit_expr(expr))
    }

    fn visit_binrel(
        &mut self,
        op: BinRel,
        lexpr: &'expr SymExpr,
        rexpr: &'expr SymExpr,
    ) -> SymExpr {
        SymExpr::lift_binrel(op, self.visit_expr(lexpr), self.visit_expr(rexpr))
    }

    fn visit_cast(&mut self, expr: &'expr SymExpr, cast: &'expr Cast) -> SymExpr {
        SymExpr::cast(self.visit_expr(expr), cast.clone())
    }

    fn visit_extract_low(&mut self, expr: &'expr SymExpr, bits: u32) -> SymExpr {
        self.visit_expr(expr).extract_low(bits)
    }

    fn visit_extract_high(&mut self, expr: &'expr SymExpr, bits: u32) -> SymExpr {
        self.visit_expr(expr).extract_high(bits)
    }

    fn visit_extract(&mut self, expr: &'expr SymExpr, lsb: u32, msb: u32) -> SymExpr {
        self.visit_expr(expr).extract(lsb, msb)
    }

    fn visit_concat(&mut self, lexpr: &'expr SymExpr, rexpr: &'expr SymExpr) -> SymExpr {
        self.visit_expr(lexpr).concat(self.visit_expr(rexpr))
    }

    fn visit_ite(&mut self, cond: &'expr SymExpr, lexpr: &'expr SymExpr, rexpr: &'expr SymExpr) -> SymExpr {
        self.visit_expr(cond).ite(self.visit_expr(lexpr), self.visit_expr(rexpr))
    }

    fn visit_load(&mut self, expr: &'expr SymExpr, bits: usize, space: AddressSpaceId) -> SymExpr {
        SymExpr::load(self.visit_expr(expr), bits, space)
    }

    fn visit_intrinsic(&mut self, name: Arc<str>, args: &'expr [SymExpr], bits: usize) -> SymExpr {
        SymExpr::intrinsic(name, args.iter().map(|arg| self.visit_expr(arg)), bits)
    }

    fn visit_expr(&mut self, expr: &'expr SymExpr) -> SymExpr {
        match &**expr {
            Expr::Val(v) => self.visit_val(v),
            Expr::Var(v) => self.visit_var(v),
            Expr::IVar(v) => self.visit_ivar(v),
            Expr::UnOp(op, e) => self.visit_unop(*op, e),
            Expr::BinOp(op, l, r) => self.visit_binop(*op, l, r),
            Expr::UnRel(op, e) => self.visit_unrel(*op, e),
            Expr::BinRel(op, l, r) => self.visit_binrel(*op, l, r),
            Expr::IfElse(ref c, ref l, ref r) => self.visit_ite(c, l, r),
            Expr::ExtractLow(e, bits) => self.visit_extract_low(e, *bits),
            Expr::ExtractHigh(e, bits) => self.visit_extract_high(e, *bits),
            Expr::Extract(e, lsb, msb) => self.visit_extract(e, *lsb, *msb),
            Expr::Concat(l, r) => self.visit_concat(l, r),
            Expr::Cast(e, c) => self.visit_cast(e, c),
            Expr::Load(e, sz, spc) => self.visit_load(e, *sz, *spc),
            Expr::Intrinsic(name, args, sz) => self.visit_intrinsic(name.clone(), args, *sz),
        }
    }
}
