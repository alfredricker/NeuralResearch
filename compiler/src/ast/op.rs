#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum UnaryOp {
    Neg,   // -x
    Not,   // !x
    Abs,   // abs(x)
    Sign,  // sign(x)
    Round, // round(x)
    Floor, // floor(x)
    Ceil,  // ceil(x)
    Trunc, // trunc(x)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BinaryOp {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Mod,
    Eq,
    Ne,
    Lt,
    Le,
    Gt,
    Ge,
    And,
    Or,
    Xor,
}