use crate::ast::op::{BinaryOp,UnaryOp};

#[derive(Debug, Clone)]
pub enum Expr {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),

    // Identifier reference like: foo
    Ident(String),

    // Useful for your DSL:
    Topology(TopologyExpr),
    Call(CallExpr),

    // Optional now, useful soon:
    Unary { op: UnaryOp, expr: Box<Expr> },
    Binary { left: Box<Expr>, op: BinaryOp, right: Box<Expr> }
}

#[derive(Debug, Clone)]
pub enum TopologyExpr {
    Sparse(f64),
    Identity,
    WeightedSum,
    Dense,
}

#[derive(Debug, Clone)]
pub struct CallExpr {
    pub name: String,
    pub args: Vec<Expr>,
}