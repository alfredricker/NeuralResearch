use crate::ast::{io::{InputKind,OutputKind},link::{LinkDecl}};

#[derive(Debug, Clone)]
pub enum Statement {
    Input(InputKind),
    Output(OutputKind),
    Link(LinkDecl),
    Var(VarDecl),
}

#[derive(Debug, Clone)]
pub enum VarValue {
    Int(i64),
    Float(f64),
    Bool(bool),
    String(String),

    // Identifier reference like: foo
    Ident(String),

    // Useful for your DSL:
    Topology(TopologyExpr),
    FunctionCall(FunctionCall),
}

#[derive(Debug, Clone)]
pub struct VarDecl {
    pub name: String,
    pub value: VarValue,
}

#[derive(Debug, Clone)]
pub enum TopologyExpr {
    Sparse(f64),
    Identity,
    WeightedSum,
    Dense,
}

#[derive(Debug, Clone)]
pub struct FunctionCall {
    pub name: String,
    pub args: Vec<VarValue>,
}