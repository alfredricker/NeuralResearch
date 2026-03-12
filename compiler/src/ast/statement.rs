use crate::ast::{
    io::{InputDecl, OutputDecl},
    link::LinkDecl,
};
use crate::ast::var::VarDecl;

#[derive(Debug, Clone)]
pub enum Statement {
    Input(InputDecl),
    Output(OutputDecl),
    Link(LinkDecl),
    Var(VarDecl),
}
