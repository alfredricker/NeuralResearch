use crate::ast::{io::{InputKind,OutputKind},link::{LinkDecl}};
use crate::ast::var::VarDecl;

#[derive(Debug, Clone)]
pub enum Statement {
    Input(InputKind),
    Output(OutputKind),
    Link(LinkDecl),
    Var(VarDecl),
}
