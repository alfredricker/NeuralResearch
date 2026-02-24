use crate::ast::graph::*;
use crate::ast::io::*;
use crate::ast::link::*;
use crate::ast::display::*;
use crate::ast::defaults::*;

#[derive(Debug, Clone)]
pub struct Program {
    pub graph: GraphDecl,
    pub subgraphs: Vec<SubgraphDecl>,
    pub inputs: Vec<InputDecl>,
    pub outputs: Vec<OutputDecl>,
    pub links: Vec<LinkDecl>,
    pub display: Vec<DisplayDecl>,
    pub defaults: Vec<DefaultDecl>,
}