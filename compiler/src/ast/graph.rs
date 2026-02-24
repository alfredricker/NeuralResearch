use crate::ast::link::*;

#[derive(Debug, Clone)]
pub struct GraphDecl {
    pub node_groups: Vec<NodeGroupDecl>,
    pub edges: Vec<LinkDecl>,
}

#[derive(Debug, Clone)]
pub struct SubgraphDecl {
    pub node_groups: Vec<NodeGroupDecl>,
    pub edges: Vec<LinkDecl>,
}

#[derive(Debug, Clone)]
pub struct NodeGroupDecl {
    pub name: String,
    pub count: u32,
}