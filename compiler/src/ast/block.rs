use crate::ast::statement::Statement;

#[derive(Debug, Clone)]
pub enum Item {
    Block(Block),
    Statement(Statement),
}

#[derive(Debug, Clone)]
pub struct Block {
    pub kind: BlockKind,
    pub items: Vec<Item>,
}

#[derive(Debug, Clone)]
pub enum BlockKind {
    Graph,
    Subgraph,
    Top,
    Learn,
    Display,
    Transform,
}