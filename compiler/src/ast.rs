#[derive(Debug, Clone)]
pub struct Program {
    pub input: InputDecl,
    pub output: OutputDecl,
    pub graph: GraphDecl,
    pub links: Vec<LinkDecl>,
}

#[derive(Debug, Clone)]
pub struct InputDecl {
    pub name: String,
    pub width: u32,
    pub height: u32,
}

#[derive(Debug, Clone)]
pub struct OutputDecl {
    pub name: String,
    pub classes: u32,
}

#[derive(Debug, Clone)]
pub struct GraphDecl {
    pub node_groups: Vec<NodeGroupDecl>,
    pub edges: Vec<EdgeDecl>,
}

#[derive(Debug, Clone)]
pub struct NodeGroupDecl {
    pub name: String,
    pub count: u32,
}

#[derive(Debug, Clone)]
pub struct EdgeDecl {
    pub from: String,
    pub to: String,
    pub topology: Topology,
}

#[derive(Debug, Clone)]
pub struct LinkDecl {
    pub from: String,
    pub to: String,
    pub topology: Topology,
}

#[derive(Debug, Clone)]
pub enum Topology {
    Sparse(f64),
    Identity,
    WeightedSum,
}
