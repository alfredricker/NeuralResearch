
#[derive(Debug, Clone)]
pub struct LinkDecl {
    pub from: String,
    pub to: String,
    pub topology: Topology,
}

#[derive(Debug, Clone)]
pub enum Topology {
    Sparse(f64),
    Dense,
    Identity,
    WeightedSum,
}