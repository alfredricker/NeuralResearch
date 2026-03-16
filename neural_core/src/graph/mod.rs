pub mod csr;
pub mod builder;

pub use csr::CsrGraph;
pub use builder::GraphBuilder;

/// Adjacency type tag — describes the wiring topology of a graph.
#[derive(Clone, Debug, PartialEq)]
pub enum Topology {
    Full,
    Ring(usize),            // degree k per side
    Circulant(Vec<usize>),  // explicit offset set
    Sparse(f32),            // Erdos-Renyi probability
    Identity,
    Custom,
}
