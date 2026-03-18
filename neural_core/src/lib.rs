pub mod activation;
pub mod learning;
pub mod graph;

/// The subgraph wiring framework: ports, nodes, graph, builder, flatten.
pub mod subgraph;

/// Node implementations for the subgraph framework (FeedForward, GridModule, etc.).
pub mod modules;

/// Burn-based traditional deep learning: MLP, pipeline, MNIST data loading.
pub mod burn;
