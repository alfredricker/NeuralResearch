pub mod port;
pub mod node;
pub mod graph;
pub mod builder;
pub mod flatten;

pub use port::{Aggregation, PortSpec, PortValues};
pub use node::Node;
pub use graph::{FlatGraph, FlatWire, SubgraphDef, Wire};
pub use builder::{NetworkBuilder, WireBuilder};
pub use flatten::BuildError;
