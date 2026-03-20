/// PopulationRule is use for structures such as modules that change their state based
/// on a local population
/// UpdateRule is just for the individual neuron and the information it receives

pub mod rules;
pub mod population;
pub mod state;
pub mod update;
pub use state::{Bounded, State};
pub use update::UpdateRule;
pub use population::PopulationRule;