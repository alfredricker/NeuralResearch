use super::port::{PortSpec, PortValues};
pub use crate::state::State;

/// The core compute unit in the generic graph network.
///
/// Every node declares its input and output ports via `input_ports` /
/// `output_ports`, then transforms data in `update` and applies learning in
/// `learn`.
pub trait Node: Send {
    fn input_ports(&self) -> &[PortSpec];
    fn output_ports(&self) -> &[PortSpec];

    /// Forward pass: read `inputs`, write `outputs`.
    fn update(&mut self, inputs: &PortValues, outputs: &mut PortValues);

    /// Hebbian (or other) learning step given the same inputs that were fed
    /// to the most recent `update`.
    fn learn(&mut self, inputs: &PortValues);
}
