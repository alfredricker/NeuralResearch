pub mod event;
pub mod topology;
pub mod build;
pub mod alloc;

use crate::network::build::NetworkBuilder;
use crate::neuron::dendrite::Dendrite;
use crate::neuron::soma::Soma;
use crate::neuron::synapse::Synapse;
use crate::neuron::axon::Axon;

// SoA pattern for efficient GPU memory access. 
// Each field is a vector of length equal to the total number of neurons in the network.
// Essentially this is just a large structure of connected neurons
pub struct Network {
    synapses: Synapse,
    dendrites: Dendrite,
    somas: Soma,
    axons: Axon,
}

impl Network {
    pub fn build(builder: NetworkBuilder) -> Self {
        for c in &builder.connections {
            // map synapses of the population to the corresponding dendrites
            // of the target population according to the connection rule
        }
        // TODO: allocator + connection resolver not implemented yet (see docs/09-gaps).
        todo!("Network::build: allocate SoA arrays and resolve connections")
    }
}
