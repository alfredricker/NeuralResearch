pub mod event;
pub mod topology;
pub mod build;

use rand::SeedableRng;
use rand::rngs::SmallRng;
use crate::network::build::{NetworkBuilder, build_network};
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
        let mut rng = SmallRng::seed_from_u64(862396277738699236);
        build_network(builder, &mut rng)
    }
}
