pub mod event;
pub mod topology;
pub mod build;

use std::ops::Range;
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
    // Global neuron-index range of each population, in `add()` order. The boundary layer
    // (`crate::io`) binds input/effector maps to these ranges after the network is built.
    ranges: Vec<Range<u32>>,
}

impl Network {
    pub fn build(builder: NetworkBuilder) -> Self {
        let mut rng = SmallRng::seed_from_u64(862396277738699236);
        build_network(builder, &mut rng)
    }

    /// Global neuron-index range of population `id` (the value returned by
    /// `NetworkBuilder::add`). Pixel/effector maps are bound to these ranges so an
    /// input space's local coordinates resolve to concrete global neuron indices.
    pub fn population_range(&self, id: u32) -> Range<u32> {
        self.ranges[id as usize].clone()
    }
}
