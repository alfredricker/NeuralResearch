struct Axon {
    pub axon_targets: Vec<u32>, // flat list of synapse indices
    pub axon_offsets: Vec<u32>, // axon_offsets[i] gives the start index in axon_targets for neuron i
}