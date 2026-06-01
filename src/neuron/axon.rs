pub struct Axon {
    pub axon_targets: Vec<u32>, // flat list of synapse indices. allows for arbitrary connections
    // each neuron has 1 axon, so the axon_offsets index lines up with soma indices.
    pub axon_offsets: Vec<u32>, // axon_offsets[i] gives the start index in axon_targets for neuron i
}