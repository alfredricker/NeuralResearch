//! Per-population base offsets into the global SoA arrays.
//!
//! Produced by the allocator (`super::alloc`). The connection resolver (deferred) uses
//! these bases to translate population-local neuron indices into global indices when
//! binding synapse slots and building the axon CSR. Also the seed of a future
//! residency/tile table.

#[derive(Clone, Debug)]
pub struct PopulationLayout {
    pub neuron_base: u32,        // index of this population's first neuron in the global soma arrays
    pub dendrite_base: u32,      // index of its first dendrite in the global dendrite arrays
    pub synapse_base: u32,       // index of its first synapse slot in the global synapse arrays
    pub dendrites_per_neuron: u32, // fixed D for this population (analytic dendrite stride)
    pub size: u32,               // number of neurons
}

#[derive(Clone, Debug, Default)]
pub struct NetworkLayout {
    pub populations: Vec<PopulationLayout>,
    pub total_neurons: u32,
    pub total_dendrites: u32,
    pub total_synapses: u32,
}
