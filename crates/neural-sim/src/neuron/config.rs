use crate::math::sample::{SamplerU8, SamplerI8};

pub struct NeuronConfig {
    // ==== NAME ====
    pub name: &'static str, // "L1Simple", "L5Pyramidal", "CA1Hippocampal" etc.

    // ==== TOPOLOGY ====
    pub n_basal_dendrites: u8, // proximal feedforward input
    pub n_apical_dendrites: Option<u8>, // distal top down input

    // ==== RANDOM PARAMETERS ====
    pub synapse_x_sampler: SamplerU8, // synapse position distribution
    pub dendrites_per_branch: SamplerU8,
    pub synapses_per_dendrite: SamplerU8,

    // ==== SOMA ====
    pub soma_threshold: i8,

    // ==== BASAL DENDRITES ====
    pub basal_dendrite_threshold: u16,
    pub basal_dendrite_constant: SamplerI8,

    // ==== APICAL DENDRITES ====
    pub apical_dendrite_threshold: Option<u16>,
    pub apical_dendrite_constant: Option<SamplerI8>,

    // ==== LEARNING ====
    pub learning_rate: i16,
}

impl NeuronConfig {
    pub fn new(
        name: &'static str,
        n_basal_dendrites: u8,
        n_apical_dendrites: Option<u8>,
        synapse_x_sampler: SamplerU8,
        dendrites_per_branch: SamplerU8,
        synapses_per_dendrite: SamplerU8,
        soma_threshold: i8,
        basal_dendrite_threshold: u16,
        basal_dendrite_constant: SamplerI8,
        apical_dendrite_threshold: Option<u16>,
        apical_dendrite_constant: Option<SamplerI8>,
        learning_rate: i16,
    ) -> Self {
        Self {
            name,
            n_basal_dendrites,
            n_apical_dendrites,
            synapse_x_sampler,
            dendrites_per_branch,
            synapses_per_dendrite,
            soma_threshold,
            basal_dendrite_threshold,
            basal_dendrite_constant,
            apical_dendrite_threshold,
            apical_dendrite_constant,
            learning_rate,
        }
    }
}