pub struct NeuronConfig {
    // ==== TOPOLOGY ====
    pub n_basal_dendrites: u8, // proximal feedforward input
    pub n_apical_dendrites: Option<u8>, // distal top down input

    // discrete mean and std for defining distribution
    // x is position along the dendrite, which influences firing dynamics
    pub mean_synapse_x: u8,
    pub std_synapse_x: u8,

    pub dendrites_per_branch: u8,
    pub synapses_per_dendrite: u8,

    // ==== SOMA ====
    pub soma_threshold: i8,

    // ==== BASAL DENDRITES ====
    pub basal_dendrite_threshold: u16,
    pub mean_basal_dendrite_constant: i8,
    pub std_basal_dendrite_constant: u8,

    // ==== APICAL DENDRITES ====
    pub apical_dendrite_threshold: Option<u16>,
    pub mean_apical_dendrite_constant: i8,
    pub std_apical_dendrite_constant: u8,

    // ==== LEARNING ====
    pub learning_rate: i16,
}