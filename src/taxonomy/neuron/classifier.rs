use crate::init::neuron::config::NeuronConfig;

// Output layer: one neuron per digit class (0-9).
// Apical dendrites receive top-down feedback; basal receive feedforward feature input.
// Higher thresholds enforce selectivity — only fire for the correct class.
pub const CONFIG: NeuronConfig = NeuronConfig {
    // topology
    n_basal_dendrites:     8,
    n_apical_dendrites:    Some(2),
    dendrites_per_branch:  6,
    synapses_per_dendrite: 20,
    mean_synapse_x:        128,
    std_synapse_x:         30,

    // soma — higher threshold, selective firing
    soma_threshold: 60,

    // basal dendrites — integrate feedforward class evidence
    basal_dendrite_threshold:      16_000,
    mean_basal_dendrite_constant:  100,
    std_basal_dendrite_constant:   12,

    // apical dendrites — gate on top-down feedback
    apical_dendrite_threshold:     Some(20_000),
    mean_apical_dendrite_constant: 80,
    std_apical_dendrite_constant:  10,

    learning_rate: 128,  // slower learning at output layer
};
