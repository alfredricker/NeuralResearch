use crate::init::neuron::config::NeuronConfig;

// Receives 28x28 pixel input. Basal dendrites handle local receptive field features;
// lower thresholds make it responsive to sparse pixel activations.
pub const CONFIG: NeuronConfig = NeuronConfig {
    // topology
    n_basal_dendrites:     6,
    n_apical_dendrites:     None,
    dendrites_per_branch:  8,
    synapses_per_dendrite: 16,
    mean_synapse_x:        128,
    std_synapse_x:         50,

    // soma — lower threshold to respond to weak pixel-level signals
    soma_threshold: 20,

    // basal dendrites — sensitive, low threshold
    basal_dendrite_threshold:      8_000,
    mean_basal_dendrite_constant:  60,
    std_basal_dendrite_constant:   8,

    // no apical input at this layer
    apical_dendrite_threshold:     None,
    mean_apical_dendrite_constant: 0,
    std_apical_dendrite_constant:  0,

    learning_rate: 256,
};
