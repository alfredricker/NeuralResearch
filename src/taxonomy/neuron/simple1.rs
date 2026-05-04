use crate::init::neuron::config::NeuronConfig;

pub const CONFIG: NeuronConfig = NeuronConfig {
    // topology
    n_basal_dendrites:    4,
    n_apical_dendrites:   None,
    dendrites_per_branch: 6,
    synapses_per_dendrite: 12,
    mean_synapse_x:       128,
    std_synapse_x:        40,

    // soma
    soma_threshold: 40,

    // basal dendrites
    basal_dendrite_threshold:      12_000,
    mean_basal_dendrite_constant:  80,
    std_basal_dendrite_constant:   10,

    // apical dendrites (unused — n_apical_branches = 0)
    apical_dendrite_threshold:     None,
    mean_apical_dendrite_constant: 0,
    std_apical_dendrite_constant:  0,

    // learning
    learning_rate: 256,
};
