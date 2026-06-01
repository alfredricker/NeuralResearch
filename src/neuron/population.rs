use crate::neuron::config::NeuronConfig;
use crate::neuron::dendrite::Dendrite;
use crate::neuron::soma::Soma;
use crate::neuron::synapse::Synapse;

pub struct Population {
    pub name: &'static str, // "L1Simple", "L5Pyramidal", "CA1Hippocampal" etc.
    pub config: &'static NeuronConfig, // the type of neurons in this population
    pub size: u32, // number of neurons in this population
}

impl Population {
    pub fn new(config: &'static NeuronConfig, size: u32) -> Self {
        Self {name: config.name, config, size}
    }

    // edits the SoA by appending the neurons in this population to the end of the arrays.
    pub fn generate_neurons(
        &self, 
        rng: &mut impl rand::Rng,
        soma: &mut Soma,
        dendrite: &mut Dendrite,
        synapse: &mut Synapse,
    ) {
        let config = self.config;
        let size = self.size as usize;

        let neuron_base_idx = soma.soma_potentials.len() as u32;
        let dendrite_base_idx = dendrite.dendrite_activities.len() as u32;
        let synapse_base_idx = synapse.synapse_weights.len() as u32;

        // --- generate random vars ---
        let dpb = config.dendrites_per_branch.sample(rng) as usize;
        let basal_ds = config.n_basal_dendrites as usize * dpb;
        let apical_ds = config.n_apical_dendrites.unwrap_or(0) as usize * dpb;
        let ds_per_neuron = basal_ds + apical_ds;

        for local_n in 0..size {
            // --- soma ---
            soma.soma_potentials.push(0);
            soma.soma_thresholds.push(config.soma_threshold);
            soma.soma_betas.push(0);
            soma.soma_last_events.push(0);
            soma.soma_lrs.push(config.learning_rate);
            soma.dendrite_offsets.push(dendrite_base + (local_n * d_per_neuron) as u32);

            let neuron_idx = neuron_base + local_n as u32;

            for local_d in 0..d_per_neuron {
                let is_apical = local_d >= basal_ds;

                dendrite.dendrite_activities.push(0);
                dendrite.dendrite_last_events.push(0);
                if is_apical {
                    
                }
                dendrite.dendrite_constants.
            }
        }
    }
}