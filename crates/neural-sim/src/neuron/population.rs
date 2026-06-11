use std::collections::BTreeSet;

use rand::RngExt;

use crate::constants::SYNAPSE_SLOTS_PER_DENDRITE;
use crate::math::sample::sample_u8_uniform;
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
        Self { name: config.name, config, size }
    }

    /// Appends this population's neurons to the end of the growing SoA arrays.
    ///
    /// Layout (per neuron): `D` dendrites (basal first, then apical), each owning a fixed
    /// block of `S = SYNAPSE_SLOTS_PER_DENDRITE` synapse slots. Offsets are therefore
    /// analytic — `dendrite_offsets[n] = dendrite_base + n*D`, `synapse_offsets[d] = d*S` —
    /// and the live (bound) synapses are packed at the front of each block, counted by
    /// `live_synapse_counts[d]`; the dead tail is zeroed.
    ///
    /// `D` is fixed for the whole population by sampling `dendrites_per_branch` once here.
    /// Trailing sentinels for `dendrite_offsets` / `synapse_offsets` are NOT added — the
    /// orchestrator appends them once after every population is generated.
    pub fn generate_neurons(
        &self,
        rng: &mut impl RngExt,
        soma: &mut Soma,
        dendrite: &mut Dendrite,
        synapse: &mut Synapse,
    ) {
        let config = self.config;
        let size = self.size as usize;

        // dendrites_per_branch sampled ONCE so every neuron in the population shares the
        // same geometry D (keeps dendrite_offsets analytic within the population).
        let dpb = config.dendrites_per_branch.sample(rng) as usize;
        let basal_ds = config.n_basal_dendrites as usize * dpb;
        let apical_ds = config.n_apical_dendrites.unwrap_or(0) as usize * dpb;
        let ds_per_neuron = basal_ds + apical_ds;

        // bases: where this population's slice begins in each global array.
        let neuron_base = soma.soma_potentials.len();
        let dendrite_base = dendrite.dendrite_activities.len();

        // --- soma arrays ---
        for local_n in 0..size {
            soma.soma_potentials.push(0);
            soma.soma_thresholds.push(config.soma_threshold);
            soma.soma_betas.push(0);
            soma.soma_last_events.push(0);
            soma.soma_lrs.push(config.learning_rate);
            // first dendrite of this neuron, in global dendrite-index space.
            soma.dendrite_offsets.push((dendrite_base + local_n * ds_per_neuron) as u32);
        }

        // dendrites first (samples + stores live_synapse_counts), then the synapse blocks
        // that read those counts back as their source of truth.
        self.generate_dendrites(rng, config, dendrite, neuron_base, dendrite_base, ds_per_neuron, basal_ds);
        self.generate_synapses(rng, config, dendrite, synapse, dendrite_base, ds_per_neuron);
    }

    fn generate_dendrites(
        &self,
        rng: &mut impl RngExt,
        config: &NeuronConfig,
        dendrite: &mut Dendrite,
        neuron_base: usize,
        dendrite_base: usize,
        ds_per_neuron: usize,
        basal_ds: usize,
    ) {
        const S: usize = SYNAPSE_SLOTS_PER_DENDRITE;
        let total_ds = self.size as usize * ds_per_neuron;

        for local_d in 0..total_ds {
            let global_d = dendrite_base + local_d;
            let owner = neuron_base + local_d / ds_per_neuron;
            // which dendrite this is WITHIN its neuron — basal slots come before apical.
            let is_apical = local_d % ds_per_neuron >= basal_ds;

            dendrite.dendrite_activities.push(0);
            dendrite.dendrite_last_events.push(0);
            dendrite.synapse_offsets.push((global_d * S) as u32);
            // live count is sampled here; generate_synapses fills exactly this many slots.
            let live = config.synapses_per_dendrite.sample(rng).min(S as u8);
            dendrite.live_synapse_counts.push(live);
            dendrite.dendrite_to_neuron.push(owner as u32);

            if is_apical {
                // panic is correct: apical params must be Some if apical dendrites are configured.
                let constant = config.apical_dendrite_constant.as_ref()
                    .expect("apical dendrites configured but apical_dendrite_constant is None")
                    .sample(rng);
                let threshold = config.apical_dendrite_threshold
                    .expect("apical dendrites configured but apical_dendrite_threshold is None");
                dendrite.dendrite_constants.push(constant);
                dendrite.dendrite_thresholds.push(threshold);
                dendrite.dendrite_is_apical.push(1);
            } else {
                dendrite.dendrite_constants.push(config.basal_dendrite_constant.sample(rng));
                dendrite.dendrite_thresholds.push(config.basal_dendrite_threshold);
                dendrite.dendrite_is_apical.push(0);
            }
        }
    }

    fn generate_synapses(
        &self,
        rng: &mut impl RngExt,
        config: &NeuronConfig,
        dendrite: &mut Dendrite,
        synapse: &mut Synapse,
        dendrite_base: usize,
        ds_per_neuron: usize,
    ) {
        const S: usize = SYNAPSE_SLOTS_PER_DENDRITE;
        let total_ds = self.size as usize * ds_per_neuron;

        for local_d in 0..total_ds {
            let global_d = dendrite_base + local_d;
            let live = dendrite.live_synapse_counts[global_d] as usize;

            // Draw `live` UNIQUE positions; the BTreeSet yields them sorted ascending for
            // free, satisfying the load-bearing invariant (xs sorted + unique per dendrite).
            // Rejection sampling can't always reach `live` distinct values, so cap attempts
            // and shrink the live count to whatever we actually got.
            let mut xs: BTreeSet<u8> = BTreeSet::new();
            let mut attempts = 0;
            while xs.len() < live && attempts < live * 8 {
                xs.insert(config.synapse_x_sampler.sample(rng));
                attempts += 1;
            }
            let actual = xs.len();
            if actual < live {
                dendrite.live_synapse_counts[global_d] = actual as u8;
            }

            // live prefix: ascending xs, weights U(0, 8) (all-excitatory init).
            for x in &xs {
                synapse.synapse_weights.push(sample_u8_uniform(0, 8, rng) as i8);
                synapse.synapse_x.push(*x);
                synapse.synapse_alphas.push(0);
                synapse.synapse_last_events.push(0);
            }
            // dead tail: zero the remaining slots so the block is always S wide.
            for _ in actual..S {
                synapse.synapse_weights.push(0);
                synapse.synapse_x.push(0);
                synapse.synapse_alphas.push(0);
                synapse.synapse_last_events.push(0);
            }
        }
    }
}
