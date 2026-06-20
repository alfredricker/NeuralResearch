pub mod event;
pub mod topology;
pub mod build;

use std::ops::Range;
use rand::SeedableRng;
use rand::rngs::SmallRng;
use crate::network::build::{NetworkBuilder, build_network};
use crate::network::event::queue::EventQueue;
use crate::network::event::event::{Event, SYNAPSE_SIGNAL};
use crate::network::event::r#loop::run_event_loop;
use crate::neuron::dendrite::Dendrite;
use crate::neuron::soma::Soma;
use crate::neuron::synapse::Synapse;
use crate::neuron::axon::Axon;
use crate::telemetry::{NetworkView, TelemetrySink};

/// Borrowed, read-only view of the network's **fixed structure** — the static counterpart to
/// [`NetworkView`] (which is per-tick dynamic state). Every field is a slice into the SoA arrays the
/// network owns, so constructing one (via [`Network::topology`]) copies nothing. A viewer walks
/// these to lay out neurons: `dendrite_offsets[n]..dendrite_offsets[n+1]` are neuron `n`'s
/// dendrites; for dendrite `d`, its live synapses are `synapse_offsets[d]` for
/// `live_synapse_counts[d]` slots.
pub struct TopologyView<'a> {
    pub n_neurons: usize,
    // soma
    pub soma_thresholds: &'a [i8],
    /// Neuron → index of its first dendrite (length `n_neurons + 1`, trailing sentinel).
    pub dendrite_offsets: &'a [u32],
    // dendrite
    pub dendrite_is_apical: &'a [u8],
    pub dendrite_thresholds: &'a [u16],
    pub dendrite_constants: &'a [i8],
    pub live_synapse_counts: &'a [u8],
    pub dendrite_to_neuron: &'a [u32],
    /// Dendrite → index of its first synapse slot (length `n_dendrites + 1`, trailing sentinel).
    pub synapse_offsets: &'a [u32],
    // synapse
    pub synapse_x: &'a [u8],
}

// SoA pattern for efficient GPU memory access.
// Each field is a vector of length equal to the total number of neurons in the network.
// Essentially this is just a large structure of connected neurons
pub struct Network {
    synapses: Synapse,
    dendrites: Dendrite,
    somas: Soma,
    axons: Axon,
    // Global neuron-index range of each population, in `add()` order. The boundary layer
    // (`crate::io`) binds input/effector maps to these ranges after the network is built.
    ranges: Vec<Range<u32>>,
}

impl Network {
    pub fn build(builder: NetworkBuilder) -> Self {
        let mut rng = SmallRng::seed_from_u64(862396277738699236);
        build_network(builder, &mut rng)
    }

    /// Global neuron-index range of population `id` (the value returned by
    /// `NetworkBuilder::add`). Pixel/effector maps are bound to these ranges so an
    /// input space's local coordinates resolve to concrete global neuron indices.
    pub fn population_range(&self, id: u32) -> Range<u32> {
        self.ranges[id as usize].clone()
    }

    /// Total neuron count — the length of every soma array and of the `spike_counts` accumulator
    /// the trial harness allocates.
    pub fn n_neurons(&self) -> usize {
        self.somas.soma_potentials.len()
    }

    /// Total dendrite count across the network (for recording manifests / sizing).
    pub fn n_dendrites(&self) -> usize {
        self.dendrites.dendrite_activities.len()
    }

    /// Total synapse-slot count across the network, live + dead tail (for recording manifests).
    pub fn n_synapses(&self) -> usize {
        self.synapses.synapse_weights.len()
    }

    /// Drive the event loop forward by exactly **one wavefront** against this network's state.
    /// Threads `sink` for telemetry and accumulates somatic spikes into `spike_counts` (length must
    /// equal [`n_neurons`](Self::n_neurons)). This is the single seam the trial harness drives once
    /// per tick — it exists because the SoA arrays are private to the network; the loop's wide
    /// parameter list stays an internal detail.
    pub fn step(
        &mut self,
        queue: &EventQueue,
        sink: &mut impl TelemetrySink,
        spike_counts: &mut [u32],
    ) {
        run_event_loop(
            queue,
            sink,
            // soma
            &mut self.somas.soma_potentials,
            &self.somas.soma_thresholds,
            &mut self.somas.soma_betas,
            &mut self.somas.soma_last_events,
            &self.somas.soma_lrs,
            // dendrite
            &self.dendrites.dendrite_constants,
            &mut self.dendrites.dendrite_last_events,
            &mut self.dendrites.dendrite_activities,
            &self.dendrites.dendrite_thresholds,
            &self.dendrites.dendrite_is_apical,
            &self.dendrites.live_synapse_counts,
            // `dendrite_offsets` param: indexed by NEURON (neuron -> first dendrite), so it is the
            // soma's map, not the dendrite->synapse one passed as `synapse_offsets` below.
            &self.somas.dendrite_offsets,
            &self.dendrites.dendrite_to_neuron,
            // synapse
            &mut self.synapses.synapse_weights,
            &mut self.synapses.synapse_alphas,
            &mut self.synapses.synapse_last_events,
            &self.synapses.synapse_x,
            &self.dendrites.synapse_offsets,
            // axon
            &self.axons.axon_targets,
            &self.axons.axon_offsets,
            // readout
            spike_counts,
        );
    }

    /// Inject one presynaptic AP delivery onto synapse slot `synapse` — a `SYNAPSE_SIGNAL` carrying
    /// `burst`, exactly as a real upstream axon would deliver it. This is the playground's primitive
    /// for exciting an isolated neuron with no input population wired in: stimulate a synapse, then
    /// [`step`](Self::step) to watch the EPSP climb its dendrite into the soma. Pushed at `clock`
    /// (the timestamp the cascade carries); drained by the next `step`. `burst` < 1 is treated as 1
    /// by the receiving handler.
    pub fn stimulate(&self, queue: &EventQueue, synapse: u32, burst: i16, clock: u16) {
        queue.producer_handle().push(Event::with_payload(SYNAPSE_SIGNAL, synapse, clock, burst));
    }

    /// Borrow this network's *fixed* structure into a read-only [`TopologyView`]: the offset tables,
    /// compartment flags, and synapse positions a viewer needs to lay a neuron out. Static for the
    /// life of the network (unlike [`view`](Self::view), which is the per-tick dynamic state), so a
    /// caller reads it once at build. Copies nothing.
    pub fn topology(&self) -> TopologyView<'_> {
        TopologyView {
            n_neurons: self.n_neurons(),
            soma_thresholds: &self.somas.soma_thresholds,
            dendrite_offsets: &self.somas.dendrite_offsets,
            dendrite_is_apical: &self.dendrites.dendrite_is_apical,
            dendrite_thresholds: &self.dendrites.dendrite_thresholds,
            dendrite_constants: &self.dendrites.dendrite_constants,
            live_synapse_counts: &self.dendrites.live_synapse_counts,
            dendrite_to_neuron: &self.dendrites.dendrite_to_neuron,
            synapse_offsets: &self.dendrites.synapse_offsets,
            synapse_x: &self.synapses.synapse_x,
        }
    }

    /// Clear the transient per-trial dynamics — soma potentials and dendrite activities — back to
    /// rest, isolating one trial from the next. Learning state (`synapse_weights`, `synapse_alphas`,
    /// `soma_betas`) **persists by design**, and the `*_last_events` timestamp bookkeeping is left
    /// untouched so a monotonic clock keeps lazy decay correct across the boundary. `spike_counts`
    /// is the harness's to zero (it does so at the start of each trial).
    pub fn reset_dynamics(&mut self) {
        self.somas.soma_potentials.iter_mut().for_each(|v| *v = 0);
        self.dendrites.dendrite_activities.iter_mut().for_each(|v| *v = 0);
    }

    /// Borrow this network's SoA arrays into a read-only [`NetworkView`] for a telemetry keyframe
    /// at `timestamp`. `spike_counts` (harness-owned) completes the readout channel. Constructing
    /// the view copies nothing — it is the seam a `RecordingSink` reads to snapshot state.
    pub fn view<'a>(&'a self, timestamp: u16, spike_counts: &'a [u32]) -> NetworkView<'a> {
        NetworkView {
            timestamp,
            soma_potentials: &self.somas.soma_potentials,
            soma_betas: &self.somas.soma_betas,
            dendrite_activities: &self.dendrites.dendrite_activities,
            dendrite_is_apical: &self.dendrites.dendrite_is_apical,
            synapse_weights: &self.synapses.synapse_weights,
            synapse_alphas: &self.synapses.synapse_alphas,
            spike_counts,
        }
    }

    /// Enumerate the network's synaptic edges as `(src_neuron, dst_neuron, synapse_idx)` — one entry
    /// per live axon target. Walks the axon CSR (source neuron → outgoing synapse slots) and resolves
    /// each slot back to the dendrite that owns it and that dendrite's neuron. Topology is fixed at
    /// build time, so the result is stable across trials: the dashboard records it once to draw the
    /// network graph and to resolve a clicked neuron's afferents/efferents (with `synapse_idx` keying
    /// into a snapshot's `synapse_weights`/`synapse_alphas`).
    pub fn edges(&self) -> Vec<(u32, u32, u32)> {
        use crate::neuron::dendrite::synapse_to_dendrite;
        let mut edges = Vec::with_capacity(self.axons.axon_targets.len());
        for src in 0..self.n_neurons() as u32 {
            let lo = self.axons.axon_offsets[src as usize] as usize;
            let hi = self.axons.axon_offsets[src as usize + 1] as usize;
            for &syn in &self.axons.axon_targets[lo..hi] {
                let dendrite = synapse_to_dendrite(syn as usize, &self.dendrites.synapse_offsets);
                let dst = self.dendrites.dendrite_to_neuron[dendrite];
                edges.push((src, dst, syn));
            }
        }
        edges
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::constants::ALPHA_BOOST;
    use crate::math::sample::{SamplerI8, SamplerU8};
    use crate::neuron::config::NeuronConfig;
    use crate::telemetry::NullSink;

    // One neuron: 2 basal dendrites × 3 live synapses = 6 synapse slots, no apical.
    fn single_neuron() -> Network {
        let cfg: &'static NeuronConfig = Box::leak(Box::new(NeuronConfig::new(
            "test",
            2,                       // n_basal_dendrites
            None,                    // n_apical_dendrites
            SamplerU8::new(128, 50), // synapse_x_sampler
            SamplerU8::new(1, 0),    // dendrites_per_branch = 1
            SamplerU8::new(3, 0),    // synapses_per_dendrite = 3
            10,                      // soma_threshold
            1000,                    // basal_dendrite_threshold
            SamplerI8::new(60, 0),   // basal_dendrite_constant
            None,                    // apical_dendrite_threshold
            None,                    // apical_dendrite_constant
            120,                     // learning_rate
        )));
        let mut b = NetworkBuilder { populations: Vec::new(), connections: Vec::new() };
        b.add(cfg, 1);
        build_network(b, &mut SmallRng::seed_from_u64(7))
    }

    // `stimulate` injects a SYNAPSE_SIGNAL that the next `step` drains onto the target synapse,
    // boosting its alpha. Alphas init to 0, so a single boost lands the stimulated slot exactly at
    // ALPHA_BOOST (decay of 0 is 0, regardless of clock) while its neighbours stay untouched.
    #[test]
    fn stimulate_then_step_boosts_only_the_target_synapse() {
        let mut net = single_neuron();
        assert!(net.synapses.synapse_alphas.iter().all(|&a| a == 0));

        // Slot 2 is live in dendrite 0 (3 live synapses occupy the block's prefix; the rest of the
        // fixed-width block is dead tail).
        let queue = EventQueue::new(16);
        let target = 2u32;
        assert!((target as usize) < net.topology().live_synapse_counts[0] as usize);
        net.stimulate(&queue, target, 5, 0);

        let mut spike_counts = vec![0u32; net.n_neurons()];
        net.step(&queue, &mut NullSink, &mut spike_counts);

        for (i, &alpha) in net.synapses.synapse_alphas.iter().enumerate() {
            if i as u32 == target {
                assert_eq!(alpha, ALPHA_BOOST, "stimulated synapse should be boosted");
            } else {
                assert_eq!(alpha, 0, "synapse {i} should be untouched");
            }
        }
    }
}
