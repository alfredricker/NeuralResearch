//! Efferent half of the IO boundary: effectors and the readout arrow.
//!
//! An [`Effector`] is the mirror of an [`InputSpace`](crate::io::input::InputSpace). Where an
//! input space *asserts* somatic spikes on input neurons, an effector *reads* the somatic spikes
//! output neurons emit, and maps them back out to external classes / actions. It owns:
//!   1. a [`ReadoutMap`] — *the arrow*: a CSR mapping each external class to the output neuron(s)
//!      that vote for it (identity for MNIST; population coding later),
//!   2. a `neuron_range` — the global indices of those output neurons, attached via [`bind`].
//!
//! It holds no mutable network state. Spike observation lives in `run_event_loop`'s per-neuron
//! `spike_counts` accumulator (the harness zeroes it per trial); the effector only *reads* the
//! window of that buffer belonging to its output neurons.
//!
//! [`bind`]: Effector::bind

use std::ops::Range;
use std::sync::LazyLock;

use rand::RngExt;

use crate::constants::MSLR;
use crate::io::input::jitter;
use crate::math::sample::{SamplerI8, SamplerU8};
use crate::network::event::{Event, EventProducer};
use crate::neuron::config::NeuronConfig;

// ============================================================================================
// Output-neuron taxonomy
// ============================================================================================

/// The `NeuronConfig` for output (effector) neurons.
///
/// Unlike input neurons, output neurons DO integrate: they have basal dendrites that receive the
/// hidden layer's projection and a soma that bursts. The config is deliberately simpler than a
/// hidden-layer neuron — a **low `soma_threshold`** so it fires readily, few/simple dendrites, and
/// **no apical compartment** (the first-pass feedback path is direct injection, not apical BDP).
/// VALUES ARE UNTUNED STARTING POINTS — calibrate once the trial loop runs end-to-end.
static OUTPUT_CONFIG: LazyLock<NeuronConfig> = LazyLock::new(|| {
    NeuronConfig::new(
        "output",
        4,                       // n_basal_dendrites — receives the hidden projection
        None,                    // n_apical_dendrites — NONE (no apical feedback first pass)
        SamplerU8::new(128, 50), // synapse_x_sampler — spread positions along each dendrite
        SamplerU8::new(1, 0),    // dendrites_per_branch — simple: one dendrite per branch
        SamplerU8::new(16, 0),   // synapses_per_dendrite
        10,                      // soma_threshold — LOW: output neurons should fire readily
        500,                     // basal_dendrite_threshold
        SamplerI8::new(60, 0),   // basal_dendrite_constant — proximal
        None,                    // apical_dendrite_threshold
        None,                    // apical_dendrite_constant
        MSLR as i16,             // learning_rate — fast updates (delta ~10 at MSLR)
    )
});

/// The shared `&'static NeuronConfig` for output neurons; pass to `NetworkBuilder::add`.
pub fn output_config() -> &'static NeuronConfig {
    &OUTPUT_CONFIG
}

// ============================================================================================
// ReadoutMap — the arrow (map B): external class -> output neuron(s)
// ============================================================================================

/// The explicit map from external classes to output neurons, in CSR form. The mirror of
/// [`SensoryMap`](crate::io::input::SensoryMap): indexed by *class*, listing the neurons that vote
/// for it (so a class can be read out by a population of neurons, not just one). Neuron indices are
/// **local** (`0..n_neurons`); [`Effector::bind`] offsets them into global index space.
pub struct ReadoutMap {
    offsets: Vec<u32>, // len = n_classes + 1
    neurons: Vec<u32>, // flattened local output-neuron indices, grouped by class
    n_neurons: u32,
}

impl ReadoutMap {
    /// One output neuron per class, in order: class `c` is read from neuron `c`. The map MNIST uses.
    pub fn identity(n_classes: u32) -> Self {
        Self {
            offsets: (0..=n_classes).collect(),
            neurons: (0..n_classes).collect(),
            n_neurons: n_classes,
        }
    }

    /// Number of distinct output neurons across all classes.
    pub fn n_neurons(&self) -> u32 {
        self.n_neurons
    }

    /// Number of external classes (rows in the CSR).
    pub fn n_classes(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Local output-neuron indices that vote for class `c`.
    pub fn members(&self, c: usize) -> &[u32] {
        let lo = self.offsets[c] as usize;
        let hi = self.offsets[c + 1] as usize;
        &self.neurons[lo..hi]
    }
}

// ============================================================================================
// Effector
// ============================================================================================

/// One external output modality: a readout arrow and (after `bind`) the global indices of the
/// output neurons it reads. The efferent mirror of `InputSpace`.
pub struct Effector {
    pub name: &'static str,
    readout: ReadoutMap,
    /// Global neuron-index range of this effector's output neurons. Empty (`0..0`) until `bind`.
    neuron_range: Range<u32>,
}

impl Effector {
    /// An effector whose readout arrow is the identity (class `c` <- output neuron `c`).
    pub fn identity(name: &'static str, n_classes: u32) -> Self {
        Self { name, readout: ReadoutMap::identity(n_classes), neuron_range: 0..0 }
    }

    /// How many output neurons to allocate for this effector (the `size` passed to
    /// `NetworkBuilder::add(output_config(), ..)`).
    pub fn n_neurons(&self) -> u32 {
        self.readout.n_neurons()
    }

    /// Number of external classes this effector reads out.
    pub fn n_classes(&self) -> usize {
        self.readout.n_classes()
    }

    /// Attach the global neuron-index range returned by `Network::population_range` after the
    /// network is built, resolving the local readout CSR into concrete global indices.
    pub fn bind(mut self, range: Range<u32>) -> Self {
        assert_eq!(
            range.end - range.start,
            self.n_neurons(),
            "Effector '{}' expects {} output neurons but was bound to a range of {}",
            self.name,
            self.n_neurons(),
            range.end - range.start,
        );
        self.neuron_range = range;
        self
    }

    /// Per-class total AP count over the trial: for each class, sum `spike_counts` across its
    /// member output neurons. `spike_counts` is the full per-neuron accumulator from
    /// `run_event_loop` (length = n_neurons); only this effector's window is read.
    pub fn class_activity(&self, spike_counts: &[u32]) -> Vec<u32> {
        let base = self.neuron_range.start as usize;
        (0..self.n_classes())
            .map(|c| {
                self.readout
                    .members(c)
                    .iter()
                    .map(|&local| spike_counts[base + local as usize])
                    .sum()
            })
            .collect()
    }

    /// The predicted class: argmax over `class_activity`, ties broken to the lowest class index.
    /// Returns `None` when the output layer was silent (all-zero activity) — so "no prediction"
    /// is explicit rather than a spurious class 0.
    pub fn predict(&self, spike_counts: &[u32]) -> Option<u32> {
        let activity = self.class_activity(spike_counts);
        let best = activity.iter().copied().max()?; // None only if there are no classes
        if best == 0 {
            return None; // output layer was silent — no prediction
        }
        // position() returns the FIRST occurrence → ties break to the lowest class index.
        Some(activity.iter().position(|&a| a == best).unwrap() as u32)
    }

    /// Inject a supervised teaching signal for the **correct** class (§8.5 Option 1, direct
    /// injection). The efferent mirror of [`InputSpace::encode`](crate::io::input::InputSpace::encode):
    /// where `encode` *asserts* `SOMATIC_SPIKE`s on input neurons, `teach` pushes a depolarizing
    /// `SOMA_SIGNAL` — a "teacher current" — on the labelled class's output neuron(s), forcing them
    /// to burst.
    ///
    /// Why this drives learning: a burst raises the neuron's `beta`, and `handle_somatic_spike`'s
    /// back-propagating sweep applies LTP (`w += (beta − H_BETA)·alpha/lr`) across that neuron's
    /// afferent synapses — i.e. the hidden→output connections whose `alpha` eligibility is high
    /// because their hidden neuron just fired for *this* frame. Teaching the right answer therefore
    /// strengthens exactly the features that predicted it. Routing through `SOMA_SIGNAL` rather than
    /// a raw `SOMATIC_SPIKE` is deliberate: only `update_soma_potential` raises `beta`, and `beta`
    /// is what gates the weight update.
    ///
    /// Call once per tick during *training* trials, right after `encode`, with the trial's true
    /// label. `strength` is the per-event voltage delta — set it at or above the output
    /// `soma_threshold` so the neuron reliably bursts (larger ⇒ bigger burst ⇒ faster `beta` climb).
    /// Timestamps jitter across `[base_ts, base_ts + window)` to align with the input volley.
    ///
    /// **Caveat:** the teacher's own spikes land in `spike_counts`, so [`predict`](Self::predict)
    /// over a *taught* trial is teacher-contaminated and not a measure of learning. Read accuracy
    /// from a separate, un-taught evaluation pass.
    ///
    /// Panics if `label` is not a valid class index, or if called before [`bind`](Self::bind).
    pub fn teach(
        &self,
        label: u32,
        base_ts: u16,
        window: u16,
        strength: i16,
        producer: &EventProducer,
        rng: &mut impl RngExt,
    ) {
        let base = self.neuron_range.start;
        for &local in self.readout.members(label as usize) {
            let ts = jitter(base_ts, window, rng);
            producer.push(Event::soma_signal(base + local, ts, strength));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_readout_is_one_to_one() {
        let m = ReadoutMap::identity(10);
        assert_eq!(m.n_neurons(), 10);
        assert_eq!(m.n_classes(), 10);
        for c in 0..10 {
            assert_eq!(m.members(c), &[c as u32]);
        }
    }

    #[test]
    fn n_neurons_matches_classes_for_identity() {
        let eff = Effector::identity("digits", 10);
        assert_eq!(eff.n_neurons(), 10);
        assert_eq!(eff.n_classes(), 10);
    }

    #[test]
    #[should_panic]
    fn bind_rejects_wrong_sized_range() {
        Effector::identity("digits", 10).bind(0..3); // 10 neurons, range of 3
    }

    #[test]
    fn predict_argmaxes_over_bound_window() {
        // 3 classes bound to global neurons 5..8. Counts elsewhere must be ignored.
        let eff = Effector::identity("t", 3).bind(5..8);
        //                 0  1  2  3  4  5  6  7
        let spike_counts = [9, 9, 9, 9, 9, 2, 7, 1]; // window = indices 5,6,7 → class 0,1,2
        assert_eq!(eff.class_activity(&spike_counts), vec![2, 7, 1]);
        assert_eq!(eff.predict(&spike_counts), Some(1));
    }

    #[test]
    fn predict_none_when_silent() {
        let eff = Effector::identity("t", 3).bind(0..3);
        assert_eq!(eff.predict(&[0, 0, 0]), None);
    }

    #[test]
    fn predict_ties_break_to_lowest_class() {
        let eff = Effector::identity("t", 3).bind(0..3);
        assert_eq!(eff.predict(&[5, 5, 1]), Some(0)); // classes 0 and 1 tie → 0
    }

    #[test]
    fn population_coded_readout_sums_member_neurons() {
        // 2 classes, 2 output neurons each (local indices 0,1 → class 0; 2,3 → class 1),
        // bound to global 10..14. Built directly since population-coded maps have no public
        // constructor yet (identity is all MNIST needs).
        let readout = ReadoutMap {
            offsets: vec![0, 2, 4],
            neurons: vec![0, 1, 2, 3],
            n_neurons: 4,
        };
        let eff = Effector { name: "pop", readout, neuron_range: 0..0 }.bind(10..14);
        //                  0  1  2  3  4  5  6  7  8  9 10 11 12 13
        let spike_counts = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 2, 4, 4];
        // class 0 = neurons 10+11 = 1+2 = 3; class 1 = neurons 12+13 = 4+4 = 8
        assert_eq!(eff.class_activity(&spike_counts), vec![3, 8]);
        assert_eq!(eff.predict(&spike_counts), Some(1));
    }
}
