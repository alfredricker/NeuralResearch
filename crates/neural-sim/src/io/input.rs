//! Afferent half of the IO boundary: input spaces and the sensory arrow.
//!
//! An [`InputSpace`] is one external modality. It owns:
//!   1. a [`Shape`] — the external coordinate system (e.g. a 28x28 pixel grid),
//!   2. a [`SensoryMap`] — *the arrow*: a CSR mapping each coordinate to the input neuron(s)
//!      it drives (identity for MNIST; later: ON/OFF channels, receptive-field pooling, ...),
//!   3. a `neuron_range` — the global indices of those input neurons, attached via [`bind`]
//!      once the network is built.
//!
//! Presenting a frame is purely host-side: [`InputSpace::encode`] walks the frame, follows the
//! sensory arrow, and pushes events into the queue. The network's axon CSR + handlers do the
//! rest — `encode` itself never touches network state.
//!
//! [`bind`]: InputSpace::bind

use std::ops::Range;
use std::sync::LazyLock;

use rand::RngExt;

use crate::constants::MSLR;
use crate::math::sample::{SamplerI8, SamplerU8};
use crate::network::event::{Event, EventProducer, SOMATIC_SPIKE};
use crate::neuron::config::NeuronConfig;

/// Largest burst (AP count) an input neuron asserts for a full-intensity element. The burst rides
/// the event payload and scales the downstream EPSP, so element intensity maps to drive strength.
const MAX_INPUT_BURST: u16 = 4;

// ============================================================================================
// Input-neuron taxonomy
// ============================================================================================

/// The `NeuronConfig` for input neurons.
///
/// Input neurons are **pure axon sources with ZERO dendrites** (`n_basal_dendrites = 0`,
/// `n_apical_dendrites = None`). They never integrate input of their own — we *assert* that they
/// fired by pushing a `SOMATIC_SPIKE` directly (see [`InputSpace::encode`]). Because they own no
/// dendrites, `dendrite_offsets[n] == dendrite_offsets[n + 1]`, so their afferent synapse range is
/// empty: `handle_somatic_spike`'s back-propagating weight sweep iterates nothing and the call
/// simply fans the AP out across the axon CSR. **Triggering a somatic spike on an input neuron is
/// therefore always safe — it cannot panic or error** — even though the neuron has no dendrites,
/// synapses, or integration state. The soma fields below exist only so the neuron occupies an
/// index in the SoA arrays; their values are never read by the dynamics.
static INPUT_CONFIG: LazyLock<NeuronConfig> = LazyLock::new(|| {
    NeuronConfig::new(
        "input",
        0,                      // n_basal_dendrites — NONE: input neurons have no dendrites
        None,                   // n_apical_dendrites — NONE
        SamplerU8::new(128, 0), // synapse_x_sampler        (unused: no synapses)
        SamplerU8::new(1, 0),   // dendrites_per_branch     (unused: no dendrites)
        SamplerU8::new(0, 0),   // synapses_per_dendrite    (unused: no synapses)
        1,                      // soma_threshold           (unused: never integrates)
        0,                      // basal_dendrite_threshold (unused: no dendrites)
        SamplerI8::new(0, 0),   // basal_dendrite_constant  (unused: no dendrites)
        None,                   // apical_dendrite_threshold
        None,                   // apical_dendrite_constant
        MSLR as i16,            // learning_rate            (unused: no own synapses)
    )
});

/// The shared `&'static NeuronConfig` for input neurons; pass to `NetworkBuilder::add`.
pub fn input_config() -> &'static NeuronConfig {
    &INPUT_CONFIG
}

// ============================================================================================
// Shape — the external coordinate system
// ============================================================================================

/// The coordinate system of an input space.
pub enum Shape {
    /// A flat 1-D space of `n` elements.
    Flat(u32),
    /// A 2-D grid (row-major, `h * w` elements). MNIST is `Grid2D { h: 28, w: 28 }`.
    Grid2D { h: u32, w: u32 },
}

impl Shape {
    /// Number of elements (tensor cells) in the space.
    pub fn n_elements(&self) -> u32 {
        match self {
            Shape::Flat(n) => *n,
            Shape::Grid2D { h, w } => h * w,
        }
    }
}

// ============================================================================================
// SensoryMap — the arrow (map A): coordinate -> input neuron(s)
// ============================================================================================

/// The explicit map from input-space coordinates to input neurons, in CSR form. Neuron indices
/// are **local** (`0..n_neurons`); [`InputSpace::bind`] offsets them into global index space.
///
/// Same offset-array idiom as `dendrite_offsets` / `axon_offsets`: `offsets` has length
/// `n_elements + 1`, and `neurons[offsets[e]..offsets[e + 1]]` are the neurons element `e` drives.
pub struct SensoryMap {
    offsets: Vec<u32>, // len = n_elements + 1
    neurons: Vec<u32>, // flattened local input-neuron indices
    n_neurons: u32,
}

impl SensoryMap {
    /// One input neuron per element, in order: element `i` drives neuron `i`. The map MNIST uses.
    pub fn identity(n: u32) -> Self {
        Self {
            offsets: (0..=n).collect(),
            neurons: (0..n).collect(),
            n_neurons: n,
        }
    }

    /// Number of distinct input neurons this map transduces to.
    pub fn n_neurons(&self) -> u32 {
        self.n_neurons
    }

    /// Number of source coordinates (rows in the CSR).
    pub fn n_elements(&self) -> usize {
        self.offsets.len() - 1
    }

    /// Local input-neuron indices driven by element `e`.
    pub fn targets(&self, e: usize) -> &[u32] {
        let lo = self.offsets[e] as usize;
        let hi = self.offsets[e + 1] as usize;
        &self.neurons[lo..hi]
    }
}

// ============================================================================================
// InputSpace
// ============================================================================================

/// One external input modality: a coordinate space, its sensory arrow, and (after `bind`) the
/// global indices of the input neurons it feeds.
pub struct InputSpace {
    pub name: &'static str,
    pub shape: Shape,
    sensory: SensoryMap,
    /// Global neuron-index range of this space's input neurons. Empty (`0..0`) until `bind`.
    neuron_range: Range<u32>,
}

impl InputSpace {
    /// An input space whose sensory arrow is the identity (element `i` -> input neuron `i`).
    pub fn identity(name: &'static str, shape: Shape) -> Self {
        let sensory = SensoryMap::identity(shape.n_elements());
        Self { name, shape, sensory, neuron_range: 0..0 }
    }

    /// How many input neurons to allocate for this space (the `size` passed to
    /// `NetworkBuilder::add(input_config(), ..)`).
    pub fn n_neurons(&self) -> u32 {
        self.sensory.n_neurons()
    }

    /// Attach the global neuron-index range returned by `Network::population_range` after the
    /// network is built, resolving the local sensory CSR into concrete global indices.
    pub fn bind(mut self, range: Range<u32>) -> Self {
        assert_eq!(
            range.end - range.start,
            self.n_neurons(),
            "InputSpace '{}' expects {} input neurons but was bound to a range of {}",
            self.name,
            self.n_neurons(),
            range.end - range.start,
        );
        self.neuron_range = range;
        self
    }

    /// Transduce one frame into spike events and push them into the queue.
    ///
    /// For each non-zero element, the sensory arrow selects its input neuron(s) and each is made
    /// to fire by pushing a `SOMATIC_SPIKE` at its global index, carrying a burst (in the payload)
    /// scaled by element intensity. `handle_somatic_spike` then fans this AP out across the axon
    /// CSR into the next layer. **Input neurons have no dendrites, so asserting a somatic spike on
    /// them is always safe (the back-prop weight sweep runs over an empty synapse range) — this
    /// call can never error.** Spike timestamps are jittered uniformly across `[base_ts, base_ts +
    /// window)` so a frame presents as a small stochastic volley rather than one synchronous edge.
    ///
    /// Panics if `frame.len()` does not match the space's element count, or if called before
    /// [`bind`](Self::bind).
    pub fn encode(
        &self,
        frame: &[u8],
        base_ts: u16,
        window: u16,
        producer: &EventProducer,
        rng: &mut impl RngExt,
    ) {
        assert_eq!(
            frame.len(),
            self.sensory.n_elements(),
            "frame length {} != input space '{}' element count {}",
            frame.len(),
            self.name,
            self.sensory.n_elements(),
        );
        debug_assert_eq!(
            self.neuron_range.end - self.neuron_range.start,
            self.n_neurons(),
            "InputSpace '{}' encoded before bind()",
            self.name,
        );

        let base = self.neuron_range.start;
        for (e, &intensity) in frame.iter().enumerate() {
            if intensity == 0 {
                continue; // dark element: no drive, no event
            }
            let burst = intensity_to_burst(intensity);
            for &local in self.sensory.targets(e) {
                let ts = jitter(base_ts, window, rng);
                producer.push(Event::with_payload(SOMATIC_SPIKE, base + local, ts, burst));
            }
        }
    }
}

/// Map element intensity (1..=255) to a burst count in 1..=`MAX_INPUT_BURST`.
fn intensity_to_burst(intensity: u8) -> i16 {
    (1 + intensity as u16 * (MAX_INPUT_BURST - 1) / 255) as i16
}

/// A timestamp uniformly within `[base_ts, base_ts + window)`, wrapping like the rest of the
/// timeline. `window <= 1` degenerates to `base_ts` (avoids an empty sampling range).
fn jitter(base_ts: u16, window: u16, rng: &mut impl RngExt) -> u16 {
    if window <= 1 {
        base_ts
    } else {
        base_ts.wrapping_add(rng.random_range(0..window))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::network::event::{EventQueue, SOMATIC_SPIKE};
    use rand::rngs::SmallRng;
    use rand::SeedableRng;

    #[test]
    fn shape_element_counts() {
        assert_eq!(Shape::Flat(784).n_elements(), 784);
        assert_eq!(Shape::Grid2D { h: 28, w: 28 }.n_elements(), 784);
    }

    #[test]
    fn identity_map_is_one_to_one() {
        let m = SensoryMap::identity(4);
        assert_eq!(m.n_neurons(), 4);
        assert_eq!(m.n_elements(), 4);
        for e in 0..4 {
            assert_eq!(m.targets(e), &[e as u32]);
        }
    }

    #[test]
    fn n_neurons_matches_shape() {
        let space = InputSpace::identity("t", Shape::Grid2D { h: 28, w: 28 });
        assert_eq!(space.n_neurons(), 784);
    }

    #[test]
    #[should_panic]
    fn bind_rejects_wrong_sized_range() {
        InputSpace::identity("t", Shape::Flat(4)).bind(10..12); // 4 neurons, range of 2
    }

    #[test]
    fn encode_fires_active_pixels_at_bound_global_indices() {
        // 4-pixel space bound to global neurons 10..14; pixels 1 and 3 are lit.
        let space = InputSpace::identity("t", Shape::Flat(4)).bind(10..14);
        let frame = [0u8, 255, 0, 128];

        let queue = EventQueue::new(64);
        let producer = queue.producer_handle();
        let mut rng = SmallRng::seed_from_u64(7);
        let (base_ts, window) = (100u16, 50u16);

        space.encode(&frame, base_ts, window, &producer, &mut rng);

        let events = queue.drain();
        // one event per lit pixel, identity map → one neuron each
        assert_eq!(events.len(), 2);

        // emitted in element order: pixel 1 → global 11, pixel 3 → global 13
        assert_eq!(events[0].event_type, SOMATIC_SPIKE);
        assert_eq!(events[0].source, 11);
        assert_eq!(events[1].source, 13);

        for e in events {
            assert_eq!(e.event_type, SOMATIC_SPIKE);
            assert!(e.payload >= 1, "burst must be at least 1 AP");
            // timestamp jittered within the trial window
            assert!(e.timestamp >= base_ts && e.timestamp < base_ts + window);
        }

        // full-intensity pixel (255) bursts harder than the mid-intensity one (128)
        assert_eq!(events[0].payload, intensity_to_burst(255));
        assert_eq!(events[1].payload, intensity_to_burst(128));
    }

    #[test]
    fn dark_frame_emits_nothing() {
        let space = InputSpace::identity("t", Shape::Flat(4)).bind(0..4);
        let queue = EventQueue::new(8);
        let producer = queue.producer_handle();
        let mut rng = SmallRng::seed_from_u64(1);

        space.encode(&[0, 0, 0, 0], 0, 10, &producer, &mut rng);

        assert_eq!(queue.drain().len(), 0);
    }

    #[test]
    fn intensity_to_burst_spans_full_range() {
        assert_eq!(intensity_to_burst(0), 1); // (only reached if a 0 element were encoded)
        assert_eq!(intensity_to_burst(1), 1);
        assert_eq!(intensity_to_burst(255), MAX_INPUT_BURST as i16);
    }
}
