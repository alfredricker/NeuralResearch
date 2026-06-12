//! The trial harness: drive the event loop forward over a fixed window, presenting one input
//! frame and reading out a prediction.
//!
//! This is the orchestration the dashboard's `neural-cli` records. It lives in `neural-sim` (which
//! is serde-free) so it is reusable and testable with [`NullSink`](crate::telemetry::NullSink) —
//! the recording machinery is layered on top via the [`TelemetrySink`] the caller passes in.
//!
//! The loop is the one specified in `docs/08-mnist-pipeline.md` §8.4: each tick re-presents the
//! frame, runs **one wavefront** ([`Network::step`]), and advances a shared monotonic clock so the
//! lazy exponential decay (alpha/beta/voltage, all keyed off timestamp deltas) sees real elapsed
//! time between wavefronts. The clock is monotonic across the whole run rather than reset per trial
//! so that persisted learning state (`alpha`/`beta`) decays correctly across trial boundaries.

use rand::RngExt;

use crate::io::input::InputSpace;
use crate::io::output::Effector;
use crate::network::Network;
use crate::network::event::queue::EventQueue;
use crate::telemetry::TelemetrySink;

/// Timing for one trial.
#[derive(Clone, Copy, Debug)]
pub struct TrialConfig {
    /// Number of wavefronts (ticks) to run. Each tick re-presents the frame and advances the clock.
    pub ticks: u16,
    /// Jitter window for the per-tick input volley (see [`InputSpace::encode`]); the frame presents
    /// as a small stochastic burst spread over `[clock, clock + window)` rather than one hard edge.
    pub window: u16,
    /// Voltage delta of the per-tick supervised teaching signal (see [`Effector::teach`]), used
    /// only on *training* trials (when `run_trial` is given a label). Set at or above the output
    /// `soma_threshold` so the taught neuron reliably bursts; ignored for un-taught trials.
    pub teach_strength: i16,
}

impl Default for TrialConfig {
    fn default() -> Self {
        Self { ticks: 100, window: 8, teach_strength: 24 }
    }
}

/// Run one trial — supervised when `teach` is `Some(label)`, a plain forward pass when `None`.
///
/// Presents `frame` each tick for `cfg.ticks` ticks, advancing the shared monotonic `clock` once
/// per wavefront. `spike_counts` (length must equal `network.n_neurons()`) is zeroed on entry and
/// accumulates somatic spikes over the trial; the effector argmaxes its output window for the
/// prediction (`None` if the output layer stayed silent).
///
/// When `teach` is `Some(label)`, the correct class's output neuron is driven to burst each tick
/// via [`Effector::teach`] (strength `cfg.teach_strength`), applying LTP to the hidden→output
/// synapses active for this frame — this is what makes trials actually learn (§8.5 Option 1).
/// **Note:** the teacher's spikes inflate `spike_counts`, so the returned prediction on a taught
/// trial is teacher-contaminated; read real accuracy from a separate `teach = None` eval pass.
///
/// The network's learning state (weights, `alpha`, `beta`) persists across trials by design — call
/// [`Network::reset_dynamics`] between trials to clear only the transient potentials.
#[allow(clippy::too_many_arguments)]
pub fn run_trial(
    network: &mut Network,
    queue: &EventQueue,
    space: &InputSpace,
    effector: &Effector,
    frame: &[u8],
    teach: Option<u32>,
    clock: &mut u16,
    cfg: TrialConfig,
    rng: &mut impl RngExt,
    sink: &mut impl TelemetrySink,
    spike_counts: &mut [u32],
) -> Option<u32> {
    spike_counts.iter_mut().for_each(|c| *c = 0);

    for _ in 0..cfg.ticks {
        // push this tick's input volley (and, when training, the teaching signal on the correct
        // output neuron), then drain the one wavefront it (and any prior cascade) produced. The
        // producer is dropped before `step` so only shared borrows of `queue` overlap.
        {
            let producer = queue.producer_handle();
            space.encode(frame, *clock, cfg.window, &producer, rng);
            if let Some(label) = teach {
                effector.teach(label, *clock, cfg.window, cfg.teach_strength, &producer, rng);
            }
        }
        network.step(queue, sink, spike_counts);
        *clock = clock.wrapping_add(1);
    }

    effector.predict(spike_counts)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::io::input::{Shape, input_config};
    use crate::io::output::output_config;
    use crate::network::build::NetworkBuilder;
    use crate::network::topology::conn::ConnRule;
    use crate::neuron::dendrite::Compartment;
    use crate::telemetry::NullSink;
    use rand::SeedableRng;
    use rand::rngs::SmallRng;

    /// The core regression for the wavefront/head-advance fix: many trials over a *reused* queue
    /// must not overrun the ring (head recycles), and the per-tick input volley must accumulate on
    /// the lit input neurons every trial. Output firing depends on untuned thresholds, so we assert
    /// only the input-side invariant — the part that is deterministic.
    #[test]
    fn multi_trial_reuses_queue_without_overrun_and_counts_input_spikes() {
        let mut b = NetworkBuilder { populations: Vec::new(), connections: Vec::new() };
        let inp = b.add(input_config(), 4);
        let out = b.add(output_config(), 4);
        b.connect(inp, out, Compartment::Basal, ConnRule::FixedInDegree { k: 1 });
        let mut net = Network::build(b);

        let in_range = net.population_range(inp);
        let space = InputSpace::identity("t", Shape::Flat(4)).bind(in_range.clone());
        let effector = Effector::identity("d", 4).bind(net.population_range(out));

        // a small ring, deliberately far smaller than the total events processed across all trials,
        // so the test only passes if slots are actually recycled.
        let queue = EventQueue::new(1 << 10);
        let mut rng = SmallRng::seed_from_u64(1);
        let mut clock = 0u16;
        let mut counts = vec![0u32; net.n_neurons()];
        let cfg = TrialConfig { ticks: 20, window: 4, teach_strength: 24 };
        let base = in_range.start as usize;

        for _ in 0..5 {
            let frame = [255u8, 0, 255, 0]; // pixels 0 and 2 lit
            run_trial(
                &mut net, &queue, &space, &effector, &frame, None, &mut clock, cfg,
                &mut rng, &mut NullSink, &mut counts,
            );
            assert!(counts[base] > 0, "lit input neuron 0 should accumulate spikes");
            assert!(counts[base + 2] > 0, "lit input neuron 2 should accumulate spikes");
            assert_eq!(counts[base + 1], 0, "dark input neuron 1 stays silent");
            net.reset_dynamics();
        }
    }

    /// The supervised path: teaching a class must drive its output neuron to fire (so `beta` climbs
    /// and the BaP sweep can run) and must strengthen the hidden→output synapses feeding it. We
    /// build input→output directly (one hidden-free hop is enough to exercise the teacher → burst →
    /// LTP chain on the output neuron's *own* afferents) and assert the taught class's afferent
    /// weights grow over an un-taught baseline.
    #[test]
    fn teaching_drives_output_firing_and_strengthens_afferents() {
        let mut b = NetworkBuilder { populations: Vec::new(), connections: Vec::new() };
        let inp = b.add(input_config(), 4);
        let out = b.add(output_config(), 4);
        b.connect(inp, out, Compartment::Basal, ConnRule::FixedInDegree { k: 4 });
        let mut net = Network::build(b);

        let space = InputSpace::identity("t", Shape::Flat(4)).bind(net.population_range(inp));
        let effector = Effector::identity("d", 4).bind(net.population_range(out));

        let queue = EventQueue::new(1 << 12);
        let mut rng = SmallRng::seed_from_u64(3);
        let mut clock = 0u16;
        let mut counts = vec![0u32; net.n_neurons()];
        let cfg = TrialConfig { ticks: 40, window: 4, teach_strength: 30 };
        let out_base = net.population_range(out).start as usize;

        // teach class 0 on a fully-lit frame for several trials.
        let frame = [255u8; 4];
        for _ in 0..6 {
            run_trial(
                &mut net, &queue, &space, &effector, &frame, Some(0), &mut clock, cfg,
                &mut rng, &mut NullSink, &mut counts,
            );
            net.reset_dynamics();
        }

        // the taught output neuron must have fired (teacher current → burst).
        assert!(counts[out_base] > 0, "taught output neuron 0 should fire under the teacher current");
    }
}
