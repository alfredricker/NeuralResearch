//! Telemetry tap — the one outward seam of the otherwise serde-free, GPU-shaped core.
//!
//! `run_event_loop` takes a `&mut impl TelemetrySink`. The production / GPU path passes
//! [`NullSink`], whose methods are all no-ops and monomorphize away to nothing — the hot
//! path pays zero cost. The dashboard path passes a `RecordingSink` (in `neural-telemetry`,
//! where serde is allowed) that writes the `.ntr` recording.
//!
//! Two observation channels:
//!   - [`TelemetrySink::on_event`] — fine-grained event trace (every drained [`Event`]),
//!     for spike rasters and step animation.
//!   - [`TelemetrySink::on_snapshot`] — periodic columnar state dump. A [`NetworkView`] is a
//!     struct of *borrowed* SoA slices, so the sink reads the network's arrays without
//!     owning or copying anything; it decides what (if anything) to persist.

use crate::network::event::event::Event;

/// Borrowed, read-only view of the network's SoA state at one instant. Every field is a
/// slice into the live `Vec`s the simulation already owns — constructing one is free, and a
/// sink that ignores it costs nothing. Fields mirror the arrays the dashboard views read
/// (see plans/dashboard-architecture.md); extend as more views come online.
pub struct NetworkView<'a> {
    /// Simulation timestamp this snapshot was taken at (the `timestamp` of the event that
    /// triggered it, or a harness-chosen mark).
    pub timestamp: u16,

    // soma
    pub soma_potentials: &'a [i8],
    pub soma_betas: &'a [u8],
    // dendrite
    pub dendrite_activities: &'a [u16],
    pub dendrite_is_apical: &'a [u8],
    // synapse
    pub synapse_weights: &'a [i8],
    pub synapse_alphas: &'a [u8],
    // readout — per-neuron accumulated AP count since the last trial reset
    pub spike_counts: &'a [u32],
}

/// Observer threaded through `run_event_loop`. Both methods take `&mut self` so a sink can
/// accumulate into its own buffers. The default `NullSink` impl is the zero-cost identity.
pub trait TelemetrySink {
    /// Called once per drained event, in fire order. Keep it cheap — it runs in the loop.
    fn on_event(&mut self, event: &Event);

    /// Called when the harness requests a state keyframe. Borrows the SoA arrays; the sink
    /// copies out only what it persists.
    fn on_snapshot(&mut self, view: &NetworkView);
}

/// Zero-cost default sink for production and GPU builds. Every method is a no-op and inlines
/// away under monomorphization, so `run_event_loop(&mut NullSink, ..)` compiles to the same
/// code as an untapped loop.
pub struct NullSink;

impl TelemetrySink for NullSink {
    #[inline(always)]
    fn on_event(&mut self, _event: &Event) {}
    #[inline(always)]
    fn on_snapshot(&mut self, _view: &NetworkView) {}
}
