//! `.ntr` recording: the replay data plane for the dashboard.
//!
//! [`RecordingSink`] implements [`neural_sim::telemetry::TelemetrySink`]. Because `neural-sim`
//! is deliberately serde-free, the sink's job is to *copy out* the borrowed `Event` / SoA
//! slices into owned, `serde`-derivable records ([`EventRecord`], [`Snapshot`]) that this
//! crate then serializes.
//!
//! Format (first cut — see plans/dashboard-architecture.md for the target):
//!   - **Manifest** (`<name>.ntr.json`): human-readable, git-diffable — topology dims, the
//!     `constants.rs` values used, trial label, and keyframe index.
//!   - **Body** (`<name>.ntr`): postcard-encoded [`Recording`] — keyframe [`Snapshot`]s plus
//!     the inter-keyframe [`EventRecord`] trace.
//!
//! PROVISIONAL: the body is currently one postcard blob, not yet seekable. Lazy per-keyframe
//! fetch (load manifest → seek to nearest keyframe → replay forward) is a later refinement;
//! the manifest already carries the keyframe offsets that change will need.

use std::path::Path;

use neural_sim::network::event::event::Event;
use neural_sim::telemetry::{NetworkView, TelemetrySink};
use serde::{Deserialize, Serialize};

/// Owned, serializable mirror of a `neural_sim` `Event` (which itself carries no serde impl).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EventRecord {
    pub event_type: u8,
    pub source: u32,
    pub timestamp: u16,
    pub payload: i16,
}

impl From<&Event> for EventRecord {
    fn from(e: &Event) -> Self {
        Self { event_type: e.event_type, source: e.source, timestamp: e.timestamp, payload: e.payload }
    }
}

/// Owned columnar copy of the SoA state at one keyframe. A snapshot *is* the arrays (clone of
/// each [`NetworkView`] slice). `event_index` records how far into the event trace this
/// keyframe sits, so replay can locate "state at time T" = nearest keyframe + forward events.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    pub timestamp: u16,
    pub event_index: u32,
    pub soma_potentials: Vec<i8>,
    pub soma_betas: Vec<u8>,
    pub dendrite_activities: Vec<u16>,
    pub dendrite_is_apical: Vec<u8>,
    pub synapse_weights: Vec<i8>,
    pub synapse_alphas: Vec<u8>,
    pub spike_counts: Vec<u32>,
}

impl Snapshot {
    fn from_view(view: &NetworkView, event_index: u32) -> Self {
        Self {
            timestamp: view.timestamp,
            event_index,
            soma_potentials: view.soma_potentials.to_vec(),
            soma_betas: view.soma_betas.to_vec(),
            dendrite_activities: view.dendrite_activities.to_vec(),
            dendrite_is_apical: view.dendrite_is_apical.to_vec(),
            synapse_weights: view.synapse_weights.to_vec(),
            synapse_alphas: view.synapse_alphas.to_vec(),
            spike_counts: view.spike_counts.to_vec(),
        }
    }
}

/// Human-readable run metadata, written alongside the body as `<name>.ntr.json`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Manifest {
    /// Trial / run label.
    pub label: String,
    /// Free-form topology dimensions (e.g. {"neurons": 800, "dendrites": 3200}). Kept as a
    /// map so it survives topology changes without a format bump.
    pub dims: std::collections::BTreeMap<String, u64>,
    /// The `constants.rs` values this run used, as name → value, for reproducibility/diffing.
    pub constants: std::collections::BTreeMap<String, i64>,
    /// `event_index` of each keyframe, in order — the seek index for lazy replay.
    pub keyframe_offsets: Vec<u32>,
}

/// The serialized body: keyframes + the event trace between them.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct Recording {
    pub events: Vec<EventRecord>,
    pub snapshots: Vec<Snapshot>,
}

#[derive(Debug, thiserror::Error)]
pub enum RecordingError {
    #[error("io error writing recording: {0}")]
    Io(#[from] std::io::Error),
    #[error("postcard encode error: {0}")]
    Postcard(#[from] postcard::Error),
    #[error("json encode error: {0}")]
    Json(#[from] serde_json::Error),
}

/// A [`TelemetrySink`] that buffers the event trace and state keyframes in memory, then writes
/// the `.ntr` pair on [`RecordingSink::write`]. Buffering (rather than streaming) keeps the
/// first cut simple; a streaming writer can replace it behind the same trait later.
pub struct RecordingSink {
    manifest: Manifest,
    recording: Recording,
}

impl RecordingSink {
    pub fn new(manifest: Manifest) -> Self {
        Self { manifest, recording: Recording::default() }
    }

    /// Write `<stem>.ntr` (postcard body) and `<stem>.ntr.json` (manifest). `stem` is a path
    /// without extension, e.g. `recordings/trial-0042`.
    pub fn write(&self, stem: impl AsRef<Path>) -> Result<(), RecordingError> {
        let stem = stem.as_ref();
        let body = postcard::to_allocvec(&self.recording)?;
        std::fs::write(stem.with_extension("ntr"), body)?;
        let json = serde_json::to_vec_pretty(&self.manifest)?;
        std::fs::write(stem.with_extension("ntr.json"), json)?;
        Ok(())
    }

    /// Borrow the buffered recording (e.g. for tests or in-process replay without a round-trip).
    pub fn recording(&self) -> &Recording {
        &self.recording
    }
}

impl TelemetrySink for RecordingSink {
    fn on_event(&mut self, event: &Event) {
        self.recording.events.push(EventRecord::from(event));
    }

    fn on_snapshot(&mut self, view: &NetworkView) {
        let event_index = self.recording.events.len() as u32;
        self.manifest.keyframe_offsets.push(event_index);
        self.recording.snapshots.push(Snapshot::from_view(view, event_index));
    }
}
