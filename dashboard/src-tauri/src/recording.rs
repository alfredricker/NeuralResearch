// ---------------------------------------------------------------------------------------------
// Simulation replay pillar
// ---------------------------------------------------------------------------------------------

use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};

use neural_telemetry::Manifest;
use serde::Serialize;

use crate::{RECORDINGS_DIR, repo_root};

/// One row in the recording list: just the manifest's headline facts, read without decoding the body.
#[derive(Serialize)]
pub(crate) struct RecordingSummary {
    /// Repo-relative extension-less stem (e.g. `recordings/trial-00007`) — the handle `load_recording`
    /// accepts.
    stem: String,
    label: String,
    true_label: Option<u32>,
    prediction: Option<u32>,
    /// `true_label == prediction`, or `None` if either side is unknown.
    correct: Option<bool>,
}

/// Somatic spikes summed across the network in one tick (clock value), relative to trial start.
#[derive(Serialize)]
pub(crate) struct TickSpikes {
    tick: u32,
    spikes: u32,
}

/// The aggregated digest of one recording the viewer renders. Everything is reduced in Rust; the raw
/// ~half-million-event trace never crosses into the webview.
#[derive(Serialize)]
pub(crate) struct RecordingDetail {
    label: String,
    dims: BTreeMap<String, u64>,
    constants: BTreeMap<String, i64>,
    true_label: Option<u32>,
    prediction: Option<u32>,
    correct: Option<bool>,
    /// Per-output-neuron accumulated spike count from the post-trial keyframe — the digit read-out,
    /// argmax of which is the prediction.
    output_spikes: Vec<u32>,
    /// Total accumulated somatic spikes per layer (post-trial keyframe).
    input_total: u64,
    hidden_total: u64,
    output_total: u64,
    /// Hidden-layer sparsity: how many of `hidden_count` hidden neurons fired at all.
    hidden_active: u32,
    hidden_count: u32,
    /// Event-trace totals (whole trace, both keyframes inclusive).
    event_total: u64,
    somatic_total: u64,
    /// Somatic spikes binned per tick across the trial — the cascade unfolding over time.
    spikes_over_time: Vec<TickSpikes>,
}

/// Resolve a repo-relative recording stem to an absolute path, with the same trust boundary as docs:
/// no absolute paths, no `..`, must sit under `recordings/`. Shared with the network-graph command.
pub(crate) fn resolve_recording(stem: &str) -> Result<PathBuf, String> {
    let p = Path::new(stem);
    if p.is_absolute() || p.components().any(|c| matches!(c, Component::ParentDir)) {
        return Err(format!("illegal recording path: {stem}"));
    }
    if !stem.starts_with(&format!("{RECORDINGS_DIR}/")) {
        return Err(format!("recordings must be under {RECORDINGS_DIR}/: {stem}"));
    }
    Ok(repo_root().join(p))
}

/// List every recording in `recordings/`, newest filename last, reading only the JSON manifests.
#[tauri::command]
pub(crate) fn list_recordings() -> Result<Vec<RecordingSummary>, String> {
    let dir = repo_root().join(RECORDINGS_DIR);
    let read = match std::fs::read_dir(&dir) {
        Ok(r) => r,
        Err(_) => return Ok(Vec::new()), // no recordings yet — empty list, not an error
    };
    let mut stems: Vec<String> = read
        .filter_map(Result::ok)
        .filter_map(|e| {
            e.path()
                .file_name()
                .and_then(|n| n.to_str())
                .and_then(|n| n.strip_suffix(".ntr.json"))
                .map(String::from)
        })
        .collect();
    stems.sort();
    let mut out = Vec::with_capacity(stems.len());
    for base in stems {
        let json = dir.join(format!("{base}.ntr.json"));
        let m: Manifest = match std::fs::read(&json).ok().and_then(|b| serde_json::from_slice(&b).ok())
        {
            Some(m) => m,
            None => continue, // skip a malformed manifest rather than failing the whole list
        };
        let correct = m.true_label.zip(m.prediction).map(|(t, p)| t == p);
        out.push(RecordingSummary {
            stem: format!("{RECORDINGS_DIR}/{base}"),
            label: m.label,
            true_label: m.true_label,
            prediction: m.prediction,
            correct,
        });
    }
    Ok(out)
}

/// Load one recording and aggregate it into the viewer digest.
#[tauri::command]
pub(crate) fn load_recording(stem: String) -> Result<RecordingDetail, String> {
    let abs = resolve_recording(&stem)?;
    let (manifest, recording) =
        neural_telemetry::load(&abs).map_err(|e| format!("loading {stem}: {e}"))?;

    // Neuron index layout is contiguous input → hidden → output (the build order in neural-cli), so
    // the layer ranges fall straight out of the dims map.
    let total = manifest.dims.get("neurons").copied().unwrap_or(0) as usize;
    let n_input = manifest.dims.get("input_neurons").copied().unwrap_or(0) as usize;
    let n_hidden = manifest.dims.get("hidden_neurons").copied().unwrap_or(0) as usize;
    let n_output = manifest.dims.get("output_neurons").copied().unwrap_or(0) as usize;
    let out_start = total.saturating_sub(n_output);
    let hid_end = (n_input + n_hidden).min(total);

    // Per-neuron spike counts come from the last (post-trial) keyframe.
    let counts: &[u32] = recording.snapshots.last().map(|s| s.spike_counts.as_slice()).unwrap_or(&[]);
    let sum = |lo: usize, hi: usize| counts.get(lo..hi).into_iter().flatten().map(|&c| c as u64).sum();
    let input_total = sum(0, n_input.min(counts.len()));
    let hidden_total = sum(n_input.min(counts.len()), hid_end.min(counts.len()));
    let output_total = sum(out_start.min(counts.len()), counts.len());
    let output_spikes = counts.get(out_start..).map(<[u32]>::to_vec).unwrap_or_default();
    let hidden_active = counts
        .get(n_input.min(counts.len())..hid_end.min(counts.len()))
        .map(|s| s.iter().filter(|&&c| c > 0).count() as u32)
        .unwrap_or(0);

    // Somatic spikes binned per tick. Clock is monotonic across trials, so timestamps start at the
    // pre-trial keyframe's value, not zero — normalize to it so the x-axis reads 0..ticks.
    use neural_sim::network::event::event::SOMATIC_SPIKE;
    let t0 = recording.snapshots.first().map(|s| s.timestamp).unwrap_or(0);
    let mut per_tick: BTreeMap<u32, u32> = BTreeMap::new();
    let mut somatic_total: u64 = 0;
    for e in &recording.events {
        if e.event_type == SOMATIC_SPIKE {
            let burst = e.payload.max(0) as u32;
            somatic_total += burst as u64;
            let tick = e.timestamp.saturating_sub(t0) as u32;
            *per_tick.entry(tick).or_insert(0) += burst;
        }
    }
    let spikes_over_time =
        per_tick.into_iter().map(|(tick, spikes)| TickSpikes { tick, spikes }).collect();

    let correct = manifest.true_label.zip(manifest.prediction).map(|(t, p)| t == p);
    Ok(RecordingDetail {
        label: manifest.label,
        dims: manifest.dims,
        constants: manifest.constants,
        true_label: manifest.true_label,
        prediction: manifest.prediction,
        correct,
        output_spikes,
        input_total,
        hidden_total,
        output_total,
        hidden_active,
        hidden_count: n_hidden as u32,
        event_total: recording.events.len() as u64,
        somatic_total,
        spikes_over_time,
    })
}