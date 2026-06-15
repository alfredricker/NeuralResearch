// ---------------------------------------------------------------------------------------------
// Network replay (graph view) — topology + per-tick state for the interactive stepper.
// ---------------------------------------------------------------------------------------------

use std::collections::BTreeMap;

use serde::Serialize;

use crate::recording::resolve_recording;

/// Above this many edges the graph is unreadable and the payload large, so `load_network` sends a
/// deterministic stride-sample instead (and flags `edges_truncated`). Blobs (~200 edges) is well
/// under; MNIST (hundreds of thousands) is sampled down to this.
const MAX_GRAPH_EDGES: usize = 2000;

/// One neuron node. Layout (x/y) is the frontend's job; the backend only classifies the layer so
/// the view can column/grid neurons by role. `layer`: 0 = input, 1 = hidden, 2 = output.
#[derive(Serialize)]
pub(crate) struct NeuronMeta {
    index: u32,
    layer: u8,
}

/// One synaptic edge with its weight at the trial's start and end keyframes, so the view can color
/// by current weight or by the learning delta (`w_post - w_pre`).
#[derive(Serialize)]
pub(crate) struct EdgeMeta {
    src: u32,
    dst: u32,
    synapse: u32,
    w_pre: i8,
    w_post: i8,
}

/// State at one tick (one wavefront): every neuron's soma potential, and how many somatic spikes it
/// emitted *this* tick (the per-tick delta of the cumulative spike counter). Only present when the
/// recording carries per-tick snapshots (small nets — see `--task blobs`).
#[derive(Serialize)]
pub(crate) struct TickFrame {
    tick: u32,
    potentials: Vec<i8>,
    spikes: Vec<u32>,
}

/// The full network-replay payload: fixed topology + a per-tick timeline to scrub. Built entirely
/// in Rust; the frontend draws nodes/edges and animates `ticks` under the step control.
#[derive(Serialize)]
pub(crate) struct NetworkReplay {
    label: String,
    dims: BTreeMap<String, u64>,
    true_label: Option<u32>,
    prediction: Option<u32>,
    correct: Option<bool>,
    n_input: u32,
    n_hidden: u32,
    n_output: u32,
    neurons: Vec<NeuronMeta>,
    edges: Vec<EdgeMeta>,
    /// Total live edges before any sampling (so the view can report "showing N of M").
    edge_total: u32,
    edges_truncated: bool,
    /// Per-tick timeline; empty when the recording has only pre/post keyframes (large nets).
    ticks: Vec<TickFrame>,
    has_per_tick: bool,
}

/// Load one recording and project it into the graph-view payload: topology (edges with pre/post
/// weights), per-neuron layer, and the per-tick state timeline when present.
#[tauri::command]
pub(crate) fn load_network(stem: String) -> Result<NetworkReplay, String> {
    let abs = resolve_recording(&stem)?;
    let (manifest, recording) =
        neural_telemetry::load(&abs).map_err(|e| format!("loading {stem}: {e}"))?;

    let total = manifest.dims.get("neurons").copied().unwrap_or(0) as u32;
    let n_input = manifest.dims.get("input_neurons").copied().unwrap_or(0) as u32;
    let n_hidden = manifest.dims.get("hidden_neurons").copied().unwrap_or(0) as u32;
    let n_output = manifest.dims.get("output_neurons").copied().unwrap_or(0) as u32;
    let hid_end = n_input + n_hidden; // contiguous input → hidden → output layout
    let out_start = total.saturating_sub(n_output);

    let layer_of = |i: u32| -> u8 {
        if i < n_input {
            0
        } else if i >= out_start {
            2
        } else if i < hid_end {
            1
        } else {
            1 // any gap falls back to hidden; layout is contiguous so this is unreached in practice
        }
    };
    let neurons: Vec<NeuronMeta> =
        (0..total).map(|index| NeuronMeta { index, layer: layer_of(index) }).collect();

    // Pre/post weights bracket the trial's plasticity; index by the edge's synapse slot.
    let pre = recording.snapshots.first();
    let post = recording.snapshots.last();
    let weight_at = |snap: Option<&neural_telemetry::Snapshot>, syn: u32| -> i8 {
        snap.and_then(|s| s.synapse_weights.get(syn as usize)).copied().unwrap_or(0)
    };

    let edge_total = recording.edges.len();
    // sample down to MAX_GRAPH_EDGES with a uniform stride if needed (deterministic, keeps a spread
    // across the source-neuron order rather than just a prefix).
    let stride = (edge_total / MAX_GRAPH_EDGES).max(1);
    let edges: Vec<EdgeMeta> = recording
        .edges
        .iter()
        .step_by(stride)
        .map(|&(src, dst, synapse)| EdgeMeta {
            src,
            dst,
            synapse,
            w_pre: weight_at(pre, synapse),
            w_post: weight_at(post, synapse),
        })
        .collect();

    // Per-tick timeline: snapshots are [pre, tick0, …, tickN-1, post]; the interior frames are the
    // per-tick keyframes. Each frame's spikes = its cumulative counts minus the previous snapshot's.
    let snaps = &recording.snapshots;
    let mut ticks = Vec::new();
    if snaps.len() > 2 {
        let t0 = snaps[0].timestamp;
        for idx in 1..snaps.len() - 1 {
            let cur = &snaps[idx];
            let prev = &snaps[idx - 1];
            let spikes: Vec<u32> = cur
                .spike_counts
                .iter()
                .zip(prev.spike_counts.iter())
                .map(|(&c, &p)| c.saturating_sub(p))
                .collect();
            ticks.push(TickFrame {
                tick: cur.timestamp.saturating_sub(t0) as u32,
                potentials: cur.soma_potentials.clone(),
                spikes,
            });
        }
    }
    let has_per_tick = !ticks.is_empty();

    let correct = manifest.true_label.zip(manifest.prediction).map(|(t, p)| t == p);
    Ok(NetworkReplay {
        label: manifest.label,
        dims: manifest.dims,
        true_label: manifest.true_label,
        prediction: manifest.prediction,
        correct,
        n_input,
        n_hidden,
        n_output,
        neurons,
        edges,
        edge_total: edge_total as u32,
        edges_truncated: stride > 1,
        ticks,
        has_per_tick,
    })
}