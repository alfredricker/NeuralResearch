//! Tauri backend for the neural research dashboard.
//!
//! Two pillars, both served by narrow Rust commands so the trust boundary is explicit:
//!   - **Docs** (`docs/`, `notes/`, `science/`): an editable markdown/LaTeX file tree. File IO is
//!     done here (not via the fs plugin) so the frontend can only read/write `.md` under those roots.
//!   - **Simulation replay**: `list_recordings`/`load_recording` read the `.ntr` pairs `neural-cli`
//!     writes and aggregate them in-process into a compact digest for the viewer. Decoding the
//!     half-million-event body and reducing it to per-tick / per-layer summaries happens here so the
//!     webview never has to hold the raw trace.
//!
//! The backend links `neural-sim`/`neural-telemetry` directly (see `sim_constants`, `load_recording`)
//! — the in-process seam the architecture plan is built around.

use std::collections::BTreeMap;
use std::path::{Component, Path, PathBuf};

use neural_telemetry::Manifest;
use serde::Serialize;

/// The document roots surfaced and edited in-app, relative to the repo root. `notes/runs/` (per-run
/// notes written from simulation mode) and anything nested under these also surface, since the doc
/// listing recurses.
const DOC_DIRS: [&str; 3] = ["docs", "notes", "science"];

/// Where `neural-cli` writes the `.ntr`/`.ntr.json` pairs the viewer replays, relative to repo root.
const RECORDINGS_DIR: &str = "recordings";

/// Repo root — two levels up from this crate (`dashboard/src-tauri` → repo root). Resolved from the
/// compile-time manifest dir, which is correct under `tauri dev` / `cargo run`; falls back to the
/// current dir if that ever fails. (A bundled install would point this elsewhere; this tool is a
/// personal `tauri dev` workflow, so the manifest-dir anchor is the right call.)
fn repo_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent() // dashboard/
        .and_then(Path::parent) // repo root
        .map(Path::to_path_buf)
        .unwrap_or_else(|| std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
}

// ---------------------------------------------------------------------------------------------
// Docs pillar
// ---------------------------------------------------------------------------------------------

/// One markdown file in the doc tree.
#[derive(Serialize)]
struct DocEntry {
    /// Path relative to the repo root (e.g. `docs/01-theory.md`, `notes/runs/trial-00007.md`) — the
    /// handle `read_doc`/`save_doc` accept. Always forward-slashed and rooted at a `DOC_DIRS` entry.
    path: String,
    /// Display name relative to its root (e.g. `01-theory.md`, `runs/trial-00007.md`).
    name: String,
    /// Which root it lives under: `docs`, `notes`, or `science`.
    dir: String,
}

/// Recursively collect `.md` files under `root.join(dir)`, pushing one [`DocEntry`] each. `prefix`
/// is the repo-relative directory walked so far (forward-slashed); recursion appends sub-dir names.
fn collect_docs(base: &Path, dir: &str, prefix: &str, out: &mut Vec<DocEntry>) {
    let read = match std::fs::read_dir(base.join(prefix)) {
        Ok(r) => r,
        Err(_) => return, // a root (or sub-dir) may not exist — just skip it
    };
    let mut entries: Vec<_> = read.filter_map(Result::ok).map(|e| e.path()).collect();
    entries.sort();
    for p in entries {
        let Some(name) = p.file_name().and_then(|n| n.to_str()) else { continue };
        let rel = format!("{prefix}/{name}");
        if p.is_dir() {
            collect_docs(base, dir, &rel, out);
        } else if p.extension().and_then(|x| x.to_str()) == Some("md") {
            // `name` for display drops the root segment (e.g. `notes/runs/x.md` → `runs/x.md`).
            let display = rel.strip_prefix(&format!("{dir}/")).unwrap_or(&rel).to_string();
            out.push(DocEntry { path: rel, name: display, dir: dir.to_string() });
        }
    }
}

/// List every `.md` file under the doc roots (recursively), sorted within each root.
#[tauri::command]
fn list_docs() -> Result<Vec<DocEntry>, String> {
    let root = repo_root();
    let mut out = Vec::new();
    for dir in DOC_DIRS {
        collect_docs(&root, dir, dir, &mut out);
    }
    Ok(out)
}

/// Resolve a repo-relative doc path to an absolute path, rejecting anything that escapes the doc
/// roots or isn't markdown. This is the trust boundary: no absolute paths, no `..` traversal, must
/// sit under a `DOC_DIRS` root, must end in `.md`.
fn resolve_doc(rel: &str) -> Result<PathBuf, String> {
    let rel_path = Path::new(rel);
    if rel_path.is_absolute() || rel_path.components().any(|c| matches!(c, Component::ParentDir)) {
        return Err(format!("illegal path: {rel}"));
    }
    if !DOC_DIRS.iter().any(|d| rel.starts_with(&format!("{d}/"))) {
        return Err(format!("path must be under docs/, notes/, or science/: {rel}"));
    }
    if rel_path.extension().and_then(|x| x.to_str()) != Some("md") {
        return Err(format!("only .md files are editable: {rel}"));
    }
    Ok(repo_root().join(rel_path))
}

/// Read a markdown doc's contents.
#[tauri::command]
fn read_doc(path: String) -> Result<String, String> {
    let abs = resolve_doc(&path)?;
    std::fs::read_to_string(&abs).map_err(|e| format!("reading {}: {e}", abs.display()))
}

/// Write a markdown doc back to disk (git still owns it — this just edits the file). Parent dirs are
/// created on demand, so a brand-new `notes/runs/<label>.md` run note writes without a manual mkdir.
#[tauri::command]
fn save_doc(path: String, content: String) -> Result<(), String> {
    let abs = resolve_doc(&path)?;
    if let Some(parent) = abs.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("creating {}: {e}", parent.display()))?;
    }
    std::fs::write(&abs, content).map_err(|e| format!("writing {}: {e}", abs.display()))
}

// ---------------------------------------------------------------------------------------------
// Simulation replay pillar
// ---------------------------------------------------------------------------------------------

/// One row in the recording list: just the manifest's headline facts, read without decoding the body.
#[derive(Serialize)]
struct RecordingSummary {
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
struct TickSpikes {
    tick: u32,
    spikes: u32,
}

/// The aggregated digest of one recording the viewer renders. Everything is reduced in Rust; the raw
/// ~half-million-event trace never crosses into the webview.
#[derive(Serialize)]
struct RecordingDetail {
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
/// no absolute paths, no `..`, must sit under `recordings/`.
fn resolve_recording(stem: &str) -> Result<PathBuf, String> {
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
fn list_recordings() -> Result<Vec<RecordingSummary>, String> {
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
fn load_recording(stem: String) -> Result<RecordingDetail, String> {
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

/// Proof that `neural-sim` links and is callable in-process with no serialization boundary: returns
/// the constants the sim was compiled with.
#[tauri::command]
fn sim_constants() -> BTreeMap<String, i64> {
    use neural_sim::constants as c;
    BTreeMap::from([
        ("T_BETA".to_string(), c::T_BETA as i64),
        ("H_ALPHA".to_string(), c::H_ALPHA as i64),
        ("H_BETA".to_string(), c::H_BETA as i64),
        ("ALPHA_DECAY".to_string(), c::ALPHA_DECAY as i64),
        ("MSLR".to_string(), c::MSLR as i64),
        ("ALPHA_BOOST".to_string(), c::ALPHA_BOOST as i64),
    ])
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
    tauri::Builder::default()
        .plugin(tauri_plugin_opener::init())
        .invoke_handler(tauri::generate_handler![
            list_docs,
            read_doc,
            save_doc,
            list_recordings,
            load_recording,
            sim_constants
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
