//! Tauri backend for the neural research dashboard.
//!
//! Two pillars, both served by narrow Rust commands so the trust boundary is explicit:
//!   - **Docs** (`docs/`, `notes/`, `science/`): an editable markdown/LaTeX file tree. File IO is
//!     done here (not via the fs plugin) so the frontend can only touch allowed types under those
//!     roots: `.md`/`.tex` are read/written as text, `.pdf` is read as raw bytes for view-only display.
//!   - **Simulation replay**: `list_recordings`/`load_recording` read the `.ntr` pairs `neural-cli`
//!     writes and aggregate them in-process into a compact digest for the viewer. Decoding the
//!     half-million-event body and reducing it to per-tick / per-layer summaries happens here so the
//!     webview never has to hold the raw trace.
//!
//! The backend links `neural-sim`/`neural-telemetry` directly (see `sim_constants`, `load_recording`)
//! — the in-process seam the architecture plan is built around.

use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

pub mod docs;
pub mod network;
pub mod recording;

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
            docs::list_docs,
            docs::read_doc,
            docs::read_doc_bytes,
            docs::save_doc,
            recording::list_recordings,
            recording::load_recording,
            network::load_network,
            sim_constants
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
