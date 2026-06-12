//! Tauri backend for the neural research dashboard.
//!
//! Phase 2 (docs pillar) surfaces the repo's `docs/` and `notes/` markdown as an editable file
//! tree. File IO is done here in Rust (rather than via the fs plugin) so the trust boundary is
//! explicit and narrow — the frontend can only read/write `.md` files under those two roots.
//!
//! The backend also links `neural-sim` directly (see `sim_constants`); the replay / run-trial
//! commands of later phases build on that in-process seam.

use std::path::{Component, Path, PathBuf};

use serde::Serialize;

/// The document roots surfaced and edited in-app, relative to the repo root.
const DOC_DIRS: [&str; 2] = ["docs", "notes"];

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

/// One markdown file in the doc tree.
#[derive(Serialize)]
struct DocEntry {
    /// Path relative to the repo root (e.g. `docs/01-theory.md`) — the handle `read_doc`/`save_doc`
    /// accept. Always forward-slashed and rooted at a `DOC_DIRS` entry.
    path: String,
    /// File name for display (e.g. `01-theory.md`).
    name: String,
    /// Which root it lives under: `docs` or `notes`.
    dir: String,
}

/// List every `.md` file under `docs/` and `notes/`, sorted within each root.
#[tauri::command]
fn list_docs() -> Result<Vec<DocEntry>, String> {
    let root = repo_root();
    let mut out = Vec::new();
    for dir in DOC_DIRS {
        let read = match std::fs::read_dir(root.join(dir)) {
            Ok(r) => r,
            Err(_) => continue, // a root may not exist — just skip it
        };
        let mut names: Vec<String> = read
            .filter_map(Result::ok)
            .filter_map(|e| {
                let p = e.path();
                (p.extension().and_then(|x| x.to_str()) == Some("md"))
                    .then(|| p.file_name().and_then(|n| n.to_str()).map(String::from))
                    .flatten()
            })
            .collect();
        names.sort();
        out.extend(names.into_iter().map(|name| DocEntry {
            path: format!("{dir}/{name}"),
            name,
            dir: dir.to_string(),
        }));
    }
    Ok(out)
}

/// Resolve a repo-relative doc path to an absolute path, rejecting anything that escapes the doc
/// roots or isn't markdown. This is the trust boundary: no absolute paths, no `..` traversal, must
/// sit under `docs/` or `notes/`, must end in `.md`.
fn resolve_doc(rel: &str) -> Result<PathBuf, String> {
    let rel_path = Path::new(rel);
    if rel_path.is_absolute()
        || rel_path.components().any(|c| matches!(c, Component::ParentDir))
    {
        return Err(format!("illegal path: {rel}"));
    }
    if !DOC_DIRS.iter().any(|d| rel.starts_with(&format!("{d}/"))) {
        return Err(format!("path must be under docs/ or notes/: {rel}"));
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

/// Write a markdown doc back to disk (git still owns it — this just edits the file).
#[tauri::command]
fn save_doc(path: String, content: String) -> Result<(), String> {
    let abs = resolve_doc(&path)?;
    std::fs::write(&abs, content).map_err(|e| format!("writing {}: {e}", abs.display()))
}

/// Proof that `neural-sim` links and is callable in-process with no serialization boundary: returns
/// the constants the sim was compiled with. The Phase-3 run-trial / load-recording commands build
/// on this same direct link.
#[tauri::command]
fn sim_constants() -> std::collections::BTreeMap<String, i64> {
    use neural_sim::constants as c;
    std::collections::BTreeMap::from([
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
            sim_constants
        ])
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
