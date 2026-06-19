// ---------------------------------------------------------------------------------------------
// Docs pillar
// ---------------------------------------------------------------------------------------------

use std::path::{Component, Path, PathBuf};

use serde::Serialize;
use tauri::ipc::Response;

use crate::{DOC_DIRS, repo_root};

/// File types surfaced in the doc tree. `.md`/`.tex` are editable text; `.pdf` is view-only
/// (read as raw bytes by `read_doc_bytes`, never written).
const VIEWABLE_EXTS: [&str; 3] = ["md", "tex", "pdf"];
/// The subset that may be read as text and written back.
const EDITABLE_EXTS: [&str; 2] = ["md", "tex"];

/// One file in the doc tree.
#[derive(Serialize)]
pub(crate) struct DocEntry {
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
        } else if p.extension().and_then(|x| x.to_str()).is_some_and(|x| VIEWABLE_EXTS.contains(&x)) {
            // `name` for display drops the root segment (e.g. `notes/runs/x.md` → `runs/x.md`).
            let display = rel.strip_prefix(&format!("{dir}/")).unwrap_or(&rel).to_string();
            out.push(DocEntry { path: rel, name: display, dir: dir.to_string() });
        }
    }
}

/// List every viewable file (`.md`/`.tex`/`.pdf`) under the doc roots (recursively), sorted within
/// each root.
#[tauri::command]
pub(crate) fn list_docs() -> Result<Vec<DocEntry>, String> {
    let root = repo_root();
    let mut out = Vec::new();
    for dir in DOC_DIRS {
        collect_docs(&root, dir, dir, &mut out);
    }
    Ok(out)
}

/// Resolve a repo-relative doc path to an absolute path, rejecting anything that escapes the doc
/// roots or isn't an allowed type. This is the trust boundary: no absolute paths, no `..` traversal,
/// must sit under a `DOC_DIRS` root, and its extension must be in `allowed`.
fn resolve_doc(rel: &str, allowed: &[&str]) -> Result<PathBuf, String> {
    let rel_path = Path::new(rel);
    if rel_path.is_absolute() || rel_path.components().any(|c| matches!(c, Component::ParentDir)) {
        return Err(format!("illegal path: {rel}"));
    }
    if !DOC_DIRS.iter().any(|d| rel.starts_with(&format!("{d}/"))) {
        return Err(format!("path must be under docs/, notes/, or science/: {rel}"));
    }
    if !rel_path.extension().and_then(|x| x.to_str()).is_some_and(|x| allowed.contains(&x)) {
        return Err(format!("unsupported file type ({allowed:?} only): {rel}"));
    }
    Ok(repo_root().join(rel_path))
}

/// Read an editable doc's text contents (`.md`/`.tex`).
#[tauri::command]
pub(crate) fn read_doc(path: String) -> Result<String, String> {
    let abs = resolve_doc(&path, &EDITABLE_EXTS)?;
    std::fs::read_to_string(&abs).map_err(|e| format!("reading {}: {e}", abs.display()))
}

/// Read a viewable doc's raw bytes (used for `.pdf`, which is binary and view-only). Returned as a
/// raw IPC [`Response`] so the webview gets an `ArrayBuffer` rather than a JSON number array.
#[tauri::command]
pub(crate) fn read_doc_bytes(path: String) -> Result<Response, String> {
    let abs = resolve_doc(&path, &VIEWABLE_EXTS)?;
    std::fs::read(&abs)
        .map(Response::new)
        .map_err(|e| format!("reading {}: {e}", abs.display()))
}

/// Write an editable doc (`.md`/`.tex`) back to disk (git still owns it — this just edits the file).
/// Parent dirs are created on demand, so a brand-new `notes/runs/<label>.md` run note writes without
/// a manual mkdir.
#[tauri::command]
pub(crate) fn save_doc(path: String, content: String) -> Result<(), String> {
    let abs = resolve_doc(&path, &EDITABLE_EXTS)?;
    if let Some(parent) = abs.parent() {
        std::fs::create_dir_all(parent).map_err(|e| format!("creating {}: {e}", parent.display()))?;
    }
    std::fs::write(&abs, content).map_err(|e| format!("writing {}: {e}", abs.display()))
}