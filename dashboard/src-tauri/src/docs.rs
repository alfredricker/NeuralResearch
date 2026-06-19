// ---------------------------------------------------------------------------------------------
// Docs pillar
// ---------------------------------------------------------------------------------------------

use std::path::{Component, Path, PathBuf};
use std::process::Command;

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

// ---------------------------------------------------------------------------------------------
// LaTeX → PDF rendering
// ---------------------------------------------------------------------------------------------

/// Known engines in preference order. Tectonic first: it's a single self-contained binary that
/// fetches packages on demand (so it works on a fresh machine) and leaves no `.aux`/`.log` clutter.
/// The rest are TeX Live tools we'll drive if that's what's installed.
const TEX_ENGINES: [&str; 5] = ["tectonic", "latexmk", "pdflatex", "xelatex", "lualatex"];

/// Dirs probed beyond `$PATH` — a `tauri dev` / GUI launch often inherits a thin PATH that misses
/// the usual TeX install locations on macOS (MacTeX, Homebrew) and Linux.
const TEX_SEARCH_DIRS: [&str; 4] =
    ["/Library/TeX/texbin", "/opt/homebrew/bin", "/usr/local/bin", "/usr/bin"];

/// Aux files non-tectonic engines drop next to the source — removed after a successful compile so the
/// doc tree stays just `.tex` + `.pdf`.
const TEX_AUX_EXTS: [&str; 7] = ["aux", "log", "out", "toc", "fls", "fdb_latexmk", "synctex.gz"];

/// Locate the first available LaTeX engine, returning its name and absolute path.
fn find_engine() -> Option<(&'static str, PathBuf)> {
    let mut dirs: Vec<PathBuf> = std::env::var_os("PATH")
        .map(|p| std::env::split_paths(&p).collect())
        .unwrap_or_default();
    dirs.extend(TEX_SEARCH_DIRS.iter().map(PathBuf::from));
    for engine in TEX_ENGINES {
        for dir in &dirs {
            let cand = dir.join(engine);
            if cand.is_file() {
                return Some((engine, cand));
            }
        }
    }
    None
}

/// Compile a `.tex` doc to a PDF written alongside it (same stem, `.pdf` extension), and return the
/// repo-relative path of the produced PDF (which `read_doc_bytes` then serves to the viewer).
///
/// The engine runs with its working dir set to the source folder so relative `\input`/graphics
/// resolve, and emits the PDF into that same folder. Raw `pdflatex`/`xelatex`/`lualatex` are run
/// twice so cross-references / the TOC settle; `tectonic` and `latexmk` handle multi-pass themselves.
#[tauri::command]
pub(crate) fn render_tex(path: String) -> Result<String, String> {
    let abs = resolve_doc(&path, &["tex"])?;
    let dir = abs.parent().ok_or("tex file has no parent directory")?;
    let stem = abs.file_stem().and_then(|s| s.to_str()).ok_or("bad tex filename")?.to_string();
    let file_name = abs.file_name().and_then(|s| s.to_str()).ok_or("bad tex filename")?.to_string();

    let (engine, bin) = find_engine().ok_or_else(|| {
        "no LaTeX engine found — install one (e.g. `brew install tectonic`) and retry".to_string()
    })?;

    // Engines that don't self-iterate need a second pass to settle refs/TOC.
    let passes = if matches!(engine, "pdflatex" | "xelatex" | "lualatex") { 2 } else { 1 };
    let mut last_output = None;
    for _ in 0..passes {
        let mut cmd = Command::new(&bin);
        cmd.current_dir(dir);
        match engine {
            "tectonic" => {
                cmd.args(["--outdir", "."]).arg(&file_name);
            }
            "latexmk" => {
                cmd.args(["-pdf", "-interaction=nonstopmode", "-halt-on-error", "-outdir=."])
                    .arg(&file_name);
            }
            _ => {
                cmd.args(["-interaction=nonstopmode", "-halt-on-error", "-output-directory=."])
                    .arg(&file_name);
            }
        }
        last_output = Some(cmd.output().map_err(|e| format!("running {engine}: {e}"))?);
    }

    // Clean aux droppings (tectonic leaves none, but the TeX Live engines do).
    if engine != "tectonic" {
        for ext in TEX_AUX_EXTS {
            let _ = std::fs::remove_file(dir.join(format!("{stem}.{ext}")));
        }
    }

    let pdf = dir.join(format!("{stem}.pdf"));
    if !pdf.is_file() {
        let log = last_output
            .map(|o| {
                format!("{}{}", String::from_utf8_lossy(&o.stdout), String::from_utf8_lossy(&o.stderr))
            })
            .unwrap_or_default();
        let lines: Vec<&str> = log.lines().collect();
        let tail = lines[lines.len().saturating_sub(30)..].join("\n");
        return Err(format!("{engine} did not produce a PDF:\n{tail}"));
    }

    Ok(format!("{}.pdf", path.strip_suffix(".tex").unwrap_or(&path)))
}