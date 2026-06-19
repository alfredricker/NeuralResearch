// Typed wrappers around the Tauri backend commands (see src-tauri/src/lib.rs).
import { invoke } from "@tauri-apps/api/core";

// ---- Docs pillar ------------------------------------------------------------------------------

export interface DocEntry {
  /** Repo-relative path, e.g. "docs/01-theory.md" — the handle read/save take. */
  path: string;
  /** Display name relative to its root, e.g. "01-theory.md" or "runs/trial-00007.md". */
  name: string;
  /** Root it lives under: "docs", "notes", or "science". */
  dir: string;
}

/** How a doc is handled: markdown/tex are editable text, pdf is view-only binary. */
export type DocKind = "md" | "tex" | "pdf";

/** Classify a doc path by extension (defaults to markdown for anything unrecognized). */
export function docKind(path: string): DocKind {
  const ext = path.slice(path.lastIndexOf(".") + 1).toLowerCase();
  return ext === "tex" || ext === "pdf" ? ext : "md";
}

/** Every viewable file (`.md`/`.tex`/`.pdf`) under the doc roots, sorted within each root. */
export const listDocs = () => invoke<DocEntry[]>("list_docs");

/** Read an editable doc's text source (`.md`/`.tex`). */
export const readDoc = (path: string) => invoke<string>("read_doc", { path });

/** Read a doc's raw bytes (used for `.pdf`); returns an ArrayBuffer. */
export const readDocBytes = (path: string) =>
  invoke<ArrayBuffer>("read_doc_bytes", { path });

/** Write an editable doc's text back to disk (`.md`/`.tex`; creates parent dirs as needed). */
export const saveDoc = (path: string, content: string) =>
  invoke<void>("save_doc", { path, content });

/** Compile a `.tex` doc to a PDF alongside it; resolves to the produced PDF's repo-relative path. */
export const renderTex = (path: string) => invoke<string>("render_tex", { path });

// ---- Run notes (reuse the docs trust boundary) -----------------------------------------------

/** Repo-relative path of a run's markdown note, under notes/runs/. */
export const runNotePath = (label: string) => `notes/runs/${label}.md`;

/** Read a run note, returning "" if it doesn't exist yet. */
export async function readRunNote(label: string): Promise<string> {
  try {
    return await readDoc(runNotePath(label));
  } catch {
    return "";
  }
}

/** Write a run note (creates notes/runs/ on first save). */
export const saveRunNote = (label: string, content: string) =>
  saveDoc(runNotePath(label), content);

// ---- Misc ------------------------------------------------------------------------------------

/** The sim's compiled-in constants — proof the neural-sim crate links in-process. */
export const simConstants = () => invoke<Record<string, number>>("sim_constants");
