// Typed wrappers around the Tauri backend commands (see src-tauri/src/lib.rs).
import { invoke } from "@tauri-apps/api/core";

export interface DocEntry {
  /** Repo-relative path, e.g. "docs/01-theory.md" — the handle read/save take. */
  path: string;
  /** File name for display, e.g. "01-theory.md". */
  name: string;
  /** Root it lives under: "docs" or "notes". */
  dir: string;
}

/** Every `.md` file under docs/ and notes/, sorted within each root. */
export const listDocs = () => invoke<DocEntry[]>("list_docs");

/** Read a doc's markdown source. */
export const readDoc = (path: string) => invoke<string>("read_doc", { path });

/** Write a doc's markdown source back to disk. */
export const saveDoc = (path: string, content: string) =>
  invoke<void>("save_doc", { path, content });

/** The sim's compiled-in constants — proof the neural-sim crate links in-process. */
export const simConstants = () => invoke<Record<string, number>>("sim_constants");
