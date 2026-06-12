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

/** Every `.md` file under the doc roots (recursively), sorted within each root. */
export const listDocs = () => invoke<DocEntry[]>("list_docs");

/** Read a doc's markdown source. */
export const readDoc = (path: string) => invoke<string>("read_doc", { path });

/** Write a doc's markdown source back to disk (creates parent dirs as needed). */
export const saveDoc = (path: string, content: string) =>
  invoke<void>("save_doc", { path, content });

// ---- Simulation replay pillar ----------------------------------------------------------------

export interface RecordingSummary {
  /** Repo-relative extension-less stem, e.g. "recordings/trial-00007" — the load handle. */
  stem: string;
  label: string;
  true_label: number | null;
  prediction: number | null;
  correct: boolean | null;
}

export interface TickSpikes {
  tick: number;
  spikes: number;
}

export interface RecordingDetail {
  label: string;
  dims: Record<string, number>;
  constants: Record<string, number>;
  true_label: number | null;
  prediction: number | null;
  correct: boolean | null;
  /** Per-output-neuron spike counts (the digit read-out). */
  output_spikes: number[];
  input_total: number;
  hidden_total: number;
  output_total: number;
  hidden_active: number;
  hidden_count: number;
  event_total: number;
  somatic_total: number;
  spikes_over_time: TickSpikes[];
}

/** All recordings in recordings/, manifest headlines only. */
export const listRecordings = () => invoke<RecordingSummary[]>("list_recordings");

/** Load + aggregate one recording into the viewer digest. */
export const loadRecording = (stem: string) =>
  invoke<RecordingDetail>("load_recording", { stem });

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
