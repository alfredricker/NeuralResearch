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

// ---- Network graph replay (the stepper) ------------------------------------------------------

export interface NeuronMeta {
  index: number;
  /** 0 = input, 1 = hidden, 2 = output. */
  layer: number;
}

export interface EdgeMeta {
  src: number;
  dst: number;
  synapse: number;
  /** Synapse weight at the trial's pre/post keyframes (i8) — color by w_post or the delta. */
  w_pre: number;
  w_post: number;
}

export interface TickFrame {
  /** Clock value relative to trial start. */
  tick: number;
  /** Per-neuron soma potential at this wavefront (i8). */
  potentials: number[];
  /** Per-neuron somatic spikes emitted *this* wavefront. */
  spikes: number[];
}

export interface NetworkReplay {
  label: string;
  dims: Record<string, number>;
  true_label: number | null;
  prediction: number | null;
  correct: boolean | null;
  n_input: number;
  n_hidden: number;
  n_output: number;
  neurons: NeuronMeta[];
  edges: EdgeMeta[];
  /** Live edge count before sampling (edges may be a stride-sample for large nets). */
  edge_total: number;
  edges_truncated: boolean;
  /** Per-tick timeline; empty for large nets (only pre/post keyframes recorded). */
  ticks: TickFrame[];
  has_per_tick: boolean;
}

/** Load one recording's topology + per-tick state for the network graph view. */
export const loadNetwork = (stem: string) =>
  invoke<NetworkReplay>("load_network", { stem });

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
