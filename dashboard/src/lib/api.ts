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

// ---- Playground (live stepper) ---------------------------------------------------------------
//
// Build a live network from a spec, stimulate synapses, and step it one wavefront at a time.
// Anatomy is the fixed structure (fetched once at build); a frame is the dynamic state after a
// step, with frame arrays parallel to the anatomy arrays so the view zips them by position.

/** A discretized-normal sampler, captured by its (mean, std) — mirrors Rust `SamplerSpec`. */
export interface SamplerSpec {
  mean: number;
  std: number;
}

/** A custom neuron type (mirrors Rust `NeuronTypeSpec`). */
export interface NeuronTypeSpec {
  n_basal_dendrites: number;
  n_apical_dendrites: number | null;
  synapse_x_sampler: SamplerSpec;
  dendrites_per_branch: SamplerSpec;
  synapses_per_dendrite: SamplerSpec;
  soma_threshold: number;
  basal_dendrite_threshold: number;
  basal_dendrite_constant: SamplerSpec;
  apical_dendrite_threshold: number | null;
  apical_dendrite_constant: SamplerSpec | null;
  learning_rate: number;
}

/** Which neuron type a population is: a builtin or a `{ Custom: "name" }` ref into neuron_types. */
export type NeuronTypeRef = "Input" | "Output" | { Custom: string };

export interface PopulationSpec {
  neuron_type: NeuronTypeRef;
  size: number;
  label?: string | null;
}

export type CompartmentSpec = "Basal" | "Apical";

/** Connection rule — one variant, e.g. `{ FixedInDegree: { k: 8 } }` or `"OneToOne"`. */
export type ConnRuleSpec =
  | { DenseRandom: { p: number } }
  | { FixedInDegree: { k: number } }
  | { ReceptiveField: { radius: number } }
  | { Topographic: { patch: number } }
  | "OneToOne";

export interface ConnectionSpec {
  from: number;
  to: number;
  compartment: CompartmentSpec;
  rule: ConnRuleSpec;
}

/** The reproducible network recipe (mirrors Rust `NetworkSpec`). */
export interface NetworkSpec {
  seed: number;
  neuron_types: Record<string, NeuronTypeSpec>;
  populations: PopulationSpec[];
  connections: ConnectionSpec[];
}

// --- anatomy (static structure) ---

export interface SynapseAnatomy {
  synapse: number;
  /** Position along the dendrite, 0..=255 — the layout coordinate. */
  x: number;
  /** Presynaptic neuron, or null for an unbound (directly-stimulated) slot. */
  src_neuron: number | null;
}

export interface DendriteAnatomy {
  dendrite: number;
  is_apical: boolean;
  /** Sign is proximal (>0) vs distal (<=0). */
  branch_constant: number;
  threshold: number;
  synapses: SynapseAnatomy[];
}

export interface NeuronAnatomy {
  neuron: number;
  soma_threshold: number;
  dendrites: DendriteAnatomy[];
}

// --- frame (dynamic state; arrays parallel to the anatomy) ---

export interface SynapseState {
  alpha: number;
  weight: number;
  signaled: boolean;
}

export interface DendriteState {
  v_b: number;
  fired: boolean;
  synapses: SynapseState[];
}

export interface NeuronFrame {
  neuron: number;
  soma_potential: number;
  soma_beta: number;
  soma_burst: number;
  dendrites: DendriteState[];
}

export interface NetworkFrame {
  clock: number;
  neurons: NeuronFrame[];
}

/** Build a live network from a spec; returns its static anatomy. */
export const pgBuild = (spec: NetworkSpec) =>
  invoke<NeuronAnatomy[]>("pg_build", { spec });

/** Inject one AP delivery onto a synapse slot (drained by the next step). */
export const pgStimulate = (synapse: number, burst: number) =>
  invoke<void>("pg_stimulate", { synapse, burst });

/** Advance one wavefront; returns the resulting state with this step's firings flagged. */
export const pgStep = () => invoke<NetworkFrame>("pg_step");

/** Current state without stepping (initial render). */
export const pgState = () => invoke<NetworkFrame>("pg_state");

/** Clear transient dynamics back to rest, keeping learned weights. */
export const pgReset = () => invoke<void>("pg_reset");

// ---- Misc ------------------------------------------------------------------------------------

/** The sim's compiled-in constants — proof the neural-sim crate links in-process. */
export const simConstants = () => invoke<Record<string, number>>("sim_constants");
