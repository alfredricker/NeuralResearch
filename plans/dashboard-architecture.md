# Research Dashboard — Architecture & Build Plan

Status: **Phase 1 partially done** (2026-06-10): workspace split + telemetry tap landed.
Drafted 2026-06-09.

## Progress log

- **2026-06-10 — workspace + telemetry tap.** Single crate → workspace. Package renamed
  `research` → `neural-sim` (now lib-only; hello-world `main.rs` dropped). Added
  `neural-sim/src/telemetry.rs`: `TelemetrySink` trait, zero-cost `NullSink`, borrowed
  `NetworkView`. `run_event_loop` now takes `&mut impl TelemetrySink` and calls `on_event`
  per drained event (no caller existed, so no call sites broke). New `neural-telemetry`
  crate holds all serde: `EventRecord`/`Snapshot`/`Manifest`/`Recording` + `RecordingSink`
  writing the `.ntr` (postcard body) + `.ntr.json` (manifest) pair. New `neural-cli` is a
  stub binary (links the crates; trial harness not built yet). Workspace compiles.
  **Still TODO in Phase 1:** the trial harness (wire `Network` + `InputSpace` + `Effector`
  + the loop), then real recording generation in `neural-cli`, then `on_snapshot` wiring
  (no clock/periodic trigger yet — keyframes are caller-driven for now).
- **Pre-existing test breakage (not from this work):** 6 of 90 `neural-sim` tests fail in
  `neuron::synapse` (alpha decay) and `neuron::dendrite` (voltage leak). Confirmed via
  `git diff -M` that those files were pure renames — red before the refactor. Fix separately.

Goal: a personal research dashboard for the spiking-net project that (1) visualizes
simulations, (2) reads/edits the markdown + LaTeX docs and notes in WYSIWYG, and
(3) gives tools for actively guiding research — with **math, code, and concept central**.

## Decisions locked in

- **Stack: Tauri** (Rust backend + webview frontend). Chosen not just "because Rust"
  but because the backend can **link `neural-sim` directly and read its flat SoA `Vec`s
  with zero serialization** — state extraction is trivial. The web frontend wins the
  three pillars decisively: KaTeX (math), Monaco/CodeMirror (code), D3/deck.gl (viz).
- **Viz mode: replay from recordings.** The sim writes a telemetry file per trial; the
  dashboard loads and scrubs it. No live data plane to engineer (no binary IPC channels,
  no downsampling under pressure). Deterministic, great for analysis. Live streaming can
  be added later on the same `TelemetrySink` abstraction if wanted.
- **Reach: personal local desktop.** Pure Tauri, link the sim crate directly. Max
  performance, zero serialization for compute.
- **Docs editing: full WYSIWYG md+LaTeX** (Milkdown/Tiptap), writing back to the same
  `.md` files so git still owns them.
- **Frontend framework: TBD** — leaning **Svelte** (lighter, less ceremony for a solo
  tool, reactivity suits live-updating plots). React if we expect heavy component libs.

## Why the data architecture is the real point

The simulator's entire state lives in flat, indexable `Vec`s (`soma_potentials`,
`synapse_weights`, `dendrite_activities`, `axon_targets`, …). There's no serialization
layer yet and we don't need one to *read* state. So unlike most "dashboard for a
simulator" projects, there is **no hard language/process boundary to cross for compute** —
the Tauri Rust backend calls the sim in-process. We only serialize at the edge, for the
recording files and the handful of things the UI paints.

Because replay was chosen, the control/data-plane split from the original sketch collapses
to: **control plane** = low-frequency Tauri commands (load recording, run trial, load/save
doc, set constant); **data plane** = `.ntr` files on disk, lazy-loaded by the frontend.

## Workspace layout

Current single crate becomes a workspace. Critical: keep **serde/serialization out of
`neural-sim`** so the eventual GPU port stays clean — recording lives in a separate crate.

```
neural/research/
├── Cargo.toml                  # [workspace]
├── crates/
│   ├── neural-sim/             # ← current src/, same philosophy
│   │   └── src/                #   SoA core, run_event_loop, NO serde
│   │       └── telemetry.rs    #   + TelemetrySink trait, NullSink (zero-cost)
│   ├── neural-telemetry/       # RecordingSink + .ntr format (serde lives HERE)
│   └── neural-cli/             # headless: run trials → write recordings/*.ntr
├── dashboard/
│   ├── src-tauri/              # Tauri backend: links neural-sim + neural-telemetry
│   │   └── src/lib.rs          #   commands: load_recording, list_docs, save_doc, run_trial
│   ├── src/                    # TS frontend (Svelte or React)
│   └── package.json
├── docs/                       # existing — surfaced & editable in-app
├── notes/                      # existing — surfaced & editable in-app
├── plans/                      # this doc
└── recordings/                 # generated .ntr files (gitignored)
```

## Telemetry tap (sim-side, the foundational change)

No telemetry exists today, and the sim is deliberately GPU-shaped — must not pollute the
hot path. Solution: a feature/generic-gated observer, monomorphized so the null path
compiles to nothing.

```rust
// crates/neural-sim/src/telemetry.rs
pub trait TelemetrySink {
    fn on_event(&mut self, e: &Event);             // optional event trace
    fn on_snapshot(&mut self, net: &NetworkView);  // periodic state dump (borrowed slices)
}

pub struct NullSink;                 // zero-cost default — production / GPU builds
impl TelemetrySink for NullSink {    // all no-ops, inlined away
    fn on_event(&mut self, _: &Event) {}
    fn on_snapshot(&mut self, _: &NetworkView) {}
}
```

`run_event_loop` takes `&mut impl TelemetrySink`. `NullSink` → nothing. The dashboard
passes a `RecordingSink` (in `neural-telemetry`, where serde is allowed) that fills the
`.ntr` file. `NetworkView` is a struct of borrowed slices so the sink reads SoA arrays
without owning anything.

## Recording format `.ntr` (keyframes + deltas, like video)

- **Manifest** (JSON, human-readable, git-diffable): topology dims, the `constants.rs`
  values used for the run, trial label, keyframe offsets.
- **Keyframe snapshots** at fixed event/timestamp intervals: columnar binary dump of the
  SoA `Vec`s (a snapshot *is* the arrays). Use **postcard** — compact, no_std-spirited,
  fits the GPU ethos.
- **Event trace** between keyframes for fine-grained raster/animation.
- Frontend loads manifest, lazy-fetches keyframe blobs while scrubbing. "State at time T"
  = nearest keyframe + replay forward a few events.

## Views, mapped to existing data structures

| View | Reads | Notes |
|---|---|---|
| Spike raster | `spike_counts[]` / event trace | needs accumulation gap closed (see below) |
| Weight matrix / distribution | `synapse_weights[]`, `synapse_alphas[]` | per-dendrite, live-count windowed |
| Dendrite voltage traces | `dendrite_activities[]`, `dendrite_is_apical[]` | basal vs apical split |
| Connectivity graph | `axon_targets[]`/`axon_offsets[]` (CSR) | deck.gl / sigma.js force graph |
| MNIST metrics | `effector.predict()`, `class_activity()` | accuracy, confusion matrix |
| Constant tuner | `constants.rs` → re-run | the "guiding research" loop |

## Docs / concept pillar

Point the app at `docs/` and `notes/`. **Milkdown** (or Tiptap) for WYSIWYG markdown with
embedded KaTeX, saving back to the same `.md` files. Render `docs/*.md` LaTeX-heavy chapters
with KaTeX; standalone `.tex` notes get a KaTeX/MathJax preview pane. The payoff is linking
concept ↔ code ↔ viz: a doc chapter can deep-link to a handler or embed a live weight plot.

## Build order (each phase independently useful)

1. **Workspace refactor + telemetry tap.** Split crates; add `TelemetrySink`/`NullSink`;
   write `RecordingSink` + `.ntr` format; build `neural-cli` to generate recordings
   headlessly. This phase *forces* closing the three loop gaps below — replay needs real
   data, so dashboard work and MNIST-loop work converge here.
2. **Tauri shell + docs pillar.** App boots; file tree over `docs/`+`notes/`; Milkdown
   WYSIWYG with inline KaTeX; saves back to `.md`. Independent of the sim — immediate
   value, exercises the Tauri command plumbing.
3. **Replay viewer.** Load `.ntr`; timeline scrubber; the two highest-value views: spike
   raster + weight matrix/distribution.
4. **Connectivity graph + dendrite voltage traces** (deck.gl/sigma for axon CSR;
   basal-vs-apical for dendrites).
5. **The research loop.** Constant tuner → re-run via `neural-cli` → load two recordings →
   diff side by side. A/B a hyperparameter change and *see* the effect on rasters/accuracy.

## Coupled prerequisite: loop gaps (from docs/09-gaps) — RECHECKED 2026-06-10

Status corrected against current code; this section was drafted before the sim evolved:
- ~~`run_event_loop` does not accumulate `spike_counts`~~ — **CLOSED.** It now increments
  `spike_counts[n] += e.payload.max(0)` on `SOMATIC_SPIKE` (`loop.rs`). (Also: `FORWARD_AP`
  no longer exists; the event types are now `SOMATIC_SPIKE`/`DENDRITIC_SPIKE`/`SOMA_SIGNAL`/
  `SYNAPSE_SIGNAL`.)
- Ring buffer **never advances `head`** — `EventQueue::drain` reads `head..tail` but nothing
  advances `head`; slots not recycled, multi-trial loop overruns. **STILL OPEN** — fix when
  building the multi-trial harness.
- **No clock advance** — only `InputSpace::encode` writes fresh timestamps; cascades run at
  frozen time, no decay between wavefronts. **STILL OPEN** — also gates periodic `on_snapshot`
  (no natural keyframe trigger without a clock).

## Alternative considered: egui (rejected)

Pure-Rust immediate-mode GUI, zero serialization boundary, `egui_plot` excellent for
real-time traces. Rejected because it undercuts two of the three pillars: no Monaco-class
code editor, painful LaTeX/markdown rendering, clumsy rich-text editing. Would only win if
the dashboard were viz-only.

## Open question for next session

- Frontend framework: **Svelte (leaning) vs React** — decide before scaffolding Phase 2.

## Next action

Start Phase 1: scaffold the workspace split + `TelemetrySink`/`NullSink`, stub the
telemetry crate and CLI, keep `neural-sim` compiling.
