//! Where-gated MNIST classifier with three-factor Hebbian learning.
//!
//! ## Architecture
//! ```text
//! "in0".."in7" (98px each) ──> Patch_k: FeedForward(98→32) → WhereModule(32→10)
//!                                                 │ "gate" (10-dim)
//!                                                 ▼
//!                                          ClassifyModule  ──> "pred" (10), "enzyme" (1)
//! ```
//!
//! ## Learning signal
//! After each forward pass the training loop computes a margin-based enzyme:
//!
//! ```text
//! enzyme = pred[true_label] − max(pred[other_classes])
//! ```
//!
//! Positive when the correct class leads (small positive reinforcement),
//! negative when a wrong class leads (weakens the current WTA winner).
//! This is injected via `learn(ext)` which routes it directly into each
//! WhereModule's `"enzyme"` input buffer before the weight update runs.
//!
//! No label is ever needed inside the graph — ClassifyModule is parameter-free.

use std::io::Write as _;
use std::path::Path;
use neural_core::burn::{Model, MnistDataset, run_pipeline};
use neural_core::modules::{ClassifyModule, FeedForward, WhereModule};
use neural_core::subgraph::{Aggregation, FlatGraph, NetworkBuilder, PortSpec, PortValues};

// ─── Constants ────────────────────────────────────────────────────────────────

const N_PATCHES: usize = 8;
const PATCH_SIZE: usize = 784 / N_PATCHES; // 98
const HIDDEN: usize = 32;
const CLASSES: usize = 10;
const K_WINNERS: usize = 2; // WTA sparsity in WhereModule

/// Static port names for per-patch pixel inputs ("in0" … "in7").
const PATCH_IN_PORTS: [&str; N_PATCHES] =
    ["in0", "in1", "in2", "in3", "in4", "in5", "in6", "in7"];

// ─── Graph construction ────────────────────────────────────────────────────────

fn build_patch(k: usize) -> FlatGraph {
    let mut ff = FeedForward::new_random(PATCH_SIZE, HIDDEN, 0.1);
    ff.lambda = 0.0; // each image processed independently — no temporal leak

    let wm = WhereModule::new(
        HIDDEN,    // n_in
        CLASSES,   // n_classes
        K_WINNERS,
        0.02,      // eta  — Hebbian learning rate
        1e-4,      // mu   — weight decay
        0.5,       // theta_w — smooth-gate threshold
    );

    NetworkBuilder::new(&format!("patch_{k}"))
        .add_node("ff", ff)
        .add_node("where", wm)
        .wire("ff", "out", "where", "in")
        .expose_input(PATCH_IN_PORTS[k], "ff", "in")
        .expose_input("enzyme", "where", "enzyme")
        .expose_output("gate", "where", "gate")
        .build()
        .expect("patch subgraph build failed")
}

fn build_graph() -> FlatGraph {
    let patch_graphs: Vec<FlatGraph> = (0..N_PATCHES).map(build_patch).collect();

    let mut outer = NetworkBuilder::new("mnist");
    for (k, pg) in patch_graphs.into_iter().enumerate() {
        outer = outer.add_node(&format!("patch_{k}"), pg);
    }
    outer = outer.add_node("classify", ClassifyModule::new(CLASSES));

    // Expose patch pixel ports so the outer graph can be reused as a Node.
    for k in 0..N_PATCHES {
        outer = outer.expose_input(PATCH_IN_PORTS[k], &format!("patch_{k}"), PATCH_IN_PORTS[k]);
    }

    // Wire each patch's gate votes → ClassifyModule (Sum: all patches contribute).
    // No recurrent enzyme wire — enzyme is injected externally via learn().
    // Build directly from the final WireBuilder since we have no more chaining to do.
    let mut wb = outer.wire("patch_0", "gate", "classify", "votes");
    for k in 1..N_PATCHES {
        wb = wb.wire(&format!("patch_{k}"), "gate", "classify", "votes");
    }
    wb.build().expect("outer graph build failed")
}

// ─── Model wrapper ─────────────────────────────────────────────────────────────

struct SubgraphModel {
    graph: FlatGraph,
    /// Reusable external buffer: pixel patches + enzyme slot.
    ext: PortValues,
}

impl SubgraphModel {
    fn new() -> Self {
        let graph = build_graph();

        // Pre-allocate a single external PortValues with pixel patches + enzyme.
        let mut specs: Vec<PortSpec> = (0..N_PATCHES)
            .map(|k| PortSpec { name: PATCH_IN_PORTS[k], dim: PATCH_SIZE, agg: Aggregation::Concat })
            .collect();
        specs.push(PortSpec { name: "enzyme", dim: 1, agg: Aggregation::Concat });

        let ext = PortValues::zeros_from(&specs);
        Self { graph, ext }
    }

    fn load_pixels(&mut self, pixels: &[f32]) {
        for k in 0..N_PATCHES {
            self.ext
                .get_mut(PATCH_IN_PORTS[k])
                .unwrap()
                .copy_from_slice(&pixels[k * PATCH_SIZE..(k + 1) * PATCH_SIZE]);
        }
    }

    fn predict(&mut self, pixels: &[f32]) -> usize {
        self.ext.get_mut("enzyme").unwrap()[0] = 0.0; // no learning signal during eval
        self.load_pixels(pixels);
        self.graph.tick(&self.ext);
        argmax(self.graph.read_output("classify", "pred").unwrap())
    }
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// ─── Pipeline integration ──────────────────────────────────────────────────────

impl Model for SubgraphModel {
    fn train_epoch(&mut self, samples: &[(Vec<f32>, usize)]) -> f32 {
        let mut total_loss = 0.0f32;
        let mut correct = 0usize;
        const PRINT_EVERY: usize = 1000;

        for (n, (pixels, label)) in samples.iter().enumerate() {
            // ── Forward pass ──────────────────────────────────────────────────
            // enzyme=0 during tick: does not affect forward computation
            // (WhereModule.tick() ignores the enzyme port).
            self.ext.get_mut("enzyme").unwrap()[0] = 0.0;
            self.load_pixels(pixels);
            self.graph.tick(&self.ext);

            // ── Supervised enzyme ─────────────────────────────────────────────
            // Margin signal: positive → correct class leads → reinforce.
            //                negative → wrong class leads  → weaken winner.
            let pred = self.graph.read_output("classify", "pred").unwrap().to_vec();
            let pred_correct = pred[*label];
            let max_wrong = pred.iter().enumerate()
                .filter(|&(i, _)| i != *label)
                .map(|(_, &p)| p)
                .fold(f32::NEG_INFINITY, f32::max);
            let enzyme = pred_correct - max_wrong; // ∈ [-1, 1]

            // ── Learning pass ─────────────────────────────────────────────────
            // FlatGraph::learn() now routes external "enzyme" into every
            // WhereModule's enzyme input buffer before calling node.learn().
            self.ext.get_mut("enzyme").unwrap()[0] = enzyme;
            self.graph.learn(&self.ext);

            // ── Tracking ──────────────────────────────────────────────────────
            let loss: f32 = pred.iter().enumerate()
                .map(|(i, &p)| {
                    let t = if i == *label { 1.0 } else { 0.0 };
                    (t - p).powi(2)
                })
                .sum::<f32>() / pred.len() as f32;
            total_loss += loss;
            if argmax(&pred) == *label { correct += 1; }

            if (n + 1) % PRINT_EVERY == 0 {
                let seen = n + 1;
                let avg_loss = total_loss / seen as f32;
                let train_acc = correct as f32 / seen as f32 * 100.0;
                print!("\r  [{:>5}/{:>5}]  loss={:.4}  train_acc={:.1}%   ",
                       seen, samples.len(), avg_loss, train_acc);
                std::io::stdout().flush().ok();
            }
        }
        println!();
        total_loss / samples.len() as f32
    }

    fn evaluate(&mut self, samples: &[(Vec<f32>, usize)]) -> f32 {
        let correct = samples.iter()
            .filter(|(pixels, label)| self.predict(pixels) == *label)
            .count();
        correct as f32 / samples.len() as f32
    }

    fn name(&self) -> &str { "Subgraph (Where-gated Hebbian)" }
}

// ─── main ──────────────────────────────────────────────────────────────────────

fn main() -> anyhow::Result<()> {
    let data_root = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent().unwrap()
        .join("data/MNIST");

    println!("Loading MNIST...");
    let train_ds = MnistDataset::load(&data_root.join("train.parquet"))?;
    let test_ds  = MnistDataset::load(&data_root.join("test.parquet"))?;
    println!("Train: {} | Test: {}", train_ds.len(), test_ds.len());

    let train: Vec<(Vec<f32>, usize)> = train_ds.samples.iter()
        .map(|s| (s.pixels.clone(), s.label))
        .collect();
    let test: Vec<(Vec<f32>, usize)> = test_ds.samples.iter()
        .map(|s| (s.pixels.clone(), s.label))
        .collect();

    let mut model = SubgraphModel::new();
    run_pipeline(&mut model, &train, &test, 5);

    Ok(())
}
