//! Where-gated MNIST classifier with three-factor Hebbian learning.
//!
//! Architecture:
//! ```text
//! External: "in0".."in7" (98px each), label used only for accuracy tracking.
//!
//! Patch 0: FeedForward(98→32) → WhereModule(32→10) ──────────────────────────┐
//! Patch 1: FeedForward(98→32) → WhereModule(32→10) ─────────────────────────┐│
//! ...                                                                         ├┤──> ClassifyModule
//! Patch 7: FeedForward(98→32) → WhereModule(32→10) ───────────────────────┘│       │
//!                                    ↑ enzyme (recurrent, prev-tick)          │       │ "enzyme" (1)
//!                                    └────────────────────────────────────────────────┘
//! ```
//!
//! Labels are **never injected** into the graph.  The enzyme signal is
//! self-computed from prediction confidence inside `ClassifyModule`.

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
const K_WINNERS: usize = 2;

/// Static port names for patch pixel inputs ("in0" … "in7").
const PATCH_IN_PORTS: [&str; N_PATCHES] =
    ["in0", "in1", "in2", "in3", "in4", "in5", "in6", "in7"];

// ─── Graph construction ────────────────────────────────────────────────────────

fn build_patch(k: usize) -> FlatGraph {
    let mut ff = FeedForward::new_random(PATCH_SIZE, HIDDEN, 0.1);
    ff.lambda = 0.0; // no temporal leak — each image is independent

    let wm = WhereModule::new(
        HIDDEN,   // n_in
        CLASSES,  // n_classes
        K_WINNERS,
        0.02,     // eta
        1e-4,     // mu
        0.5,      // theta_w
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

    // Expose each patch's pixel input port on the outer graph.
    for k in 0..N_PATCHES {
        outer = outer.expose_input(PATCH_IN_PORTS[k], &format!("patch_{k}"), PATCH_IN_PORTS[k]);
    }

    // Wire patch gates → classify votes (Sum: all patches vote together).
    // Wire classify enzyme → each patch's enzyme input (recurrent: prev-tick).
    for k in 0..N_PATCHES {
        outer = outer
            .wire(&format!("patch_{k}"), "gate", "classify", "votes")
            .wire("classify", "enzyme", &format!("patch_{k}"), "enzyme")
            .recurrent();
    }

    outer.build().expect("outer graph build failed")
}

// ─── Model wrapper ─────────────────────────────────────────────────────────────

struct SubgraphModel {
    graph: FlatGraph,
    ext: PortValues,
}

impl SubgraphModel {
    fn new() -> Self {
        let graph = build_graph();

        // Pre-allocate external PortValues for training (reused each sample).
        let specs: Vec<PortSpec> = (0..N_PATCHES)
            .map(|k| PortSpec { name: PATCH_IN_PORTS[k], dim: PATCH_SIZE, agg: Aggregation::Concat })
            .collect();
        let ext = PortValues::zeros_from(&specs);

        Self { graph, ext }
    }

    /// Fill external port values with the 8 pixel patches for this image.
    fn load_pixels(&mut self, pixels: &[f32]) {
        for k in 0..N_PATCHES {
            self.ext
                .get_mut(PATCH_IN_PORTS[k])
                .unwrap()
                .copy_from_slice(&pixels[k * PATCH_SIZE..(k + 1) * PATCH_SIZE]);
        }
    }

    fn predict(&mut self, pixels: &[f32]) -> usize {
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
        let mut total_entropy = 0.0f32;
        let mut correct = 0usize;
        const PRINT_EVERY: usize = 1000;

        for (n, (pixels, label)) in samples.iter().enumerate() {
            self.load_pixels(pixels);
            self.graph.tick(&self.ext);
            self.graph.learn(&self.ext);

            // Track accuracy and prediction entropy as a proxy for loss.
            let pred = self.graph.read_output("classify", "pred").unwrap();
            let entropy: f32 = pred.iter()
                .map(|&p| if p > 1e-9 { -p * p.ln() } else { 0.0 })
                .sum();
            total_entropy += entropy;
            if argmax(pred) == *label { correct += 1; }

            if (n + 1) % PRINT_EVERY == 0 {
                let seen = n + 1;
                let avg_ent = total_entropy / seen as f32;
                let train_acc = correct as f32 / seen as f32 * 100.0;
                print!("\r  [{:>5}/{:>5}]  entropy={:.3}  train_acc={:.1}%   ",
                       seen, samples.len(), avg_ent, train_acc);
                std::io::stdout().flush().ok();
            }
        }
        println!();
        total_entropy / samples.len() as f32
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
