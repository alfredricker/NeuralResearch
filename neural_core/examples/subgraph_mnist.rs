//! MNIST classifier built from wired subgraph modules.
//!
//! Architecture (all via NetworkBuilder):
//!
//!   external["in"] (784)
//!       │
//!   ┌───┴───────────────────────────────────┐
//!   │  hidden: FeedForward(784 → 256)       │
//!   │  Hebbian leaky-integration layer      │
//!   └───┬───────────────────────────────────┘
//!       │ wire "out" → "in"
//!   ┌───┴───────────────────────────────────┐
//!   │  output: SupervisedLayer(256 → 10)    │
//!   │  delta-rule supervised classifier     │
//!   └───┬──────────────────┬────────────────┘
//!       │ "out"             │ "target" ← external["target"] (one-hot label)
//!       ▼                   │            (no wire, filled from external)
//!   class scores            └──────── (training only; zeros at inference)
//!
//! Usage: cargo run --example subgraph_mnist

use std::io::Write;
use std::path::Path;
use neural_core::burn::{Model, MnistDataset, run_pipeline};
use neural_core::modules::{FeedForward, SupervisedLayer};
use neural_core::subgraph::{Aggregation, FlatGraph, NetworkBuilder, PortSpec, PortValues};

// ─── Model wrapper ────────────────────────────────────────────────────────────

struct SubgraphModel {
    graph: FlatGraph,
}

impl SubgraphModel {
    fn new() -> Self {
        // Hidden layer: random init, no leaky memory (λ=0) so each image is
        // processed independently.
        let mut hidden = FeedForward::new_random(784, 256, 0.1);
        hidden.lambda = 0.0;

        let graph = NetworkBuilder::new("mnist_subgraph")
            .add_node("hidden", hidden)
            .add_node("output", SupervisedLayer::new_random(256, 10, 0.1))
            .wire("hidden", "out", "output", "in")
            .build()
            .unwrap();

        Self { graph }
    }

    fn predict(&mut self, pixels: &[f32]) -> usize {
        let mut ext = PortValues::zeros_from(&[
            PortSpec { name: "in", dim: 784, agg: Aggregation::Concat },
        ]);
        ext.get_mut("in").unwrap().copy_from_slice(pixels);
        self.graph.tick(&ext);
        argmax(self.graph.read_last_output("out").unwrap())
    }
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}

// ─── Pipeline integration ─────────────────────────────────────────────────────

impl Model for SubgraphModel {
    fn train_epoch(&mut self, samples: &[(Vec<f32>, usize)]) -> f32 {
        let mut ext = PortValues::zeros_from(&[
            PortSpec { name: "in",     dim: 784, agg: Aggregation::Concat },
            PortSpec { name: "target", dim: 10,  agg: Aggregation::Sum },
        ]);

        let mut total_loss = 0.0;
        let mut correct = 0usize;
        const PRINT_EVERY: usize = 1000;

        for (n, (pixels, label)) in samples.iter().enumerate() {
            ext.get_mut("in").unwrap().copy_from_slice(pixels);

            // One-hot encode the label.
            let target = ext.get_mut("target").unwrap();
            target.fill(0.0);
            target[*label] = 1.0;

            self.graph.tick(&ext);
            self.graph.learn(&ext);

            // Track MSE loss and training accuracy.
            let out = self.graph.read_last_output("out").unwrap();
            let loss: f32 = out.iter().enumerate()
                .map(|(i, &o)| {
                    let t = if i == *label { 1.0 } else { 0.0 };
                    (t - o).powi(2)
                })
                .sum::<f32>() / out.len() as f32;
            total_loss += loss;
            if argmax(out) == *label { correct += 1; }

            if (n + 1) % PRINT_EVERY == 0 {
                let seen = n + 1;
                let avg_loss = total_loss / seen as f32;
                let train_acc = correct as f32 / seen as f32 * 100.0;
                print!("\r  [{:>5}/{:>5}]  loss={:.4}  train_acc={:.1}%   ",
                       seen, samples.len(), avg_loss, train_acc);
                std::io::stdout().flush().ok();
            }
        }
        println!(); // newline after the overwriting progress line
        total_loss / samples.len() as f32
    }

    fn evaluate(&mut self, samples: &[(Vec<f32>, usize)]) -> f32 {
        let correct = samples.iter()
            .filter(|(pixels, label)| self.predict(pixels) == *label)
            .count();
        correct as f32 / samples.len() as f32
    }

    fn name(&self) -> &str { "Subgraph (Hebbian+Delta)" }
}

// ─── main ─────────────────────────────────────────────────────────────────────

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
