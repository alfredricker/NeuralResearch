//! FFN baseline on MNIST (Burn backend).
//! Usage: cargo run --example mnist_ffn
use std::path::Path;
use neural_core::data::MnistDataset;
use neural_core::networks::FeedForwardNet;
use neural_core::pipeline::{Model, run_pipeline};

const BATCH_SIZE: usize = 64;

struct FfnModel {
    net: FeedForwardNet,
}

impl FfnModel {
    fn new() -> Self {
        Self { net: FeedForwardNet::new(784, 256, 10, 0.001) }
    }
}

impl Model for FfnModel {
    fn train_epoch(&mut self, samples: &[(Vec<f32>, usize)]) -> f32 {
        let mut total_loss = 0.0;
        let mut n_batches = 0;
        for chunk in samples.chunks(BATCH_SIZE) {
            total_loss += self.net.train_step(chunk);
            n_batches += 1;
        }
        total_loss / n_batches as f32
    }

    fn evaluate(&mut self, samples: &[(Vec<f32>, usize)]) -> f32 {
        let correct = samples.iter()
            .filter(|(pixels, label)| self.net.predict(pixels) == *label)
            .count();
        correct as f32 / samples.len() as f32
    }

    fn name(&self) -> &str { "FFN (Burn/Adam)" }
}

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

    let mut model = FfnModel::new();
    run_pipeline(&mut model, &train, &test, 5);

    Ok(())
}
