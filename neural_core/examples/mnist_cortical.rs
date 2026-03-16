/// Cortical model on MNIST.
/// Usage: cargo run --example mnist_cortical
use std::path::Path;
use neural_core::data::MnistDataset;
use neural_core::networks::CorticalNet;
use neural_core::pipeline::{Model, run_pipeline};

struct CorticalModel {
    net: CorticalNet,
}

impl CorticalModel {
    fn new() -> Self {
        let mut rng = rand::thread_rng();
        Self {
            net: CorticalNet::new_two_level(784, 128, 64, 10, &mut rng),
        }
    }
}

impl Model for CorticalModel {
    fn train_epoch(&mut self, samples: &[(Vec<f32>, usize)]) -> f32 {
        let mut total_loss = 0.0f32;
        for (pixels, label) in samples {
            // Forward
            let logits = self.net.forward(pixels);

            // Cross-entropy loss (for logging only — learning is Hebbian)
            let max_l = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
            let exp: Vec<f32> = logits.iter().map(|&l| (l - max_l).exp()).collect();
            let sum_exp: f32 = exp.iter().sum();
            let prob = exp[*label] / sum_exp.max(1e-10);
            total_loss += -prob.max(1e-10).ln();

            // Hebbian updates + readout delta rule
            self.net.learn(pixels);
            self.net.learn_readout(*label, 0.01);
        }
        total_loss / samples.len() as f32
    }

    fn evaluate(&mut self, samples: &[(Vec<f32>, usize)]) -> f32 {
        let correct: usize = samples.iter()
            .filter(|(pixels, label)| self.net.predict(pixels) == *label)
            .count();
        correct as f32 / samples.len() as f32
    }

    fn name(&self) -> &str { "CorticalNet" }
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

    let mut model = CorticalModel::new();
    run_pipeline(&mut model, &train, &test, 3);

    Ok(())
}
