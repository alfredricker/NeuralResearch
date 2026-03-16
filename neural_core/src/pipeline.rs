use std::time::{Duration, Instant};

/// A trainable/evaluable model.
pub trait Model {
    fn train_epoch(&mut self, samples: &[(Vec<f32>, usize)]) -> f32;  // returns mean loss
    fn evaluate(&mut self, samples: &[(Vec<f32>, usize)]) -> f32;     // returns accuracy
    fn name(&self) -> &str;
}

pub struct PipelineResult {
    pub model_name: String,
    pub train_loss: f32,
    pub test_accuracy: f32,
    pub epoch_time: Duration,
}

/// Run `model` for `epochs` epochs, printing progress each epoch.
pub fn run_pipeline<M: Model>(
    model: &mut M,
    train: &[(Vec<f32>, usize)],
    test: &[(Vec<f32>, usize)],
    epochs: usize,
) -> Vec<PipelineResult> {
    let mut results = Vec::new();

    for epoch in 0..epochs {
        let t0 = Instant::now();
        let loss = model.train_epoch(train);
        let elapsed = t0.elapsed();
        let acc = model.evaluate(test);

        println!(
            "[{}] epoch {:>2}/{} | loss={:.4} | test_acc={:.2}% | {:.1?}",
            model.name(), epoch + 1, epochs, loss, acc * 100.0, elapsed
        );

        results.push(PipelineResult {
            model_name: model.name().to_string(),
            train_loss: loss,
            test_accuracy: acc,
            epoch_time: elapsed,
        });
    }

    results
}
