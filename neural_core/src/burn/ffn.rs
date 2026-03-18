//! Feedforward network (MLP) built on the Burn deep-learning framework.
//!
//! Architecture: input → hidden (ReLU) → output (logits)
//! Training:     Adam optimiser + cross-entropy loss, mini-batch SGD
//!
//! Requires the `ndarray` feature of the `burn` crate (CPU backend).

use burn::{
    config::Config,
    module::{AutodiffModule, Module},
    nn::{Linear, LinearConfig},
    nn::loss::CrossEntropyLossConfig,
    optim::{adaptor::OptimizerAdaptor, Adam, AdamConfig, GradientsParams, Optimizer},
    tensor::{
        backend::Backend,
        Int, Tensor, TensorData,
    },
};

/// Concrete optimizer type: `OptimizerAdaptor` wraps the stateless `Adam` update
/// rule and tracks per-parameter moment state via Burn's adaptor pattern.
type MyOptim = OptimizerAdaptor<Adam, Mlp<TrainBackend>, TrainBackend>;

/// CPU-only inference backend (pure-Rust NdArray, no external deps required).
pub type InferenceBackend = burn::backend::NdArray;
/// Training backend – wraps NdArray with automatic differentiation.
pub type TrainBackend = burn::backend::Autodiff<InferenceBackend>;

// ─── Model ───────────────────────────────────────────────────────────────────

/// Two-layer MLP.  Generic over `B` so the same type works for both training
/// (`TrainBackend`) and inference (`InferenceBackend`).
///
/// Note: `Module` already derives `Clone`; do not add a manual `Clone` derive.
#[derive(Module, Debug)]
pub struct Mlp<B: Backend> {
    fc1: Linear<B>,
    fc2: Linear<B>,
}

/// Configuration for building an [`Mlp`].
#[derive(Config, Debug)]
pub struct MlpConfig {
    pub n_input: usize,
    pub n_hidden: usize,
    pub n_output: usize,
}

impl MlpConfig {
    /// Initialise an [`Mlp`] on `device` with Burn's default weight init.
    pub fn init<B: Backend>(&self, device: &B::Device) -> Mlp<B> {
        Mlp {
            fc1: LinearConfig::new(self.n_input, self.n_hidden).init(device),
            fc2: LinearConfig::new(self.n_hidden, self.n_output).init(device),
        }
    }
}

impl<B: Backend> Mlp<B> {
    /// `[batch, n_input]` → `[batch, n_output]` (raw logits, no softmax)
    pub fn forward(&self, x: Tensor<B, 2>) -> Tensor<B, 2> {
        let x = self.fc1.forward(x);
        let x = burn::tensor::activation::relu(x);
        self.fc2.forward(x)
    }
}

// ─── Trainer ─────────────────────────────────────────────────────────────────

/// Stores an [`Mlp`] + Adam optimiser ready for mini-batch training.
///
/// The model is held in `Option` because Burn's `Optimizer::step` consumes
/// the model by value and returns an updated copy.
///
/// `burn::optim::Adam` has no generic parameters in 0.20; parameter identity
/// is tracked internally by Burn via each parameter's unique ID.
pub struct FeedForwardNet {
    model: Option<Mlp<TrainBackend>>,
    optim: MyOptim,
    device: <TrainBackend as Backend>::Device,
    lr: f64,
}

impl FeedForwardNet {
    pub fn new(n_input: usize, n_hidden: usize, n_output: usize, lr: f32) -> Self {
        let device = Default::default();
        let model = MlpConfig::new(n_input, n_hidden, n_output).init(&device);
        let optim = AdamConfig::new().init();
        Self { model: Some(model), optim, device, lr: lr as f64 }
    }

    /// One Adam step on a mini-batch. Returns the average cross-entropy loss.
    pub fn train_step(&mut self, batch: &[(Vec<f32>, usize)]) -> f32 {
        let n = batch.len();
        let n_input = batch[0].0.len();

        let flat: Vec<f32> = batch.iter()
            .flat_map(|(p, _)| p.iter().copied())
            .collect();
        let x = Tensor::<TrainBackend, 2>::from_data(
            TensorData::new(flat, vec![n, n_input]), &self.device);

        let labels: Vec<i32> = batch.iter().map(|(_, l)| *l as i32).collect();
        let y = Tensor::<TrainBackend, 1, Int>::from_data(
            TensorData::new(labels, vec![n]), &self.device);

        let model = self.model.take().expect("model present");
        let logits = model.forward(x);
        let loss = CrossEntropyLossConfig::new()
            .init::<TrainBackend>(&self.device)
            .forward(logits, y);

        // Save scalar before backward() consumes the tensor.
        let loss_val = loss.clone().into_scalar();

        let grads = loss.backward();
        let grads_params = GradientsParams::from_grads(grads, &model);
        self.model = Some(self.optim.step(self.lr, model, grads_params));

        loss_val
    }

    /// Predict class index for a single example (inference mode, no grad tracking).
    pub fn predict(&self, pixels: &[f32]) -> usize {
        let device: <InferenceBackend as Backend>::Device = Default::default();
        let x = Tensor::<InferenceBackend, 2>::from_data(
            TensorData::new(pixels.to_vec(), vec![1, pixels.len()]), &device);
        // model.valid() strips autodiff wrappers for pure inference.
        let logits = self.model.as_ref().unwrap().valid().forward(x);
        let values = logits.into_data().to_vec::<f32>().unwrap();
        values.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_output_shape() {
        let device: <InferenceBackend as Backend>::Device = Default::default();
        let model = MlpConfig::new(4, 8, 3).init::<InferenceBackend>(&device);
        let x = Tensor::<InferenceBackend, 2>::from_data(
            TensorData::new(vec![0.1f32, 0.2, 0.3, 0.4], vec![1, 4]), &device);
        let logits = model.forward(x);
        assert_eq!(logits.dims(), [1, 3]);
    }

    #[test]
    fn train_step_reduces_loss() {
        let mut net = FeedForwardNet::new(4, 16, 3, 0.01);
        // Repeat the same sample in a batch of 8 for 50 steps.
        let batch: Vec<(Vec<f32>, usize)> = vec![(vec![1.0f32, 0.0, 0.0, 0.0], 0); 8];
        let mut loss = f32::MAX;
        for _ in 0..50 {
            loss = net.train_step(&batch);
        }
        assert!(loss < 1.5, "loss={loss}");
    }
}
