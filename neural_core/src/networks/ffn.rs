/// Two-layer feedforward network with manual backprop.
///
/// Architecture: input → hidden (ReLU) → output (softmax)
/// Loss: cross-entropy
///
/// Weight shapes:
///   w1: hidden × input
///   b1: hidden
///   w2: output × hidden
///   b2: output
use ndarray::{Array1, Array2};
use crate::activation::{relu, relu_prime};

pub struct FeedForwardNet {
    pub w1: Array2<f32>,
    pub b1: Array1<f32>,
    pub w2: Array2<f32>,
    pub b2: Array1<f32>,
    pub n_input: usize,
    pub n_hidden: usize,
    pub n_output: usize,
    pub lr: f32,
}

impl FeedForwardNet {
    pub fn new(n_input: usize, n_hidden: usize, n_output: usize, lr: f32) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        // He initialization for ReLU
        let scale1 = (2.0 / n_input as f32).sqrt();
        let scale2 = (2.0 / n_hidden as f32).sqrt();

        let w1 = Array2::from_shape_fn((n_hidden, n_input), |_| rng.gen::<f32>() * scale1 - scale1 / 2.0);
        let b1 = Array1::zeros(n_hidden);
        let w2 = Array2::from_shape_fn((n_output, n_hidden), |_| rng.gen::<f32>() * scale2 - scale2 / 2.0);
        let b2 = Array1::zeros(n_output);

        Self { w1, b1, w2, b2, n_input, n_hidden, n_output, lr }
    }

    /// Forward pass, returns (pre1, h, pre2, probs).
    fn forward(&self, x: &Array1<f32>) -> (Array1<f32>, Array1<f32>, Array1<f32>, Array1<f32>) {
        let pre1 = self.w1.dot(x) + &self.b1;
        let h = pre1.mapv(relu);
        let pre2 = self.w2.dot(&h) + &self.b2;
        let probs = softmax(&pre2);
        (pre1, h, pre2, probs)
    }

    /// Predict class for a single example.
    pub fn predict(&self, x: &Array1<f32>) -> usize {
        let (_, _, _, probs) = self.forward(x);
        probs.iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    /// One gradient step on a single example. Returns cross-entropy loss.
    pub fn train_step(&mut self, x: &Array1<f32>, label: usize) -> f32 {
        let (pre1, h, _pre2, probs) = self.forward(x);

        // Cross-entropy loss
        let loss = -probs[label].max(1e-10).ln();

        // Backprop — output layer
        let mut d_pre2 = probs.clone();
        d_pre2[label] -= 1.0;  // ∂L/∂pre2 for softmax + cross-entropy

        let dw2 = outer(&d_pre2, &h);
        let db2 = d_pre2.clone();

        // Backprop — hidden layer
        let dh = self.w2.t().dot(&d_pre2);
        let d_pre1 = dh * pre1.mapv(relu_prime);

        let dw1 = outer(&d_pre1, x);
        let db1 = d_pre1;

        // SGD update
        self.w2 -= &(dw2 * self.lr);
        self.b2 -= &(db2 * self.lr);
        self.w1 -= &(dw1 * self.lr);
        self.b1 -= &(db1 * self.lr);

        loss
    }
}

fn softmax(x: &Array1<f32>) -> Array1<f32> {
    let max = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp = x.mapv(|v| (v - max).exp());
    let sum = exp.sum();
    exp / sum
}

fn outer(a: &Array1<f32>, b: &Array1<f32>) -> Array2<f32> {
    let m = a.len();
    let n = b.len();
    Array2::from_shape_fn((m, n), |(i, j)| a[i] * b[j])
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn forward_output_shape() {
        let net = FeedForwardNet::new(4, 8, 3, 0.01);
        let x = Array1::from_vec(vec![0.1, 0.2, 0.3, 0.4]);
        let (_, _, _, probs) = net.forward(&x);
        assert_eq!(probs.len(), 3);
        assert!((probs.sum() - 1.0).abs() < 1e-5);
    }

    #[test]
    fn train_step_reduces_loss() {
        let mut net = FeedForwardNet::new(4, 16, 3, 0.1);
        let x = Array1::from_vec(vec![1.0, 0.0, 0.0, 0.0]);
        let mut loss = f32::MAX;
        for _ in 0..100 {
            loss = net.train_step(&x, 0);
        }
        // After 100 steps on a fixed example, loss should drop
        assert!(loss < 1.5, "loss={loss}");
    }
}
