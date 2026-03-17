//! Hierarchical cortical network for classification.
//!
//! [`CorticalNet`] wraps a [`Hierarchy`] of [`CorticalRegion`]s and adds a
//! linear read-out head trained with the softmax cross-entropy delta rule.
//! The hierarchy handles all unsupervised feature extraction via online Hebbian
//! learning; only the read-out layer receives a gradient-style update.
//!
//! # Architecture
//! ```text
//! input (n_input)
//!   └─► Level 0: CorticalRegion(n_l0)
//!         └─► Level 1: CorticalRegion(n_l1)
//!               └─► ReadOut  (linear, n_l1 → n_classes)
//! ```
//!
//! # Learning rules
//! - **Feedforward + recurrent weights**: online Hebbian  (call [`learn`])
//! - **Read-out weights**: softmax cross-entropy delta rule (call [`learn_readout`])
//!
//! # Example
//! ```rust,ignore
//! let mut net = CorticalNet::new_two_level(784, 128, 64, 10, &mut rng);
//! let logits = net.forward(&pixels);    // forward pass
//! net.learn(&pixels);                   // Hebbian weight updates
//! net.learn_readout(label, 0.01);       // read-out delta rule
//! let pred = net.predict(&pixels);      // argmax class label
//! ```
//!
//! [`learn`]: CorticalNet::learn
//! [`learn_readout`]: CorticalNet::learn_readout

use crate::hierarchy::{Hierarchy, HierarchyBuilder};
use crate::region::{CorticalRegion, RegionModule};

/// Hierarchical cortical network with a trainable linear read-out.
///
/// Internally composed via [`HierarchyBuilder`]; extend to more levels or
/// more regions per level by constructing a `Hierarchy` manually and
/// wrapping it alongside your own read-out.
pub struct CorticalNet {
    /// The cortical feature hierarchy (forward pass + Hebbian learning).
    pub hierarchy: Hierarchy,
    /// Read-out weight matrix, row-major shape `[n_classes × n_top]`.
    pub read_w: Vec<f32>,
    /// Number of output classes.
    pub n_classes: usize,
    /// Output width of the top hierarchy level.
    n_top: usize,
}

impl CorticalNet {
    /// Construct a 2-level cortical network for classification.
    ///
    /// | Level | Regions | Model neurons | Receives          |
    /// |-------|---------|--------------|-------------------|
    /// | 0     | 1       | `n_l0`       | raw `n_input` dims |
    /// | 1     | 1       | `n_l1`       | level-0 output    |
    ///
    /// Grid periods are chosen to be coprime for maximum CRT capacity.
    /// Read-out weights are He-initialised.
    pub fn new_two_level(
        n_input: usize,
        n_l0: usize,
        n_l1: usize,
        n_classes: usize,
        rng: &mut impl rand::Rng,
    ) -> Self {
        let r0 = CorticalRegion::new(n_l0, n_input, &[(5, 1), (7, 2), (11, 3)], 0.05, rng);
        let r1 = CorticalRegion::new(n_l1, n_l0,    &[(13, 2), (17, 3)],         0.05, rng);

        let hierarchy = HierarchyBuilder::new()
            .level(vec![Box::new(r0) as Box<dyn RegionModule + Send>])
            .level(vec![Box::new(r1) as Box<dyn RegionModule + Send>])
            .build();

        let scale = (2.0 / n_l1 as f32).sqrt();
        let read_w: Vec<f32> = (0..n_classes * n_l1)
            .map(|_| rng.gen::<f32>() * scale - scale / 2.0)
            .collect();

        Self { hierarchy, read_w, n_classes, n_top: n_l1 }
    }

    /// Propagate `input` through the hierarchy and apply the linear read-out.
    ///
    /// Returns raw (pre-softmax) class logits.
    pub fn forward(&mut self, input: &[f32]) -> Vec<f32> {
        let top_out = self.hierarchy.forward(input);
        linear_readout(top_out, &self.read_w, self.n_classes, self.n_top)
    }

    /// Predict the most likely class label (argmax of logits).
    pub fn predict(&mut self, input: &[f32]) -> usize {
        let logits = self.forward(input);
        argmax(&logits)
    }

    /// Apply Hebbian learning at every level of the hierarchy.
    ///
    /// Must be called **after** [`forward`][Self::forward].
    pub fn learn(&mut self, input: &[f32]) {
        self.hierarchy.learn(input);
    }

    /// Update read-out weights using the softmax cross-entropy delta rule.
    ///
    /// `Δw[c,i] = −lr · (p_c − 1_{c=label}) · top_out[i]`
    ///
    /// Must be called **after** [`forward`][Self::forward] (uses cached
    /// top-level activations from the last forward pass).
    pub fn learn_readout(&mut self, label: usize, lr: f32) {
        let last    = self.hierarchy.n_levels() - 1;
        let top_out = self.hierarchy.level_output(last).to_vec();
        let logits  = linear_readout(&top_out, &self.read_w, self.n_classes, self.n_top);
        let probs   = softmax(&logits);

        for c in 0..self.n_classes {
            let delta = probs[c] - if c == label { 1.0 } else { 0.0 };
            for i in 0..self.n_top {
                self.read_w[c * self.n_top + i] -= lr * delta * top_out[i];
            }
        }
    }
}

// ─── Private helpers ──────────────────────────────────────────────────────────

fn linear_readout(top_out: &[f32], read_w: &[f32], n_classes: usize, n_top: usize) -> Vec<f32> {
    (0..n_classes)
        .map(|c| (0..n_top).map(|i| read_w[c * n_top + i] * top_out[i]).sum::<f32>())
        .collect()
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exp: Vec<f32> = logits.iter().map(|&l| (l - max).exp()).collect();
    let sum: f32 = exp.iter().sum();
    exp.iter().map(|&e| e / sum).collect()
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
        .map(|(i, _)| i)
        .unwrap_or(0)
}
