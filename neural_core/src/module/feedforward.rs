use crate::activation::sigma;
use crate::learning::{HebbianRule, LearningRule};
use crate::network::{Aggregation, Node, PortSpec, PortValues};

/// A feedforward projection layer.
///
/// Accepts an input vector, applies a learned weight matrix and σ activation,
/// and emits the resulting neuron activations.
///
/// ```text
/// input (n_in)  →  W (n_out × n_in)  →  σ  →  activations (n_out)
/// ```
///
/// State is leaky-integrated across ticks:
///   `α(t+1) = (1 − λ)·α(t) + σ(W · σ(input))`
///
/// Weights are updated each `learn()` call via the Hebbian rule with decay.
pub struct FeedForward {
    /// Current neuron activations α ∈ (−1, 1)^n_out.
    pub activations: Vec<f32>,
    /// Weight matrix, row-major, shape n_out × n_in.
    pub weights: Vec<f32>,
    pub n_in: usize,
    pub n_out: usize,
    /// Leak rate λ ∈ [0, 1].  0 = no memory, 1 = no update.
    pub lambda: f32,
    /// Hebbian learning rate η.
    pub eta: f32,

    rule: HebbianRule,
    input_specs: Vec<PortSpec>,
    output_specs: Vec<PortSpec>,
}

impl FeedForward {
    /// Create a new layer with weights initialised to `weight_init`.
    pub fn new(n_in: usize, n_out: usize, weight_init: f32) -> Self {
        Self {
            activations: vec![0.0; n_out],
            weights: vec![weight_init; n_out * n_in],
            n_in,
            n_out,
            lambda: 0.1,
            eta: 0.005,
            rule: HebbianRule::default(),
            input_specs: vec![
                PortSpec { name: "in", dim: n_in, agg: Aggregation::Concat },
            ],
            output_specs: vec![
                PortSpec { name: "out", dim: n_out, agg: Aggregation::Concat },
            ],
        }
    }
}

impl Node for FeedForward {
    fn input_ports(&self) -> &[PortSpec] {
        &self.input_specs
    }

    fn output_ports(&self) -> &[PortSpec] {
        &self.output_specs
    }

    fn tick(&mut self, inputs: &PortValues, outputs: &mut PortValues) {
        let input = inputs.get("in").expect("FeedForward: missing 'in' port");

        for i in 0..self.n_out {
            let net: f32 = (0..self.n_in)
                .map(|j| self.weights[i * self.n_in + j] * sigma(input[j]))
                .sum();
            self.activations[i] = (1.0 - self.lambda) * self.activations[i] + sigma(net);
        }

        outputs
            .get_mut("out")
            .expect("FeedForward: missing 'out' port")
            .copy_from_slice(&self.activations);
    }

    fn learn(&mut self, inputs: &PortValues) {
        let input = inputs.get("in").expect("FeedForward: missing 'in' port");

        for i in 0..self.n_out {
            for j in 0..self.n_in {
                let w = self.weights[i * self.n_in + j];
                self.weights[i * self.n_in + j] =
                    self.rule.update_weight(w, input[j], self.activations[i], self.eta);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn output_nonzero_after_tick() {
        let mut ff = FeedForward::new(4, 8, 0.1);
        let inputs = PortValues::zeros_from(ff.input_ports());
        // manually set input values
        let mut inputs = inputs;
        inputs.get_mut("in").unwrap().copy_from_slice(&[1.0, 0.5, -0.5, 0.2]);
        let mut outputs = PortValues::zeros_from(ff.output_ports());

        ff.tick(&inputs, &mut outputs);

        let out = outputs.get("out").unwrap();
        assert_eq!(out.len(), 8);
        assert!(out.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn learn_changes_weights() {
        let mut ff = FeedForward::new(4, 4, 0.1);
        let before = ff.weights.clone();

        let mut inputs = PortValues::zeros_from(ff.input_ports());
        inputs.get_mut("in").unwrap().copy_from_slice(&[1.0, 1.0, 1.0, 1.0]);
        let mut outputs = PortValues::zeros_from(ff.output_ports());
        ff.tick(&inputs, &mut outputs);
        ff.learn(&inputs);

        assert_ne!(ff.weights, before);
    }

    #[test]
    fn plugs_into_network_builder() {
        use crate::network::NetworkBuilder;

        let fg = NetworkBuilder::new("test")
            .add_node("a", FeedForward::new(4, 8, 0.01))
            .add_node("b", FeedForward::new(8, 4, 0.01))
            .wire("a", "out", "b", "in")
            .build()
            .unwrap();

        assert_eq!(fg.node_count(), 2);
    }
}
