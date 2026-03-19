use crate::activation::sigma;
use crate::subgraph::{Aggregation, Node, PortSpec, PortValues};

/// A supervised output layer trained with the delta rule.
///
/// Ports:
///   in     (n_hidden)  — wired from the previous layer
///   target (n_classes) — one-hot label supplied via external `PortValues`
///   out    (n_classes) — σ-activated class scores
///
/// Forward:  `out_i = σ(Σ_j w_ij · σ(in_j))`
/// Learning: `Δw_ij = η · (target_i − out_i) · σ(in_j)`
///
/// The `target` port is filled during `FlatGraph::tick` (because no wire
/// targets it) and persists in the node's input buffer so that the
/// subsequent `FlatGraph::learn` call can apply the delta rule.
/// When `target` is all-zero (inference), no weight update is performed.
pub struct SupervisedLayer {
    activations: Vec<f32>,
    weights: Vec<f32>,  // row-major: n_classes × n_hidden
    n_hidden: usize,
    n_classes: usize,
    pub eta: f32,
    input_specs: Vec<PortSpec>,
    output_specs: Vec<PortSpec>,
}

impl SupervisedLayer {
    /// Constant weight initialisation (useful for testing).
    pub fn new(n_hidden: usize, n_classes: usize, weight_init: f32) -> Self {
        Self {
            activations: vec![0.0; n_classes],
            weights: vec![weight_init; n_classes * n_hidden],
            n_hidden,
            n_classes,
            eta: 0.01,
            input_specs: vec![
                PortSpec { name: "in",     dim: n_hidden,  agg: Aggregation::Concat },
                PortSpec { name: "target", dim: n_classes, agg: Aggregation::Sum },
            ],
            output_specs: vec![
                PortSpec { name: "out", dim: n_classes, agg: Aggregation::Concat },
            ],
        }
    }

    /// Random weight initialisation: uniform `[-scale, scale]`.
    pub fn new_random(n_hidden: usize, n_classes: usize, scale: f32) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let weights = (0..n_classes * n_hidden)
            .map(|_| rng.gen_range(-scale..scale))
            .collect();
        Self {
            activations: vec![0.0; n_classes],
            weights,
            n_hidden,
            n_classes,
            eta: 0.01,
            input_specs: vec![
                PortSpec { name: "in",     dim: n_hidden,  agg: Aggregation::Concat },
                PortSpec { name: "target", dim: n_classes, agg: Aggregation::Sum },
            ],
            output_specs: vec![
                PortSpec { name: "out", dim: n_classes, agg: Aggregation::Concat },
            ],
        }
    }
}

impl Node for SupervisedLayer {
    fn input_ports(&self)  -> &[PortSpec] { &self.input_specs  }
    fn output_ports(&self) -> &[PortSpec] { &self.output_specs }

    fn update(&mut self, inputs: &PortValues, outputs: &mut PortValues) {
        let input = inputs.get("in").expect("SupervisedLayer: missing 'in' port");
        for i in 0..self.n_classes {
            let net: f32 = (0..self.n_hidden)
                .map(|j| self.weights[i * self.n_hidden + j] * sigma(input[j]))
                .sum();
            self.activations[i] = sigma(net);
        }
        outputs
            .get_mut("out")
            .expect("SupervisedLayer: missing 'out' port")
            .copy_from_slice(&self.activations);
    }

    /// Delta-rule weight update.  No-op when `target` is all-zero (inference).
    fn learn(&mut self, inputs: &PortValues) {
        let input  = inputs.get("in").expect("SupervisedLayer: missing 'in' port");
        let target = match inputs.get("target") {
            Some(t) if t.iter().any(|&x| x != 0.0) => t.to_vec(),
            _ => return,
        };
        for i in 0..self.n_classes {
            let error = target[i] - self.activations[i];
            for j in 0..self.n_hidden {
                self.weights[i * self.n_hidden + j] +=
                    self.eta * error * sigma(input[j]);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_produces_output() {
        let mut layer = SupervisedLayer::new(4, 3, 0.1);
        let mut inputs = PortValues::zeros_from(layer.input_ports());
        inputs.get_mut("in").unwrap().copy_from_slice(&[1.0, 0.5, -0.5, 0.2]);
        let mut outputs = PortValues::zeros_from(layer.output_ports());

        layer.update(&inputs, &mut outputs);

        let out = outputs.get("out").unwrap();
        assert_eq!(out.len(), 3);
        assert!(out.iter().any(|&x| x != 0.0));
    }

    #[test]
    fn learn_with_target_changes_weights() {
        let mut layer = SupervisedLayer::new(4, 3, 0.1);
        let before = layer.weights.clone();

        let mut inputs = PortValues::zeros_from(layer.input_ports());
        inputs.get_mut("in").unwrap().copy_from_slice(&[1.0, 1.0, 1.0, 1.0]);
        inputs.get_mut("target").unwrap().copy_from_slice(&[1.0, 0.0, 0.0]);
        let mut outputs = PortValues::zeros_from(layer.output_ports());
        layer.update(&inputs, &mut outputs);
        layer.learn(&inputs);

        assert_ne!(layer.weights, before);
    }

    #[test]
    fn learn_without_target_is_noop() {
        let mut layer = SupervisedLayer::new(4, 3, 0.1);
        let before = layer.weights.clone();

        let mut inputs = PortValues::zeros_from(layer.input_ports());
        inputs.get_mut("in").unwrap().copy_from_slice(&[1.0, 1.0, 1.0, 1.0]);
        // target stays all-zero
        let mut outputs = PortValues::zeros_from(layer.output_ports());
        layer.update(&inputs, &mut outputs);
        layer.learn(&inputs);

        assert_eq!(layer.weights, before);
    }
}
