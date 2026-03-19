use crate::activation::sigma;
use crate::subgraph::{Aggregation, Node, PortSpec, PortValues};

/// Learned location/context module (W_M in the cortical column model).
///
/// Receives a feature vector and a scalar enzyme signal, applies a learned
/// weight matrix, winner-take-all sparsification, and a smooth gating
/// function to produce a sparse class-vote vector.
///
/// ```text
/// in (n_in)  →  W (n_classes × n_in)  →  WTA  →  gate fn  →  gate (n_classes)
///                                                         ↑
///                                              enzyme (1) ─── scales Hebbian η
/// ```
///
/// **Learning (three-factor Hebbian):**
/// ```text
/// Δw[i,j] = ν · η · σ(in[j]) · gate[i]
/// w[i,j]  = (1 − μ) · w[i,j] + Δw[i,j]
/// ```
/// where `ν = enzyme[0]` gates the effective learning rate.
pub struct WhereModule {
    pub n_in: usize,
    pub n_classes: usize,
    /// Weight matrix, row-major, shape n_classes × n_in.
    pub weights: Vec<f32>,
    /// Gate outputs after WTA + gating fn (n_classes).
    pub activations: Vec<f32>,
    /// Base Hebbian learning rate η.
    pub eta: f32,
    /// Weight decay coefficient μ.
    pub mu: f32,
    /// WTA sparsity — top-k winners kept, rest zeroed.
    pub k: usize,
    /// Smooth gating threshold θ_W: gate[i] = pre[i]² / (pre[i]² + θ_W²).
    pub theta_w: f32,

    input_specs: Vec<PortSpec>,
    output_specs: Vec<PortSpec>,
}

impl WhereModule {
    /// Create a new `WhereModule` with Xavier-uniform weight init.
    pub fn new(
        n_in: usize,
        n_classes: usize,
        k: usize,
        eta: f32,
        mu: f32,
        theta_w: f32,
    ) -> Self {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        let scale = (1.0 / n_in as f32).sqrt();
        let weights = (0..n_classes * n_in)
            .map(|_| rng.gen_range(-scale..scale))
            .collect();

        Self {
            n_in,
            n_classes,
            weights,
            activations: vec![0.0; n_classes],
            eta,
            mu,
            k,
            theta_w,
            input_specs: vec![
                PortSpec { name: "in",     dim: n_in,      agg: Aggregation::Concat },
                PortSpec { name: "enzyme", dim: 1,         agg: Aggregation::Sum   },
            ],
            output_specs: vec![
                PortSpec { name: "gate", dim: n_classes, agg: Aggregation::Concat },
            ],
        }
    }
}

impl Node for WhereModule {
    fn input_ports(&self) -> &[PortSpec] {
        &self.input_specs
    }

    fn output_ports(&self) -> &[PortSpec] {
        &self.output_specs
    }

    fn update(&mut self, inputs: &PortValues, outputs: &mut PortValues) {
        let input = inputs.get("in").expect("WhereModule: missing 'in' port");

        // 1. Linear projection: pre[i] = Σ_j w[i,j] · σ(in[j])
        let mut pre = vec![0.0f32; self.n_classes];
        for i in 0..self.n_classes {
            pre[i] = (0..self.n_in)
                .map(|j| self.weights[i * self.n_in + j] * sigma(input[j]))
                .sum();
        }

        // 2. WTA: zero all but top-k entries by magnitude.
        if self.k < self.n_classes {
            let mut indices: Vec<usize> = (0..self.n_classes).collect();
            indices.sort_unstable_by(|&a, &b| {
                pre[b].abs().partial_cmp(&pre[a].abs()).unwrap()
            });
            for &idx in &indices[self.k..] {
                pre[idx] = 0.0;
            }
        }

        // 3. Smooth gate: gate[i] = pre[i]² / (pre[i]² + θ_W²)
        let tw2 = self.theta_w * self.theta_w;
        for i in 0..self.n_classes {
            let p2 = pre[i] * pre[i];
            self.activations[i] = p2 / (p2 + tw2);
        }

        outputs
            .get_mut("gate")
            .expect("WhereModule: missing 'gate' port")
            .copy_from_slice(&self.activations);
    }

    fn learn(&mut self, inputs: &PortValues) {
        let input = inputs.get("in").expect("WhereModule: missing 'in' port");
        let enzyme_buf = inputs.get("enzyme").expect("WhereModule: missing 'enzyme' port");
        let nu = enzyme_buf[0]; // scalar enzyme — gates effective learning rate

        for i in 0..self.n_classes {
            for j in 0..self.n_in {
                let delta = nu * self.eta * sigma(input[j]) * self.activations[i];
                let w = &mut self.weights[i * self.n_in + j];
                *w = (1.0 - self.mu) * *w + delta;
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gate_output_in_range() {
        let mut wm = WhereModule::new(8, 4, 2, 0.01, 1e-4, 0.5);
        let mut inputs = PortValues::zeros_from(wm.input_ports());
        inputs.get_mut("in").unwrap().copy_from_slice(&[1.0, -0.5, 0.3, 0.8, -1.0, 0.2, 0.4, -0.3]);
        inputs.get_mut("enzyme").unwrap()[0] = 0.8;
        let mut outputs = PortValues::zeros_from(wm.output_ports());

        wm.update(&inputs, &mut outputs);

        let gate = outputs.get("gate").unwrap();
        assert_eq!(gate.len(), 4);
        // All gate values in [0, 1).
        assert!(gate.iter().all(|&g| g >= 0.0 && g < 1.0));
        // WTA: at most k=2 non-zero.
        let nonzero = gate.iter().filter(|&&g| g > 0.0).count();
        assert!(nonzero <= 2);
    }

    #[test]
    fn learn_changes_weights() {
        let mut wm = WhereModule::new(4, 3, 2, 0.1, 1e-4, 0.5);
        let before = wm.weights.clone();

        let mut inputs = PortValues::zeros_from(wm.input_ports());
        inputs.get_mut("in").unwrap().copy_from_slice(&[1.0, 0.5, -0.5, 0.2]);
        inputs.get_mut("enzyme").unwrap()[0] = 1.0;
        let mut outputs = PortValues::zeros_from(wm.output_ports());

        wm.update(&inputs, &mut outputs);
        wm.learn(&inputs);

        assert_ne!(wm.weights, before);
    }

    #[test]
    fn zero_enzyme_freezes_weights() {
        let mut wm = WhereModule::new(4, 3, 2, 0.1, 0.0, 0.5); // mu=0 so decay doesn't change
        let mut inputs = PortValues::zeros_from(wm.input_ports());
        inputs.get_mut("in").unwrap().copy_from_slice(&[1.0, 0.5, -0.5, 0.2]);
        inputs.get_mut("enzyme").unwrap()[0] = 0.0; // enzyme=0 → no learning
        let mut outputs = PortValues::zeros_from(wm.output_ports());

        wm.update(&inputs, &mut outputs);
        let before = wm.weights.clone();
        wm.learn(&inputs);

        assert_eq!(wm.weights, before);
    }
}
