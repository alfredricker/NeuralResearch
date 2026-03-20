use crate::activation::sigma;
use crate::drive::DriveRule;
use crate::state::{Bounded, State};
use crate::subgraph::{Aggregation, Node, PortSpec, PortValues};

pub mod ports {
    pub const IN:     &str = "in";
    pub const ENZYME: &str = "enzyme";
    pub const GATE:   &str = "gate";
}

/// Learned location/context module (W_M in the cortical column model).
///
/// Receives a feature vector and a scalar enzyme signal, applies a learned
/// weight matrix, winner-take-all sparsification, and a smooth gating
/// function to produce a sparse class-vote vector.
///
/// ```text
/// in (n_in)  →  DriveRule  →  WTA  →  gate fn  →  gate (n_classes)
///                                               ↑
///                                    enzyme (1) ─── scales Hebbian η
/// ```
///
/// ## Rule usage
///
/// - **`DriveRule`** — governs the linear projection (step 1). Default:
///   `Activated(Sigma)`, the standard cortical drive.
/// - **`UpdateRule`** — not used. WTA + smooth gating is a population
///   operation across all neurons simultaneously; it cannot be expressed as
///   a per-neuron scalar state transition.
/// - **`LearnRule`** — not used. The three-factor Hebbian rule uses the
///   pre-computed gate activation directly (not re-compressed through σ),
///   and is scaled by the enzyme signal ν. Forcing it through `LearnRule`
///   would either double-apply σ to the gate or require a Custom closure.
///
/// **Learning (three-factor Hebbian):**
/// ```text
/// Δw[i,j] = ν · η · σ(in[j]) · gate[i]
/// w[i,j]  = (1 − μ) · w[i,j] + Δw[i,j]
/// ```
/// where `ν = enzyme[0]` gates the effective learning rate.
pub struct WhereModule {
    pub n_in:      usize,
    pub n_classes: usize,
    /// Weight matrix, row-major, shape n_classes × n_in.
    pub weights:   Vec<f32>,
    /// Per-neuron state after WTA + smooth gate. Each is ContinuousBounded ∈ [0, 1).
    /// Readout via `state.readout()` gives the gate activation.
    pub states:    Vec<State>,
    /// Drive rule for the linear projection step.
    pub drive_rule: DriveRule,
    /// Base Hebbian learning rate η.
    pub eta:       f32,
    /// Weight decay coefficient μ.
    pub mu:        f32,
    /// WTA sparsity — top-k winners kept, rest zeroed.
    pub k:         usize,
    /// Smooth gating threshold θ_W: gate[i] = pre[i]² / (pre[i]² + θ_W²).
    pub theta_w:   f32,

    input_specs:  Vec<PortSpec>,
    output_specs: Vec<PortSpec>,
}

impl WhereModule {
    /// Create a new `WhereModule` with Xavier-uniform weight init.
    /// Uses `DriveRule::Activated(Sigma)` as the default projection.
    pub fn new(
        n_in:     usize,
        n_classes: usize,
        k:        usize,
        eta:      f32,
        mu:       f32,
        theta_w:  f32,
    ) -> Self {
        use crate::activation::Activation;
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
            states: vec![State::ContinuousBounded(Bounded::new(0.0, 0.0, 1.0)); n_classes],
            drive_rule: DriveRule::Activated(Activation::Sigma),
            eta,
            mu,
            k,
            theta_w,
            input_specs: vec![
                PortSpec { name: ports::IN,     dim: n_in,      agg: Aggregation::Concat },
                PortSpec { name: ports::ENZYME, dim: 1,         agg: Aggregation::Sum   },
            ],
            output_specs: vec![
                PortSpec { name: ports::GATE, dim: n_classes, agg: Aggregation::Concat },
            ],
        }
    }
}

impl Node for WhereModule {
    fn input_ports(&self)  -> &[PortSpec] { &self.input_specs }
    fn output_ports(&self) -> &[PortSpec] { &self.output_specs }

    fn update(&mut self, inputs: &PortValues, outputs: &mut PortValues) {
        let input = inputs.get(ports::IN).expect("WhereModule: missing 'in' port");

        // 1. Linear projection via DriveRule: pre[i] = Σ_j drive_rule(input, weights_row_i)
        let mut pre = vec![0.0f32; self.n_classes];
        for i in 0..self.n_classes {
            let row = &self.weights[i * self.n_in..(i + 1) * self.n_in];
            pre[i] = self.drive_rule.compute(input, row);
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
        //    Store result in per-neuron State::ContinuousBounded.
        let tw2 = self.theta_w * self.theta_w;
        for i in 0..self.n_classes {
            let p2 = pre[i] * pre[i];
            let gate = p2 / (p2 + tw2);
            self.states[i] = State::ContinuousBounded(Bounded::new(gate, 0.0, 1.0));
        }

        // Write gate readouts to output port.
        let out = outputs.get_mut(ports::GATE).expect("WhereModule: missing 'gate' port");
        for i in 0..self.n_classes {
            out[i] = self.states[i].readout();
        }
    }

    fn learn(&mut self, inputs: &PortValues) {
        let input     = inputs.get(ports::IN).expect("WhereModule: missing 'in' port");
        let enzyme    = inputs.get(ports::ENZYME).expect("WhereModule: missing 'enzyme' port");
        let nu        = enzyme[0]; // scalar enzyme — gates effective learning rate

        for i in 0..self.n_classes {
            let gate_i = self.states[i].readout();
            for j in 0..self.n_in {
                let delta = nu * self.eta * sigma(input[j]) * gate_i;
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
        inputs.get_mut(ports::IN).unwrap()
            .copy_from_slice(&[1.0, -0.5, 0.3, 0.8, -1.0, 0.2, 0.4, -0.3]);
        inputs.get_mut(ports::ENZYME).unwrap()[0] = 0.8;
        let mut outputs = PortValues::zeros_from(wm.output_ports());

        wm.update(&inputs, &mut outputs);

        let gate = outputs.get(ports::GATE).unwrap();
        assert_eq!(gate.len(), 4);
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
        inputs.get_mut(ports::IN).unwrap().copy_from_slice(&[1.0, 0.5, -0.5, 0.2]);
        inputs.get_mut(ports::ENZYME).unwrap()[0] = 1.0;
        let mut outputs = PortValues::zeros_from(wm.output_ports());

        wm.update(&inputs, &mut outputs);
        wm.learn(&inputs);

        assert_ne!(wm.weights, before);
    }

    #[test]
    fn zero_enzyme_freezes_weights() {
        let mut wm = WhereModule::new(4, 3, 2, 0.1, 0.0, 0.5); // mu=0 so decay doesn't interfere
        let mut inputs = PortValues::zeros_from(wm.input_ports());
        inputs.get_mut(ports::IN).unwrap().copy_from_slice(&[1.0, 0.5, -0.5, 0.2]);
        inputs.get_mut(ports::ENZYME).unwrap()[0] = 0.0;
        let mut outputs = PortValues::zeros_from(wm.output_ports());

        wm.update(&inputs, &mut outputs);
        let before = wm.weights.clone();
        wm.learn(&inputs);

        assert_eq!(wm.weights, before);
    }

    #[test]
    fn states_are_bounded_after_update() {
        let mut wm = WhereModule::new(4, 4, 2, 0.01, 1e-4, 0.5);
        let mut inputs = PortValues::zeros_from(wm.input_ports());
        inputs.get_mut(ports::IN).unwrap().copy_from_slice(&[10.0, -10.0, 5.0, -5.0]);
        let mut outputs = PortValues::zeros_from(wm.output_ports());

        wm.update(&inputs, &mut outputs);

        for state in &wm.states {
            let v = state.readout();
            assert!(v >= 0.0 && v < 1.0, "state readout {v} out of [0, 1)");
        }
    }
}
