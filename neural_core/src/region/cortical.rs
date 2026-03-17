use crate::activation::sigma;
use crate::graph::{CsrGraph, GraphBuilder};
use crate::learning::{HebbianRule, LearningRule};
use crate::network::{Aggregation, Node, PortSpec, PortValues};
use super::GridBank;

/// CorticalRegion wires together three streams per the Cortical.tex equations:
///
///   f_i(t) = f_i^F · g(f_i^W) + f_i^R - κ · σ(α_i) · E_M(t)   [eq. 8]
///
/// where:
///   f_i^F = feedforward input from F_ω (via W_FF weights)
///   f_i^W = gating signal from W_T grid modules (hardwired)
///   f_i^R = recurrent input from M↔M (via W_RR weights, learned)
///   κ·σ(α_i)·E_M = lateral inhibition (E_M = mean activation of M)
pub struct CorticalRegion {
    /// Model neurons α_i, i ∈ 0..n_model
    pub m_activations: Vec<f32>,
    pub n_model: usize,

    /// Feedforward weights: F_ω (n_input) → M (n_model)
    /// Shape: n_model × n_input (row-major)
    pub w_ff: Vec<f32>,
    pub n_input: usize,

    /// Recurrent weights M↔M as a CSR graph (learned via Hebbian)
    pub w_rr: CsrGraph,

    /// Where module bank (hardwired, not learned)
    pub w_t: GridBank,

    /// Hebbian learning rule for W_RR
    pub learning_rule: HebbianRule,

    /// Hyperparameters
    pub lambda: f32,   // leak rate
    pub kappa: f32,    // lateral inhibition strength
    pub eta: f32,      // Hebbian learning rate
    pub theta: f32,    // sparsity threshold
}

impl CorticalRegion {
    pub fn new(
        n_model: usize,
        n_input: usize,
        grid_specs: &[(usize, i32)],
        recurrent_p: f32,
        rng: &mut impl rand::Rng,
    ) -> Self {
        let w_ff = vec![0.01f32; n_model * n_input];

        let w_rr = GraphBuilder::new(n_model)
            .sparse(recurrent_p, 0.01, rng)
            .build();

        let w_t = GridBank::new(grid_specs);

        Self {
            m_activations: vec![0.0; n_model],
            n_model,
            w_ff,
            n_input,
            w_rr,
            w_t,
            learning_rule: HebbianRule::default(),
            lambda: 0.1,
            kappa: 0.05,
            eta: 0.005,
            theta: 0.05,
        }
    }

    /// One time step: compute f_i for all model neurons and update activations.
    ///
    /// Implements equation 8 from Cortical.tex:
    /// `f_i = f_i^F · g(f_i^W) + f_i^R - κ · σ(α_i) · E_M`
    ///
    /// Prefer calling this directly on a concrete `CorticalRegion`.
    /// When composing regions in a [`Hierarchy`][crate::hierarchy::Hierarchy],
    /// the [`RegionModule::tick`] trait method is used instead.
    pub fn step(&mut self, feedforward: &[f32]) {
        assert_eq!(feedforward.len(), self.n_input);

        let n = self.n_model;
        let gate = self.w_t.activations(); // W_T one-hot gate per module
        // Aggregate gate: sum of all module activations at each W_T neuron
        // For M neurons we project: g_i = dot(gate, w_gate_i) ≈ use mean gate
        // Simplified: use mean of W_T activations as a scalar gate g ∈ [0,1]
        let g_scalar: f32 = gate.iter().sum::<f32>() / gate.len().max(1) as f32;

        // Mean activation for lateral inhibition
        let e_m: f32 = self.m_activations.iter().map(|&a| sigma(a)).sum::<f32>() / n as f32;

        let mut new_act = vec![0.0f32; n];

        for i in 0..n {
            // Feedforward: sum_j W_FF[i,j] * σ(x_j)
            let f_ff: f32 = (0..self.n_input)
                .map(|j| self.w_ff[i * self.n_input + j] * sigma(feedforward[j]))
                .sum();

            // Recurrent: sum over neighbors in W_RR
            let f_rec: f32 = self.w_rr
                .neighbors(i)
                .map(|(src, w)| sigma(self.m_activations[src as usize]) * w)
                .sum();

            // Lateral inhibition
            let f_inh = self.kappa * sigma(self.m_activations[i]) * e_m;

            // Combined drive (eq. 8)
            let drive = f_ff * g_scalar + f_rec - f_inh;

            // Leaky integration
            new_act[i] = (1.0 - self.lambda) * self.m_activations[i] + sigma(drive);
        }

        self.m_activations = new_act;
    }

    /// Hebbian weight update for W_FF after a tick.
    pub fn learn_ff(&mut self, feedforward: &[f32]) {
        for i in 0..self.n_model {
            for j in 0..self.n_input {
                let w = self.w_ff[i * self.n_input + j];
                let updated = self.learning_rule.update_weight(
                    w,
                    feedforward[j],
                    self.m_activations[i],
                    self.eta,
                );
                self.w_ff[i * self.n_input + j] = updated;
            }
        }
    }

    /// Hebbian update for recurrent weights W_RR.
    pub fn learn_rr(&mut self) {
        let acts = self.m_activations.clone();
        for i in 0..self.n_model {
            for k in self.w_rr.edge_range(i) {
                let j = self.w_rr.targets[k] as usize;
                let w = self.w_rr.weights[k];
                self.w_rr.weights[k] =
                    self.learning_rule.update_weight(w, acts[i], acts[j], self.eta);
            }
        }
    }

    /// Advance grid bank by one spatial step.
    pub fn advance_grid(&mut self, displacement: i32) {
        self.w_t.advance(displacement);
    }

    /// Current model neuron activations (the region's output port).
    pub fn output(&self) -> &[f32] {
        &self.m_activations
    }

    /// Sparsity of model layer (fraction of |α_i| > θ).
    pub fn sparsity(&self) -> f32 {
        let active = self.m_activations.iter().filter(|&&a| a.abs() > self.theta).count();
        active as f32 / self.n_model as f32
    }
}

// ─── Node (generic graph) impl ───────────────────────────────────────────────

/// Port constants for `CorticalRegion` when used as a graph `Node`.
///
/// - `feedforward` (Concat): concatenated sensory input from lower regions.
/// - `feedback`    (Sum):    top-down modulatory signal (optional, same dim as n_model).
/// - `feed_out`    (Concat): the region's model-layer activations as output.
impl CorticalRegion {
    fn _input_port_specs(&self) -> Vec<PortSpec> {
        vec![
            PortSpec { name: "feedforward", dim: self.n_input, agg: Aggregation::Concat },
            PortSpec { name: "feedback",    dim: self.n_model, agg: Aggregation::Sum   },
        ]
    }

    fn _output_port_specs(&self) -> Vec<PortSpec> {
        vec![
            PortSpec { name: "feed_out", dim: self.n_model, agg: Aggregation::Concat },
        ]
    }
}

/// Cached per-instance port specs (built once, borrowed on every tick).
///
/// Because `Node::input_ports` returns `&[PortSpec]` we store the specs inside
/// the struct via a lazily-initialised `OnceCell`-like field.  To keep things
/// simple we just build them in `new` and store them directly.
pub struct CorticalRegionNode {
    inner: CorticalRegion,
    input_specs:  Vec<PortSpec>,
    output_specs: Vec<PortSpec>,
}

impl CorticalRegionNode {
    pub fn new(inner: CorticalRegion) -> Self {
        let input_specs  = inner._input_port_specs();
        let output_specs = inner._output_port_specs();
        Self { inner, input_specs, output_specs }
    }
}

impl Node for CorticalRegionNode {
    fn input_ports(&self) -> &[PortSpec] {
        &self.input_specs
    }

    fn output_ports(&self) -> &[PortSpec] {
        &self.output_specs
    }

    fn tick(&mut self, inputs: &PortValues, outputs: &mut PortValues) {
        let ff = inputs.get("feedforward").expect("feedforward port missing");
        self.inner.step(ff);
        let out = outputs.get_mut("feed_out").expect("feed_out port missing");
        out.copy_from_slice(self.inner.output());
    }

    fn learn(&mut self, inputs: &PortValues) {
        let ff = inputs.get("feedforward").expect("feedforward port missing");
        self.inner.learn_ff(ff);
        self.inner.learn_rr();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cortical_region_runs() {
        let mut rng = rand::thread_rng();
        let mut region = CorticalRegion::new(
            16,                // n_model
            8,                 // n_input
            &[(5, 1), (7, 2)], // grid specs
            0.1,               // recurrent connectivity
            &mut rng,
        );
        let input = vec![0.5f32; 8];
        region.step(&input);
        region.learn_ff(&input);
        region.learn_rr();
        assert!(region.m_activations.iter().any(|&a| a != 0.0));
    }

    #[test]
    fn cortical_region_node_trait() {
        use crate::network::{Node, PortValues, PortSpec, Aggregation};

        let mut rng = rand::thread_rng();
        let region = CorticalRegion::new(16, 8, &[(5, 1)], 0.1, &mut rng);
        let mut node = CorticalRegionNode::new(region);

        assert_eq!(node.input_ports().len(), 2);
        assert_eq!(node.output_ports().len(), 1);

        let inputs = PortValues::zeros_from(&[
            PortSpec { name: "feedforward", dim: 8, agg: Aggregation::Concat },
            PortSpec { name: "feedback",    dim: 16, agg: Aggregation::Sum   },
        ]);
        let mut outputs = PortValues::zeros_from(node.output_ports());
        node.tick(&inputs, &mut outputs);
        assert_eq!(outputs.get("feed_out").unwrap().len(), 16);
    }
}
