use crate::activation::sigma;
use crate::graph::{CsrGraph, GraphBuilder};
use crate::learning::{HebbianRule, LearningRule};
use super::{GridBank, RegionModule};

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

// ─── RegionModule impl ───────────────────────────────────────────────────────

/// `CorticalRegion` participates in hierarchies through the `RegionModule` trait.
///
/// The trait methods delegate to the inherent `step` / `output` / `learn_*`
/// methods so that direct usage of the concrete type and trait-object usage
/// behave identically.
impl RegionModule for CorticalRegion {
    #[inline] fn n_in(&self)  -> usize { self.n_input }
    #[inline] fn n_out(&self) -> usize { self.n_model }

    fn tick(&mut self, input: &[f32]) {
        self.step(input);
    }

    fn output(&self) -> &[f32] {
        // Calls the inherent `output` method — not recursive.
        CorticalRegion::output(self)
    }

    fn learn(&mut self, input: &[f32]) {
        self.learn_ff(input);
        self.learn_rr();
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
    fn region_module_trait() {
        let mut rng = rand::thread_rng();
        let mut region: Box<dyn RegionModule + Send> = Box::new(
            CorticalRegion::new(16, 8, &[(5, 1)], 0.1, &mut rng)
        );
        assert_eq!(region.n_in(), 8);
        assert_eq!(region.n_out(), 16);
        let input = vec![0.3f32; 8];
        region.tick(&input);
        assert_eq!(region.output().len(), 16);
    }
}
