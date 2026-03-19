use super::Learn;

/// BCM rule (Bienenstock-Cooper-Munro) — stateful, per-neuron learning.
///
/// Each output neuron maintains a sliding modification threshold θ_M that
/// tracks its mean squared output. This creates an LTP/LTD crossover point
/// that shifts with the neuron's recent activity:
///
///   Δw_ij = η · pre_j · post_i · (post_i − θ_M_i)
///   θ_M_i ← (1 − τ)·θ_M_i + τ·post_i²
///
/// When post > θ_M: LTP — the neuron is more active than usual, reinforce.
/// When post < θ_M: LTD — the neuron is less active than usual, weaken.
///
/// Over time, active neurons raise their threshold (harder to potentiate),
/// while inactive neurons lower theirs (easier to potentiate). This drives
/// neurons toward selectivity and produces implicit decorrelation across the
/// layer without any neuron needing to observe its neighbours — the module-
/// level analogue of Sanger's rule.
///
/// Thresholds are initialised lazily: the `thetas` vec grows on first access
/// for each `neuron_idx`, so `BcmRule` does not need to know `n_out` upfront.
pub struct BcmRule {
    /// Per-neuron sliding modification threshold θ_M.
    /// Lazily grown — index i is valid once neuron i has been seen.
    pub thetas: Vec<f32>,
    /// Threshold adaptation rate τ ∈ (0, 1). Smaller = slower adaptation.
    pub tau: f32,
    /// Weight decay rate μ.
    pub mu: f32,
}

impl BcmRule {
    pub fn new(tau: f32, mu: f32) -> Self {
        Self { thetas: Vec::new(), tau, mu }
    }
}

impl Default for BcmRule {
    fn default() -> Self { Self::new(0.01, 0.001) }
}

impl Learn for BcmRule {
    fn update_weight(&mut self, w: f32, pre: f32, post: f32, eta: f32, neuron_idx: usize) -> f32 {
        // Grow threshold vector on first encounter of this neuron index.
        if neuron_idx >= self.thetas.len() {
            self.thetas.resize(neuron_idx + 1, 0.0);
        }

        let theta = self.thetas[neuron_idx];
        let delta = eta * pre * post * (post - theta);

        // Slide θ_M toward mean squared output.
        self.thetas[neuron_idx] = (1.0 - self.tau) * theta + self.tau * post * post;

        w * (1.0 - self.mu) + delta
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ltp_when_post_exceeds_threshold() {
        let mut rule = BcmRule { thetas: vec![0.1], tau: 0.0, mu: 0.0 };
        // post=0.5 > theta=0.1 → LTP
        let w = rule.update_weight(0.0, 1.0, 0.5, 0.1, 0);
        assert!(w > 0.0);
    }

    #[test]
    fn ltd_when_post_below_threshold() {
        let mut rule = BcmRule { thetas: vec![0.8], tau: 0.0, mu: 0.0 };
        // post=0.3 < theta=0.8 → LTD on a positive weight
        let w = rule.update_weight(1.0, 1.0, 0.3, 0.1, 0);
        assert!(w < 1.0);
    }

    #[test]
    fn threshold_rises_with_high_activity() {
        let mut rule = BcmRule::default();
        let initial_theta = 0.0;
        rule.update_weight(0.0, 1.0, 1.0, 0.01, 0);
        assert!(rule.thetas[0] > initial_theta);
    }

    #[test]
    fn threshold_falls_toward_zero_with_no_activity() {
        let mut rule = BcmRule { thetas: vec![1.0], tau: 0.1, mu: 0.0 };
        for _ in 0..100 {
            rule.update_weight(0.0, 0.0, 0.0, 0.01, 0);
        }
        assert!(rule.thetas[0] < 0.01);
    }

    #[test]
    fn lazy_init_grows_threshold_vec() {
        let mut rule = BcmRule::default();
        rule.update_weight(0.0, 1.0, 1.0, 0.01, 4);
        assert_eq!(rule.thetas.len(), 5);
        assert_eq!(rule.thetas[0], 0.0); // unvisited neurons stay at 0
    }

    #[test]
    fn weight_decay_applied_independently() {
        let mut rule = BcmRule { thetas: vec![0.0], tau: 0.0, mu: 0.1 };
        // post=0, pre=0 → no Hebbian term, only decay
        let w = rule.update_weight(1.0, 0.0, 0.0, 0.01, 0);
        assert!((w - 0.9).abs() < 1e-6);
    }
}
