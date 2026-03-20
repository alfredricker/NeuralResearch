use super::Learn;
use crate::activation::sigma;

/// Oja's rule — weight-normalised Hebbian learning.
///
///   Δw_ij = η · σ(post_i) · (σ(pre_j) − σ(post_i) · w_ij)
///
/// The subtracted term prevents unbounded weight growth: as w grows, the
/// correction term grows with it, pushing the weight row toward a unit vector
/// aligned with the first principal component of the input distribution.
///
/// Unlike raw Hebbian, no explicit weight decay is needed — the normalisation
/// term acts as an implicit decay. `mu` is provided as an optional additional
/// decay for regularisation.
pub struct OjaRule {
    pub mu: f32,  // optional extra weight decay (default 0.0)
}

impl Default for OjaRule {
    fn default() -> Self { Self { mu: 0.0 } }
}

impl Learn for OjaRule {
    #[inline]
    fn update_weight(&mut self, w: f32, pre: f32, post: f32, eta: f32, _neuron_idx: usize) -> f32 {
        let pre_s  = sigma(pre);
        let post_s = sigma(post);
        w * (1.0 - self.mu) + eta * post_s * (pre_s - post_s * w)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_bounded_under_repeated_updates() {
        let mut rule = OjaRule::default();
        let mut w = 0.0f32;
        // Strong correlated signal — Oja should converge, not blow up.
        for _ in 0..1000 {
            w = rule.update_weight(w, 1.0, 1.0, 0.1, 0);
        }
        assert!(w.is_finite());
        assert!(w.abs() <= 2.0);
    }

    #[test]
    fn positive_correlation_grows_weight() {
        let mut rule = OjaRule::default();
        let w = rule.update_weight(0.0, 1.0, 1.0, 0.01, 0);
        assert!(w > 0.0);
    }

    #[test]
    fn normalisation_term_shrinks_oversized_weight() {
        let mut rule = OjaRule::default();
        // pre=0, post=1, w=1: correction = σ(1)·(σ(0) − σ(1)·1) = 0.5·(0 − 0.5) < 0
        let w = rule.update_weight(1.0, 0.0, 1.0, 0.1, 0);
        assert!(w < 1.0);
    }
}
