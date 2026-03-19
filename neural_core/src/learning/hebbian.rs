use super::Learn;
use crate::activation::sigma;

/// Hebbian rule with weight decay:
///   Δw = η · σ(pre) · σ(post)
///   w  ← w · (1 - μ) + Δw
pub struct HebbianRule {
    pub mu: f32,  // weight decay rate
}

impl Default for HebbianRule {
    fn default() -> Self {
        Self { mu: 0.001 }
    }
}

impl Learn for HebbianRule {
    #[inline]
    fn update_weight(&mut self, w: f32, pre: f32, post: f32, eta: f32, _neuron_idx: usize) -> f32 {
        w * (1.0 - self.mu) + eta * sigma(pre) * sigma(post)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn weight_grows_correlated() {
        let mut rule = HebbianRule::default();
        let w = rule.update_weight(0.0, 1.0, 1.0, 0.01, 0);
        assert!(w > 0.0);
    }

    #[test]
    fn weight_decays_uncorrelated() {
        let mut rule = HebbianRule::default();
        // Strong positive weight, no correlation
        let w = rule.update_weight(1.0, 0.0, 0.0, 0.01, 0);
        assert!(w < 1.0);
    }
}
