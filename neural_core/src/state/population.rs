/// Describes a population-level operation applied across all neuron values
/// in a layer simultaneously.
///
/// Unlike `UpdateRule` (per-neuron scalar transition) and `DriveRule`
/// (per-neuron weighted sum), `PopulationRule` operates on the entire
/// activation vector at once — it needs to compare or normalise across
/// neurons, so individual neurons cannot apply it independently.
pub enum PopulationRule {
    /// Winner-Take-All: keep the top-k activations by magnitude, zero the rest.
    ///
    /// Enforces sparse representations — at most `k` neurons are active after
    /// each application. When `k >= len`, all values are kept unchanged.
    Wta { k: usize },

    /// Softmax normalisation: σ(x)_i = exp(x_i) / Σ_j exp(x_j).
    ///
    /// Maps any real-valued vector to a probability distribution over neurons.
    /// Numerically stable: subtracts max(x) before exponentiating.
    Softmax,

    /// Custom population operation: mutates the activation slice in place.
    Custom(Box<dyn Fn(&mut [f32]) + Send>),
}

impl PopulationRule {
    /// Apply the rule to `values` in place.
    pub fn apply(&self, values: &mut [f32]) {
        match self {
            PopulationRule::Wta { k } => wta(values, *k),
            PopulationRule::Softmax   => softmax(values),
            PopulationRule::Custom(f) => f(values),
        }
    }
}

/// Zero all but the top-k entries by magnitude.
fn wta(values: &mut [f32], k: usize) {
    if k >= values.len() { return; }

    // Find the k-th largest magnitude via partial sort of indices.
    let mut indices: Vec<usize> = (0..values.len()).collect();
    indices.sort_unstable_by(|&a, &b| {
        values[b].abs().partial_cmp(&values[a].abs()).unwrap()
    });
    for &idx in &indices[k..] {
        values[idx] = 0.0;
    }
}

/// Numerically stable softmax in place.
fn softmax(values: &mut [f32]) {
    let max = values.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let mut sum = 0.0f32;
    for v in values.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    if sum > 0.0 {
        for v in values.iter_mut() { *v /= sum; }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wta_zeros_losers() {
        let mut v = vec![0.1, 0.9, 0.3, 0.7, 0.2];
        PopulationRule::Wta { k: 2 }.apply(&mut v);
        let nonzero: usize = v.iter().filter(|&&x| x != 0.0).count();
        assert_eq!(nonzero, 2);
        // Top-2 by magnitude should be 0.9 and 0.7
        assert_ne!(v[1], 0.0);
        assert_ne!(v[3], 0.0);
    }

    #[test]
    fn wta_k_ge_len_keeps_all() {
        let mut v = vec![1.0, 2.0, 3.0];
        PopulationRule::Wta { k: 10 }.apply(&mut v);
        assert!(v.iter().all(|&x| x != 0.0));
    }

    #[test]
    fn wta_respects_negative_magnitude() {
        // -0.9 has larger magnitude than 0.5, should survive WTA k=1
        let mut v = vec![0.5, -0.9];
        PopulationRule::Wta { k: 1 }.apply(&mut v);
        assert_eq!(v[0], 0.0);
        assert_eq!(v[1], -0.9);
    }

    #[test]
    fn softmax_sums_to_one() {
        let mut v = vec![1.0, 2.0, 3.0];
        PopulationRule::Softmax.apply(&mut v);
        let sum: f32 = v.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);
    }

    #[test]
    fn softmax_preserves_order() {
        let mut v = vec![1.0, 3.0, 2.0];
        PopulationRule::Softmax.apply(&mut v);
        assert!(v[1] > v[2] && v[2] > v[0]);
    }

    #[test]
    fn softmax_stable_with_large_values() {
        let mut v = vec![1000.0, 1001.0, 999.0];
        PopulationRule::Softmax.apply(&mut v);
        assert!(v.iter().all(|x| x.is_finite()));
        let sum: f32 = v.iter().sum();
        assert!((sum - 1.0).abs() < 1e-5);
    }

    #[test]
    fn custom_rule_applies() {
        let mut v = vec![1.0, 2.0, 3.0];
        PopulationRule::Custom(Box::new(|xs| xs.iter_mut().for_each(|x| *x *= 2.0)))
            .apply(&mut v);
        assert_eq!(v, vec![2.0, 4.0, 6.0]);
    }
}
