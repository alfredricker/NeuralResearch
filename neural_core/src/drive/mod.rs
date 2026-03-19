use crate::activation::{Activation, sigma};

/// Describes how a neuron's synaptic drive is computed from a row of weights
/// and a slice of input values.
///
/// Drive is the scalar current delivered to one neuron before the UpdateRule
/// integrates it into state:
///
///   drive_i = DriveRule::compute(inputs, weights_row_i)
///
/// The weight row must have the same length as inputs; this is not checked in
/// release builds.
pub enum DriveRule {
    /// Raw dot product: Σ_j x_j · w_{ij}
    ///
    /// No input nonlinearity. Use when inputs are already bounded (e.g. the
    /// output of a previous ContinuousBounded state) and you want to preserve
    /// linear scaling through the projection.
    Linear,

    /// Parameterised nonlinearity: Σ_j activation(x_j) · w_{ij}
    ///
    /// The standard cortical drive is `Activated(Activation::Sigma)`:
    /// inputs are compressed through σ(x) = x/(|x|+1) before weighting,
    /// bounding each synapse's contribution to (-1, 1).
    Activated(Activation),

    /// Custom projection function: (inputs, weights_row) → drive.
    Custom(Box<dyn Fn(&[f32], &[f32]) -> f32 + Send>),
}

impl DriveRule {
    /// Compute the scalar drive for one neuron.
    ///
    /// `inputs`      — the full input slice (length n_in)
    /// `weights_row` — the weight row for this neuron (length n_in)
    pub fn compute(&self, inputs: &[f32], weights_row: &[f32]) -> f32 {
        debug_assert_eq!(inputs.len(), weights_row.len());
        match self {
            DriveRule::Linear =>
                inputs.iter().zip(weights_row).map(|(x, w)| x * w).sum(),

            DriveRule::Activated(act) =>
                inputs.iter().zip(weights_row).map(|(x, w)| act.apply(*x) * w).sum(),

            DriveRule::Custom(f) =>
                f(inputs, weights_row),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn weights() -> Vec<f32> { vec![1.0, -1.0, 0.5] }
    fn inputs()  -> Vec<f32> { vec![2.0,  4.0, 2.0] }

    #[test]
    fn activated_sigma_applies_sigma() {
        // drive = σ(2)·1 + σ(4)·(-1) + σ(2)·0.5
        let expected: f32 = sigma(2.0) * 1.0
            + sigma(4.0) * -1.0
            + sigma(2.0) * 0.5;
        let got = DriveRule::Activated(Activation::Sigma).compute(&inputs(), &weights());
        assert!((got - expected).abs() < 1e-6);
    }

    #[test]
    fn linear_is_raw_dot_product() {
        // drive = 2·1 + 4·(-1) + 2·0.5 = 2 - 4 + 1 = -1
        let got = DriveRule::Linear.compute(&inputs(), &weights());
        assert!((got - -1.0_f32).abs() < 1e-6);
    }

    #[test]
    fn activated_relu_differs_from_sigma() {
        let a = DriveRule::Activated(Activation::Sigma).compute(&inputs(), &weights());
        let b = DriveRule::Activated(Activation::Relu).compute(&inputs(), &weights());
        assert!((a - b).abs() > 1e-6);
    }

    #[test]
    fn custom_can_implement_max_pooling() {
        let rule = DriveRule::Custom(Box::new(|xs, ws| {
            xs.iter().zip(ws).map(|(x, w)| x * w).fold(f32::NEG_INFINITY, f32::max)
        }));
        // max(2·1, 4·(-1), 2·0.5) = max(2, -4, 1) = 2
        let got = rule.compute(&inputs(), &weights());
        assert!((got - 2.0).abs() < 1e-6);
    }

    #[test]
    fn zero_weights_gives_zero_drive() {
        let zeros = vec![0.0; 3];
        for rule in [DriveRule::Activated(Activation::Sigma), DriveRule::Linear] {
            let got = rule.compute(&inputs(), &zeros);
            assert_eq!(got, 0.0);
        }
    }
}
