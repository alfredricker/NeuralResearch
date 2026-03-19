pub enum Activation {
    Sigma,
    SigmaPrime,
    Relu,
    ReluPrime,
    Sigmoid,
    Softplus,
    Decay,
}

impl Activation {
    pub fn apply(&self, x: f32) -> f32 {
        match self {
            Activation::Sigma => sigma(x),
            Activation::SigmaPrime => sigma_prime(x),
            Activation::Relu => relu(x),
            Activation::ReluPrime => relu_prime(x),
            Activation::Sigmoid => sigmoid(x),
            Activation::Softplus => softplus(x),
            Activation::Decay => decay(x),
        }
    }
}

/// Smooth bounded activation: σ(x) = x / (|x| + 1)
/// Range: (-1, 1), differentiable everywhere, σ(0) = 0
#[inline]
pub fn sigma(x: f32) -> f32 {
    x / (x.abs() + 1.0)
}

/// Derivative of sigma: σ'(x) = 1 / (|x| + 1)²
#[inline]
pub fn sigma_prime(x: f32) -> f32 {
    let d = x.abs() + 1.0;
    1.0 / (d * d)
}

#[inline]
pub fn relu(x: f32) -> f32 {
    x.max(0.0)
}

#[inline]
pub fn relu_prime(x: f32) -> f32 {
    if x > 0.0 { 1.0 } else { 0.0 }
}

/// Soft-plus: ln(1 + e^x), smooth ReLU approximation
#[inline]
pub fn softplus(x: f32) -> f32 {
    (1.0 + x.exp()).ln()
}

/// Weight decay multiplier: (1 - μ)
#[inline]
pub fn decay(w: f32, mu: f32) -> f32 {
    w * (1.0 - mu)
}

#[inline]
pub fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sigma_range() {
        assert!(sigma(100.0) < 1.0);
        assert!(sigma(-100.0) > -1.0);
        assert_eq!(sigma(0.0), 0.0);
        // sigma(1) = 0.5
        assert!((sigma(1.0) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn sigma_symmetry() {
        for x in [-5.0f32, -1.0, 0.5, 2.0, 10.0] {
            assert!((sigma(x) + sigma(-x)).abs() < 1e-6);
        }
    }
}
