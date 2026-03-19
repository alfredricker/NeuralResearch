use crate::activation::Activation;
use super::{Bounded, State};

/// A single neuron whose activation leaky-integrates its synaptic drive.
///
/// Dynamics:
///   α(t+1) = (1 − λ)·α(t) + σ(f)
///
/// where f is the synaptic drive (State::Continuous),
/// λ ∈ [0, 1] is the leak rate, and σ(x) = x / (|x| + 1).
///
/// Output state is ContinuousBounded with range (-1, 1).
pub struct LeakyIntegrated {
    /// Current activation α(t) ∈ (-1, 1).
    pub alpha: f32,
    /// Leak rate λ. 0 = no memory, 1 = no update.
    pub lambda: f32,
    pub activation: Activation,
}

impl LeakyIntegrated {
    pub fn new(lambda: f32, activation: Option<Activation>) -> Self {
        Self { alpha: 0.0, lambda, activation: activation.unwrap_or(Activation::Sigma) }
    }

    /// `input` is the synaptic drive f = Σ σ(αₚ)·w(p, h) — State::Continuous.
    fn update(&mut self, input: &State) -> State {
        let f = match input {
            State::Continuous(f) => *f,
            other => panic!("LeakyIntegrated expects Continuous input, got {:?}", other),
        };

        self.alpha = (1.0 - self.lambda) * self.alpha + self.activation.apply(f);

        State::ContinuousBounded(Bounded::new(self.alpha, -1.0, 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_input_decays_to_zero() {
        let mut n = LeakyIntegrated::new(0.1);
        n.alpha = 0.8;
        for _ in 0..200 {
            n.update(&State::Continuous(0.0));
        }
        assert!(n.alpha.abs() < 1e-4);
    }

    #[test]
    fn output_stays_bounded() {
        let mut n = LeakyIntegrated::new(0.1);
        for drive in [-1000.0f32, 0.0, 1000.0] {
            let s = n.update(&State::Continuous(drive));
            if let State::ContinuousBounded(b) = s {
                assert!(b.value() > b.min && b.value() < b.max);
            }
        }
    }

    #[test]
    fn lambda_zero_is_memoryless() {
        let mut n = LeakyIntegrated { alpha: 0.5, lambda: 0.0 };
        let s = n.update(&State::Continuous(1.0));
        // With λ=0: α(t+1) = σ(1.0) = 0.5, previous state forgotten.
        if let State::ContinuousBounded(b) = s {
            assert!((b.value() - sigma(1.0)).abs() < 1e-6);
        }
    }
}
