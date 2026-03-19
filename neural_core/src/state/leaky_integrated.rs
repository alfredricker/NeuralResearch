use crate::activation::Activation;
use super::{Bounded, State};

/// Leaky integration update — pure function.
///
/// α(t+1) = (1 − λ)·α(t) + activation(input)
///
/// Extracts α from State::ContinuousBounded (or Continuous as a fallback).
/// Returns State::ContinuousBounded clamped to (−1, 1).
pub fn update(state: &State, input: f32, lambda: f32, activation: &Activation) -> State {
    let alpha = match state {
        State::ContinuousBounded(b) => b.value(),
        State::Continuous(v)        => *v,
        _                           => 0.0,
    };

    let driven = (1.0 - lambda) * alpha + activation.apply(input);
    State::ContinuousBounded(Bounded::new(driven, -1.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::activation::sigma;

    fn zero_state() -> State {
        State::ContinuousBounded(Bounded::new(0.0, -1.0, 1.0))
    }

    fn state_with(alpha: f32) -> State {
        State::ContinuousBounded(Bounded::new(alpha, -1.0, 1.0))
    }

    #[test]
    fn zero_input_decays_to_zero() {
        let mut s = state_with(0.8);
        for _ in 0..200 {
            s = update(&s, 0.0, 0.1, &Activation::Sigma);
        }
        if let State::ContinuousBounded(b) = s {
            assert!(b.value().abs() < 1e-4);
        }
    }

    #[test]
    fn output_stays_bounded() {
        let mut s = zero_state();
        for drive in [-1000.0f32, 0.0, 1000.0] {
            s = update(&s, drive, 0.1, &Activation::Sigma);
            if let State::ContinuousBounded(b) = &s {
                assert!(b.value() > b.min && b.value() < b.max);
            }
        }
    }

    #[test]
    fn lambda_one_is_memoryless() {
        // With λ=1: α(t+1) = 0·α(t) + activation(input) = activation(input)
        // Prior state is entirely discarded.
        let s = state_with(0.5);
        let next = update(&s, 1.0, 1.0, &Activation::Sigma);
        if let State::ContinuousBounded(b) = next {
            assert!((b.value() - sigma(1.0)).abs() < 1e-6);
        }
    }

    #[test]
    fn positive_drive_increases_activation() {
        let s = zero_state();
        let next = update(&s, 2.0, 0.0, &Activation::Sigma);
        if let State::ContinuousBounded(b) = next {
            assert!(b.value() > 0.0);
        }
    }
}
