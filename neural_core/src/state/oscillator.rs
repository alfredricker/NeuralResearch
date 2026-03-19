use std::f32::consts::TAU; // 2π
use super::{Bounded, State};

/// Cyclic phase oscillator for path integration and grid cell dynamics.
///
/// Phase advances each tick by a base frequency plus the input displacement:
///   φ(t+1) = (φ(t) + ω₀ + Δ) mod 2π
///
/// where ω₀ is the intrinsic angular velocity (radians/tick) and Δ is the
/// displacement signal carried by the input (State::Continuous). Pass
/// Continuous(0.0) for a free-running oscillator.
///
/// Output is ContinuousBounded with range [0, 2π).
pub struct Oscillator {
    /// Current phase φ ∈ [0, 2π).
    pub phase: f32,
    /// Intrinsic angular velocity ω₀ (radians per tick).
    pub omega: f32,
}

impl Oscillator {
    pub fn new(omega: f32) -> Self {
        Self { phase: 0.0, omega }
    }

    pub fn with_phase(omega: f32, phase: f32) -> Self {
        Self { phase: phase.rem_euclid(TAU), omega }
    }

    /// `input` is a displacement Δ added to the intrinsic advance — State::Continuous.
    fn update(&mut self, input: &State) -> State {
        let delta = match input {
            State::Continuous(d) => *d,
            other => panic!("Oscillator expects Continuous input, got {:?}", other),
        };

        self.phase = (self.phase + self.omega + delta).rem_euclid(TAU);

        State::ContinuousBounded(Bounded::new(self.phase, 0.0, TAU))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn phase_wraps_at_two_pi() {
        let mut osc = Oscillator::new(TAU / 4.0); // quarter turn per tick
        for _ in 0..4 {
            osc.update(&State::Continuous(0.0));
        }
        assert!(osc.phase < 1e-4 || (osc.phase - TAU).abs() < 1e-4);
    }

    #[test]
    fn displacement_shifts_phase() {
        let mut osc = Oscillator::new(0.0);
        osc.update(&State::Continuous(1.0));
        assert!((osc.phase - 1.0).abs() < 1e-6);
    }

    #[test]
    fn output_always_in_range() {
        let mut osc = Oscillator::new(1.3);
        for _ in 0..100 {
            let s = osc.update(&State::Continuous(0.7));
            if let State::ContinuousBounded(b) = s {
                assert!(b.value() >= b.min && b.value() < b.max);
            }
        }
    }
}
