use super::{Bounded, State};

/// Soft-threshold gate neuron.
///
/// Produces zero below the threshold and a smoothly rising output above it,
/// bounded in [0, 1):
///
///   g(f) = max(f − θ, 0)² / (max(f − θ, 0)² + θ²)
///
/// This is the same smooth gating function used in WhereModule — it is zero
/// when the drive is at or below the threshold, and approaches 1 as the
/// excess drive grows large. No temporal state is carried between ticks.
///
/// Output: ContinuousBounded in [0, 1).
pub struct Threshold {
    /// Gating threshold θ.
    pub theta: f32,
}

impl Threshold {
    pub fn new(theta: f32) -> Self {
        assert!(theta > 0.0, "Threshold: theta must be positive");
        Self { theta }
    }

    /// `input` is the synaptic drive — State::Continuous.
    fn update(&mut self, input: &State) -> State {
        let f = match input {
            State::Continuous(f) => *f,
            other => panic!("Threshold expects Continuous input, got {:?}", other),
        };

        let excess = (f - self.theta).max(0.0);
        let th2 = self.theta * self.theta;
        let gate = excess * excess / (excess * excess + th2);

        State::ContinuousBounded(Bounded::new(gate, 0.0, 1.0))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_below_threshold() {
        let mut t = Threshold::new(0.5);
        let s = t.update(&State::Continuous(0.3));
        if let State::ContinuousBounded(b) = s {
            assert_eq!(b.value(), 0.0);
        }
    }

    #[test]
    fn positive_above_threshold() {
        let mut t = Threshold::new(0.5);
        let s = t.update(&State::Continuous(1.0));
        if let State::ContinuousBounded(b) = s {
            assert!(b.value() > 0.0);
        }
    }

    #[test]
    fn output_bounded() {
        let mut t = Threshold::new(0.5);
        for drive in [0.0f32, 0.5, 1.0, 10.0, 1000.0] {
            let s = t.update(&State::Continuous(drive));
            if let State::ContinuousBounded(b) = s {
                assert!(b.value() >= 0.0 && b.value() < 1.0);
            }
        }
    }

    #[test]
    fn approaches_one_for_large_drive() {
        let mut t = Threshold::new(0.5);
        let s = t.update(&State::Continuous(1000.0));
        if let State::ContinuousBounded(b) = s {
            assert!(b.value() > 0.999);
        }
    }
}
