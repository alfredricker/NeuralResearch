use crate::state::{Bounded, State};

/// Soft-threshold gate update — pure function.
///
/// Produces zero below the threshold and a smoothly rising output above it,
/// bounded in [0, 1):
///
///   g(f) = max(f − θ, 0)² / (max(f − θ, 0)² + θ²)
///
/// No temporal state is carried between ticks; `state` is ignored.
/// Returns State::ContinuousBounded in [0, 1).
pub fn update(_state: &State, input: f32, theta: f32) -> State {
    let excess = (input - theta).max(0.0);
    let th2 = theta * theta;
    let gate = excess * excess / (excess * excess + th2);
    State::ContinuousBounded(Bounded::new(gate, 0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s() -> State { State::Continuous(0.0) } // placeholder — threshold ignores state

    #[test]
    fn zero_below_threshold() {
        let out = update(&s(), 0.3, 0.5);
        if let State::ContinuousBounded(b) = out {
            assert_eq!(b.value(), 0.0);
        }
    }

    #[test]
    fn positive_above_threshold() {
        let out = update(&s(), 1.0, 0.5);
        if let State::ContinuousBounded(b) = out {
            assert!(b.value() > 0.0);
        }
    }

    #[test]
    fn output_bounded() {
        for drive in [0.0f32, 0.5, 1.0, 10.0, 1000.0] {
            let out = update(&s(), drive, 0.5);
            if let State::ContinuousBounded(b) = out {
                assert!(b.value() >= 0.0 && b.value() < 1.0);
            }
        }
    }

    #[test]
    fn approaches_one_for_large_drive() {
        let out = update(&s(), 1000.0, 0.5);
        if let State::ContinuousBounded(b) = out {
            assert!(b.value() > 0.999);
        }
    }

    #[test]
    fn exactly_at_threshold_is_zero() {
        let out = update(&s(), 0.5, 0.5);
        if let State::ContinuousBounded(b) = out {
            assert_eq!(b.value(), 0.0);
        }
    }
}
