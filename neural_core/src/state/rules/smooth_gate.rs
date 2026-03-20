use crate::state::{Bounded, State};

/// Smooth gate update — pure function.
///
/// Maps a scalar input to a continuous output in [0, 1) via:
///
///   g(x) = x² / (x² + θ²)
///
/// Unlike `Threshold`, there is no dead zone — any non-zero input produces
/// a non-zero output. The function is symmetric around zero and saturates
/// toward 1 as |x| grows large relative to θ.
///
/// No temporal state is carried between ticks; `state` is ignored.
/// Returns State::ContinuousBounded in [0, 1).
pub fn update(_state: &State, input: f32, theta: f32) -> State {
    let i2 = input * input;
    let t2 = theta * theta;
    State::ContinuousBounded(Bounded::new(i2 / (i2 + t2), 0.0, 1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn s() -> State { State::Continuous(0.0) }

    #[test]
    fn zero_input_gives_zero() {
        let out = update(&s(), 0.0, 0.5);
        if let State::ContinuousBounded(b) = out {
            assert_eq!(b.value(), 0.0);
        }
    }

    #[test]
    fn symmetric_around_zero() {
        let pos = update(&s(),  1.0, 0.5);
        let neg = update(&s(), -1.0, 0.5);
        if let (State::ContinuousBounded(p), State::ContinuousBounded(n)) = (pos, neg) {
            assert!((p.value() - n.value()).abs() < 1e-6);
        }
    }

    #[test]
    fn output_always_in_range() {
        for x in [-1000.0f32, -1.0, 0.0, 1.0, 1000.0] {
            let out = update(&s(), x, 0.5);
            if let State::ContinuousBounded(b) = out {
                assert!(b.value() >= 0.0 && b.value() < 1.0);
            }
        }
    }

    #[test]
    fn approaches_one_for_large_input() {
        let out = update(&s(), 1000.0, 0.5);
        if let State::ContinuousBounded(b) = out {
            assert!(b.value() > 0.999);
        }
    }

    #[test]
    fn theta_controls_sensitivity() {
        // Larger theta → lower output for same input
        let tight = update(&s(), 1.0, 0.1);
        let loose = update(&s(), 1.0, 2.0);
        if let (State::ContinuousBounded(t), State::ContinuousBounded(l)) = (tight, loose) {
            assert!(t.value() > l.value());
        }
    }
}
