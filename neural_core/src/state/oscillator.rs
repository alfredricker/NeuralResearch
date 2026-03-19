use std::f32::consts::TAU; // 2π
use super::State;

/// Cyclic phase oscillator update — pure function.
///
/// φ(t+1) = (φ(t) + ω₀ + Δ) mod 2π
///
/// Extracts φ from State::Phase. Returns State::Phase with the advanced phase.
/// Pass input = 0.0 for a free-running oscillator.
pub fn update(state: &State, input: f32, omega: f32) -> State {
    let phi = match state {
        State::Phase(phi) => *phi,
        _ => 0.0,
    };
    State::Phase((phi + omega + input).rem_euclid(TAU))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::TAU;

    fn phase(phi: f32) -> State { State::Phase(phi) }

    #[test]
    fn phase_wraps_at_two_pi() {
        // Quarter turn per tick → full revolution in 4 ticks
        let mut s = phase(0.0);
        for _ in 0..4 {
            s = update(&s, 0.0, TAU / 4.0);
        }
        if let State::Phase(phi) = s {
            assert!(phi < 1e-4 || (phi - TAU).abs() < 1e-4);
        }
    }

    #[test]
    fn displacement_shifts_phase() {
        let s = update(&phase(0.0), 1.0, 0.0);
        if let State::Phase(phi) = s {
            assert!((phi - 1.0).abs() < 1e-6);
        }
    }

    #[test]
    fn free_running_advances_by_omega() {
        let omega = 0.7;
        let s = update(&phase(0.0), 0.0, omega);
        if let State::Phase(phi) = s {
            assert!((phi - omega).abs() < 1e-6);
        }
    }

    #[test]
    fn phase_always_in_range() {
        let mut s = phase(0.0);
        for _ in 0..100 {
            s = update(&s, 0.7, 1.3);
            if let State::Phase(phi) = &s {
                assert!(*phi >= 0.0 && *phi < TAU);
            }
        }
    }
}
