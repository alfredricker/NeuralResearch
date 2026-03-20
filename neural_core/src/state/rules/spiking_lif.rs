use crate::state::State;

/// Leaky integrate-and-fire update — pure function.
///
/// Extracts (v, ref_remaining) from State::Spiking; returns the next State::Spiking.
///
/// Refractory convention:
///   ref_remaining > 0  →  readout = 1.0  (spiking / refractory)
///   ref_remaining = 0  →  readout = 0.0  (silent)
///
/// On firing: ref_remaining ← t_ref + 1, so even t_ref = 0 produces one tick
/// of readout = 1.0 before the neuron goes silent again.
/// Each subsequent tick decrements ref_remaining until it reaches 0.
pub fn update(
    state: &State,
    input: f32,
    lambda: f32,
    v_thresh: f32,
    v_reset: f32,
    t_ref: u32,
) -> State {
    let (v, ref_remaining) = match state {
        State::Spiking { v, ref_remaining } => (*v, *ref_remaining),
        _ => (0.0, 0),
    };

    if ref_remaining > 0 {
        return State::Spiking { v: v_reset, ref_remaining: ref_remaining - 1 };
    }

    let new_v = (1.0 - lambda) * v + input;

    if new_v >= v_thresh {
        State::Spiking { v: v_reset, ref_remaining: t_ref + 1 }
    } else {
        State::Spiking { v: new_v, ref_remaining: 0 }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn silent(v: f32) -> State { State::Spiking { v, ref_remaining: 0 } }

    fn readout(s: &State) -> f32 { s.readout() }

    #[test]
    fn fires_when_threshold_crossed() {
        // λ=0, drive=1.0, v_thresh=1.0 → fires on first tick
        let s = update(&silent(0.0), 1.0, 0.0, 1.0, 0.0, 0);
        assert_eq!(readout(&s), 1.0);
    }

    #[test]
    fn silent_below_threshold() {
        let s = update(&silent(0.0), 0.1, 0.1, 1.0, 0.0, 0);
        assert_eq!(readout(&s), 0.0);
    }

    #[test]
    fn refractory_period_respected() {
        // Fires, then t_ref = 3 ticks of readout = 1.0 (refractory),
        // then falls silent.
        let mut s = update(&silent(0.0), 1.0, 0.0, 1.0, 0.0, 3);
        // Firing tick: readout = 1.0, ref_remaining = 4
        assert_eq!(readout(&s), 1.0);
        // Refractory ticks 1–3 (ref_remaining 3, 2, 1): still 1.0
        for _ in 0..3 {
            s = update(&s, 10.0, 0.0, 1.0, 0.0, 3);
            assert_eq!(readout(&s), 1.0);
        }
        // ref_remaining → 0: silent
        s = update(&s, 10.0, 0.0, 1.0, 0.0, 3);
        assert_eq!(readout(&s), 0.0);
    }

    #[test]
    fn leak_reduces_membrane_potential() {
        // λ=0.5, no firing: v should be smaller than input alone
        let s0 = silent(0.5);
        let s1 = update(&s0, 0.1, 0.5, 10.0, 0.0, 0);
        if let State::Spiking { v, .. } = s1 {
            assert!((v - (0.5 * 0.5 + 0.1)).abs() < 1e-6);
        }
    }
}
