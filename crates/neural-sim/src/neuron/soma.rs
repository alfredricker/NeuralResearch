use crate::math::decay::shift_decay_i8;
use crate::constants::{SOMATIC_DECAY, SOMA_V_RESET, T_BETA};

// beta is a 6-bit burst counter (0..=63).
const BETA_MAX: u8 = 63;

pub struct Soma {
    pub soma_potentials: Vec<i8>,
    pub soma_thresholds: Vec<i8>,
    pub soma_betas: Vec<u8>,
    pub soma_last_events: Vec<u16>,
    pub soma_lrs: Vec<i16>, // learning rates
    pub dendrite_offsets: Vec<u32>,
}

// The soma's complete local state machine for one integration event. Lazily leaks both the
// potential (SOMATIC_DECAY) and the burst counter beta (1 per T_BETA ticks) since the last soma
// event, adds the incoming voltage delta v_s, and — if threshold is crossed — resets the
// potential, bumps beta by the burst size, and returns the burst count (number of APs). Returns 0
// otherwise. This owns ALL beta dynamics; callers (handlers) only read beta for plasticity.
pub fn update_soma_potential(
    timestamp: u16,
    so_idx: usize,
    soma_potentials: &mut [i8],
    soma_last_events: &mut [u16],
    soma_thresholds: &[i8],
    soma_betas: &mut [u8],
    v_s: i16, // voltage change to apply to the soma potential
) -> u8 {
    let elapsed = timestamp.wrapping_sub(soma_last_events[so_idx]);

    // lazy leaks since the last soma event
    let decayed_potential = shift_decay_i8(soma_potentials[so_idx], elapsed, SOMATIC_DECAY);
    let beta_decrement = (elapsed / T_BETA).min(BETA_MAX as u16) as u8;
    let beta = soma_betas[so_idx].saturating_sub(beta_decrement);
    soma_last_events[so_idx] = timestamp;

    let threshold = soma_thresholds[so_idx];
    let new_v = decayed_potential as i16 + v_s;
    if threshold > 0 && new_v >= threshold as i16 {
        let burst = (new_v / threshold as i16) as u8;
        soma_potentials[so_idx] = SOMA_V_RESET;        // reset potential after spike
        soma_betas[so_idx] = beta.saturating_add(burst).min(BETA_MAX); // bursting reinforces beta
        burst
    } else {
        soma_potentials[so_idx] = new_v.clamp(i8::MIN as i16, i8::MAX as i16) as i8;
        soma_betas[so_idx] = beta;                     // commit the lazy decay
        0 // no AP generated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_fire_accumulates_potential() {
        // elapsed=0 → no decay; 5 + 10 = 15 < 100 → no spike
        let mut potentials = [5i8];
        let mut last_events = [100u16];
        let thresholds = [100i8];
        let mut betas = [8u8];
        let burst = update_soma_potential(100, 0, &mut potentials, &mut last_events, &thresholds, &mut betas, 10);
        assert_eq!(burst, 0);
        assert_eq!(potentials[0], 15);
        assert_eq!(betas[0], 8); // unchanged (no decay, no fire)
    }

    #[test]
    fn fire_resets_potential_and_reinforces_beta() {
        // 0 + 32 = 32 >= 20 → burst = 1, potential → SOMA_V_RESET, beta += 1
        let mut potentials = [0i8];
        let mut last_events = [0u16];
        let thresholds = [20i8];
        let mut betas = [5u8];
        let burst = update_soma_potential(0, 0, &mut potentials, &mut last_events, &thresholds, &mut betas, 32);
        assert_eq!(burst, 1);
        assert_eq!(potentials[0], SOMA_V_RESET);
        assert_eq!(betas[0], 6);
    }

    #[test]
    fn multi_ap_burst_increments_beta_by_burst_size() {
        // 0 + 35 = 35 >= 10 → burst = 3, beta += 3
        let mut potentials = [0i8];
        let mut last_events = [0u16];
        let thresholds = [10i8];
        let mut betas = [0u8];
        let burst = update_soma_potential(0, 0, &mut potentials, &mut last_events, &thresholds, &mut betas, 35);
        assert_eq!(burst, 3);
        assert_eq!(betas[0], 3);
    }

    #[test]
    fn beta_lazily_decays_with_elapsed() {
        // elapsed=1000, T_BETA=500 → 2 decrements; beta 10 → 8. v_s=0, no fire.
        let mut potentials = [0i8];
        let mut last_events = [0u16];
        let thresholds = [100i8];
        let mut betas = [10u8];
        let burst = update_soma_potential(1000, 0, &mut potentials, &mut last_events, &thresholds, &mut betas, 0);
        assert_eq!(burst, 0);
        assert_eq!(betas[0], 8);
        assert_eq!(last_events[0], 1000);
    }

    #[test]
    fn beta_caps_at_max_on_large_burst() {
        // threshold=1, v_s=127 → burst=127, beta capped at BETA_MAX
        let mut potentials = [0i8];
        let mut last_events = [0u16];
        let thresholds = [1i8];
        let mut betas = [0u8];
        let burst = update_soma_potential(0, 0, &mut potentials, &mut last_events, &thresholds, &mut betas, 127);
        assert_eq!(burst, 127);
        assert_eq!(betas[0], BETA_MAX);
    }
}