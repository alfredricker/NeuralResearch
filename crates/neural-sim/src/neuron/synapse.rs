use crate::math::decay::shift_decay_u8;
use crate::constants::{ALPHA_DECAY, H_ALPHA, H_BETA};

pub struct Synapse {
    pub synapse_weights: Vec<i8>,
    // must order them by increasing x for a given dendrite.
    // x must also be unique along a dendrite
    pub synapse_x: Vec<u8>,
    pub synapse_alphas: Vec<u8>,
    pub synapse_last_events: Vec<u16>,
}

// Applies lazy alpha decay for a synapse and updates its timestamp. Returns the new alpha.
pub fn update_synapse_alpha(
    s_idx: usize,
    timestamp: u16,
    synapse_alphas: &mut [u8],
    synapse_last_events: &mut [u16],
) -> u8 {
    let elapsed = timestamp.wrapping_sub(synapse_last_events[s_idx]);
    let alpha = shift_decay_u8(synapse_alphas[s_idx], elapsed, ALPHA_DECAY);
    synapse_alphas[s_idx] = alpha;
    synapse_last_events[s_idx] = timestamp;
    alpha
}

pub fn update_weight(
    timestamp: u16,
    beta: u8,
    lr: i16,
    synapse_idx: usize,
    synapse_alphas: &mut [u8],
    synapse_last_events: &mut [u16],
    synapse_weights: &mut [i8],
) {
    let alpha = update_synapse_alpha(synapse_idx, timestamp, synapse_alphas, synapse_last_events);
    // synaptic activity must be above threshold to contribute to weight updates
    if alpha <= H_ALPHA { return; }
    // the burst term captures burst dependent plasticity -- if the neuron is bursting,
    // then it is receiving top down reinforcement and should trigger LTP 
    let burst_term = (beta as i16) - H_BETA;
    let delta: i16 = burst_term * (alpha as i16) / lr;
    synapse_weights[synapse_idx] = synapse_weights[synapse_idx].saturating_add(delta.clamp(-127, 127) as i8);
}

#[cfg(test)]
mod tests {
    use super::*;

    // --- update_synapse_alpha ---

    #[test]
    fn alpha_no_elapsed_unchanged() {
        let mut alphas = [200u8];
        let mut last_events = [100u16];
        let result = update_synapse_alpha(0, 100, &mut alphas, &mut last_events);
        assert_eq!(result, 200);
        assert_eq!(alphas[0], 200);
        assert_eq!(last_events[0], 100);
    }

    #[test]
    fn alpha_one_half_life() {
        // ALPHA_DECAY=8 → halves every 256 ticks
        let mut alphas = [200u8];
        let mut last_events = [0u16];
        let result = update_synapse_alpha(0, 256, &mut alphas, &mut last_events);
        assert_eq!(result, 100);
        assert_eq!(alphas[0], 100);
        assert_eq!(last_events[0], 256);
    }

    #[test]
    fn alpha_two_half_lives() {
        let mut alphas = [200u8];
        let mut last_events = [0u16];
        let result = update_synapse_alpha(0, 512, &mut alphas, &mut last_events);
        assert_eq!(result, 50);
        assert_eq!(alphas[0], 50);
    }

    #[test]
    fn alpha_decays_to_zero() {
        // 16+ half-lives (16*256 = 4096 ticks) → shift_decay returns 0
        let mut alphas = [200u8];
        let mut last_events = [0u16];
        let result = update_synapse_alpha(0, 4096, &mut alphas, &mut last_events);
        assert_eq!(result, 0);
        assert_eq!(alphas[0], 0);
    }

    #[test]
    fn alpha_timestamp_wraps() {
        // last_event=65530, timestamp=10 → wrapping elapsed = 16 ticks
        // shift_decay_u8(200, 16, 8): shifts=0, drop=6 → 194
        let mut alphas = [200u8];
        let mut last_events = [u16::MAX - 5];
        let result = update_synapse_alpha(0, 10, &mut alphas, &mut last_events);
        assert_eq!(result, 194);
        assert_eq!(last_events[0], 10);
    }

    #[test]
    fn alpha_updates_correct_index() {
        let mut alphas = [50u8, 200u8, 150u8];
        let mut last_events = [0u16; 3];
        let result = update_synapse_alpha(1, 256, &mut alphas, &mut last_events);
        assert_eq!(result, 100);
        assert_eq!(alphas[0], 50);  // untouched
        assert_eq!(alphas[1], 100);
        assert_eq!(alphas[2], 150); // untouched
    }

    // --- update_weight ---

    #[test]
    fn weight_unchanged_alpha_below_threshold() {
        let mut alphas = [H_ALPHA - 1]; // 29 < 30
        let mut last_events = [0u16];
        let mut weights = [10i8];
        update_weight(0, 10, 100, 0, &mut alphas, &mut last_events, &mut weights);
        assert_eq!(weights[0], 10);
    }

    #[test]
    fn weight_unchanged_alpha_at_threshold() {
        // condition is alpha <= H_ALPHA, so equality also skips update
        let mut alphas = [H_ALPHA]; // 30
        let mut last_events = [0u16];
        let mut weights = [10i8];
        update_weight(0, 10, 100, 0, &mut alphas, &mut last_events, &mut weights);
        assert_eq!(weights[0], 10);
    }

    #[test]
    fn weight_ltp_positive_burst() {
        // beta=10, alpha=200, lr=100 → burst_term=6, delta=6*200/100=12
        let mut alphas = [200u8];
        let mut last_events = [0u16];
        let mut weights = [0i8];
        update_weight(0, 10, 100, 0, &mut alphas, &mut last_events, &mut weights);
        assert_eq!(weights[0], 12);
    }

    #[test]
    fn weight_ltd_negative_burst() {
        // beta=0, alpha=200, lr=100 → burst_term=-4, delta=-4*200/100=-8
        let mut alphas = [200u8];
        let mut last_events = [0u16];
        let mut weights = [0i8];
        update_weight(0, 0, 100, 0, &mut alphas, &mut last_events, &mut weights);
        assert_eq!(weights[0], -8);
    }

    #[test]
    fn weight_unchanged_neutral_burst() {
        // beta == H_BETA → burst_term = 0 → delta = 0
        let mut alphas = [200u8];
        let mut last_events = [0u16];
        let mut weights = [42i8];
        update_weight(0, H_BETA as u8, 100, 0, &mut alphas, &mut last_events, &mut weights);
        assert_eq!(weights[0], 42);
    }

    #[test]
    fn weight_saturates_at_max() {
        // beta=10, alpha=200, lr=1 → delta=1200, clamped to 127; 100+127 saturates to 127
        let mut alphas = [200u8];
        let mut last_events = [0u16];
        let mut weights = [100i8];
        update_weight(0, 10, 1, 0, &mut alphas, &mut last_events, &mut weights);
        assert_eq!(weights[0], i8::MAX);
    }

    #[test]
    fn weight_saturates_at_min() {
        // beta=0, alpha=200, lr=1 → delta=-800, clamped to -127; -100+(-127) saturates to -128
        let mut alphas = [200u8];
        let mut last_events = [0u16];
        let mut weights = [-100i8];
        update_weight(0, 0, 1, 0, &mut alphas, &mut last_events, &mut weights);
        assert_eq!(weights[0], i8::MIN);
    }
}