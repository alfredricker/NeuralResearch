use crate::constants::X_DECAY;
use crate::math::decay::shift_decay_u8;
use crate::neuron::synapse::update_synapse_alpha;

pub enum Compartment {
    Apical,
    Basal,
}

pub struct Dendrite {
    pub dendrite_activities: Vec<u16>,
    pub dendrite_last_events: Vec<u16>,
    pub dendrite_constants: Vec<i8>,
    pub dendrite_thresholds: Vec<u16>,
    pub synapse_offsets: Vec<u32>,
    // fixed slot model, see docs
    pub live_synapse_counts: Vec<u8>, // number of synapse SLOTS that are active on this dendrite
}

/// when a synapse receives a spike event, it must update the voltage of the parent dendrite
/// this updated function has asymmetric dynamics dependent on x and alpha
/// delta V_dendrite = w_i * (1 + gamma_i)
/// where gamma is the sum of alpha times an exponential decay of their distance param x
/// for only x_j > x_i (synapses higher along the dendrite have more influence on the soma)
/// this allows for ordered synaptic integration
pub fn update_dendrite_activity(
    s_idx: usize, // which synapse triggered the update
    timestamp: u16,
    live_end: usize, // = base + live_count, computed by caller
    synapse_xs: &[u8],
    synapse_alphas: &mut [u8],
    synapse_weights: &[i8],
    synapse_last_events: &mut [u16],
) -> i16 {
    let x_i = synapse_xs[s_idx];
    let w_i = synapse_weights[s_idx];

    let mut gamma: u16 = 0;

    // loop to calculate gamma
    // synapses must be ordered by increasing x
    for j in (s_idx + 1)..live_end {
        // lazy update decay for each synapse j > i to get current alpha_j, then calculate contribution to gamma
        let alpha_j = update_synapse_alpha(j, timestamp, synapse_alphas, synapse_last_events);
        let dx = synapse_xs[j] - x_i;
        // shift_decay_u8(alpha_j, dx, X_DECAY) ≈ alpha_j * exp(-dx / 2^X_DECAY)
        gamma = gamma.saturating_add(shift_decay_u8(alpha_j, dx as u16, X_DECAY) as u16);
    }

    (w_i as i16).saturating_mul(1 + gamma.min(i16::MAX as u16) as i16)
}

// binary search to find the dendrite index for a given synapse index
// synapse_offsets is a sorted array of the starting synapse index for each dendrite
// dendrite i owns synapses in the range [synapse_offsets[i], synapse_offsets[i+1])
pub fn synapse_to_dendrite(
    s_idx: usize,
    synapse_offsets: &[u32],
) -> usize {
    synapse_offsets.partition_point(|&o| o as usize <= s_idx) - 1
}

#[cfg(test)]
mod tests {
    use super::*;

    // offsets [0, 3, 7, 12]: dendrite 0 owns s[0..3], dendrite 1 owns s[3..7], dendrite 2 owns s[7..12]
    const OFFSETS: [u32; 4] = [0, 3, 7, 12];

    #[test]
    fn synapse_to_dendrite_first_synapse_of_first_dendrite() {
        assert_eq!(synapse_to_dendrite(0, &OFFSETS), 0);
    }

    #[test]
    fn synapse_to_dendrite_last_synapse_of_first_dendrite() {
        assert_eq!(synapse_to_dendrite(2, &OFFSETS), 0);
    }

    #[test]
    fn synapse_to_dendrite_first_synapse_of_second_dendrite() {
        assert_eq!(synapse_to_dendrite(3, &OFFSETS), 1);
    }

    #[test]
    fn synapse_to_dendrite_first_synapse_of_third_dendrite() {
        assert_eq!(synapse_to_dendrite(7, &OFFSETS), 2);
    }

    #[test]
    fn synapse_to_dendrite_last_synapse_of_third_dendrite() {
        assert_eq!(synapse_to_dendrite(11, &OFFSETS), 2);
    }

    #[test]
    fn update_dendrite_activity_single_synapse_no_gamma() {
        // single synapse: gamma=0, delta = w_i * 1 = w_i
        let xs = [10u8];
        let mut alphas = [200u8];
        let weights = [7i8];
        let mut last_events = [0u16];
        let delta = update_dendrite_activity(0, 0, xs.len(), &xs, &mut alphas, &weights, &mut last_events);
        assert_eq!(delta, 7);
    }

    #[test]
    fn update_dendrite_activity_last_synapse_has_no_neighbors() {
        // s_idx at end of slice → no j > s_idx, gamma=0
        let xs = [5u8, 10, 20];
        let mut alphas = [200u8; 3];
        let weights = [10i8, 5, 3];
        let mut last_events = [0u16; 3];
        let delta = update_dendrite_activity(2, 0, xs.len(), &xs, &mut alphas, &weights, &mut last_events);
        assert_eq!(delta, 3); // 3 * (1 + 0)
    }

    #[test]
    fn update_dendrite_activity_active_neighbor_increases_delta() {
        // s_idx=0, x_i=5, w_i=10
        // j=1: alpha_j=200 (elapsed=0), dx=5
        //   shift_decay_u8(200, 5, X_DECAY=4): shifts=0, rem=5, drop=(100*5)>>4=31, result=169
        // gamma=169, delta = 10 * (1+169) = 1700
        let xs = [5u8, 10];
        let mut alphas = [50u8, 200];
        let weights = [10i8, 5];
        let mut last_events = [0u16; 2];
        let delta = update_dendrite_activity(0, 0, xs.len(), &xs, &mut alphas, &weights, &mut last_events);
        assert_eq!(delta, 1700);
    }

    #[test]
    fn update_dendrite_activity_live_end_excludes_dead_tail() {
        // 3 slots but only 2 are live (live_end=2). The distal slot at index 2 has high alpha
        // but is a dead/unbound slot and must NOT contribute to gamma.
        // s_idx=0, x_i=5, w_i=10; only j=1 is in range: alpha=200, dx=5 → shift_decay_u8(200,5,4)=169
        // gamma=169, delta = 10 * (1+169) = 1700 — identical to the 2-live-synapse case above.
        let xs = [5u8, 10, 20];
        let mut alphas = [50u8, 200, 255];
        let weights = [10i8, 5, 3];
        let mut last_events = [0u16; 3];
        let delta = update_dendrite_activity(0, 0, 2, &xs, &mut alphas, &weights, &mut last_events);
        assert_eq!(delta, 1700); // dead slot 2 ignored despite alpha=255
    }
}
