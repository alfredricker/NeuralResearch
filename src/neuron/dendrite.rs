use crate::constants::X_DECAY;
use crate::math::decay::shift_decay_u8;
use crate::neuron::synapse::update_synapse_alpha;

struct Dendrites {
    pub dendrite_activities: Vec<u16>,
    pub dendrite_last_events: Vec<u16>,
    pub dendrite_constants: Vec<i8>,
    pub dendrite_thresholds: Vec<u16>,
    pub synapse_offsets: Vec<u32>,
}

/// when a synapse receives a spike event, it must update the voltage of the parent dendrite
/// this updated function has asymmetric dynamics dependent on x and alpha
/// delta V_dendrite = w_i * (1 + gamma_i)
/// where gamma is the sum of alpha times an exponential decay of their distance param x
/// for only x_j > x_i (synapses higher along the dendrite have more influence on the soma)
/// this allows for ordered synaptic integration
pub fn update_dendrite_activity(
    dendrite_idx: usize,
    s_idx: usize, // which synapse triggered the update
    timestamp: u16,
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
    for j in (s_idx + 1)..synapse_xs.len() {
        let alpha_j = update_synapse_alpha(j, timestamp, synapse_alphas, synapse_last_events);
        let dx = synapse_xs[j] - x_i;
        // shift_decay_u8(alpha_j, dx, X_DECAY) ≈ alpha_j * exp(-dx / 2^X_DECAY)
        gamma = gamma.saturating_add(shift_decay_u8(alpha_j, dx as u16, X_DECAY) as u16);
    }

    (w_i as i16).saturating_mul(1 + gamma.min(i16::MAX as u16) as i16)
}