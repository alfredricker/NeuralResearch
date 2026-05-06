use crate::math::decay::shift_decay_u8;
use crate::constants::{ALPHA_DECAY, H_ALPHA, H_BETA};

pub struct Synapse {
    synapse_weights: Vec<i8>,
    // must order them by increasing x for a given dendrite. 
    // x must also be unique along a dendrite
    synapse_x: Vec<u8>,
    synapse_alphas: Vec<u8>,
    synapse_last_events: Vec<u16>,
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