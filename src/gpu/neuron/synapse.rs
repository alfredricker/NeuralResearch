pub struct Synapse {
    synapse_weights: Vec<i8>,
    synapse_x: Vec<u8>,
    synapse_alphas: Vec<u8>,
    synapse_last_events: Vec<u16>,
}


pub fn update_weight(
    timestamp: u16, 
    beta: u8, 
    lr: i16, 
    synapse_idx: usize, 
    synapse_alphas: &mut [u8], 
    synapse_last_events: &mut [u16], 
    synapse_weights: &mut [i8]
) { 
    // timestamps wrap according to the global u64 clock.
    let s_elapsed = timestamp.wrapping_sub(synapse_last_events[s_idx]);
    // calculate how much alpha has decayed since the last event on this synapse, and update it
    let alpha = shift_decay_u8(synapse_alphas[s_idx], s_elapsed, ALPHA_DECAY);
    synapse_alphas[s_idx] = alpha;
    synapse_last_events[s_idx] = timestamp;
    // synaptic activity must be above threshold to contribute to weight updates
    // intuition: if a synapse hasn't been active recently, it shouldn't be strongly reinforced by a somatic spike
    if alpha <= H_ALPHA { continue; }
    // the burst term captures burst dependent plasticity -- if the neuron is bursting,
    // then it is receiving top down reinforcement and should trigger LTP
    let burst_term = (beta as i16) - H_BETA; // if beta is above the H_BETA threshold, this will be positive and trigger LTP.
    let delta: i16 = burst_term * (alpha as i16) / lr;
    synapse_weights[s_idx] = synapse_weights[s_idx].saturating_add(delta.clamp(-127, 127) as i8);
}