use crate::network::event::{Event, SOMATIC_SPIKE, DENDRITIC_SPIKE, FORWARD_AP, push_event};
use std::sync::atomic::AtomicU32;
use crate::constants::{T_BETA, H_ALPHA, ALPHA_BOOST};
use crate::neuron::synapse::{update_weight, update_synapse_alpha};
use crate::neuron::dendrite::update_dendrite_activity;

// Somatic spike: update beta, BaP weight updates across all owned synapses, emit ForwardAP.
// Alpha decay on each synapse is lazy — computed here from synapse_last_events.
// synapse slices must already be scoped to this neuron via neuron_synapse_range.
pub fn handle_somatic_spike(
    neuron_idx: usize,
    timestamp: u16,
    beta: &mut u8,
    soma_last_event: &mut u16,
    soma_lr: &i16,
    synapse_weights: &mut [i8],
    synapse_alphas: &mut [u8],
    synapse_last_events: &mut [u16],
    event_buf: *mut Event,
    event_tail: &AtomicU32,
    event_capacity: u32,
) {
    let elapsed = timestamp.wrapping_sub(*soma_last_event);
    let decrements = (elapsed / T_BETA).min(15) as u8;
    *beta = beta.saturating_sub(decrements).saturating_add(1).min(63);
    let beta = *beta;
    *soma_last_event = timestamp;

    let lr = *soma_lr;

    for s_idx in 0..synapse_weights.len() {
        update_weight(timestamp, beta, lr, s_idx, synapse_alphas, synapse_last_events, synapse_weights);
    }

    unsafe {
        push_event(event_buf, event_tail, event_capacity,
            Event { event_type: FORWARD_AP, source: neuron_idx as u32, timestamp });
    }
}


// Dendritic spike: propagate to soma scaled by branch_constant (proximal vs distal),
// boost alpha on synapses active at spike time, emit SOMATIC_SPIKE if threshold crossed.
//
// branch_constant > 0: proximal — scales directly onto soma potential
// branch_constant <= 0: distal — attenuated to 1, strong local NMDA-like reinforcement
// synapse slices must already be scoped to this dendrite via dendrite_synapse_range.
pub fn handle_dendritic_spike(
    neuron_idx: usize,
    timestamp: u16,
    dendrite_constant: &i8,
    dendrite_last_event: &mut u16,
    soma_potential: &mut i8,
    soma_threshold: &i8,
    synapse_alphas: &mut [u8],
    synapse_last_events: &mut [u16],
    event_buf: *mut Event,
    event_tail: &AtomicU32,
    event_capacity: u32,
) {
    *dendrite_last_event = timestamp;

    let branch_constant = *dendrite_constant;
    let soma_delta: i8 = branch_constant.max(1);
    *soma_potential = soma_potential.saturating_add(soma_delta);

    for s_idx in 0..synapse_alphas.len() {
        let alpha = update_synapse_alpha(s_idx, timestamp, synapse_alphas, synapse_last_events);
        if alpha > H_ALPHA {
            synapse_alphas[s_idx] = alpha.saturating_add(branch_constant.unsigned_abs());
        }
    }

    if *soma_potential >= *soma_threshold {
        unsafe {
            push_event(event_buf, event_tail, event_capacity,
                Event { event_type: SOMATIC_SPIKE, source: neuron_idx as u32, timestamp });
        }
    }
}


// Forward AP received at a synapse: boost alpha, update dendrite voltage, emit DENDRITIC_SPIKE if threshold crossed.
// synapse slices must already be scoped to this dendrite via dendrite_synapse_range.
pub fn handle_forward_ap(
    s_idx: usize,
    dendrite_idx: usize,
    timestamp: u16,
    synapse_xs: &[u8],
    synapse_alphas: &mut [u8],
    synapse_last_events: &mut [u16],
    synapse_weights: &[i8],
    dendrite_activity: &mut u16,
    dendrite_threshold: &u16,
    event_buf: *mut Event,
    event_tail: &AtomicU32,
    event_capacity: u32,
) {
    let alpha = update_synapse_alpha(s_idx, timestamp, synapse_alphas, synapse_last_events);
    synapse_alphas[s_idx] = alpha.saturating_add(ALPHA_BOOST);

    let delta = update_dendrite_activity(
        s_idx, timestamp,
        synapse_xs, synapse_alphas, synapse_weights, synapse_last_events,
    );
    *dendrite_activity = dendrite_activity.saturating_add_signed(delta);

    if *dendrite_activity >= *dendrite_threshold {
        unsafe {
            push_event(event_buf, event_tail, event_capacity,
                Event { event_type: DENDRITIC_SPIKE, source: dendrite_idx as u32, timestamp });
        }
    }
}

// Apical feedback event received at a synapse: boost alpha by axon_constant, apply
// multiplicative somatic update, emit one SOMATIC_SPIKE per threshold crossing.                          
pub fn handle_apical_fb(                                                                                  
    s_idx: usize,                                                                                         
    neuron_idx: usize,                                                                                    
    timestamp: u16,
    axon_constant: u8,                                                                                    
    synapse_alphas: &mut [u8],
    synapse_last_events: &mut [u16],
    soma_potential: &mut i8,                                                                              
    soma_threshold: i8,
    event_buf: *mut Event,                                                                                
    event_tail: &AtomicU32,
    event_capacity: u32,
) {                                                                                                       
    let alpha = update_synapse_alpha(s_idx, timestamp, synapse_alphas, synapse_last_events);
    let effective_alpha = alpha.saturating_add(axon_constant);                                            
    let v_s = (*soma_potential).max(0) as i32;
    let new_v = *soma_potential as i32 + effective_alpha as i32 * v_s;                                    
                                                                    
    let burst_count = new_v / soma_threshold as i32;                                                      
    *soma_potential = (new_v % soma_threshold as i32) as i8;                                              
                                                            
    for _ in 0..burst_count {                                                                             
        unsafe {                                                                                          
            push_event(event_buf, event_tail, event_capacity,
                Event { event_type: SOMATIC_SPIKE, value: 0, source: neuron_idx as u32, timestamp });     
        }                                                                                            
    }                                                                                                     
}    