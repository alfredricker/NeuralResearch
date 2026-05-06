use crate::gpu::event::{Event, SOMATIC_SPIKE, FORWARD_AP, push_event};
use crate::math::decay::shift_decay_u8;
use std::sync::atomic::AtomicU32;
use crate::constants::{T_BETA, H_ALPHA, ALPHA_DECAY};
use crate::gpu::neuron::synapse::update_weight;

// Somatic spike: update beta, BaP weight updates across all owned synapses, emit ForwardAP.
// Alpha decay on each synapse is lazy — computed here from synapse_last_events.
pub fn handle_somatic_spike(
    neuron_idx: usize,
    timestamp: u16,
    beta: &mut u8,
    soma_last_event: &mut u16,
    soma_lr: &i16,
    dendrite_offsets: &[u32],
    synapse_offsets: &[u32],
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
    let d_start = dendrite_offsets[neuron_idx] as usize;
    let d_end   = dendrite_offsets[neuron_idx + 1] as usize;

    for d_idx in d_start..d_end {
        let s_start = synapse_offsets[d_idx] as usize;
        let s_end   = synapse_offsets[d_idx + 1] as usize;

        for s_idx in s_start..s_end {
            update_weight(
                timestamp,
                beta,
                lr,
                s_idx,
                synapse_alphas,
                synapse_last_events,
                synapse_weights
            );
        }
    }

    push_event(event_buf, event_tail, event_capacity, 
        Event { event_type: FORWARD_AP, source: neuron_idx as u32, timestamp });
}


// Dendritic spike: propagate to soma scaled by branch_constant (proximal vs distal),
// boost alpha on synapses active at spike time, emit SOMATIC_SPIKE if threshold crossed.
//
// branch_constant > 0: proximal — contribution scales with the constant
// branch_constant <= 0: distal — attenuated to 1, local computation without strongly driving soma
pub fn handle_dendritic_spike(
    dendrite_idx: usize,
    neuron_idx: usize,
    timestamp: u16,
    dendrite_constant: &i8,
    dendrite_last_event: &mut u16,
    soma_potential: &mut i8,
    soma_threshold: &i8,
    synapse_offsets: &[u32],
    synapse_alphas: &mut [u8],
    synapse_last_events: &mut [u16],
    event_buf: *mut Event,
    event_tail: &AtomicU32,
    event_capacity: u32,
) {
    *dendrite_last_event = timestamp;

    let branch_constant = *dendrite_constant;
    let soma_delta: i8 = branch_constant.max(1);
    // @QUESTION : how can the apical dendrite produce a burst pattern?
    *soma_potential = soma_potential.saturating_add(soma_delta);

    // NMDA-like: synapses that were recently active get their alpha boosted,
    // reinforcing the inputs that caused this dendritic spike
    let s_start = synapse_offsets[dendrite_idx] as usize;
    let s_end   = synapse_offsets[dendrite_idx + 1] as usize;

    for s_idx in s_start..s_end {
        let s_elapsed = timestamp.wrapping_sub(synapse_last_events[s_idx]);
        let alpha = shift_decay_u8(synapse_alphas[s_idx], s_elapsed, ALPHA_DECAY);
        synapse_alphas[s_idx] = alpha;
        synapse_last_events[s_idx] = timestamp;

        if alpha > H_ALPHA {
            synapse_alphas[s_idx] = alpha.saturating_add(branch_constant.unsigned_abs());
        }
    }

    if *soma_potential >= *soma_threshold {
        push_event(event_buf, event_tail, event_capacity, 
            Event { event_type: SOMATIC_SPIKE, source: neuron_idx as u32, timestamp });
    }
}
