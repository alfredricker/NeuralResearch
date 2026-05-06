use crate::constants::{T_BETA, H_ALPHA, H_BETA, ALPHA_DECAY, MSLR};
use crate::math::decay::shift_decay_u8;
use std::sync::atomic::{AtomicU32, Ordering};                                                      
                                                                                                    
pub struct EventQueue {
    buf: Box<[Event]>,                                                                             
    tail: AtomicU32,
    head: AtomicU32,
}
                                                                                                    
impl EventQueue {
    pub fn new(capacity: usize) -> Self {    
      let buf = (0..capacity)                                                                        
          .map(|_| Event { event_type: 0, source: 0, timestamp: 0 })
          .collect::<Vec<_>>()                                                                       
          .into_boxed_slice();                                                                     
      Self { buf, tail: AtomicU32::new(0), head: AtomicU32::new(0) }                                 
    }       

    pub fn drain(&self) -> &[Event] {                                                              
        let head = self.head.load(Ordering::Relaxed) as usize;
        let tail = self.tail.load(Ordering::Relaxed) as usize;                                     
        &self.buf[head % self.buf.len()..tail % self.buf.len()]
    }
    
    // returns the raw parts a kernel function needs to push events
    pub fn producer_handle(&self) -> (*mut Event, &AtomicU32, u32) {
        (
            self.buf.as_ptr() as *mut Event,
            &self.tail,
            self.buf.len() as u32,
        )
    }
}

pub struct Event {
    pub event_type: u8,
    pub source: u32,    // neuron_idx for SOMATIC_SPIKE/FORWARD_AP, dendrite_idx for DENDRITIC_SPIKE
    pub timestamp: u16,
}


// Somatic spike: update beta, BaP weight updates across all owned synapses, emit ForwardAP.
// Alpha decay on each synapse is lazy — computed here from synapse_last_events.
pub fn handle_somatic_spike(
    neuron_idx: usize,
    timestamp: u16,
    soma_betas: &mut [u8],
    soma_last_events: &mut [u16],
    soma_lrs: &[i16],
    dendrite_offsets: &[u32],
    synapse_offsets: &[u32],
    synapse_weights: &mut [i8],
    synapse_alphas: &mut [u8],
    synapse_last_events: &mut [u16],
    event_buf: *mut Event,
    event_tail: &AtomicU32,
    event_capacity: u32,
) {
    let elapsed = timestamp.wrapping_sub(soma_last_events[neuron_idx]);
    let decrements = (elapsed / T_BETA).min(15) as u8;
    let beta = soma_betas[neuron_idx]
        .saturating_sub(decrements)
        .saturating_add(1)
        .min(63);
    soma_betas[neuron_idx] = beta;
    soma_last_events[neuron_idx] = timestamp;

    let lr = soma_lrs[neuron_idx];
    let d_start = dendrite_offsets[neuron_idx] as usize;
    let d_end   = dendrite_offsets[neuron_idx + 1] as usize;

    for d_idx in d_start..d_end {
        let s_start = synapse_offsets[d_idx] as usize;
        let s_end   = synapse_offsets[d_idx + 1] as usize;

        for s_idx in s_start..s_end {
            let s_elapsed = timestamp.wrapping_sub(synapse_last_events[s_idx]);
            let alpha = shift_decay_u8(synapse_alphas[s_idx], s_elapsed, ALPHA_DECAY);
            synapse_alphas[s_idx] = alpha;
            synapse_last_events[s_idx] = timestamp;

            if alpha <= H_ALPHA { continue; }

            let burst_term = (beta as i16) - H_BETA;
            let delta: i16 = burst_term * (alpha as i16) / lr;
            synapse_weights[s_idx] = synapse_weights[s_idx].saturating_add(delta.clamp(-127, 127) as i8);
        }
    }

    queue.push(Event { event_type: FORWARD_AP, source: neuron_idx as u32, timestamp });
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
    dendrite_constants: &[i8],
    dendrite_last_events: &mut [u16],
    soma_potentials: &mut [i8],
    soma_thresholds: &[i8],
    synapse_offsets: &[u32],
    synapse_alphas: &mut [u8],
    synapse_last_events: &mut [u16],
    event_buf: *mut Event,
    event_tail: &AtomicU32,
    event_capacity: u32,
) {
    dendrite_last_events[dendrite_idx] = timestamp;

    let branch_constant = dendrite_constants[dendrite_idx];
    let soma_delta: i8 = branch_constant.max(1);
    soma_potentials[neuron_idx] = soma_potentials[neuron_idx].saturating_add(soma_delta);

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

    if soma_potentials[neuron_idx] >= soma_thresholds[neuron_idx] {
        queue.push(Event { event_type: SOMATIC_SPIKE, source: neuron_idx as u32, timestamp });
    }
}

