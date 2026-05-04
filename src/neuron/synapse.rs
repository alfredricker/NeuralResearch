use crate::math::decay::shift_decay_u8;
use crate::constants::{ALPHA_DECAY, H_BETA, H_ALPHA};
use crate::neuron::neuron::{Spike};

pub struct Synapse {
    pub weight: i8,
    pub x: u8, // position along the dendrite
    pub alpha: u8, // activity level (from presynaptic spikes)
    pub last_event: u16,
}

impl Synapse {
    pub fn new(x: u8) -> Self {
        Self {
            weight: 0,
            x,
            alpha: 0,
            last_event: 0,
        }
    }

    pub fn update(&mut self, spike: &Spike) {
        self.update_weight(spike);
        self.last_event = spike.global_tick as u16;
    }

    // runs when there is a somatic spike
    pub fn update_weight(&mut self, spike: &Spike) {
        self.decay_alpha(&spike.global_tick); 
        // only update if alpha > h_alpha
        if self.alpha <= H_ALPHA { return };

        let burst_term: i16 = (spike.beta as i16) - H_BETA; // don't have to worry about overflow -- max(beta) is 2^8 - 1
        // don't have to worry about overflow on the burst_term*self.alpha either. max is (2^6 - 5)*(2^8 - 1)
        let delta_weight: i16 = burst_term*(self.alpha as i16) / spike.learning_rate;
        // should already be in this range with appropriate learning rate checks.
        let dw_i8: i8 = delta_weight.clamp(-127, 128) as i8;
        self.weight = self.weight.saturating_add(dw_i8);
    }

    pub fn decay_alpha(&mut self, global_tick: &u64) {
        self.alpha = shift_decay_u8(self.alpha, self.elapsed(global_tick), ALPHA_DECAY)
    }

    // time elapsed since last event
    fn elapsed(&self, global_tick: &u64) -> u16 {
        (*global_tick as u16).wrapping_sub(self.last_event)
    }
}