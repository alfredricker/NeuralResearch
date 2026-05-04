use crate::constants::{T_BETA};

pub struct Soma {
    pub membrane_potential: i8,  // 1 bytes
    pub threshold: i8,            // 1 byte
    pub last_event: u16,
    pub beta: u8 // tracks events in T_BETA timeframe.
}

impl Soma {
    pub fn update_beta(&mut self) {
        // @TODO: confirm that this update mechanism doesn't lose information
        let decrements = (self.last_event / T_BETA).min(15) as u8;
        self.beta = self.beta.saturating_sub(decrements).saturating_add(1);
        // prevent exploding weight updates and i16 overflows in delta_weight -- cap at 2^6 - 1
        if self.beta > 63 { self.beta = 63 }; // there might be a more efficient way to do this in this func
    }
}