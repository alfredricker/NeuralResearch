pub const SOMATIC_SPIKE:   u8 = 0; // source = neuron_idx
pub const DENDRITIC_SPIKE: u8 = 1; // source = dendrite_idx
pub const FORWARD_AP:      u8 = 2; // source = neuron_idx

pub struct Event {
    pub event_type: u8,
    pub source: u32,    // neuron_idx for SOMATIC_SPIKE/FORWARD_AP, dendrite_idx for DENDRITIC_SPIKE
    pub timestamp: u16,
}