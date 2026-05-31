pub const SOMATIC_SPIKE:   u8 = 0; // source = neuron_idx
pub const DENDRITIC_SPIKE: u8 = 1; // source = dendrite_idx
pub const FORWARD_AP:      u8 = 2; // source = neuron_idx
pub const APICAL_FB:       u8 = 3; // source = neuron_idx

pub struct Event {
    pub event_type: u8,  // not an enum because buffer is shared with gpu kernels
    pub source: u32,     // neuron_idx for SOMATIC_SPIKE/FORWARD_AP/APICAL_FB, dendrite_idx for DENDRITIC_SPIKE
    pub timestamp: u16,
}