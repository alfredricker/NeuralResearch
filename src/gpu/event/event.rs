pub struct Event {
    pub event_type: u8,
    pub source: u32,    // neuron_idx for SOMATIC_SPIKE/FORWARD_AP, dendrite_idx for DENDRITIC_SPIKE
    pub timestamp: u16,
}