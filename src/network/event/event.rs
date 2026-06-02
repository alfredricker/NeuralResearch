pub const SOMATIC_SPIKE:   u8 = 0; // source = neuron_idx,   payload unused
pub const DENDRITIC_SPIKE: u8 = 1; // source = dendrite_idx, payload unused
pub const FORWARD_AP:      u8 = 2; // source = neuron_idx,   payload unused (fans out to all axon targets, basal AND apical)
pub const SOMA_SIGNAL:     u8 = 3; // source = neuron_idx,   payload = v_s (voltage delta to integrate)

pub struct Event {
    pub event_type: u8,  // not an enum because buffer is shared with gpu kernels
    pub source: u32,     // neuron_idx, except DENDRITIC_SPIKE where it is dendrite_idx
    pub timestamp: u16,
    pub payload: i16,    // event-specific scalar; only SOMA_SIGNAL uses it (the voltage delta v_s)
}

impl Event {
    // Spike/AP event — no payload. Keeps the payload-less call sites uncluttered.
    pub fn spike(event_type: u8, source: u32, timestamp: u16) -> Self {
        Self { event_type, source, timestamp, payload: 0 }
    }

    // SOMA_SIGNAL carrying a voltage delta from a dendritic spike or an apical plateau.
    pub fn soma_signal(neuron_idx: u32, timestamp: u16, v_s: i16) -> Self {
        Self { event_type: SOMA_SIGNAL, source: neuron_idx, timestamp, payload: v_s }
    }
}
