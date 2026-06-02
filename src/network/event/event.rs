pub const SOMATIC_SPIKE:   u8 = 0; // source = neuron_idx,   payload = burst (AP count)
pub const DENDRITIC_SPIKE: u8 = 1; // source = dendrite_idx, payload unused
pub const SOMA_SIGNAL:     u8 = 2; // source = neuron_idx,   payload = v_s (voltage delta to integrate)
pub const SYNAPSE_SIGNAL:  u8 = 3; // source = synapse_idx,  payload = burst (one queued AP delivery per target synapse)

pub struct Event {
    pub event_type: u8,  // not an enum because buffer is shared with gpu kernels
    pub source: u32,     // neuron_idx, except DENDRITIC_SPIKE (dendrite_idx) and SYNAPSE_SIGNAL (synapse_idx)
    pub timestamp: u16,
    pub payload: i16,    // event-specific scalar: burst count (spikes/APs) or v_s (SOMA_SIGNAL)
}

impl Event {
    // General constructor.
    pub fn with_payload(event_type: u8, source: u32, timestamp: u16, payload: i16) -> Self {
        Self { event_type, source, timestamp, payload }
    }

    // Payload-less event (DENDRITIC_SPIKE, queue init).
    pub fn spike(event_type: u8, source: u32, timestamp: u16) -> Self {
        Self::with_payload(event_type, source, timestamp, 0)
    }

    // SOMA_SIGNAL carrying a voltage delta from a dendritic spike or an apical plateau.
    pub fn soma_signal(neuron_idx: u32, timestamp: u16, v_s: i16) -> Self {
        Self::with_payload(SOMA_SIGNAL, neuron_idx, timestamp, v_s)
    }
}
