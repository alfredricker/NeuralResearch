pub struct Synapse {
    pub weight: i8,
    pub x: u8, // position along the dendrite
    pub: alpha: u8, // activity level (from presynaptic spikes)
    pub: beta: u8, // BPaP level (from soma spikes)
    pub: last_event: u16,
}

impl Synapse {
    pub fn new(x: u8) -> Self {
        Self {
            weight: 0,
            x,
            alpha: 0,
            beta: 0,
            last_event: 0,
        }
    }


}