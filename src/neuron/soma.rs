pub struct Soma {
    pub soma_potentials: Vec<i8>,
    pub soma_thresholds: Vec<i8>,
    pub soma_betas: Vec<u8>,
    pub soma_last_events: Vec<u16>,
    pub soma_lrs: Vec<i16>, // learning rates
    pub dendrite_offsets: Vec<u32>,
}
