pub struct Soma {
    soma_potentials: Vec<i8>,
    soma_thresholds: Vec<i8>,
    soma_betas: Vec<u8>,
    soma_last_events: Vec<u16>,
    soma_lrs: Vec<i16>, // learning rates
    dendrite_offsets: Vec<u32>,
}