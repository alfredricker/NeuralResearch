struct Neuron {
    learning_rate: i16 // eta, minimum is constants::MSLR. Won't go negative but I don't want to cast
}

// carries information of the spike to the synapses
pub struct Spike {
    pub learning_rate: i16,
    pub beta: u8,
    pub global_tick: u64, // the clock when spike
}

impl Spike {
    pub fn new(learning_rate: i16, beta: u8, global_tick: u64) -> Self {
        Self {
            learning_rate,
            beta,
            global_tick,
        }
    }
}