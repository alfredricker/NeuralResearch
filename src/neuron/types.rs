pub enum NeuronTypes {
    PYRAMID5,
    SENSORY1,
    LAY1,
}

// hyperparams
pub struct NeuronDefaults {
    pub init_branch_threshold: u8,
    pub init_branch_constant: u8,
    pub init_soma_threshold: u8,
}

impl NeuronDefaults {
    pub fn new(ibt: u8, ibc: u8, ist: u8) -> Self {
        Self {
            init_branch_threshold: ibt,
            init_branch_constant: ibc,
            init_soma_threshold: ist,
        }
    }
}

impl NeuronTypes {
    pub fn defaults(&self) -> NeuronDefaults {
        match self {
            self::LAY1 => { NeuronDefaults::new(20, 5, 60) },
            self::SENSORY1 => { NeuronDefaults::new(20, 5, 60) },
            self::PYRAMID5 => { NeuronDefaults::new(20, 5, 60) }
        }
    }
}