use crate::math::midpoint::MidPoint;

pub enum NeuronTypes {
    Pyramid5,
    Sensory1,
    Lay1,
}

impl NeuronTypes {
    pub fn defaults(&self) -> NeuronDefaults {
        match self {
            Self::Lay1 => { NeuronDefaults::new(u16::mid(), 100, u8::mid()) },
            Self::Sensory1 => { NeuronDefaults::new(u16::mid(), 50, u8::mid()) },
            Self::Pyramid5 => { NeuronDefaults::new(u16::mid(), 100, u8::mid()) }
        }
    }
}

// hyperparams
pub struct NeuronDefaults {
    pub init_branch_threshold: u16,
    pub init_branch_constant: i8,
    pub init_soma_threshold: u8,
}

impl NeuronDefaults {
    pub fn new(ibt: u16, ibc: i8, ist: u8) -> Self {
        Self {
            init_branch_threshold: ibt,
            init_branch_constant: ibc,
            init_soma_threshold: ist,
        }
    }
}