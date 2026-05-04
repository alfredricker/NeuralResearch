use crate::math::midpoint::MidPoint;
use crate::init::neuron::defaults::NeuronDefaults;

pub enum NeuronType {
    Pyramid5,
    Sensory1,
    Lay1,
}

impl NeuronType {
    pub fn defaults(&self) -> NeuronDefaults {
        match self {
            Self::Lay1 => { NeuronDefaults::new(u16::mid(), 100, u8::mid()) },
            Self::Sensory1 => { NeuronDefaults::new(u16::mid(), 50, u8::mid()) },
            Self::Pyramid5 => { NeuronDefaults::new(u16::mid(), 100, u8::mid()) }
        }
    }
}
