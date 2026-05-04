use crate::init::neuron::defaults::NeuronDefaults;
use crate::math::midpoint::MidPoint;

pub fn defaults() -> NeuronDefaults {
    NeuronDefaults::new(u16::mid(), 120, u8::mid())
}
