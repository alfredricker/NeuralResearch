use crate::init::neuron::defaults::NeuronDefaults;
use crate::math::midpoint::MidPoint;

pub fn defaults() -> NeuronDefaults {
    NeuronDefaults::new(u16::mid() / 2, 80, u8::mid() / 2)
}
