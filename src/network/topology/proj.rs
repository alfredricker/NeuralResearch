use crate::neuron::dendrite::{Compartment};
use crate::network::topology::conn::ConnRule;

pub struct Projection {
    pub target: &'static str, // name of the target population
    pub compartment: Compartment, // basal | apical
    pub rule: ConnRule,
}