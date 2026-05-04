use crate::neuron::dendrite::Dendrite;
use crate::init::neuron::defaults::NeuronDefaults;

pub struct Branch {
    branch_constant: i8,
    threshold: u16,
    activity: u8,
    last_event: u32,
    dendrites: Vec<Dendrite>,
}

impl Branch {
    pub fn new(defaults: &NeuronDefaults) -> Self {
        Self {
            branch_constant: defaults.init_branch_constant,
            threshold: defaults.init_branch_threshold,
            activity: 0,
            last_event: 0,
            dendrites: Vec::new(),
        }
    }
}
