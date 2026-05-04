use crate::neuron::synapse::Synapse;

// dendrite owns the synapses. This is essentially a dendritic branch.
// it has its own spiking dynamics, meaning a single neuron acts as a two layer NN
pub struct Dendrite {
    pub activity: u16,
    pub last_event: u16,
    pub branch_constant: i8,
    pub threshold: u16,
    pub synapses: Vec<Synapse>,
}

impl Dendrite {
    pub fn new(threshold: u16, branch_constant: i8) -> Self {
        Self {
            activity: 0,
            last_event: 0,
            branch_constant,
            threshold,
            synapses: Vec::new(),
        }
    }
}

#[derive(Copy, Clone)]
pub struct DendriteAddr(u32);

// Layout: [neuron_id: 20 bits | dendrite_id: 12 bits]
// Max: 1,048,576 neurons × 4,096 dendrites each
impl DendriteAddr {
    pub fn new(neuron_id: u32, dendrite_id: u16) -> Self {
        DendriteAddr((neuron_id << 12) | (dendrite_id as u32 & 0xFFF))
    }
    pub fn neuron_id(self) -> usize { (self.0 >> 12) as usize }
    pub fn dendrite_id(self) -> usize { (self.0 & 0xFFF) as usize }
}
