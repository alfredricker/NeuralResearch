pub struct Dendrite {
    pub activity: u8,
    pub last_event: u32,
    pub branch_constant: i8,
    pub threshold: u8,
    pub activity: u8,
}

impl Dendrite {
    pub fn new(activity: u8) -> Self {
        Self {
            activity: activity,
            last_event: 0
        }
    }
}

#[derive(Copy, Clone)]
pub struct DendriteAddr(u32);

impl DendriteAddr {
    pub fn new(neuron_id: u32, branch_id: u8, dendrite_id: u8) -> Self {
        DendriteAddr((neuron_id << 12) | ((branch_id as u32) << 8) | (dendrite_id as u32))
    }
    pub fn neuron_id(self) -> usize { (self.0 >> 12) as usize }
    pub fn branch_id(self) -> usize { ((self.0 >> 8) & 0xF) as usize }
    pub fn dendrite_id(self) -> usize { (self.0 & 0xFF) as usize }
}