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