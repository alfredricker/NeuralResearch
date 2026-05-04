## Default Behavoir
```rust
pub enum NeuronType {
    Pyramid5,
    Sensory1,
    Lay1,
}

// hyperparams
pub struct NeuronDefaults {
    pub init_branch_threshold: u16,
    pub init_branch_constant: i8,
    pub init_soma_threshold: u16,
}

impl NeuronDefaults {
    pub fn new(ibt: u16, ibc: i8, ist: u16) -> Self {
        Self {
            init_branch_threshold: ibt,
            init_branch_constant: ibc,
            init_soma_threshold: ist,
        }
    }
}

impl NeuronType {
    pub fn defaults(&self) -> NeuronDefaults {
        match self {
            Self::Lay1 => { NeuronDefaults::new(20, 5, 60) },
            Self::Sensory1 => { NeuronDefaults::new(20, 5, 60) },
            Self::Pyramid5 => { NeuronDefaults::new(20, 5, 60) }
        }
    }
}
```

## Logic for Defaults
Branch threshold - `u16`, can have it be the midpoint `2^15`
Can set synaptic weights to midpoints `2^7`.