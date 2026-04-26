# Data Components
* **i8:** (-128, 127)
* **i16:** (-32768, 32767)
* **u32:** (0, 4294967295) or ~4 billion

## Structs and Memory Analysis
```rust
  struct Soma {
      membrane_potential: i16,  // 2 bytes
      threshold: i8,            // 1 byte
      resting_potential: i8,    // 1 byte
  }
  // actual size: 4 bytes — no padding
```
Can pack a dendritic address into a single u32.
| neuron_id: 20 bits | branch_id: 4 bits | dendrite_id: 8 bits |
```rust
  // the whole address fits in one u32
  // | neuron_id: 20 bits | branch_id: 4 bits | dendrite_id: 8 bits |
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
```

```rust
  struct Partition {
      neurons:           Vec<Neuron>,
      branches:          Vec<DendriteBranch>,
      synapses:          Vec<Synapse>,
      dendrite_activity: Vec<i16>,   // or Vec<i8> — one flat array for all dendrites
  }

  struct Neuron {
      soma:           Soma,   // 4 bytes
      synapse_start:  u32,    // 4 bytes
      branch_start:   u16,    // 2 bytes
      branch_count:   u8,     // 1 byte
      synapse_count:  u8,     // 1 byte
  }                           // 12 bytes total

  struct DendriteBranch {
      dendrite_start: u16,    // 2 bytes
      dendrite_count: u8,     // 1 byte
      threshold:      i8,     // 1 byte
      branch_constant:i8,     // 1 byte
      _pad:           u8,     // 1 byte explicit padding
  }                           // 6 bytes total
```

Neurons will have different memory requirements based on their type. As a first approximation, we can split neurons into 3 general types.
* Simple neurons are used for intermediate transmission and trivial calculations
* Interneurons perform inhibition or easy to intermediate calculation
* Pyramidal neurons perform complicated computation and learning operations

| Neuron Type | Branches | Dendrites / branch | Synapses | Calculation | Total |
| ----------- | -------- | ------------------ | -------- | ----------- | ----- |
| Simple | 4 | 32 | 20 | 12 + (4x6) + 128 + (20x8) | **~325 bytes** |
| Interneuron | 6 | 50 | 40 | 12 + (6x6) + 300 + (40x8) | **~660 bytes** |
| Pyramidal | 20 | 100 | 100 | 12 + (20x6) + 2000 + (100x8) | **~2.5Kb** |


Perhaps neurons should keep track of statistics such as average number of incoming spikes to adjust thresholds.