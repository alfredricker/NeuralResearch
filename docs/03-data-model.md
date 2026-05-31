# 3. Data model

This chapter is the concrete realization of [chapter 2](02-architecture.md)'s
SoA + offset-array principles. It documents the arrays the code declares today,
the index conventions that connect them, and where the layout is still
incomplete.

## 3.1 The four component groups

`Network` (`src/network/mod.rs`) owns three structs; a fourth (`Axon`) exists but
is not yet wired in:

```rust
pub struct Network {
    dendrites: Dendrite,
    somas:     Soma,
    synapses:  Synapse,
}
```

Each struct is a bundle of parallel `Vec`s — one logical table, column-major.

### Soma — one entry per neuron (`src/neuron/soma.rs`)

```rust
pub struct Soma {
    soma_potentials:  Vec<i8>,    // membrane potential V; reset to 0 on spike
    soma_thresholds:  Vec<i8>,    // fire when potential ≥ threshold
    soma_betas:       Vec<u8>,    // burst counter (chapter 1.5); capped at 63
    soma_last_events: Vec<u16>,   // timestamp of last somatic spike (for beta decay)
    soma_lrs:         Vec<i16>,   // per-neuron learning rate (divisor)
    dendrite_offsets: Vec<u32>,   // CSR: first dendrite index of each neuron (len = n+1)
}
```

### Dendrite — one entry per dendrite (`src/neuron/dendrite.rs`)

```rust
pub struct Dendrite {
    dendrite_activities: Vec<u16>,  // accumulated depolarization; reset to 0 on dendritic spike
    dendrite_last_events: Vec<u16>, // timestamp of last dendritic event
    dendrite_constants:  Vec<i8>,   // branch constant: >0 proximal/basal, ≤0 distal/apical
    dendrite_thresholds: Vec<u16>,  // fire when activity ≥ threshold
    synapse_offsets:     Vec<u32>,  // CSR: first synapse index of each dendrite (len = d+1)
}
```

The sign of `dendrite_constants[d]` is how the model distinguishes compartment
behavior at runtime (proximal scales onto the soma; distal is attenuated — see
[chapter 6](06-learning-dynamics.md)). The `Compartment { Apical, Basal }` enum
in the same file is the *build-time* tag for the same distinction.

### Synapse — one entry per synapse (`src/neuron/synapse.rs`)

```rust
pub struct Synapse {
    synapse_weights:     Vec<i8>,   // w_i, the learned weight
    synapse_x:           Vec<u8>,   // position along dendrite — MUST be sorted ascending per dendrite, unique
    synapse_alphas:      Vec<u8>,   // eligibility trace alpha (chapter 1.3)
    synapse_last_events: Vec<u16>,  // timestamp of last synaptic event (for alpha decay)
}
```

The **sorted-`x`-per-dendrite invariant** is load-bearing: the gamma loop in
[chapter 6](06-learning-dynamics.md) walks synapses in `x` order and assumes
ascending. [Chapter 7](07-network-construction.md) explains why this is
guaranteed at allocation time rather than checked at runtime.

### Axon — inter-neuron connectivity (`src/neuron/axon.rs`)

```rust
struct Axon {
    axon_targets: Vec<u32>,   // flat list of target *synapse* indices
    axon_offsets: Vec<u32>,   // CSR: first target index for each source neuron (len = n+1)
}
```

`axon_targets[axon_offsets[n] .. axon_offsets[n+1]]` is the set of downstream
synapses that neuron `n` drives when it fires a forward AP. Note targets are
**absolute synapse indices**, which is why structural plasticity (deleting or
moving a synapse) is delicate — see [chapter 7](07-network-construction.md) and
[chapter 9](09-gaps-and-open-questions.md).

> **Gap.** `Axon` is `struct` (not `pub`), is not a field of `Network`, and is
> not populated by any builder. The event loop receives `axon_targets` /
> `axon_offsets` as bare parameters, so nothing constructs them yet.

## 3.2 The index hierarchy

Three index spaces, linked by the offset arrays from [chapter 2](02-architecture.md):

```
neuron index  n ──dendrite_offsets──▶  dendrite index d ──synapse_offsets──▶  synapse index s
              n+1 sentinel                              d+1 sentinel
```

Resolving ranges (implemented in `src/network/event/slice.rs`):

```rust
// all synapses owned by neuron n — spans every one of its dendrites
fn neuron_synapse_range(n, dendrite_offsets, synapse_offsets) -> (s_start, s_end) {
    let d_start = dendrite_offsets[n];
    let d_end   = dendrite_offsets[n + 1];
    (synapse_offsets[d_start], synapse_offsets[d_end])
}

// just the synapses on dendrite d
fn dendrite_synapse_range(d, synapse_offsets) -> (s_start, s_end) {
    (synapse_offsets[d], synapse_offsets[d + 1])
}
```

These two functions are how a handler gets the exact slice it is allowed to
touch — central to the event loop in [chapter 5](05-event-system.md).

There is also a `dendrite_to_neuron: Vec<u32>` reverse map (a parameter to the
event loop) so that a dendritic spike, which knows only its dendrite index, can
find its soma in O(1) without searching `dendrite_offsets`.

## 3.3 The packed dendrite address (planned)

`docs/code/data.md` proposed encoding a full dendrite address in one `u32`:

```
| neuron_id: 20 bits | branch_id: 4 bits | dendrite_id: 8 bits |
```

```rust
pub struct DendriteAddr(u32);
impl DendriteAddr {
    pub fn new(neuron_id: u32, branch_id: u8, dendrite_id: u8) -> Self {
        DendriteAddr((neuron_id << 12) | ((branch_id as u32) << 8) | (dendrite_id as u32))
    }
    pub fn neuron_id(self)   -> usize { (self.0 >> 12) as usize }
    pub fn branch_id(self)   -> usize { ((self.0 >> 8) & 0xF) as usize }
    pub fn dendrite_id(self) -> usize { (self.0 & 0xFF) as usize }
}
```

This caps a neuron at 16 branches × 256 dendrites/branch and the network at ~1M
neurons. It is **not yet in the code** — the current model uses a flat
`dendrite_to_neuron` map instead. The branch level is, for now, purely a *count*
used during allocation (`n_basal_dendrites × dendrites_per_branch` total
dendrites per neuron); it is not a separate addressable tier at runtime.

## 3.4 Memory budget (first-order estimates)

Per-synapse runtime state is 4 bytes: `weight (i8) + x (u8) + alpha (u8) +
last_event (u16)` = 1+1+1+2 ≈ 4 (the `u16` may force 2-byte alignment depending
on how columns are stored; as separate `Vec`s there is no inter-column padding).

For the `visual_mnist`-style neuron (6 basal × 8 dendrites/branch × 16 synapses =
**768 synapses/neuron**, 48 dendrites/neuron):

| Hidden neurons `N` | Synapses | Synapse state | Notes |
| ------------------ | -------- | ------------- | ----- |
| 100 | 76 800 | ~300 KB | trivially cache-resident |
| 200 | 153 600 | ~600 KB | recommended MNIST starting size |
| 1000 | 768 000 | ~3 MB | cache behavior starts to matter |

Earlier sizing sketches (`docs/code/data.md`) for three coarse neuron classes
(Simple ~325 B, Interneuron ~660 B, Pyramidal ~2.5 KB per neuron) remain useful
as ballpark figures for heterogeneous networks.

---

Next: [chapter 4 — Math primitives](04-math-primitives.md), the leaf functions
that operate on these arrays.
