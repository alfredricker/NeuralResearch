# 3. Data model

This chapter is the concrete realization of [chapter 2](02-architecture.md)'s
SoA + offset-array principles. It documents the arrays the code declares today,
the index conventions that connect them, and where the layout is still
incomplete.

## 3.1 The four component groups

`Network` (`src/network/mod.rs`) owns four SoA structs plus the per-population
neuron ranges:

```rust
pub struct Network {
    synapses:  Synapse,
    dendrites: Dendrite,
    somas:     Soma,
    axons:     Axon,                // now a wired-in field, not a loose parameter
    ranges:    Vec<Range<u32>>,     // global neuron range of each population, in add() order
}
```

Each struct is a bundle of parallel `Vec`s — one logical table, column-major.
`ranges` is what `Network::population_range(id)` returns so the `io/` boundary can
bind input/effector maps to concrete global indices
([chapter 11](11-io-boundary.md)).

### Soma — one entry per neuron (`src/neuron/soma.rs`)

```rust
pub struct Soma {
    soma_potentials:  Vec<i8>,    // membrane potential V; reset to SOMA_V_RESET (−32) on spike
    soma_thresholds:  Vec<i8>,    // fire when potential ≥ threshold
    soma_betas:       Vec<u8>,    // burst counter (chapter 1.5); capped at 63
    soma_last_events: Vec<u16>,   // timestamp of last soma event (drives both potential & beta decay)
    soma_lrs:         Vec<i16>,   // per-neuron learning rate (divisor)
    dendrite_offsets: Vec<u32>,   // CSR: first dendrite index of each neuron (len = n+1)
}
```

Both the potential (`SOMATIC_DECAY`) and `beta` (1 per `T_BETA` ticks) leak
lazily from `soma_last_events`; the whole soma state machine lives in
`update_soma_potential` ([chapter 6.5](06-learning-dynamics.md)).

### Dendrite — one entry per dendrite (`src/neuron/dendrite.rs`)

```rust
pub struct Dendrite {
    dendrite_activities:  Vec<u16>, // branch voltage V_B (basal AND apical integrate here); leaks, resets to 0 on a basal spike
    dendrite_last_events: Vec<u16>, // timestamp of last dendritic event (drives the V_B leak)
    dendrite_constants:   Vec<i8>,  // basal branch constant: >0 proximal, ≤0 distal
    dendrite_thresholds:  Vec<u16>, // basal: hard spike threshold; apical: θ_B (plateau half-activation)
    synapse_offsets:      Vec<u32>, // CSR: first synapse index of each dendrite (len = d+1); analytic d*S
    live_synapse_counts:  Vec<u8>,  // number of bound synapse SLOTS, packed at the front of each block (chapter 7.4)
    dendrite_to_neuron:   Vec<u32>, // reverse map d → owning neuron (stored for the event loop)
    dendrite_is_apical:   Vec<u8>,  // 0 = basal, 1 = apical; u8 not bool so the buffer can be GPU-shared
}
```

Two runtime distinctions live here. The **sign of `dendrite_constants[d]`**
selects proximal vs. distal *propagation* behavior (proximal scales onto the
soma; distal is attenuated — [chapter 6.4](06-learning-dynamics.md)). The
**`dendrite_is_apical[d]` flag** selects the *integration* behavior — basal hard
threshold vs. apical graded plateau ([chapter 6.2](06-learning-dynamics.md)) — and
is what lets a single axon drive whichever compartment its target slot belongs to.
The `Compartment { Apical, Basal }` enum in the same file is the *build-time* tag
the allocator and connection resolver use ([chapter 7](07-network-construction.md)).
`live_synapse_counts` is the fixed-slot live count ([chapter 7.4](07-network-construction.md));
`dendrite_to_neuron` is the O(1) reverse lookup the `DENDRITIC_SPIKE` handler needs.

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
pub struct Axon {
    pub axon_targets: Vec<u32>,   // flat list of target *synapse* indices
    pub axon_offsets: Vec<u32>,   // CSR: first target index for each source neuron (len = n+1)
}
```

`axon_targets[axon_offsets[n] .. axon_offsets[n+1]]` is the set of downstream
synapses that neuron `n` drives when it fires. On a `SOMATIC_SPIKE` the loop fans
one `SYNAPSE_SIGNAL` out to each of these targets
([chapter 5.3](05-event-system.md)). Targets are **absolute synapse indices**,
which is why structural plasticity (deleting or moving a synapse) is delicate —
see [chapter 7.5](07-network-construction.md) and
[chapter 9](09-gaps-and-open-questions.md).

`Axon` is now a `pub` field of `Network`, populated by `build_network` (phase 4,
[chapter 7.4](07-network-construction.md)): connections are resolved to
`(source_neuron, target_synapse)` pairs and inverted into this CSR.

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

> **See the diagrams.** [`resources/index-relationships.md`](resources/index-relationships.md)
> renders this hierarchy, a concrete worked example, and the forward-AP path —
> all as Mermaid diagrams. A key consequence to internalize: the global synapse
> array is ordered by `(neuron, dendrite, x)`, so an absolute synapse index
> implicitly identifies its owning dendrite and neuron — which is exactly why
> `axon_targets` can store nothing but a synapse index.

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
last_event `(u16) = 1+1+1+2 ≈ 4` (the `u16` may force 2-byte alignment depending
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

> **Fixed-slot over-provisioning.** The table counts *live* synapses. The
> allocator reserves `S = SYNAPSE_SLOTS_PER_DENDRITE = 255` slots per dendrite
> regardless of live count ([chapter 7.3](07-network-construction.md)), so the
> *allocated* synapse arrays are far larger — a 16-live-synapse dendrite still
> occupies 255 slots. With `S = 255` the dead tail dominates; lower `S` in
> `constants.rs` if memory matters before scaling up. The live-count figures above
> are the working-set size that actually participates in the dynamics.

---

Next: [chapter 4 — Math primitives](04-math-primitives.md), the leaf functions
that operate on these arrays.
