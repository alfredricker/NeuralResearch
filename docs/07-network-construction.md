# 7. Network construction

Everything so far assumed the SoA arrays already exist. This chapter is about
*building* them: from a declarative description of neuron types and connectivity
down to the flat `Vec`s [chapter 6](06-learning-dynamics.md) operates on. It is
the project's **critical path** — most of it is designed but not yet implemented.

The pipeline has two clean halves:

```
DECLARE                                        COMPILE
NeuronConfig ─▶ Population ─┐                  ┌─▶ allocator ─▶ Soma/Dendrite/Synapse arrays
                            ├─▶ NetworkBuilder ─┤
Connection (ConnRule) ─────┘                  └─▶ resolver  ─▶ Axon CSR (axon_targets/offsets)
```

The declarative half is typed and ~complete. The compile half is the gap.

## 7.1 The declarative front-end

### `NeuronConfig` (`src/neuron/config.rs`) — a neuron *type*

Pure data plus samplers ([chapter 4.2](04-math-primitives.md)). It describes
shape and distributions, not any particular neuron:

```rust
pub struct NeuronConfig {
    pub name: &'static str,
    // topology
    pub n_basal_dendrites:  u8,
    pub n_apical_dendrites:  Option<u8>,        // None ⇒ no apical compartment
    // random parameters — sampled per neuron at allocation
    pub synapse_x_sampler:     SamplerU8,       // position x along dendrite
    pub dendrites_per_branch:  SamplerU8,
    pub synapses_per_dendrite: SamplerU8,
    // soma
    pub soma_threshold: i8,
    // basal / apical dendrites
    pub basal_dendrite_threshold: u16,
    pub basal_dendrite_constant:  SamplerI8,
    pub apical_dendrite_threshold: Option<u16>,
    pub apical_dendrite_constant:  Option<SamplerI8>,
    // learning
    pub learning_rate: i16,
}
```

### `Population` (`src/neuron/population.rs`) — a *count* of one type

```rust
pub struct Population { pub name: &'static str, pub config: &'static NeuronConfig, pub size: u32 }
```

A population is a pure node — it stores no topology. Identity is a generated
`u32` id (allocation order), which is what lets motifs be duplicated
programmatically (a motif = a function returning its port ids; duplication = a
`for` loop).

### `Connection` / `ConnRule` (`src/network/topology/conn.rs`) — wiring intent

```rust
pub struct Connection { pub from: u32, pub to: u32, pub compartment: Compartment, pub rule: ConnRule }

pub enum ConnRule {
    DenseRandom { p: f32 },     // each possible edge made with probability p
    FixedInDegree { k: u32 },   // each target gets exactly k incoming edges
    ReceptiveField { radius: u32 },
    Topographic { patch: u8 },
    OneToOne,                   // i→i, requires equal population sizes
}
```

`Compartment { Apical, Basal }` ([chapter 3.1](03-data-model.md)) chooses which
dendrite class the connection lands on.

### `NetworkBuilder` (`src/network/build.rs`) — the assembly API

```rust
impl NetworkBuilder {
    pub fn add(&mut self, config: &'static NeuronConfig, size: u32) -> u32 { /* returns pop id */ }
    pub fn connect(&mut self, from: u32, to: u32, c: Compartment, rule: ConnRule) { /* records a Connection */ }
}
```

`add` and `connect` just accumulate `Vec<Population>` and `Vec<Connection>`. They
do no allocation — the builder is a recipe, not the dish.

## 7.2 The allocator (designed, not built)

This is the keystone. A function — sketched as `build_layer(config, n_neurons,
rng) -> ...` — must materialize, for one population, every array
[chapter 3](03-data-model.md) declares, in offset-consistent order:

- **Soma** (len `n`): potentials/betas/last_events = 0; thresholds =
  `config.soma_threshold`; lrs = `config.learning_rate`; `dendrite_offsets` with
  stride `n_basal_dendrites × dendrites_per_branch`.
- **Dendrite** (len `n × dends_per_neuron`): activities/last_events = 0;
  thresholds = `basal_dendrite_threshold`; constants sampled from
  `basal_dendrite_constant`; `synapse_offsets` with stride
  `synapses_per_dendrite`; plus the `dendrite_to_neuron` reverse map.
- **Synapse** (len `n × dends × syns`): weights = small uniform (e.g. `U(−8,8)`);
  `x` sampled from `synapse_x_sampler` **and sorted ascending within each
  dendrite**; alphas/last_events = 0.

The sort step is non-negotiable: it is what makes the gamma loop's invariant
([chapter 6.2](06-learning-dynamics.md)) hold. Build each dendrite's `x` values,
sort, then write.

> **Gap.** `Network::build` (`src/network/mod.rs`) is an empty stub that does not
> even return a `Network`; `ConnRule::apply` is an empty body. No allocator
> exists. This is the top of the priority list in
> [chapter 9](09-gaps-and-open-questions.md).

## 7.3 Fixed synapse slots — the layout decision

This is the most consequential **specific design choice** for the allocator, and
it follows directly from [chapter 2](02-architecture.md)'s offset-array and
GPU-coalescing principles.

**Decision: synapses are pre-allocated as fixed-size slots per dendrite, and
connections bind to existing slots rather than creating synapses.**

Because a neuron type fixes `synapses_per_dendrite`, every dendrite gets its full
pool of *empty slots* at allocation. The immediate payoff:

```
synapse_offsets[d] = d * stride        // ANALYTIC — no prefix-sum pass, no stored array needed
```

The offsets become pure arithmetic ([chapter 2.2](02-architecture.md)), and
`synapse_to_dendrite` ([chapter 5.3](05-event-system.md)) collapses from a binary
search to `s / stride`. Connections (phase 2) then *bind a presynaptic source to
an existing empty slot* via a per-target cursor and emit `(source_neuron,
target_synapse)` pairs, grouped by source into the axon CSR. They create nothing
and **never reorder**, so the sorted-`x` invariant set up in §7.2 is structurally
preserved.

The rejected alternative — materializing synapses in connection order — would
force a post-hoc co-permutation of weights/`x`/alphas/last_events plus a counting
pass to size the arrays. Fragile and slower.

## 7.4 Iterating only active synapses — `live_count`

[Chapter 6.2](06-learning-dynamics.md)'s gamma loop, and dendritic integration
generally, should iterate only *bound, live* synapses — not empty or tombstoned
slots. With fixed slots there are two orderings that appear to conflict:

1. The gamma loop needs synapses **sorted ascending by `x`**.
2. We want to iterate only **live** synapses, skipping dead ones.

They only conflict if dead slots can sit *between* live ones. The resolution is
one invariant: **dead slots always live at the tail of the block.** Then a single
layout satisfies both:

```
[ base ............................. base + stride )
  └──── live, sorted by x ────┘└──── dead ────┘
        [base, base + live_count)   [base + live_count, base + stride)
```

Add one array — `live_count: Vec<u8>` (one per dendrite) — and keep the live
synapses packed at the front, sorted by `x`. The gamma loop then bounds itself by
`base + live_count`, **not** by the full stride and **not** by `s_idx + live_count`
(the count's origin is the dendrite base, not the firing synapse):

```rust
let base     = synapse_offsets[d] as usize;          // = d * stride, analytic
let live_end = base + live_count[d] as usize;
for j in (s_idx + 1)..live_end { /* ... */ }         // only live, more-distal synapses
```

Why this beats the alternatives **for the GPU specifically**:

| Approach | Hot-loop cost | GPU behavior |
| -------- | ------------- | ------------ |
| **Packed + `live_count`** (recommended) | iterate `live_count` contiguous slots | coalesced reads; loop bound is *uniform across the warp* ⇒ no divergence |
| Tombstone + `if dead { skip }` | iterate full stride, branch per slot | warp divergence; reads dead data; wasted lanes |
| Per-dendrite bitmask | popcount/ballot to find live | uniform too, but you must still map set-bits → compacted order to do the *ordered* suffix sum — i.e. re-derive the packing every kernel call |

The gamma sum is an **ordered suffix reduction** (`Σ` over `x_j > x_i`). Packing
front-loads that compaction once, at structural-change time, so the per-spike hot
loop stays trivial. Since reads (every spike) vastly outnumber structural edits,
that is the right place to pay. The bitmask only wins if structural churn
dominated reads, which it will not in a training loop.

> **Gap / change-on-adoption.** `live_count` does not exist yet, and
> `update_dendrite_activity` ([chapter 6.2](06-learning-dynamics.md)) currently
> bounds by slice length — correct only because [chapter 5](05-event-system.md)
> pre-trims the slice to exactly the dendrite's synapses. Adopting fixed slots
> with a padded tail means the slice becomes the *full stride block*, and the
> loop must switch to the `live_end` bound above. This is the one concrete
> code change the slot model forces on the hot path.

## 7.5 Structural plasticity — keeping it non-fragile

If synapses can be added/removed during learning, the fixed-slot layout makes it
cheap *provided one rule is never broken*: **never rewrite `synapse_offsets`
in-place at runtime.** Two things make naive in-place deletion a nightmare —
fixed-stride offsets would all have to shift, and `axon_targets` store *absolute*
synapse indices, so moving a synapse silently invalidates every incoming axon
pointer. So separate *liveness* from *layout*:

1. **Tombstone (runtime, O(1)).** Don't remove — mark dead (sentinel weight, dead
   flag, or `x = 255`) and decrement nothing structural. With the packing model
   (§7.4), "delete" = shift the live tail left by one and `live_count -= 1`
   (preserves sort), O(stride) over a ~16-element block.
2. **Migrate within the block (runtime, O(stride)).** Insert into a dead tail
   slot, `live_count += 1`, re-sort the ≤16-element live prefix by `x`. All edits
   stay inside one fixed block; offsets never move.
3. **Compaction (offline, between epochs, one linear pass).** When tombstones
   accumulate, rebuild: walk dendrites, copy live synapses into fresh arrays,
   recompute offsets, build an `old_index → new_index` remap, and apply it to
   `axon_targets` in one scan. This is the GC model — non-fragile precisely
   because it rebuilds with an explicit remap instead of mutating in place.

The one genuinely global concern — the absolute-index remap of `axon_targets`
when a synapse's index changes — is confined to the deliberate compaction pass.
A within-block delete that shifts the tail also changes absolute indices; the
clean options are (a) make `axon_targets` address a *logical* `(dendrite, slot)`,
or (b) defer the fixup to compaction and tolerate a brief stale window. If
plasticity ever needs *more* synapses than the stride allows, over-provision each
block with headroom (`stride + slack`) at allocation; compaction reclaims it.

> **Status.** Entirely a design for now — there is no structural-plasticity code.
> It is documented here because it constrains the allocator's slot/stride choices
> (§7.3) that *are* about to be built.

---

Next: [chapter 8 — The MNIST pipeline](08-mnist-pipeline.md), the first concrete
network these pieces assemble into.
