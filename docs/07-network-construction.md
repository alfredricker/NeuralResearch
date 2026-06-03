# 7. Network construction

Everything so far assumed the SoA arrays already exist. This chapter is about
*building* them: from a declarative description of neuron types and connectivity
down to the flat `Vec`s [chapter 6](06-learning-dynamics.md) operates on.

> **Status change.** Earlier revisions of this book called construction "the
> critical gap — designed but not built." That is no longer true. The allocator
> (`Population::generate_neurons`), the orchestrator (`build_network`), and the
> connection resolver (`ConnRule::apply`) are all **implemented and unit-tested**.
> This chapter now documents the code as it stands; the remaining gaps are
> narrower and listed in [chapter 9](09-gaps-and-open-questions.md).

The pipeline has two clean halves:

```
DECLARE                                          COMPILE
NeuronConfig ─▶ Population ─┐                     ┌─▶ generate_neurons ─▶ Soma/Dendrite/Synapse arrays
                           ├─▶ NetworkBuilder ─▶ build_network ─┤
Connection (ConnRule) ─────┘                     └─▶ ConnRule::apply ─▶ Axon CSR (axon_targets/offsets)
```

## 7.1 The declarative front-end

### `NeuronConfig` (`src/neuron/config.rs`) — a neuron *type*

Pure data plus samplers ([chapter 4.2](04-math-primitives.md)) — shape and
distributions, not any particular neuron:

```rust
pub struct NeuronConfig {
    pub name: &'static str,
    pub n_basal_dendrites:  u8,
    pub n_apical_dendrites:  Option<u8>,        // None ⇒ no apical compartment
    pub synapse_x_sampler:     SamplerU8,       // position x along dendrite
    pub dendrites_per_branch:  SamplerU8,
    pub synapses_per_dendrite: SamplerU8,
    pub soma_threshold: i8,
    pub basal_dendrite_threshold: u16,
    pub basal_dendrite_constant:  SamplerI8,
    pub apical_dendrite_threshold: Option<u16>, // Some required iff n_apical_dendrites is Some
    pub apical_dendrite_constant:  Option<SamplerI8>,
    pub learning_rate: i16,
}
```

The concrete configs that exist today live in `src/io/` — `input_config()`
(zero-dendrite axon sources) and `output_config()` (low-threshold integrators) —
documented in [chapter 11](11-io-boundary.md). A dedicated *hidden-layer* config
for MNIST is still to be written ([chapter 8](08-mnist-pipeline.md)).

### `Population` (`src/neuron/population.rs`) — a *count* of one type

```rust
pub struct Population { pub name: &'static str, pub config: &'static NeuronConfig, pub size: u32 }
```

A population is a pure node — it stores no topology. Its identity is its index in
the builder's `populations` vector (allocation order), which is what
`NetworkBuilder::add` returns and what `Network::population_range`
([chapter 11](11-io-boundary.md)) later resolves to a global neuron range.

### `Connection` / `ConnRule` (`src/network/topology/conn.rs`) — wiring intent

```rust
pub struct Connection { pub from: u32, pub to: u32, pub compartment: Compartment, pub rule: ConnRule }

pub enum ConnRule {
    DenseRandom { p: f32 },     // each possible edge made with probability p
    FixedInDegree { k: u32 },   // each target gets exactly k distinct incoming edges
    ReceptiveField { radius: u32 },
    Topographic { patch: u8 },  // TODO — currently returns InvalidRule
    OneToOne,                   // i→i, requires equal population sizes
}
```

`Compartment { Apical, Basal }` ([chapter 3.1](03-data-model.md)) chooses which
dendrite class the connection lands on. `ConnRule::apply(src, dst, rng, edges)`
is **implemented** for `DenseRandom`, `FixedInDegree`, `OneToOne`, and
`ReceptiveField`; it pushes `(src_neuron, dst_neuron)` pairs into `edges`.
`Topographic` is still a stub returning `ConnError::InvalidRule`. The spatial
rules assume both populations are laid out on a `√N × √N` grid with position =
index (so MNIST's 784 pixels are a 28×28 sheet).

### `NetworkBuilder` (`src/network/build.rs`) — the assembly API

```rust
impl NetworkBuilder {
    pub fn add(&mut self, config: &'static NeuronConfig, size: u32) -> u32 { /* push Population, return id */ }
    pub fn connect(&mut self, from: u32, to: u32, c: Compartment, rule: ConnRule) { /* push Connection */ }
}
```

`add` and `connect` just accumulate `Vec<Population>` and `Vec<Connection>` — the
builder is a recipe, not the dish.

## 7.2 The allocator — `Population::generate_neurons`

Each population materializes its own slice of every SoA array, *appending* to the
growing buffers so populations lay down back-to-back in `add` order. The geometry
is fixed per population so offsets stay analytic:

- **`D` dendrites per neuron** = `n_basal_dendrites × dendrites_per_branch`
  (+ apical if configured). `dendrites_per_branch` is sampled **once per
  population**, so every neuron in it shares the same `D` and
  `dendrite_offsets[n] = dendrite_base + n·D` is exact.
- **Soma arrays** (len `size`): potentials/betas/last_events = 0; thresholds =
  `config.soma_threshold`; lrs = `config.learning_rate`; `dendrite_offsets`
  pointing at each neuron's first dendrite.
- **Dendrite arrays** (`generate_dendrites`): basal dendrites come first, then
  apical (the `is_apical = local_d % D ≥ basal_ds` test). Each gets
  `synapse_offsets[d] = d·S` (the analytic fixed stride, §7.3), a sampled
  `live_synapse_count`, a `dendrite_to_neuron` back-pointer, and
  compartment-tagged `constant` / `threshold` / `is_apical`. Configuring apical
  dendrites without `apical_dendrite_{threshold,constant}` panics — a deliberate
  loud failure.
- **Synapse arrays** (`generate_synapses`): for each dendrite, draw `live` UNIQUE
  positions into a `BTreeSet<u8>` — which yields them **sorted ascending for
  free**, satisfying the load-bearing invariant. The live prefix gets weights
  `U(0, 8)` (all-excitatory init), the sampled `x`, alpha/last_event = 0; the dead
  tail (up to `S`) is zeroed so every block is exactly `S` wide.

The sort is not a separate step — it falls out of the `BTreeSet`. Rejection
sampling can't always reach `live` distinct `u8` values, so attempts are capped
and `live_synapse_counts[d]` is shrunk to whatever was actually drawn (the source
of truth `generate_synapses` reads back).

The trailing sentinels for `dendrite_offsets` / `synapse_offsets` are
deliberately **not** added per population; `build_network` appends them once after
all populations are generated.

## 7.3 Fixed synapse slots — the layout decision, now in code

The most consequential layout choice, and it is implemented:

**Synapses are pre-allocated as fixed-size slots per dendrite, and connections
bind to existing slots rather than creating synapses.**

`SYNAPSE_SLOTS_PER_DENDRITE = S` is the uniform stride (currently `u8::MAX = 255`,
which over-provisions heavily — tune in `constants.rs` if memory matters). Because
every dendrite gets its full pool of empty slots at allocation:

```
synapse_offsets[d] = d * S        // analytic — value is pure arithmetic
```

The offsets are *value*-analytic, though the code still **stores** them as a `Vec`
and `synapse_to_dendrite` still does a `partition_point` binary search rather than
the `s / S` division the analytic form permits — a possible future simplification,
not yet taken ([chapter 9](09-gaps-and-open-questions.md)).

Connections then *bind a presynaptic source to an existing empty slot* (§7.4
below) and emit `(source_neuron, target_synapse)` pairs grouped into the axon CSR.
They create nothing and **never reorder**, so the sorted-`x` invariant set up in
§7.2 is structurally preserved. The rejected alternative — materializing synapses
in connection order — would force a post-hoc co-permutation of
weights/`x`/alphas/last_events plus a counting pass to size the arrays.

## 7.4 `build_network` — orchestration and slot binding

`build_network(builder, rng)` (`src/network/build.rs`) runs four phases:

1. **Generate** each population in turn (§7.2), recording its global neuron
   `Range<u32>` into `ranges` (later surfaced by `Network::population_range`).
2. **Append the trailing sentinels** to `dendrite_offsets` and `synapse_offsets`.
3. **Resolve connections into axon edges.** For each `Connection`, call
   `rule.apply` to get `(src_neuron, dst_neuron)` pairs, then for each pair bind
   the source to *the first free synapse slot on a matching-compartment dendrite*
   of the destination neuron:

   ```rust
   let want_apical = matches!(c.compartment, Compartment::Apical) as u8;
   for den in dendrite_offsets[d] .. dendrite_offsets[d+1] {
       if dendrite_is_apical[den] != want_apical { continue; }
       if consumed[den] < live_synapse_counts[den] {        // a free live slot exists
           let slot = synapse_offsets[den] + consumed[den];
           consumed[den] += 1;
           axon_edges.push((src, slot));
           break;
       }                                                    // else dendrite full → try next; else drop edge
   }
   ```

   `consumed[d]` is a per-dendrite cursor so no slot is wired to two presynaptic
   axons. An edge that finds no free matching-compartment slot is **silently
   dropped** — a capacity-limited binding, not an error.
4. **Build the axon CSR.** Sort `axon_edges` by source neuron, prefix-sum the
   per-source counts into `axon_offsets` (len `n+1`), and flatten the targets into
   `axon_targets`. The result is the `Axon` ([chapter 3.1](03-data-model.md)) that
   `run_event_loop`'s `SOMATIC_SPIKE` arm fans out over.

`Network::build` wraps this with a fixed-seed `SmallRng` for reproducibility. The
returned `Network { synapses, dendrites, somas, axons, ranges }` is exactly what
the event loop consumes.

The build is covered by tests that pin the load-bearing invariants: offset arrays
carry their trailing sentinels, the axon CSR is monotonic and correctly sized,
one-to-one wires each source exactly once onto a *distinct basal slot of the right
population*, and `synapse_x` is strictly ascending within every live block.

## 7.5 Structural plasticity — keeping it non-fragile

Still entirely a *design* — there is no structural-plasticity code — but the
fixed-slot layout (§7.3) is what makes it cheap, and the allocator's stride/
headroom choices are made with it in mind. The rule that must never break:
**never rewrite `synapse_offsets` in-place at runtime.** Two things make naive
in-place deletion a nightmare — fixed-stride offsets would all have to shift, and
`axon_targets` store *absolute* synapse indices, so moving a synapse silently
invalidates every incoming axon pointer. So separate *liveness* from *layout*:

1. **Tombstone (runtime, O(1)→O(stride)).** Don't remove — with the packed model,
   "delete" = shift the live tail left by one and `live_synapse_counts -= 1`
   (preserves the sort), O(stride) over a ~16-live-element block.
2. **Migrate within the block (runtime, O(stride)).** Insert into a dead tail
   slot, `live_count += 1`, re-sort the small live prefix by `x`. All edits stay
   inside one fixed block; offsets never move.
3. **Compaction (offline, between epochs, one linear pass).** When tombstones
   accumulate, rebuild: copy live synapses into fresh arrays, recompute offsets,
   build an `old_index → new_index` remap, and apply it to `axon_targets` in one
   scan. The GC model — non-fragile because it rebuilds with an explicit remap
   instead of mutating in place.

The one genuinely global concern — the absolute-index remap of `axon_targets`
when a synapse's index changes — is confined to that compaction pass. If
plasticity ever needs *more* synapses than the stride allows, over-provision each
block (`stride + slack`); compaction reclaims it. With `S = 255` today, headroom
is abundant.

---

Next: [chapter 8 — The MNIST pipeline](08-mnist-pipeline.md), the first concrete
network these pieces assemble into.
