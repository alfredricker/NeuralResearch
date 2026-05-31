# 2. Architecture choices

This chapter explains the four foundational decisions that shape every data
structure and algorithm in the codebase. They all answer one question: *how do
we run [chapter 1](01-theory.md)'s model on a GPU?* Each choice is made for the
GPU even though the current code runs on the CPU — the CPU implementation is a
faithful prototype of the eventual kernels.

## 2.1 Struct-of-Arrays, not Array-of-Structs

The intuitive layout is a `Vec<Neuron>` where each `Neuron` owns its dendrites,
each dendrite owns its synapses — hierarchical ownership. **The codebase rejects
this.** Instead, every per-component field is its own flat `Vec`:

```rust
// NOT this:
struct Neuron { soma: Soma, dendrites: Vec<Dendrite> }   // AoS — rejected

// This:
struct Soma {
    soma_potentials:  Vec<i8>,    // one entry per neuron
    soma_thresholds:  Vec<i8>,
    soma_betas:       Vec<u8>,
    // ...
}
```

Three reasons, all GPU-driven:

1. **Coalesced memory access.** A GPU warp of 32 threads processing 32 neurons
   wants `soma_potentials[n]` for 32 consecutive `n` to be 32 contiguous bytes,
   so one memory transaction serves the whole warp. AoS would interleave
   unrelated fields and waste bandwidth.
2. **No pointer chasing.** A flat index *is* the address. There are no `&`
   references to follow across the heap — fatal on a GPU where host pointers are
   meaningless.
3. **Independent residency.** Hot fields (e.g. `soma_potentials`, updated every
   spike) can live in fast memory while cold fields (e.g. `synapse_xs`, never
   mutated at runtime) stay elsewhere.

A neuron, then, is not an object. It is an *index* `n`, and "its" data is
whatever lives at position `n` (or a derived range) across many arrays.

## 2.2 Offset arrays: the CSR-style indirection

A neuron owns a variable number of dendrites; a dendrite owns a variable number
of synapses. Variable-length nesting in flat arrays is the classic compressed
sparse row (CSR) problem, solved here with **offset arrays**:

```
dendrite_offsets[n]  → index of neuron n's first dendrite
synapse_offsets[d]   → index of dendrite d's first synapse
```

Both arrays have length `count + 1` with a **sentinel at the end**, so a range is
always `[offsets[i] .. offsets[i+1]]` with no special case for the last element:

```rust
let d_start = dendrite_offsets[n] as usize;
let d_end   = dendrite_offsets[n + 1] as usize;   // sentinel makes this safe
```

The reverse lookup — "which dendrite owns synapse `s`?" — is a binary search over
`synapse_offsets`:

```rust
synapse_to_dendrite(s, synapse_offsets)   // O(log n) via partition_point
```

This indirection is the backbone of [chapter 3](03-data-model.md). One important
forward reference: [chapter 7](07-network-construction.md) argues that if every
dendrite gets a *fixed* number of synapse slots, `synapse_offsets` becomes
**analytic** (`d * stride`) and the binary search becomes pure arithmetic — a
significant simplification that also makes GPU addressing predictable.

## 2.3 Event-driven execution: there is no tick loop

This is the architecture's defining choice and the easiest to get wrong when
reading the code.

A conventional simulator steps a global clock: `for t in 0..T { update_everything() }`.
Every neuron is visited every tick whether or not anything happened to it. For a
sparse spiking network — where most neurons are silent most of the time — this is
almost entirely wasted work.

Instead, the simulation is a **queue of events**. An event says "something
happened to component X at time T." Processing an event mutates a *small, local*
slice of state and may *emit new events*. The simulation runs until the queue
drains. Nothing is touched unless an event reaches it.

The three event types (detailed in [chapter 5](05-event-system.md)) mirror the
signal flow from [chapter 1](01-theory.md):

- `FORWARD_AP` — an action potential arriving at a neuron's downstream synapses.
- `DENDRITIC_SPIKE` — a dendrite crossed threshold.
- `SOMATIC_SPIKE` — a soma crossed threshold (and will emit forward APs).

### The lazy-decay consequence

Because there is no tick, **decay cannot be applied incrementally.** A synapse
that is idle for 200 ticks is never visited during those 200 ticks. So decay is
computed *on read*: every decaying variable stores a `last_event` timestamp, and
when an event finally touches it, the handler computes `elapsed = now − last_event`
and decays in one shot (§4.1). This is why timestamps are carried on events and
stored per-component — they *are* the clock. Time is reconstructed locally,
never stepped globally.

### What this buys, and what it costs

- **Buys:** work proportional to activity, not to network size; trivially
  variable per-component timescales; a natural mapping to GPU "one event = one
  unit of work."
- **Costs:** ordering and concurrency become subtle. Two events targeting the
  same dendrite at the same timestamp race on `dendrite_activities[d]`
  ([chapter 5](05-event-system.md) and [chapter 9](09-gaps-and-open-questions.md)).
  Trial boundaries can't be "tick T" — they need a sentinel event or a timestamp
  cutoff ([chapter 8](08-mnist-pipeline.md)).

## 2.4 Fixed-width integers everywhere

No `f32` lives in the hot path. State is `i8` / `u8` / `u16` / `i16` / `u32`:

| Type | Range | Used for |
| ---- | ----- | -------- |
| `i8`  | −128..127 | soma potential, soma threshold, synapse weight, dendrite constant |
| `u8`  | 0..255 | synapse `alpha`, synapse `x`, `beta` (capped at 63) |
| `u16` | 0..65 535 | dendrite activity & threshold, all `last_event` timestamps |
| `i16` | −32 768..32 767 | learning rate, intermediate `delta_V` before clamping |
| `u32` | 0..~4.3 B | all offset/index/target arrays |

Rationale:

- **Density.** Smaller types mean more state per cache line / per memory
  transaction — directly more throughput on the bandwidth-bound GPU.
- **Decay is bit-shifting.** The exponential decay (§4.1) is implemented as
  integer shifts on these types — no FPU, no transcendental functions, fully
  deterministic across CPU and GPU.
- **The `learning_rate` is a divisor.** `delta_w = burst_term * alpha / lr`. A
  *larger* `lr` means *smaller* weight steps. The constant `MSLR` (minimum
  synaptic learning rate, §4) is the smallest `lr` for which the maximum possible
  `delta_w` still fits in `i8` without saturating spuriously.

The cost is calibration: with 8-bit weights and thresholds, the constants
(`H_ALPHA`, `H_BETA`, thresholds, `ALPHA_BOOST`) must be tuned so signals neither
vanish to zero nor saturate. Those tunings live in `constants.rs` and are
discussed where each value is used.

`u16` timestamps **wrap** at 65 535. Every elapsed-time computation uses
`wrapping_sub`, so wrap is handled correctly *for decay* — but it is still a
watch-out for trial bookkeeping ([chapter 9](09-gaps-and-open-questions.md)).

## 2.5 Where unsafe lives

The design confines all unsafe code to one place: pushing onto the event queue
(`EventProducer::push`). Everything else is safe Rust operating on slices. This
is deliberate — the producer is the one structure that must eventually become a
lock-free, GPU-shared, atomically-claimed buffer, so the unsafe surface is kept
to exactly that boundary ([chapter 5](05-event-system.md)).

---

Next: [chapter 3 — Data model](03-data-model.md), which turns these principles
into the concrete arrays the code actually declares.
