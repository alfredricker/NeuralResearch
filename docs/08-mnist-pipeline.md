# 8. The MNIST pipeline

The first end-to-end target. It exercises every prior chapter: the allocator
([chapter 7](07-network-construction.md)) builds the layers, the IO boundary
([chapter 11](11-io-boundary.md)) encodes pixels and reads predictions, the trial
loop runs the dynamics ([chapters 5](05-event-system.md),
[6](06-learning-dynamics.md)) under a clock ([chapter 12](12-time-and-clocking.md)).
The construction and IO pieces now exist; the *trial loop and feedback* are the
remaining unimplemented glue. This chapter is the plan plus its open decisions.

## 8.1 Topology

```
[Input: 784 pixels] ──SOMATIC_SPIKE──▶ [Hidden: N neurons] ──SOMATIC_SPIKE──▶ [Output: 10 neurons]
                                                ◀──── apical feedback (optional) ────
```

A neuron's axon fans out as `SYNAPSE_SIGNAL`s to synapses in the next layer
([chapter 5.3](05-event-system.md)); the input layer's "spikes" are asserted by
`InputSpace::encode` ([chapter 11.3](11-io-boundary.md)). The learning feedback
path (to make the correct output neuron burst,
[chapter 1.6](01-theory.md)) is a design fork covered in §8.5.

**Input layer:** `input_config()` — 784 zero-dendrite axon sources, one per pixel
(identity `SensoryMap`). Already concrete.

**Hidden layer:** still to be defined — a `NeuronConfig` in the
`visual_mnist` spirit (e.g. 6 basal × 8 dendrites/branch × 16 live synapses ≈ 768
synapses/neuron, 48 dendrites/neuron). Start at **N = 200**
([chapter 3.4](03-data-model.md)) — enough capacity, fast to iterate. Note the
fixed-slot stride is `S = 255` per dendrite regardless of live count, so the
*allocated* synapse arrays are larger than the live working set
([chapter 7.3](07-network-construction.md)); size memory accordingly or lower `S`.

**Output layer:** `output_config()` — 10 neurons (identity `ReadoutMap`), low
`soma_threshold` so they fire readily, no apical. Already concrete
([chapter 11.4](11-io-boundary.md)).

## 8.2 Connectivity choice

Wire input→hidden with a `ConnRule` ([chapter 7.1](07-network-construction.md)),
in increasing structure:

- **A — Dense / fixed-in-degree** *(recommended first pass).* `DenseRandom { p }`
  or `FixedInDegree { k }` from pixels onto hidden basal dendrites. No spatial
  prior, but the gamma co-activity dynamics
  ([chapter 6.2](06-learning-dynamics.md)) still learn structure. `build_network`
  binds each edge to a free basal slot and inverts to the axon CSR automatically.
- **B — Local receptive fields.** `ReceptiveField { radius }` (implemented) ties
  each hidden neuron to a pixel neighborhood on the 28×28 grid — a convolution-like
  prior. `x`-position could later encode distance-from-center.
- **C — Topographic.** `Topographic { patch }` — a retinotopic patch map. The rule
  is **not yet implemented** (returns `InvalidRule`).

Start with A; add structure only when there's a reason.

## 8.3 Input encoding

Pixels are external event sources. `InputSpace::encode`
([chapter 11.3](11-io-boundary.md)) walks the frame and, for each lit pixel,
pushes a `SOMATIC_SPIKE` at the bound global neuron index, with:

- a **burst payload** scaled by pixel intensity (`intensity_to_burst`, 1..=4) —
  brighter pixels drive a larger downstream EPSP;
- a **jittered timestamp** in `[base_ts, base_ts + window)` so the frame presents
  as a small stochastic volley rather than one synchronous edge.

`handle_somatic_spike` then fans each one out across the axon CSR into the hidden
layer. There is no separate "pre-generate `FORWARD_AP` events" step any more (the
old encoding sketch predates the four-event model) — encoding is just
`SOMATIC_SPIKE` production through the same producer the dynamics use.

A note on the window length: with `ALPHA_BOOST = 64`, a synapse clears
`H_ALPHA = 30` in ~1 spike but needs several to build the `alpha` that makes gamma
amplification meaningful, and `alpha` now decays *slowly* (`ALPHA_DECAY = 11`,
~2048-tick half-life), so eligibility persists comfortably across a trial. A
**trial window of ~100–200 ticks** remains a reasonable starting estimate.

## 8.4 Trial loop and readout

```rust
let mut clock: u16 = 0;
let mut spike_counts = vec![0u32; n_neurons];   // per-neuron AP accumulator (chapter 11.4)
for _ in 0..T_trial {
    space.encode(frame, clock, window, &producer, &mut rng); // push this tick's input
    run_event_loop(&queue, /* ...all SoA arrays..., &mut spike_counts */); // drain one wavefront
    clock = clock.wrapping_add(1);                            // advance the clock (chapter 12)
}
let prediction = effector.predict(&spike_counts);            // argmax over the output window
```

Because each `run_event_loop` call processes one wavefront
([chapter 5.3](05-event-system.md)), calling it per tick lets a spike cascade
march forward over successive ticks, and the advancing `clock` makes the lazy
decay between ticks real ([chapter 12](12-time-and-clocking.md)).

**Reading the prediction:** the effector sums `spike_counts` over each class's
member output neurons and argmaxes, returning `None` if the layer was silent
([chapter 11.4](11-io-boundary.md)).

**Between trials:** the clean, deterministic choice is to clear `dendrite_activities`,
`soma_potentials`, and `spike_counts`. Unlike the older design, `dendrite_activity`
and `soma_potential` now **leak** ([chapter 6](06-learning-dynamics.md)), so a long
enough inter-trial gap forgets them on its own — but an explicit reset is still the
right call for trial isolation. `alpha` and `beta` **persist intentionally**:
they carry the longer-timescale learning state across trials.

> **Gaps for the loop.** (1) `run_event_loop` does not yet accumulate
> `spike_counts` — there is no `SOMATIC_SPIKE` counter, so readout has nothing to
> read ([chapter 11.4](11-io-boundary.md)). (2) The ring buffer never advances
> `head` ([chapter 5.2](05-event-system.md)), so a multi-tick loop overruns it.
> (3) Nothing advances a clock today; `base_ts` is whatever the caller passes
> ([chapter 12](12-time-and-clocking.md)). All three are prerequisites for a
> running trial loop.

## 8.5 Training feedback — the fork

To learn, the correct output neuron must **burst** so its `beta` climbs and
`update_weight` applies LTP ([chapter 6.6](06-learning-dynamics.md)). Two routes
([chapter 1.6](01-theory.md)):

- **Option 1 — direct injection** *(sufficient for classification).* During
  training, push extra `SOMATIC_SPIKE`s (or `SOMA_SIGNAL`s) at the *correct*
  output neuron so it fires repeatedly and `beta` climbs. No apical machinery;
  doesn't modulate the hidden layer. This is the recommended first pass and needs
  no new event types — the boundary can do it the same way `encode` asserts input.
- **Option 2 — apical feedback** *(the biologically faithful path).* The output
  axon drives *hidden* neurons' apical synapses. With the current model this needs
  **no new event type** — an apical `SYNAPSE_SIGNAL` already produces a graded
  `SOMA_SIGNAL` via `apical_plateau` ([chapter 6.3](06-learning-dynamics.md)),
  which can push the soma into a burst. What it *does* require: the hidden config
  to set `n_apical_dendrites` (so apical dendrites are allocated), and feedback
  connections wired onto the `Compartment::Apical` slots
  ([chapter 7.4](07-network-construction.md)). The apical *mechanism* exists
  end-to-end now; only the topology to exercise it is missing.

Start with Option 1; Option 2 is the path to real BDP — and is much closer to
reachable than it was when this book was first written.

## 8.6 Trial boundaries

There is no global tick to mark "end of trial" ([chapter 2.3](02-architecture.md)).
The boundary is either a **timestamp cutoff** (run the clock for `T_trial` ticks,
then reset) or a **sentinel event**. With the caller-driven clock of
[chapter 12](12-time-and-clocking.md), the timestamp-cutoff form is the natural
one: the harness owns the loop count, so "trial over" is simply "the loop
finished." Still an open decision insofar as the loop itself is unwritten.

## 8.7 What must exist before any of this runs

Construction and IO are **done** ([chapters 7](07-network-construction.md),
[11](11-io-boundary.md)). The remaining dependency order:

1. A working ring buffer that recycles slots — advance `head`
   ([chapter 5.2](05-event-system.md)) — or a multi-tick loop overruns it.
2. A `spike_counts` accumulator in `run_event_loop`, zeroed per trial
   ([chapter 11.4](11-io-boundary.md)).
3. A clock the trial loop advances ([chapter 12](12-time-and-clocking.md)).
4. A hidden-layer `NeuronConfig` and the input→hidden→output wiring (§8.1–8.2).
5. The trial loop, readout, and training feedback (§8.4–8.5).

---

Next: [chapter 9 — Gaps and open questions](09-gaps-and-open-questions.md), the
consolidated punch list.
