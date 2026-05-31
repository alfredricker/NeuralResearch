# 8. The MNIST pipeline

The first end-to-end target. It exercises every prior chapter: the allocator
([chapter 7](07-network-construction.md)) builds the layers, input encoding feeds
the event queue ([chapter 5](05-event-system.md)), the trial loop runs the
dynamics ([chapter 6](06-learning-dynamics.md)), and a readout interprets the
output. None of this is implemented yet — this chapter is the plan plus its open
decisions.

## 8.1 Topology

```
[Input: 784 pixels] ──FORWARD_AP──▶ [Hidden: N neurons] ──FORWARD_AP──▶ [Output: 10 neurons]
                                              ◀──── apical feedback (optional) ────
```

Each neuron's axon broadcasts forward APs to synapses in the next layer
([chapter 5.3](05-event-system.md)). The learning feedback path (to make the
correct output neuron burst, [chapter 1.6](01-theory.md)) is a design fork
covered in §8.5.

**Hidden layer:** the `visual_mnist`-style config — 6 basal × 8 dendrites/branch
× 16 synapses = 768 synapses/neuron, 48 dendrites/neuron. Start at **N = 200**
(~600 KB synapse state, [chapter 3.4](03-data-model.md)) — enough capacity, fast
to iterate.

**Output layer:** 10 neurons, one per digit, needs a *separate, simpler* config —
lower `soma_threshold` so it fires readily, fewer/larger dendrites, no apical,
possibly higher learning rate (or no learning, letting the hidden layer do the
representation).

## 8.2 Connectivity choice

Three options, in increasing structure (full analysis in
`notes/topology.md`):

- **A — Dense random** *(recommended first pass).* Each hidden synapse draws a
  random input pixel; invert to build the pixel→synapse axon CSR. With N=200 and
  768 synapses each, each pixel drives ~196 synapses. No spatial structure, but
  the gamma co-activity dynamics ([chapter 6.2](06-learning-dynamics.md)) still
  learn structure. Maps to `ConnRule::DenseRandom`.
- **B — Local receptive fields.** Each hidden neuron gets a center pixel and
  draws from a radius; `x`-position could encode distance-from-center. A
  convolution-like prior, harder to build. Maps to `ConnRule::ReceptiveField`.
- **C — Topographic.** A 2D hidden sheet, each neuron tied to a pixel patch — a
  retinotopic map, good for visualizing learned weights. Maps to
  `ConnRule::Topographic`.

Start with A; add structure only when there's a reason.

## 8.3 Input encoding

Pixels are external event sources. Per the architecture
([chapter 2.3](02-architecture.md)), input is *pre-generated* as `FORWARD_AP`
events with stochastic timestamps spread across the trial window, pushed into the
queue up front. A sketch:

```rust
fn encode_frame(pixels: &[u8; 784], timestamp, tick, queue, pixel_axon_targets, pixel_axon_offsets) {
    // for each pixel: Bernoulli(p = brightness/255 * max_rate); if it fires,
    // push a FORWARD_AP for each downstream synapse in its axon range
}
```

Brighter pixels fire more often (rate coding). The `notes/topology.md` threshold
math shows why timing matters: with `ALPHA_BOOST = 64`, a synapse clears
`H_ALPHA = 30` in ~1 spike but needs 3–4 spikes to saturate `alpha`, so gamma
amplification only becomes meaningful ~10 ticks in. **A trial window of
100–200 ticks** is the starting estimate.

## 8.4 Trial loop and readout

```
for tick in 0..T_trial:
    encode_frame(pixels, base_ts + tick, ...)   // push this tick's input events
    run_event_loop(queue, ...)                  // drain + propagate one wavefront (chapter 5.3)
    // count SOMATIC_SPIKEs from output neurons into spike_counts[0..10]
prediction = argmax(spike_counts)
```

Because each `run_event_loop` call processes one wavefront
([chapter 5.3](05-event-system.md)), calling it per tick lets a spike cascade
forward over successive ticks.

**Between trials:** explicitly reset `dendrite_activities` and `soma_potentials`
to 0 (`dendrite_activity` has no decay — [chapter 6.3](06-learning-dynamics.md)).
`alpha` and `beta` **persist intentionally** — they carry the longer-timescale
learning state across trials.

## 8.5 Training feedback — the fork

To learn, the correct output neuron must **burst** so its `beta` climbs and
`update_weight` applies LTP ([chapter 6.5](06-learning-dynamics.md)). Two routes
([chapter 1.6](01-theory.md)):

- **Option 1 — direct injection** *(sufficient for classification).* Push
  `FORWARD_AP` events into the correct output neuron's synapses, driving it to
  burst. No apical machinery; doesn't modulate the hidden layer.
- **Option 2 — apical feedback** *(the biologically faithful path).* The output
  axon drives hidden neurons' apical synapses; `handle_apical_fb`
  ([chapter 6.6](06-learning-dynamics.md)) multiplicatively bursts them. Requires
  the hidden config to set `n_apical_dendrites` (currently `None`), a second
  apical synapse compartment, an axon-constant array, and an event arm that
  routes to `handle_apical_fb` — none of which exist yet
  ([chapters 3](03-data-model.md), [5](05-event-system.md),
  [6](06-learning-dynamics.md)).

Start with Option 1; Option 2 is the path to real BDP.

## 8.6 Trial boundaries

There is no global tick to mark "end of trial" ([chapter 2.3](02-architecture.md)).
The boundary must be either a **sentinel event type** or a **timestamp cutoff**.
This is an open decision (`CLAUDE.md`, [chapter 9](09-gaps-and-open-questions.md)).

## 8.7 What must exist before any of this runs

In dependency order:

1. The allocator + `Network::build` ([chapter 7.2](07-network-construction.md)) —
   nothing has data without it.
2. `ConnRule::apply` and the axon-CSR resolver
   ([chapter 7.1](07-network-construction.md)) — no connectivity without it.
3. A working ring buffer that recycles slots
   ([chapter 5.2](05-event-system.md)) — a multi-tick loop will otherwise
   overrun `head`.
4. Input encoding, trial loop, readout, feedback (this chapter).

---

Next: [chapter 9 — Gaps and open questions](09-gaps-and-open-questions.md), the
consolidated punch list.
