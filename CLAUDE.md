# neural/research — CLAUDE.md

## What this is

A biologically-inspired spiking neural network simulator in Rust, structured for eventual GPU execution. The immediate goal is a working MNIST learner. Long-term goal is biologically realistic Burst-Dependent Plasticity (BDP).

## Build / test

```
cargo build
cargo test
```

No external dependencies. Edition 2024. 35 unit tests, all passing.

## Architecture overview

### Data layout
Struct-of-Arrays (SoA) flat Vecs. Neurons, dendrites, and synapses each occupy contiguous arrays indexed by their flat index. This mirrors GPU memory layout.

### Offset arrays (critical pattern)
`dendrite_offsets[n]` → first dendrite index of neuron `n`  
`synapse_offsets[d]` → first synapse index of dendrite `d`  
Both arrays are length+1 (sentinel at end), so `[n..n+1]` gives the range.

`synapse_to_dendrite(s, synapse_offsets)` uses `partition_point` — O(log n) binary search.

### Event system
Three event types (u8, not enum — buffer will be shared with GPU kernels):
- `SOMATIC_SPIKE` (0) — source = neuron_idx
- `DENDRITIC_SPIKE` (1) — source = dendrite_idx
- `FORWARD_AP` (2) — source = neuron_idx

`EventQueue` holds a fixed `Box<[Event]>` ring buffer with atomic head/tail.  
`EventProducer<'a>` wraps a raw `*mut Event` — all unsafe is isolated to `EventProducer::push`.  
`run_event_loop` drains the queue each tick, dispatches to handlers, and handlers push new events via the producer.

### Learning model
STDP-like burst-dependent plasticity:
- `alpha` (u8 per synapse) — synaptic activity, lazy exponential decay via `shift_decay_u8`
- `beta` (u8 per soma) — burst counter, decays by 1 per T_BETA=500 ticks
- Weight update on SOMATIC_SPIKE: `delta = burst_term * alpha / lr`, where `burst_term = beta - H_BETA`
- LTP when bursting (beta > H_BETA=4), LTD otherwise

### Dendritic integration
`update_dendrite_activity`: asymmetric — synapses with higher `x` (more distal on dendrite) amplify proximal synapses. `gamma = Σ shift_decay_u8(alpha_j, dx, X_DECAY=4)` for all `j` with `x_j > x_i`. `delta_V = w_i * (1 + gamma)`. **synapse_xs must be sorted ascending within each dendrite.**

`handle_apical_fb`: multiplicative top-down feedback — `new_v = v_s + effective_alpha * v_s`, emits SOMATIC_SPIKEs equal to `new_v / soma_threshold`.

## Module map

```
src/
  constants.rs          — T_BETA=500, H_ALPHA=30, H_BETA=4, ALPHA_DECAY=8, X_DECAY=4, MSLR=120, ALPHA_BOOST=64
  math/
    decay.rs            — shift_decay / shift_decay_u8: O(1) base-2 exponential decay
    midpoint.rs         — midpoint arithmetic
  neuron/
    synapse.rs          — update_synapse_alpha, update_weight
    dendrite.rs         — update_dendrite_activity, synapse_to_dendrite
    soma.rs             — (stub)
    axon.rs             — (stub)
  network/event/
    event.rs            — Event struct + SOMATIC_SPIKE/DENDRITIC_SPIKE/FORWARD_AP constants
    queue.rs            — EventQueue (ring buffer)
    push.rs             — EventProducer (unsafe isolation)
    loop.rs             — run_event_loop
    handlers.rs         — handle_somatic_spike, handle_dendritic_spike, handle_forward_ap, handle_apical_fb
    slice.rs            — neuron_synapse_range, dendrite_synapse_range
  init/neuron/
    config.rs           — NeuronConfig struct
  taxonomy/neuron/
    visual_mnist.rs     — CONFIG for MNIST hidden layer neurons (6 basal × 8 branch × 16 syn = 768 syn/neuron)
    simple1.rs          — simple neuron config
    pyramidl1.rs        — (stub)
    classifier.rs       — neuron type classifiers
  memory/
    partition.rs        — (stub) GPU memory partition utilities
  gpu/mod.rs            — (stub) reserved for CUDA
```

## Next work: MNIST learner

Build order from `notes/next_steps.md`:

1. **Network allocator** (`src/init/neuron/mod.rs`) — `build_layer(config, n_neurons, rng)` → produces all SoA Vecs. Key constraint: `synapse_xs` must be sorted ascending per dendrite.
2. **Input encoding** — `encode_frame(pixels, timestamp, tick, queue, pixel_axon_targets, pixel_axon_offsets)` — rate-coded Bernoulli firing per pixel.
3. **Trial loop** — 200 ticks per trial; reset `dendrite_activities` and `soma_potentials` between trials (alpha/beta persist).
4. **Output readout** — `spike_counts: [u32; 10]`, argmax after trial.
5. **Training feedback** — push FORWARD_AP into correct output neuron's targets.

Layer sizes: hidden N=200 (visual_mnist config), output 10 neurons (simpler config, lower threshold).

Open decisions:
- Connectivity: dense random (recommended first pass) vs local receptive fields
- Feedback path: direct output injection vs apical FB through hidden layer (visual_mnist currently has `n_apical_dendrites: None`)
- Output layer config: needs separate NeuronConfig with lower soma_threshold and simpler dendrite structure

## Known issues / watch-outs

- `visual_mnist` `learning_rate=256 > MSLR=120` — valid but slow updates (delta≈4 vs ≈10 at MSLR)
- `dendrite_activity` has no decay — needs explicit reset between trials
- u16 timestamps wrap at ~327 trials × 200 ticks; `wrapping_sub` handles it correctly
- `run_event_loop` FORWARD_AP arm has an inner loop — noted as a future batching/parallelism opportunity
- Weight init U(-8,8) means ~50% inhibitory from start — may want U(0,8) initially
- No lateral inhibition — neurons may not specialize without WTA competition
