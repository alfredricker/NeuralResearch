# Next Steps: MNIST Learner

## Immediate blocker: network allocator

`NeuronConfig` describes a neuron's shape but nothing allocates the SoA arrays the event loop
actually consumes. This is the first thing to build.

Target: `src/init/neuron/mod.rs` — a function roughly like:

```rust
pub fn build_layer(
    config: &NeuronConfig,
    n_neurons: usize,
    rng: &mut impl Rng,
) -> Layer  // or return individual Vec fields — decide on grouping later
```

It needs to produce all arrays expected by `run_event_loop`:

**Soma** (len = n_neurons):
- `soma_potentials: Vec<i8>` — all 0
- `soma_thresholds: Vec<i8>` — all config.soma_threshold
- `soma_betas: Vec<u8>` — all 0
- `soma_last_events: Vec<u16>` — all 0
- `soma_lrs: Vec<i16>` — all config.learning_rate
- `dendrite_offsets: Vec<u32>` — stride = n_basal_dendrites * dendrites_per_branch

**Dendrite** (len = n_neurons * n_basal_dendrites * dendrites_per_branch):
- `dendrite_activities: Vec<u16>` — all 0
- `dendrite_thresholds: Vec<u16>` — all config.basal_dendrite_threshold
- `dendrite_constants: Vec<i8>` — sample from N(mean_basal_dendrite_constant, std_basal_dendrite_constant)
- `dendrite_last_events: Vec<u16>` — all 0
- `synapse_offsets: Vec<u32>` — stride = synapses_per_dendrite
- `dendrite_to_neuron: Vec<u32>` — flat map d -> neuron_idx

**Synapse** (len = n_neurons * n_basal_dendrites * dendrites_per_branch * synapses_per_dendrite):
- `synapse_weights: Vec<i8>` — small uniform random, e.g. U(-8, 8)
- `synapse_xs: Vec<u8>` — sampled from N(mean_synapse_x, std_synapse_x), sorted per dendrite
- `synapse_alphas: Vec<u8>` — all 0
- `synapse_last_events: Vec<u16>` — all 0

The `synapse_xs` **must be sorted in ascending order within each dendrite** — the gamma
computation in `update_dendrite_activity` assumes this. Build each dendrite's xs, sort, then
write.

**Axon** — built separately because it encodes inter-layer connectivity, not intra-neuron
structure. See topology.md.

---

## Step 2: input encoding

Input pixels are not model neurons — they're external event sources that inject FORWARD_AP events.

```rust
pub fn encode_frame(
    pixels: &[u8; 784],
    timestamp: u16,
    tick: u16,
    queue: &EventQueue,
    // axon_targets/offsets mapping pixel i -> synapse indices in hidden layer
    pixel_axon_targets: &[u32],
    pixel_axon_offsets: &[u32],
)
```

For each pixel i, sample a Bernoulli with p = pixels[i] / 255.0 * max_rate. If it fires,
push a FORWARD_AP for each synapse in `pixel_axon_targets[pixel_axon_offsets[i]..pixel_axon_offsets[i+1]]`.

This is called once per tick inside the trial loop.

---

## Step 3: trial loop

```
for tick in 0..T_trial:
    encode_frame(pixels, base_ts + tick, ...)
    run_event_loop(queue, ...)
    // count output spikes into output_counts[0..10]
```

T_trial needs tuning — see threshold math in topology.md. Start with 200 ticks.

Between trials: reset `dendrite_activities` and `soma_potentials` to 0. alpha and beta persist
intentionally (they represent longer-timescale learning state).

---

## Step 4: output readout

Track a `spike_counts: [u32; 10]` array. Every time a SOMATIC_SPIKE fires from an output
neuron, increment the count. After T_trial ticks, `argmax(spike_counts)` is the prediction.

---

## Step 5: training feedback

When the prediction is wrong (or always, to reinforce the correct class):
- Push FORWARD_AP events into the output layer for the correct output neuron's synapse targets
- This drives `handle_forward_ap` → `handle_dendritic_spike` → `handle_somatic_spike` (bursting)
- Bursting increments beta, which drives LTP via `update_weight`'s burst_term

Alternatively use `handle_apical_fb` directly if the hidden layer has apical dendrites configured.
**visual_mnist currently has n_apical_dendrites: None** — so the feedback path needs a config
decision before it can work. See topology.md.

---

## Open questions

- Does `dendrite_activity` need to be reset between trials, or left to decay naturally?
  Currently there is no decay mechanism for dendrite_activity — it only resets on spike.
  Probably needs explicit reset between trials, or a decay step added.

- Should the output layer use a different NeuronConfig? It probably needs a simpler structure
  (fewer/no dendrites, lower threshold, no apical) and a higher learning rate.

- Wrapping u16 timestamps: T_trial ticks per trial * n_trials can exceed u16::MAX (65535).
  After 327 trials of 200 ticks each, the timestamp wraps. The wrapping_sub in
  update_synapse_alpha handles this correctly, but axon_targets timestamps also wrap —
  verify this doesn't corrupt alpha state across the wrap boundary.
