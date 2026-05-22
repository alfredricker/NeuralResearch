# Topology Options

## Layer structure

A minimal MNIST network needs three layers:

```
[Input: 784 pixels]  →  [Hidden: N visual_mnist neurons]  →  [Output: 10 neurons]
         FORWARD_AP                                FORWARD_AP
                                       ← apical FB ←
```

The axon of each neuron broadcasts FORWARD_AP events to synapses in the next layer.
The reverse path (feedback for learning) runs through apical dendrites — currently
disabled in visual_mnist (n_apical_dendrites: None). That's a decision point.

---

## Connectivity options

### Option A: Dense random

Each synapse in the hidden layer is independently assigned a random input pixel.
Build by: for each synapse s in hidden layer, draw pixel_idx ~ U(0, 783). Then
invert the mapping to build axon_targets/axon_offsets for input pixel i.

- Simple to implement
- Each pixel drives ~(N_hidden * synapses_per_neuron) / 784 synapses on average
  With N=200 hidden, 768 synapses each: 200*768/784 ≈ 196 synapses per pixel
- No spatial structure — hidden neurons have global receptive fields
- The dendritic synapse ordering by x still applies; randomly assigned synapses are
  sorted by their x values within each dendrite, not by spatial position

### Option B: Local receptive fields

Assign each hidden neuron a center pixel (cx, cy). Its synapses receive from pixels
within a radius r. Each of the n_basal_dendrites * dendrites_per_branch dendrites
handles a different angular sector or spatial scale.

- Approximates a convolutional layer without weight sharing
- Natural fit for early visual processing; spatially adjacent inputs activate the same dendrite
- x position along dendrite could map to distance from center (closer = lower x, more distal = higher x)
- Harder to implement: need to compute neighborhoods and partition them across dendrites
- With r≈6 (area ~113 pixels) and 96 synapses per neuron, coverage is tight

### Option C: Topographic hidden layer

The hidden layer has a 2D spatial arrangement (e.g., 20×20 = 400 neurons). Each
neuron connects to the corresponding 4×4 pixel patch plus noise. This mirrors
a retinotopic map.

Not necessary for MNIST but gives a natural basis for visualizing learned weights.

**Recommendation for first pass: Option A.** The asymmetric dendritic dynamics
(gamma term) will still learn structure because active pixel-synapse pairs
will reinforce each other via co-activity. Add spatial structure later if
there's a reason to visualize or constrain it.

---

## Hidden layer sizing

visual_mnist config: 6 basal × 8 dendrites_per_branch × 16 synapses = 768 synapses/neuron.

For N hidden neurons:
- Total synapses: 768N
- Total dendrites: 48N
- Memory (weights + alphas + xs + last_events): 4 * 768N bytes = ~3N KB

N=100: ~300 KB synapse data, fast to iterate
N=500: ~1.5 MB synapse data, still fine
N=1000: ~3 MB, start to think about cache behavior

Start with N=200. Enough representational capacity for MNIST, fast enough to
iterate on learning dynamics.

---

## Threshold math

**Dendrite:** threshold = 8000, delta per FORWARD_AP (no gamma) = w_i * 1.

If w_i ≈ 8 (mid-range positive initial weight), delta ≈ 8 per event.
To cross threshold: 8000 / 8 = 1000 events — far too many for 16 synapses in
a short trial.

With gamma amplification:
- If one neighboring synapse (dx=16) has alpha=200: shift_decay_u8(200, 16, 4) = 100
- gamma = 100, delta = 8 * 101 = 808
- Threshold crossing: 8000 / 808 ≈ 10 events

With full activity (all 15 neighbors active, alpha≈200, spread across dx=5 to dx=120):
- gamma ≈ Σ shift_decay_u8(200, dx, 4) for dx in {5,10,...,75}
- At dx=5: 169, dx=10: 149, dx=20: 113, dx=40: 63, dx=80: 13 → rough total ≈ 500
- delta = 8 * 501 = 4008 per event
- Threshold crossing: 2 events

So the dendrite threshold of 8000 is calibrated for coincident activity across
many synapses — it won't fire from a single synapse repeatedly, but will fire
quickly when the full dendrite is activated together. This is the NMDA-like
coincidence detection the code comments describe.

Consequence: **the trial window needs to be long enough for alpha to build up**
across multiple spikes before the gamma term becomes meaningful.
alpha reaches H_ALPHA=30 after about 1 spike (ALPHA_BOOST=64 per FORWARD_AP).
Full alpha (200+) requires 3-4 spikes. With rate coding at 50% of max rate
and a pixel brightness of 128 (50% of max), one synapse fires every ~2 ticks.
Alpha saturates around tick 8. So the dendrite can start integrating meaningfully
after ~10 ticks, and a trial window of 100-200 ticks should work.

The threshold may still need empirical tuning once the trial loop is running.

---

## Synapse x distribution

mean_synapse_x=128, std_synapse_x=50 → most x values in [78, 178] on a [0,255] scale.
X_DECAY=4 → gamma contribution halves every 16 x-units.

Effective gamma range: synapses within dx ≈ 48 of each other contribute meaningfully
(at dx=48, weight = shift_decay_u8(200, 48, 4) ≈ 6 — small but nonzero).

With std=50 and 16 synapses per dendrite, the typical inter-synapse gap is about
50/16 ≈ 3 x-units. Nearly all neighbors contribute to gamma. This means most
synapses on an active dendrite get amplified by their neighbors — good for learning,
but means gamma ≈ constant across the dendrite when all synapses are active.

If you want position-dependent selectivity (proximal synapses more influential
than distal), consider increasing std_synapse_x or spreading x values more
uniformly across [0,255]. With uniform spacing on 16 synapses: gap = 16 x-units,
so contributions drop to ~50% per step. That would give the distal amplification
term more bite.

---

## Feedback path and apical dendrites

The STDP learning rule works via bursting: the output neuron must burst (high beta)
when the correct class fires, which drives LTP in all active synapses via
`update_weight(burst_term > 0)`.

There are two ways to get the correct output neuron to burst:

**Option 1: Direct output injection (no apical)**
Push FORWARD_AP events directly into the output neuron's dendrites using the label.
No apical dendrites needed. Simpler, but the feedback doesn't modulate the hidden layer.

**Option 2: Apical feedback through hidden layer**
The output neuron's axon sends FORWARD_AP events to apical synapses of hidden neurons.
`handle_apical_fb` then multiplicatively boosts hidden soma potentials — biologically
this is "top-down attention" from higher cortical areas.

This is what the existing `handle_apical_fb` function implements. Requires:
- visual_mnist to have n_apical_dendrites set (currently None)
- A second set of synapse arrays for the apical compartment
- Axon connectivity from output layer back to hidden layer apical synapses

Option 1 is sufficient for basic classification. Option 2 enables the more
biologically realistic Burst-Dependent Plasticity dynamics the architecture was
designed for.

---

## Output layer

10 neurons, one per digit class. Recommended config:
- 1 basal dendrite cluster (n_basal_dendrites=2, dendrites_per_branch=4)
- More synapses per dendrite (e.g., 32) to receive from a wider hidden population
- Lower threshold: soma_threshold ≈ 5-10 (needs to fire readily during inference)
- Higher learning rate (or no learning — the hidden layer does the representation work)
- No apical dendrites

The output layer receives FORWARD_AP from hidden layer axons. Each output neuron
should represent one class; during training the correct one is driven to burst.

---

## Things still unclear

- **Weight initialization sign:** random weights in U(-8, 8) means ~50% inhibitory
  synapses from the start. This may prevent early dendrite firing. Consider
  starting with U(0, 8) and letting LTD drive weights negative over training.

- **Learning rate calibration:** MSLR=120 ensures the max delta fits in i8.
  But visual_mnist uses lr=256 which is above MSLR. Check whether this causes
  delta to consistently underflow to 0 for typical alpha/beta values.
  delta = burst_term * alpha / lr. With burst_term=6, alpha=200, lr=256: delta=4.
  With lr=120: delta=10. The current lr=256 gives slow but valid updates.

- **No lateral inhibition:** the current event system has no mechanism for
  winner-take-all competition between hidden neurons. Without it, many neurons
  may respond to the same input and not specialize. May need explicit inhibitory
  connections or beta-based suppression later.
