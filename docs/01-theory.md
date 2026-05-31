# 1. Theory

This chapter describes the biological model and the learning rule, independent
of any code. Every later chapter is an implementation of something defined here.

## 1.1 Why spiking, and why dendrites

A standard artificial neuron is a dot-product followed by a nonlinearity: it
collapses all of its inputs into one scalar before doing anything interesting.
Biological pyramidal neurons do not. Input arrives on a **dendritic tree** whose
branches perform *local* nonlinear integration before the soma ever sees a
summary. The same total input produces different somatic responses depending on
*where on the tree* and *in what order* it arrived.

This simulator takes that structure seriously. A neuron is:

```
              soma  ─── axon ──▶ (forward action potentials to other neurons)
               ▲
       ┌───────┼───────┐
   dendrite  dendrite  dendrite        each dendrite integrates its own synapses
   ┌──┬──┐    ┌──┬──┐                  and can spike independently (NMDA-like)
   s  s  s    s  s  s                  synapses, ordered along the dendrite by x
```

Two compartment classes matter:

- **Basal dendrites** — proximal, feedforward. Their activity pushes more or
  less directly onto the soma's membrane potential.
- **Apical dendrites** — distal, top-down. They carry feedback from "higher"
  areas and *modulate* the soma multiplicatively rather than additively. This is
  the substrate for the burst signal that drives learning (§1.5).

## 1.2 The state variables

The whole model is built from four quantities. Keep these names in mind; they
recur in every chapter.

| Symbol | Lives on | Meaning | Decays? |
| ------ | -------- | ------- | ------- |
| `alpha` | each synapse | recent synaptic activity (an eligibility trace) | yes — exponentially, slowly |
| `beta`  | each soma | burst counter: how many somatic spikes happened recently | yes — by 1 every fixed interval |
| `x`     | each synapse | position along its dendrite (proximal → distal) | no — structural |
| `gamma` | computed per event | how much a synapse's neighbors amplify it | n/a — recomputed |

`x` is the key structural idea. Synapses are not interchangeable: each sits at a
position `x` along its dendrite. Distal synapses (high `x`) amplify proximal ones
(low `x`). This makes integration **ordered** and **asymmetric**, which is what
lets a dendrite be sensitive to input *patterns*, not just input *sums*.

## 1.3 Synaptic activity: `alpha`

When a synapse receives a presynaptic action potential, its `alpha` is boosted.
Between events, `alpha` decays exponentially. So `alpha` is high exactly when a
synapse has been active *recently and repeatedly* — it is an **eligibility
trace**, the memory of "this synapse just participated."

A synapse only counts as meaningfully active once `alpha` rises above a threshold
`H_ALPHA`. Below that it contributes nothing to integration or learning.

## 1.4 Dendritic integration: `gamma` and ordered amplification

When a synapse `i` is driven, the depolarization it contributes to its dendrite
is not just its weight. It is its weight scaled by an amplification term that
depends on the activity of its **more-distal neighbors**:

```
delta_V_i = w_i * (1 + gamma_i)
gamma_i   = Σ  decay(alpha_j , x_j − x_i)    over all j on the same dendrite with x_j > x_i
```

Read this carefully:

- Only synapses *more distal* than `i` (`x_j > x_i`) contribute to `gamma_i`. The
  influence is **directional**, proximal-ward.
- A distal neighbor contributes in proportion to *its* recent activity `alpha_j`,
  attenuated by the distance `x_j − x_i`. Far-apart synapses barely interact;
  adjacent ones strongly co-amplify.
- If no distal neighbor is active, `gamma_i = 0` and the synapse contributes just
  `w_i`. The amplification is a *coincidence bonus*.

Biologically this stands in for the way a distal dendritic depolarization
(an NMDA spike or a back-propagating event) opens the door for proximal inputs to
have outsized effect. Computationally it means a dendrite responds most strongly
when a *spatially ordered sequence* of its synapses is co-active — sequence
selectivity, for free, from the geometry.

When a dendrite's accumulated activity crosses its threshold, it emits a
**dendritic spike**, which propagates toward the soma.

## 1.5 Burst-dependent plasticity: `beta` and the learning rule

The soma integrates dendritic spikes. When it crosses threshold it emits a
**somatic spike** (an action potential), which travels out the axon as a forward
AP to downstream neurons.

`beta` counts how many somatic spikes a neuron has produced in the recent past.
A high `beta` means the neuron is **bursting**. Bursting is the teaching signal:
in cortex, bursts (driven by coincident feedforward *and* top-down apical input)
are what gate plasticity.

The weight update on a somatic spike is:

```
burst_term = beta − H_BETA
delta_w    = burst_term * alpha / learning_rate          (only if alpha > H_ALPHA)
```

- **Bursting** (`beta > H_BETA`) → `burst_term > 0` → **LTP** (potentiation):
  recently-active synapses (`alpha` high) are strengthened.
- **Not bursting** (`beta < H_BETA`) → `burst_term < 0` → **LTD** (depression):
  the same synapses are weakened.
- The magnitude scales with `alpha`: only synapses that actually participated
  (high eligibility) move. Silent synapses are left alone.

This is the BDP rule. It is STDP-like, but the sign of the update is gated by a
*burst* state rather than by precise pre/post spike timing. The eligibility
trace `alpha` decouples "which synapses were involved" from "when the teaching
signal arrives," so the teaching signal can lag the activity by tens of ticks.

## 1.6 How a teaching signal reaches the neuron

Two routes make a neuron burst, and the choice is a live design decision
(revisited in [chapter 8](08-mnist-pipeline.md)):

1. **Direct drive** — inject forward APs straight into the target neuron so it
   fires repeatedly and `beta` climbs. Simple; no apical machinery.
2. **Apical feedback** — a higher layer drives the neuron's *apical* synapses,
   which multiplicatively amplify the soma and produce a burst of somatic spikes.
   This is the biologically faithful path and the reason apical compartments
   exist in the model.

## 1.7 Time

There is **no discrete network clock advancing state step by step.** Every state
variable carries a "last touched" timestamp, and decay is computed *lazily* —
only when the variable is next read — from the elapsed time since that timestamp.
A synapse idle for 200 ticks does not get 200 decay updates; it gets one, the
moment something touches it. This is the single most important consequence of the
event-driven architecture, and [chapter 2](02-architecture.md) explains why the
design is built around it.

---

Next: [chapter 2 — Architecture choices](02-architecture.md), where these
concepts get mapped onto a memory layout and an execution model chosen for the
GPU.
