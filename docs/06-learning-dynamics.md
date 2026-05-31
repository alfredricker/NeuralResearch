# 6. Learning dynamics

This chapter is the biophysics from [chapter 1](01-theory.md), realized as the
four handlers in `src/network/event/handlers.rs` plus the two synapse/dendrite
leaf functions they call. The dispatch plumbing is
[chapter 5](05-event-system.md); the primitives (`shift_decay`) are
[chapter 4](04-math-primitives.md). Read those first.

By the slice-scoping rule ([chapter 5.3](05-event-system.md)), every function
here receives slices already narrowed to the relevant neuron or dendrite and
uses **local** indices.

## 6.1 Synaptic alpha — the eligibility trace

```rust
// src/neuron/synapse.rs
pub fn update_synapse_alpha(s, timestamp, synapse_alphas, synapse_last_events) -> u8 {
    let elapsed = timestamp.wrapping_sub(synapse_last_events[s]);   // lazy decay (chapter 2.3)
    let alpha   = shift_decay_u8(synapse_alphas[s], elapsed, ALPHA_DECAY);  // half-life 256 ticks
    synapse_alphas[s] = alpha;
    synapse_last_events[s] = timestamp;
    alpha
}
```

This is the workhorse: *whenever* a synapse is touched, bring its `alpha` up to
date by decaying from `last_event` to `now`, then stamp `now`. `wrapping_sub`
makes the `u16` timestamp wrap harmless ([chapter 2.4](02-architecture.md)).
Every other function calls this before reading `alpha`.

## 6.2 Dendritic integration — the gamma sum

```rust
// src/neuron/dendrite.rs
pub fn update_dendrite_activity(s_idx, timestamp, synapse_xs, synapse_alphas,
                                synapse_weights, synapse_last_events) -> i16 {
    let x_i = synapse_xs[s_idx];
    let w_i = synapse_weights[s_idx];
    let mut gamma: u16 = 0;
    for j in (s_idx + 1)..synapse_xs.len() {              // only MORE-DISTAL synapses (x sorted ascending)
        let alpha_j = update_synapse_alpha(j, timestamp, synapse_alphas, synapse_last_events);
        let dx = synapse_xs[j] - x_i;
        gamma = gamma.saturating_add(shift_decay_u8(alpha_j, dx as u16, X_DECAY) as u16);
    }
    (w_i as i16).saturating_mul(1 + gamma.min(i16::MAX as u16) as i16)   // delta_V = w_i * (1 + gamma)
}
```

This is [chapter 1.4](01-theory.md)'s formula exactly. Key points:

- The loop runs `(s_idx + 1)..len` — i.e. only synapses *after* `s_idx` in the
  slice. Because the slice is one dendrite's synapses **sorted ascending by `x`**
  ([chapter 3](03-data-model.md)), "after" means "more distal" (`x_j > x_i`). The
  directionality is encoded purely by the sort order plus the loop bound.
- `dx` is fed to `shift_decay_u8` as if it were elapsed *time*, with
  `k = X_DECAY = 4`: the same decay machine, applied on the spatial axis
  ([chapter 4.1](04-math-primitives.md)). Distal neighbors attenuate with
  distance.
- The return is `i16` because `w_i` is signed (inhibitory synapses give negative
  `delta_V`) and the amplified product can exceed `i8`. The caller accumulates it
  into the `u16` dendrite activity with `saturating_add_signed`.

> **Design note tying back to our slot discussion.** The loop bounds itself by
> `synapse_xs.len()`, which is correct *today* only because
> [chapter 5](05-event-system.md) hands it a slice already trimmed to the
> dendrite's live synapses. The moment [chapter 7](07-network-construction.md)'s
> **fixed-slot** layout is adopted — where each dendrite block is padded with
> dead slots at the tail — this loop must instead bound by `live_count`, i.e.
> iterate `(s_idx + 1)..live_end`. The bound's *origin is the dendrite base*, not
> `s_idx`. This is recorded as a concrete change-on-migration in
> [chapter 7](07-network-construction.md).

## 6.3 `handle_forward_ap` — input arrives at a synapse

Fired once per axon target ([chapter 5.3](05-event-system.md)).

```rust
let alpha = update_synapse_alpha(s_idx, ...);
synapse_alphas[s_idx] = alpha.saturating_add(ALPHA_BOOST);   // +64: this synapse just fired
let delta = update_dendrite_activity(s_idx, ...);            // gamma integration (§6.2)
*dendrite_activity = dendrite_activity.saturating_add_signed(delta);
if *dendrite_activity >= *dendrite_threshold {
    *dendrite_activity = 0;                                  // reset on spike
    producer.push(Event { event_type: DENDRITIC_SPIKE, source: dendrite_idx, timestamp });
}
```

Boost the firing synapse's eligibility, recompute the dendrite's depolarization
including the amplification from its active distal neighbors, and emit a dendritic
spike if threshold is crossed. The reset-to-0-on-spike is the model's
refractory/normalization mechanism.

> **Gap.** `dendrite_activity` has **no decay** — it only ever resets to 0 on a
> spike. Between trials it must be explicitly cleared, or it accumulates stale
> depolarization. See [chapters 8](08-mnist-pipeline.md) and
> [9](09-gaps-and-open-questions.md).

## 6.4 `handle_dendritic_spike` — propagation to the soma

```rust
let branch_constant = *dendrite_constant;
let soma_delta: i8   = branch_constant.max(1);              // proximal: scale; distal: clamp to +1
*soma_potential = soma_potential.saturating_add(soma_delta);

for s_idx in 0..synapse_alphas.len() {                      // reinforce synapses active at spike time
    let alpha = update_synapse_alpha(s_idx, ...);
    if alpha > H_ALPHA {
        synapse_alphas[s_idx] = alpha.saturating_add(branch_constant.unsigned_abs());
    }
}

if *soma_potential >= *soma_threshold {
    *soma_potential = 0;
    producer.push(Event { event_type: SOMATIC_SPIKE, source: neuron_idx, timestamp });
}
```

The sign of `dendrite_constant` selects compartment behavior at runtime
([chapter 3.1](03-data-model.md)):

- **Proximal / basal** (`constant > 0`): pushes `constant` onto the soma — a
  strong, direct feedforward contribution.
- **Distal / apical** (`constant ≤ 0`): `max(1)` attenuates it to a weak `+1` on
  the soma, but the *alpha boost* (`unsigned_abs()`) is large — i.e. distal
  spikes do little to fire the soma directly but strongly reinforce local
  eligibility, an NMDA-plateau-like effect. This asymmetry is the whole point of
  having two compartment types.

## 6.5 `handle_somatic_spike` — the BDP weight update

```rust
let elapsed     = timestamp.wrapping_sub(*soma_last_event);
let decrements  = (elapsed / T_BETA).min(15) as u8;        // beta decays 1 per T_BETA=500 ticks
*beta = beta.saturating_sub(decrements).saturating_add(1).min(63);   // +1 for this spike, cap 63
*soma_last_event = timestamp;

for s_idx in 0..synapse_weights.len() {                    // ALL synapses of the neuron (chapter 5.3)
    update_weight(timestamp, *beta, *soma_lr, s_idx, synapse_alphas, synapse_last_events, synapse_weights);
}
producer.push(Event { event_type: FORWARD_AP, source: neuron_idx, timestamp });
```

with the per-synapse rule ([chapter 1.5](01-theory.md)):

```rust
// src/neuron/synapse.rs
pub fn update_weight(timestamp, beta, lr, s, alphas, last_events, weights) {
    let alpha = update_synapse_alpha(s, timestamp, alphas, last_events);
    if alpha <= H_ALPHA { return; }                        // only eligible synapses learn
    let burst_term = (beta as i16) - H_BETA;               // >0 bursting → LTP, <0 → LTD
    let delta: i16 = burst_term * (alpha as i16) / lr;     // scaled by eligibility, divided by lr
    weights[s] = weights[s].saturating_add(delta.clamp(-127, 127) as i8);
}
```

Everything from [chapter 1.5](01-theory.md) is here: `beta` first decays for
elapsed time then increments for this spike (and caps at 63); the weight step's
sign is the burst gate, its magnitude is the eligibility `alpha`, and its scale is
set by `lr` (with `MSLR` from [chapter 4.4](04-math-primitives.md) ensuring the
maximum fits `i8`). Crucially the update runs over **every synapse the neuron
owns** (the slice spans all its dendrites), so one somatic spike is a global,
eligibility-weighted credit assignment across the whole dendritic tree. Finally
the neuron fires a forward AP downstream.

## 6.6 `handle_apical_fb` — multiplicative top-down feedback

The route that makes a neuron burst from above ([chapter 1.6](01-theory.md)):

```rust
let alpha = update_synapse_alpha(s_idx, ...);
let effective_alpha = alpha.saturating_add(axon_constant);
let v_s   = (*soma_potential).max(0) as i32;
let new_v = *soma_potential as i32 + effective_alpha as i32 * v_s;   // MULTIPLICATIVE in v_s

let burst_count = new_v / soma_threshold as i32;                     // possibly several spikes
*soma_potential = (new_v % soma_threshold as i32) as i8;             // carry the remainder
for _ in 0..burst_count {
    producer.push(Event { event_type: SOMATIC_SPIKE, source: neuron_idx, timestamp });
}
```

Unlike basal input (additive, §6.4), apical feedback **scales the existing soma
potential** (`effective_alpha · v_s`). A neuron already depolarized by
feedforward drive gets violently amplified by coincident top-down input and emits
a *burst* of somatic spikes in one shot — exactly the high-`beta` state that
gates LTP in §6.5. This is the structural basis of Burst-Dependent Plasticity:
learning happens when bottom-up and top-down signals coincide.

> **Gaps.** `handle_apical_fb` takes an `axon_constant: u8`, but no axon-constant
> array exists in [chapter 3](03-data-model.md)'s data model, and no event type
> routes to this handler — the dispatch loop ([chapter 5.3](05-event-system.md))
> has no apical arm. Apical feedback is implemented at the leaf level but not yet
> connected to the event system or the allocator. See
> [chapters 8](08-mnist-pipeline.md) and [9](09-gaps-and-open-questions.md).

---

Next: [chapter 7 — Network construction](07-network-construction.md), which
allocates the arrays all of this operates on.
