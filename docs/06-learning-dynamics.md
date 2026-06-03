# 6. Learning dynamics

This chapter is the biophysics from [chapter 1](01-theory.md), realized as the
**fat primitives** in `src/neuron/{synapse,dendrite,soma}.rs` plus the **thin
handlers** in `src/network/event/handlers.rs` that route between them. The
dispatch plumbing is [chapter 5](05-event-system.md); the decay primitive
(`shift_decay`) is [chapter 4](04-math-primitives.md). Read those first.

## 6.0 The separation rule: fat primitives, thin handlers

The codebase draws a hard line:

- A `neuron/update_*` **primitive** owns the *complete local physics* of one
  component — decay/leak, integrate, threshold, reset — and returns a verdict. It
  never touches the `EventProducer`.
- A `handlers.rs` **handler** does *only event routing*: scope the SoA slices,
  call the primitive, translate its verdict into emitted events. No physics.

By the slice-scoping rule ([chapter 5.3](05-event-system.md)), every function here
receives slices already narrowed to the relevant neuron or dendrite and uses
**local** indices.

## 6.1 Synaptic alpha — the eligibility trace

```rust
// src/neuron/synapse.rs
pub fn update_synapse_alpha(s, timestamp, synapse_alphas, synapse_last_events) -> u8 {
    let elapsed = timestamp.wrapping_sub(synapse_last_events[s]);          // lazy decay (chapter 2.3)
    let alpha   = shift_decay_u8(synapse_alphas[s], elapsed, ALPHA_DECAY); // half-life 2^11 = 2048 ticks
    synapse_alphas[s] = alpha;
    synapse_last_events[s] = timestamp;
    alpha
}
```

The workhorse: *whenever* a synapse is touched, bring its `alpha` up to date by
decaying from `last_event` to `now`, then stamp `now`. `wrapping_sub` makes the
`u16` timestamp wrap harmless ([chapter 2.4](02-architecture.md)). Every other
function calls this before reading `alpha`. (`ALPHA_DECAY = 11` now — a slow
~2048-tick half-life, so eligibility lingers across a trial.)

## 6.2 Dendritic integration — gamma, leak, and the burst-scaled EPSP

`update_dendrite_activity` owns the dendrite's full local state machine:
integrate, leak, then either fire+reset (basal) or produce a graded plateau
(apical). It returns a `DendriteOutput` the handler routes.

```rust
// src/neuron/dendrite.rs
pub enum DendriteOutput {
    Basal { fired: bool },   // hard threshold; fired ⇒ caller emits DENDRITIC_SPIKE (V_B was reset to 0)
    Apical { plateau: i16 }, // graded; plateau ⇒ caller emits SOMA_SIGNAL (V_B left intact, it leaks)
}

pub fn update_dendrite_activity(
    s_idx, timestamp, burst, live_end,                       // burst = presynaptic AP count
    synapse_xs, synapse_alphas, synapse_weights, synapse_last_events,
    dendrite_activity, dendrite_last_event, dendrite_threshold, is_apical,
) -> DendriteOutput {
    let x_i = synapse_xs[s_idx];
    let w_i = synapse_weights[s_idx];

    let mut gamma: u16 = 0;
    for j in (s_idx + 1)..live_end {                          // only MORE-DISTAL, LIVE synapses
        let alpha_j = update_synapse_alpha(j, timestamp, synapse_alphas, synapse_last_events);
        let dx = synapse_xs[j] - x_i;
        gamma = gamma.saturating_add(shift_decay_u8(alpha_j, dx as u16, X_DECAY) as u16);
    }

    let k = if is_apical { APICAL_DECAY } else { BASAL_DECAY };
    let elapsed = timestamp.wrapping_sub(*dendrite_last_event);
    *dendrite_last_event = timestamp;
    let decayed = shift_decay(*dendrite_activity, elapsed, k);             // leak V_B since last event

    // ΔV_B = burst · w_i · (1 + gamma), computed in i32 then clamped to i16
    let gain = 1i32 + gamma as i32;
    let update_term = ((w_i as i32) * gain * burst as i32).clamp(i16::MIN as i32, i16::MAX as i32) as i16;
    *dendrite_activity = decayed.saturating_add_signed(update_term);

    if is_apical {
        DendriteOutput::Apical { plateau: apical_plateau(*dendrite_activity, dendrite_threshold, APICAL_DV_S, APICAL_SLOPE_K) }
    } else if *dendrite_activity >= dendrite_threshold {
        *dendrite_activity = 0;                                            // hard reset on basal spike
        DendriteOutput::Basal { fired: true }
    } else {
        DendriteOutput::Basal { fired: false }
    }
}
```

This is [chapter 1.4](01-theory.md)'s formula, now with three refinements over
the older design:

- **The gamma loop bounds by `live_end`, not slice length.** `live_end = base +
  live_count` is the count of bound synapses packed at the front of the
  fixed-slot block ([chapter 7.4](07-network-construction.md)); the padded dead
  tail (whatever garbage `alpha` it holds) is never read. Because the slice is one
  dendrite's synapses **sorted ascending by `x`** ([chapter 3](03-data-model.md)),
  "after `s_idx`" means "more distal" (`x_j > x_i`) — directionality from the sort
  order plus the loop bound. `dx` is fed to `shift_decay_u8` as if it were elapsed
  *time*, with `k = X_DECAY = 4`: same decay machine, spatial axis.
- **The branch voltage now leaks.** Before integrating, `*dendrite_activity` is
  decayed by `shift_decay` over the elapsed time, with a compartment-specific
  half-life: `BASAL_DECAY` (2⁹ = 512 ticks) or `APICAL_DECAY` (2¹¹ = 2048 ticks).
  Older docs noted "`dendrite_activity` has no decay"; that gap is **closed** —
  the dendrite forgets on its own between events.
- **Burst scales the EPSP.** `ΔV_B = burst · w_i · (1 + γ)`. The presynaptic burst
  count (threaded through the payload, [chapter 5](05-event-system.md)) multiplies
  the depolarization, so a bursting upstream neuron drives its targets harder. The
  product is computed in `i32` (it can far exceed `i16`: `w_i` up to ±127,
  `1+γ` up to ~65 536, burst up to 127) then clamped to `i16`.

### Basal vs. apical: two different verdicts

The same integration produces fundamentally different outputs, which is why the
primitive returns a compartment-tagged enum rather than a bool:

- **Basal** is a hard threshold. Cross `dendrite_threshold` → reset `V_B` to 0 and
  report `fired: true` (the handler emits a `DENDRITIC_SPIKE`). The reset is the
  model's refractory/normalization mechanism.
- **Apical** is graded. `V_B` is **not** reset (it leaks); the primitive returns a
  continuous `plateau` voltage via `apical_plateau` (§6.3), which the handler
  delivers to the soma as a `SOMA_SIGNAL`.

## 6.3 `apical_plateau` — the sigmoidal transfer function

The apical compartment follows Payeur et al. (2021): instead of a discrete spike,
it produces a *graded* somatic depolarization, a logistic function of branch
voltage.

```rust
// σ^(ap)(V_B) = δV_S / (1 + exp(−κ(V_B − θ_B)))
pub fn apical_plateau(v_b: u16, theta: u16, dv_s: i16, k: u8) -> i16 {
    const UNIT: i32 = 256;
    let u  = (v_b as i32 - theta as i32).unsigned_abs() as u16;  // |V_B − θ_B|
    let d  = shift_decay(UNIT as u16, u, k) as i32;              // D = 256·2^(−u/2^k) ∈ [0, 256]
    let dv = dv_s as i32;
    let out = if v_b >= theta { dv * UNIT / (UNIT + d) }         // upper half: σ ≥ ½ → [δV_S/2, δV_S]
              else            { dv * d    / (UNIT + d) };        // lower half: σ < ½ → [0, δV_S/2]
    out.min(i16::MAX as i32 - (i8::MAX as i32 + 1)) as i16       // ceiling: safe to add to an i8 soma
}
```

There is **no `exp` call**: the logistic core `e^(−κ·)` *is* `shift_decay`, and
the `V_B < θ_B` branch uses the logistic symmetry `σ(−x) = 1 − σ(x)`. `θ_B` reuses
`dendrite_thresholds[d]` (the half-activation point); `δV_S = APICAL_DV_S` is the
plateau ceiling; `κ = ln2 / 2^k` with `k = APICAL_SLOPE_K` sets the slope. The
result is clamped below `i16::MAX − 128` so it can be added to an `i8` soma
potential without overflow. `APICAL_DV_S` and `APICAL_SLOPE_K` are explicitly
untuned placeholders.

This graded, multiplicative-feeling drive is the structural basis of
Burst-Dependent Plasticity: a neuron already depolarized by feedforward basal
input, when its apical branch is also driven from above, gets pushed
over threshold repeatedly in one integration (the soma's `new_v / threshold`
burst, §6.5) — exactly the high-`beta` state that gates LTP.

## 6.4 The four handlers

Each is pure routing. Slices arrive pre-scoped from the loop
([chapter 5.3](05-event-system.md)).

### `handle_synapse_signal` — one AP delivery lands on a synapse

```rust
let alpha = update_synapse_alpha(s_idx, ...);
synapse_alphas[s_idx] = alpha.saturating_add(ALPHA_BOOST);    // +64: this synapse just fired
match update_dendrite_activity(s_idx, timestamp, burst, live_end, ...) {   // §6.2
    DendriteOutput::Basal { fired: true }  => producer.push(Event::spike(DENDRITIC_SPIKE, dendrite_idx, timestamp)),
    DendriteOutput::Basal { fired: false } => {}
    DendriteOutput::Apical { plateau }     => producer.push(Event::soma_signal(neuron_idx, timestamp, plateau)),
}
```

Boost the receiving synapse's eligibility, integrate its parent dendrite (EPSP
scaled by the presynaptic burst), and route the verdict: a basal spike becomes a
`DENDRITIC_SPIKE`; an apical plateau becomes a `SOMA_SIGNAL` carrying the graded
voltage. `is_apical` (from `dendrite_is_apical[d]`) is what selects the branch, so
one upstream axon drives whichever compartment its target slot belongs to.

### `handle_dendritic_spike` — a basal dendrite fired

```rust
let branch_constant = *dendrite_constant;
for s_idx in 0..synapse_alphas.len() {                        // reinforce synapses active at spike time
    let alpha = update_synapse_alpha(s_idx, ...);
    if alpha > H_ALPHA {
        synapse_alphas[s_idx] = alpha.saturating_add(branch_constant.unsigned_abs());
    }
}
producer.push(Event::soma_signal(neuron_idx, timestamp, branch_constant.max(1) as i16));
```

The sign of `dendrite_constant` selects compartment behavior at runtime
([chapter 3.1](03-data-model.md)):

- **Proximal / basal** (`constant > 0`): passes its magnitude to the soma as the
  `SOMA_SIGNAL` payload — a strong, direct feedforward contribution.
- **Distal** (`constant ≤ 0`): `max(1)` attenuates it to a weak `+1` at the soma,
  but the *alpha boost* (`unsigned_abs()`) is large — distal spikes do little to
  fire the soma directly but strongly reinforce local eligibility, an
  NMDA-plateau-like effect. This asymmetry is the point of two compartment types.

Note this handler now *signals* the soma via an event rather than mutating soma
state in place; the soma's own physics live entirely in `update_soma_potential`.

### `handle_soma_signal` — integrate a voltage delta at the soma

```rust
let burst = update_soma_potential(timestamp, neuron_idx, soma_potentials, soma_last_events,
                                  soma_thresholds, soma_betas, v_s);       // §6.5
if burst > 0 {
    producer.push(Event::with_payload(SOMATIC_SPIKE, neuron_idx, timestamp, burst as i16));
}
```

A voltage delta (from a dendritic spike or an apical plateau) is integrated
through the soma's state machine. If the soma bursts, emit a **single**
`SOMATIC_SPIKE` carrying the burst count as payload — downstream scales by it
rather than replaying *N* events.

### `handle_somatic_spike` — the BDP weight update and the axonal fan-out

```rust
// 1. BaP: back-propagating weight update over the neuron's OWN afferent synapses
for s_idx in 0..synapse_weights.len() {
    update_weight(timestamp, beta, soma_lr, s_idx, synapse_alphas, synapse_last_events, synapse_weights);
}
// 2. axonal output: enqueue one SYNAPSE_SIGNAL per downstream target synapse, carrying the burst
for &s in axon_targets {
    producer.push(Event::with_payload(SYNAPSE_SIGNAL, s, timestamp, burst as i16));
}
```

Two consequences of firing. **(1)** the back-propagating action potential drives
a BDP weight update across every synapse the neuron owns (the slice spans all its
dendrites) — one somatic spike is a global, eligibility-weighted credit
assignment across the whole dendritic tree. **(2)** the axonal AP fans out to
every downstream target as an independent `SYNAPSE_SIGNAL` (the parallelizable
push-only step from [chapter 5.3](05-event-system.md)).

Crucially, `beta` here is **read-only**. All of `beta`'s dynamics live in
`update_soma_potential` (§6.5), so the burst factor enters the weight update only
through `beta` and is **not** applied a second time — the BaP update is
β-driven, not burst-scaled.

## 6.5 The soma state machine — `update_soma_potential`

```rust
// src/neuron/soma.rs   (BETA_MAX = 63)
pub fn update_soma_potential(timestamp, so_idx, soma_potentials, soma_last_events,
                             soma_thresholds, soma_betas, v_s) -> u8 {
    let elapsed = timestamp.wrapping_sub(soma_last_events[so_idx]);
    let decayed_potential = shift_decay_i8(soma_potentials[so_idx], elapsed, SOMATIC_DECAY); // half-life 2^10
    let beta_decrement    = (elapsed / T_BETA).min(BETA_MAX as u16) as u8;                    // 1 per 500 ticks
    let beta              = soma_betas[so_idx].saturating_sub(beta_decrement);
    soma_last_events[so_idx] = timestamp;

    let threshold = soma_thresholds[so_idx];
    let new_v = decayed_potential as i16 + v_s;
    if threshold > 0 && new_v >= threshold as i16 {
        let burst = (new_v / threshold as i16) as u8;          // possibly several APs in one shot
        soma_potentials[so_idx] = SOMA_V_RESET;                // reset to −32, not 0
        soma_betas[so_idx]      = beta.saturating_add(burst).min(BETA_MAX);  // bursting reinforces beta
        burst
    } else {
        soma_potentials[so_idx] = new_v.clamp(i8::MIN as i16, i8::MAX as i16) as i8;
        soma_betas[so_idx]      = beta;                        // commit the lazy decay
        0
    }
}
```

This primitive owns **all** of `beta`'s dynamics (the older design split them
across a handler):

- **Both quantities leak lazily.** The potential decays with `SOMATIC_DECAY`
  (2¹⁰ = 1024 ticks); `beta` drops by 1 per `T_BETA = 500` ticks elapsed. The
  reset potential is `SOMA_V_RESET = −32` (a hyperpolarized refractory floor),
  not 0.
- **A burst is `new_v / threshold`.** A single large integration (a strong apical
  plateau, or a bigger `v_s`) can push the soma several multiples over threshold,
  yielding a burst of >1 in one event — the multi-AP shortcut that the
  payload-carrying `SOMATIC_SPIKE` then propagates without replaying events.
- **Bursting reinforces `beta`** by the burst size, capped at `BETA_MAX = 63`
  (the 6-bit counter). High `beta` ⇒ LTP gate open in §6.6.

## 6.6 `update_weight` — the BDP rule itself

```rust
// src/neuron/synapse.rs
pub fn update_weight(timestamp, beta, lr, s, alphas, last_events, weights) {
    let alpha = update_synapse_alpha(s, timestamp, alphas, last_events);
    if alpha <= H_ALPHA { return; }                            // only eligible synapses learn
    let burst_term = (beta as i16) - H_BETA;                   // >0 bursting → LTP, <0 → LTD
    let delta: i16 = burst_term * (alpha as i16) / lr;         // scaled by eligibility, divided by lr
    weights[s] = weights[s].saturating_add(delta.clamp(-127, 127) as i8);
}
```

Everything from [chapter 1.5](01-theory.md): the update's *sign* is the burst gate
(`beta` vs `H_BETA = 4`), its *magnitude* is the eligibility `alpha`, and its
*scale* is `lr` (with `MSLR` from [chapter 4.4](04-math-primitives.md) ensuring the
maximum step fits `i8`). Silent synapses (`alpha ≤ H_ALPHA`) are left untouched.
Run over every synapse a neuron owns by `handle_somatic_spike`, this is the global,
eligibility-weighted credit assignment that makes a burst a teaching signal.

---

Next: [chapter 7 — Network construction](07-network-construction.md), which
allocates the arrays all of this operates on — and which, unlike when these docs
were first written, **now exists**.
