# 4. Math primitives

These are the leaf computations in `src/math/`. They are pure, allocation-free,
fully unit-tested, and chosen so they translate directly to GPU kernels. Every
dynamic in [chapter 6](06-learning-dynamics.md) is built from them.

## 4.1 `shift_decay` — O(1) integer exponential decay

The single most-used function. It approximates `v · 2^(−t / 2^k)` — i.e. a value
that halves every `2^k` time steps — using only integer shifts and one multiply.
No FPU, no `exp()`.

```rust
pub fn shift_decay(v: u16, t: u16, k: u8) -> u16 {
    let shifts = t >> k;                       // how many whole half-lives elapsed
    if shifts >= 16 { return 0; }              // 16 halvings ⇒ underflow to 0

    let remainder = t & ((1 << k) - 1);        // fractional part within a half-life
    let v_current = v >> shifts;               // integer halvings
    let v_next    = v_current >> 1;            // value after one more halving
    let diff      = v_current - v_next;        // the span we interpolate across
    let drop      = ((diff as u32 * remainder as u32) >> k) as u16;  // linear interp; u32 avoids overflow
    v_current - drop
}

pub fn shift_decay_u8(v: u8, t: u16, k: u8) -> u8 {   // u8 convenience wrapper
    shift_decay(v as u16, t, k) as u8
}
```

How it works:

1. **Whole half-lives** are exact right-shifts: after `shifts = t >> k` halvings,
   the value is `v >> shifts`.
2. **The fractional part** (`remainder` ticks into the current half-life) is
   filled by *linearly interpolating* between `v_current` and `v_next`. Real
   exponential decay is convex, so a straight-line interpolation slightly
   over-estimates the drop early in the interval — a deliberate, cheap
   approximation, accurate enough for an 8-bit state variable.
3. **The `as u32` cast is critical**: `diff * remainder` can exceed `u16` before
   the `>> k` brings it back down. The source comment flags this explicitly.
4. After ~16 half-lives the value is indistinguishable from 0, so the function
   short-circuits.

Worked example (from the tests): `shift_decay_u8(200, 5, 4)` — `k=4` so half-life
is 16; `shifts=0`, `remainder=5`, `v_current=200`, `v_next=100`, `diff=100`,
`drop = (100·5)>>4 = 31`, result `169`.

This one function powers **both** decays in the model, parameterized by `k`:

- **Alpha decay** uses `k = ALPHA_DECAY = 8` → half-life 256 ticks. Slow: an
  active synapse stays "eligible" for hundreds of ticks after its last spike.
- **Distance attenuation in `gamma`** uses `k = X_DECAY = 4` → halves every 16
  `x`-units. Here `t` is not time but the distance `x_j − x_i` between two
  synapses ([chapter 6](06-learning-dynamics.md)). Same math, spatial axis.

## 4.2 The samplers — `SamplerU8` / `SamplerI8` (`src/math/sample.rs`)

Network construction needs to draw thousands of values from (discretized) normal
distributions — synapse positions, dendrite constants, etc. The samplers
precompute a **Vose alias table** so each draw is O(1): two random reads and one
comparison.

```rust
pub struct SamplerU8 { prob: [f32; 256], alias: [u8; 256] }

impl SamplerU8 {
    pub fn new(mean: u8, std: u8) -> Self { /* discretize N(mean,std) over 256 bins, build alias table */ }
    pub fn sample(&self, rng: &mut impl Rng) -> u8 {
        let i = rng.random::<u8>() as usize;   // pick a bin — u8 ⇒ no modulo bias
        let u = rng.random::<f32>();
        if u < self.prob[i] { i as u8 } else { self.alias[i] }
    }
}
```

Design notes:

- **Exactly 256 bins.** Bin selection is a raw `u8`, so there is *zero* modulo
  bias — a deliberate match to the 8-bit state types from
  [chapter 2](02-architecture.md).
- `SamplerI8` is identical but maps bin `i → (i − 128)`, so bin 0 is −128 and bin
  255 is +127.
- `std == 0` collapses to a delta at `mean` (deterministic) — useful for fixed
  parameters.
- Build is O(256), sampling is O(1). You construct one sampler per `NeuronConfig`
  field ([chapter 7](07-network-construction.md)) and draw from it for every
  neuron in the population.

For simple bounded draws there are also `sample_u8_uniform(lo, hi, rng)` and
`sample_i8_uniform(lo, hi, rng)` — used for weight initialization, e.g. `U(−8, 8)`.

The tests pin the statistical behavior (observed mean/std within tolerance) and
two `visual_mnist`-specific expectations: dendrite-constant `N(60, 8)` stays
strictly positive (so those dendrites act proximal), and synapse-`x` `N(128, 50)`
spans most of `[0, 255]`.

## 4.3 `MidPoint` (`src/math/midpoint.rs`)

A tiny trait returning the midpoint (`1 << (BITS − 1)`) of an unsigned integer
type — `128` for `u8`, `32768` for `u16`, etc. Intended for default
initialization ("start a weight/threshold at the middle of its range"). Currently
the free `midpoint::<T>()` wrapper is private and unused; the trait is the usable
surface.

## 4.4 The constants (`src/constants.rs`)

Every tuning knob in one place. They are the bridge between
[chapter 1](01-theory.md)'s symbols and the 8-bit reality of
[chapter 2](02-architecture.md).

| Constant | Value | Role |
| -------- | ----- | ---- |
| `T_BETA` | 500 | ticks elapsed to subtract 1 from `beta` |
| `H_ALPHA` | 30 | min `alpha` for a synapse to affect weights/boost |
| `H_BETA` | 4 (`i16`) | burst threshold; `burst_term = beta − H_BETA` |
| `ALPHA_DECAY` | 8 | `alpha` half-life = `2^8` = 256 ticks |
| `X_DECAY` | 4 | gamma distance half-life = `2^4` = 16 `x`-units |
| `ALPHA_BOOST` | 64 | `alpha` added when a synapse receives a forward AP |
| `MSLR` | 120 | minimum synaptic learning rate (see below) |

**`MSLR` derivation.** Weight delta is `burst_term · alpha / lr`. The maxima are
`burst_term_max = 2^6 − 5` (since `beta` is capped at 63 and `H_BETA = 4`, this is
the largest documented burst term) and `alpha_max = 2^8 − 1 = 255`. `MSLR` is
chosen so that `max(burst_term · alpha) / MSLR ≈ 127`, i.e. the biggest possible
single update *just* fills an `i8` without spurious saturation. Picking
`lr < MSLR` risks overflow; picking `lr > MSLR` gives slower (smaller) updates.

> **Watch-out.** The MNIST-oriented neuron config historically used `lr = 256`,
> well above `MSLR = 120`. That is *valid* but slow: with `burst_term = 6,
> alpha = 200`, `delta = 6·200/256 = 4` versus `10` at `MSLR`. Carried into
> [chapter 9](09-gaps-and-open-questions.md) as a calibration item.

`H_BETA = 4` and "a spike contributes 1 to beta" are admitted placeholders
(`docs/code/beta.md`) pending experiments.

---

Next: [chapter 5 — The event system](05-event-system.md), which orchestrates
these primitives.
