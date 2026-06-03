# 11. The IO boundary

Everything up to here is *internal*: indices, slices, events, weights. This
chapter is the layer that connects the network to the outside world — pixels in,
class predictions out. It lives in `src/io/` and is the piece
[chapter 8](08-mnist-pipeline.md) assembles into a working MNIST loop.

The boundary is deliberately **symmetric**. There is an afferent half (`input`)
and an efferent half (`output`), and they are mirror images:

| | afferent (`input.rs`) | efferent (`output.rs`) |
| --- | --- | --- |
| object | `InputSpace` | `Effector` |
| the arrow (CSR) | `SensoryMap`: coordinate → input neuron(s) | `ReadoutMap`: class → output neuron(s) |
| neuron config | `input_config()` | `output_config()` |
| direction | *asserts* spikes onto neurons | *reads* spikes off neurons |
| MNIST instance | `Grid2D{28,28}`, identity map | 10 classes, identity map |

Both halves own a `neuron_range` that is empty (`0..0`) until `bind` attaches the
global neuron indices the [allocator](07-network-construction.md) handed out. The
arrow is always *explicit code* — a CSR you can read — never implied by
allocation order. That is the whole design principle of this module: the mapping
from the external world onto neuron indices is a first-class, inspectable object.

## 11.1 `Shape` — the external coordinate system

```rust
pub enum Shape {
    Flat(u32),               // a 1-D space of n elements
    Grid2D { h: u32, w: u32 }, // a 2-D grid (row-major). MNIST is Grid2D { h: 28, w: 28 }
}
```

`Shape::n_elements()` is `n` for `Flat`, `h*w` for `Grid2D`. It is the size the
frame slice must match and the number of rows in the sensory CSR.

## 11.2 `SensoryMap` — the afferent arrow

The map from input-space coordinates to input neurons, in the same CSR idiom as
`dendrite_offsets` / `axon_offsets` ([chapter 2.2](02-architecture.md)):

```rust
pub struct SensoryMap {
    offsets: Vec<u32>, // len = n_elements + 1
    neurons: Vec<u32>, // flattened LOCAL input-neuron indices
    n_neurons: u32,
}
```

`neurons[offsets[e] .. offsets[e+1]]` are the (local) input neurons that
element `e` drives. `SensoryMap::identity(n)` is the one MNIST uses — element `i`
drives neuron `i`, one-to-one. The structure already supports the richer maps the
comments anticipate (ON/OFF channels, receptive-field pooling): those are just
non-identity CSRs.

Indices are **local** (`0..n_neurons`). `InputSpace::bind` is what offsets them
into the global index space.

## 11.3 `InputSpace` — one external modality

```rust
pub struct InputSpace {
    pub name:  &'static str,
    pub shape: Shape,
    sensory:   SensoryMap,
    neuron_range: Range<u32>, // empty until bind()
}
```

Construction and binding:

```rust
let space = InputSpace::identity("mnist", Shape::Grid2D { h: 28, w: 28 });
let id    = builder.add(input_config(), space.n_neurons());   // allocate 784 input neurons
let net    = Network::build(builder);
let space  = space.bind(net.population_range(id));             // resolve local → global
```

`n_neurons()` (= the sensory map's neuron count) is exactly the `size` to pass to
`NetworkBuilder::add` ([chapter 7.1](07-network-construction.md)). `bind` asserts
the range width matches, so a mis-sized allocation fails loudly rather than
silently mis-mapping pixels.

### Input neurons are pure axon sources

`input_config()` is a `NeuronConfig` with **zero dendrites**
(`n_basal_dendrites = 0`, `n_apical_dendrites = None`). The soma fields exist only
so the neuron occupies an SoA index; the dynamics never read them. The
consequence matters for safety: with no dendrites,
`dendrite_offsets[n] == dendrite_offsets[n+1]`, so the neuron's afferent synapse
range is empty. Asserting a somatic spike on an input neuron therefore runs
`handle_somatic_spike`'s BaP sweep over an empty slice (a no-op) and simply fans
the AP out across the axon CSR. **Triggering a spike on an input neuron can never
panic or error**, even though it owns no integration state.

### `encode` — transduce a frame into events

```rust
pub fn encode(&self, frame: &[u8], base_ts: u16, window: u16,
              producer: &EventProducer, rng: &mut impl RngExt)
```

For each non-zero element of `frame`, `encode` follows the sensory arrow to the
input neuron(s) it drives and pushes a `SOMATIC_SPIKE`
([chapter 5](05-event-system.md)) at that neuron's *global* index. Two details
carry meaning:

- **Intensity → burst.** `intensity_to_burst` maps element value `1..=255` to a
  burst count `1..=MAX_INPUT_BURST` (currently 4). The burst rides the event
  payload and scales the downstream EPSP ([chapter 6.2](06-learning-dynamics.md)),
  so a bright pixel drives harder than a dim one — rate/intensity coding folded
  into a single event instead of a Bernoulli spike train.
- **Timestamp jitter.** Each spike's timestamp is drawn uniformly from
  `[base_ts, base_ts + window)`, so a frame presents as a small stochastic volley
  rather than one perfectly synchronous edge. `window <= 1` degenerates to
  `base_ts`. This is where a frame's spikes get spread across the **trial
  window** ([chapter 12](12-time-and-clocking.md) on where `base_ts` comes from).

Dark elements (intensity 0) push nothing — no drive, no event — so cost scales
with the number of lit pixels, exactly the event-driven principle from
[chapter 2.3](02-architecture.md).

`encode` is **purely host-side**: it only pushes onto the queue. The network's
own axon CSR + handlers carry the signal from there. It touches no network state.

## 11.4 `ReadoutMap` and `Effector` — the efferent mirror

```rust
pub struct ReadoutMap {              // class -> output neuron(s)
    offsets: Vec<u32>,               // len = n_classes + 1
    neurons: Vec<u32>,               // local output-neuron indices, grouped by class
    n_neurons: u32,
}

pub struct Effector {
    pub name: &'static str,
    readout:  ReadoutMap,
    neuron_range: Range<u32>,        // empty until bind()
}
```

`ReadoutMap::identity(10)` gives one output neuron per digit. The CSR shape
already supports **population coding** — several neurons voting for one class —
which is why readout sums over `members(c)` rather than reading a single neuron.

Unlike input neurons, output neurons *integrate*. `output_config()` gives them
basal dendrites that receive the hidden layer's projection, a deliberately **low
`soma_threshold`** (10) so they fire readily, and **no apical compartment** (the
first-pass training feedback is direct, not apical BDP —
[chapter 8.5](08-mnist-pipeline.md)). Those values are explicitly untuned
starting points.

### Reading a prediction

The effector holds **no mutable state**. Spike observation is expected to live in
a per-neuron `spike_counts: Vec<u32>` accumulator (the harness zeroes it per
trial); the effector only reads the window of that buffer belonging to its output
neurons:

```rust
pub fn class_activity(&self, spike_counts: &[u32]) -> Vec<u32>  // per class, sum over member neurons
pub fn predict(&self, spike_counts: &[u32]) -> Option<u32>      // argmax; None if the layer was silent
```

`predict` returns `None` (not a spurious class 0) when every output neuron was
silent, so "no prediction" is explicit. Ties break to the lowest class index.

> **Gap.** `spike_counts` is the contract `output.rs` is written against, but
> `run_event_loop` ([chapter 5.3](05-event-system.md)) does **not** currently
> accumulate it — there is no `SOMATIC_SPIKE` counter in the loop. Wiring that
> accumulator (and zeroing it per trial) is a prerequisite for readout. Tracked
> in [chapter 9](09-gaps-and-open-questions.md).

## 11.5 Why this layer is shaped this way

- **The arrow is data, not convention.** Because `SensoryMap`/`ReadoutMap` are
  explicit CSRs, the pixel→neuron and class→neuron mappings are testable and
  swappable without touching the dynamics. Identity today; receptive fields or
  population codes later are the *same code* with a different CSR.
- **`bind` separates allocation from mapping.** The IO objects are built before
  the network exists (you need `n_neurons()` to size the population), then bound
  to concrete global ranges after `Network::build`. `population_range`
  ([chapter 7](07-network-construction.md)) is the handshake.
- **Encoding is just event production.** `encode` uses the very same
  `EventProducer` ([chapter 5.2](05-event-system.md)) the internal handlers use.
  External input is not a special path — it is one more producer of
  `SOMATIC_SPIKE`s, which is exactly why a pixel and an upstream neuron are
  indistinguishable to the layer that receives them.

---

Next: [chapter 12 — Time and the network clock](12-time-and-clocking.md), which
addresses the one thing this chapter glossed over: where `base_ts` comes from,
and what advances it.
