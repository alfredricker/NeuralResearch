# 10. Appendix — GPU execution and memory residency

Forward-looking notes for the eventual CUDA backend (`src/gpu/`, currently a
stub). Everything here builds on the layout and event model from
[chapters 2](02-architecture.md) and [5](05-event-system.md) and is *design
intent*, not implemented code.

## 10.1 Why the whole design is already GPU-shaped

The earlier chapters made their choices for the GPU; this collects the payoff:

- **SoA + analytic offsets** ([chapters 2.1](02-architecture.md),
  [7.3](07-network-construction.md)) → coalesced, predictable, pointer-free
  addressing.
- **`u8`/`u16` state + shift-based decay** ([chapters 2.4](02-architecture.md),
  [4.1](04-math-primitives.md)) → high state density, no FPU, deterministic
  host/device parity.
- **Atomic claim-and-write event buffer** ([chapter 5.2](05-event-system.md)) →
  the `EventProducer` is a direct prototype of the kernel pattern: *pass a device
  pointer and an atomic counter; each thread claims a slot via `atomicAdd` and
  writes directly; no ownership crosses thread boundaries.*

## 10.2 Segmented reduction

The recurring parallel primitive. The gamma sum
([chapter 6.2](06-learning-dynamics.md)) and the per-neuron weight update
([chapter 6.5](06-learning-dynamics.md)) are both reductions over a *segment* —
one dendrite's synapses, or one neuron's synapses — laid out contiguously by the
offset arrays. On the GPU this maps to standard segmented-reduction kernels
(one warp/block per segment), and the fixed-stride slot layout
([chapter 7.3](07-network-construction.md)) makes segment boundaries analytic, so
no segment-id scan is needed. The gamma case is specifically an *ordered suffix*
reduction, which is why the packed-live `live_count` layout
([chapter 7.4](07-network-construction.md)) matters: it keeps the segment a dense,
divergence-free prefix.

## 10.3 Memory residency — partitioning the network into VRAM

A biologically realistic network is far larger than VRAM, so only a working set
can be resident at once. The CPU/GPU distinction drives the strategy:

- **CPU caches** auto-evict cold 64-byte lines and load hot ones — residency is
  handled by hardware.
- **GPU caches** have no analogous automatic mechanism, and moving data between
  RAM and VRAM has a fixed per-transfer overhead.

So residency must be *explicit and coarse*. Rather than a few multi-megabyte
partitions, a highly dynamic spiking network may favor **kilobyte-range
partitions hot-loaded by recent activity** — and one neuron with all its
dendrites and synapses is conveniently a kilobyte-scale unit
([chapter 3.4](03-data-model.md)).

### The Hawkes assumption

The justification for activity-driven residency: a component that just received
an event is more likely to receive another soon than one that has been quiet.
Spiking activity is self-exciting (Hawkes-process-like), so "recently active"
is a good predictor of "needed next" — exactly the signal a residency manager
should track. This dovetails with the event-driven model
([chapter 2.3](02-architecture.md)): the event stream *is* the activity signal.

### How to choose partition boundaries

Two families:

- **Graph partitioning (e.g. METIS).** Partition to minimize edges crossing
  boundaries, i.e. minimize cross-partition event traffic. Well studied and
  near-optimal for a fixed graph, but expensive to compute and must be redone
  when connectivity changes — costly under structural plasticity
  ([chapter 7.5](07-network-construction.md)).
- **Biological / structural partitioning.** Cut along the network's own
  structure — cortical columns, layers, regions. Cheap, stable under rewiring,
  and aligned with how motifs are duplicated
  ([chapter 7.1](07-network-construction.md)), at the cost of more boundary
  traffic than an optimal graph cut.

A `NetworkLayout` of per-population base offsets (mentioned in `notes/5-30-26.md`)
is the natural seed for a residency/tile table: it already records where each
population's arrays start.

## 10.4 Clocks

Although the simulation is event-driven with no global clock
([chapter 2.3](02-architecture.md)), a coarse global time is still useful for
bookkeeping (trial counting, residency aging). The GPU-specific concern is
contention: a single global atomic clock read per event by thousands of threads is
a hot spot. The intended split avoids it — a `u64` global clock for absolute time,
advanced *coarsely* (per kernel launch / per wavefront), while individual
components track *time since last event* in `u16` ([chapter 6.1](06-learning-dynamics.md));
the `u16` is all the decay math reads on the hot path, and `wrapping_sub` keeps it
correct across wraps ([chapter 2.4](02-architecture.md)).

The full treatment of clocking — what writes a timestamp today, what (doesn't yet)
advance it, and the design options for a clock — is its own chapter:
[chapter 12 — Time and the network clock](12-time-and-clocking.md). The `u64`-frames-`u16`-deltas
scheme above is Option C there.

---

Next: [chapter 11 — The IO boundary](11-io-boundary.md), the layer that connects
all of this to the outside world, and [chapter 12 — Time and the network
clock](12-time-and-clocking.md).
