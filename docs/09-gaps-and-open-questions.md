# 9. Gaps and open questions

A consolidated, prioritized punch list of everything stubbed, undecided, or
known-broken. Each item links to where it is discussed in context. Items are
grouped by kind and ordered roughly by what blocks what.

> **Major progress since earlier revisions.** The four "blocking" gaps that
> headed this list — no allocator, empty `Network::build`, empty
> `ConnRule::apply`, unwired `Axon` — are **all closed**
> ([chapter 7](07-network-construction.md)). The network builds, wires, and
> resolves its axon CSR, with tests. Apical feedback, once "disconnected
> everywhere," now has a complete leaf-to-event mechanism
> ([chapter 6.3](06-learning-dynamics.md)). What remains is the trial-loop glue,
> the event-buffer lifecycle, and calibration.

## 9.1 Blocking the first end-to-end run

1. **The ring buffer never advances `head`.** `EventQueue::drain` reads `head`
   and `tail` but neither it nor anything else moves `head`, so slots are never
   recycled. A multi-tick trial loop ([chapter 8.4](08-mnist-pipeline.md)) marches
   `tail` forward indefinitely and eventually overruns the buffer.
   → [chapter 5.2](05-event-system.md)

2. **`run_event_loop` does not accumulate `spike_counts`.** `output.rs`'s readout
   is written against a per-neuron AP accumulator the loop is expected to fill on
   each `SOMATIC_SPIKE`, but the loop has no such counter. Without it the effector
   has nothing to read.
   → [chapters 8.4](08-mnist-pipeline.md), [11.4](11-io-boundary.md)

3. **No clock advances time.** The only writer of a fresh timestamp is
   `InputSpace::encode`, fed a caller-supplied `base_ts`; handlers reuse the
   triggering timestamp, so a whole cascade runs at one frozen time and nothing
   increments a clock between wavefronts. A trial loop needs a clocking scheme.
   → [chapter 12](12-time-and-clocking.md)

4. **No hidden-layer config or input→hidden→output wiring.** `input_config()` and
   `output_config()` exist; the hidden `NeuronConfig` and the `connect` calls that
   assemble the MNIST topology do not.
   → [chapters 8.1](08-mnist-pipeline.md), [7.1](07-network-construction.md)

## 9.2 Event-system correctness — before multi-tick / parallel runs

5. **`drain` slicing breaks when `tail` wraps.** `buf[head % len .. tail % len]`
   produces an invalid range once `tail % len < head % len`. Safe today only
   because capacity is assumed larger than one drain's worth of events.
   → [chapter 5.2](05-event-system.md)

6. **The queue is FIFO, not time-ordered.** `drain` returns insertion order. Any
   per-hop conduction delay ([chapter 12.3](12-time-and-clocking.md), Options B/D)
   requires replacing the ring with a timestamp-ordered priority queue.
   → [chapters 5.2](05-event-system.md), [12](12-time-and-clocking.md)

7. **Parallel write conflict on `dendrite_activities[d]`.** Two `SYNAPSE_SIGNAL`
   events targeting the same dendrite at the same timestamp race on the
   read-modify-write of `dendrite_activities[d]` and `dendrite_last_events[d]`.
   Fine serially; a data race once the loop is parallelized. Needs a resolution
   strategy (atomic accumulate / per-dendrite coalescing / owner-thread).
   → [chapters 2.3](02-architecture.md), [5.4](05-event-system.md)

8. **`Relaxed` atomics provide no ordering.** Acceptable single-threaded; no
   happens-before for a real multi-producer/consumer queue.
   → [chapter 5.2](05-event-system.md)

9. **Serial fan-out in the `SOMATIC_SPIKE` path.** `handle_somatic_spike` enqueues
   one `SYNAPSE_SIGNAL` per axon target in a loop — push-only and cheap, but the
   flagged prime candidate for GPU batching / per-target parallel dispatch.
   → [chapters 5.3](05-event-system.md), [10.2](10-gpu-execution-and-residency.md)

## 9.3 Layout / allocator follow-ups

10. **`synapse_to_dendrite` is still a binary search.** With fixed slots the
    offset is analytic (`d·S`), so the reverse lookup *could* collapse to `s / S`,
    but the code still stores `synapse_offsets` and uses `partition_point`. A
    pending simplification, not a correctness bug.
    → [chapters 7.3](07-network-construction.md), [3.2](03-data-model.md)

11. **`S = SYNAPSE_SLOTS_PER_DENDRITE = 255` over-provisions ~16×.** For a
    16-live-synapse dendrite the dead tail dominates the allocated synapse arrays.
    Fine for small MNIST nets; tune before scaling memory.
    → [chapter 7.3](07-network-construction.md)

12. **Dropped edges are silent.** When a connection finds no free matching-
    compartment slot on the target neuron, `build_network` drops the edge with no
    diagnostic. Acceptable as capacity-limiting, but a silent loss of intended
    connectivity.
    → [chapter 7.4](07-network-construction.md)

13. **`Topographic` is unimplemented.** `ConnRule::apply` returns
    `ConnError::InvalidRule` for it.
    → [chapter 7.1](07-network-construction.md)

14. **Structural plasticity is unimplemented.** Tombstone / migrate / compact is
    fully designed but no code exists. Relevant now mainly as a constraint on
    stride/headroom choices.
    → [chapter 7.5](07-network-construction.md)

## 9.4 Apical / feedback — mechanism done, topology missing

15. **Apical feedback has no topology to exercise it.** The mechanism is complete
    end-to-end — an apical `SYNAPSE_SIGNAL` integrates via `update_dendrite_activity`
    and emits a graded `SOMA_SIGNAL` through `apical_plateau`. What's missing: a
    config that sets `n_apical_dendrites` and the feedback `connect` calls onto
    `Compartment::Apical` slots.
    → [chapters 6.3](06-learning-dynamics.md), [8.5](08-mnist-pipeline.md)

## 9.5 Open design decisions

16. **Trial boundary: timestamp cutoff vs. sentinel event.** With a caller-driven
    clock the cutoff form is natural, but the loop is unwritten.
    → [chapters 8.6](08-mnist-pipeline.md), [12](12-time-and-clocking.md)

17. **Feedback path: direct injection vs. apical.** Option 1 unblocks
    classification; Option 2 is real BDP and now much closer to reachable.
    → [chapter 8.5](08-mnist-pipeline.md)

18. **Connectivity: dense/fixed-in-degree vs. receptive field vs. topographic.**
    Recommendation is dense/fixed-in-degree first.
    → [chapter 8.2](08-mnist-pipeline.md)

19. **Clocking scheme.** Caller-driven wavefront tick, global `u64` + local `u16`,
    or full discrete-event priority queue — pick per feature needs.
    → [chapter 12.3](12-time-and-clocking.md)

20. **Packed dendrite address (`DendriteAddr`) — adopt or not.** Proposed `u32`
    bit-packing is documented but unused; the runtime uses the flat
    `dendrite_to_neuron` map. The "branch" tier is only an allocation count, not an
    addressable level.
    → [chapter 3.3](03-data-model.md)

## 9.6 Calibration watch-outs

21. **`dendrite_activity` and `soma_potential` now decay** (with `BASAL_DECAY` /
    `APICAL_DECAY` / `SOMATIC_DECAY`). They no longer *require* a per-trial reset to
    avoid stale accumulation, but a reset is still the clean trial-isolation choice
    — and the half-lives interact with trial-window length.
    → [chapters 6.2](06-learning-dynamics.md), [8.4](08-mnist-pipeline.md)

22. **`u16` timestamp wrap.** Decay uses `wrapping_sub` and is safe, but trial
    bookkeeping (~327 trials × 200 ticks exhausts `u16`) must not assume monotonic
    timestamps — push monotonicity into a wider clock.
    → [chapters 2.4](02-architecture.md), [12.4](12-time-and-clocking.md)

23. **Weight init is now `U(0, 8)` — all excitatory.** Resolves the older
    "~50% inhibitory from the start" concern; LTD must now *drive* weights
    negative rather than starting there. Watch that early dendrites can actually
    reach threshold.
    → [chapter 7.2](07-network-construction.md)

24. **No lateral inhibition / winner-take-all.** Nothing makes hidden neurons
    specialize; many may respond to the same input. May need explicit inhibitory
    connections or beta-based suppression.
    → [chapter 8](08-mnist-pipeline.md)

25. **Untuned placeholders.** `H_BETA = 4`, the per-spike `beta` increment,
    `APICAL_DV_S = 64`, and `APICAL_SLOPE_K = 9` are admitted placeholders pending
    experiments.
    → [chapters 4.4](04-math-primitives.md), [6.3](06-learning-dynamics.md)

26. **Stale unit tests after a constants change.** Several `synapse` and
    `dendrite` tests hard-code the *old* decay half-lives (`ALPHA_DECAY = 8`,
    `BASAL_DECAY` giving a 1024-tick half-life) and now fail against the current
    constants (`ALPHA_DECAY = 11` → 2048, `BASAL_DECAY = 9` → 512). The *code* is
    the source of truth; the test expectations need updating.
    → [chapters 4.4](04-math-primitives.md), [6.1](06-learning-dynamics.md)

## 9.7 Documentation drift

27. **`CLAUDE.md` / root docs may still describe the old event model.** This
    `docs/` book reflects the current four-event system (`SOMATIC_SPIKE`,
    `DENDRITIC_SPIKE`, `SOMA_SIGNAL`, `SYNAPSE_SIGNAL`), the built allocator, the
    `io/` boundary, and the clocking question. Reconcile any root-level docs that
    still reference `FORWARD_AP` / `APICAL_FB`, `taxonomy/`, or `init/`.
    → [README](README.md)
