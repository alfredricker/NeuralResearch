# 9. Gaps and open questions

A consolidated, prioritized punch list of everything stubbed, undecided, or
known-broken. Each item links to where it is discussed in context. Items are
grouped by kind and ordered roughly by what blocks what.

## 9.1 Blocking gaps — nothing runs end-to-end until these exist

1. **The allocator does not exist.** No function turns a `NeuronConfig` +
   neuron count into the SoA arrays the event loop consumes. This is the
   keystone — every downstream item depends on it.
   → [chapter 7.2](07-network-construction.md)

2. **`Network::build` is an empty stub.** `src/network/mod.rs` has a `build` that
   iterates connections but never returns a `Network` — it does not compile as
   written. It must drive the allocator and the connection resolver.
   → [chapter 7.2](07-network-construction.md)

3. **`ConnRule::apply` is empty.** `src/network/topology/conn.rs` declares all
   five connection rules but implements none. Needed to produce the axon CSR.
   → [chapter 7.1](07-network-construction.md)

4. **`Axon` is not wired into the model.** It is a private `struct`, not a field
   of `Network`, and nothing populates `axon_targets` / `axon_offsets`. The event
   loop receives them only as bare parameters.
   → [chapter 3.1](03-data-model.md)

## 9.2 Event-system correctness — must fix before multi-tick / parallel runs

5. **The ring buffer never advances `head`.** `EventQueue::drain` reads `head`
   and `tail` but neither it nor anything else moves `head`, so slots are never
   recycled. A multi-tick trial loop ([chapter 8.4](08-mnist-pipeline.md)) will
   march `tail` forward indefinitely.
   → [chapter 5.2](05-event-system.md)

6. **`drain` slicing breaks when `tail` wraps.** `buf[head % len .. tail % len]`
   produces an invalid range once `tail % len < head % len`. Only safe today
   because capacity is assumed larger than one drain's worth of events.
   → [chapter 5.2](05-event-system.md)

7. **Parallel write conflict on `dendrite_activities[d]`.** Two `FORWARD_AP`
   events targeting the same dendrite at the same timestamp race on the
   read-modify-write. Fine serially; a data race once the loop is parallelized.
   Needs a resolution strategy (atomic accumulate / per-dendrite coalescing /
   owner-thread).
   → [chapters 2.3](02-architecture.md), [5.4](05-event-system.md)

8. **`Relaxed` atomics provide no ordering.** Acceptable single-threaded; provides
   no happens-before for a real multi-producer/consumer queue.
   → [chapter 5.2](05-event-system.md)

9. **Serial fan-out in the `FORWARD_AP` arm.** The inner loop over axon targets is
   flagged in-source as the prime batching/parallelism target.
   → [chapter 5.3](05-event-system.md)

## 9.3 Apical feedback — implemented at the leaf, disconnected everywhere else

10. **No event arm routes to `handle_apical_fb`.** The handler exists and is
    correct, but the dispatch loop has no apical event type, there is no
    axon-constant array in the data model, and no config currently sets
    `n_apical_dendrites` (it is `None`). Apical/BDP feedback cannot fire until
    these three are added.
    → [chapters 3.1](03-data-model.md), [5.3](05-event-system.md),
    [6.6](06-learning-dynamics.md)

## 9.4 Layout changes the slot model forces

11. **`live_count` does not exist.** Iterating only active synapses needs a
    per-dendrite live count and packed-live layout.
    → [chapter 7.4](07-network-construction.md)

12. **`update_dendrite_activity`'s loop bound must change with fixed slots.** It
    currently bounds by slice length — correct only because the event loop
    pre-trims the slice. Under fixed slots with a padded dead tail, it must bound
    by `base + live_count`. The bound's origin is the dendrite base, **not**
    `s_idx + live_count`.
    → [chapters 6.2](06-learning-dynamics.md),
    [7.4](07-network-construction.md)

13. **Structural plasticity is unimplemented.** Tombstone / migrate / compact is
    fully designed but no code exists. Relevant now only because it constrains
    the allocator's stride/headroom choices.
    → [chapter 7.5](07-network-construction.md)

## 9.5 Open design decisions

14. **Trial boundary: sentinel event vs. timestamp cutoff.** Unresolved.
    → [chapter 8.6](08-mnist-pipeline.md)

15. **Feedback path: direct injection vs. apical.** Option 1 unblocks
    classification; Option 2 is real BDP.
    → [chapter 8.5](08-mnist-pipeline.md)

16. **Connectivity: dense random vs. receptive fields vs. topographic.**
    Recommendation is dense random first.
    → [chapter 8.2](08-mnist-pipeline.md)

17. **Output-layer config.** Needs its own `NeuronConfig`: lower soma threshold,
    simpler dendrites, no apical, possibly higher/zero learning rate.
    → [chapter 8.1](08-mnist-pipeline.md)

18. **Packed dendrite address (`DendriteAddr`) — adopt or not.** Proposed
    `u32` bit-packing is documented but unused; the runtime uses a flat
    `dendrite_to_neuron` map. The "branch" tier is currently only an allocation
    count, not an addressable level.
    → [chapter 3.3](03-data-model.md)

## 9.6 Calibration watch-outs

19. **`learning_rate = 256 > MSLR = 120`** in the MNIST-oriented config — valid
    but slow updates (`delta ≈ 4` vs `≈ 10` at `MSLR`). Verify it doesn't
    underflow to 0 for typical `alpha`/`beta`.
    → [chapter 4.4](04-math-primitives.md)

20. **`dendrite_activity` has no decay** — only resets on spike, so it must be
    explicitly cleared between trials.
    → [chapters 6.3](06-learning-dynamics.md), [8.4](08-mnist-pipeline.md)

21. **`u16` timestamp wrap.** Decay uses `wrapping_sub` and is safe, but
    trial-level bookkeeping (≈327 trials × 200 ticks exhausts `u16`) must not
    assume monotonic timestamps.
    → [chapters 2.4](02-architecture.md), [6.1](06-learning-dynamics.md)

22. **Weight init `U(−8, 8)`** ⇒ ~50% inhibitory synapses from the start, which
    may suppress early dendrite firing. Consider `U(0, 8)` initially and let LTD
    drive weights negative.
    → [chapter 7.2](07-network-construction.md)

23. **No lateral inhibition / winner-take-all.** Nothing makes hidden neurons
    specialize; many may respond to the same input. May need explicit inhibitory
    connections or beta-based suppression.
    → [chapter 8](08-mnist-pipeline.md)

24. **`H_BETA = 4` and "spike adds 1 to beta" are placeholders** pending
    experiments.
    → [chapter 4.4](04-math-primitives.md)

## 9.7 Documentation drift

25. **`CLAUDE.md` references nonexistent `taxonomy/` and `init/` directories.**
    The neuron-config/taxonomy split it describes has been refactored away; configs
    now live under `src/neuron/` and the SoA arrays are owned by `Network`. The
    root doc should be reconciled with the current tree (and with this `docs/`
    book).
    → [README](README.md)
