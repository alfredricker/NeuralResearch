# 12. Time and the network clock

[Chapter 2.3](02-architecture.md) made the foundational claim: *there is no
global tick.* That is true and deliberate — but it leaves a real question the
codebase has **not yet answered**: if no loop steps a clock, what sets the
`timestamp` on an event, and what makes time *advance*? This chapter lays out the
current (incomplete) state and the design options for closing the gap. It is the
one place the "no clock" architecture meets the practical need for *some* notion
of time.

## 12.1 What time is actually used for

Before choosing a clocking scheme, be clear about who consumes time. There are
four distinct consumers, and they want different things:

1. **Lazy decay** (the hot path). Every `update_*` primitive computes
   `elapsed = now.wrapping_sub(last_event)` and decays from it
   ([chapters 4.1](04-math-primitives.md), [6](06-learning-dynamics.md)). This is
   the *only* consumer that exists in code today. It needs a `u16` that increases
   between successive touches of a component; it does **not** need a global
   ordering.
2. **Trial bookkeeping.** Counting trials, marking "end of input," resetting
   per-trial state ([chapter 8.4](08-mnist-pipeline.md)). Needs a coarse,
   monotonic frame.
3. **Conduction delay / temporal coding.** If a synapse should fire `δ` ticks
   *after* the presynaptic AP, downstream events must carry a *larger* timestamp,
   and the queue must process them in time order. The model has no such delays
   yet (see §12.3).
4. **Residency aging** (future, GPU). "How long since this tile saw an event"
   drives hot-loading ([chapter 10.3](10-gpu-execution-and-residency.md)). Needs
   absolute wall-clock-ish time, coarse is fine.

A clocking scheme is really a decision about which of these you serve and how.

## 12.2 What exists today

Two facts, both load-bearing and both currently underdeveloped:

**Timestamps enter only at the boundary.** The sole writer of a "fresh"
timestamp is `InputSpace::encode` ([chapter 11.3](11-io-boundary.md)), which
stamps each input spike with `base_ts + jitter(window)`. `base_ts` and `window`
are *parameters supplied by the caller*. Nothing inside the network generates
time; the harness that calls `encode` does.

**Propagation is zero-delay in simulated time.** Look at the handlers
([chapter 6](06-learning-dynamics.md)): every emitted event reuses the timestamp
of the event that triggered it.

```rust
// handle_soma_signal, handle_somatic_spike, handle_synapse_signal, ...
producer.push(Event::with_payload(SOMATIC_SPIKE, n, timestamp, burst));
//                                                ^^^^^^^^^ inherited, never incremented
```

So an entire forward cascade — `SOMATIC_SPIKE → SYNAPSE_SIGNAL → DENDRITIC_SPIKE
→ SOMA_SIGNAL → SOMATIC_SPIKE → …` — happens at **one frozen timestamp**. Within a
cascade, no time elapses, so no decay occurs between hops. Time only moves when
the *next* `encode` call (or whatever drives the loop) supplies a larger
`base_ts`.

This is internally consistent but limited: it means the network currently has no
conduction latency, no spike-timing structure beyond the input jitter, and — most
importantly — **no component advances the clock on its own.** That is the gap the
user-facing "set a clock" question is about.

A third fact constrains every option below: the queue is a **FIFO ring**
([chapter 5.2](05-event-system.md)), not a time-ordered priority queue. `drain`
returns events in *insertion* order, not timestamp order. Any scheme that puts
delays on events (so a later-stamped event can be enqueued before an
earlier-stamped one) needs the queue to become time-ordered first.

## 12.3 Options for a clock

Five schemes, roughly in increasing power and cost. They are not mutually
exclusive — the recommended path composes the first and the third.

### Option A — Caller-driven wavefront clock *(recommended first step)*

The harness owns a single counter and bumps it once per `run_event_loop` call.
Because each call drains exactly one wavefront ([chapter 5.3](05-event-system.md)),
"one wavefront = one tick" is a natural quantum:

```rust
let mut clock: u16 = 0;
for _ in 0..T_trial {
    space.encode(frame, clock, window, &producer, rng); // stamp this tick's input
    run_event_loop(&queue, /* ...all SoA arrays... */); // drain one wavefront
    clock = clock.wrapping_add(1);
}
```

- **Serves:** decay across ticks (consumer 1) and trial framing (consumer 2).
- **Cost:** essentially zero — no new data structures. It is the smallest change
  that makes decay meaningful between wavefronts and gives `encode` a real
  `base_ts` to advance.
- **Limitation:** still zero-delay *within* a wavefront. Granularity is the
  loop-call, not the spike.

This is the scheme [chapter 8.4](08-mnist-pipeline.md)'s trial loop assumes, and
it is where to start.

### Option B — Per-hop conduction delay

Give each emitted event a timestamp *offset* from its trigger — a synaptic or
axonal latency `δ`:

```rust
producer.push(Event::with_payload(SYNAPSE_SIGNAL, s, timestamp.wrapping_add(AXON_DELAY), burst));
```

- **Serves:** real temporal dynamics (consumer 3) — spike-timing, refractory
  windows, the asymmetric gamma integration getting genuine *temporal* (not just
  spatial) ordering.
- **Cost:** requires the queue to process events **in timestamp order**, which
  the current FIFO ring does not. You need a priority queue (binary heap keyed on
  timestamp) or time-bucketed sub-queues. This is a real structural change.
- **Note:** delays could be uniform constants, per-edge (an `axon_delays` array
  parallel to `axon_targets`), or sampled — each is a modelling choice.

### Option C — Global `u64` clock + `u16` local deltas

The scheme [chapter 10.4](10-gpu-execution-and-residency.md) sketched: a `u64`
absolute clock for bookkeeping and aging, while components keep tracking *time
since last event* in `u16` for decay.

- **Serves:** trial bookkeeping and residency aging (consumers 2, 4) without
  letting the `u16` wrap corrupt long-horizon accounting.
- **Why both widths:** the decay math only ever needs the `u16` delta, and
  `wrapping_sub` keeps it correct across the 65 536-tick wrap
  ([chapter 2.4](02-architecture.md)). The `u64` never enters the hot path; it
  only frames it. Concretely: ~327 trials × 200 ticks already exhausts a `u16`,
  so anything counting *trials* must not assume the event `timestamp` is
  monotonic — it wraps. The `u64` is where monotonicity lives.
- **Cost:** low; it is bookkeeping beside the hot path, not inside it. Compose it
  with Option A (the `u16` clock is `(global_u64 & 0xFFFF)`).

### Option D — Discrete-event time (the clock *is* the next event)

Full discrete-event simulation: there is no fixed tick at all. The clock is
defined as the timestamp of the event currently being processed; the simulator
always advances to the *earliest* pending event.

- **Serves:** consumers 1–3 exactly and efficiently — this is the textbook
  event-driven scheme.
- **Cost:** mandates the time-ordered priority queue from Option B and makes
  "drain one wavefront" ill-defined (you drain one *event*, or all events at the
  current timestamp). It is the most faithful to [chapter 2.3](02-architecture.md)
  and the most disruptive to the current ring-buffer loop.

### Option E — Hybrid *(the likely endpoint)*

Option A's caller clock to frame trials and supply `base_ts`, plus Option C's
`u64` for bookkeeping, evolving to Option B/D's priority queue + per-hop delays
once temporal coding matters. Start simple; add ordering only when a feature
needs it.

## 12.4 Cross-cutting constraints

Whatever scheme is chosen must respect these, all already noted elsewhere:

- **`u16` wrap is real, not theoretical.** Decay is wrap-safe via `wrapping_sub`;
  *trial counting is not* and must live in a wider type
  ([chapters 2.4](02-architecture.md), [9](09-gaps-and-open-questions.md)).
- **The ring buffer is FIFO and never advances `head`**
  ([chapter 5.2](05-event-system.md)). Options B and D require replacing it with
  a time-ordered structure that recycles slots; even Option A needs the
  `head`-advance fix before a multi-tick loop can run without overrunning the
  buffer.
- **`dendrite_activity` and `soma_potential` now decay** (with `BASAL_DECAY` /
  `APICAL_DECAY` / `SOMATIC_DECAY` — [chapter 6](06-learning-dynamics.md)), so
  unlike the older design they do *not* strictly require a per-trial reset to
  avoid stale accumulation — a long enough inter-trial gap leaks them toward 0 on
  its own. A reset is still the clean, deterministic choice for trial isolation,
  but the clock now participates in inter-trial forgetting.
- **GPU.** A single global atomic clock is a contention point across thousands of
  threads. The `u64`-frames-`u16`-deltas split
  ([chapter 10.4](10-gpu-execution-and-residency.md)) is partly motivated by this:
  the hot path reads only the local delta, and the global clock is updated
  coarsely (per kernel launch / per wavefront), not per event.

## 12.5 Recommendation

For the immediate MNIST target, adopt **Option A**: a `u16` clock in the trial
harness, incremented once per `run_event_loop` call, fed to `encode` as
`base_ts`. It costs nothing, makes decay meaningful across wavefronts, and gives
[chapter 8](08-mnist-pipeline.md)'s trial loop a real time axis — after the
ring-buffer `head` fix it depends on. Layer in the `u64` bookkeeping clock
(Option C) when trial counts exceed the `u16` horizon. Defer the priority queue
and per-hop delays (Options B/D) until a feature — conduction latency, true
spike-timing codes — actually requires sub-wavefront time resolution.

---

This is the end of the documentation proper. Back to the [README](README.md), the
[chapter 9 punch list](09-gaps-and-open-questions.md), or the
[GPU appendix](10-gpu-execution-and-residency.md).
