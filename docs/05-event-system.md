# 5. The event system

This is the engine that realizes [chapter 2](02-architecture.md)'s event-driven
model. It lives in `src/network/event/`. The flow is: a fixed ring buffer holds
events; `run_event_loop` drains them and dispatches each to a handler; handlers
scope a local slice ([chapter 3](03-data-model.md)), call the `neuron/`
primitives, and push new events back via a producer. The handlers and primitives
— the actual biophysics — are [chapter 6](06-learning-dynamics.md); this chapter
is the plumbing.

## 5.1 The `Event` (`event.rs`)

There are **four** event types, each a bare `u8`, plus a `payload`:

```rust
pub const SOMATIC_SPIKE:   u8 = 0; // source = neuron_idx,   payload = burst (AP count)
pub const DENDRITIC_SPIKE: u8 = 1; // source = dendrite_idx, payload unused
pub const SOMA_SIGNAL:     u8 = 2; // source = neuron_idx,   payload = v_s (voltage delta to integrate)
pub const SYNAPSE_SIGNAL:  u8 = 3; // source = synapse_idx,  payload = burst (one AP delivery per target synapse)

pub struct Event {
    pub event_type: u8,   // a u8, NOT an enum — the buffer is shared with GPU kernels
    pub source:     u32,  // neuron / dendrite / synapse index — meaning depends on event_type
    pub timestamp:  u16,  // the lazy-decay clock (chapter 2.3)
    pub payload:    i16,  // event-specific scalar: burst count, or v_s for SOMA_SIGNAL
}
```

`event_type` is a `u8` on purpose: a Rust `enum` has no guaranteed layout across
the host/device boundary, but a `u8` discriminant does. The `source` field is
*overloaded* by type — neuron index, dendrite index, or synapse index — which is
why the dispatch loop must know the type before interpreting it.

The `payload` is the new structural piece versus older designs. It threads a
scalar through the cascade: a **burst count** (how many APs the soma fired) rides
`SOMATIC_SPIKE` and each fanned-out `SYNAPSE_SIGNAL`; a **voltage delta** (`v_s`,
from a dendritic spike or an apical plateau) rides `SOMA_SIGNAL`. This is what
lets a burst be one event carrying a multiplier instead of *N* identical events
(§5.3).

Constructors:

```rust
Event::with_payload(event_type, source, timestamp, payload) // general
Event::spike(event_type, source, timestamp)                 // payload = 0 (DENDRITIC_SPIKE, queue init)
Event::soma_signal(neuron_idx, timestamp, v_s)              // SOMA_SIGNAL carrying a voltage delta
```

The `timestamp` is the whole temporal story ([chapter 2.3](02-architecture.md)):
handlers reconstruct elapsed time from it and never consult a global clock. Note
that handlers currently *reuse* the triggering event's timestamp on the events
they emit, so a whole cascade runs at one frozen time — see
[chapter 12](12-time-and-clocking.md) for what does (and does not) advance it.

## 5.2 The queue and the producer

The design splits **reading** (safe, shared) from **writing** (unsafe, isolated)
— the unsafe-boundary decision from [chapter 2.5](02-architecture.md).

### `EventQueue` (`queue.rs`) — the buffer + read side

```rust
pub struct EventQueue {
    buf:  Box<[Event]>,   // fixed-capacity ring buffer, allocated once
    tail: AtomicU32,      // next write slot (claimed by producers)
    head: AtomicU32,      // next read slot
}

impl EventQueue {
    pub fn new(capacity: usize) -> Self { /* fills buf with Event::spike(0,0,0) */ }
    pub fn drain(&self) -> &[Event] {
        let head = self.head.load(Relaxed) as usize;
        let tail = self.tail.load(Relaxed) as usize;
        &self.buf[head % self.buf.len() .. tail % self.buf.len()]
    }
    pub fn producer_handle(&self) -> EventProducer<'_> { /* hands out a writer */ }
}
```

### `EventProducer` (`push.rs`) — the write side, all unsafe here

```rust
pub struct EventProducer<'a> { buf: *mut Event, tail: &'a AtomicU32, capacity: u32 }

impl<'a> EventProducer<'a> {
    pub fn push(&self, event: Event) {
        let idx = self.tail.fetch_add(1, Relaxed);          // atomically claim a slot
        unsafe { self.buf.add((idx % self.capacity) as usize).write(event); }
    }
}
```

This is the only `unsafe` in the simulator, and it is exactly the pattern
[chapter 2](02-architecture.md) anticipated for the GPU: *pass a device pointer
and an atomic counter; each thread claims a slot with `atomicAdd` and writes
directly; no ownership crosses thread boundaries.* The CPU `EventProducer` is a
1:1 prototype of that kernel-side claim-and-write.

> **Gaps in the ring buffer** (carried into [chapter 9](09-gaps-and-open-questions.md)):
> - `drain()` reads `head` and `tail` but **never advances `head`**, and nothing
>   else does either. So `head` stays 0 and `drain` always returns
>   `buf[0 .. tail % len]`. Slots are never recycled — a multi-tick trial loop
>   ([chapter 8.4](08-mnist-pipeline.md)) marches `tail` forward indefinitely.
> - The `head % len .. tail % len` slicing **misbehaves once `tail` wraps past
>   `capacity`** (the start index would exceed the end). Safe today only because
>   capacity is assumed larger than one drain.
> - `Relaxed` ordering is fine single-threaded but gives no happens-before
>   guarantee for a real multi-producer/consumer setup.
> - The buffer is FIFO: `drain` returns *insertion* order, not timestamp order.
>   Any per-hop conduction delay ([chapter 12](12-time-and-clocking.md)) needs a
>   time-ordered queue instead.

## 5.3 The dispatch loop (`loop.rs`)

`run_event_loop` is the heart. It takes *every* SoA array
([chapter 3](03-data-model.md)) as a parameter (no global state — GPU-friendly),
drains the queue once, and dispatches each event to one of four handlers. Each
handler does **only routing**; the physics lives in the `neuron/` primitives it
calls ("fat primitives, thin handlers" — [chapter 6](06-learning-dynamics.md)).

```rust
let producer = queue.producer_handle();
for e in queue.drain() {
    match e.event_type {
        SOMATIC_SPIKE => {                              // source = neuron n; payload = burst
            let n = e.source as usize;
            let (s0, s1) = neuron_synapse_range(n, dendrite_offsets, synapse_offsets);
            let axons = &axon_targets[axon_offsets[n] as usize .. axon_offsets[n+1] as usize];
            handle_somatic_spike(e.timestamp, e.payload as u16, soma_betas[n], soma_lrs[n],
                                 &mut synapse_weights[s0..s1], &mut synapse_alphas[s0..s1],
                                 &mut synapse_last_events[s0..s1], axons, &producer);
        }
        SOMA_SIGNAL => {                                // source = neuron n; payload = v_s
            let n = e.source as usize;
            handle_soma_signal(n, e.timestamp, e.payload, soma_potentials, soma_last_events,
                               soma_thresholds, soma_betas, &producer);
        }
        DENDRITIC_SPIKE => {                            // source = dendrite d
            let d = e.source as usize;
            let n = dendrite_to_neuron[d] as usize;     // reverse map (chapter 3.2)
            let (s0, s1) = dendrite_synapse_range(d, synapse_offsets);
            handle_dendritic_spike(n, e.timestamp, &dendrite_constants[d],
                                   &mut synapse_alphas[s0..s1], &mut synapse_last_events[s0..s1], &producer);
        }
        SYNAPSE_SIGNAL => {                             // source = synapse s; payload = burst
            let s = e.source as usize;
            let d = synapse_to_dendrite(s, synapse_offsets);     // O(log) binary search
            let target_n = dendrite_to_neuron[d] as usize;
            let (s0, s1)  = dendrite_synapse_range(d, synapse_offsets); // full stride block
            let local_s   = s - s0;
            let live_end  = dendrite_live_counts[d] as usize;    // slice-local; live synapses packed at front
            let is_apical = dendrite_is_apical[d] == 1;
            let burst     = e.payload.max(1) as u16;             // presynaptic burst scales the EPSP
            handle_synapse_signal(local_s, d, target_n, e.timestamp, burst, live_end,
                                  &synapse_xs[s0..s1], &mut synapse_alphas[s0..s1],
                                  &mut synapse_last_events[s0..s1], &synapse_weights[s0..s1],
                                  &mut dendrite_activities[d], &mut dendrite_last_events[d],
                                  dendrite_thresholds[d], is_apical, &producer);
        }
        _ => {}
    }
}
```

### Slice scoping is the safety model

Every handler receives **pre-narrowed slices**, computed by
`neuron_synapse_range` / `dendrite_synapse_range`
([chapter 3.2](03-data-model.md)). A handler literally cannot touch state outside
its component — the borrow checker enforces it, because it holds only a slice of
the relevant range. This is how the codebase gets memory-safety guarantees on top
of a flat, index-addressed layout.

Two consequences worth internalizing:

- Inside `handle_synapse_signal` the synapse index is **local** (`s − s0`),
  because the slice starts at the dendrite's first slot. The gamma loop in
  [chapter 6](06-learning-dynamics.md) therefore iterates relative to the slice.
- Under the **fixed-slot** layout ([chapter 7.3](07-network-construction.md)) the
  dendrite slice is the *full stride block* (`d*S .. (d+1)*S`), padded with a dead
  tail. The loop must not run to `slice.len()`; it runs to `live_end =
  live_synapse_counts[d]`, the count of bound synapses packed at the front. The
  loop passes `live_end` to the primitive so the dead slots never contribute.

### Signal flow

The four handlers reproduce [chapter 1](01-theory.md)'s cascade. The arrows are
emitted events:

```
SOMA_SIGNAL     ─► handle_soma_signal     ─► SOMATIC_SPIKE (payload = burst)         [if soma crosses threshold]
SOMATIC_SPIKE   ─► handle_somatic_spike   ─► BaP weight sweep + one SYNAPSE_SIGNAL per axon target (payload = burst)
SYNAPSE_SIGNAL  ─► handle_synapse_signal  ─► DENDRITIC_SPIKE (basal, fired) | SOMA_SIGNAL (apical, plateau)
DENDRITIC_SPIKE ─► handle_dendritic_spike ─► SOMA_SIGNAL (payload = branch constant)
```

Key design points, each a deliberate change from the older three-event model:

- **One axon, both compartments.** A somatic spike fans out to *all* of a
  neuron's axon targets as independent `SYNAPSE_SIGNAL` events — one queued AP
  delivery per target synapse. Whether each landing is basal or apical is decided
  per-target by `dendrite_is_apical[d]`, so a single axon drives both compartment
  types. There is **no separate forward-AP or apical-feedback event** (the old
  `FORWARD_AP` / `APICAL_FB` are gone).
- **Burst as a multiplier, not repetition.** Instead of replaying *N* identical
  spikes, the burst count rides the payload the whole way
  (`SOMATIC_SPIKE → SYNAPSE_SIGNAL`) and scales the EPSP at the receiving
  dendrite ([chapter 6.2](06-learning-dynamics.md)). One event, one multiply.
- **The fan-out is push-only.** `handle_somatic_spike` does *not* compute the
  per-target integration inline; it just enqueues the `SYNAPSE_SIGNAL`s. The
  per-synapse work happens later, one independent event each — cheap to enqueue,
  parallelizable to drain (the prime GPU-batching target).

Each handler may `producer.push(...)` new events; because `drain()` snapshots
`head..tail` at loop entry, events pushed *during* this drain are not processed
until the next call to `run_event_loop`. The loop is therefore one **wavefront**
of propagation per call (see the trial loop in
[chapter 8](08-mnist-pipeline.md)).

## 5.4 The parallel-write hazard

The architecture's central unsolved concurrency problem
([chapter 2.3](02-architecture.md)) surfaces here concretely: if two
`SYNAPSE_SIGNAL` events target synapses on the **same dendrite** at the same
timestamp, both handlers read-modify-write `dendrite_activities[d]` (and
`dendrite_last_events[d]`). Serially (today) this is fine. Parallelized naively it
is a data race. Resolution strategies — atomic accumulate, per-dendrite event
coalescing, owner-thread-per-dendrite — are an open decision in
[chapter 9](09-gaps-and-open-questions.md).

---

Next: [chapter 6 — Learning dynamics](06-learning-dynamics.md), the handler and
primitive bodies themselves.
