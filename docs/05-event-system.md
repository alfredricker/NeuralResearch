# 5. The event system

This is the engine that realizes [chapter 2](02-architecture.md)'s event-driven
model. It lives in `src/network/event/`. The flow is: a fixed ring buffer holds
events; `run_event_loop` drains them and dispatches each to a handler; handlers
mutate a local slice ([chapter 3](03-data-model.md)) and push new events back via
a producer. The handlers themselves — the actual biophysics — are
[chapter 6](06-learning-dynamics.md); this chapter is the plumbing.

## 5.1 The `Event` (`event.rs`)

```rust
pub const SOMATIC_SPIKE:   u8 = 0;   // source = neuron_idx
pub const DENDRITIC_SPIKE: u8 = 1;   // source = dendrite_idx
pub const FORWARD_AP:      u8 = 2;   // source = neuron_idx

pub struct Event {
    pub event_type: u8,   // a u8, NOT an enum — the buffer is shared with GPU kernels
    pub source:     u32,  // meaning depends on event_type (see above)
    pub timestamp:  u16,  // the lazy-decay clock (chapter 2.3)
}
```

`event_type` is a bare `u8` on purpose: a Rust `enum` has no guaranteed layout
across the host/device boundary, but a `u8` discriminant does. The `source`
field is *overloaded* by type — neuron index for somatic/forward, dendrite index
for dendritic — which is why the dispatch loop must know the type before
interpreting it.

The `timestamp` is the whole temporal story ([chapter 2.3](02-architecture.md)):
handlers reconstruct elapsed time from it and never consult a global clock.

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

> **Gaps in the ring buffer.**
> - `drain()` reads `head` and `tail` but **never advances `head`**, and nothing
>   else does either. So `head` stays 0 and `drain` always returns
>   `buf[0 .. tail % len]`. The buffer does not actually recycle slots yet.
> - The `head % len .. tail % len` slicing **panics or misbehaves once `tail`
>   wraps past `capacity`** (the start index would exceed the end). Capacity is
>   currently assumed large enough that one drain never wraps.
> - `Relaxed` ordering is fine single-threaded but provides no
>   happens-before guarantees for a real multi-producer/consumer setup.
> These are acceptable for the current single-pass CPU prototype; they must be
> resolved before parallelism. Carried into
> [chapter 9](09-gaps-and-open-questions.md).

## 5.3 The dispatch loop (`loop.rs`)

`run_event_loop` is the heart. It takes *every* SoA array
([chapter 3](03-data-model.md)) as a parameter (no global state — GPU-friendly),
drains the queue once, and dispatches each event:

```rust
let producer = queue.producer_handle();
for e in queue.drain() {
    match e.event_type {
        SOMATIC_SPIKE => {
            let n = e.source as usize;
            let (s0, s1) = neuron_synapse_range(n, dendrite_offsets, synapse_offsets);
            handle_somatic_spike(n, e.timestamp, &mut soma_betas[n], &mut soma_last_events[n],
                                 &soma_lrs[n], &mut synapse_weights[s0..s1],
                                 &mut synapse_alphas[s0..s1], &mut synapse_last_events[s0..s1], &producer);
        }
        DENDRITIC_SPIKE => {
            let d = e.source as usize;
            let n = dendrite_to_neuron[d] as usize;          // reverse map (chapter 3.2)
            let (s0, s1) = dendrite_synapse_range(d, synapse_offsets);
            handle_dendritic_spike(n, e.timestamp, &dendrite_constants[d], &mut dendrite_last_events[d],
                                   &mut soma_potentials[n], &soma_thresholds[n],
                                   &mut synapse_alphas[s0..s1], &mut synapse_last_events[s0..s1], &producer);
        }
        FORWARD_AP => {
            let n = e.source as usize;
            for &s in &axon_targets[axon_offsets[n] as usize .. axon_offsets[n+1] as usize] {
                let d = synapse_to_dendrite(s as usize, synapse_offsets);   // O(log) binary search
                let (s0, s1) = dendrite_synapse_range(d, synapse_offsets);
                handle_forward_ap(s as usize - s0, d, e.timestamp, &synapse_xs[s0..s1], /* ...scoped slices... */, &producer);
            }
        }
        _ => {}
    }
}
```

### Slice scoping is the safety model

Notice every handler receives **pre-narrowed slices**, computed by
`neuron_synapse_range` / `dendrite_synapse_range`
([chapter 3.2](03-data-model.md)). A handler literally cannot touch state outside
its component — the borrow checker enforces it, because it only holds a slice of
the relevant range. This is how the codebase gets memory-safety guarantees on
top of a flat, index-addressed layout.

One consequence worth internalizing: inside `handle_forward_ap`, the synapse
index is **local** (`s − s0`), because the slice it receives starts at the
dendrite's first synapse. The gamma loop in [chapter 6](06-learning-dynamics.md)
therefore iterates `0..slice.len()`, which is exactly the dendrite's synapses —
the sorted-`x` invariant holds *within the slice*.

### Signal flow recap

The three arms reproduce [chapter 1](01-theory.md)'s cascade:

```
FORWARD_AP  ─(per target synapse)─▶ boost alpha, integrate gamma → maybe DENDRITIC_SPIKE
DENDRITIC_SPIKE ─▶ push onto soma, boost active synapses → maybe SOMATIC_SPIKE
SOMATIC_SPIKE ─▶ update beta, BDP weight update on all synapses → emit FORWARD_AP (out the axon)
```

Each handler may `producer.push(...)` new events; because `drain()` snapshots
`head..tail` at loop entry, events pushed *during* this drain are not processed
until the next call to `run_event_loop`. The loop is therefore one **wavefront**
of propagation per call (see the trial loop in
[chapter 8](08-mnist-pipeline.md)).

> **Gap (flagged in the source).** The `FORWARD_AP` arm has an inner loop over
> axon targets — a serial fan-out. The code comment marks it as the prime
> candidate for batching / per-target parallel dispatch on the GPU.

## 5.4 The parallel-write hazard

The architecture's central unsolved concurrency problem ([chapter 2.3](02-architecture.md))
surfaces here concretely: if two `FORWARD_AP` events target synapses on the
**same dendrite** at the same timestamp, both handlers read-modify-write
`dendrite_activities[d]`. Serially (today) this is fine. Parallelized naively it
is a data race. Resolution strategies (atomic accumulate, per-dendrite event
coalescing, owner-thread-per-dendrite) are an open decision in
[chapter 9](09-gaps-and-open-questions.md).

---

Next: [chapter 6 — Learning dynamics](06-learning-dynamics.md), the handler
bodies themselves.
