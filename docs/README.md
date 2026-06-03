# NeuralResearch — Design Documentation

A biologically-inspired spiking neural network simulator in Rust, structured for
eventual GPU execution. The near-term goal is a working MNIST learner; the
long-term goal is biologically realistic **Burst-Dependent Plasticity (BDP)**.

This documentation is written as a progression. Each chapter assumes the
previous ones. Read in order the first time.

## Table of contents

1. [Theory](01-theory.md) — the biological model, the state variables
   (`alpha`, `beta`, `x`, `gamma`), and the learning rule. *No code.*
2. [Architecture choices](02-architecture.md) — SoA layout, offset arrays, the
   event-driven (no-tick) execution model, fixed-width integer types, GPU
   orientation.
3. [Data model](03-data-model.md) — the concrete SoA arrays (`Soma`,
   `Dendrite`, `Synapse`, `Axon`), the offset/stride conventions, and how a flat
   index resolves to a neuron / dendrite / synapse.
4. [Math primitives](04-math-primitives.md) — `shift_decay`, the alias-method
   samplers, midpoints. The leaf computations everything else is built from.
5. [The event system](05-event-system.md) — event types, the ring-buffer queue,
   the unsafe-isolated producer, the dispatch loop, and slice scoping.
6. [Learning dynamics](06-learning-dynamics.md) — the handlers in detail:
   synaptic alpha, dendritic gamma-integration, somatic BDP weight updates, and
   apical feedback.
7. [Network construction](07-network-construction.md) — config → population →
   builder → connection rules → allocator. The fixed-slot synapse model and the
   `live_count` active-synapse iteration design.
8. [The MNIST pipeline](08-mnist-pipeline.md) — topology, input encoding, the
   trial loop, readout, and training feedback.
9. [Gaps and open questions](09-gaps-and-open-questions.md) — a consolidated,
   prioritized list of everything that is stubbed, undecided, or known-broken.
10. [Appendix — GPU execution and residency](10-gpu-execution-and-residency.md) —
    segmented reduction, the event-buffer kernel pattern, and how the network is
    partitioned into VRAM (Hawkes-driven hot-loading, METIS vs. biological cuts).
11. [The IO boundary](11-io-boundary.md) — `src/io/`: input spaces and the
    sensory arrow (afferent), effectors and the readout arrow (efferent), and how
    pixels become events and spikes become predictions. Underpins chapter 8.
12. [Time and the network clock](12-time-and-clocking.md) — the one thing the
    event-driven model leaves open: what sets a `timestamp` and what advances
    time. The design options for giving the network a clock.

### Resources

- [Index relationships (Mermaid diagrams)](resources/index-relationships.md) —
  how neuron / dendrite / synapse / axon connect through offset arrays and
  reverse lookups, with a worked example and the axonal fan-out path (a somatic
  spike's `SYNAPSE_SIGNAL`s) traced through indices.

## Status at a glance

**Implemented and unit-tested** — the leaf biophysics, the event machinery, *and*
construction: `math/decay`, `math/sample`, `neuron/{synapse,dendrite,soma}`,
`network/event/*` (queue, producer, handlers, loop, slice), the allocator
(`neuron/population`), the orchestrator (`network/build`), the connection resolver
(`network/topology/conn`), and the `io/` boundary (`InputSpace`/`Effector`). The
network now **builds, wires, and resolves its axon CSR**.

**The remaining gap** — the end-to-end *trial loop*: the event ring buffer never
recycles slots (`head` is never advanced), `run_event_loop` does not accumulate
the `spike_counts` the readout needs, nothing advances a clock, and there is no
hidden-layer config yet. See [chapter 9](09-gaps-and-open-questions.md).

**Event model** — four event types: `SOMATIC_SPIKE`, `DENDRITIC_SPIKE`,
`SOMA_SIGNAL`, `SYNAPSE_SIGNAL` (the older `FORWARD_AP` / `APICAL_FB` are gone). A
burst count rides the event payload as a multiplier
([chapter 5](05-event-system.md)).

> Source-of-truth note: any root-level `CLAUDE.md` that still references
> `FORWARD_AP`/`APICAL_FB`, or `taxonomy/`/`init/` directories, is stale. This
> documentation reflects the tree as it actually stands.
