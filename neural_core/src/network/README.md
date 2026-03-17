# `network/` — Generic Graph Network Framework

Wires together cortical regions (or any `Node`) into a directed compute graph.
Two-layer design: a **build-time logical graph** you assemble with `NetworkBuilder`,
compiled down to a **run-time flat graph** (`FlatGraph`) that executes efficiently.

---

## Types

### `port.rs`

**`Aggregation`** — Policy for combining multiple incoming wires into one port.
- `Concat`: destination dim == sum of all source dims. Slices are written end-to-end.
- `Sum`: all source dims == destination dim. Slices are element-wise accumulated.

**`PortSpec`** — Declares a single port: its name, dimensionality, and aggregation policy.

**`PortValues`** — Named slice container passed to `Node::tick` / `Node::learn`.
Holds one `Vec<f32>` per port, addressable by name.

---

### `node.rs`

**`Node`** — Trait every compute unit must implement.
Declares its input/output ports, processes data in `tick`, and applies learning in `learn`.
`CorticalRegionNode` (in `region/cortical.rs`) is the primary implementor.

---

### `graph.rs`

**`Wire`** — A directed connection between two ports in a `SubgraphDef`.
Carries a `recurrent` flag for connections that feed back values from the previous tick.

**`NodeOrSubgraph`** — A child slot inside a `SubgraphDef`: either a concrete `Node`
or a nested `SubgraphDef` to be recursively flattened.

**`SubgraphDef`** — Build-time logical graph. Holds an ordered map of named children
and a list of wires between them. Can be nested arbitrarily deep.
Produced by `NetworkBuilder::into_subgraph()` for embedding in a larger graph.

**`FlatWire`** — A connection in the flattened graph, using integer node indices
instead of names.

**`FlatGraph`** — Run-time flat graph produced by flattening and validating a
`SubgraphDef`. Stores nodes, separated feedforward and recurrent wire lists,
a topologically sorted execution order, and per-node input/output buffers.
Call `tick()` to propagate activations and `learn()` to apply weight updates.

---

### `builder.rs`

**`NetworkBuilder`** — Fluent API for assembling a `SubgraphDef` and compiling it.
Add nodes with `.add_node()`, embed subgraphs with `.add_subgraph()`,
connect ports with `.wire()`, and call `.build()` to get a validated `FlatGraph`.

**`WireBuilder`** — Returned by `.wire()`. Lets you optionally call `.recurrent()`
to mark a back-edge before continuing to chain builder calls.

---

### `flatten.rs`

**`BuildError`** — Enum of all validation failures: unresolved node/port names,
`Concat`/`Sum` dimension mismatches, and feedforward cycles (missing `.recurrent()` flag).

**`flatten_and_build`** — Internal function that recursively walks a `SubgraphDef` tree
into a flat `(nodes, wires)` list, validates aggregation dimensions, runs Kahn's
topological sort on feedforward wires, and allocates all runtime buffers.
