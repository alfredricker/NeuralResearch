# STN Language Overview

STN (Subgraph Topology Network) is a declarative language for specifying neural network architectures as graphs. The core model is simple: **nodes hold data, edges transform it**.

## Mental Model

A neural network in STN is a directed graph where:
- **Nodes** are typed containers holding activation values, hidden states, or learnable parameters
- **Edges** define how data flows and transforms between nodes
- **Topology** specifies the connection pattern (sparse, ring, all-to-all, etc.)
- **Morphisms** define the computation that runs along an edge

This model is general enough to express feedforward networks, RNNs, spiking neural networks, Hopfield networks, and cortical column architectures within the same framework.

## Key Syntax at a Glance

| Symbol | Meaning |
|--------|---------|
| `:` | Type annotation or property assignment |
| `~>` | Directed topology (edge from left to right) |
| `~` | Symmetric topology (bidirectional edge) |
| `->` | Function / morph return type |
| `\|>` | Morphism pipeline (apply transformation) |
| `@` | Matrix multiplication |
| `*` | Ring product (Hadamard elementwise) |
| `**` | Exponentiation |
| `%` | Modulus |

## Naming Convention

Built-in functions and types are **Capitalized** (`Nodes`, `Sparse`, `ReLU`, `Conv2d`).
User-defined functions, morphs, and variables are **lowercase** (`my_layer`, `encode`, `hidden`).

## Documentation Sections

- [Data Types](02_data_types.md) — primitives, tensors, alternate algebras
- [Operators](03_operators.md) — all symbols and their semantics
- [Variables & Properties](04_variables_and_properties.md) — assignment and the `:` operator
- [Nodes](05_nodes.md) — `out`, `state`, `dyn`, `dynamic`
- [Edges & Topology](06_edges_and_topology.md) — connection patterns
- [Functions & Morphs](07_functions_and_morphs.md) — `fn`, `morph`, `|>` pipelines
- [Subgraphs](08_subgraphs.md) — reusable parameterized blocks
- [Graphs](09_graph.md) — top-level graph declaration
- [Algebras](10_algebras.md) — custom algebraic structures
