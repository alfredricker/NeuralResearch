# Core Primitives — What Actually Needs to Be a Keyword?

## The Problem

Quiver is accumulating keywords like a framework. Every concept gets its own keyword: `graph`, `subgraph`, `node`, `edge`, `morph`, `fn`, `dynamic`, `rule`, `loss`, `optimizer`, `train`, `gate`, `probe`... This is the path toward a bloated DSL that's rigid rather than powerful.

A good language has a small number of orthogonal primitives that compose freely. Everything else is library.

## What Are We Actually Doing?

Strip away the neural network framing. What does a user of this language fundamentally need to express?

1. **Structure** — collections of things, and connections between them
2. **Transformation** — mapping values from one form to another
3. **State + Time** — things that evolve, persist, update
4. **Algebra** — the rules for how values combine

That's it. Everything else — attention, convolutions, STDP, loss functions, optimizers, topologies — is a *use* of these primitives, not a primitive itself.

## Proposed Core Keywords

```
let         — bind a name
fn          — pure transformation (no state)
dynamic     — stateful transformation (evolves over time)
type        — define a composite type or alias
algebra     — define algebraic structure (multiplication rules)
in / out    — declare ports (interfaces)
for         — iteration
if / else   — branching
import      — bring in external definitions
```

That's roughly 10 keywords. Everything else is std or user-defined.

## How the Core Works

### `let` — Structure is just binding

Nodes, edges, graphs, subgraphs — these are all just named collections with relationships. You don't need a keyword for each:

```quiver
// A "node set" is just a typed array
let x = [f32; 128]          // 128 floats
let h = [tsr[f32; 64]; 50]  // 50 nodes, each holding a 64-dim vector

// A "graph" is just the top-level scope. No keyword needed.
// A "subgraph" is just a function that returns structure.
```

### `->` and `~` — Connections are operators, not keywords

```quiver
let x = [f32; 100]
let y = [f32; 100]

x -> y          // directed connection (topology is a property, not a keyword)
x ~ y           // symmetric connection
```

Topology patterns like `Sparse(0.2)`, `Ring(1)`, `All` — these are **std library functions** that return connection masks. They don't need to be built-in:

```quiver
import std.topology : { Sparse, Ring, All, KNN }

x -> y : Sparse(0.2)
x -> y : Ring(1) | Sparse(0.05)   // composition via set operators
```

### `fn` — Pure transformations

```quiver
fn relu(x: tsr[f32; ..]) -> tsr[f32; ..] {
    max(x, 0.0)
}

fn softmax(x: tsr[f32; N]) -> tsr[f32; N] {
    let e = exp(x - max(x))
    e / sum(e)
}
```

`max`, `exp`, `sum` — these are std, not keywords.

### `dynamic` — The powerful primitive

This is where the language earns its keep. `dynamic` defines something that has **state and a rule for how it evolves**. This single keyword replaces `node` (with dynamics), `edge` (with forward), `rule`, `optimizer`, `train`, and anything else that updates over time.

```quiver
// A spiking neuron — what was "node LIF"
dynamic lif(input: f32) -> f32 {
    state v: f32 = 0.0
    state t_ref: u32 = 0

    param threshold: f32 = 1.0    // "param" could just be "let" with a learnable marker
    param tau: f32 = 20.0

    v = v * (1.0 - 1.0/tau) + input
    if v >= threshold {
        out = 1.0
        v = 0.0
        t_ref = 5
    } else {
        out = 0.0
    }
}

// A learning rule — what was "rule hebbian"
dynamic hebbian(pre: f32, post: f32, w: f32, lr: f32 = 0.01) -> f32 {
    w + lr * pre * post
}

// An optimizer — what was "optimizer Adam"
dynamic adam(grad: tsr[f32; ..], lr: f32 = 0.001) -> tsr[f32; ..] {
    state m: tsr[f32; ..] = 0.0
    state v: tsr[f32; ..] = 0.0
    state t: u32 = 0

    t = t + 1
    m = 0.9 * m + 0.1 * grad
    v = 0.999 * v + 0.001 * grad ** 2
    let m_hat = m / (1.0 - 0.9 ** t)
    let v_hat = v / (1.0 - 0.999 ** t)
    -lr * m_hat / (sqrt(v_hat) + 1e-8)
}
```

The insight: a neuron, a learning rule, and an optimizer are **all the same thing** — a stateful transformation that evolves over time. `dynamic` unifies them.

### `type` — Composite types replace `node`, `edge` keywords

```quiver
// What was "node LSTMCell" — now it's just a type with dynamics
type LSTMCell {
    out: tsr[f32; H]
    state h: tsr[f32; H] = zeros()
    state c: tsr[f32; H] = zeros()

    dynamic step(input: tsr[f32; H]) {
        // ... LSTM equations ...
    }
}

// Instantiate it — no special "node" syntax
let lstm = [LSTMCell(H=128); 50]    // 50 LSTM cells
```

### `algebra` — Stays as-is

This is genuinely a primitive. Defining multiplication tables for number systems is foundational and can't be pushed to a library without losing compile-time verification.

```quiver
algebra Dual over f32 {
    basis { 1, eps }
    relations { eps * eps = 0.0 }
}
```

## What Moves to `std`

Everything that's currently a keyword or built-in but is really just a specific use of the primitives:

```
std.topology    — Sparse, All, Ring, Identity, KNN, Lattice, SmallWorld, ...
std.init        — Zeros, Ones, KaimingUniform, XavierNormal, ...
std.activation  — ReLU, GELU, Sigmoid, Tanh, Softmax, ...
std.transform   — Linear, Conv2d, BatchNorm, Dropout, LayerNorm, ...
std.loss        — MSE, CrossEntropy, KLDivergence, ...
std.optim       — SGD, Adam, AdamW, ...
std.schedule    — CosineAnnealing, LinearWarmup, ...
std.random      — Normal, Uniform, Bernoulli, Sample, ...
std.aggregate   — Sum, Mean, Max, Concat, ...
std.math        — exp, log, sqrt, abs, sin, cos, pi, ...
```

## What This Buys You

1. **Composability** — `dynamic` composes with `fn`, with types, with algebra. No special cases.
2. **Extensibility** — Users define new neuron types, learning rules, optimizers without needing new syntax.
3. **Fewer concepts** — A beginner learns ~10 keywords. The complexity lives in the library, not the grammar.
4. **The graph is still there** — `->` and `~` are operators, not keywords. The graph structure is implicit in the connections, not in a `graph {}` wrapper.
5. **Time is unified** — Everything stateful uses `dynamic`. A neuron and an optimizer are the same abstraction.

## Open Questions

- **Does `->` need to be a special operator, or can connections be expressed as function calls?** E.g., `connect(x, y, Sparse(0.2))`. The operator syntax is nicer to read, but it's one more piece of grammar.
- **How does the compiler know what's the "top level" structure?** Without a `graph` keyword, is it just the file scope? That's probably fine.
- **`state` and `param` inside `dynamic` — are these keywords or could they be annotations?** `state` feels primitive enough to keep. `param` (learnable) might just be `let` with some marker.
- **Execution model** — synchronous vs async, time stepping. This could be a property on the top-level scope rather than a keyword: `@sync` or `execution: synchronous` as metadata.
