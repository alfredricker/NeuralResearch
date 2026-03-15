# Claude Suggestions 1

---

## Edge & Topology Syntax

The challenge is that a fully specified edge has three orthogonal concerns:

1. **Topology** — which nodes connect to which (`Sparse`, `Ring`, `All`, ...)
2. **Morphism** — the transformation applied on the forward pass (`conv2d`, `relu`, ...)
3. **Learning rule** — how weights update (`SGD`, `Hebbian`, `STDP`, ...)

The current syntax handles topology and morph but hasn't addressed learning rules yet. The sketches below explore different ways to pack all three in.

---

### Option A — Current syntax, extended with a `learn` clause

Keep what exists, just add a `learn` keyword:

```stn
x ~> y : Sparse(0.2) |> relu learn Hebbian
```

Or on a named edge:
```stn
e = x ~> y : Sparse(0.2)
e |> relu
e learn Hebbian(lr=0.01)
```

**Pro:** Minimal change from current syntax.
**Con:** Three different operators on one line reads dense. `learn` as a keyword feels slightly awkward.

---

### Option B — Block body on an edge

Let a topology expression accept an optional `{ }` body for forward and learning:

```stn
x ~> y : Sparse(0.2) {
    forward(input) { input |> relu }
    learn: Hebbian(lr=0.01)
}
```

Named:
```stn
e = x ~> y : Sparse(0.2) {
    forward(input) { input |> conv2d(16, 3) |> relu }
    learn: STDP(tau=20.0)
}
```

**Pro:** Clean separation. Each concern lives in its own named slot. Extensible — easy to add `init`, `mask`, etc. later.
**Con:** More verbose for simple cases (most edges are just topology + `dyn`).

---

### Option C — Topology in brackets, morph after arrow

Encode topology structurally in the arrow itself:

```stn
x -[Sparse(0.2)]-> y |> relu
x -[Ring(1), dyn]-> y
x -[All]-> y |> conv2d(16, 3) |> relu learn STDP
```

**Pro:** Topology is visually "inside" the connection, morph is outside. Hard to mix up the two.
**Con:** `-[...]->`  is somewhat noisy. Would need to decide whether `~` variant exists: `~[...]~>`.

---

### Option D — `edge` declaration as its own construct

Treat an edge as a first-class named object like `node` or `subgraph`:

```stn
edge ConvEdge(Cin: u32, Cout: u32) {
    topology: Sparse(0.2)
    dyn kernel: tsr[f32; Cout, Cin, 3, 3] = KaimingUniform()
    dyn bias:   tsr[f32; Cout]            = Zeros()

    forward(x: tsr[f32; Cin, H, W]) -> tsr[f32; Cout, H, W] {
        x |> Conv2d(kernel, bias)
    }

    learn: SGD
}

// Applied to a topology:
x ~> y : ConvEdge(3, 16)
```

**Pro:** Mirrors how `node` works. Clean if edges get complicated (e.g. memory heads).
**Con:** Overkill for most edges. Two-step (define + apply) may feel heavy.

---

### Option E — Pipe-first syntax (morph-centric)

Instead of topology being primary, make the morph primary and topology a modifier:

```stn
x |[Sparse(0.2)]|> y relu
```

Or more readably, drop the arrows entirely for morphs:

```stn
x ~> y : Sparse(0.2)               // topology only — structural
x |> y via relu                     // morph only — no topology keyword needed
x |[Sparse(0.2)]|> y via relu      // combined
```

**Pro:** `|>` already means "transform," so it reads naturally for the morph-heavy case.
**Con:** Two different connection operators (`~>` and `|>`) could confuse when to use which.

---

### Option F — Sections per concern (explicit separation)

Use named section keywords inside a graph or subgraph:

```stn
subgraph layer(n: u32) {
    // === Structure ===
    x = Nodes(n) : tsr[f32; n]
    e = x ~> x : Sparse(0.2), dyn

    // === Dynamics ===
    forward {
        e |> relu
    }

    // === Learning ===
    learn {
        e @ Hebbian(lr=0.01)
    }
}
```

**Pro:** Maximum clarity for complex subgraphs. Documents intent.
**Con:** Verbose. May be better suited for a `node` body than inline on an edge.

---

### Recommendation

Options **B** and **D** seem like the best foundations. **B** (block body on an edge) handles the 80% case elegantly while staying close to current syntax. **D** (`edge` as a first-class construct) is the natural extension for reusable edge types — analogous to how `node` handles reusable node types. They compose well: use inline block syntax for one-off edges, use `edge` declarations for repeated patterns.

---

## Language Name

"Subgraph Topology Network" is a description, not a name. Some directions:

---

### Neural / Brain Metaphors

**Axon** — The wire between neurons. Short (4 letters), pronounceable, accurate: the job of the language is to wire nodes together.

**Synapse** — The connection point. Slightly longer, but very evocative. Could abbreviate to `syn`.

**Soma** — The cell body (the node). Poetic but people won't know it.

**Dendrite** — Too long, too specific.

---

### Graph / Topology Metaphors

**Quiver** — A *quiver* is the actual mathematical term for a directed graph (a set of vertices and arrows). It's precise, unusual, and memorable. Short enough to type. No existing language has this name.

**Weave** — Evokes the interlacing of topology. Easy to say and spell. Could describe the act of composing graphs.

**Lattice** — Mathematical, clean. Neural network layers often are lattices. Slightly generic.

**Braid** — Topological term. Short, memorable. Implies interleaving of subgraphs.

---

### Data Flow Metaphors

**Flow** — Clean and universal. Too generic to be distinctive.

**Flux** — Change over time. Fits the dynamic step / spiking neuron angle. Short.

**Drift** — Subtle. Fits diffusion-based networks aesthetically.

**Pulse** — Spiking network feel. Short, energetic.

---

### Fabrication / Craft Metaphors

**Loom** — A loom weaves threads into fabric; the language weaves nodes into networks. 4 letters, memorable, unused.

**Stitch** — Connecting subgraphs together.

**Graft** — Grafting subgraphs onto a growing network.

---

### Short / Invented

**Nexus** — Connection hub. Clear meaning, sounds slightly corporate.

**Topos** — Mathematical (a topos is a category with topological properties). Obscure but satisfying if you know it.

**Nodus** — Latin for "knot" or "node." Unusual, precise.

---

### Ranked shortlist

| Name | Why it works |
|------|-------------|
| **Quiver** | Mathematically exact, completely distinctive, short |
| **Axon** | Instantly legible neural metaphor, 4 letters |
| **Loom** | Weaving metaphor, 4 letters, unused, evocative |
| **Flux** | Dynamic, short, fits spiking/recurrent angle |
| **Weave** | Topology metaphor, readable, verb-as-noun |
