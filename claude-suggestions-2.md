# Claude Suggestions 2 — Topology Syntax & Primitives

---

## Reframing the Problem

A connection between two node sets has exactly two independent parts:

1. **Topology** — a pattern over index pairs: given source size `n` and target size `m`, which `(i, j)` pairs are connected
2. **Edge** — what lives on each connection: `dyn` weights, a `dynamic forward()`, a learning rule

These are orthogonal. The topology says *which* connections exist; the edge says *what* each connection computes. The syntax should reflect this.

---

## Applying Topology + Edge Type

The cleanest read: "connect A to B using topology T, instantiating edge type E at each connected pair."

```quiver
a ~> b : Sparse(0.2) via ConvEdge(Cin=3, Cout=16)
```

- `:` gives the **topology**
- `via` names the **edge type** instantiated at each pair

Simple cases still work unchanged:

```quiver
a ~> b : Sparse(0.2)          // topology only, default scalar weight
a ~> b : All via DenseEdge    // dense topology, custom edge
a ~> b : Identity             // 1-to-1 wiring
```

If an `edge` declaration has a natural topology, it can declare a default:

```quiver
edge ConvEdge(Cin: u32, Cout: u32, K: u32) {
    default topology: Local(K)

    dyn kernel: tsr[f32; Cout, Cin, K, K] = KaimingUniform()
    dyn bias:   tsr[f32; Cout]            = Zeros()

    dynamic forward(x: tsr[f32; Cin, H, W]) -> tsr[f32; Cout, H, W] {
        x |> Conv2d(kernel, bias)
    }
}

a ~> b via ConvEdge(3, 16, 3)            // uses Local(3) from default
a ~> b : Sparse(0.2) via ConvEdge(3, 16, 3)  // override
```

---

## Defining Custom Topology Patterns

### What a topology compiles to

A **static** topology runs at graph construction time and produces a CSR adjacency matrix. A **dynamic** topology re-runs at each forward pass and produces an adjacency matrix that can change. The compiler infers which is which from whether the topology body contains a `dynamic` block.

### Syntax: loop body calling `edge()`

The most CS-friendly and most directly compilable form. The topology body is a loop that calls `edge(src, dst)` for each connection to create:

```quiver
topology circulant(offsets: [i32], n: u32) {
    for i in 0..n {
        for k in offsets {
            edge(i, (i + k) % n)
        }
    }
}

topology grid4(H: u32, W: u32) {
    for i in 0..H {
        for j in 0..W {
            if j + 1 < W  { edge(i*W + j, i*W + j+1) }   // right
            if i + 1 < H  { edge(i*W + j, (i+1)*W + j) }  // down
        }
    }
}

topology stride(s: u32, n: u32, m: u32) {
    for i in 0..m {
        edge(i*s, i)
    }
}
```

The compiler runs these loops at build time, collects all `edge()` calls, and emits a CSR matrix.

### Dynamic topologies

A `dynamic` block inside a topology body marks code that re-runs each forward pass. The loop outside runs once at construction (e.g. to set up a structure), the loop inside `dynamic` re-evaluates each step:

```quiver
topology knn(k: u32, n: u32) {
    dynamic {
        for i in 0..n {
            neighbors = TopK(j where j != i, by: Dist(pos[i], pos[j]), k=k)
            for j in neighbors {
                edge(i, j)
            }
        }
    }
}

topology threshold(t: f32, n: u32, m: u32) {
    dynamic {
        for i in 0..n {
            for j in 0..m {
                if similarity(i, j) > t { edge(i, j) }
            }
        }
    }
}
```

Using a topology with a `dynamic` block automatically signals to the compiler that the connection requires runtime adjacency rather than a prebuilt CSR.

### Composition

Topologies can be combined without writing a loop body:

```quiver
union(A, B)         // edges in A or B
intersect(A, B)     // edges in both A and B
diff(A, B)          // edges in A but not B
complement(A)       // all edges not in A (relative to All)
```

```quiver
topology SmallWorld(k: u32, p: f32, n: u32) =
    union(Ring(k, n), Sparse(p, n))

topology LocalNoSelf(r: u32, n: u32) =
    diff(Local(r, n), Identity(n))
```

Using named functions rather than operator symbols (`|`, `&`, `\`) avoids symbol overload and is unambiguous to a CS reader.

---

## A Note on Set-Builder Notation

The earlier draft used `{ (i, j) | i <- 0..n, j <- 0..m, condition }` — this is Haskell-style list comprehension syntax where `<-` means "drawn from." It was dropped for two reasons:

1. The `|` separator conflicts with `|` as union in topology composition, and `|i - j|` (absolute value) creates a third meaning in the same expression.
2. It requires mathematical set notation literacy. The for-loop form compiles the same way and is immediately readable to anyone who has written code.

---

## Standard Topology Primitives

### Density

| Topology | Description |
|----------|-------------|
| `All(n, m)` | Complete bipartite — every source to every target |
| `None` | No connections |
| `Identity(n)` | Node `i` to node `i` — requires equal sizes |
| `Sparse(p, n, m)` | Each edge independently with probability `p` (Erdős–Rényi) |
| `KRegular(k, n)` | Each node has exactly `k` outgoing connections, randomly chosen |

### Circulant (Shift-Invariant)

All of these are special cases of `Circulant(offsets, n)` — node `i` connects to `(i + offset) % n` for each offset. A 1D convolution is circulant connectivity with a learnable weight at each offset.

| Topology | Equivalent |
|----------|-----------|
| `Ring(k, n)` | `Circulant([k], n)` |
| `Cycle(n)` | `Circulant([1], n)` |
| `Local(r, n)` | `Circulant([-r..-1, 1..r], n)` — neighborhood, no self |

The 2D analog is `GridLocal(H, W, r)` — each node at position `(i, j)` connects to all nodes within L∞ distance `r`. This is the topological skeleton of a 2D convolution.

### Structural

| Topology | Description |
|----------|-------------|
| `Stride(s, n, m)` | Target node `i` receives from source node `i*s` — downsampling |
| `Star(n)` | Node 0 connects to all others |
| `Path(n)` | Node `i` to node `i+1`, open chain |
| `Tree(b, n)` | Node `i` connects to `i*b .. i*b + b-1` — branching factor `b` |
| `Grid4(H, W)` | 2D lattice, 4-connected (von Neumann neighborhood) |
| `Grid8(H, W)` | 2D lattice, 8-connected (Moore neighborhood) |

### Dynamic (Runtime-Evaluated)

| Topology | Description |
|----------|-------------|
| `KNN(k)` | Connect to k nearest neighbors by distance over node positions |
| `Threshold(t)` | Edge `(i,j)` exists if `similarity(i, j) > t` |
| `TopK(k)` | Each node keeps its top-k incoming connections by weight magnitude |

---

## Derivation Tree

```
All
├── Sparse(p)       = All, keep each edge with prob p
├── KRegular(k)     = All, keep exactly k per node
└── Identity        = All, keep only diagonal

Circulant(offsets)
├── Ring(k)         = Circulant([k])
├── Cycle           = Circulant([1])
└── Local(r)        = Circulant([-r..-1, 1..r])

GridLocal(H, W, r)
└── 2D conv shape   = GridLocal where each connection shares a kernel position

Stride(s)
└── Pooling shape   = GridLocal on the source side, Stride on the output side

SmallWorld(k, p)    = union(Ring(k), Sparse(p))
```
