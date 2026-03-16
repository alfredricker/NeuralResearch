# Edges and Topology

Edges connect nodes and carry data between them. Every edge has two independent concerns: the **topology** (which nodes connect to which) and the **edge type** (what computation and weights live on each connection). These are specified separately using `:` and `via`.

```quiver
a ~> b : Sparse(0.2) via ConvEdge(Cin=3, Cout=16, K=3);
```

- `:` specifies the **topology pattern**
- `via` specifies the **edge type** instantiated at each connected pair

Both are optional. Omitting `via` gives a bare connection with a scalar weight. Omitting `:` uses the edge type's `default topology` if one is declared.

---

## Directed vs Symmetric

`~>` creates directed edges (one-way):

```quiver
a ~> b : All;
```

`~` creates symmetric edges (bidirectional, shared weights):

```quiver
x ~ x : All;    // fully symmetric self-connections (e.g. Hopfield)
```

---

## Built-in Topology Patterns

### `All`
Every source node connects to every target node. For `n` source and `m` target nodes this creates `n*m` edges.

```quiver
a ~> b : All;
```

### `Sparse(p)`
Each edge exists independently with probability `p`. No self-connections.

```quiver
a ~> b : Sparse(0.1);    // ~10% of possible edges
```

### `Identity`
Node `i` in the source connects to node `i` in the target. Requires equal counts.

```quiver
a ~> b : Identity;
```

### `Ring(k)`
Node `i` connects to node `(i + k) mod n`.

```quiver
x ~> x : Ring(1);    // each node connects to its right neighbor
```

### `None`
No connections. Useful as a placeholder.

```quiver
a ~> b : None;
```

---

## Named Topology Variables

Assign a topology to a variable to apply further properties or reference it later:

```quiver
lateral = x ~> x : Sparse(0.2);
lateral : Scale(0.1);
```

---

## Learnable Weights with `dyn`

By default, topology patterns create fixed structural connections with a scalar weight per edge. The `dyn` keyword makes those weights learnable:

```quiver
a ~> b : All, dyn;           // fully learnable dense connections
a ~> b : Sparse(0.2), dyn;   // sparse structure fixed, weights learned
a ~> b : Ring(1), dyn;       // ring structure fixed, weights learned
```

Named learnable weight matrices can be shared across multiple edge sets:

```quiver
// Both edges share the same weight matrix W_lat
L1[0].out ~> L1[1].in : All, dyn W_lat;
L1[1].out ~> L1[0].in : All, dyn W_lat;
```

---

## Declaring Edge Types

For edges that carry more than a scalar weight — custom forward logic, multiple `dyn` parameters, or a default topology — use an `edge` block.

```quiver
edge ConvEdge(Cin: u32, Cout: u32, K: u32) {
    default topology: Local(K)

    dyn kernel: tsr[f32; Cout, Cin, K, K] = KaimingUniform();
    dyn bias:   tsr[f32; Cout]            = Zeros();

    dynamic forward(x: tsr[f32; Cin, H, W]) -> tsr[f32; Cout, H, W] {
        x |> Conv2d(kernel, bias)
    }
}
```

An `edge` block can contain:

| Field | Role |
|-------|------|
| `default topology` | Topology used when no `:` is given at the call site |
| `dyn` | Learnable parameter belonging to this edge |
| Fixed fields | Non-learned constants (e.g. `stride`, `pad`) |
| `dynamic forward()` | The computation run on each connected pair during the forward pass |

### Applying an Edge Type

```quiver
// Uses ConvEdge's default topology (Local(3))
feature_map ~> next via ConvEdge(3, 16, 3);

// Override the topology
feature_map ~> next : Sparse(0.2) via ConvEdge(3, 16, 3);
```

Post-hoc application on a named connection:

```quiver
ff = input ~> hidden : All;
ff : via ConvEdge(3, 16, 3);
```

### Edge Type with Fixed Parameters

Non-`dyn` fields in an `edge` block are fixed constants that are not learned:

```quiver
edge PoolEdge(K: u32, stride: u32) {
    default topology: Stride(stride)

    dynamic forward(x: tsr[f32; K]) -> f32 {
        Max(x)
    }
}
```

---

## Edge Dynamics

The `dynamic forward()` block is the forward pass of the edge. It receives the source node's `out` value and returns the value written to the target node's `in`. The compiler aggregates all incoming edges at a target node by summing their outputs unless overridden.

```quiver
edge DenseEdge(in_dim: u32, out_dim: u32) {
    dyn W: tsr[f32; out_dim, in_dim] = KaimingUniform();
    dyn b: tsr[f32; out_dim]         = Zeros();

    dynamic forward(x: tsr[f32; in_dim]) -> tsr[f32; out_dim] {
        W @ x + b
    }
}
```

For edges with no `dynamic forward()`, the edge simply passes the source value through scaled by its weight — equivalent to:

```quiver
dynamic forward(x) -> x {
    weight * x
}
```
