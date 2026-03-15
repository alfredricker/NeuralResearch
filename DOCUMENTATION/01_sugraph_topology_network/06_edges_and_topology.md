# Edges and Topology

Edges connect nodes and carry data between them. The topology of a set of edges — which nodes connect to which — is specified using built-in topology functions.

## Directed vs Symmetric

`~>` creates directed edges (one-way):
```stn
a ~> b    // data flows from a to b
```

`~` creates symmetric edges (bidirectional, shared weights):
```stn
a ~ b     // data flows both ways with the same weights
x ~ x     // fully symmetric self-connections (e.g. Hopfield)
```

## Built-in Topology Patterns

All patterns are applied with the `:` operator on a topology expression.

### `All`
Every node in the source connects to every node in the target. For `n` source and `m` target nodes, this creates `n*m` edges.

```stn
a ~> b : All
```

### `Sparse(p)`
Each edge exists independently with probability `p`. No self-connections.

```stn
a ~> b : Sparse(0.1)    // ~10% of possible edges
```

### `Identity`
Node `i` in the source connects to node `i` in the target. Source and target must have the same count.

```stn
a ~> b : Identity
```

### `Ring(k)`
Node `i` connects to node `(i + k) mod n`. Useful for locally connected topologies.

```stn
x ~> x : Ring(1)    // each node connects to its right neighbor
```

### `None`
No connections. Useful as a placeholder when topology will be defined later.

```stn
a ~> b : None
```

## Named Topologies

Assign a topology to a variable to apply multiple properties or reference it later:

```stn
lateral = x ~> x : Sparse(0.2)
lateral : Scale(0.1)
```

## Learnable Topology

The `dyn` modifier on a topology makes the edge weights learnable parameters:

```stn
a ~> b : dyn          // fully learnable dense connections
a ~> b : dyn W_ff     // named learnable weight matrix
```

Named learnable weights can be shared across multiple edge sets:

```stn
// Both edges share the same weight matrix
L1[0].out ~> L1[1].in : dyn W_lat
L1[1].out ~> L1[0].in : dyn W_lat
```

## Fixed vs Learnable

By default, topology patterns like `Sparse` and `Ring` create fixed (non-learnable) structural connections. To make the weights on those connections learnable, combine with `dyn`:

```stn
x ~> x : Ring(1), dyn    // ring structure is fixed, weights are learned
```

## Edge Properties

Beyond topology, edges accept properties that affect initialization or transformation:

```stn
a ~> b : Sparse(0.2), Scale(0.01)   // sparse + small initial weights
```
