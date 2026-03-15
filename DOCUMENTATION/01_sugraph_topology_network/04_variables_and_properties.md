# Variables and Properties

## Variables

Variables are declared with `=`. They can hold nodes, edge topologies, or any named expression.

```stn
x = Nodes(10)          // 10 nodes
e = x ~> x             // directed self-connections on x
```

Variables serve two purposes:
1. **Reuse** — reference the same object in multiple places
2. **Property attachment** — give a name to something so properties can be applied to it

## Properties

A property is metadata attached to an object using `:`. Properties refine what an object is — its data type, connection pattern, scale, etc.

```stn
x : f32                // x holds 32-bit floats
x : tsr[f32; 128]      // x holds 128-dimensional vectors
e : Sparse(0.2)        // e has 20% random connectivity
e : Scale(0.1)         // e weights are initialized scaled by 0.1
```

The compiler enforces that a property is valid for the kind of object it is applied to. Applying a topology property to a node set, or a data type to a topology, is a compile error.

## Inline Declaration

Properties can be applied on the same line as declaration:

```stn
x = Nodes(10) : tsr[f32; 128]
e = x ~> x : Sparse(0.2)
```

## Multiple Properties

Comma-separate properties to apply several at once:

```stn
e = x ~> x : Sparse(0.2), Scale(0.1)
```

This is equivalent to:

```stn
e = x ~> x
e : Sparse(0.2)
e : Scale(0.1)
```

## Scope

Variables are scoped to the block they are declared in (`graph`, `subgraph`, `node`, `morph`). Inner blocks can reference variables from enclosing blocks, but not the reverse.
