# Operators

Quiver uses a small set of symbols that each carry a distinct semantic role. Understanding these is the key to reading any Quiver program.

## `:` — Type Annotation and Property Assignment

The colon attaches a type or property to a variable. It is used both at declaration time and after the fact.

```quiver
x = Node[10] : f32             // nodes holding f32 scalars
x = Node[10] : tsr[f32; 128]  // nodes holding 128-dim vectors
```

Multiple properties can be applied in a single statement by separating them with commas:

```quiver
e = x ~> x : Sparse(0.2) via Scale(5.0)
```

Properties can also be applied on separate lines after declaration:

```quiver
e = x ~> x
e : Sparse(0.2)
e : via Scale(5.0)
```

The compiler knows the set of valid properties for each object kind. Applying an incompatible property is a compile error.

## `~>` — Directed Topology

Creates directed edges from a source group of nodes to a target group. The connection pattern is specified with `:`.

```quiver
a ~> b : Sparse(0.1);   // 10% of possible a→b edges, random
a ~> b : Identity;      // node i connects to node i only
a ~> b : Ring(1);       // node i connects to node (i+1) mod n
a ~> b : All;           // every node in a connects to every node in b
a ~> b : None;          // no connections (placeholder)
```

A topology expression can be named for later property assignment:

```quiver
edges = a ~> b : Sparse(0.2);
edges : via Scale(0.1);
```

## `~` — Symmetric Topology

Creates bidirectional (undirected) edges. Equivalent to edges in both directions sharing the same weights.

```quiver
a ~ b : Sparse(0.3);     // undirected sparse connections between a and b
x ~ x : All;             // fully connected undirected (e.g. Hopfield network)
```

## `->` — Return Type

Used in `fn` and `morph` declarations to specify the output type. This is purely a type-level annotation, not a graph edge.

```quiver
fn softmax(x: tsr[f32; n]) -> tsr[f32; n] { ... }

morph encode(x: tsr[f32; 784]) -> tsr[f32; 128] { ... }
```

## `|>` — Morphism Pipeline

Threads a value through a sequence of transformations left-to-right. The output type of each stage must be compatible with the input type of the next.

```quiver
x |> Flatten() |> Linear(784, 128) |> ReLU();
```

Pipelines can span multiple lines:

```quiver
x |> Conv2d(in=1, out=16, kernel=3, padding=1)
  |> ReLU()
  |> MaxPool2d(kernel=2, stride=2);
```

## Arithmetic Operators

| Operator | Meaning |
|----------|---------|
| `@` | Matrix multiplication / linear map composition |
| `*` | Ring product — elementwise (Hadamard) for tensors |
| `**` | Exponentiation, elementwise |
| `+` | Addition, elementwise |
| `-` | Subtraction, elementwise |
| `/` | Division, elementwise |
| `%` | Modulus |

`@` and `*` have distinct roles: `@` composes linear maps (contracts inner dimensions), while `*` multiplies element-by-element.

```quiver
C = A @ B;      // matrix multiply: [m,k] @ [k,n] => [m,n]
C = A * B;      // Hadamard:        [m,n] * [m,n] => [m,n]
```

## Indexing and Slicing

```quiver
x[i]        // single element
x[a..b]     // slice from index a to b (exclusive)
x[:, i]     // all rows, column i
```
