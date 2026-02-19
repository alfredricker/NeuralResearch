# Network Definition Language

## Overview

A network is a hierarchical directed graph. The language provides syntax for:
- Declaring subgraph templates
- Instantiating and nesting subgraphs
- Defining connection topologies at any level of hierarchy
- Attaching external inputs and outputs

The core principle: **topology declarations mirror subgraph structure recursively**.

---

## Subgraphs

A subgraph is a reusable template containing nodes or other subgraphs.

### Basic syntax
```
subgraph <name> {
    <component>: <definition>;
    ...
    
    <internal wiring>
    
    port <name> = <component>;
    ...
}
```

### Leaf subgraph (contains nodes)
```
subgraph column {
    omega: nodes(49);
    m: nodes(500);
    w: nodes(23);
    z: nodes(100);
    
    // internal wiring
    omega -> m: sparse(0.1);
    m -> m: sparse(0.02) & !identity;
    w -> m: gate;
    m -> z: sparse(0.1);
    
    // exposed interfaces
    port in = omega;
    port out = z;
    port lateral = m;
    port feedback_in = omega;
    port feedback_out = m;
}
```

### Composite subgraph (contains subgraphs)
```
subgraph layer {
    cols: column[16];
    
    // lateral wiring between columns
    cols -> cols: ring(1) + ring(-1), {
        lateral -> lateral: sparse(0.05),
    };
    
    // expose aggregate ports
    port in = cols[*].in;
    port out = cols[*].out;
}
```

### Nesting depth is arbitrary
```
subgraph cortex {
    layers: layer[4];
    
    // feedforward
    layers[i] -> layers[i+1]: {
        cols -> cols: all, {
            out -> in: sparse(0.2),
        }
    };
    
    // feedback
    layers[i+1] -> layers[i]: {
        cols -> cols: all, {
            feedback_out -> feedback_in: sparse(0.1),
        }
    };
    
    port in = layers[0].in;
    port out = layers[3].out;
}
```

---

## Topologies

Topologies define connection patterns between node sets.

### Primitives

| Syntax | Meaning |
|--------|---------|
| `all` | every source to every target |
| `none` | no connections |
| `identity` | source[i] to target[i], requires equal size |
| `sparse(p)` | each pair connected with probability p |
| `k_random(k)` | each target receives exactly k random sources |
| `ring(d)` | source[i] to target[(i+d) mod n] |
| `gate` | gating connection (multiplicative rather than additive) |

### Combinators

| Syntax | Meaning |
|--------|---------|
| `A + B` | union of edges |
| `A & B` | intersection of edges |
| `!A` | complement (all edges not in A) |

### Examples
```
ring(1) + ring(-1)          // bidirectional ring
all & !identity             // all-to-all except self
sparse(0.1) & !identity     // sparse, no self-connections
```

---

## Wiring Syntax

### Basic form
```
<source> -> <target>: <topology>;
```

### Hierarchical form

When source and target contain substructure, topology must specify how subcomponents connect:
```
<source> -> <target>: <inter_topology>, {
    <sub_source> -> <sub_target>: <topology>;
    ...
};
```

The `inter_topology` determines which subgraph pairs connect.
The nested block determines how nodes within each pair connect.

### Directionality rule

**In `A -> B: { x -> y }`, x is always a component of A, y is always a component of B.**

The arrow direction cascades through all nesting levels.

### Index expressions
```
cols[i] -> cols[i+1]    // each column to its successor
cols[i] -> cols[i-1]    // each column to its predecessor  
cols[*]                 // all columns (for port aggregation)
layers[0]               // specific index
```

### Examples

Same-level lateral connections:
```
cols -> cols: ring(1), {
    m -> m: sparse(0.05),
};
```

Cross-level feedforward:
```
layers[i] -> layers[i+1]: {
    cols -> cols: all, {
        z -> omega: sparse(0.2),
    }
};
```

Cross-level feedback:
```
layers[i+1] -> layers[i]: {
    cols -> cols: identity, {
        m -> omega: sparse(0.1),
    }
};
```

---

## Input and Output

External data sources and sinks attach to the network through adapters.

### Declaring external interfaces
```
input <name>: <type>;
output <name>: <type>;
```

Types include:
- `Image(h, w)` - 2D image
- `Image(h, w, c)` - 2D image with channels
- `Sequence(n)` - 1D sequence
- `Class(n)` - classification into n categories
- `Vector(n)` - n-dimensional vector

### Input wiring

Input requires a **spatial mapping** to distribute data across subgraphs, then **topology** for node-level connections.
```
<input> -> <target>: <spatial_mapping>, {
    <input_component> -> <target_component>: <topology>;
};
```

Spatial mappings:
| Syntax | Meaning |
|--------|---------|
| `patch(h, w)` | partition into non-overlapping patches |
| `patch(h, w, overlap=n)` | overlapping patches |
| `stride(h, w, step_h, step_w)` | strided windows |
| `broadcast` | same input to all targets |

Example:
```
input MNIST: Image(28, 28);

MNIST -> layers[0].cols: patch(7, 7, overlap=2), {
    pixels -> omega: identity,
};
```

This partitions the 28x28 image into overlapping 7x7 patches, one per column, then wires each patch's pixels directly to that column's omega nodes.

### Output wiring

Output requires an **aggregation mapping** to collect from subgraphs, then **topology** for node-level connections.
```
<source> -> <output>: <aggregation>, {
    <source_component> -> <output_component>: <topology>;
};
```

Aggregations:
| Syntax | Meaning |
|--------|---------|
| `pool` | all sources contribute to single output |
| `concat` | concatenate source outputs |
| `spatial(h, w)` | arrange outputs spatially |

Example:
```
output classification: Class(10);

layers[2].cols -> classification: pool, {
    z -> logits: weighted_sum,
};
```

This pools all columns' z neurons into a single vote, using weighted sum to produce 10 class logits.

---

## Complete Example
```
// Node group templates
subgraph column {
    omega: nodes(49);
    m: nodes(500);
    w: nodes(23);
    z: nodes(100);
    
    omega -> m: sparse(0.1);
    m -> m: sparse(0.02) & !identity;
    w -> m: gate;
    m -> z: sparse(0.1);
    
    port in = omega;
    port out = z;
    port lateral = m;
    port feedback_in = omega;
    port feedback_out = m;
}

subgraph layer {
    cols: column[16];
    
    cols -> cols: ring(1) + ring(-1), {
        lateral -> lateral: sparse(0.05),
    };
    
    port in = cols[*].in;
    port out = cols[*].out;
    port feedback_in = cols[*].feedback_in;
    port feedback_out = cols[*].feedback_out;
}

// External interfaces
input MNIST: Image(28, 28);
output classification: Class(10);

// Network
network MNISTClassifier {
    layers: layer[3];
    
    // Input
    MNIST -> layers[0]: patch(7, 7, overlap=2), {
        pixels -> in: identity,
    };
    
    // Feedforward
    layers[i] -> layers[i+1]: {
        cols -> cols: all, {
            out -> in: sparse(0.2),
        }
    };
    
    // Feedback
    layers[i+1] -> layers[i]: {
        cols -> cols: all, {
            feedback_out -> feedback_in: sparse(0.1),
        }
    };
    
    // Output
    layers[2] -> classification: pool, {
        out -> logits: weighted_sum,
    };
}
```

---

## Summary

| Concept | Syntax |
|---------|--------|
| Define subgraph | `subgraph name { ... }` |
| Create nodes | `name: nodes(n);` |
| Instantiate subgraph | `name: other_subgraph;` |
| Array of subgraphs | `name: other_subgraph[n];` |
| Expose port | `port name = component;` |
| Aggregate port | `port name = components[*].port;` |
| Wire (flat) | `a -> b: topology;` |
| Wire (hierarchical) | `a -> b: inter, { x -> y: topo; };` |
| Declare input | `input name: Type(...);` |
| Declare output | `output name: Type(...);` |
| Input wiring | `input -> target: spatial, { ... };` |
| Output wiring | `source -> output: aggregation, { ... };` |