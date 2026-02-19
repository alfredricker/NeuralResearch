# Network Definition Language

## Overview

A network is a hierarchical directed graph. The language provides syntax for:
- Declaring subgraph templates
- Instantiating and nesting subgraphs
- Defining connection topologies at any level of hierarchy
- Specifying transforms (how nodes process inputs)
- Specifying learning rules (how weights change)
- Attaching external inputs and outputs

The core principle: **topology, transform, and learning declarations mirror subgraph structure recursively**.

---

## Defaults

Set defaults at any scope. Inner scopes inherit and can override.
```
default topology: sparse(0.02);
default transform: sigma(sum(inputs));
default learn: hebbian(eta=0.01);
default timescale: {
    activation: 1,
    weights: 1000,
};
```

Defaults cascade inward. A default set at network level applies to all subgraphs unless overridden.

---

## Subgraphs

A subgraph is a reusable template. There are two forms:
- Leaf subgraph: declares only `nodes(...)` and optional behavior blocks.
- Composite subgraph: declares subgraph components and wiring between them.

### Leaf subgraph syntax (nodes only)
```
subgraph <name>(<optional params>) {
    nodes(<count>);

    // optional
    structure { ... }
    transform { ... }
    learn { ... }
    timescale { ... }
}
```

Leaf subgraphs cannot declare nested components (`name: other_subgraph;`) or ports.

### Leaf subgraph example
```
subgraph m(n) {
    nodes(n);

    structure {
        nodes -> nodes: sparse(0.02) & !identity;
    }

    transform {
        nodes: sigma(sum(inputs));
    }

    learn {
        nodes -> nodes: hebbian(eta=0.01, decay=0.0001);
    }

    timescale {
        activation: 1,
        weights: 1000,
    }
}
```

### Composite subgraph syntax (contains subgraphs)
```
subgraph <name> {
    <component>: <subgraph>(...);
    <component>: <subgraph>[n];
    ...

    structure { ... }
    transform { ... }   // optional
    learn { ... }       // optional

    port <name> = <component>;
    ...
}
```

### Composite subgraph example
```
subgraph layer {
    cols: column[16];
    
    structure {
        cols -> cols: ring(1) + ring(-1), {
            lateral -> lateral: sparse(0.05),
        };
    }
    
    learn {
        cols -> cols: {
            lateral -> lateral: hebbian(eta=0.005),
        };
    }
    
    port in = cols[*].in;
    port out = cols[*].out;
    port feedback_in = cols[*].feedback_in;
    port feedback_out = cols[*].feedback_out;
}
```

### Nesting depth is arbitrary
```
subgraph cortex {
    layers: layer[4];
    
    structure {
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
    }
    
    learn {
        layers[i] -> layers[i+1]: {
            cols -> cols: {
                out -> in: hebbian(eta=0.01),
            }
        };
        
        layers[i+1] -> layers[i]: {
            cols -> cols: {
                feedback_out -> feedback_in: modulated_hebbian(eta=0.01, signal=prediction_error),
            }
        };
    }
    
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

## Transforms

Transforms define how nodes compute their activation from inputs.

### Syntax

In a `transform` block, specify per-node or per-connection transforms:
```
transform {
    <node_group>: <activation_function>;
    <source> -> <target>: <connection_transform>;
}
```

### Activation primitives

| Syntax | Meaning |
|--------|---------|
| `identity` | output = input (for input nodes) |
| `sum(inputs)` | linear sum of weighted inputs |
| `sigma(x)` | saturating nonlinearity: x / (\|x\| + 1) |
| `relu(x)` | max(0, x) |
| `tanh(x)` | hyperbolic tangent |
| `sigmoid(x)` | 1 / (1 + exp(-x)) |
| `phase_ring` | ring attractor dynamics for grid cells |

### Composable transforms
```
sigma(sum(inputs))              // saturated sum
sigma(sum(inputs) - inhibition) // with lateral inhibition
gated(w, sigma(sum(inputs)))    // multiplicative gating by w
```

### Gating

The `gated` transform multiplies input by a gating signal:
```
transform {
    m: gated(w, sigma(sum(inputs)));
}
```

This means: m's activation = g(w_signal) * sigma(sum of feedforward inputs), where g is a gating function (e.g., g(x) = x² / (x² + θ²)).

### Inhibition

Lateral inhibition enforces sparsity:
```
transform {
    m: sigma(sum(inputs) - kappa * total_activity(m));
}
```

`total_activity(m)` is the sum of activations across all m nodes.

### Full activation dynamics

For explicit control over the update equation:
```
transform {
    m: dynamics {
        tau: 1,
        leak: 0.1,
        update: (1 - leak) * prev + sigma(sum(inputs) - kappa * total_activity(m)),
    };
}
```

### Custom transforms
```
fn gated_saturate(inputs, gate_inputs, params) {
    let gate = gate_inputs.sum().pow(2) / (gate_inputs.sum().pow(2) + params.theta.pow(2));
    let drive = inputs.weighted_sum();
    let inhibition = params.kappa * inputs.total_activity();
    return sigma(gate * drive - inhibition);
}
```

Then use:
```
transform {
    m: gated_saturate(theta=0.5, kappa=0.1);
}
```

---

## Learning Rules

Learning rules define how weights change over time.

### Syntax
```
learn {
    <source> -> <target>: <rule>;
}
```

### Primitive rules

| Rule | Update formula | Description |
|------|----------------|-------------|
| `none` | Δw = 0 | Fixed weights, no learning |
| `hebbian(eta)` | Δw = η · pre · post · (1 - w) | Fire together, wire together |
| `hebbian(eta, decay)` | Δw = η · pre · post · (1 - w) - μ · w | With weight decay |
| `anti_hebbian(eta)` | Δw = -η · pre · post · w | Decorrelation |
| `oja(eta)` | Δw = η · post · (pre - w · post) | Normalized Hebbian |
| `stdp(A_plus, A_minus, tau)` | Δw = f(Δt) | Spike-timing dependent |
| `covariance(eta)` | Δw = η · (pre - E[pre]) · (post - E[post]) | Covariance rule |

### Parameters

| Parameter | Meaning | Typical range |
|-----------|---------|---------------|
| `eta` | Learning rate | 0.001 - 0.1 |
| `decay` | Weight decay rate | 0.0001 - 0.001 |
| `bounds` | Weight bounds | [-1, 1] or [0, 1] |
| `tau` | Time constant for STDP | 10 - 50 ms |

### Examples
```
learn {
    omega -> m: hebbian(eta=0.01);
    m -> m: hebbian(eta=0.01, decay=0.0001, bounds=[0, 1]);
    w -> m: none;
    m -> z: oja(eta=0.005);
}
```

### Modulated learning

Learning can be gated by a modulatory signal (reward, prediction error, attention):
```
learn {
    omega -> m: hebbian(eta=0.01) * modulator(nu);
}
```

Or declare the modulator at subgraph level:
```
subgraph column {
    modulator nu: prediction_error;
    
    learn(modulated_by=nu) {
        omega -> m: hebbian(eta=0.01);
        m -> m: hebbian(eta=0.01);
    }
}
```

When `nu` is high, learning is amplified. When `nu` is low, learning is suppressed.

### Modulator sources

| Source | Meaning |
|--------|---------|
| `prediction_error` | Difference between predicted and actual input |
| `reward` | External reward signal |
| `attention` | Top-down attention signal |
| `novelty` | Inverse familiarity |
| `custom(signal_name)` | User-defined signal |

### Prediction error computation
```
subgraph column {
    signal predicted = feedback_in.activation;
    signal actual = omega.activation;
    modulator error: magnitude(actual - predicted);
    
    learn(modulated_by=error) {
        omega -> m: hebbian(eta=0.01);
    }
}
```

### Custom learning rules
```
fn my_hebbian(pre, post, w, params) {
    let eta = params.eta;
    let decay = params.decay ?? 0.0;
    let bounds = params.bounds ?? [0.0, 1.0];
    
    let delta = eta * sigma(pre) * sigma(post) * (1.0 - w);
    delta = delta - decay * w;
    
    return clamp(w + delta, bounds[0], bounds[1]);
}
```

Use:
```
learn {
    omega -> m: my_hebbian(eta=0.01, decay=0.0001);
}
```

### Hierarchical learning

Learning rules cascade through hierarchy:
```
learn {
    layers[i] -> layers[i+1]: {
        cols -> cols: {
            out -> in: hebbian(eta=0.01),
        }
    };
    
    layers[i+1] -> layers[i]: {
        cols -> cols: {
            feedback_out -> feedback_in: anti_hebbian(eta=0.005),
        }
    };
}
```

---

## Timescales

Separate fast dynamics (activation) from slow dynamics (learning).

### Syntax
```
timescale {
    activation: <tau>;
    expectation: <tau>;
    weights: <tau>;
}
```

### Example
```
subgraph column {
    timescale {
        activation: 1,      // updates every tick
        expectation: 100,   // running averages for learning signals
        weights: 1000,      // weight updates 1000x slower than activation
    }
}
```

### Per-connection timescales
```
learn {
    omega -> m: hebbian(eta=0.01), timescale(1000);
    m -> m: hebbian(eta=0.001), timescale(10000);  // slower for recurrent
}
```

---

## Signals

Signals are named values computed from network state, used for modulation or readout.

### Syntax
```
signal <name> = <expression>;
modulator <name>: <signal_expression>;
```

### Examples
```
signal total_m_activity = sum(m.activation);
signal prediction = feedback_in.activation;
signal error = magnitude(omega.activation - prediction);

modulator nu: error;
```

### Built-in signal functions

| Function | Meaning |
|----------|---------|
| `sum(x)` | Sum of activations |
| `mean(x)` | Mean activation |
| `max(x)` | Maximum activation |
| `magnitude(x)` | L2 norm |
| `sparsity(x)` | Fraction of active units |

---

## Wiring Syntax

### Basic form
```
structure {
    <source> -> <target>: <topology>;
}
```

### Hierarchical form

When source and target contain substructure, topology must specify how subcomponents connect:
```
structure {
    <source> -> <target>: <inter_topology>, {
        <sub_source> -> <sub_target>: <topology>;
        ...
    };
}
```

### Directionality rule

**In `A -> B: { x -> y }`, x is always a component of A, y is always a component of B.**

The arrow direction cascades through all nesting levels.

### Index expressions

| Syntax | Meaning |
|--------|---------|
| `cols[i]` | The i-th column (loop variable) |
| `cols[i+1]` | Successor of i-th column |
| `cols[i-1]` | Predecessor of i-th column |
| `cols[*]` | All columns (for port aggregation) |
| `cols[0]` | First column (literal index) |
| `layers[i] -> layers[i+1]` | Each layer to its successor |

### Examples

Same-level lateral connections:
```
structure {
    cols -> cols: ring(1) + ring(-1), {
        lateral -> lateral: sparse(0.05),
    };
}
```

Cross-level feedforward:
```
structure {
    layers[i] -> layers[i+1]: {
        cols -> cols: all, {
            out -> in: sparse(0.2),
        }
    };
}
```

Cross-level feedback:
```
structure {
    layers[i+1] -> layers[i]: {
        cols -> cols: identity, {
            feedback_out -> feedback_in: sparse(0.1),
        }
    };
}
```

---

## Input and Output

External data sources and sinks attach to the network through adapters.

### Declaring external interfaces
```
input <name>: <type>;
output <name>: <type>;
```

### Input/Output types

| Type | Description |
|------|-------------|
| `Image(h, w)` | 2D grayscale image |
| `Image(h, w, c)` | 2D image with c channels |
| `Sequence(n)` | 1D sequence of length n |
| `Sequence(*, vocab)` | Variable-length sequence over vocabulary |
| `Class(n)` | Classification into n categories |
| `Vector(n)` | n-dimensional real vector |
| `Scalar` | Single real value |

### Input wiring

Input requires a **spatial mapping** to distribute data across subgraphs, then **topology** for node-level connections.
```
structure {
    <input> -> <target>: <spatial_mapping>, {
        <input_component> -> <target_component>: <topology>;
    };
}
```

### Spatial mappings

| Mapping | Meaning |
|---------|---------|
| `patch(h, w)` | Partition into non-overlapping h×w patches |
| `patch(h, w, overlap=n)` | Overlapping patches with n pixels overlap |
| `stride(h, w, step_h, step_w)` | Strided windows |
| `broadcast` | Same input to all targets |
| `identity` | Direct 1-to-1 mapping |

### Input example
```
input MNIST: Image(28, 28);

structure {
    MNIST -> layers[0].cols: patch(7, 7, overlap=2), {
        pixels -> in: identity,
    };
}
```

### Output wiring

Output requires an **aggregation mapping** to collect from subgraphs, then **topology** for node-level connections.
```
structure {
    <source> -> <output>: <aggregation>, {
        <source_component> -> <output_component>: <topology>;
    };
}
```

### Aggregations

| Aggregation | Meaning |
|-------------|---------|
| `pool` | All sources contribute to single output |
| `concat` | Concatenate source outputs |
| `spatial(h, w)` | Arrange outputs spatially |
| `vote` | Consensus voting across sources |

### Output transforms

| Transform | Meaning |
|-----------|---------|
| `weighted_sum` | Weighted sum of inputs |
| `softmax` | Softmax normalization |
| `argmax` | Index of maximum |
| `threshold(t)` | Binary threshold |

### Output example
```
output classification: Class(10);

structure {
    layers[2].cols -> classification: pool, {
        out -> logits: weighted_sum,
    };
}

transform {
    classification: softmax;
}
```

---

## Complete Example
```
// Global defaults
default topology: sparse(0.02);
default transform: sigma(sum(inputs));
default learn: hebbian(eta=0.01);

// Leaf subgraphs
subgraph omega(n) {
    nodes(n);

    transform {
        nodes: identity;
    }
}

subgraph m(n) {
    nodes(n);

    structure {
        nodes -> nodes: sparse(0.02) & !identity;
    }

    transform {
        nodes: sigma(sum(inputs) - kappa * total_activity(nodes));
    }

    learn(modulated_by=prediction_error) {
        nodes -> nodes: hebbian(eta=0.01, decay=0.0001);
    }

    timescale {
        activation: 1,
        weights: 1000,
    }
}

subgraph w(n) {
    nodes(n);

    transform {
        nodes: phase_ring([5, 7, 11]);
    }
}

subgraph z(n) {
    nodes(n);
}

// Composite subgraph
subgraph column {
    omega: omega(49);
    m: m(500);
    w: w(23);
    z: z(100);
    
    // Signals for modulation
    signal predicted = feedback_in.activation;
    signal actual = omega.activation;
    modulator prediction_error: magnitude(actual - predicted);
    
    structure {
        omega -> m: sparse(0.1);
        m -> m: sparse(0.02) & !identity;
        w -> m: all;
        m -> z: sparse(0.1);
    }
    
    transform {
        m: gated(w, m);
        z: sigma(sum(inputs));
    }

    learn(modulated_by=prediction_error) {
        omega -> m: hebbian(eta=0.01);
        w -> m: none;
        m -> z: hebbian(eta=0.01);
    }

    port in = omega;
    port out = z;
    port lateral = m;
    port feedback_in = omega;
    port feedback_out = m;
}

// Composite subgraph
subgraph layer {
    cols: column[16];
    
    structure {
        cols -> cols: ring(1) + ring(-1), {
            lateral -> lateral: sparse(0.05),
        };
    }
    
    learn {
        cols -> cols: {
            lateral -> lateral: hebbian(eta=0.005),
        };
    }
    
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
    
    structure {
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
    
    transform {
        classification: softmax;
    }
    
    learn {
        layers[i] -> layers[i+1]: {
            cols -> cols: {
                out -> in: hebbian(eta=0.01),
            }
        };
        
        layers[i+1] -> layers[i]: {
            cols -> cols: {
                feedback_out -> feedback_in: anti_hebbian(eta=0.005),
            }
        };
        
        layers[2] -> classification: {
            out -> logits: hebbian(eta=0.02),
        };
    }
}
```

---

## Summary

| Block | Purpose | Scope |
|-------|---------|-------|
| `structure { }` | Define which connections exist | subgraph, network |
| `transform { }` | Define how nodes compute activations | subgraph, network |
| `learn { }` | Define how weights update | subgraph, network |
| `timescale { }` | Define update rates | subgraph |
| `signal` | Named computed value | subgraph |
| `modulator` | Learning rate modulation | subgraph |
| `default` | Set defaults for nested scopes | any |

| Declaration | Syntax |
|-------------|--------|
| Define subgraph | `subgraph name { ... }` |
| Create leaf nodes | `nodes(n);` |
| Instantiate parameterized subgraph | `name: other_subgraph(args);` |
| Instantiate subgraph | `name: other_subgraph;` |
| Array of subgraphs | `name: other_subgraph[n];` |
| Expose port | `port name = component;` |
| Aggregate port | `port name = components[*].port;` |
| Declare input | `input name: Type(...);` |
| Declare output | `output name: Type(...);` |
| Wire (flat) | `a -> b: topology;` |
| Wire (hierarchical) | `a -> b: inter, { x -> y: topo; };` |
| Node transform | `node: function;` |
| Connection learning | `a -> b: rule(params);` |
| Modulated learning | `learn(modulated_by=x) { ... }` |