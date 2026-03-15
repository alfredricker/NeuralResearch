# Nodes

Nodes are the fundamental data-holding units of a Quiver graph. A node definition specifies what data it holds, what state it carries across time steps, and what learnable parameters belong to it.

## Instantiating Node Sets

Any node type is instantiated into a set using square brackets:

```quiver
x = Node[10];          // 10 built-in scalar nodes
h = LSTMCell[50];      // 50 LSTMCell nodes
s = LIF[128];          // 128 Leaky Integrate-and-Fire neurons
```

The number in brackets is the count — a `u32` that can be a literal or a parameter passed from an enclosing `subgraph`.

## The Built-in `Node` Type

`Node` is the default node type. It holds a single scalar value whose type is inferred from its connected edges — the compiler ensures that all edges meeting at a `Node` have compatible types. `Node` carries no state and no dynamics: its output at any step is simply whatever its incoming edges wrote to it.

```quiver
x = Node[10]               // 10 nodes, type inferred from context
x = Node[10] : tsr[f32; 128]  // explicitly typed
```

Any custom node definition that declares only an `out` field and nothing else is semantically equivalent to `Node` with that type. The `node` keyword exists for when you need state, learnable parameters, or a custom update rule.

## Node Fields

| Keyword | Role |
|---------|------|
| `out` | The current output value — what outgoing edges read |
| `state` | Persistent value that survives across time steps |
| `dyn` | Learnable intrinsic parameter (belongs to the node, not an edge) |
| `dynamic` | An update function that runs each time step, writing to `out` and `state` |

## Recurrent Nodes

Nodes with `state` persist values between time steps:

```quiver
node LSTMCell(hidden: u32) {
    out:     tsr[f32; hidden]
    state h: tsr[f32; hidden] = Zeros()
    state c: tsr[f32; hidden] = Zeros()
}

node GRUCell(hidden: u32) {
    out:     tsr[f32; hidden]
    state h: tsr[f32; hidden] = Zeros()
}
```

Instantiated the same way as any other node type:

```quiver
lstm = LSTMCell[50]    // 50 LSTM cells, each with hidden dim inferred from edges
gru  = GRUCell[32]
```

If the node type takes parameters, pass them in parentheses before the count brackets:

```quiver
lstm = LSTMCell(hidden=128)[50]    // 50 cells, hidden dim explicitly 128
```

## Nodes with Dynamics

The `dynamic` block defines the update rule run each time step. It receives the aggregated input from incoming edges and writes to `out` and any `state` fields.

```quiver
// Leaky Integrate-and-Fire spiking neuron
node LIF {
    out:   f32                    // spike output (0 or 1)
    state v_m:   f32 = 0.0       // membrane potential
    state t_ref: u32 = 0         // refractory countdown

    dyn threshold: f32 = 1.0     // learnable firing threshold
    dyn tau:       f32 = 20.0    // membrane time constant

    dynamic step(input: f32) {
        if t_ref > 0 {
            v_m   = v_m * (1.0 - 1.0/tau)
            t_ref = t_ref - 1
            out   = 0.0
        } else {
            v_m = v_m * (1.0 - 1.0/tau) + input
            if v_m >= threshold {
                out   = 1.0
                v_m   = 0.0
                t_ref = 5
            } else {
                out = 0.0
            }
        }
    }
}
```

```quiver
node AdaptiveLIF {
    out:   f32
    state v_m:   f32 = 0.0
    state theta: f32 = 1.0       // adaptive threshold (state, not fixed)

    dyn tau_m:    f32 = 20.0
    dyn tau_th:   f32 = 100.0
    dyn delta_th: f32 = 0.1      // threshold increment per spike

    dynamic step(input: f32) {
        v_m   = v_m   * (1.0 - 1.0/tau_m) + input
        theta = theta * (1.0 - 1.0/tau_th)
        if v_m >= threshold {
            out   = 1.0
            v_m   = 0.0
            theta = theta + delta_th
        } else {
            out = 0.0
        }
    }
}
```

## Stochastic Nodes

```quiver
// VAE latent node — edges write to mu and log_var, node samples out
node Stochastic(dim: u32) {
    out:     tsr[f32; dim]
    mu:      tsr[f32; dim]
    log_var: tsr[f32; dim]

    dynamic sample() {
        eps = RandnLike(mu)
        out = mu + Exp(0.5 * log_var) * eps
    }
}

// Bayesian node — weight uncertainty lives in the node itself
node BayesianDense(dim: u32) {
    out:      tsr[f32; dim]
    dyn mu_w:    tsr[f32; dim] = Zeros()
    dyn log_sig: tsr[f32; dim] = Zeros()

    dynamic forward(input: tsr[f32; dim]) {
        w   = mu_w + Exp(log_sig) * RandnLike(mu_w)
        out = w * input
    }
}
```

## Oscillatory Nodes

```quiver
node Oscillator {
    out:   c32                // complex amplitude — encodes phase and magnitude
    state phase: f32 = 0.0

    dyn freq:    f32 = 1.0   // natural frequency (learnable)
    dyn damping: f32 = 0.01

    dynamic step(input: c32, dt: f32) {
        phase = phase + 2*pi*freq*dt
        out   = (1.0 - damping) * out + input
        out   = out * Exp(i * phase) / out.norm()
    }
}
```

## Memory-Augmented Nodes

```quiver
node ExternalMemory(mem_size: u32, head_dim: u32) {
    out:     tsr[f32; head_dim]
    state M: tsr[f32; mem_size, head_dim] = Zeros()
    state w: tsr[f32; mem_size]           = Softmax(Ones())

    // Write and read heads are edges that update M and w via their forward()
}
```
