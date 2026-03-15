# Nodes

Nodes are the fundamental data-holding units of an STN graph. A node definition specifies what data it holds, what state it carries across time steps, and what learnable parameters belong to it.

## Node Fields

A `node` block can contain four kinds of declarations:

| Keyword | Role |
|---------|------|
| `out` | The current output value — what outgoing edges read |
| `state` | Persistent value that survives across time steps |
| `dyn` | Learnable intrinsic parameter (belongs to the node, not an edge) |
| `dynamic` | A step function that updates `out` and `state` each tick |

## Simple Feedforward Nodes

Nodes that just hold an activation vector — no recurrence, no dynamics:

```stn
node Dense(dim: u32) {
    out: tsr[f32; dim]
}

node ImageNode(C: u32, H: u32, W: u32) {
    out: tsr[f32; C, H, W]
}
```

## Recurrent Nodes

Nodes with `state` persist values between time steps:

```stn
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

## Nodes with Dynamics

The `dynamic` block defines the update rule run each time step. It receives external input and updates `out` and any `state` fields.

```stn
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

### Adaptive threshold variant

```stn
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
        if v_m >= theta {
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

Nodes that sample from a distribution each forward pass:

```stn
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

Nodes that evolve a phase over time — useful for binding and synchrony:

```stn
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

Nodes that maintain an external memory matrix:

```stn
node ExternalMemory(mem_size: u32, head_dim: u32) {
    out:     tsr[f32; head_dim]
    state M: tsr[f32; mem_size, head_dim] = Zeros()
    state w: tsr[f32; mem_size]           = Softmax(Ones())

    // Write and read heads are edges that update M and w via their forward()
}
```
