I am adopting the convention that user defined functions should be lowercase while built-in functions should be capitalized.

# Data Types
The purpose of a network is to take in data, process it, and output data. Therefore I adopt the simple general framework where *Nodes* are pieces of data and *Edges* transform the data either symmetrically or directionally.

There are many data types that are useful for deep learning. The most important of these are floats and ints. These translate directly to the primitive rust types.
### floats and ints
* **Floats:** f16, f32, f64
* **Unsigned Ints:** u8, u16, u32, u64
* **Ints:** i8, i16, i32, i64

### alternate algebras
There are to be more exotic types and groups included as data types in the framework.  
* **Complex Numbers:** c16, c32, c64
* **Split Complex Numbers:** sc16, sc32, sc64
* **Quaternions:** q16, q32

## Tensor type
Tensors are a generalization of matrices and can perform arbitrary dimensional linear transformations.
To declare a tensor, use the `tsr` type. We use the square brackets for the `tsr` because there are an arbitrary number of elements
* `tsr[f32; 3]` // a tensor that contains 32 bit floats and 3 dimensions. This is the same as a 3 dimensional vector.
* `tsr[sc16; 4, 4, 4, 4]` // this is a 4x4x4x4 = 64 dimensional tensor where each entry is a split complex number.

The compiler may also eventually support different dimensions of the tensor to support different datatypes
` tsr[4: f16, 6: i32]; `


# Variables and properties
Variables are commonplace in programming languages because of their adaptability and utility. Variables exist within the STN framework to apply multiple properties to a given objects, as well as to define objects that are to be reused.

You can set a variable to any built in type or type you have custom built. For example, to set 10 empty nodes (these carry null data type)

```
x = Nodes(10);
```

To set our variable to carry data, we must apply a *property* to it

```
x : tsr[f16; 3, 3];
```

This will successfully compile because `tsr` is a data type, x is a set of `Nodes()` objects, and a set of `Nodes()` apply data types as a property.

Another example is applying properties to topologies. Let's say we want to define a random connection of edges among our nodes so that 20% of them are connected. We can use the edge syntax and apply the built in `Sparse()` topology function. What if we also want to define a constant multiplier to the Tensor? The `Scale()` function is built in and compatible with almost all data types.
```
a = x -> x;
a : sparse(0.2);
a : scale(5);
```
The compiler is smart and knows whether you are applying a transformation function or a topology function. If the given code after a colon is not an applicable property, the compiler will throw an error.

There is a convenient shorthand to the previous code examples. You can apply properties in the same line as a declaration and apply multiple properties at once.
```
x = Nodes(10) : tsr[f16; 3, 3];
y = x -> x : sparse(0.2), scale(5);
```

# Transformations
Transformations are the way to manipulate data between edges. 
## Out Of the Box
Many transformations are included out of the box
* `conv2d(k, out s, p)`
* `relu()`
* `maxpool(k, s)`
* `flatten()`
* `scale()`
* `sigmoid()`

## Building Transforms
You can build custom transformations using the `morph` keyword.
```
cnn = morph(x : tsr[f32; 1, C, H]) :  

## Applying Transformations to Edges
Transformations are properties that can be applied to edges using the edge `->` and property `:` syntax.
```stn
x = Nodes(50);
y = Nodes(20);
x -> y : 
```


# Subgraphs

Imagine a CNN as a graph structure. The nodes could be the feature vectors / activation functions and the edges are the weight transformations.

```stn
subgraph layer {
    // declare the nodes and the type of data structure that it holds
    x = Nodes(10) : tsr[f32; 1, 28, 28];
}
```

You should also be able to pass args to definitions such as layer().

```stn
subgraph layer(n: i32) {
    x = Nodes(n) : f32;
}
```

How should we define the topology and transformation type? Perhaps the following example is good syntax.
```stn
subgraph layer {
    x = Nodes(10): tsr[f32; 12, 12];
    // topology definition
    x -> x : Sparse(0.2);
}
Where we use the syntax `x -> x` to define connection topologies. To apply a transformation along a set of edges, we need to assign a name to the topology.
```stn
subgraph layer {
    x = Nodes(10) : tsr[f32; 12 12];
    // spx is a var name -- can be anything. I chose spx as shorthand for "Sparse x"
    spx = x -> x : Sparse(0.2);
    spx : 
}

## Graph With 10 Layers
To define a graph with 10 layers, we can write the following.
```stn
subgraph layer(n: i32) {
    x = Nodes(n) : f32;
}

graph {
    layer[0..9] // creates 10 layers
    index(i, 0..8) {
        // Compiler should interpret topological statements to the nodes
        layer[i] -> layer[i+1] : Identity; // same as layer[i].x -> layer[i+1].x : Identity.
    }

}
```
To allow for breaking up code into smaller blocks, you can have multiple graph statements. Note that this will still edit the master graph (there is only one graph, it can be built from many subgraphs).


# Graph Declaration
The final object of interest is our complete neural network. This is the graph that we can apply learning rules to, feed streams of information, train, etc. Let's start by declaring a disconnected graph of 50 `nodes`. 

```stn
graph {
    nodes(50);
}
```

This graph simply generates 50 empty node objects that could be loaded into memory.

### Declaring edges

Edges connect nodes. The arrow `->` creates directed connections from left to right.
```stn
graph {
    graph_nodes = nodes(50);
    graph_nodes -> graph_nodes all;
}
```

This creates 50 nodes where every node connects to every other node (2500 edges).

The topology `all` specifies the connection pattern. Other patterns:
```
nodes -> nodes: sparse(0.1);   // 10% of possible edges, random -- no identity connections
nodes -> nodes: identity;      // node i connects to node i only
nodes -> nodes: ring(1);       // node i connects to node (i+1) mod n
nodes -> nodes: none;          // no connections
```

# Inputs

Inputs should be interpreted in a node, edge framework. A certain subset of the input domain maps to a defined set of n points in the network. The job for the coder is to define the domain space of the inputs, and how it maps to the network.

```stn
image = Input: Tsr[f32; 8, 8, 8]


```


# Writing a CNN
It makes sense to adopt the following syntax:
`:` assigns a data type
`->` defines a topology
`|>` defines a morphism

```stn
// convert a 3x32x32 tensor to a 16x16x16 tensor
morph first_conv(x : tsr[f32; 32, 32, 3]) : tsr[f32; 16, 16, 16] {
    x |> Conv2d(in=3, out=16, kernel=3, padding=1)
    |> ReLU() |> MaxPool2d(kernel=2, stride=2); 
}

morph second_conv(x : tsr[f32; 16, 16, 16]) : tsr[f32; 8, 32, 32] {
    x |> Conv2d(in=16, out=32, kernel=3, padding=1)
    |> ReLU() |> MaxPool2d(kernel=2, stride=2);
}

// output 10 dimensional logit vector
morph classify(x : tsr[f32; 8, 32, 32]) : tsr[f32; 10] {
    x |> Reshape(-1, 2048) |> Linear(in=2048, out=128)
    |> ReLU() |> Linear(in=128, out=10);
}
```
Another option
```stn
morph classify tsr[32; 8, 32, 32] |> tsr[f32; 10] {
    Reshape(-1, 2048) |> Linear(in=2048, out=128) |> ReLU() |> Linear(in=128, out=10);
}

// calling the morph
x : tsr[f32; 8, 32, 32] = rand(tsr[f32; 8, 32, 32]);
y = x |> classify;
```

Define an edge template
```stn
edge conve(x : tsr[f32; Cin, H, W]) : tsr[f32; Cout, H2, W2] {
    dyn kernel: tsr[f32; Cout, Cin, K, K] = kaiming_uniform(); // initialization
    dyn bias: tsr[f32; Cout] = zeros(); // initialization
    stride: i32 = 1;
    pad: i32 = 1;
}

dynamic forward {

}


# some claude notes
```
fiber / out   →  current output value (what edges read and write to)
state         →  persistent value that survives across time steps
dyn           →  learnable intrinsic parameter (belongs to the node, not the edge)
```

## Standard Feedforward Node
```stn
node Dense(dim: u32) {
    out: tsr[f32; dim]
}

node ImageNode(C: u32, H: u32, W: u32) {
    out: tsr[f32; C, H, W]
}
```

## Recurrent Nodes (RNN / LSTM / GRU)
```stn
node LSTMCell(hidden: u32) {
    out: tsr[f32; hidden]
    state h: tsr[f32; hidden] = zeros()
    state c: tsr[f32; hidden] = zeros()
}

node GRUCell(hidden: u32) {
    out: tsr[f32; hidden]
    state h: tsr[f32; hidden] = zeros()
}
```

## Spiking Neurons
```stn
// Leaky Integrate-and-Fire
node LIF {
    out: f32                         // spike output (0 or 1)
    state v_m:   f32 = 0.0          // membrane potential
    state t_ref: u32 = 0            // refractory countdown

    dyn threshold: f32 = 1.0        // learnable firing threshold
    dyn tau:       f32 = 20.0       // membrane time constant

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

// Adaptive LIF — threshold itself adapts with spike history
node AdaptiveLIF {
    out: f32
    state v_m:    f32 = 0.0
    state theta:  f32 = 1.0         // adaptive threshold (state, not fixed)

    dyn tau_m:    f32 = 20.0
    dyn tau_th:   f32 = 100.0
    dyn delta_th: f32 = 0.1         // threshold increment per spike

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

## Stochastic / Probabilistic Nodes (VAE, Bayesian)
node Stochastic(dim: u32) {
    out:     tsr[f32; dim]       // sampled latent (the fiber edges consume)
    mu:      tsr[f32; dim]       // mean — written to by an incoming edge
    log_var: tsr[f32; dim]       // log variance — written to by an incoming edge

    dynamic sample() {
        eps = randn_like(mu)
        out = mu + exp(0.5 * log_var) * eps
    }
}

// Bayesian node — weight uncertainty lives in the node
node BayesianDense(dim: u32) {
    out: tsr[f32; dim]
    dyn mu_w:    tsr[f32; dim] = zeros()
    dyn log_sig: tsr[f32; dim] = zeros()

    dynamic forward(input: tsr[f32; dim]) {
        w = mu_w + exp(log_sig) * randn_like(mu_w)
        out = w * input
    }
}
```

## Oscillatory / Phase Nodes
```
node Oscillator {
    out: c32                     // complex amplitude — encodes both phase and magnitude
    state phase: f32 = 0.0

    dyn freq:    f32 = 1.0       // natural frequency (learnable)
    dyn damping: f32 = 0.01

    dynamic step(input: c32, dt: f32) {
        phase = phase + 2*pi*freq*dt
        out   = (1.0 - damping) * out + input
        out   = out * exp(i * phase) / out.norm()
    }
}
```

## Memory-Augmented Nodes
```
node ExternalMemory(mem_size: u32, head_dim: u32) {
    out:    tsr[f32; head_dim]                        // current read vector
    state M: tsr[f32; mem_size, head_dim] = zeros()  // memory matrix
    state w: tsr[f32; mem_size] = softmax(ones())    // attention weights

    // Write and read heads are edges — they update M and w via their forward()
}
```


# Declaring Functions
```
// Activation functions
fn relu(x: tsr[f32; ..]) -> tsr[f32; ..] {
    max(x, 0.0)
}

fn gelu(x: tsr[f32; ..]) -> tsr[f32; ..] {
    0.5 * x * (1.0 + tanh(sqrt(2.0/pi) * (x + 0.044715 * x**3)))
}

fn softmax(x: tsr[f32; n]) -> tsr[f32; n] {
    e = exp(x - max(x))     // numerically stable
    e / sum(e)
}

// Normalization
fn layer_norm(x: tsr[f32; d], eps: f32 = 1e-5) -> tsr[f32; d] {
    mu  = mean(x)
    sig = std(x)
    (x - mu) / sqrt(sig**2 + eps)
}

fn batch_norm(x: tsr[f32; B, d], eps: f32 = 1e-5) -> tsr[f32; B, d] {
    mu  = mean(x, dim=0)
    sig = std(x, dim=0)
    (x - mu) / sqrt(sig**2 + eps)
}
```

# Primitives
* Arithmetic:   + - * /  **  @  (matmul)
* Indexing:     x[i]  x[a..b]  x[:, i]
* Shape:        reshape  transpose (.T)  squeeze  flatten  concat  split
* Reduction:    sum  mean  max  min  prod  (all take optional dim=)
* Elementwise:  exp  log  sqrt  abs  sin  cos  tanh  clip
* Comparison:   >  <  ==  where(cond, a, b)
* Random:       randn_like  rand_like  randn(shape)
* Init:         zeros  ones  eye  kaiming_uniform  xavier


// activations
relu  leaky_relu(alpha)  elu(alpha)  selu  sigmoid  tanh
softmax  log_softmax  gelu  swish  mish  hardswitch

// linear algebra
matmul (@)  outer  dot  norm  normalize  cross  det  inv  svd  eig

// tensor ops
reshape  flatten  squeeze  unsqueeze  permute  transpose (.T)
concat  stack  split  chunk  pad  roll  gather  scatter

// reductions
sum  mean  max  min  prod  std  var  (all take dim= and keepdim=)
argmax  argmin  topk  sort

// convolutions
conv1d  conv2d  conv3d
max_pool1d  max_pool2d  avg_pool1d  avg_pool2d
upsample  interpolate

// normalization
batch_norm  layer_norm  group_norm  instance_norm  rms_norm

// regularization
dropout(p)  droppath(p)  alpha_dropout(p)

// initializers (used in dyn declarations)
zeros  ones  eye  full
rand  randn  kaiming_uniform  kaiming_normal  xavier_uniform  xavier_normal
trunc_normal(mean, std)

// loss functions
mse  mae  cross_entropy  binary_cross_entropy  kl_div  cosine_similarity


# Algebras
```stn
// The free algebra over f32 with one generator j, quotiented by j*j = 1
algebra SplitComplex over f32 {
    basis { 1, j }
    relations { j*j = 1.0 }
}

algebra Complex over f32 {
    basis { 1, i }
    relations { i*i = -1.0 }
}

// Quaternions fall out naturally — non-commutativity is captured by ordered relations
algebra Quaternion over f32 {
    basis { 1, i, j, k }
    relations {
        i*i = -1.0,  j*j = -1.0,  k*k = -1.0
        i*j =  k,    j*i = -k
        j*k =  i,    k*j = -i
        k*i =  j,    i*k = -j
    }
}

// Dual numbers (for forward-mode autodiff)
algebra Dual over f32 {
    basis { 1, eps }
    relations { eps*eps = 0.0 }
}

// Exterior algebra (for differential forms, physics-informed nets)
algebra Exterior3 over f32 {
    basis { 1, e1, e2, e3, e12, e13, e23, e123 }
    relations {
        e1*e1 = 0.0,  e2*e2 = 0.0,  e3*e3 = 0.0
        e1*e2 = e12,  e2*e1 = -e12
        e1*e3 = e13,  e3*e1 = -e13
        e2*e3 = e23,  e3*e2 = -e23
    }
}
```

```
*     ring product, lifted elementwise (Hadamard for tensors)
@     matrix multiplication / linear map composition
**    exponentiation (elementwise)
·     tensor contraction along arbitrary axes (optional, for physics/einsum style)
+     additive group operation (elementwise)
```


# Complicated Topologies
```
subgraph GridModule(n: u32) {
    x = node [n] : Activity         // n neurons in a ring
    state phase: u32 = 0
    dyn delta: i32 = 1              // displacement sensitivity

    x -> x : cyclic(), fixed        // hardwired cyclic, not learned

    in  drive: f32                  // receives displacement signal
    out activation: [Activity; n]   // exposes current activations

    dynamic step {
        phase = (phase + delta * drive) % n
        for i in 0..n {
            x[i] = if i == phase { alpha_max } else { x[i] * (1 - lambda) }
        }
    }
}

// Nesting: Where neurons contain two named sub-subgraphs
subgraph Where(L: u32, periods: [u32; L], N_ctx: u32) {

    // W_T: array of L grid modules with coprime periods
    W_T = GridModule(periods[0..L])     // [GridModule; L]

    // W_M: learned context component
    subgraph W_M {
        x = node [N_ctx] : Activity
        x <-> x : dyn                   // learned recurrent
    }

    in  F_w                             // feedforward from region boundary
    in  displacement: f32               // external displacement input
    out state                           // exposes full W state to M

    F_w         -> W_T[*].drive         // fan displacement to all modules
    F_w         -> W_M.x : dyn
    displacement -> W_T[*].drive
}
```

```
graph CorticalNet {

    L0 = CorticalRegion(N_M=1000, N_W=64,  L=3, periods=[5,7,11])[0..4]
    L1 = CorticalRegion(N_M=2000, N_W=128, L=3, periods=[7,11,13])[0..2]
    L2 = CorticalRegion(N_M=4000, N_W=256, L=3, periods=[11,13,17])

    // What I was calling "feedforward" — just named directed edges between ports
    ff_0_to_1 = (L0[0].F_z, L0[1].F_z) -> L1[0].F_w : dyn W_ff01
    ff_1_to_2 = (L1[0].F_z, L1[1].F_z) -> L2.F_w    : dyn W_ff12

    // What I was calling "lateral" — edges between regions at the same level
    lat_L1 = L1[0].F_z -> L1[1].F_w : dyn W_lat
    lat_L1 = L1[1].F_z -> L1[0].F_w : dyn W_lat      // reverse direction, same name

    // What I was calling "feedback" — edges going the other direction
    fb_2_to_1 = L2.F_z    -> L1[*].feedback : dyn W_fb21
    fb_1_to_0 = L1[*].F_z -> L0[*].feedback : dyn W_fb10
}
```

# Final notes
I need to pick distinction for function return value and topology definition

* **Morphism:** `|>`
* **Directed Topology:** `~>`
* **Symmetric Topology:** `~`
* **Functional Return:** `->`
* **Type Annotation / Property:** `:`
* **Group Operation or Ring Product (Hadamard):** `*`
* **Matrix Multiply:** `@`
* **Exponentiation:** `**`
* **Modulus:** `%`


# Declaring topologies using edges

```
morph ctrans(x : tsr[f32]) {

}

edge conve(x : tsr[f32; Cin, H, W]) : tsr[f32; Cout, H2, W2] {
    dyn kernel: tsr[f32; Cout, Cin, K, K] = kaiming_uniform(); // initialization
    dyn bias: tsr[f32; Cout] = zeros(); // initialization
    stride: i32 = 1;
    pad: i32 = 1;

    // learning rule
    dynamic forward(){

    }
}

subgraph mem(){
    subgraph p(n: u32) {
        // node defined previously
        ExternalMemory(8, 16)[n]; // n externalmemory neurons
    }

    subgraph q(n: u32) {
        // imported or defined previously
        Stochastic(5)[n]; // 50 stochastic neurons 
    }



    
}