# Subgraph Topology Network

## Overview

A network is a hierarchical directed graph. The language provides syntax for:
- Declaring subgraph templates
- Instantiating and nesting subgraphs
- Defining connection topologies at any level of hierarchy
- Specifying transforms (how nodes process inputs)
- Specifying learning rules (how weights change)
- Attaching external inputs and outputs
- Describing metrics to output to the terminal or visualize
- Declaring the overall network
- Parallelism methodologies

The core principle: **topology, transform, and learning declarations mirror subgraph structure recursively**.

The language has flexibility to define a diverse class of networks with syntactical efficiency: 
- Graph NNs
- Convolutional NNs
- Cortical Column NNs (Thousand Brains Theory)
- Transformer based
- Hopfield
- Hebbian
And more.

## Graphs

A graph is a collection of nodes and edges. This is the fundamental compiled unit - everything else in the language exists to help you build graphs concisely.

### Declaring nodes
```
graph {
    nodes(50);
}
```

This creates 50 nodes. They have no connections yet.

### Declaring edges

Edges connect nodes. The arrow `->` creates directed connections from left to right.
```
graph {
    nodes(50);
    nodes -> nodes: all;
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

### Named node groups

When you have multiple groups of nodes, give them names:
```
graph {
    input: nodes(784);
    hidden: nodes(500);
    output: nodes(10);
    
    input -> hidden: sparse(0.4);
    hidden -> output: sparse(0.4);
}
```

This creates a simple feedforward network: 784 input nodes, 500 hidden nodes, 10 output nodes, with sparse connections between layers.

### Self-connections

A group can connect to itself:
```
graph {
    hidden: nodes(500);
    
    hidden -> hidden: sparse(0.02);
}
```

This creates recurrent connections. To include self-loops (node connecting to itself):
```
hidden -> hidden: sparse(0.02) & !!identity;
```

The `&` combines patterns (intersection), and `!` inverts a pattern. Since by default,
```
sparse(0.2) == sparse(0.2) & !identity;
```
we must use the double inversion `!!identity`.

---

## Input and Output

A graph needs to receive data from the outside world and produce results.

### Declaring input
```
input Image(28, 28);
```

This declares an input interface expecting a 28x28 image (784 values).

### Declaring output
```
output numbers: Class(10);
```

This declares an output interface producing a classification over 10 classes.

### Connecting input to the graph
```
input MNIST: Image(28, 28);

graph {
    input_nodes: nodes(784);
    hidden: nodes(500);
    output_nodes: nodes(10);
    
    input_nodes -> hidden: sparse(0.4);
    hidden -> output_nodes: sparse(0.4);
}

MNIST -> input_nodes: identity;
```

The final line maps the 784 image pixels directly to the 784 input nodes.

### Connecting the graph to output
```
output_nodes -> numbers: weighted_sum;
```

This reads out activations from `output_nodes`, applies weighted sum, and produces class scores.

---

## Complete Example: MNIST Classifier
```
input MNIST: Image(28, 28);
output numbers: Class(10);

graph {
    omega: nodes(784);
    m: nodes(500);
    z: nodes(10);
    
    omega -> m: sparse(0.4);
    m -> z: sparse(0.4);
}

MNIST -> omega: identity;
z -> numbers: weighted_sum;
```

This is a minimal network for MNIST classification. 784 input nodes receive pixel values, 500 hidden nodes process them, 10 output nodes produce class scores.

No transforms are specified, so nodes use the default (identity). No learning rules are specified, so weights are fixed. Later sections will add these.