I am adopting the convention that user defined vars or types should be lowercase while built-in stuff should be capitalized.

# Subgraphs

Imagine a CNN as a graph structure. The nodes could be the feature vectors / activation functions and the edges are the weight transformations.

```stn
subgraph layer {
    // declare the nodes and the type of data structure that it holds
    x = Nodes(10) : Tsr[f32; 1, 28, 28];
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
    x = Nodes(10): Tsr[f32; 12, 12];
    // topology definition
    x -> x : Sparse(0.2);
    // transformation definition
    x -> x : Conv2d(kernel=3, out_channels=16, stride=1, pad=1);
}
Where we use the syntax `x -> x` for any edge relevant modifiers. It is up to the compiler to determine if the statement applies to the topology (how edges are created) or to the transformations (how data is transformed among the topology).

## Graph With 10 Layers
To define a graph with 10 layers, we can write the following (assume subgraph layer is defined in the file as it was before).
```stn
graph {
    index(i,10) {
        
    }
}
```

# Inputs

Inputs should be interpreted in a node, edge framework. A certain subset of the input domain maps to a defined set of n points in the network. The job for the coder is to define the domain space of the inputs, and how it maps to the network.

```stn
image = Input: Tsr[f32; 8, 8, 8]


```