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
* tsr[f32; 3] // a tensor that contains 32 bit floats and 3 dimensions. This is the same as a 3 dimensional vector.
* tsr[sc16; 4, 4, 4, 4] // this is a 4x4x4x4 = 64 dimensional tensor where each entry is a split complex number.



# Variables and properties
Variables are commonplace in programming languages because of their adaptability and utility. Variables exist within the STN framework to apply multiple properties to a given objects, as well as to define objects that are to be reused.




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
```stn
subgraph mono(n: i32) {
    x = Nodes(n) : f32;

}
graph {

}

# Inputs

Inputs should be interpreted in a node, edge framework. A certain subset of the input domain maps to a defined set of n points in the network. The job for the coder is to define the domain space of the inputs, and how it maps to the network.

```stn
image = Input: Tsr[f32; 8, 8, 8]


```