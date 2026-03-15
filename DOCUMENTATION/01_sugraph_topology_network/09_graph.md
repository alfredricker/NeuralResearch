# Graph Declaration

The `graph` block is the top-level entry point of an STN program. There is only one logical graph — multiple `graph` blocks in the same program all contribute to the same structure.

## Minimal Graph

```stn
graph {
    x = Nodes(50)
}
```

This creates 50 disconnected nodes with no data type.

## Typed Nodes

```stn
graph {
    input  = Nodes(784) : f32
    hidden = Nodes(128) : f32
    output = Nodes(10)  : f32
}
```

## Adding Topology

```stn
graph {
    input  = Nodes(784) : f32
    hidden = Nodes(128) : f32
    output = Nodes(10)  : f32

    input  ~> hidden : All, dyn
    hidden ~> output : All, dyn
}
```

## Using Subgraphs

Graphs are typically built from subgraph instances:

```stn
graph {
    enc = Encoder(784, 64)
    dec = Decoder(64, 784)

    enc.out ~> dec.in : Identity
}
```

## Arrays and Loops

Create multiple layers and connect them with `index`:

```stn
graph {
    layers = DenseLayer(128, 128)[0..9]   // 10 layers

    index(i, 0..8) {
        layers[i].y ~> layers[i+1].x : Identity
    }
}
```

## Inputs and Outputs

Declare named input and output ports on the graph using `in` and `out`:

```stn
graph {
    in  image: tsr[f32; 1, 28, 28]
    out logits: tsr[f32; 10]

    enc = Encoder(784, 128)
    cls = Classifier(128, 10)

    image   ~> enc.in  : first_conv
    enc.out ~> cls.in  : Identity
    cls.out ~> logits  : Identity
}
```

## Multiple `graph` Blocks

Large networks can be split across multiple `graph` blocks. All blocks contribute to the same graph:

```stn
// Define the backbone
graph {
    backbone = ResNetBackbone(layers=18)
}

// Attach a classification head separately
graph {
    head = ClassifierHead(512, 10)
    backbone.out ~> head.in : Identity
}
```

## Named Graph

A graph can be given a name, which is used when compiling to a named output:

```stn
graph MinimalClassifier {
    in  x: tsr[f32; 784]
    out y: tsr[f32; 10]

    hidden = Nodes(128) : tsr[f32; 128]

    x      ~> hidden : All, dyn
    hidden ~> y      : All, dyn
}
```
