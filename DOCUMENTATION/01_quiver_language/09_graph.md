# Graph Declaration

The `graph` block is the top-level entry point of a Quiver program. There is only one logical graph — multiple `graph` blocks in the same program all contribute to the same structure.

## Minimal Graph

```quiver
graph {
    x = Node[50];
}
```

This creates 50 disconnected nodes with no data type.

## Typed Nodes

```quiver
graph {
    input  = Node[784] : f32;
    hidden = Node[128] : f32;
    output = Node[10]  : f32;
}
```

## Adding Topology

```quiver
graph {
    input  = Node[784] : f32;
    hidden = Node[128] : f32;
    output = Node[10]  : f32;

    input  ~> hidden : All, dyn;
    hidden ~> output : All, dyn;
}
```

## Using Subgraphs

Graphs are typically built from subgraph instances:

```quiver
graph {
    enc = Encoder(784, 64);
    dec = Decoder(64, 784);

    enc.out ~> dec.in : Identity;
}
```

## Arrays and Loops

Create multiple layers and connect them with `index`:

```quiver
graph {
    layers = DenseLayer(128, 128)[0..9];    // 10 layers

    index(i, 0..8) {
        layers[i].y ~> layers[i+1].x : Identity;
    }
}
```

## Inputs and Outputs

Declare named input and output ports on the graph using `in` and `out`:

```quiver
graph {
    in  image:  tsr[f32; 1, 28, 28];
    out logits: tsr[f32; 10];

    enc = Encoder(784, 128);
    cls = Classifier(128, 10);

    image   ~> enc.in  via first_conv;
    enc.out ~> cls.in  : Identity;
    cls.out ~> logits  : Identity;
}
```

## Multiple `graph` Blocks

Large networks can be split across multiple `graph` blocks. All blocks contribute to the same graph:

```quiver
// Define the backbone
graph {
    backbone = ResNetBackbone(layers=18);
}

// Attach a classification head separately
graph {
    head = ClassifierHead(512, 10);
    backbone.out ~> head.in : Identity;
}
```

## Named Graph

A graph can be given a name, which is used when compiling to a named output:

```quiver
graph MinimalClassifier {
    in  x: tsr[f32; 784];
    out y: tsr[f32; 10];

    hidden = Node[128] : tsr[f32; 128];

    x      ~> hidden : All, dyn;
    hidden ~> y      : All, dyn;
}
```
