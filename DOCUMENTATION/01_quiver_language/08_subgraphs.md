# Subgraphs

A `subgraph` is a named, reusable block of nodes and edges. Subgraphs are the primary unit of composition in Quiver — complex networks are built by nesting and connecting subgraphs.

## Basic Declaration

```quiver
subgraph layer(n: u32) {
    x = Node[n] : f32;
}
```

## Parameterized Subgraphs

Subgraphs accept typed parameters, making them templates:

```quiver
subgraph DenseLayer(in_dim: u32, out_dim: u32) {
    x = Node[in_dim]  : tsr[f32; in_dim];
    y = Node[out_dim] : tsr[f32; out_dim];
    x ~> y : All, dyn;
}
```

## Named Ports

Subgraphs expose named `in` and `out` ports so that enclosing graphs can connect to them:

```quiver
subgraph GridModule(n: u32) {
    x = Node[n] : f32;
    x ~> x : Ring(1), fixed;

    in  drive:      f32;
    out activation: tsr[f32; n];

    state phase: u32 = 0;
    dyn delta:   i32 = 1;

    dynamic step {
        phase = (phase + delta * drive) % n;
        for i in 0..n {
            x[i] = if i == phase { alpha_max } else { x[i] * (1 - lambda) };
        }
    }
}
```

## Instantiating Subgraphs

Subgraphs are instantiated by calling them like a function:

```quiver
graph {
    enc = DenseLayer(784, 128);
    dec = DenseLayer(128, 784);
    enc.y ~> dec.x : Identity;
}
```

## Arrays of Subgraphs

Use bracket notation to create multiple instances:

```quiver
graph {
    layers = DenseLayer(128, 128)[0..4];    // 5 identical layers
}
```

Connect consecutive layers using an `index` loop:

```quiver
graph {
    layers = DenseLayer(128, 128)[0..4];

    index(i, 0..3) {
        layers[i].y ~> layers[i+1].x : Identity;
    }
}
```

## Nested Subgraphs

Subgraphs can contain other subgraphs:

```quiver
subgraph Where(L: u32, periods: [u32; L], N_ctx: u32) {

    W_T = GridModule(periods[0..L]);     // array of L grid modules

    subgraph W_M {
        x = Node[N_ctx] : f32;
        x ~ x : dyn;                     // learned recurrent connections
    }

    in  F_w;
    in  displacement: f32;
    out state;

    F_w          ~> W_M.x        : dyn;
    displacement ~> W_T[*].drive;
}
```

`W_T[*]` fans the connection out to all instances in the array.

## Accessing Subgraph Internals

Dot notation accesses named fields and ports:

```quiver
region.W_T[2].activation    // activation port of the 3rd GridModule inside region
layer[0].x                  // the x node set of the first layer
```
