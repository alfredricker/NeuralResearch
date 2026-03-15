# Functions and Morphs

STN has two mechanisms for defining computation: `fn` for pure mathematical functions and `morph` for edge transformations that may carry learnable parameters.

## `fn` — Pure Functions

`fn` declares a stateless, parameterless function. These are used for activations, normalization, and other element-level operations with no learned weights.

```stn
fn relu(x: tsr[f32; ..]) -> tsr[f32; ..] {
    Max(x, 0.0)
}

fn gelu(x: tsr[f32; ..]) -> tsr[f32; ..] {
    0.5 * x * (1.0 + Tanh(Sqrt(2.0/pi) * (x + 0.044715 * x**3)))
}

fn softmax(x: tsr[f32; n]) -> tsr[f32; n] {
    e = Exp(x - Max(x))    // numerically stable
    e / Sum(e)
}

fn layer_norm(x: tsr[f32; d], eps: f32 = 1e-5) -> tsr[f32; d] {
    mu  = Mean(x)
    sig = Std(x)
    (x - mu) / Sqrt(sig**2 + eps)
}
```

## `morph` — Edge Transformations

A `morph` defines a transformation applied along a set of edges. Unlike `fn`, a morph can declare learnable parameters with `dyn` and carry state.

```stn
morph first_conv(x: tsr[f32; 3, 32, 32]) -> tsr[f32; 16, 16, 16] {
    x |> Conv2d(in=3, out=16, kernel=3, padding=1)
      |> ReLU()
      |> MaxPool2d(kernel=2, stride=2)
}

morph classify(x: tsr[f32; 32, 8, 8]) -> tsr[f32; 10] {
    x |> Flatten()
      |> Linear(in=2048, out=128)
      |> ReLU()
      |> Linear(in=128, out=10)
}
```

### Morphs with learnable parameters

Use `dyn` inside a morph to declare weights that belong to the morph itself:

```stn
morph conv_edge(x: tsr[f32; Cin, H, W]) -> tsr[f32; Cout, H2, W2] {
    dyn kernel: tsr[f32; Cout, Cin, K, K] = KaimingUniform()
    dyn bias:   tsr[f32; Cout]            = Zeros()
    stride: i32 = 1
    pad:    i32 = 1

    x |> Conv2d(kernel, bias, stride, pad)
}
```

## `|>` — Morphism Pipeline

The `|>` operator threads a value through a sequence of transformations. Each step receives the output of the previous.

```stn
y = x |> Flatten() |> Linear(784, 256) |> ReLU() |> Linear(256, 10)
```

Multi-line pipelines are idiomatic for readability:

```stn
y = x
    |> Conv2d(in=1, out=32, kernel=3, padding=1)
    |> BatchNorm()
    |> ReLU()
    |> MaxPool2d(kernel=2, stride=2)
    |> Flatten()
    |> Linear(in=1568, out=10)
```

A morph can be called inline in a pipeline:

```stn
y = image |> first_conv |> second_conv |> classify
```

## Applying a Morph to an Edge

Morphs are attached to a topology using `:`:

```stn
x ~> y : first_conv
```

## Built-in Transformations

### Activations
```
ReLU()  LeakyReLU(alpha)  ELU(alpha)  SELU()
Sigmoid()  Tanh()  GELU()  Swish()  Mish()
Softmax()  LogSoftmax()
```

### Linear
```
Linear(in, out)
```

### Convolutions
```
Conv1d(in, out, kernel, stride=1, padding=0)
Conv2d(in, out, kernel, stride=1, padding=0)
Conv3d(in, out, kernel, stride=1, padding=0)
```

### Pooling
```
MaxPool1d(kernel, stride)  MaxPool2d(kernel, stride)
AvgPool1d(kernel, stride)  AvgPool2d(kernel, stride)
```

### Shape
```
Flatten()  Reshape(...)  Squeeze()  Unsqueeze(dim)
Permute(...)  Transpose()
```

### Normalization
```
BatchNorm()  LayerNorm()  GroupNorm(groups)  RMSNorm()
```

### Regularization
```
Dropout(p)  DropPath(p)
```

### Upsampling
```
Upsample(scale)  Interpolate(size, mode)
```
