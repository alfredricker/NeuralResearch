# Data Types

Every node in a Quiver graph holds a typed value. Types are assigned to nodes and edges using the `:` operator.

## Numeric Primitives

### Floats
| Type | Description |
|------|-------------|
| `f16` | 16-bit float |
| `f32` | 32-bit float (most common) |
| `f64` | 64-bit float |

### Signed Integers
| Type | Description |
|------|-------------|
| `i8` | 8-bit signed integer |
| `i16` | 16-bit signed integer |
| `i32` | 32-bit signed integer |
| `i64` | 64-bit signed integer |

### Unsigned Integers
| Type | Description |
|------|-------------|
| `u8` | 8-bit unsigned integer |
| `u16` | 16-bit unsigned integer |
| `u32` | 32-bit unsigned integer |
| `u64` | 64-bit unsigned integer |

## Alternate Algebras

These types carry the same bit widths as floats but with different algebraic structure. See [Algebras](10_algebras.md) for defining custom algebraic types.

### Complex Numbers
`c16`, `c32`, `c64` — elements of the form `a + bi` where `i*i = -1`.

### Split-Complex Numbers
`sc16`, `sc32`, `sc64` — elements of the form `a + bj` where `j*j = 1`. Useful for hyperbolic geometry and certain physics-informed networks.

### Quaternions
`q16`, `q32` — four-component numbers `a + bi + cj + dk`. Useful for 3D rotation and orientation-sensitive networks.

## Tensor Type

Tensors are the primary data container for deep learning. They are declared with the `tsr` keyword, element type, and a semicolon-separated shape.

```quiver
tsr[f32; 3]           // 1D tensor with 3 elements (a vector)
tsr[f32; 4, 4]        // 2D tensor, 4x4 matrix
tsr[f32; 1, 28, 28]   // 3D tensor, e.g. a grayscale image
tsr[f32; 16, 3, 3, 3] // 4D tensor, e.g. a conv kernel bank
tsr[c32; 64]          // vector of 64 complex numbers
```

The shape dimensions can be symbolic when used inside parameterized node or subgraph definitions:

```quiver
node ImageNode(C: u32, H: u32, W: u32) {
    out: tsr[f32; C, H, W]
}
```

### Ellipsis for Rank-Polymorphic Functions

In function signatures, `..` in a shape denotes an arbitrary number of dimensions:

```quiver
fn relu(x: tsr[f32; ..]) -> tsr[f32; ..] {
    Max(x, 0.0)
}
```
