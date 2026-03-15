# Algebras

STN lets you define custom algebraic structures as first-class types. An `algebra` block specifies a basis and the multiplication relations between basis elements. The compiler uses these relations to generate correct arithmetic for the type.

## Syntax

```stn
algebra <Name> over <scalar_type> {
    basis { <element>, ... }
    relations { <product> = <value>, ... }
}
```

`over` specifies the scalar field (typically `f32` or `f64`). The first basis element is always the scalar unit `1`.

## Built-in Algebra Types

The standard alternate number types (`c32`, `sc32`, `q32`, etc.) are defined using `algebra` internally. The examples below show how they are derived.

### Complex Numbers

```stn
algebra Complex over f32 {
    basis { 1, i }
    relations { i*i = -1.0 }
}
```

### Split-Complex Numbers

```stn
algebra SplitComplex over f32 {
    basis { 1, j }
    relations { j*j = 1.0 }
}
```

The `j` element squares to `+1` rather than `-1`, giving a hyperbolic structure instead of circular.

### Quaternions

```stn
algebra Quaternion over f32 {
    basis { 1, i, j, k }
    relations {
        i*i = -1.0,  j*j = -1.0,  k*k = -1.0
        i*j =  k,    j*i = -k
        j*k =  i,    k*j = -i
        k*i =  j,    i*k = -j
    }
}
```

Non-commutativity is captured by specifying both `i*j` and `j*i` explicitly.

### Dual Numbers

Useful for forward-mode automatic differentiation:

```stn
algebra Dual over f32 {
    basis { 1, eps }
    relations { eps*eps = 0.0 }
}
```

A dual number `a + b*eps` carries a value `a` and its derivative `b`. Setting `eps*eps = 0` ensures higher-order terms vanish automatically.

### Exterior Algebra

Useful for differential forms and physics-informed networks:

```stn
algebra Exterior3 over f32 {
    basis { 1, e1, e2, e3, e12, e13, e23, e123 }
    relations {
        e1*e1 = 0.0,  e2*e2 = 0.0,  e3*e3 = 0.0
        e1*e2 =  e12, e2*e1 = -e12
        e1*e3 =  e13, e3*e1 = -e13
        e2*e3 =  e23, e3*e2 = -e23
    }
}
```

## Using Custom Algebras

Once defined, an algebra type can be used anywhere a primitive type is valid:

```stn
node PhaseNode {
    out: Complex
    state z: Complex = 1.0 + 0.0*i
}

x = Nodes(64) : Quaternion
```

Tensors can also be parameterized by algebra types:

```stn
tsr[Complex; 128]     // 128-dimensional complex vector
tsr[Quaternion; 3, 3] // 3x3 matrix of quaternions
```

## Operator Semantics on Algebra Types

The standard operators lift naturally:

| Operator | Behavior on algebra type |
|----------|--------------------------|
| `+` | Componentwise addition |
| `*` | Algebra multiplication (uses `relations`) |
| `@` | Matrix multiplication using algebra `*` as the inner product |
| `**` | Repeated algebra multiplication |
