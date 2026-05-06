# GPU Programming
To maximize computational power of the simulation, I must use the GPU to its full potential.

## SoA vs AoS
I need to abandon the hierarchical struct ownership format and instead have a flat structure of arrays
```rust
struct Soma {
    soma_thresholds: Vec<i8>,
    soma_voltages: Vec<i8>,
    dendrite_offsets: Vec<u32> // tells you starting index of dendrites for the ith soma.
}
```

## Segmented Reduction
A common operation in parallel programming is reducing each inner array of a multidimensional array, such as rows of a matrix.


## The Event Buffer
Pass a device pointer and an atomic counter. Each thread claims a slot through atomicAdd, then writes directly. No ownership can cross thread boundaries.