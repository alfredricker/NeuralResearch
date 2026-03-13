# Codegen

## Purpose
The codegen stage converts executable IR into a standalone Rust project that can be built and run with Cargo.

## Input
- `ExecutableModule` produced by IR lowering.

## Output
- Generated Rust project directory containing:
  - `Cargo.toml`
  - `src/main.rs` — lean runtime (~94 lines)
  - `graph.bin` — binary graph data file

## Current behavior
- Uses the first executable graph in the module.
- Emits graph data in **Compressed Sparse Row (CSR)** binary format:
  - `graph.bin`: little-endian serialized u32 node/edge counts, offset arrays, target nodes, and edge weights
  - `src/main.rs`: runtime loader that reads the binary format and executes event-driven simulation
- Efficient execution:
  - `load_graph()` reads binary CSR format at startup (no compile-time embedding)
  - `tick()` loop with active node set for cache-friendly, sparse propagation
  - SoA (Structure of Arrays) memory layout: `activations` and `input_buffer` vectors

## How to generate and run from `.stn`

### Step 1: Generate the Rust project
From the `compiler/` directory:

```bash
cargo run -- ../stn/1-minimal-graph.stn ../generated-rust
```

This compiles the `.stn` program, lowers through IR, and writes a Rust project to `../generated-rust` containing:
- `Cargo.toml` — package manifest
- `src/main.rs` — lean runtime with binary loader
- `graph.bin` — serialized graph in CSR format

### Step 2: Build and run the generated project

```bash
cd ../generated-rust
cargo build
./target/debug/stn_generated
```

Or simply:

```bash
cd ../generated-rust
cargo run
```

Output will print: `nodes=<count> edges=<count> activation_sum=<value>`

### File sizes
The binary graph format is efficient:
- `graph.bin` stores node count, edge count, CSR offsets, target nodes, and weights
- `src/main.rs` is lean (~94 lines) with no embedded graph constants
- For large graphs (1294 nodes, 158K edges): `graph.bin` ~1.2 MB, `src/main.rs` ~4 KB
