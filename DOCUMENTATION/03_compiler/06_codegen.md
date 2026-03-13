# Codegen

## Purpose
The codegen stage converts executable IR into a standalone Rust project that can be built and run with Cargo.

## Input
- `ExecutableModule` produced by IR lowering.

## Output
- Generated Rust project directory containing:
  - `Cargo.toml`
  - `src/main.rs`

## Current behavior
- Uses the first executable graph in the module.
- Emits:
  - node count constant
  - edge list constant
  - basic `tick()` loop (message pass + node update)
  - runnable `main()` entry point

## How to generate from `.stn`
From the `compiler/` directory:

```bash
cargo run -- ../stn/1-minimal-graph.stn ../generated-rust
```

This compiles the `.stn` program, lowers through IR, and writes a Rust project to `../generated-rust`.

Then build/run generated code:

```bash
cd ../generated-rust
cargo run
```
