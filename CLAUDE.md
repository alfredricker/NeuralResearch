# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NeuralResearch is a **DSL compiler** for specifying neural network architectures. The language is called **STN (Subgraph Topology Network)**. STN source files (`.stn`) are compiled into executable Rust projects.

## Build & Run Commands

```bash
# Compile an STN file into a Rust project
cd compiler
cargo run -- ../stn/1-minimal-graph.stn ../generated-rust

# Build and run the generated project
cd ../generated-rust
cargo run

# Run compiler tests
cd compiler
cargo test

# Run a single test
cd compiler
cargo test <test_name>

# Check for compile errors without building
cd compiler
cargo check
```

## Compiler Pipeline

The pipeline is: **Lex → Parse → AST → Declarative IR → Executable IR → Codegen**

Each stage is a distinct transformation; `compiler/src/main.rs` orchestrates them in sequence.

```
compiler/src/
├── main.rs          # CLI entry point — orchestrates the full pipeline
├── lib.rs           # Library exports
├── lexer/           # Tokenizes .stn source using `logos` crate
├── parser/          # Builds AST from token stream
├── ast/             # 14-file AST module (block, expr, io, link, statement, var, etc.)
├── ir/
│   ├── declarative/ # Resolves names → stable IDs, validates graph topology
│   └── executable/  # Materializes nodes/edges, assigns storage slots, expands groups
└── codegen.rs       # Emits a Rust project with CSR binary graph format
```

**Two-IR Design:** The declarative IR preserves source-level intent and validates correctness. The executable IR materializes the concrete data structures needed for codegen (node slots, expanded groups, edge lists in CSR layout).

**Generated output:** `codegen.rs` emits a self-contained Rust project with a lean runtime (~94 lines) using an active node set for efficient graph traversal. The graph is stored in Compressed Sparse Row (CSR) binary format.

## STN Language Basics

```stn
graph MinimalNet {
    input: Nodes(784)
    output: Nodes(10)

    hidden = Nodes(128)
    input -> hidden : sparse(0.2)
    hidden -> output : identity
}
```

Key syntax:
- `->` directed edge / topology
- `~` symmetric (undirected) topology
- `|>` morphism/transformation pipeline
- `@` matrix multiplication, `*` Hadamard product
- `:` type annotation or property assignment (e.g., `sparse(0.2)`, `f32`)

## Documentation

- `DOCUMENTATION/03_compiler/` — per-stage compiler docs (lexer, parser, AST, IR, codegen)
- `DOCUMENTATION/01_sugraph_topology_network/01_basics.md` — STN language fundamentals
- `DOCUMENTATION/00_motivation/` — project goals and philosophy
- `jumbled-notes.md` — active research scratchpad; contains in-progress language design ideas
- `Cortical.tex` — LaTeX paper draft

## Workspace Layout

There are two Cargo crates:
- `compiler/` — the STN compiler (Rust, edition 2024, depends on `logos`)
- `generated-rust/` — output directory for compiled STN programs (overwritten on each compile run)

The `stn/` directory holds example `.stn` source files for testing.
