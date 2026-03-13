# The STN Compiler

The STN compiler is a Rust pipeline that turns `.stn` source into runtime-ready graph structures.

## Current Pipeline
1. **Lexing**: Convert raw source text into a stream of typed tokens with source spans.
2. **Parsing**: Convert tokens into AST nodes (`Program`, `Item`, `Statement`, `Expr`).
3. **AST**: Preserve the structural meaning of STN source in a syntax-oriented tree.
4. **Declarative IR**: Resolve named groups/interfaces into compiler IDs and validate graph links.
5. **Executable IR**: Materialize node/edge/storage structures for runtime execution.
6. **Codegen**: Emit a runnable Rust project from executable IR.

## Why these stages exist
- **Early stages** keep source-level intent and good diagnostics.
- **Middle stages** normalize semantics and remove syntax ambiguity.
- **Late stages** optimize for execution and backend generation.

## Current status
- Lexer, parser, AST, IR lowering, and basic codegen are implemented.
- The minimal program can be compiled to a generated Rust crate and executed.

## Generate from `.stn` (quick start)
From the `compiler/` directory:

```bash
cargo run -- ../stn/1-minimal-graph.stn ../generated-rust
```

This writes a runnable Rust project to `../generated-rust`. To run it:

```bash
cd ../generated-rust
cargo run
```