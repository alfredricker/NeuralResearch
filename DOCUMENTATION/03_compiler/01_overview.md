# The STN Compiler

The STN compiler is a Rust pipeline that turns `.stn` source into runtime-ready graph structures.

## Current Pipeline
1. **Lexing**: Convert raw source text into a stream of typed tokens with source spans.
2. **Parsing**: Convert tokens into AST nodes (`Program`, `Item`, `Statement`, `Expr`).
3. **AST**: Preserve the structural meaning of STN source in a syntax-oriented tree.
4. **Declarative IR**: Resolve named groups/interfaces into compiler IDs and validate graph links.
5. **Executable IR**: Materialize node/edge/storage structures for runtime execution.
6. **Codegen**: Planned stage that will emit Rust/runtime artifacts from executable IR.

## Why these stages exist
- **Early stages** keep source-level intent and good diagnostics.
- **Middle stages** normalize semantics and remove syntax ambiguity.
- **Late stages** optimize for execution and backend generation.

## Current status
- Lexer, parser, AST, and IR lowering are implemented and can compile `stn/1-minimal-graph.stn`.
- Code generation is scaffolded but not implemented yet.