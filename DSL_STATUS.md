# DSL Status and Minimal Program Compile Plan

This document captures the current state of the STN DSL compiler and the next steps to get `stn/1-minimal-graph.stn` compiling end-to-end.

## Target Program

`stn/1-minimal-graph.stn`

```stn
input MNIST: Image(28, 28);
output numbers: Class(10);

graph {
    omega: nodes(784);
    m: nodes(500);
    z: nodes(10);
    
    omega -> m: sparse(0.4);
    m -> z: sparse(0.4);
}

MNIST -> omega: identity;
z -> numbers: weighted_sum;
```

## Current Implementation Status

### What is already in place

- Lexer exists and returns `Result<Vec<SpannedToken>, LexError>`.
- Parser skeleton exists with:
  - top-level program parsing (`parse_program`)
  - block parsing (`graph`, `subgraph`, `topology`, `learn`, `display`)
  - statement parsing for `input`, `output`, and `ident -> ident : ...` links
  - shared argument parsing utilities (`parse_args`, typed arg decoding)
- AST currently models:
  - program container
  - block items
  - input/output declarations
  - link declarations with basic topology enum (`Sparse`, `Dense`, `Identity`)

### What currently fails at compile time (Rust)

From current diagnostics in `compiler/src/main.rs`:

1. `Parser::new` expects `Vec<SpannedToken>`, but `lex(&source)` returns `Result<Vec<SpannedToken>, LexError>`.
2. Parse error printing uses `{}` with `ParseError`, but `ParseError` does not implement `Display`.

These two issues prevent the compiler binary from building cleanly before DSL parsing can be fully tested.

### What currently fails at DSL parse level for the minimal program

Even after fixing Rust compile errors, `1-minimal-graph.stn` will still fail to parse because:

1. **Node declarations inside `graph` are not implemented**
   - Example syntax: `omega: nodes(784);`
   - Current `parse_item` only accepts identifier-led statements in arrow form (`ident -> ...`), not `ident : ...`.

2. **Topology/function token case mismatch**
   - Minimal program uses lowercase: `sparse`, `identity`.
   - Lexer currently defines uppercase tokens: `Sparse`, `Identity`, `Dense`.
   - Lowercase forms become plain identifiers today.

3. **`weighted_sum` not handled in link topology parser**
   - Lexer has `WeightedSum`, but `parse_topology` currently only accepts `Identity`, `Dense`, and `Sparse(...)`.
   - Link AST topology enum currently also omits `WeightedSum`.

## Immediate Next Steps (in order)

### 1) Unblock Rust build in `main`

- Handle lexer result explicitly before constructing parser.
- Update parse error formatting to `{:?}` or implement `Display` for `ParseError`.

Suggested minimal direction:
- `let tokens = lex(&source).unwrap_or_else(...);`
- keep panic formatting consistent with available traits.

### 2) Add parser support for graph node declaration statements

- Introduce AST type for node declaration (for example: `NodeDecl { name, count }`).
- Extend `Statement` enum with `Node(...)` (or equivalent).
- Add parser branch for identifier-led colon form:
  - parse `name`
  - expect `:`
  - expect `nodes`
  - parse single integer arg in `nodes(...)`
  - expect `;`

### 3) Normalize topology/function token matching

Pick one policy and apply consistently:

- **Option A (recommended now):** accept lowercase DSL spellings used in examples
  - `sparse`, `identity`, `dense`, `weighted_sum`
- **Option B:** require uppercase keywords and update all `.stn` examples

For minimal friction, Option A aligns with `1-minimal-graph.stn`.

### 4) Extend link topology support to include `weighted_sum`

- Add `WeightedSum` to AST link topology enum.
- Add parser branch in `parse_topology` for `WeightedSum`.
- Keep topology declaration grammar consistent between graph-internal and top-level links.

### 5) Add a focused parse test for this exact file

- Add a test that lexes + parses `stn/1-minimal-graph.stn`.
- Assert parse success and basic AST shape:
  - 2 top-level IO declarations
  - 1 graph block
  - 3 node declarations in graph
  - 2 internal links
  - 2 top-level links

## “Definition of Done” for this milestone

`stn/1-minimal-graph.stn` is considered compiling when:

1. `cargo check` passes for the compiler crate.
2. Running the compiler on `stn/1-minimal-graph.stn` exits successfully.
3. Parser returns an AST that includes node declarations + links + IO declarations with no fallback/placeholder errors.
4. A regression test locks the behavior.

## Practical Notes

- Current parser shape is a good base; this is an incremental completion task, not a rewrite.
- Start by fixing Rust compile errors first, then parse coverage.
- Keep keyword casing strategy explicit early to avoid churn across examples and tests.
