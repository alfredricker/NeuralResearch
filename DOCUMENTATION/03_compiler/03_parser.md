# Parser

## Purpose
The parser consumes the token stream and builds a structured AST representing STN syntax and nesting.

## Input
- `Vec<SpannedToken>` produced by the lexer.

## Output
- `Program` AST root containing top-level `Item`s.
- `ParseError` when token order does not match grammar expectations.

## What it builds now
- top-level declarations:
  - `input ...`
  - `output ...`
  - links like `A -> B: identity;`
- block items:
  - `graph { ... }` and other block headers
- graph body statements:
  - variable assignments like `omega = nodes(784);`
  - internal links like `omega -> m: sparse(0.4);`
- expressions:
  - numeric literals
  - identifiers
  - call expressions (`nodes(...)`)
  - topology expressions used in link parsing

## Why this stage matters
- It enforces grammar order and structure.
- It produces a stable, syntax-level tree that later passes can lower semantically.
