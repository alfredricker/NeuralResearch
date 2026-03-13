# AST (Abstract Syntax Tree)

## Purpose
The AST stores the parsed STN program in a source-oriented tree form that preserves programmer intent before semantic lowering.

## Core root shape
- `Program`
  - `Vec<Item>`
    - `Item::Statement(Statement)`
    - `Item::Block(Block)`

## Key AST entities
- `Statement`:
  - input declaration
  - output declaration
  - link declaration
  - variable declaration
- `Block`:
  - block kind (`Graph`, `Subgraph`, `Learn`, etc.)
  - nested `items`
- `Expr`:
  - literals, identifiers, calls, and topology-related expression forms

## AST role in current compiler
- The parser fills AST nodes only.
- The analyzer/IR lowering reads the AST and:
  - validates supported forms
  - resolves names
  - maps syntax into declarative IR IDs and topology expressions

## Why this stage matters
- Keeps parsing concerns separate from semantic validation.
- Gives a clear boundary for future language expansion without rewriting downstream IR logic.
