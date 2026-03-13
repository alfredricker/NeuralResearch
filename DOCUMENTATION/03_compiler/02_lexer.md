# Lexer

## Purpose
The lexer transforms raw STN source text into a typed token stream the parser can consume deterministically.

## Input
- A full `.stn` source string.

## Output
- `Vec<SpannedToken>` where each token contains:
  - token kind (`Token`)
  - source byte span (`start..end`) for diagnostics
- Or `LexError` with the invalid fragment and span.

## What it recognizes now
- punctuation/operators: `=`, `->`, `{}`, `()`, `:`, `;`, `,`
- block keywords: `graph`, `subgraph`, `learn`, `display`, `topology`
- declaration keywords: `input`, `output`, `nodes`
- topology/transform keywords: `sparse`, `identity`, `dense`, `weighted_sum`
- IO kinds: `Image`, `Language`, `Class`, `Logits`
- literals/identifiers: integers, floats, identifiers

## Why this stage matters
- It isolates text-level concerns from grammar rules.
- It attaches span info once so all later errors can point back to source.
