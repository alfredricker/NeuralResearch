# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project

A biologically-inspired neural network simulation in Rust. The goal is a computationally efficient substrate for novel neuroscience-inspired learning algorithms (cortical column model, Hawkins' thousand brains theory) alongside classical ML approaches.

This is early-stage research code. Correctness and conceptual clarity matter more than premature optimization.

## Commands

```bash
cargo build                        # build
cargo test                         # run all tests
cargo test <test_name>             # run a single test by name (substring match)
cargo test -- --nocapture          # run tests with println! output visible
cargo clippy                       # lint
cargo clippy -- -D warnings        # lint, fail on warnings
```

## Architecture

The simulation models biological neurons with dendritic structure:

- `neuron/neuron.rs` — `Neuron`: the core unit; soma integrates signals from branches
- `neuron/branch.rs` — `Branch`: a dendritic compartment; `branch_constant` controls local integration behavior
- `neuron/synapse.rs` — `Synapse`: a connection point on a branch; `activity` tracks recent firing, `psn` (post-synaptic neuron) is the target

The biological layering is: **Synapse → Branch → Neuron**. Synapses receive input, branches integrate their synapses locally (non-linear compartment computation), and the neuron soma integrates branch outputs to decide whether to fire.

## Working Style

The user is learning Rust seriously and wants to understand every line — no vibe coding. When suggesting code:
- Explain *why* a pattern is idiomatic Rust, not just what it does
- Point to specific traits, ownership rules, or type system features that make the approach correct
- Prefer the simplest correct implementation; introduce abstractions only when the need is concrete
