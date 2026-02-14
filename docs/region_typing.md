## Architecture Overview

This project currently uses a two-stage mapping pipeline and typed regions/neurons:

- `Omega -> region chunks` (global map)
- `chunk -> sensory neuron payload` (local map)
- region stepping and feedforward signaling

The goal is to keep interfaces stable while changing mapping and learning strategies.

## ID Conventions

Neuron IDs are globally unique strings and include the region prefix.

- Sensory neurons: ``{region_id}:s_{i}``
- Feed-in neurons: ``{region_id}:fin_{i}``
- Hidden/internal neurons: ``{region_id}:h_{i}``
- Feed-out neurons: ``{region_id}:fout_{i}``
- Effector/class neurons: ``{region_id}:z_{k}``

Region IDs are caller-defined (examples: `R0_0`, `R1_2`, `SENSORY`, `CLS`).

## Map Architecture

### Stage 1: Global map (domain -> regions)

`GlobalMap.route(sample)` returns a list of `RegionChunkAssignment`:

- `region_id`: target region
- `chunk`: data slice for that region
- `chunk_origin`: `(row, col)` top-left in the source sample

MNIST implementation:

- `MnistTiledGlobalMap` splits a `28x28` image into a grid of tiles
- optional overlap expands each tile before clipping to bounds

### Stage 2: Local sensory map (chunk -> neuron payload)

`LocalSensoryMap.map_chunk_to_neurons(region_id, chunk)` returns:

- `Dict[str, float]` where key is neuron id and value is activation/input

Default implementation:

- `FlatLocalSensoryMap` / `MnistFlatLocalSensoryMap`
- flatten chunk and map `chunk[i] -> "{region_id}:s_{i}"`

## Region Typing

All regions extend `BaseRegion`:

- `neurons: Dict[str, BaseNeuron]`
- `feed_in_ids: Set[str]`
- `feed_out_ids: Set[str]`
- `connect(src_id, dst_id, weight)` for directed edges
- `apply_inputs(payload)` for setting input/activity
- `output_signals(feed_out_only=True)` for downstream use
- `step(include_feed_in=False)` for one tick of neuron updates

Concrete region types:

- `SensoryLevelRegion` (`L_0`)
  - feed-in is sensory neurons
  - accepts chunk/image input through local mapping
- `RelayRegion`
  - generic feed-in -> hidden -> feed-out structure
- `EffectorRegion`
  - feed-out are `EffectorNeuron`s used for output map / class scores

MNIST-specific wrappers:

- `MNISTSensoryRegion`
- `MNISTNumberClassifierRegion`

## Neuron Typing

All neurons inherit from `BaseNeuron`:

- core state: `activity`, `decay`, `threshold`
- connectivity:
  - `incident_weights` (`P(h)`: incoming edges)
  - `terminal_weights` (`Q(h)`: outgoing edges)
- update rule uses bounded readout `sigma(x) = x / (|x| + 1)`

Current neuron classes:

- `SensoryNeuron`
  - direct external input via `apply_input(value)`
- `StandardNeuron`
  - default internal neuron with synaptic integration (+ optional bias)
- `EffectorNeuron`
  - output neuron with `output_label` and `readout()`


