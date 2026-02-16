# Neurons
Neurons are the fundamental building block of the network. They are given an activity value and a unique id.

The most important questions at the neuron level are:

- How are neurons connected together (code architecture wise)?
- How does a time step work?
- How is the code parallelized?

## Connecting Neurons
Importantly, neurons are given unique ids that provide information about their context.
The different types of neurons and their respective id prefixes are

- Feed forward sensory neuron: ``ffs``
- Feed back sensory neuron: ``fbs``
- Feed out effector neuron: ``foe``
- Feed forward (interior) neuron: ``ff``
- Feed out (interior) neuron: ``fo``
- Feed back (interior) neuron: ``fb``
- Model neuron: ``m``
- Where neuron: ``w``
- Miscellaneous neuron type: ``s``

The full neuron id format is to be `{region_id}_{neuron_type_prefix}_{number}`.

Therefore, one option for defining connections is through an edge class which contains a weight, incident neuron id, and terminal neuron id.
```python
class Edge:
    weight: torch.float32,
    incident_id: str,
    terminal_id: str
```

Computing activity updates can be done by iterating through all edges of the network and applying the ring sum to the terminal id.
Firing mechanisms are looking ideal because it would dramatically decrease the number of computations. It would simply be "did this edge activate" then apply the activity update. But this also adds intense programming and mathematical complexity.

One option is tracking if a neuron previously crossed the threshold:
```python
# Track previous state
was_above = self.above_threshold                          # last tick
is_above = self.activations.abs() > self.theta            # this tick
just_fired = is_above & ~was_above                        # rising edge only
self.above_threshold = is_above
```

A simpler one, which I shall implement first is resetting to a negative state
```python
# After computing who fired and scattering their contributions:
self.activations[fired_mask] = -0.2  # hyperpolarize below zero
```