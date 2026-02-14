# Neurons
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
