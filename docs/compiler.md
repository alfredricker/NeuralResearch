```python
contract = OutputContract(
    domain_id="mnist",
    selector="z=",
    label_count=10,
    neurons_per_label=8,
)

# Suppose compiler resolved all matching Z neurons:
z_neuron_ids = [
    "R_out_z0_0", "R_out_z0_1", "R_out_z0_2",  # ...
]

required = contract.label_count * contract.neurons_per_label
if len(z_neuron_ids) < required:
    raise ValueError(
        f"Not enough Z neurons: need {required}, got {len(z_neuron_ids)}"
    )

# Compile deterministic slots: label -> neuron ids
label_to_neurons = {}
cursor = 0
for label in range(contract.label_count):
    label_to_neurons[label] = z_neuron_ids[cursor: cursor + contract.neurons_per_label]
    cursor += contract.neurons_per_label
```