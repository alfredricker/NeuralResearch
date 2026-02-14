from __future__ import annotations

from src.map.output.base import RegionOutputAssignment
from src.map.output.oglobal import ClassificationVoteGlobalOutputMap
from src.map.output.local import FlatLocalOutputMap


def main() -> None:
    # Local output map example (one region's neurons -> local class scores)
    region_id = "CLS"
    neuron_outputs = {
        "CLS:z_0": 0.12,
        "CLS:z_1": 0.83,
        "CLS:z_2": 0.21,
        "CLS:z_3": 0.41,
    }

    mapper = FlatLocalOutputMap(expected_size=4)
    assignment = mapper.map_region_output(region_id=region_id, neuron_outputs=neuron_outputs)

    print("=== Local Output Map Example ===")
    print("Neuron outputs:")
    for neuron_id, value in neuron_outputs.items():
        print(f"{neuron_id}: {value:.2f}")

    print("\nLocal class scores:")
    for label, score in sorted(assignment.class_scores.items()):
        print(f"class {label}: {score:.2f}")

    # Global output map example (multiple local scores -> global vote)
    local_outputs = {
        "R0_0": RegionOutputAssignment("R0_0", {0: 0.20, 1: 0.80, 2: 0.10}),
        "R0_1": RegionOutputAssignment("R0_1", {0: 0.10, 1: 0.50, 2: 0.40}),
        "R1_0": RegionOutputAssignment("R1_0", {0: 0.40, 1: 0.20, 2: 0.60}),
    }
    global_map = ClassificationVoteGlobalOutputMap(
        region_weights={"R0_0": 1.0, "R0_1": 1.0, "R1_0": 0.8}
    )
    global_scores = global_map.aggregate(local_outputs)
    prediction = global_map.predict(local_outputs)

    print("\n=== Global Output Map Example ===")
    print("Aggregated class scores:")
    for label, score in sorted(global_scores.items()):
        print(f"class {label}: {score:.2f}")
    print(f"Predicted class: {prediction}")


if __name__ == "__main__":
    main()
