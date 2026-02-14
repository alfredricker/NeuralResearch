from __future__ import annotations

from typing import Dict, Mapping

from .base import LocalOutputMap, RegionOutputAssignment


class FlatLocalOutputMap(LocalOutputMap):
    """
    Simple local output map:
    expects feed-out neuron ids in the form `{region_id}:z_{class_idx}` and
    turns them into local class scores.
    """

    def __init__(self, expected_size: int | None = None):
        self.expected_size = expected_size

    def map_region_output(
        self,
        region_id: str,
        neuron_outputs: Mapping[str, float],
    ) -> RegionOutputAssignment:
        if self.expected_size is not None and len(neuron_outputs) != self.expected_size:
            raise ValueError(
                f"Expected {self.expected_size} output neurons, got {len(neuron_outputs)}"
            )

        class_scores: Dict[int, float] = {}
        prefix = f"{region_id}:z_"

        for neuron_id, value in neuron_outputs.items():
            if not neuron_id.startswith(prefix):
                continue
            label_str = neuron_id[len(prefix) :]
            try:
                label = int(label_str)
            except ValueError:
                continue
            class_scores[label] = class_scores.get(label, 0.0) + float(value)

        return RegionOutputAssignment(region_id=region_id, class_scores=class_scores)
