from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping


@dataclass(frozen=True)
class RegionOutputAssignment:
    """
    Output produced by one region after local output mapping.
    """

    region_id: str
    class_scores: Dict[int, float]


class LocalOutputMap:
    """
    Stage 1 output map: region feed-out neuron values -> local class scores.
    """

    def map_region_output(
        self,
        region_id: str,
        neuron_outputs: Mapping[str, float],
    ) -> RegionOutputAssignment:
        raise NotImplementedError


class GlobalOutputMap:
    """
    Stage 2 output map: aggregate local region scores -> global output.
    """

    def aggregate(
        self,
        local_outputs: Mapping[str, RegionOutputAssignment],
    ) -> Dict[int, float]:
        raise NotImplementedError

    def predict(
        self,
        local_outputs: Mapping[str, RegionOutputAssignment],
    ) -> int:
        scores = self.aggregate(local_outputs)
        if not scores:
            raise ValueError("No class scores available for prediction")
        return max(scores, key=scores.get)
