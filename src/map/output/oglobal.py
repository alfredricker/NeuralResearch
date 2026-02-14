from __future__ import annotations

from typing import Dict, Mapping

from .base import GlobalOutputMap, RegionOutputAssignment


class ClassificationVoteGlobalOutputMap(GlobalOutputMap):
    """
    Aggregates local class scores from multiple regions by weighted voting.
    """

    def __init__(self, region_weights: Mapping[str, float] | None = None):
        self.region_weights = dict(region_weights or {})

    def aggregate(
        self,
        local_outputs: Mapping[str, RegionOutputAssignment],
    ) -> Dict[int, float]:
        totals: Dict[int, float] = {}
        for region_id, assignment in local_outputs.items():
            weight = self.region_weights.get(region_id, 1.0)
            for label, value in assignment.class_scores.items():
                totals[label] = totals.get(label, 0.0) + weight * value
        return totals
