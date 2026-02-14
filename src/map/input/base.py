from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import numpy as np

@dataclass(frozen=True)
class RegionInputAssignment:
    """
    Output of global mapping stage: one chunk assigned to one region.
    """

    region_id: str
    chunk: np.ndarray
    chunk_origin: Tuple[int, int]


class GlobalInputMap:
    """
    Stage 1 map: Omega -> region chunks.
    """

    def route(self, sample: np.ndarray) -> List[RegionInputAssignment]:
        raise NotImplementedError


class LocalInputMap:
    """
    Stage 2 map: region chunk -> sensory neuron payload.
    """

    def map_chunk_to_neurons(self, region_id: str, chunk: np.ndarray) -> Dict[str, float]:
        raise NotImplementedError


def build_region_payloads(
    global_map: GlobalInputMap,
    local_maps: Dict[str, LocalInputMap],
    sample: np.ndarray,
) -> Dict[str, Dict[str, float]]:
    """
    Convenience utility: run both stages and return per-region neuron payloads.
    """
    payloads: Dict[str, Dict[str, float]] = {}
    for assignment in global_map.route(sample):
        if assignment.region_id not in local_maps:
            raise ValueError(f"No local map configured for region {assignment.region_id}")
        local_map = local_maps[assignment.region_id]
        payloads[assignment.region_id] = local_map.map_chunk_to_neurons(
            assignment.region_id, assignment.chunk
        )
    return payloads