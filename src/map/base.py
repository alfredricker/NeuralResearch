from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple
import numpy as np

@dataclass(frozen=True)
class RegionChunkAssignment:
    """
    Output of global mapping stage: one chunk assigned to one region.
    """

    region_id: str
    chunk: np.ndarray
    chunk_origin: Tuple[int, int]


class GlobalMap:
    """
    Stage 1 map: Omega -> region chunks.
    """

    def route(self, sample: np.ndarray) -> List[RegionChunkAssignment]:
        raise NotImplementedError


class LocalSensoryMap:
    """
    Stage 2 map: region chunk -> sensory neuron payload.
    """

    def map_chunk_to_neurons(self, region_id: str, chunk: np.ndarray) -> Dict[str, float]:
        raise NotImplementedError


class FlatLocalSensoryMap(LocalSensoryMap):
    """
    Deterministic local mapping:
    chunk[i] -> f\"{region_id}:s_{i}\"
    """

    def __init__(self, expected_size: int | None = None):
        self.expected_size = expected_size

    def map_chunk_to_neurons(self, region_id: str, chunk: np.ndarray) -> Dict[str, float]:
        flat = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if self.expected_size is not None and flat.size != self.expected_size:
            raise ValueError(f"Expected chunk size {self.expected_size}, got {flat.size}")
        return {f"{region_id}:s_{idx}": float(value) for idx, value in enumerate(flat)}


def build_region_payloads(
    global_map: GlobalMap,
    local_maps: Dict[str, LocalSensoryMap],
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