from typing import Dict
import numpy as np
from .base import LocalSensoryMap

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