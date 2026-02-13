from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np

from .base import GlobalMap, LocalSensoryMap, RegionChunkAssignment

class MnistTiledGlobalMap(GlobalMap):
    """
    Split a 2D image into fixed rectangular tiles routed to region IDs.
    Optionally supports overlap (in pixels) between neighboring tiles.
    """

    def __init__(
        self,
        region_grid_shape: Tuple[int, int],
        input_shape: Tuple[int, int] = (28, 28),
        overlap: int = 0,
        region_id_prefix: str = "R",
    ):
        self.grid_rows, self.grid_cols = region_grid_shape
        self.input_h, self.input_w = input_shape
        self.overlap = overlap
        self.region_id_prefix = region_id_prefix

        if self.grid_rows <= 0 or self.grid_cols <= 0:
            raise ValueError("region_grid_shape must be positive")
        if overlap < 0:
            raise ValueError("overlap must be >= 0")

        if self.input_h % self.grid_rows != 0 or self.input_w % self.grid_cols != 0:
            raise ValueError("input shape must be divisible by region grid shape")

        self.base_tile_h = self.input_h // self.grid_rows
        self.base_tile_w = self.input_w // self.grid_cols

    def _region_id(self, row: int, col: int) -> str:
        return f"{self.region_id_prefix}{row}_{col}"

    def route(self, sample: np.ndarray) -> List[RegionChunkAssignment]:
        arr = np.asarray(sample, dtype=np.float32)
        if arr.size == self.input_h * self.input_w and arr.shape != (self.input_h, self.input_w):
            arr = arr.reshape(self.input_h, self.input_w)
        if arr.shape != (self.input_h, self.input_w):
            raise ValueError(f"Expected shape {(self.input_h, self.input_w)}, got {arr.shape}")

        assignments: List[RegionChunkAssignment] = []
        for r in range(self.grid_rows):
            for c in range(self.grid_cols):
                y0 = r * self.base_tile_h
                x0 = c * self.base_tile_w
                y1 = y0 + self.base_tile_h
                x1 = x0 + self.base_tile_w

                # Apply overlap while clipping to image bounds.
                oy0 = max(0, y0 - self.overlap)
                ox0 = max(0, x0 - self.overlap)
                oy1 = min(self.input_h, y1 + self.overlap)
                ox1 = min(self.input_w, x1 + self.overlap)

                chunk = arr[oy0:oy1, ox0:ox1]
                assignments.append(
                    RegionChunkAssignment(
                        region_id=self._region_id(r, c),
                        chunk=chunk,
                        chunk_origin=(oy0, ox0),
                    )
                )
        return assignments


class MnistFlatLocalSensoryMap(LocalSensoryMap):
    """
    Local MNIST map:
    flatten chunk and map to sensory ids `region_id:s_i`.
    """

    def __init__(self, expected_size: int | None = None):
        self.expected_size = expected_size

    def map_chunk_to_neurons(self, region_id: str, chunk: np.ndarray) -> Dict[str, float]:
        flat = np.asarray(chunk, dtype=np.float32).reshape(-1)
        if self.expected_size is not None and flat.size != self.expected_size:
            raise ValueError(f"Expected chunk size {self.expected_size}, got {flat.size}")
        return {f"{region_id}:s_{idx}": float(value) for idx, value in enumerate(flat)}


def build_mnist_payloads(
    global_map: MnistTiledGlobalMap,
    sample: np.ndarray,
    expected_sizes: Dict[str, int] | None = None,
) -> Dict[str, Dict[str, float]]:
    """
    One-call helper for experiments:
    sample -> region chunk assignments -> per-region sensory payloads.
    """
    payloads: Dict[str, Dict[str, float]] = {}
    assignments = global_map.route(sample)
    for assignment in assignments:
        expected = None if expected_sizes is None else expected_sizes.get(assignment.region_id)
        local_map = MnistFlatLocalSensoryMap(expected_size=expected)
        payloads[assignment.region_id] = local_map.map_chunk_to_neurons(
            assignment.region_id, assignment.chunk
        )
    return payloads
