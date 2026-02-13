from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from src.map.base import LocalSensoryMap
from src.map.mnist import MnistTiledGlobalMap

from .builder import build_local_maps_for_network, build_network
from .graph import CorticalNetwork
from .spec import EdgeSpec, NetworkSpec, RegionSpec


@dataclass
class MNISTNetworkRuntime:
    network: CorticalNetwork
    global_map: MnistTiledGlobalMap
    local_maps: Dict[str, LocalSensoryMap]

    def ingest(self, sample: np.ndarray) -> None:
        self.network.apply_sample(
            sample=sample,
            global_map=self.global_map,
            local_maps=self.local_maps,
        )

    def step(self, ticks: int = 1) -> None:
        for _ in range(ticks):
            self.network.step(include_feed_in=True)

    def predict(self) -> int:
        return self.network.predict()

    def class_scores(self) -> Dict[int, float]:
        return self.network.class_scores()


def build_mnist_simple_spec(
    grid_shape: Tuple[int, int] = (2, 2),
    input_shape: Tuple[int, int] = (28, 28),
    num_classes: int = 10,
    overlap: int = 0,
    classifier_region_id: str = "CLS",
) -> NetworkSpec:
    grid_rows, grid_cols = grid_shape
    if input_shape[0] % grid_rows != 0 or input_shape[1] % grid_cols != 0:
        raise ValueError("input_shape must be divisible by grid_shape")

    tile_h = input_shape[0] // grid_rows
    tile_w = input_shape[1] // grid_cols
    num_tile_neurons = tile_h * tile_w

    region_specs = []
    edge_specs = []

    for r in range(grid_rows):
        for c in range(grid_cols):
            rid = f"R{r}_{c}"
            region_specs.append(
                RegionSpec(
                    region_id=rid,
                    kind="sensory",
                    width=tile_w,
                    height=tile_h,
                )
            )
            edge_specs.append(
                EdgeSpec(
                    src_region_id=rid,
                    dst_region_id=classifier_region_id,
                    pattern="dense",
                    weight=1.0,
                )
            )

    region_specs.append(
        RegionSpec(
            region_id=classifier_region_id,
            kind="effector",
            num_feed_in=num_tile_neurons,
            num_classes=num_classes,
        )
    )

    return NetworkSpec(
        regions=tuple(region_specs),
        edges=tuple(edge_specs),
        mnist_grid_shape=grid_shape,
        mnist_input_shape=input_shape,
        mnist_overlap=overlap,
    )


def build_mnist_simple_network(
    grid_shape: Tuple[int, int] = (2, 2),
    input_shape: Tuple[int, int] = (28, 28),
    num_classes: int = 10,
    overlap: int = 0,
) -> MNISTNetworkRuntime:
    spec = build_mnist_simple_spec(
        grid_shape=grid_shape,
        input_shape=input_shape,
        num_classes=num_classes,
        overlap=overlap,
    )
    network = build_network(spec)
    global_map = MnistTiledGlobalMap(
        region_grid_shape=spec.mnist_grid_shape,
        input_shape=spec.mnist_input_shape,
        overlap=spec.mnist_overlap,
    )
    local_maps = build_local_maps_for_network(spec)
    return MNISTNetworkRuntime(network=network, global_map=global_map, local_maps=local_maps)
