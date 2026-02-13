from __future__ import annotations

from typing import Dict

from src.map.local import FlatLocalSensoryMap
from src.region.mnist import MNISTNumberClassifierRegion, MNISTSensoryRegion
from src.region.region import BaseRegion, RelayRegion

from .graph import CorticalNetwork
from .spec import NetworkSpec, RegionSpec


def _build_region(region_spec: RegionSpec) -> BaseRegion:
    if region_spec.kind == "sensory":
        expected_size = region_spec.width * region_spec.height
        local_map = FlatLocalSensoryMap(expected_size=expected_size)
        return MNISTSensoryRegion(
            region_id=region_spec.region_id,
            width=region_spec.width,
            height=region_spec.height,
            input_gain=1.0,
            local_map=local_map,
        )
    if region_spec.kind == "relay":
        return RelayRegion(
            region_id=region_spec.region_id,
            num_feed_in=region_spec.num_feed_in,
            num_hidden=region_spec.num_hidden,
            num_feed_out=region_spec.num_feed_out,
        )
    if region_spec.kind == "effector":
        return MNISTNumberClassifierRegion(
            region_id=region_spec.region_id,
            num_feed_in=region_spec.num_feed_in,
            num_classes=region_spec.num_classes,
        )
    raise ValueError(f"Unknown region kind: {region_spec.kind}")


def build_network(spec: NetworkSpec) -> CorticalNetwork:
    network = CorticalNetwork()
    for region_spec in spec.regions:
        network.add_region(_build_region(region_spec))
    for edge_spec in spec.edges:
        network.connect_regions(
            src_region_id=edge_spec.src_region_id,
            dst_region_id=edge_spec.dst_region_id,
            pattern=edge_spec.pattern,
            weight=edge_spec.weight,
        )
    return network


def build_local_maps_for_network(spec: NetworkSpec) -> Dict[str, FlatLocalSensoryMap]:
    local_maps: Dict[str, FlatLocalSensoryMap] = {}
    for region_spec in spec.regions:
        if region_spec.kind != "sensory":
            continue
        local_maps[region_spec.region_id] = FlatLocalSensoryMap(
            expected_size=region_spec.width * region_spec.height
        )
    return local_maps
