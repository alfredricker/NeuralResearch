from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Tuple

RegionKind = Literal["sensory", "relay", "effector"]

from .edge_pattern import EdgePattern


@dataclass(frozen=True)
class RegionSpec:
    region_id: str
    kind: RegionKind
    width: int = 28
    height: int = 28
    num_feed_in: int = 0
    num_hidden: int = 0
    num_feed_out: int = 0
    num_classes: int = 0


@dataclass(frozen=True)
class EdgeSpec:
    src_region_id: str
    dst_region_id: str
    pattern: EdgePattern = field(default_factory=EdgePattern.dense)
    weight: float | None = None


@dataclass(frozen=True)
class NetworkSpec:
    regions: tuple[RegionSpec, ...]
    edges: tuple[EdgeSpec, ...]
    mnist_grid_shape: Tuple[int, int] = (2, 2)
    mnist_input_shape: Tuple[int, int] = (28, 28)
    mnist_overlap: int = 0
    metadata: dict[str, str] = field(default_factory=dict)
