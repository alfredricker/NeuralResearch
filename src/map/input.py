from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from src.domain.base import Domain


class InputAssignmentMode(Enum):
    EXPLICIT_NEURONS_PER_POINT = "explicit_neurons_per_point"
    DERIVE_FROM_OVERLAP = "derive_from_overlap"

@dataclass(frozen=True)
class ResolvedInputPattern:
    region_ids: tuple[str, ...]
    overlap: int
    neurons_per_point: int


@dataclass(frozen=True)
class InputPattern:
    """
    Input-side analog of EdgePattern.

    Defines where domain points are routed (region ids) and how much overlap
    exists in point-to-neuron assignment.
    """

    region_ids: tuple[str, ...] | None = None
    num_regions: int | None = None
    overlap: int | None = 0
    neurons_per_point: int | None = None
    assignment_mode: InputAssignmentMode = InputAssignmentMode.DERIVE_FROM_OVERLAP
    region_prefix: str = "R"

    def resolve(self, domain: Domain) -> ResolvedInputPattern:
        region_ids = self._resolve_region_ids()
        overlap = self._resolve_overlap()
        neurons_per_point = self._resolve_neurons_per_point(region_count=len(region_ids), overlap=overlap)

        if neurons_per_point < 1:
            raise ValueError("neurons_per_point must be >= 1")
        if overlap < 0:
            raise ValueError("overlap must be >= 0")

        # Soft consistency guard with domain metadata.
        if domain.discrete_set_cardinality < 1:
            raise ValueError("domain.discrete_set_cardinality must be >= 1")

        return ResolvedInputPattern(
            region_ids=region_ids,
            overlap=overlap,
            neurons_per_point=neurons_per_point,
        )

    def _resolve_region_ids(self) -> tuple[str, ...]:
        if self.region_ids is not None:
            if not self.region_ids:
                raise ValueError("region_ids must not be empty")
            return tuple(self.region_ids)
        if self.num_regions is None or self.num_regions < 1:
            raise ValueError("Provide either region_ids or num_regions >= 1")
        return tuple(f"{self.region_prefix}{i}" for i in range(self.num_regions))

    def _resolve_overlap(self) -> int:
        if self.overlap is not None:
            return self.overlap
        if self.neurons_per_point is not None:
            return max(0, self.neurons_per_point - 1)
        return 0

    def _resolve_neurons_per_point(self, region_count: int, overlap: int) -> int:
        if self.neurons_per_point is not None:
            return self.neurons_per_point
        # Basic heuristic: each unit overlap adds one additional receiving region,
        # bounded by number of regions.
        if self.assignment_mode == InputAssignmentMode.DERIVE_FROM_OVERLAP:
            return min(region_count, overlap + 1)
        raise ValueError("neurons_per_point must be provided for explicit mode")