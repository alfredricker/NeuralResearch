from __future__ import annotations

import random
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Iterable

from src.cortical.neuron.edge import Edge


class EdgeTopology(ABC):
    """
    Wiring strategy that maps source neuron ids to destination neuron ids.

    This is intentionally agnostic to group/region semantics so it can be used
    for both inter-group and recurrent/intra-group wiring.
    """

    @property
    def name(self) -> str:
        return self.__class__.__name__.replace("Topology", "").lower()

    @abstractmethod
    def connection_pairs(
        self,
        src_ids: Iterable[str],
        dst_ids: Iterable[str],
    ) -> list[tuple[str, str]]:
        raise NotImplementedError

    def make_edges(
        self,
        src_ids: Iterable[str],
        dst_ids: Iterable[str],
        weight: float,
    ) -> list[Edge]:
        return [
            Edge(weight=weight, incident_id=sid, terminal_id=did)
            for sid, did in self.connection_pairs(src_ids, dst_ids)
        ]


@dataclass(frozen=True)
class DenseTopology(EdgeTopology):
    def connection_pairs(
        self,
        src_ids: Iterable[str],
        dst_ids: Iterable[str],
    ) -> list[tuple[str, str]]:
        src = sorted(src_ids)
        dst = sorted(dst_ids)
        if not src:
            raise ValueError("Source set is empty")
        if not dst:
            raise ValueError("Destination set is empty")
        return [(sid, did) for sid in src for did in dst]


@dataclass(frozen=True)
class BijectiveTopology(EdgeTopology):
    def connection_pairs(
        self,
        src_ids: Iterable[str],
        dst_ids: Iterable[str],
    ) -> list[tuple[str, str]]:
        src = sorted(src_ids)
        dst = sorted(dst_ids)
        if len(src) != len(dst):
            raise ValueError(f"bijective topology requires equal sizes; got {len(src)} and {len(dst)}")
        return list(zip(src, dst))


@dataclass(frozen=True)
class RingTopology(EdgeTopology):
    def connection_pairs(
        self,
        src_ids: Iterable[str],
        dst_ids: Iterable[str],
    ) -> list[tuple[str, str]]:
        src = sorted(src_ids)
        dst = sorted(dst_ids)
        if len(src) != len(dst):
            raise ValueError(f"ring topology requires equal sizes; got {len(src)} and {len(dst)}")
        n = len(src)
        if n < 2:
            raise ValueError("ring topology requires at least 2 neurons")
        return [(src[i], dst[(i + 1) % n]) for i in range(n)]


@dataclass(frozen=True)
class StochasticTopology(EdgeTopology):
    min_fan_in: int = 1
    extra_connection_prob: float = 0.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.min_fan_in < 1:
            raise ValueError("min_fan_in must be >= 1")
        if not (0.0 <= self.extra_connection_prob <= 1.0):
            raise ValueError("extra_connection_prob must be in [0.0, 1.0]")

    def connection_pairs(
        self,
        src_ids: Iterable[str],
        dst_ids: Iterable[str],
    ) -> list[tuple[str, str]]:
        src = sorted(src_ids)
        dst = sorted(dst_ids)
        if not src:
            raise ValueError("Source set is empty")
        if not dst:
            raise ValueError("Destination set is empty")
        if self.min_fan_in > len(src):
            raise ValueError(
                f"stochastic min_fan_in={self.min_fan_in} exceeds source count={len(src)}"
            )

        rng = random.Random(self.seed)
        pairs: set[tuple[str, str]] = set()
        for did in dst:
            selected_src = set(rng.sample(src, k=self.min_fan_in))
            if self.extra_connection_prob > 0.0:
                for sid in src:
                    if sid in selected_src:
                        continue
                    if rng.random() < self.extra_connection_prob:
                        selected_src.cortical.add(sid)
            for sid in selected_src:
                pairs.add((sid, did))
        return sorted(pairs)
