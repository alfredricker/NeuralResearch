from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum
from typing import Iterable


class EdgePatternKind(Enum):
    DENSE = "dense"
    BIJECT = "biject"
    STOCHASTIC = "stochastic"


@dataclass(frozen=True)
class EdgePattern:
    """
    Edge wiring policy between two regions.

    - dense: every source connects to every destination.
    - biject: one-to-one mapping (equal cardinality required).
    - stochastic: randomized mapping, surjective with >= min_fan_in incoming
      connections per destination neuron.
    """

    kind: EdgePatternKind
    weight: float = 1.0
    min_fan_in: int = 1
    extra_connection_prob: float = 0.0
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.min_fan_in < 1:
            raise ValueError("min_fan_in must be >= 1")
        if not (0.0 <= self.extra_connection_prob <= 1.0):
            raise ValueError("extra_connection_prob must be in [0.0, 1.0]")

    @classmethod
    def dense(cls, weight: float = 1.0) -> "EdgePattern":
        return cls(kind=EdgePatternKind.DENSE, weight=weight)

    @classmethod
    def biject(cls, weight: float = 1.0) -> "EdgePattern":
        return cls(kind=EdgePatternKind.BIJECT, weight=weight)

    @classmethod
    def stochastic(
        cls,
        weight: float = 1.0,
        min_fan_in: int = 1,
        extra_connection_prob: float = 0.0,
        seed: int | None = None,
    ) -> "EdgePattern":
        return cls(
            kind=EdgePatternKind.STOCHASTIC,
            weight=weight,
            min_fan_in=min_fan_in,
            extra_connection_prob=extra_connection_prob,
            seed=seed,
        )

    def with_weight(self, weight: float) -> "EdgePattern":
        return EdgePattern(
            kind=self.kind,
            weight=weight,
            min_fan_in=self.min_fan_in,
            extra_connection_prob=self.extra_connection_prob,
            seed=self.seed,
        )

    @classmethod
    def coerce(
        cls,
        pattern: "EdgePattern | EdgePatternKind | str",
        weight: float | None = None,
    ) -> "EdgePattern":
        if isinstance(pattern, EdgePattern):
            if weight is None:
                return pattern
            return pattern.with_weight(weight)
        if isinstance(pattern, EdgePatternKind):
            return EdgePattern(kind=pattern, weight=1.0 if weight is None else weight)
        if isinstance(pattern, str):
            normalized = pattern.strip().lower()
            aliases = {
                "dense": EdgePatternKind.DENSE,
                "biject": EdgePatternKind.BIJECT,
                "one_to_one": EdgePatternKind.BIJECT,
                "stochastic": EdgePatternKind.STOCHASTIC,
            }
            if normalized not in aliases:
                raise ValueError(f"Unknown edge pattern: {pattern}")
            return EdgePattern(kind=aliases[normalized], weight=1.0 if weight is None else weight)
        raise TypeError(f"Unsupported pattern type: {type(pattern).__name__}")

    def connection_pairs(
        self,
        src_ids: Iterable[str],
        dst_ids: Iterable[str],
    ) -> list[tuple[str, str]]:
        src = sorted(src_ids)
        dst = sorted(dst_ids)
        if not src:
            raise ValueError("Source region has no feed-out neurons")
        if not dst:
            raise ValueError("Destination region has no feed-in neurons")

        if self.kind == EdgePatternKind.DENSE:
            return [(sid, did) for sid in src for did in dst]

        if self.kind == EdgePatternKind.BIJECT:
            if len(src) != len(dst):
                raise ValueError(f"biject requires equal sizes; got {len(src)} and {len(dst)}")
            return list(zip(src, dst))

        if self.kind == EdgePatternKind.STOCHASTIC:
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
                            selected_src.add(sid)
                for sid in selected_src:
                    pairs.add((sid, did))
            return sorted(pairs)

        raise ValueError(f"Unsupported edge pattern: {self.kind.value}")
