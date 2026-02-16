from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Sequence

from src.domain.base import Domain

class DecodeMode(Enum):
    ARGMAX = "argmax"
    TOPK = "topk"
    THRESHOLD = "threshold"


@dataclass(frozen=True)
class OutputPattern:
    """
    Output-side analog of EdgePattern.
    """

    region_ids: tuple[str, ...]
    decode_mode: DecodeMode = DecodeMode.ARGMAX
    top_k: int = 1
    threshold: float = 0.5

    def aggregate_scores(
        self,
        local_scores: Mapping[str, Mapping[int, float]],
        region_weights: Mapping[str, float] | None = None,
    ) -> dict[int, float]:
        if not self.region_ids:
            raise ValueError("region_ids must not be empty")
        weights = dict(region_weights or {})
        totals: dict[int, float] = {}
        for region_id in self.region_ids:
            if region_id not in local_scores:
                continue
            w = weights.get(region_id, 1.0)
            for label, value in local_scores[region_id].items():
                totals[label] = totals.get(label, 0.0) + w * float(value)
        return totals

    def decode(self, global_scores: Mapping[int, float]):
        if not global_scores:
            raise ValueError("No global scores to decode")
        if self.decode_mode == DecodeMode.ARGMAX:
            return max(global_scores, key=global_scores.get)
        if self.decode_mode == DecodeMode.TOPK:
            if self.top_k < 1:
                raise ValueError("top_k must be >= 1")
            ranked = sorted(global_scores.items(), key=lambda kv: kv[1], reverse=True)
            return ranked[: self.top_k]
        if self.decode_mode == DecodeMode.THRESHOLD:
            return [label for label, score in global_scores.items() if score >= self.threshold]
        raise ValueError(f"Unsupported decode mode: {self.decode_mode.value}")

    def validate_labels(self, labels: Sequence[int], domain: Domain) -> None:
        max_label = domain.discrete_set_cardinality - 1
        for label in labels:
            if label < 0 or label > max_label:
                raise ValueError(
                    f"Label {label} is outside output domain range [0, {max_label}]"
                )
