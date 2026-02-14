from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

from src.map.base import GlobalMap, LocalSensoryMap, build_region_payloads
from src.region.region import BaseRegion, EffectorRegion

from .edge_pattern import EdgePattern, EdgePatternKind


@dataclass(frozen=True)
class RegionEdge:
    src_region_id: str
    dst_region_id: str
    pattern: EdgePattern
    weight: float = 1.0


class CorticalNetwork:
    """
    Runtime graph for cortical-style region networks.
    Handles region registry, inter-region wiring, and stepping.
    """

    def __init__(self):
        self.regions: Dict[str, BaseRegion] = {}
        self.edges: List[RegionEdge] = []

    def add_region(self, region: BaseRegion) -> None:
        if region.region_id in self.regions:
            raise ValueError(f"Region already exists: {region.region_id}")
        self.regions[region.region_id] = region

    def connect_regions(
        self,
        src_region_id: str,
        dst_region_id: str,
        pattern: EdgePattern | EdgePatternKind | str = "dense",
        weight: float | None = None,
    ) -> None:
        if src_region_id not in self.regions:
            raise ValueError(f"Unknown source region: {src_region_id}")
        if dst_region_id not in self.regions:
            raise ValueError(f"Unknown destination region: {dst_region_id}")

        src = self.regions[src_region_id]
        dst = self.regions[dst_region_id]

        src_ids = sorted(src.feed_out_ids)
        dst_ids = sorted(dst.feed_in_ids)
        pattern_obj = EdgePattern.coerce(pattern, weight=weight)

        try:
            pairs = pattern_obj.connection_pairs(src_ids=src_ids, dst_ids=dst_ids)
        except ValueError as exc:
            message = str(exc)
            if message == "Source region has no feed-out neurons":
                raise ValueError(f"{message}: {src_region_id}") from exc
            if message == "Destination region has no feed-in neurons":
                raise ValueError(f"{message}: {dst_region_id}") from exc
            raise

        for sid, did in pairs:
            src.neurons[sid].terminal_weights[did] = pattern_obj.weight
            dst.neurons[did].incident_weights[sid] = pattern_obj.weight

        self.edges.append(
            RegionEdge(
                src_region_id=src_region_id,
                dst_region_id=dst_region_id,
                pattern=pattern_obj,
                weight=pattern_obj.weight,
            )
        )

    def apply_region_payloads(self, payloads: Dict[str, Dict[str, float]]) -> None:
        for region_id, payload in payloads.items():
            if region_id not in self.regions:
                continue
            self.regions[region_id].apply_inputs(payload)

    def apply_sample(
        self,
        sample,
        global_map: GlobalMap,
        local_maps: Dict[str, LocalSensoryMap],
    ) -> None:
        payloads = build_region_payloads(global_map=global_map, local_maps=local_maps, sample=sample)
        self.apply_region_payloads(payloads)

    def _global_outputs(self) -> Dict[str, float]:
        outputs: Dict[str, float] = {}
        for region in self.regions.values():
            outputs.update(region.output_signals(feed_out_only=False))
        return outputs

    def step(self, include_feed_in: bool = True) -> None:
        """
        Network-level tick using global outputs so inter-region edges contribute.
        """
        outputs = self._global_outputs()
        for region in self.regions.values():
            for neuron_id, neuron in region.neurons.items():
                if not include_feed_in and neuron_id in region.feed_in_ids:
                    continue
                neuron.step(outputs)

    def class_scores(self) -> Dict[int, float]:
        scores: Dict[int, float] = {}
        for region in self.regions.values():
            if isinstance(region, EffectorRegion):
                for label, value in region.class_scores().items():
                    scores[label] = scores.get(label, 0.0) + value
        return scores

    def predict(self) -> int:
        scores = self.class_scores()
        if not scores:
            raise ValueError("No effector scores available in network")
        return max(scores, key=scores.get)

    def region_ids(self) -> Iterable[str]:
        return self.regions.keys()
