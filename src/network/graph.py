from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Literal

from src.map.base import GlobalMap, LocalSensoryMap, build_region_payloads
from src.region.region import BaseRegion, EffectorRegion

EdgePattern = Literal["dense", "one_to_one"]


@dataclass(frozen=True)
class RegionEdge:
    src_region_id: str
    dst_region_id: str
    pattern: EdgePattern = "dense"
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
        pattern: EdgePattern = "dense",
        weight: float = 1.0,
    ) -> None:
        if src_region_id not in self.regions:
            raise ValueError(f"Unknown source region: {src_region_id}")
        if dst_region_id not in self.regions:
            raise ValueError(f"Unknown destination region: {dst_region_id}")

        src = self.regions[src_region_id]
        dst = self.regions[dst_region_id]

        src_ids = sorted(src.feed_out_ids)
        dst_ids = sorted(dst.feed_in_ids)

        if not src_ids:
            raise ValueError(f"Source region has no feed-out neurons: {src_region_id}")
        if not dst_ids:
            raise ValueError(f"Destination region has no feed-in neurons: {dst_region_id}")

        if pattern == "dense":
            for sid in src_ids:
                src_neuron = src.neurons[sid]
                for did in dst_ids:
                    src_neuron.terminal_weights[did] = weight
                    dst.neurons[did].incident_weights[sid] = weight
        elif pattern == "one_to_one":
            if len(src_ids) != len(dst_ids):
                raise ValueError(
                    f"one_to_one requires equal sizes; got {len(src_ids)} and {len(dst_ids)}"
                )
            for sid, did in zip(src_ids, dst_ids):
                src.neurons[sid].terminal_weights[did] = weight
                dst.neurons[did].incident_weights[sid] = weight
        else:
            raise ValueError(f"Unsupported edge pattern: {pattern}")

        self.edges.append(
            RegionEdge(
                src_region_id=src_region_id,
                dst_region_id=dst_region_id,
                pattern=pattern,
                weight=weight,
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
