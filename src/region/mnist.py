from __future__ import annotations

from typing import Dict, Iterable

import numpy as np

from src.map.base import LocalSensoryMap
from src.region.region import EffectorRegion, SensoryLevelRegion


class MNISTSensoryRegion(SensoryLevelRegion):
    """
    MNIST sensory region wrapper.
    - For full-image input: width=28, height=28
    - For tiled input: pass tile width/height
    """

    def __init__(
        self,
        region_id: str,
        width: int = 28,
        height: int = 28,
        input_gain: float = 1.0,
        local_map: LocalSensoryMap | None = None,
    ):
        super().__init__(
            region_id=region_id,
            width=width,
            height=height,
            input_gain=input_gain,
            local_map=local_map,
        )

    def ingest(self, chunk: np.ndarray) -> None:
        """
        Alias for apply_chunk for readability in experiments.
        """
        self.apply_chunk(chunk)


class MNISTNumberClassifierRegion(EffectorRegion):
    """
    Simple MNIST output region.
    Feed-out neurons are effectors z_0 ... z_9 by default.
    """

    def __init__(self, region_id: str, num_feed_in: int, num_classes: int = 10):
        super().__init__(
            region_id=region_id,
            num_feed_in=num_feed_in,
            num_classes=num_classes,
        )

    def ingest_features(self, values: Dict[str, float]) -> None:
        """
        Load feature activations onto feed-in neurons.
        Expected keys are this region's feed-in neuron ids.
        """
        self.apply_inputs(values)

    def predict(self) -> int:
        scores = self.class_scores()
        if not scores:
            raise ValueError("No class scores available")
        return max(scores, key=scores.get)


def connect_feedforward_dense(
    source_region: SensoryLevelRegion,
    classifier_region: MNISTNumberClassifierRegion,
    weight: float = 1.0,
) -> None:
    """
    Dense feedforward wiring:
    every source feed-out neuron connects to every classifier feed-in neuron.
    """
    for src_id in source_region.feed_out_ids:
        src_neuron = source_region.neurons[src_id]
        for dst_id in classifier_region.feed_in_ids:
            # cross-region edge bookkeeping
            src_neuron.terminal_weights[dst_id] = weight
            classifier_region.neurons[dst_id].incident_weights[src_id] = weight


def collect_source_outputs(regions: Iterable[SensoryLevelRegion]) -> Dict[str, float]:
    """
    Gather feed-out signals from one or more source regions into a single dict.
    """
    merged: Dict[str, float] = {}
    for region in regions:
        merged.update(region.output_signals(feed_out_only=True))
    return merged
