from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from src.neuron.edge import Edge
from src.map.input.base import LocalInputMap
from src.map.output.base import LocalOutputMap
from src.map.input.local import FlatLocalInputMap
from src.map.output.local import FlatLocalOutputMap
from src.neuron.neuron import BaseNeuron


@dataclass
class BaseRegion:
    region_id: str
    neurons: Dict[str, BaseNeuron] = field(default_factory=dict)
    edges: List[Edge] = field(default_factory=list)
    # @NOTE: The feed in and feed out edges are to be added by the network graph.



class SensoryRegion(BaseRegion):
    """
    L_0 region: feed-in neurons are sensory neurons.
    For MNIST, use width=28 and height=28 (784 neurons).
    """

    def __init__(
        self,
        region_id: str,
        feed_in_size: int = 28*28,
        input_gain: float = 1.0,
        input_map: LocalInputMap | None = None,
    ):
        super().__init__(region_id=region_id)
        self.feed_in_size = feed_in_size
        self.input_gain = input_gain
        self.input_map = input_map or FlatLocalInputMap(expected_size=feed_in_size)



class EffectorRegion(BaseRegion):
    """
    Region where feed-out neurons are effectors (output-map neurons).
    """
    def __init__(
        self,
        region_id: str,
        feed_out_size: int = 10,
        output_map: LocalOutputMap | None = None,
    ):
        super().__init__(region_id=region_id)
        self.feed_out_size = feed_out_size
        self.output_map = output_map or FlatLocalOutputMap(expected_size=feed_out_size)

