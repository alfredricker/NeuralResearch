from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Set

import numpy as np

from src.map.base import FlatLocalSensoryMap, LocalSensoryMap
from src.neuron.neuron import BaseNeuron, EffectorNeuron, SensoryNeuron, StandardNeuron, sigma


@dataclass
class BaseRegion:
    region_id: str
    neurons: Dict[str, BaseNeuron] = field(default_factory=dict)
    feed_in_ids: Set[str] = field(default_factory=set)
    feed_out_ids: Set[str] = field(default_factory=set)

    def add_neuron(self, neuron: BaseNeuron) -> None:
        self.neurons[neuron.neuron_id] = neuron

    def set_feed_in(self, neuron_ids: Iterable[str]) -> None:
        ids = set(neuron_ids)
        missing = ids - set(self.neurons)
        if missing:
            raise ValueError(f"Unknown feed-in neuron ids: {sorted(missing)}")
        self.feed_in_ids = ids

    def set_feed_out(self, neuron_ids: Iterable[str]) -> None:
        ids = set(neuron_ids)
        missing = ids - set(self.neurons)
        if missing:
            raise ValueError(f"Unknown feed-out neuron ids: {sorted(missing)}")
        self.feed_out_ids = ids

    def connect(self, src_id: str, dst_id: str, weight: float) -> None:
        if src_id not in self.neurons:
            raise ValueError(f"Unknown source neuron: {src_id}")
        if dst_id not in self.neurons:
            raise ValueError(f"Unknown destination neuron: {dst_id}")
        self.neurons[src_id].terminal_weights[dst_id] = weight
        self.neurons[dst_id].incident_weights[src_id] = weight

    def apply_inputs(self, values_by_id: Dict[str, float]) -> None:
        for neuron_id, value in values_by_id.items():
            if neuron_id not in self.neurons:
                continue
            neuron = self.neurons[neuron_id]
            if hasattr(neuron, "apply_input"):
                neuron.apply_input(float(value))
            else:
                neuron.activity = float(value)

    def output_signals(self, feed_out_only: bool = True) -> Dict[str, float]:
        ids = self.feed_out_ids if feed_out_only else set(self.neurons.keys())
        return {neuron_id: sigma(self.neurons[neuron_id].activity) for neuron_id in ids}

    def step(self, include_feed_in: bool = False) -> None:
        current_outputs = {nid: sigma(neuron.activity) for nid, neuron in self.neurons.items()}
        for neuron_id, neuron in self.neurons.items():
            if not include_feed_in and neuron_id in self.feed_in_ids:
                continue
            neuron.step(current_outputs)


class SensoryLevelRegion(BaseRegion):
    """
    L_0 region: feed-in neurons are sensory neurons.
    For MNIST, use width=28 and height=28 (784 neurons).
    """

    def __init__(
        self,
        region_id: str,
        width: int = 28,
        height: int = 28,
        input_gain: float = 1.0,
        local_map: LocalSensoryMap | None = None,
    ):
        super().__init__(region_id=region_id)
        self.width = width
        self.height = height
        self.input_gain = input_gain
        self.local_map = local_map or FlatLocalSensoryMap(expected_size=width * height)

        sensory_ids: list[str] = []
        for idx in range(width * height):
            neuron_id = f"{region_id}:s_{idx}"
            self.add_neuron(SensoryNeuron(neuron_id=neuron_id, input_gain=input_gain))
            sensory_ids.append(neuron_id)

        self.set_feed_in(sensory_ids)
        self.set_feed_out(sensory_ids)

    def apply_chunk(self, chunk: np.ndarray) -> None:
        payload = self.local_map.map_chunk_to_neurons(self.region_id, chunk)
        self.apply_inputs(payload)


class RelayRegion(BaseRegion):
    """
    Region with explicit feed-in and feed-out neuron groups.
    Useful as a generic intermediate processing shell.
    """

    def __init__(self, region_id: str, num_feed_in: int, num_hidden: int, num_feed_out: int):
        super().__init__(region_id=region_id)

        feed_in_ids: list[str] = []
        hidden_ids: list[str] = []
        feed_out_ids: list[str] = []

        for idx in range(num_feed_in):
            neuron_id = f"{region_id}:fin_{idx}"
            self.add_neuron(StandardNeuron(neuron_id=neuron_id))
            feed_in_ids.append(neuron_id)

        for idx in range(num_hidden):
            neuron_id = f"{region_id}:h_{idx}"
            self.add_neuron(StandardNeuron(neuron_id=neuron_id))
            hidden_ids.append(neuron_id)

        for idx in range(num_feed_out):
            neuron_id = f"{region_id}:fout_{idx}"
            self.add_neuron(StandardNeuron(neuron_id=neuron_id))
            feed_out_ids.append(neuron_id)

        self.set_feed_in(feed_in_ids)
        self.set_feed_out(feed_out_ids)

        # default one-hop wiring: feed_in -> hidden -> feed_out
        for fin_id in feed_in_ids:
            for h_id in hidden_ids:
                self.connect(fin_id, h_id, weight=1.0)
        for h_id in hidden_ids:
            for fout_id in feed_out_ids:
                self.connect(h_id, fout_id, weight=1.0)


class EffectorRegion(BaseRegion):
    """
    Region where feed-out neurons are effectors (output-map neurons).
    """

    def __init__(self, region_id: str, num_feed_in: int, num_classes: int):
        super().__init__(region_id=region_id)

        feed_in_ids: list[str] = []
        effector_ids: list[str] = []

        for idx in range(num_feed_in):
            neuron_id = f"{region_id}:fin_{idx}"
            self.add_neuron(StandardNeuron(neuron_id=neuron_id))
            feed_in_ids.append(neuron_id)

        for label in range(num_classes):
            neuron_id = f"{region_id}:z_{label}"
            self.add_neuron(EffectorNeuron(neuron_id=neuron_id, output_label=label))
            effector_ids.append(neuron_id)

        self.set_feed_in(feed_in_ids)
        self.set_feed_out(effector_ids)

        for fin_id in feed_in_ids:
            for eff_id in effector_ids:
                self.connect(fin_id, eff_id, weight=1.0)

    def class_scores(self) -> Dict[int, float]:
        scores: Dict[int, float] = {}
        for neuron_id in self.feed_out_ids:
            neuron = self.neurons[neuron_id]
            if isinstance(neuron, EffectorNeuron):
                scores[neuron.output_label] = scores.get(neuron.output_label, 0.0) + neuron.readout()
        return scores
