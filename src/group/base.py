from __future__ import annotations

from typing import Dict, List
from src.group.port import InputPortBinding, InputPortSpec
from src.neuron.neuron import Neuron
from src.neuron.edge import Edge
from src.neuron.edge_topology import EdgeTopology
from abc import ABC, abstractmethod

class Group:
    def __init__(self, n: int, theta: float, group_type: str, group_index: int | None = None,
        recurrent_topology: EdgeTopology | None = None):
        self.n = n
        self.theta = theta
        self.neurons = Neuron.create_neurons(n, theta)
        self.group_type = group_type
        group_index_str = f'{group_index}' if group_index is not None else ''
        self.group_id = f'{group_type}{group_index_str}'

        # recurrent structure
        self.recurrent_topology: EdgeTopology = recurrent_topology
        self.recurrent_edges: list[Edge] = self.build_recurrent_edges()
        
        # input structure
        self._input_bindings: Dict[str, list[InputPortBinding]] = {
            name: [] for name in self.expected_input_ports().keys()
        }

    @abstractmethod
    def expected_input_ports(self) -> dict[str, InputPortSpec]:
        pass

    def build_recurrent_edges(
        self,
        weight: float = 1.0,
    ) -> None:
        # IF no recurrent topology is provided, no recurrent edges are built
        if self.recurrent_topology is None:
            return
        src_ids = [neuron.id for neuron in self.neurons]
        dst_ids = src_ids
        self.recurrent_edges = self.recurrent_topology.make_edges(src_ids=src_ids, dst_ids=dst_ids, weight=weight)

    def input_bindings(self, port_name: str | None = None) -> list[InputPortBinding]:
        if port_name is None:
            out: list[InputPortBinding] = []
            for bindings in self._input_bindings.values():
                out.extend(bindings)
            return out
        return list(self._input_bindings.get(port_name, []))

    def verify_input_ports(self) -> bool:
        specs = self.expected_input_ports()
        for port_name, spec in specs.items():
            spec.validate()
            bindings = self._input_bindings.get(port_name, [])
            n_bindings = len(bindings)

            if spec.required and n_bindings == 0:
                return False
            if n_bindings < spec.min_bindings:
                return False
            if spec.max_bindings is not None and n_bindings > spec.max_bindings:
                return False
            if spec.allowed_source_types:
                for binding in bindings:
                    if binding.source_group_type not in spec.allowed_source_types:
                        return False
        return True