from __future__ import annotations
from dataclasses import dataclass, field
from src.neuron.edge import Edge


@dataclass(frozen=True)
class InputPortSpec:
    source_type: str
    required: bool = True

    def validate(self) -> None:
        pass

@dataclass
class InputPortBinding:
    port_name: str
    source_group_id: str
    source_group_type: str
    topology_name: str
    edges: list[Edge] = field(default_factory=list)