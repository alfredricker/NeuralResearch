from __future__ import annotations
from dataclasses import dataclass, field
from src.neuron.edge import Edge


@dataclass(frozen=True)
class InputPortSpec:
    name: str
    required: bool = True
    allowed_source_types: set[str] | None = None
    min_bindings: int = 1
    max_bindings: int | None = None

    def validate(self) -> None:
        if self.min_bindings < 0:
            raise ValueError(f"port {self.name}: min_bindings must be >= 0")
        if self.max_bindings is not None and self.max_bindings < self.min_bindings:
            raise ValueError(
                f"port {self.name}: max_bindings must be >= min_bindings"
            )


@dataclass
class InputPortBinding:
    port_name: str
    source_group_id: str
    source_group_type: str
    topology_name: str
    edges: list[Edge] = field(default_factory=list)