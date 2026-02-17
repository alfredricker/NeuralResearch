from dataclasses import dataclass

@dataclass
class Edge:
    weight: float
    incident_id: str
    terminal_id: str