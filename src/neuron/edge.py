import torch

class Edge:
    weight: torch.float32
    incident_id: str
    terminal_id: str