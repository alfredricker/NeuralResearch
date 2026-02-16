from abc import ABC, abstractmethod
from typing import List
from src.neuron.neuron import Neuron
from src.neuron.edge import Edge

class Group:
    def __init__(self, n: int, theta: float, id: str):
        self.n = n
        self.theta = theta
        self.neurons = Neuron.create_neurons(n, theta)
        self.group_id = id

        self.recurrent_edges = []

    def update_recurrent_edges(self, edges: List[Edge]) -> None:
        self.recurrent_edges = edges

    @abstractmethod
    def verify_input_ports(self) -> bool:
        raise NotImplementedError