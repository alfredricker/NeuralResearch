from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict


def sigma(x: float) -> float:
    """Bounded readout in (-1, 1)."""
    return x / (abs(x) + 1.0)


@dataclass
class BaseNeuron:
    neuron_id: str
    decay: float = 0.05
    threshold: float = 0.1
    activity: float = 0.0

    # adjacency + weights
    incident_weights: Dict[str, float] = field(default_factory=dict)  # P(h): src_id -> w(src, self)
    terminal_weights: Dict[str, float] = field(default_factory=dict)  # Q(h): dst_id -> w(self, dst)

    def is_active(self) -> bool:
        return self.activity > self.threshold

    def synaptic_input(self, neuron_outputs: Dict[str, float]) -> float:
        # f_h = sum_{p in P(h)} sigma(alpha_p) * w(p,h)
        total = 0.0
        for src_id, w in self.incident_weights.items():
            total += neuron_outputs.get(src_id, 0.0) * w
        return total

    def step(self, neuron_outputs: Dict[str, float]) -> None:
        # alpha(t+1) = (1-lambda)alpha(t) + f/(|f|+1)
        f = self.synaptic_input(neuron_outputs)
        self.activity = (1.0 - self.decay) * self.activity + (f / (abs(f) + 1.0))


@dataclass
class SensoryNeuron(BaseNeuron):
    """
    Input-boundary neuron (F_omega).
    Typically receives direct external signal each tick.
    """
    input_gain: float = 1.0

    def apply_input(self, value: float) -> None:
        # direct mapping alpha = p (or scaled)
        self.activity = self.input_gain * value


@dataclass
class StandardNeuron(BaseNeuron):
    """
    Internal processing neuron (for now: feedforward/recurrent only).
    Can later add gating/lateral inhibition hooks.
    """
    bias: float = 0.0

    def step(self, neuron_outputs: Dict[str, float]) -> None:
        f = self.synaptic_input(neuron_outputs) + self.bias
        self.activity = (1.0 - self.decay) * self.activity + (f / (abs(f) + 1.0))


@dataclass
class EffectorNeuron(BaseNeuron):
    """
    Output-map neuron (F_z): converts activity to task-specific output score.
    """
    output_label: int = 0  # e.g., MNIST class 0..9
    readout_gain: float = 1.0

    def readout(self) -> float:
        # value used in V(k) aggregation
        return self.readout_gain * sigma(self.activity)