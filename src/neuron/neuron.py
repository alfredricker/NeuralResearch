from typing import List

def sigma(x: float) -> float:
    """Bounded readout in (-1, 1)."""
    return x / (abs(x) + 1.0)

class Neuron:
    def __init__(self, index: int, 
        theta: float,
        activity_reset: float = -0.2, 
        initial_activity: float = 0.0):
        
        self.index = index # for id purposes
        self.activity_reset = activity_reset
        self.activity = initial_activity

    def update_activity(self, activity: float) -> None:
        self.activity = activity

    # how should we handle fire checks? resetting right away will lose the activity update for other neurons
    def fire(self) -> None:
        self.activity = self.activity_reset

    @classmethod
    def create_neurons(cls, n: int,
        theta: float, 
        activity_reset: float = -0.2, 
        initial_activity: float = 0.0) -> List[Neuron]:
        return [cls(index=i, theta=theta, activity_reset=activity_reset, initial_activity=initial_activity) for i in range(n)]