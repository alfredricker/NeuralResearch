import torch

class LehmerRing:
    def __init__(self, decay_constant: torch.float32 = 4.0):
        self.decay_constant = decay_constant

    def add(self, x: torch.float32, y: torch.float32) -> torch.float32:
        # Calculate sum and difference once
        add = x + y
        diff = x - y
        
        # Adaptive epsilon: shrinks as activity (add) grows
        # We add 1.0 to denominator to prevent division by zero at initialization
        epsilon = self.decay_constant / (1.0 + torch.abs(add)).pow(2)
        
        # The Smooth Max formula
        # 0.5 * (x + y + sqrt((x-y)^2 + epsilon))
        smooth_max = 0.5 * (add + torch.sqrt(diff.pow(2) + epsilon))
        
        return smooth_max