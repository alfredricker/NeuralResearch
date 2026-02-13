"""
Small standard CNN baseline for MNIST classification.

Architecture loosely follows LeNet-5 but sized to match ~50 K parameters
so the comparison with the cortical column and Hopfield network is fair
in terms of compute / capacity budget.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class SmallCNN(nn.Module):
    """Compact CNN: 2 conv layers + 2 FC layers.

    Default sizing (~47 K params):
        Conv1 : 1 → 10 channels, 5×5        →  24×24, MaxPool → 12×12
        Conv2 : 10 → 20 channels, 5×5       →   8×8,  MaxPool →  4×4
        FC1   : 320 → 128
        FC2   : 128 → 10
    """

    def __init__(
        self,
        in_channels: int = 1,
        conv1_out: int = 10,
        conv2_out: int = 20,
        fc1_out: int = 128,
        n_classes: int = 10,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, conv1_out, kernel_size=5)
        self.conv2 = nn.Conv2d(conv1_out, conv2_out, kernel_size=5)
        self.fc1 = nn.Linear(conv2_out * 4 * 4, fc1_out)
        self.fc2 = nn.Linear(fc1_out, n_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, 28, 28)
        Returns: (B, n_classes) logits
        """
        x = F.relu(self.conv1(x))            # (B, c1, 24, 24)
        x = F.max_pool2d(x, 2)               # (B, c1, 12, 12)
        x = F.relu(self.conv2(x))            # (B, c2, 8, 8)
        x = F.max_pool2d(x, 2)               # (B, c2, 4, 4)
        x = x.view(x.size(0), -1)            # (B, c2*4*4)
        x = F.relu(self.fc1(x))              # (B, fc1)
        logits = self.fc2(x)                  # (B, 10)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
