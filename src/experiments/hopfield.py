"""
Modern Hopfield Network for classification (Ramsauer et al., 2020).

The classical energy is:
    E = −lse(β, X^T ξ) + ½ ξ^T ξ + const

The single-step retrieval rule is equivalent to an attention mechanism:
    ξ_new = V · softmax(β · K^T ξ)

where K (stored patterns / keys) and V (associated values) are learned.
Multiple retrieval steps refine the query before classification.

This gives the Hopfield network a comparable expressivity budget to the
cortical column, while using an entirely different computational paradigm
(energy minimisation / associative memory vs. sequential saccade integration).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class ModernHopfieldNetwork(nn.Module):
    """Modern Hopfield Network for classification.

    Architecture:
        1. Linear encoder: input → query ξ ∈ R^d
        2. Hopfield retrieval (repeated n_steps times):
              scores = β · K ξ       (n_patterns,)
              attn   = softmax(scores)
              ξ      = V^T attn       (d,)
        3. Linear classifier: ξ → logits

    Parameters are comparable to the cortical column (~50 K).
    """

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dim: int = 48,
        n_patterns: int = 128,
        n_classes: int = 10,
        beta: float = 8.0,
        n_steps: int = 3,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.beta = beta
        self.n_steps = n_steps

        # Encoder: flatten image → embedding
        self.encoder = nn.Linear(input_dim, hidden_dim)

        # Stored patterns (keys) and associated values
        self.keys = nn.Parameter(torch.randn(n_patterns, hidden_dim) * 0.02)
        self.values = nn.Parameter(torch.randn(n_patterns, hidden_dim) * 0.02)

        # Read-out
        self.classifier = nn.Linear(hidden_dim, n_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_normal_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        nn.init.xavier_normal_(self.classifier.weight)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (B, 1, 28, 28)
        Returns: (B, n_classes) logits
        """
        x = x.view(x.size(0), -1)                                 # (B, 784)
        xi = self.encoder(x)                                       # (B, d)

        for _ in range(self.n_steps):
            # Hopfield retrieval step (attention)
            scores = self.beta * (xi @ self.keys.T)                # (B, n_patterns)
            attn = F.softmax(scores, dim=-1)                       # (B, n_patterns)
            xi = attn @ self.values                                # (B, d)

        logits = self.classifier(xi)                               # (B, n_classes)
        return logits

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
