"""Simple multi-layer perceptron used as a building block inside larger models."""

import torch.nn as nn


class MLP(nn.Module):
    """Two-layer MLP with ReLU activation and dropout."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.1):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.layers(x)