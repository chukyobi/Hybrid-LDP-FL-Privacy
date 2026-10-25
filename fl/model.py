"""
fl/model.py
===========
Neural network model for digital footprint privacy classification.

Task: Binary classification → predict privacy-risk level (0=low, 1=high)
from noisy (LDP-privatized) feature vectors.
"""

import torch
import torch.nn as nn
import sys, os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from config import NUM_FEATURES, NUM_CLASSES


class FootprintClassifier(nn.Module):
    """
    A lightweight MLP designed for resource-constrained edge devices.
    Architecture: Input → 64 → 32 → 16 → Output (2 classes)
    Uses LayerNorm and Dropout for regularization under noisy data.
    """

    def __init__(self, input_dim: int = NUM_FEATURES,
                 num_classes: int = NUM_CLASSES):
        super(FootprintClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(64, 32),
            nn.LayerNorm(32),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Linear(16, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def get_model(input_dim: int = NUM_FEATURES) -> FootprintClassifier:
    """Factory function — returns a fresh, untrained model."""
    return FootprintClassifier(input_dim=input_dim)


def count_parameters(model: nn.Module) -> int:
    """Returns total number of trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    model = get_model()
    print(model)
    print(f"Trainable parameters: {count_parameters(model):,}")
