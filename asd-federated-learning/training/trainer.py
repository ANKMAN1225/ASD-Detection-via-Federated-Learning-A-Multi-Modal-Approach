"""
Local training loop, loss, and optimizer setup for federated clients.
"""

from typing import Optional

import torch
import torch.nn as nn


def get_optimizer(
    model: nn.Module,
    lr: float = 0.001,
    optimizer_cls: type = torch.optim.Adam,
) -> torch.optim.Optimizer:
    """Create optimizer for model parameters."""
    return optimizer_cls(model.parameters(), lr=lr)


def get_criterion() -> nn.Module:
    """Get cross-entropy loss for classification."""
    return nn.CrossEntropyLoss()
