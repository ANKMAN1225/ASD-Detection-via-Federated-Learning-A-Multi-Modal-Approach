"""
Training utilities for federated learning - optimizer setup, loss functions, local training.
"""

from .trainer import get_optimizer, get_criterion

__all__ = ["get_optimizer", "get_criterion"]
