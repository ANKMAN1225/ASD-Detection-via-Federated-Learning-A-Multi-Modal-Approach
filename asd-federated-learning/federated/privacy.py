"""
Privacy mechanisms: Differential Privacy and Secure Aggregation utilities.
"""

from typing import Dict, List

import torch
import torch.nn as nn


class DifferentialPrivacy:
    """Differential Privacy implementation for federated learning."""

    def __init__(
        self,
        noise_multiplier: float = 1.0,
        max_grad_norm: float = 1.0,
    ) -> None:
        self.noise_multiplier = noise_multiplier
        self.max_grad_norm = max_grad_norm

    def add_noise_to_gradients(self, model: nn.Module) -> None:
        """Add Gaussian noise to model gradients (gradient clipping + noise)."""
        for param in model.parameters():
            if param.grad is not None:
                # Clip gradients
                torch.nn.utils.clip_grad_norm_([param], self.max_grad_norm)

                # Add noise
                noise = torch.normal(
                    0,
                    self.noise_multiplier * self.max_grad_norm,
                    size=param.grad.shape,
                    device=param.device,
                )
                param.grad += noise

    def add_noise_to_parameters(self, params: Dict) -> Dict:
        """Add noise to model parameters."""
        noisy_params = {}
        for name, param in params.items():
            noise = torch.normal(
                0,
                self.noise_multiplier,
                size=param.shape,
                device=param.device,
            )
            noisy_params[name] = param + noise
        return noisy_params


def secure_aggregate(
    client_params: List[Dict],
    client_weights: List[int],
    device: torch.device,
) -> Dict:
    """
    Secure Aggregation utilities - simulates secure multi-party aggregation.
    In production, this would use cryptographic protocols (e.g., Shamir secret sharing)
    to allow aggregation without revealing individual client updates.
    For now, returns standard weighted average (placeholder for secure implementation).
    """
    total_samples = sum(client_weights)
    aggregated_params = {}

    for name in client_params[0].keys():
        aggregated_params[name] = torch.zeros_like(
            client_params[0][name], device=device
        )

    for client_param, weight in zip(client_params, client_weights):
        weight_ratio = weight / total_samples
        for name in aggregated_params.keys():
            aggregated_params[name] += client_param[name].to(device) * weight_ratio

    return aggregated_params
