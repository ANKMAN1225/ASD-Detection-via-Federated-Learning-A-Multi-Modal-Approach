"""
FedAvg aggregation server for federated learning.
"""

from typing import Dict, List

import torch
import torch.nn as nn

from .client import FederatedClient


class FederatedServer:
    """Federated learning server for aggregation."""

    def __init__(self, global_model: nn.Module, clients: List[FederatedClient]) -> None:
        self.global_model = global_model
        self.clients = clients
        self.round_metrics: List[Dict] = []

    def federated_averaging(
        self, client_params: List[Dict], client_weights: List[int]
    ) -> Dict:
        """Perform federated averaging of client parameters (FedAvg)."""
        # Weighted average based on number of samples
        total_samples = sum(client_weights)
        averaged_params = {}

        # Initialize with zeros
        for name in client_params[0].keys():
            averaged_params[name] = torch.zeros_like(client_params[0][name])

        # Weighted sum
        for client_param, weight in zip(client_params, client_weights):
            weight_ratio = weight / total_samples
            for name in averaged_params.keys():
                averaged_params[name] += client_param[name] * weight_ratio

        return averaged_params

    def train_round(self, local_epochs: int = 1, lr: float = 0.001) -> Dict:
        """Execute one round of federated training."""
        print("Starting training round...")

        # Distribute global model to all clients
        global_params = {
            name: param.clone().detach()
            for name, param in self.global_model.named_parameters()
        }

        for client in self.clients:
            client.set_model_params(global_params)

        # Local training at each client
        client_params = []
        client_weights = []
        client_metrics = []

        for client in self.clients:
            print(f"Training client {client.client_id}...")
            metrics = client.local_train(epochs=local_epochs, lr=lr)
            client_params.append(client.get_model_params())
            client_weights.append(metrics["samples"])
            client_metrics.append(metrics)

        # Federated averaging
        averaged_params = self.federated_averaging(client_params, client_weights)

        # Update global model
        with torch.no_grad():
            for name, param in self.global_model.named_parameters():
                param.copy_(averaged_params[name])

        # Aggregate metrics
        total_samples = sum(client_weights)
        avg_loss = (
            sum(m["loss"] * w for m, w in zip(client_metrics, client_weights))
            / total_samples
        )

        round_metrics = {
            "avg_loss": avg_loss,
            "total_samples": total_samples,
            "num_clients": len(self.clients),
        }

        self.round_metrics.append(round_metrics)
        return round_metrics

    def evaluate_global_model(self) -> Dict:
        """Evaluate global model on all client test sets."""
        print("Evaluating global model...")

        # Distribute global model to all clients
        global_params = {
            name: param.clone().detach()
            for name, param in self.global_model.named_parameters()
        }

        all_metrics = []
        total_samples = 0
        total_correct = 0.0
        all_preds = []
        all_targets = []

        for client in self.clients:
            client.set_model_params(global_params)
            metrics = client.local_test()
            all_metrics.append(metrics)

            total_samples += metrics["samples"]
            total_correct += (metrics["accuracy"] / 100) * metrics["samples"]
            all_preds.extend(metrics["predictions"])
            all_targets.extend(metrics["targets"])

        global_accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0.0

        return {
            "global_accuracy": global_accuracy,
            "client_metrics": all_metrics,
            "predictions": all_preds,
            "targets": all_targets,
            "total_samples": total_samples,
        }
