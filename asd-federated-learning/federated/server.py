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
        """Perform federated averaging of full state_dict (FedAvg).

        Uses state_dict keys so that BatchNorm running_mean, running_var,
        and num_batches_tracked buffers are also averaged — not just the
        trainable parameters. Averaging is done in float32 and cast back
        to the original dtype to handle integer buffers (e.g. num_batches_tracked).
        """
        total_samples = sum(client_weights)
        averaged_params = {}

        for name in client_params[0].keys():
            orig_dtype = client_params[0][name].dtype
            # Accumulate weighted average in float32
            avg = torch.zeros_like(client_params[0][name], dtype=torch.float32)
            for client_param, weight in zip(client_params, client_weights):
                avg += client_param[name].float() * (weight / total_samples)
            # Cast back to original dtype (handles int64 num_batches_tracked)
            averaged_params[name] = avg.to(orig_dtype)

        return averaged_params

    def train_round(self, local_epochs: int = 1, lr: float = 0.001, dp=None) -> Dict:
        """Execute one round of federated training.

        Args:
            local_epochs: Number of local training epochs per client.
            lr: Learning rate for local optimizers.
            dp: Optional DifferentialPrivacy instance. When provided, each
                client applies gradient clipping + Gaussian noise after
                loss.backward() on every local step.
        """
        print("Starting training round...")

        # Distribute full global state_dict to all clients (includes BN buffers)
        global_state = {
            k: v.clone().detach()
            for k, v in self.global_model.state_dict().items()
        }

        for client in self.clients:
            client.set_model_params(global_state)

        # Local training at each client
        client_params = []
        client_weights = []
        client_metrics = []

        for client in self.clients:
            print(f"Training client {client.client_id}...")
            metrics = client.local_train(epochs=local_epochs, lr=lr, dp=dp)
            client_params.append(client.get_model_params())
            client_weights.append(metrics["samples"])
            client_metrics.append(metrics)

        # Federated averaging over full state_dict
        averaged_params = self.federated_averaging(client_params, client_weights)

        # Update global model
        self.global_model.load_state_dict(averaged_params)

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

    def evaluate_global_model(
        self, split: str = "test", include_predictions: bool = True
    ) -> Dict:
        """Evaluate global model on all client datasets.

        Args:
            split: Which split to evaluate on: "valid" or "test".
            include_predictions: If False, skip collecting predictions/targets.
        """
        print("Evaluating global model...")

        # Distribute full global state_dict (includes BN buffers)
        global_state = {
            k: v.clone().detach()
            for k, v in self.global_model.state_dict().items()
        }

        all_metrics: List[Dict] = []
        total_samples = 0
        total_correct = 0.0
        all_preds: List = []
        all_targets: List = []

        for client in self.clients:
            client.set_model_params(global_state)
            if split == "valid":
                metrics = client.local_validate(
                    include_predictions=include_predictions
                )
            else:
                metrics = client.local_test(
                    include_predictions=include_predictions
                )
            all_metrics.append(metrics)

            total_samples += metrics["samples"]
            total_correct += (metrics["accuracy"] / 100) * metrics["samples"]
            if include_predictions:
                all_preds.extend(metrics.get("predictions", []))
                all_targets.extend(metrics.get("targets", []))

        global_accuracy = (total_correct / total_samples) * 100 if total_samples > 0 else 0.0

        metrics: Dict[str, object] = {
            "global_accuracy": global_accuracy,
            "client_metrics": all_metrics,
            "total_samples": total_samples,
        }
        if include_predictions:
            metrics["predictions"] = all_preds
            metrics["targets"] = all_targets
        return metrics  # type: ignore[return-value]
