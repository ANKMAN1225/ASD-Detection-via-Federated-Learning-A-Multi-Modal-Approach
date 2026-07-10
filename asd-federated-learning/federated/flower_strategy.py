"""
Flower (flwr) server strategy for ASD Federated Learning.

Extends the standard FedAvg strategy to capture and log customized metrics
consistent with the original FederatedServer implementation.
"""

from typing import Callable, Dict, List, Optional, Tuple, Union

import flwr as fl
from flwr.common import (
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    MetricsAggregationFn,
    NDArrays,
    Parameters,
    Scalar,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)
from flwr.server.client_proxy import ClientProxy

import numpy as np


class EarlyStoppingException(Exception):
    """Exception raised to stop Flower simulation early."""
    pass


class ASDFedAvgStrategy(fl.server.strategy.FedAvg):
    """Custom FedAvg strategy to track specific evaluation metrics."""

    def __init__(
        self,
        experiment_name: str,
        round_metrics: List[Dict],
        *args,
        **kwargs,
    ) -> None:
        """Initialize ASDFedAvgStrategy.

        Args:
            experiment_name (str): Name of the current experiment.
            round_metrics (list): A list to append per-round aggregated metrics to.
            *args, **kwargs: Arguments passed to FedAvg.
        """
        super().__init__(*args, **kwargs)
        self.experiment_name = experiment_name
        self.round_metrics = round_metrics
        
        # Early stopping state
        self.best_accuracy = 0.0
        self.patience_counter = 0
        
        # Pull from kwargs if passed (they will be in args/kwargs via init)
        self.early_stopping_enabled = False
        self.early_stopping_patience = 5
        self.early_stopping_min_delta = 0.0

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, EvaluateRes]],
        failures: List[Union[Tuple[ClientProxy, EvaluateRes], BaseException]],
    ) -> Tuple[Optional[float], Dict[str, Scalar]]:
        """Aggregate evaluation losses and accuracies using weighted average."""
        if not results:
            return None, {}

        # Call the standard FedAvg aggregation for loss
        loss_aggregated, metrics_aggregated = super().aggregate_evaluate(
            server_round, results, failures
        )

        # Compute custom metrics like global accuracy and client variance
        total_samples = sum([res.num_examples for _, res in results])
        
        # Weighted accuracy
        accuracies = [res.metrics.get("accuracy", 0.0) for _, res in results]
        weights = [res.num_examples for _, res in results]
        
        weighted_accuracy = sum(
            [acc * w for acc, w in zip(accuracies, weights)]
        ) / total_samples if total_samples > 0 else 0.0

        client_accuracy_variance = float(np.var(accuracies))

        # Build custom aggregated metrics
        if metrics_aggregated is None:
            metrics_aggregated = {}
            
        metrics_aggregated["accuracy"] = float(weighted_accuracy)
        metrics_aggregated["client_accuracy_variance"] = client_accuracy_variance

        print(f"[Round {server_round}] Global Accuracy: {weighted_accuracy:.2f}% (var: {client_accuracy_variance:.2f})")

        # Save to our external metrics list
        self.round_metrics.append({
            "round": server_round,
            "avg_loss": loss_aggregated,
            "total_samples": total_samples,
            "num_clients": len(results),
            "global_accuracy": weighted_accuracy,
        })

        # Early Stopping Logic
        if self.early_stopping_enabled:
            if weighted_accuracy > self.best_accuracy + self.early_stopping_min_delta:
                self.best_accuracy = weighted_accuracy
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                print(f"[Early Stopping] Patience: {self.patience_counter}/{self.early_stopping_patience}")
                if self.patience_counter >= self.early_stopping_patience:
                    print(f"\n*** Early Stopping Triggered! No improvement for {self.early_stopping_patience} rounds. ***")
                    raise EarlyStoppingException("Early stopping criteria met.")

        return loss_aggregated, metrics_aggregated
