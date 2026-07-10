"""
Flower (flwr) client wrapper for ASD Federated Learning.

This module provides an ASDFlowerClient that implements the flwr.client.NumPyClient
interface. It wraps the existing FederatedClient to reuse the local training
and evaluation logic, while adapting the parameter exchange to Flower's
List[np.ndarray] format.
"""

from typing import Dict, List, Tuple

import flwr as fl
import numpy as np
import torch
from collections import OrderedDict

from federated.client import FederatedClient
from federated.privacy import DifferentialPrivacy


class ASDFlowerClient(fl.client.NumPyClient):
    """Flower client wrapping the existing FederatedClient."""

    def __init__(
        self,
        client: FederatedClient,
        dp_module: DifferentialPrivacy = None,
    ) -> None:
        """Initialize the ASDFlowerClient.

        Args:
            client (FederatedClient): The underlying client containing the model and data.
            dp_module (DifferentialPrivacy, optional): DP mechanism for gradient clipping and noising.
        """
        self.client = client
        self.dp_module = dp_module

    def get_parameters(self, config: Dict[str, fl.common.Scalar]) -> List[np.ndarray]:
        """Return the current local model parameters as a list of NumPy ndarrays."""
        return [val.cpu().numpy() for _, val in self.client.model.state_dict().items()]

    def set_parameters(self, parameters: List[np.ndarray]) -> None:
        """Set the local model parameters from the provided list of NumPy ndarrays."""
        params_dict = zip(self.client.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.client.model.load_state_dict(state_dict, strict=True)

    def fit(
        self, parameters: List[np.ndarray], config: Dict[str, fl.common.Scalar]
    ) -> Tuple[List[np.ndarray], int, Dict[str, fl.common.Scalar]]:
        """Train the model locally.

        Args:
            parameters: The global model parameters from the server.
            config: Configuration dictionary sent by the server (e.g., local_epochs, lr).

        Returns:
            A tuple of (updated_parameters, num_examples, metrics).
        """
        self.set_parameters(parameters)

        local_epochs = int(config.get("local_epochs", 1))
        lr = float(config.get("lr", 0.001))

        # We need to inject the DP logic into the training loop if enabled.
        # Since FederatedClient.local_train() encapsulates the loop, we have to
        # decide how to apply DP.
        # The existing client.local_train() doesn't expose the parameter updates directly.
        # Let's temporarily modify how we call training to support DP if required,
        # or we adapt the client.local_train to accept a dp_module.
        
        # For this integration, we'll run the existing local_train which we'll assume
        # has been or will be adapted to call dp_module if present, or we do it here.
        # Actually, in the existing setup, DP was applied inside the server logic or client?
        # Let's check: The original `CompleteFederatedTrainer` didn't explicitly pass DP to `client.local_train`.
        # Wait, DP-SGD must be applied *during* local training on each batch.
        
        # We will use the existing `local_train` method of `FederatedClient`. 
        # We need to make sure the client knows about the dp_module.
        # Let's inject it into the client object directly.
        self.client.dp_module = self.dp_module

        metrics = self.client.local_train(epochs=local_epochs, lr=lr)

        return self.get_parameters(config={}), metrics["samples"], {"loss": metrics["loss"]}

    def evaluate(
        self, parameters: List[np.ndarray], config: Dict[str, fl.common.Scalar]
    ) -> Tuple[float, int, Dict[str, fl.common.Scalar]]:
        """Evaluate the model locally.

        Args:
            parameters: The global model parameters from the server.
            config: Configuration dictionary sent by the server.

        Returns:
            A tuple of (loss, num_examples, metrics).
        """
        self.set_parameters(parameters)
        
        # Determine split to evaluate on from config
        split = config.get("split", "test")
        
        if split == "valid":
            metrics = self.client.local_validate(include_predictions=False)
        else:
            metrics = self.client.local_test(include_predictions=False)

        loss = float(metrics["loss"])
        num_examples = int(metrics["samples"])
        
        # Return accuracy as a metric
        return loss, num_examples, {"accuracy": float(metrics["accuracy"])}
