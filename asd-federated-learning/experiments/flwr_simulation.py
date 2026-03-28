"""
Flower (flwr) simulation script for parallel federated learning.
Replaces the synchronous sequential loop in experiment_runner.py with real parallel agents.
"""

import os
import torch
import flwr as fl
from typing import Dict, List, Optional, Tuple, Union, Any
from collections import OrderedDict
from torch.utils.data import DataLoader

from federated.privacy import DifferentialPrivacy

class PyTorchFlowerClient(fl.client.NumPyClient):
    def __init__(
        self, 
        model: torch.nn.Module, 
        train_loader: DataLoader, 
        val_loader: DataLoader, 
        device: str, 
        dp: Optional[DifferentialPrivacy] = None, 
        lr: float = 0.001
    ):
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.dp = dp
        self.lr = lr
        
    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in self.model.state_dict().items()]
        
    def set_parameters(self, parameters):
        params_dict = zip(self.model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        self.model.load_state_dict(state_dict, strict=True)
        
    def fit(self, parameters, config):
        self.set_parameters(parameters)
        self.model.train()
        self.model.to(self.device)
        
        epochs = config.get("local_epochs", 1)
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        criterion = torch.nn.CrossEntropyLoss()
        
        total_loss = 0.0
        total_samples = 0
        
        for _ in range(epochs):
            for batch_data, batch_labels in self.train_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                optimizer.zero_grad()
                outputs = self.model(batch_data)
                
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                    
                loss = criterion(outputs, batch_labels)
                loss.backward()
                
                if self.dp is not None:
                    self.dp.add_noise_to_gradients(self.model)
                    
                optimizer.step()
                
                batch_size = batch_data.size(0)
                total_loss += loss.item() * batch_size
                total_samples += batch_size
                
        avg_loss = total_loss / max(total_samples, 1)
        return self.get_parameters(config={}), total_samples, {"loss": avg_loss}
        
    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        self.model.eval()
        self.model.to(self.device)
        
        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch_data, batch_labels in self.val_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                outputs = self.model(batch_data)
                if isinstance(outputs, tuple):
                    outputs = outputs[0]
                    
                loss = criterion(outputs, batch_labels)
                total_loss += loss.item() * batch_data.size(0)
                
                _, predicted = torch.max(outputs.data, 1)
                total_samples += batch_labels.size(0)
                correct += (predicted == batch_labels).sum().item()
                
        avg_loss = float(total_loss / max(total_samples, 1))
        accuracy = float(correct / max(total_samples, 1))
        
        return avg_loss, total_samples, {"accuracy": accuracy}


def run_flwr_simulation(
    global_model: torch.nn.Module,
    client_train_loaders: List[DataLoader],
    client_val_loaders: List[DataLoader],
    num_clients: int,
    num_rounds: int,
    local_epochs: int,
    learning_rate: float,
    device: str,
    dp: Optional[DifferentialPrivacy] = None,
    experiment_name: str = "flwr_experiment",
    config_cpus: int = 1,
    early_stopping_patience: int = 5,
    early_stopping_enabled: bool = True,
    early_stopping_min_delta: float = 0.0,
) -> Tuple[Any, Optional[List[fl.common.NDArray]]]:
    """
    Runs a parallel FL simulation using Flower + Ray.
    Includes early stopping: tracks mean val accuracy each round; halts when
    accuracy does not improve by `early_stopping_min_delta` for
    `early_stopping_patience` consecutive evaluate rounds.
    """
    print(f"\nStarting Flower Simulation for {experiment_name}...")
    print(f"Clients: {num_clients}  |  Max rounds: {num_rounds}  |  Local epochs: {local_epochs}")
    print(f"Early stopping: enabled={early_stopping_enabled}  patience={early_stopping_patience}")

    def client_fn(cid: str) -> fl.client.Client:
        import copy
        client_idx = int(cid)
        client_model = copy.deepcopy(global_model)
        return PyTorchFlowerClient(
            model=client_model,
            train_loader=client_train_loaders[client_idx],
            val_loader=client_val_loaders[client_idx],
            device=device,
            dp=dp,
            lr=learning_rate,
        ).to_client()

    def fit_config(server_round: int) -> Dict[str, Any]:
        return {"local_epochs": local_epochs, "round": server_round}

    # ── Custom strategy with early stopping ─────────────────────────────────
    class EarlyStopStrategy(fl.server.strategy.FedAvg):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.latest_parameters: Optional[fl.common.Parameters] = None
            self.best_parameters: Optional[fl.common.Parameters] = None
            self.best_accuracy: float = float("-inf")
            self.best_round: int = 0
            self.bad_rounds: int = 0
            self._stop: bool = False

        # --- intercept aggregated weights after each fit round ---
        def aggregate_fit(
            self,
            server_round: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes]],
            failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.FitRes], BaseException]],
        ) -> Tuple[Optional[fl.common.Parameters], Dict[str, fl.common.Scalar]]:
            aggregated_parameters, aggregated_metrics = super().aggregate_fit(
                server_round, results, failures
            )
            if aggregated_parameters is not None:
                self.latest_parameters = aggregated_parameters
            return aggregated_parameters, aggregated_metrics

        # --- track val accuracy and apply early stopping logic ---
        def aggregate_evaluate(
            self,
            server_round: int,
            results: List[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes]],
            failures: List[Union[Tuple[fl.server.client_proxy.ClientProxy, fl.common.EvaluateRes], BaseException]],
        ) -> Tuple[Optional[float], Dict[str, fl.common.Scalar]]:
            aggregated_loss, aggregated_metrics = super().aggregate_evaluate(
                server_round, results, failures
            )

            if results:
                # Compute mean accuracy across all clients
                total_samples = sum(r.num_examples for _, r in results)
                if total_samples == 0:
                    print(f"  [WARNING] Round {server_round}: total_samples is 0. Results: {[r.num_examples for _, r in results]}")
                
                weighted_acc = sum(
                    r.metrics.get("accuracy", 0.0) * r.num_examples
                    for _, r in results
                ) / max(total_samples, 1)

                print(
                    f"  [Round {server_round}] Val Accuracy: {weighted_acc:.4f}"
                    f"  |  Best: {self.best_accuracy:.4f} (round {self.best_round})"
                )

                # Inject into aggregated_metrics so it appears in the History object
                if aggregated_metrics is None:
                    aggregated_metrics = {}
                aggregated_metrics["accuracy"] = weighted_acc

                if early_stopping_enabled:
                    improved = weighted_acc > self.best_accuracy + early_stopping_min_delta
                    if improved:
                        self.best_accuracy = weighted_acc
                        self.best_round = server_round
                        self.best_parameters = self.latest_parameters
                        self.bad_rounds = 0
                    else:
                        self.bad_rounds += 1
                        print(
                            f"  [Early stopping] No improvement for {self.bad_rounds}/{early_stopping_patience} rounds"
                        )
                        if self.bad_rounds >= early_stopping_patience:
                            print(
                                f"\n  *** Early stopping triggered at round {server_round}. "
                                f"Best val accuracy: {self.best_accuracy:.4f} at round {self.best_round} ***\n"
                            )
                            self._stop = True
                else:
                    # Track best even without early stopping
                    if weighted_acc > self.best_accuracy:
                        self.best_accuracy = weighted_acc
                        self.best_round = server_round
                        self.best_parameters = self.latest_parameters

            return aggregated_loss, aggregated_metrics

        # --- stop selecting clients once the early-stop flag is raised ---
        def configure_fit(self, server_round, parameters, client_manager):
            if self._stop:
                return []  # returning empty list halts Flower training
            return super().configure_fit(server_round, parameters, client_manager)

        def configure_evaluate(self, server_round, parameters, client_manager):
            if self._stop:
                return []  # halts Flower evaluation
            return super().configure_evaluate(server_round, parameters, client_manager)

    # ── Build and run the strategy ───────────────────────────────────────────
    strategy = EarlyStopStrategy(
        fraction_fit=1.0,
        fraction_evaluate=1.0,
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        on_fit_config_fn=fit_config,
        initial_parameters=fl.common.ndarrays_to_parameters(
            [val.cpu().numpy() for _, val in global_model.state_dict().items()]
        ),
    )

    history = fl.simulation.start_simulation(
        client_fn=client_fn,
        num_clients=num_clients,
        config=fl.server.ServerConfig(num_rounds=num_rounds),
        strategy=strategy,
        client_resources={"num_cpus": config_cpus, "num_gpus": 0.0},
        # NOTE: `num_cpus: 2` limits Ray to running at most 2 clients AT ONCE. 
        # All 5 clients will still participate in the round, just not simultaneously!
        ray_init_args={"num_cpus": max(2, config_cpus), "include_dashboard": False},
    )

    print(f"\nFlower Simulation finished for {experiment_name}.")
    print(
        f"Best val accuracy: {strategy.best_accuracy:.4f} at round {strategy.best_round}"
    )

    # Return the BEST parameters (not just the last-round ones)
    best_ndarrays = None
    chosen_params = strategy.best_parameters or strategy.latest_parameters
    if chosen_params is not None:
        best_ndarrays = fl.common.parameters_to_ndarrays(chosen_params)

    return history, best_ndarrays

