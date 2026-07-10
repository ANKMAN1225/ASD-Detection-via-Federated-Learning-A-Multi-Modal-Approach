"""
Flower (flwr) simulation runner.

Replaces the custom federated loop with flwr.simulation.start_simulation.
"""

from typing import Dict, List, Optional
import flwr as fl
import torch

from .flower_client import ASDFlowerClient
from .flower_strategy import ASDFedAvgStrategy, EarlyStoppingException
from federated.client import FederatedClient
from federated.privacy import DifferentialPrivacy


def start_flower_simulation(
    global_model: torch.nn.Module,
    clients: List[FederatedClient],
    config: Dict,
    dp_module: Optional[DifferentialPrivacy] = None,
) -> Dict:
    """Start a Flower simulation on a single machine.

    Args:
        global_model: The global model (PyTorch module).
        clients: List of initialized custom FederatedClients.
        config: Experiment configuration dictionary.
        dp_module: DifferentialPrivacy module (optional).

    Returns:
        Dict containing the simulation results structured to match
        the existing output format.
    """
    print("\n" + "=" * 50)
    print("STARTING FLOWER SIMULATION")
    print("=" * 50)

    num_clients = len(clients)
    num_rounds = config.get("num_rounds", 10)
    local_epochs = config.get("local_epochs", 1)
    lr = config.get("learning_rate", 0.001)

    # Convert initial model weights to list of ndarrays
    initial_parameters = fl.common.ndarrays_to_parameters(
        [val.cpu().numpy() for _, val in global_model.state_dict().items()]
    )

    # We need a client_fn for the simulation
    def client_fn(cid: str) -> fl.client.Client:
        # cid is a string like "0", "1", ...
        client_idx = int(cid)
        client = clients[client_idx]
        flower_client = ASDFlowerClient(client=client, dp_module=dp_module)
        return flower_client.to_client()

    round_metrics: List[Dict] = []

    def fit_config_fn(server_round: int) -> Dict[str, fl.common.Scalar]:
        """Return training configuration dict for each round."""
        return {
            "local_epochs": local_epochs,
            "lr": lr,
            "round": server_round,
        }
        
    def evaluate_config_fn(server_round: int) -> Dict[str, fl.common.Scalar]:
        """Return evaluation configuration dict for each round."""
        # We can specify if we want to run valid or test split
        # We'll use 'test' to match the final evaluation behavior
        return {
            "split": "test",
            "round": server_round,
        }

    # Initialize custom FedAvg Strategy
    strategy = ASDFedAvgStrategy(
        experiment_name=config.get("experiment_name", "flower_sim"),
        round_metrics=round_metrics,
        fraction_fit=1.0,  # Sample 100% of available clients for training
        fraction_evaluate=1.0,  # Sample 100% of available clients for evaluation
        min_fit_clients=num_clients,
        min_evaluate_clients=num_clients,
        min_available_clients=num_clients,
        initial_parameters=initial_parameters,
        on_fit_config_fn=fit_config_fn,
        on_evaluate_config_fn=evaluate_config_fn,
    )
    
    # Pass Early Stopping configs to strategy
    strategy.early_stopping_enabled = config.get("early_stopping_enabled", False)
    strategy.early_stopping_patience = config.get("early_stopping_patience", 5)
    strategy.early_stopping_min_delta = config.get("early_stopping_min_delta", 0.0)

    # Start simulation with custom Ray arguments to avoid memory reservation errors
    ray_init_args = {
        "include_dashboard": False,
        "num_cpus": 2,                             # Limit concurrency to 2 actors to prevent OutOfMemory on 8GB machines
        "object_store_memory": 100 * 1024 * 1024,  # 100 MB
        "_memory": 500 * 1024 * 1024,              # 500 MB limit for actors
    }
    
    # ── CRITICAL FIX FOR WSL/RAY CUDA DESERIALIZATION ──
    # Ray masks GPUs for actors (num_gpus: 0.0), causing torch to crash
    # when it tries to deserialize a CUDA tensor in an environment where
    # torch.cuda.is_available() is False. We must move clients to CPU first.
    for client in clients:
        if hasattr(client, "model"):
            client.model.cpu()
        client.device = torch.device("cpu")

    history = None
    try:
        history = fl.simulation.start_simulation(
            client_fn=client_fn,
            num_clients=num_clients,
            config=fl.server.ServerConfig(num_rounds=num_rounds),
            strategy=strategy,
            client_resources={"num_cpus": 1, "num_gpus": 0.0},
            ray_init_args=ray_init_args,
        )
    except EarlyStoppingException as e:
        print(f"\n{e}")
        # The simulation was stopped early, history will not be fully populated
        pass

    print("\nFlower Simulation Complete.")

    # Convert history into the expected output format for compatibility
    
    # We will assume the best model was the last round for simplicity,
    # as Flower doesn't trivially return the global model at the end.
    # Alternatively, we could save the model inside the strategy.
    # To conform with existing API:
    final_metrics = {
        "global_accuracy": 0.0,
        "total_samples": 0,
        "client_metrics": []
    }
    
    if round_metrics:
        last_round = round_metrics[-1]
        final_metrics["global_accuracy"] = last_round.get("global_accuracy", 0.0)
        final_metrics["total_samples"] = last_round.get("total_samples", 0)

    results = {
        "round_metrics": round_metrics,
        "final_metrics": final_metrics,
        "flower_history": history,
    }

    return results
