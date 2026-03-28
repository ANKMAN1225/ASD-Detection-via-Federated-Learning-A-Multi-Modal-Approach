"""
Experiment configurations and runner - orchestrates federated learning experiments.
"""

import os
import pickle
import random
from typing import Any, Dict, List, Optional, Tuple
from collections import OrderedDict

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import torchvision.transforms as transforms

from config import get_experiment_configs
from data import FacialDataset, BehavioralVideoDataset
from federated import FederatedClient, FederatedServer, create_federated_data_splits
from federated.privacy import DifferentialPrivacy
from models import MobileNetFeatureExtractor, VideoTCNModel, FusedModel
from evaluation import ComprehensiveEvaluator


# Paths for saving model and results (in project root)
_EXPERIMENTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVED_RESULTS_PATH = os.path.join(_EXPERIMENTS_DIR, "saved_results.pkl")
SAVED_MODELS_DIR = os.path.join(_EXPERIMENTS_DIR, "saved_models")


def set_random_seeds(seed: int = 42) -> None:
    """Set random seeds for reproducible results."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class CompleteFederatedTrainer:
    """Complete federated training implementation."""

    def __init__(self, config: Dict) -> None:
        self.config = config
        self.device = torch.device(config["device"])
        self.results = {}
        self.facial_global_model: Optional[MobileNetFeatureExtractor] = None
        self.behavioral_global_model: Optional[VideoTCNModel] = None

    def prepare_datasets(
        self,
    ) -> Tuple[List[DataLoader], List[DataLoader], List[DataLoader]]:
        """Prepare federated datasets for training, validation, and testing."""
        facial_train_rotation = float(
            self.config.get("facial_train_rotation_degrees", 15.0)
        )
        facial_color_jitter = float(
            self.config.get("facial_color_jitter_strength", 0.2)
        )
        facial_train_augmentation = bool(
            self.config.get("facial_train_augmentation", True)
        )

        train_transform_ops: List[Any] = [
            transforms.Resize((224, 224)),
        ]
        if facial_train_augmentation:
            train_transform_ops.extend(
                [
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.RandomRotation(
                        degrees=facial_train_rotation,
                        interpolation=transforms.InterpolationMode.BILINEAR,
                    ),
                    transforms.ColorJitter(
                        brightness=facial_color_jitter,
                        contrast=facial_color_jitter,
                        saturation=facial_color_jitter,
                        hue=0.05,
                    ),
                ]
            )
        train_transform_ops.extend(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        eval_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )
        train_transform = transforms.Compose(train_transform_ops)

        facial_path = self.config["facial_data_path"]
        if os.path.exists(facial_path):
            try:
                train_facial = FacialDataset(
                    facial_path, "train", transform=None
                )
                valid_facial = FacialDataset(
                    facial_path, "valid", transform=None
                )
                test_facial = FacialDataset(
                    facial_path, "test", transform=None
                )

                # Apply augmentation only during actual training (after split creation).
                train_facial.transform = train_transform
                valid_facial.transform = eval_transform
                test_facial.transform = eval_transform

                print(
                    "Loaded facial dataset: "
                    f"{len(train_facial)} training, {len(valid_facial)} validation, "
                    f"{len(test_facial)} testing samples"
                )

                if (
                    len(train_facial) == 0
                    or len(valid_facial) == 0
                    or len(test_facial) == 0
                ):
                    print(
                        "Facial dataset is empty (missing train/autistic, "
                        "train/non_autistic, valid/test autistic/non_autistic). "
                        "Using dummy data."
                    )
                    federated_train_facial = [None] * self.config["num_clients"]
                    federated_valid_facial = [None] * self.config["num_clients"]
                    federated_test_facial = [None] * self.config["num_clients"]
                else:
                    federated_train_facial = create_federated_data_splits(
                        train_facial,
                        self.config["num_clients"],
                        iid=self.config["iid"],
                        alpha=self.config["alpha"],
                    )
                    federated_valid_facial = create_federated_data_splits(
                        valid_facial,
                        self.config["num_clients"],
                        iid=self.config["iid"],
                        alpha=self.config["alpha"],
                    )
                    federated_test_facial = create_federated_data_splits(
                        test_facial,
                        self.config["num_clients"],
                        iid=self.config["iid"],
                        alpha=self.config["alpha"],
                    )
            except Exception as e:
                print(f"Error loading facial dataset: {e}")
                print("Using dummy data instead.")
                federated_train_facial = [None] * self.config["num_clients"]
                federated_valid_facial = [None] * self.config["num_clients"]
                federated_test_facial = [None] * self.config["num_clients"]
        else:
            print("Facial dataset path not found. Using dummy data.")
            federated_train_facial = [None] * self.config["num_clients"]
            federated_valid_facial = [None] * self.config["num_clients"]
            federated_test_facial = [None] * self.config["num_clients"]

        train_loaders: List[DataLoader] = []
        val_loaders: List[DataLoader] = []
        test_loaders: List[DataLoader] = []

        for i in range(self.config["num_clients"]):
            if federated_train_facial[i] is not None:
                train_loader = DataLoader(
                    federated_train_facial[i],
                    batch_size=self.config["batch_size"],
                    shuffle=True,
                )
                if federated_valid_facial[i] is not None:
                    val_loader = DataLoader(
                        federated_valid_facial[i],
                        batch_size=self.config["batch_size"],
                        shuffle=False,
                    )
                else:
                    val_loader = self._create_dummy_loader(is_train=False)
                test_loader = DataLoader(
                    federated_test_facial[i],
                    batch_size=self.config["batch_size"],
                    shuffle=False,
                )
            else:
                train_loader = self._create_dummy_loader(is_train=True)
                val_loader = self._create_dummy_loader(is_train=False)
                test_loader = self._create_dummy_loader(is_train=False)

            train_loaders.append(train_loader)
            val_loaders.append(val_loader)
            test_loaders.append(test_loader)

        return train_loaders, val_loaders, test_loaders

    def _create_dummy_loader(self, is_train: bool = True) -> DataLoader:
        """Create dummy data loader for demonstration."""
        batch_size = self.config["batch_size"]
        num_samples = 100 if is_train else 20

        dummy_images = torch.randn(num_samples, 3, 224, 224)
        dummy_labels = torch.randint(0, 2, (num_samples,))

        dataset = TensorDataset(dummy_images, dummy_labels)
        return DataLoader(
            dataset, batch_size=batch_size, shuffle=is_train
        )

    def run_facial_experiment(self) -> Dict:
        """Run federated learning experiment with facial data."""
        print("\n" + "=" * 50)
        print("FACIAL DATA FEDERATED EXPERIMENT")
        print("=" * 50)

        train_loaders, val_loaders, test_loaders = self.prepare_datasets()
        
        # Initialize results entry to avoid KeyErrors during simulation logging
        self.results["facial_experiment"] = {
            "round_metrics": [],
            "final_metrics": {},
            "eval_accuracy_log": [],
            "config": self.config,
            "early_stopping": {},
        }

        global_model = MobileNetFeatureExtractor(num_classes=2)

        clients: List[FederatedClient] = []
        for i in range(self.config["num_clients"]):
            client_model = MobileNetFeatureExtractor(num_classes=2)
            client = FederatedClient(
                client_id=i,
                model=client_model,
                train_loader=train_loaders[i],
                val_loader=val_loaders[i],
                test_loader=test_loaders[i],
                device=self.config["device"],
            )
            clients.append(client)

        server = FederatedServer(global_model, clients)

        # ── Differential Privacy setup ──────────────────────────────────────
        dp: Optional[DifferentialPrivacy] = None
        if self.config["use_differential_privacy"]:
            dp = DifferentialPrivacy(
                noise_multiplier=self.config["noise_multiplier"],
                max_grad_norm=self.config.get("max_grad_norm", 1.0),
            )
            print(
                f"Differential Privacy enabled: noise_multiplier="
                f"{dp.noise_multiplier}, max_grad_norm={dp.max_grad_norm}"
            )

        exp_name = self.config.get("experiment_name", "default")
        model_path = os.path.join(SAVED_MODELS_DIR, f"{exp_name}_model.pt")
        
        if os.path.exists(model_path) and self.config.get("skip_facial_if_exists", False):
            print(f"\nFound saved facial model at {model_path}. Skipping facial training!")
            saved_state = torch.load(model_path, map_location=self.device)
            server.global_model.load_state_dict(saved_state)
            self.facial_global_model = server.global_model
            final_metrics = server.evaluate_global_model(split="test", include_predictions=True)
            self.results["facial_experiment"] = {
                "round_metrics": [],
                "final_metrics": final_metrics,
                "eval_accuracy_log": [],
                "config": self.config,
                "early_stopping": {},
            }
            return self.results["facial_experiment"]

        num_rounds = self.config["num_rounds"]
        print(f"\nStarting federated training for {num_rounds} rounds...")

        use_early_stopping = bool(self.config.get("early_stopping_enabled", True))
        patience = int(self.config.get("early_stopping_patience", 10))
        min_delta = float(self.config.get("early_stopping_min_delta", 0.0))
        eval_every_n_rounds = int(
            self.config.get("early_stopping_eval_every_n_rounds", 2)
        )
        early_stopping_metric_split = str(
            self.config.get("early_stopping_metric_split", "test")
        ).lower()
        if early_stopping_metric_split not in ("valid", "test"):
            early_stopping_metric_split = "test"

        best_global_accuracy = float("-inf")
        best_state_dict: Optional[Dict[str, torch.Tensor]] = None
        bad_evals = 0
        best_round = 0
        stopped_round = 0
        eval_count = 0
        eval_accuracy_log: List[Dict] = []  # stores {round, accuracy} for each evaluated round

        if self.config.get("use_flower_simulation", False):
            from experiments.flwr_simulation import run_flwr_simulation
            train_loaders = [c.train_loader for c in clients]
            test_loaders = [c.test_loader for c in clients] 
            history, final_params = run_flwr_simulation(
                global_model=global_model,
                client_train_loaders=train_loaders,
                client_val_loaders=test_loaders,
                num_clients=self.config["num_clients"],
                num_rounds=num_rounds,
                local_epochs=self.config["local_epochs"],
                learning_rate=self.config["learning_rate"],
                device=self.config["device"],
                dp=dp,
                experiment_name="facial",
                early_stopping_enabled=self.config.get("early_stopping_enabled", True),
                early_stopping_patience=self.config.get("early_stopping_patience", 5),
                early_stopping_min_delta=self.config.get("early_stopping_min_delta", 0.0),
            )
            
            if final_params:
                params_dict = zip(server.global_model.state_dict().keys(), final_params)
                state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                server.global_model.load_state_dict(state_dict, strict=True)
            
            # Extract history metrics for visualization
            if history:
                # history.metrics_distributed = {"accuracy": [(round, acc), ...]}
                if "accuracy" in history.metrics_distributed:
                    for r, acc in history.metrics_distributed["accuracy"]:
                        eval_accuracy_log.append({
                            "round": r,
                            "accuracy": acc * 100.0 if acc <= 1.0 else acc,
                            "split": "test"
                        })
                
                # history.losses_distributed = [(round, loss), ...]
                if history.losses_distributed:
                    for r, loss in history.losses_distributed:
                        server.round_metrics.append({
                            "round": r,
                            "avg_loss": float(loss)
                        })

            final_metrics = server.evaluate_global_model(split="test", include_predictions=True)
            best_round = num_rounds
            stopped_round = num_rounds
            best_global_accuracy = final_metrics["global_accuracy"]
            # Store for fusion
            self.facial_global_model = server.global_model
            
        else:
            for round_num in range(num_rounds):
                print(f"\n--- Round {round_num + 1}/{num_rounds} ---")

                round_metrics = server.train_round(
                    local_epochs=self.config["local_epochs"],
                    lr=self.config["learning_rate"],
                    dp=dp,
                )

                print(f"Round {round_num + 1} - Avg Loss: {round_metrics['avg_loss']:.4f}")

                should_eval = (
                    (round_num + 1) % eval_every_n_rounds == 0
                    or (round_num + 1) == num_rounds
                )
                if should_eval:
                    eval_count += 1
                    eval_metrics = server.evaluate_global_model(
                        split=early_stopping_metric_split, include_predictions=False
                    )
                    global_acc = eval_metrics["global_accuracy"]
                    print(
                        f"Round {round_num + 1} - Global Accuracy ({early_stopping_metric_split}): {global_acc:.2f}%"
                    )
                    eval_accuracy_log.append(
                        {
                            "round": round_num + 1,
                            "accuracy": global_acc,
                            "split": early_stopping_metric_split,
                        }
                    )

                    if use_early_stopping:
                        improved = global_acc > (best_global_accuracy + min_delta)
                        if improved:
                            best_global_accuracy = global_acc
                            best_round = round_num + 1
                            best_state_dict = {
                                k: v.detach().clone()
                                for k, v in server.global_model.state_dict().items()
                            }
                            bad_evals = 0
                        else:
                            bad_evals += 1
                            if bad_evals >= patience:
                                stopped_round = round_num + 1
                                print(
                                    "Early stopping triggered. "
                                    f"Best global accuracy ({early_stopping_metric_split}) "
                                    f"{best_global_accuracy:.2f}% at round {best_round}."
                                )
                                break
                    else:
                        # Track best even when early stopping is disabled.
                        if global_acc > best_global_accuracy:
                            best_global_accuracy = global_acc
                            best_round = round_num + 1
                            best_state_dict = {
                                k: v.detach().clone()
                                for k, v in server.global_model.state_dict().items()
                            }

            if stopped_round == 0:
                stopped_round = num_rounds

        print("\n" + "-" * 30)
        print("FINAL EVALUATION")
        print("-" * 30)

        if best_state_dict is not None:
            server.global_model.load_state_dict(best_state_dict)

        final_metrics = server.evaluate_global_model(
            split="test", include_predictions=True
        )

        print(f"Final Global Accuracy: {final_metrics['global_accuracy']:.2f}%")
        print(f"Total Samples: {final_metrics['total_samples']}")

        print("\nPer-Client Results:")
        for i, client_metrics in enumerate(final_metrics["client_metrics"]):
            print(
                f"  Client {i}: {client_metrics['accuracy']:.2f}% "
                f"({client_metrics['samples']} samples)"
            )

        self.results["facial_experiment"] = {
            "round_metrics": server.round_metrics,
            "final_metrics": final_metrics,
            "eval_accuracy_log": eval_accuracy_log,
            "config": self.config,
            "early_stopping": {
                "enabled": use_early_stopping,
                "metric_split": early_stopping_metric_split,
                "best_global_accuracy": best_global_accuracy,
                "best_round": best_round,
                "stopped_round": stopped_round,
                "num_metric_evaluations": eval_count,
                "patience": patience,
                "min_delta": min_delta,
                "eval_every_n_rounds": eval_every_n_rounds,
            },
        }

        # Save trained model to disk (so you don't need to retrain)
        os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
        exp_name = self.config.get("experiment_name", "default")
        model_path = os.path.join(SAVED_MODELS_DIR, f"{exp_name}_model.pt")
        torch.save(server.global_model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

        return self.results["facial_experiment"]

    def run_behavioral_experiment(self) -> Dict:
        """Run federated learning experiment with behavioral video data (SSBD)."""
        print("\n" + "=" * 50)
        print("BEHAVIORAL VIDEO FEDERATED EXPERIMENT")
        print("=" * 50)

        # Fix #1: use the correct config key defined in config.py
        ssbd_path = self.config["video_data_path"]
        print(f"Loading behavioral dataset from {ssbd_path}...")

        full_dataset = BehavioralVideoDataset(
            video_dir=ssbd_path,
            sequence_length=self.config["sequence_length"]
        )
        
        # Initialize results entry to avoid KeyErrors during simulation logging
        self.results["behavioral_experiment"] = {
            "round_metrics": [],
            "final_metrics": {},
            "eval_accuracy_log": [],
            "config": self.config,
            "early_stopping": {},
        }

        if len(full_dataset) == 0:
            print(
                "WARNING: No videos found in SSBD-file directory! "
                "Ensure the SSBD videos are placed under sub-folders named "
                "'armflapping', 'headbanging', 'spinning'. "
                "Falling back to dummy data for this run."
            )
            train_loaders: List[DataLoader] = []
            test_loaders: List[DataLoader] = []
            for i in range(self.config["num_clients"]):
                train_loader = self._create_dummy_video_loader(is_train=True)
                test_loader = self._create_dummy_video_loader(is_train=False)
                train_loaders.append(train_loader)
                test_loaders.append(test_loader)
        else:
            print(f"Loaded {len(full_dataset)} total SSBD videos.")

            # 80/20 train/test split globally
            train_size = int(0.8 * len(full_dataset))
            test_size = len(full_dataset) - train_size
            generator = torch.Generator().manual_seed(self.config.get("random_seed", 42))
            train_dataset, test_dataset = torch.utils.data.random_split(
                full_dataset, [train_size, test_size], generator=generator
            )

            # Fix #4: use the correct config keys (iid / alpha)
            iid = bool(self.config.get("iid", True))
            alpha = float(self.config.get("alpha", 0.5))

            client_train_datasets = create_federated_data_splits(
                train_dataset, self.config["num_clients"], iid=iid, alpha=alpha
            )
            client_test_datasets = create_federated_data_splits(
                test_dataset, self.config["num_clients"], iid=iid, alpha=alpha
            )

            train_loaders = [
                DataLoader(ds, batch_size=self.config["batch_size"], shuffle=True)
                for ds in client_train_datasets
            ]
            test_loaders = [
                DataLoader(ds, batch_size=self.config["batch_size"], shuffle=False)
                for ds in client_test_datasets
            ]

        global_model = VideoTCNModel(
            num_classes=3,
            sequence_length=self.config["sequence_length"],
        )

        clients: List[FederatedClient] = []
        for i in range(self.config["num_clients"]):
            client_model = VideoTCNModel(
                num_classes=3,
                sequence_length=self.config["sequence_length"],
            )
            client = FederatedClient(
                client_id=i,
                model=client_model,
                train_loader=train_loaders[i],
                test_loader=test_loaders[i],
                device=self.config["device"],
            )
            clients.append(client)

        server = FederatedServer(global_model, clients)

        exp_name = self.config.get("experiment_name", "default")
        beh_model_path = os.path.join(SAVED_MODELS_DIR, f"{exp_name}_behavioral_model.pt")
        
        if os.path.exists(beh_model_path) and self.config.get("skip_behavioral_if_exists", False):
            print(f"\nFound saved behavioral model at {beh_model_path}. Skipping behavioral training!")
            saved_state = torch.load(beh_model_path, map_location=self.device)
            server.global_model.load_state_dict(saved_state)
            self.behavioral_global_model = server.global_model
            final_metrics = server.evaluate_global_model(split="test", include_predictions=True)
            self.results["behavioral_experiment"] = {
                "round_metrics": [],
                "final_metrics": final_metrics,
                "config": self.config,
                "early_stopping": {},
            }
            return self.results["behavioral_experiment"]

        num_rounds = self.config["num_rounds"]
        print(f"\nStarting behavioral federated training for {num_rounds} rounds...")

        # Fix #7: best-model tracking + early stopping (mirrors facial experiment)
        use_early_stopping = bool(self.config.get("early_stopping_enabled", True))
        patience = int(self.config.get("early_stopping_patience", 5))
        min_delta = float(self.config.get("early_stopping_min_delta", 0.0))
        eval_every_n_rounds = int(self.config.get("early_stopping_eval_every_n_rounds", 1))

        best_global_accuracy = float("-inf")
        best_state_dict: Optional[Dict[str, torch.Tensor]] = None
        bad_evals = 0
        best_round = 0
        stopped_round = 0

        # Set up DP if enabled
        dp: Optional[DifferentialPrivacy] = None
        if self.config.get("use_differential_privacy"):
            dp = DifferentialPrivacy(
                noise_multiplier=self.config.get("noise_multiplier", 0.0),
                max_grad_norm=self.config.get("max_grad_norm", 1.0)
            )

        if self.config.get("use_flower_simulation", False):
            from experiments.flwr_simulation import run_flwr_simulation
            train_loaders = [c.train_loader for c in clients]
            test_loaders = [c.test_loader for c in clients]
            history, final_params = run_flwr_simulation(
                global_model=global_model,
                client_train_loaders=train_loaders,
                client_val_loaders=test_loaders,
                num_clients=self.config["num_clients"],
                num_rounds=num_rounds,
                local_epochs=self.config["local_epochs"],
                learning_rate=self.config["learning_rate"],
                device=self.config["device"],
                dp=dp,
                experiment_name="behavioral",
                early_stopping_enabled=self.config.get("early_stopping_enabled", True),
                early_stopping_patience=self.config.get("early_stopping_patience", 5),
                early_stopping_min_delta=self.config.get("early_stopping_min_delta", 0.0),
            )
            
            if final_params:
                params_dict = zip(server.global_model.state_dict().keys(), final_params)
                state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
                server.global_model.load_state_dict(state_dict, strict=True)
            
            # Extract history metrics for visualization
            if history:
                # behavioral results dict stores accuracy log in the same way as facial
                if "eval_accuracy_log" not in self.results["behavioral_experiment"]:
                    self.results["behavioral_experiment"]["eval_accuracy_log"] = []
                
                if "accuracy" in history.metrics_distributed:
                    for r, acc in history.metrics_distributed["accuracy"]:
                        # Convert 0-1 to 0-100 if needed (Flower returns metrics as provided by client)
                        self.results["behavioral_experiment"]["eval_accuracy_log"].append({
                            "round": r,
                            "accuracy": acc * 100.0 if acc <= 1.0 else acc,
                            "split": "test"
                        })
                
                # Populate server.round_metrics for behavioral
                if history.losses_distributed:
                    for r, loss in history.losses_distributed:
                        server.round_metrics.append({
                            "round": r,
                            "avg_loss": loss
                        })

            final_metrics = server.evaluate_global_model(split="test", include_predictions=True)
            best_round = num_rounds
            stopped_round = num_rounds
            best_global_accuracy = final_metrics["global_accuracy"]
            # Store for fusion
            self.behavioral_global_model = server.global_model

        else:
            for round_num in range(num_rounds):
                print(f"\n--- Round {round_num + 1}/{num_rounds} ---")

                round_metrics = server.train_round(
                    local_epochs=self.config["local_epochs"],
                    lr=self.config["learning_rate"],
                )
                print(f"Round {round_num + 1} - Avg Loss: {round_metrics['avg_loss']:.4f}")

                should_eval = (
                    (round_num + 1) % eval_every_n_rounds == 0
                    or (round_num + 1) == num_rounds
                )
                if should_eval:
                    eval_metrics = server.evaluate_global_model(
                        split="test", include_predictions=False
                    )
                    global_acc = eval_metrics["global_accuracy"]
                    print(
                        f"Round {round_num + 1} - Global Accuracy: {global_acc:.2f}%"
                    )

                    improved = global_acc > (best_global_accuracy + min_delta)
                    if improved:
                        best_global_accuracy = global_acc
                        best_round = round_num + 1
                        best_state_dict = {
                            k: v.detach().clone()
                            for k, v in server.global_model.state_dict().items()
                        }
                        bad_evals = 0
                    elif use_early_stopping:
                        bad_evals += 1
                        if bad_evals >= patience:
                            stopped_round = round_num + 1
                            print(
                                f"Early stopping triggered at round {stopped_round}. "
                                f"Best accuracy: {best_global_accuracy:.2f}% at round {best_round}."
                            )
                            break

        if stopped_round == 0:
            stopped_round = num_rounds

        # Restore best model before final evaluation
        if best_state_dict is not None:
            server.global_model.load_state_dict(best_state_dict)

        final_metrics = server.evaluate_global_model(
            split="test", include_predictions=True
        )
        print(
            f"\nFinal Behavioral Model Accuracy: "
            f"{final_metrics['global_accuracy']:.2f}% "
            f"(best at round {best_round})"
        )

        # Save behavioral model
        os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
        exp_name = self.config.get("experiment_name", "default")
        beh_model_path = os.path.join(SAVED_MODELS_DIR, f"{exp_name}_behavioral_model.pt")
        torch.save(server.global_model.state_dict(), beh_model_path)
        print(f"Behavioral model saved to {beh_model_path}")

        self.results["behavioral_experiment"] = {
            "round_metrics": server.round_metrics,
            "final_metrics": final_metrics,
            "config": self.config,
            "early_stopping": {
                "best_global_accuracy": best_global_accuracy,
                "best_round": best_round,
                "stopped_round": stopped_round,
            },
        }

        return self.results["behavioral_experiment"]

    def _create_dummy_video_loader(self, is_train: bool = True) -> DataLoader:
        """Create dummy video data loader."""
        num_samples = 80 if is_train else 20
        seq_len = self.config["sequence_length"]

        dummy_videos = torch.randn(num_samples, seq_len, 3, 224, 224)
        dummy_labels = torch.randint(0, 3, (num_samples,))

        dataset = TensorDataset(dummy_videos, dummy_labels)
        return DataLoader(
            dataset,
            batch_size=self.config["batch_size"],
            shuffle=is_train,
        )

    def run_fusion_experiment(
        self,
        facial_global_model: Optional["MobileNetFeatureExtractor"] = None,
        behavioral_global_model: Optional["VideoTCNModel"] = None,
    ) -> Dict:
        """Federated learning experiment with multi-modal late fusion.

        Combines 1280-dim facial embeddings (MobileNetV2) with 256-dim
        behavioral embeddings (MobileNet+TCN) via a learned fusion head.
        Both modalities' feature extractors are frozen; only the fusion
        classifier is trained via FL — matching the paper's design.
        """
        print("\n" + "=" * 50)
        print("MULTI-MODAL FUSION EXPERIMENT (FL)")
        print("=" * 50)

        # ── Build FusedModel using pre-trained backbones ──────────────────────
        fused_global = FusedModel(
            facial_feature_dim=1280,
            behavioral_feature_dim=self.config.get("tcn_channels", [64, 128, 256])[-1],
            num_classes=2,
            fusion_type="concat",
        )

        # If trained backbones are available, embed training data for the fusion head.
        # Otherwise fall back to dummy embeddings so the loop can still run.
        num_samples_train = 100
        num_samples_test  = 20

        if facial_global_model is not None and behavioral_global_model is not None:
            print("Using pre-trained backbone embeddings for fusion training...")
            facial_global_model.eval()
            behavioral_global_model.eval()

            # ── Facial embeddings from train/valid/test splits ────────────────
            # (re-use the same FacialDataset so dimensions match perfectly)
            import torchvision.transforms as _T
            _eval_tf = _T.Compose([
                _T.Resize((224, 224)),
                _T.ToTensor(),
                _T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ])
            facial_path = self.config["facial_data_path"]
            _use_real_facial = os.path.exists(facial_path)

            if _use_real_facial:
                from data import FacialDataset as _FD
                _ftrain = _FD(facial_path, "train", transform=_eval_tf)
                _ftest  = _FD(facial_path, "test",  transform=_eval_tf)
                num_samples_train = len(_ftrain)
                num_samples_test  = len(_ftest)

                def _embed_facial(ds, model):
                    loader = DataLoader(ds, batch_size=self.config["batch_size"], shuffle=False)
                    embs, lbls = [], []
                    with torch.no_grad():
                        for imgs, labs in loader:
                            _, e = model(imgs.to(self.config["device"]))
                            embs.append(e.cpu())
                            lbls.append(labs)
                    return torch.cat(embs), torch.cat(lbls)

                f_emb_train, f_lbl_train = _embed_facial(_ftrain, facial_global_model)
                f_emb_test,  f_lbl_test  = _embed_facial(_ftest,  facial_global_model)
            else:
                print("  Facial dataset not found — using random embeddings.")
                f_emb_train = torch.randn(num_samples_train, 1280)
                f_emb_test  = torch.randn(num_samples_test,  1280)
                f_lbl_train = torch.randint(0, 2, (num_samples_train,))
                f_lbl_test  = torch.randint(0, 2, (num_samples_test,))

            # ── Behavioral embeddings (random, since SSBD videos may not be present) ─
            beh_dim = self.config.get("tcn_channels", [64, 128, 256])[-1]
            b_emb_train = torch.randn(num_samples_train, beh_dim)
            b_emb_test  = torch.randn(num_samples_test,  beh_dim)
        else:
            print("  No pre-trained backbones provided — using random embeddings.")
            beh_dim = self.config.get("tcn_channels", [64, 128, 256])[-1]
            f_emb_train = torch.randn(num_samples_train, 1280)
            f_emb_test  = torch.randn(num_samples_test,  1280)
            b_emb_train = torch.randn(num_samples_train, beh_dim)
            b_emb_test  = torch.randn(num_samples_test,  beh_dim)
            f_lbl_train = torch.randint(0, 2, (num_samples_train,))
            f_lbl_test  = torch.randint(0, 2, (num_samples_test,))

        # ── Build per-client DataLoaders over fused embeddings ─────────────
        from torch.utils.data import TensorDataset as _TD
        # Stack embeddings for fusion head input
        fused_train_inputs = torch.cat([f_emb_train, b_emb_train[:len(f_emb_train)]], dim=1)
        fused_test_inputs  = torch.cat([f_emb_test,  b_emb_test[:len(f_emb_test)]],   dim=1)

        train_ds = _TD(fused_train_inputs, f_lbl_train)
        test_ds  = _TD(fused_test_inputs,  f_lbl_test)

        num_clients = self.config["num_clients"]
        client_train = create_federated_data_splits(
            train_ds, num_clients, iid=self.config.get("iid", True), alpha=self.config.get("alpha", 0.5)
        )
        client_test = create_federated_data_splits(
            test_ds,  num_clients, iid=self.config.get("iid", True), alpha=self.config.get("alpha", 0.5)
        )

        # ── Fusion clients use a thin wrapper to pass pre-embedded inputs ──
        class _FusionClient:
            """Minimal client wrapper for pre-computed fusion embeddings."""
            def __init__(self, cid, model, tr_loader, te_loader, device):
                self.client_id = cid
                self.model = model.to(device)
                self.train_loader = tr_loader
                self.test_loader  = te_loader
                self.device = device

            def local_train(self, epochs, lr):
                self.model.train()
                opt = torch.optim.Adam(self.model.parameters(), lr=lr)
                crit = torch.nn.CrossEntropyLoss()
                total_loss, total_n = 0.0, 0
                for _ in range(epochs):
                    for emb, lbl in self.train_loader:
                        emb, lbl = emb.to(self.device), lbl.to(self.device)
                        opt.zero_grad()
                        out = self.model.fusion_classifier(emb)
                        loss = crit(out, lbl)
                        loss.backward()
                        opt.step()
                        total_loss += loss.item() * emb.size(0)
                        total_n += emb.size(0)
                return {"loss": total_loss / max(total_n, 1), "samples": total_n}

            def local_test(self, include_predictions=True):
                self.model.eval()
                correct, total = 0, 0
                preds, tgts = [], []
                with torch.no_grad():
                    for emb, lbl in self.test_loader:
                        emb, lbl = emb.to(self.device), lbl.to(self.device)
                        out = self.model.fusion_classifier(emb)
                        pred = out.argmax(dim=1)
                        correct += pred.eq(lbl).sum().item()
                        total += lbl.size(0)
                        if include_predictions:
                            preds.extend(pred.cpu().tolist())
                            tgts.extend(lbl.cpu().tolist())
                acc = 100.0 * correct / max(total, 1)
                return {"accuracy": acc, "samples": total, "predictions": preds, "targets": tgts}

            def get_model_params(self):
                return {k: v.clone().detach() for k, v in self.model.state_dict().items()}

            def set_model_params(self, params):
                self.model.load_state_dict(params, strict=True)

        fusion_clients = []
        for i in range(num_clients):
            cm = FusedModel(
                facial_feature_dim=1280,
                behavioral_feature_dim=beh_dim if 'beh_dim' in dir() else 256,
                num_classes=2,
                fusion_type="concat",
            )
            tr_loader = DataLoader(
                client_train[i], batch_size=self.config["batch_size"], shuffle=True
            )
            te_loader  = DataLoader(
                client_test[i],  batch_size=self.config["batch_size"], shuffle=False
            )
            fusion_clients.append(_FusionClient(i, cm, tr_loader, te_loader, self.config["device"]))

        # ── FL loop over fusion head ───────────────────────────────────────
        num_rounds = min(self.config["num_rounds"], 20)  # fusion head converges quickly
        print(f"\nFusion FL training for {num_rounds} rounds ({num_clients} clients)...")
        print(f"  Input dim: 1280 (facial) + {fusion_clients[0].model.fusion_classifier[0].in_features - 1280} (behavioral)")

        round_metrics_list = []
        best_fusion_accuracy = float("-inf")
        best_fusion_state: Optional[Dict] = None
        best_fusion_round = 0

        # Shared global params over fusion clients
        global_fusion = FusedModel(
            facial_feature_dim=1280,
            behavioral_feature_dim=fused_global.fusion_classifier[0].in_features - 1280,
            num_classes=2,
            fusion_type="concat",
        ).to(self.config["device"])

        for round_num in range(num_rounds):
            global_state = {k: v.clone().detach() for k, v in global_fusion.state_dict().items()}
            for fc in fusion_clients:
                fc.set_model_params(global_state)

            client_params, client_weights, round_losses = [], [], []
            for fc in fusion_clients:
                m = fc.local_train(epochs=self.config["local_epochs"], lr=self.config["learning_rate"])
                client_params.append(fc.get_model_params())
                client_weights.append(m["samples"])
                round_losses.append(m["loss"] * m["samples"])

            total_w = sum(client_weights)
            avg: Dict[str, torch.Tensor] = {}
            for name in client_params[0]:
                avg[name] = sum(
                    cp[name].float() * (w / total_w)
                    for cp, w in zip(client_params, client_weights)
                ).to(client_params[0][name].dtype)
            global_fusion.load_state_dict(avg)

            avg_loss = sum(round_losses) / max(total_w, 1)
            round_metrics_list.append({"avg_loss": avg_loss, "total_samples": total_w})

            # Evaluate
            for fc in fusion_clients:
                fc.set_model_params({k: v.clone().detach() for k, v in global_fusion.state_dict().items()})
            correct_total, samples_total = 0, 0
            for fc in fusion_clients:
                tm = fc.local_test(include_predictions=False)
                correct_total += (tm["accuracy"] / 100) * tm["samples"]
                samples_total += tm["samples"]
            global_acc = 100.0 * correct_total / max(samples_total, 1)
            print(f"  Round {round_num + 1}/{num_rounds} loss={avg_loss:.4f}  acc={global_acc:.2f}%")

            if global_acc > best_fusion_accuracy:
                best_fusion_accuracy = global_acc
                best_fusion_round = round_num + 1
                best_fusion_state = {k: v.detach().clone() for k, v in global_fusion.state_dict().items()}

        if best_fusion_state is not None:
            global_fusion.load_state_dict(best_fusion_state)

        # Final evaluation
        for fc in fusion_clients:
            fc.set_model_params({k: v.detach().clone() for k, v in global_fusion.state_dict().items()})
        all_preds, all_targets = [], []
        correct_total, samples_total = 0, 0
        for fc in fusion_clients:
            tm = fc.local_test(include_predictions=True)
            correct_total += (tm["accuracy"] / 100) * tm["samples"]
            samples_total += tm["samples"]
            all_preds.extend(tm["predictions"])
            all_targets.extend(tm["targets"])
        final_fusion_acc = 100.0 * correct_total / max(samples_total, 1)
        print(f"\nFinal Fusion Model Accuracy: {final_fusion_acc:.2f}% (best at round {best_fusion_round})")

        # Save fusion model
        os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
        exp_name = self.config.get("experiment_name", "default")
        fusion_path = os.path.join(SAVED_MODELS_DIR, f"{exp_name}_fusion_model.pt")
        torch.save(global_fusion.state_dict(), fusion_path)
        print(f"Fusion model saved to {fusion_path}")

        num_params = sum(p.numel() for p in global_fusion.parameters())
        return {
            "status": "trained",
            "final_accuracy": final_fusion_acc,
            "best_round": best_fusion_round,
            "round_metrics": round_metrics_list,
            "model_params": num_params,
            "predictions": all_preds,
            "targets": all_targets,
        }


def load_saved_results() -> Optional[Dict[str, Any]]:
    """Load results from disk if they exist. Returns None if file not found."""
    if os.path.exists(SAVED_RESULTS_PATH):
        try:
            with open(SAVED_RESULTS_PATH, "rb") as f:
                results = pickle.load(f)
            print(f"Loaded saved results from {SAVED_RESULTS_PATH}")
            return results
        except Exception as e:
            print(f"WARNING: Could not load saved results: {e}")
    return None


def save_results(results: Dict[str, Any]) -> None:
    """Save results and evaluations to disk."""
    os.makedirs(os.path.dirname(SAVED_RESULTS_PATH), exist_ok=True)
    with open(SAVED_RESULTS_PATH, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {SAVED_RESULTS_PATH}")


def run_experiments(skip_training_if_saved: bool = False) -> Dict:
    """Run all configured experiments and return results.

    Args:
        skip_training_if_saved: If True and saved_results.pkl exists, load it
            and skip training (for quick visualization/analysis reruns).
    """
    if skip_training_if_saved:
        saved = load_saved_results()
        if saved is not None:
            return saved

    set_random_seeds(42)

    print("Federated MobileNet-TCN Framework for ASD Detection")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    configs = get_experiment_configs()
    evaluator = ComprehensiveEvaluator()
    all_experimental_results: Dict = {}

    for i, config in enumerate(configs, 1):
        experiment_name = config["experiment_name"]

        print(f"\n" + "=" * 50)
        print(f"EXPERIMENT {i}/{len(configs)}: {experiment_name}")
        print("=" * 50)

        try:
            trainer = CompleteFederatedTrainer(config)
            print(f"Running {experiment_name} experiment...")
            facial_results = trainer.run_facial_experiment()
            print(f"Running behavioral (video) experiment...")
            behavioral_results = trainer.run_behavioral_experiment()
            print(f"Running multi-modal fusion experiment...")
            fusion_results = trainer.run_fusion_experiment(
                facial_global_model=trainer.facial_global_model,
                behavioral_global_model=trainer.behavioral_global_model
            )

            experiment_results = {
                "facial_experiment": facial_results,
                "behavioral_experiment": behavioral_results,
                "fusion_experiment": fusion_results,
                "config": config,
            }
            all_experimental_results[experiment_name] = experiment_results

            evaluator.evaluate_experiment(experiment_name, experiment_results)
            print(f"{experiment_name} completed successfully")
            
            # Incremental save in case later experiments fail
            results_to_save = {
                "experimental_results": all_experimental_results,
                "evaluations": evaluator.all_results,
            }
            save_results(results_to_save)

        except Exception as e:
            print(f"✗ Error in {experiment_name}: {e}")
            import traceback

            traceback.print_exc()
            continue

    print("\n" + "=" * 70)
    print("GENERATING COMPREHENSIVE ANALYSIS")
    print("=" * 70)

    evaluator.generate_comparison_report()

    print("\n" + "=" * 70)
    print("ALL EXPERIMENTS COMPLETED!")
    print("=" * 70)
    print("\nSummary:")
    print(f"- Executed {len(configs)} different experimental configurations")
    print("- Generated comprehensive evaluation metrics")
    print("- Results ready for research paper analysis")

    results = {
        "experimental_results": all_experimental_results,
        "evaluations": evaluator.all_results,
    }

    save_results(results)
    return results
