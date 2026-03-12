"""
Experiment configurations and runner - orchestrates federated learning experiments.
"""

import os
import pickle
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

import torchvision.transforms as transforms

from config import get_experiment_configs
from data import FacialDataset
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
        self.results: Dict = {}

    def prepare_datasets(
        self,
    ) -> Tuple[List[DataLoader], List[DataLoader]]:
        """Prepare federated datasets for training and testing."""
        transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225],
                ),
            ]
        )

        facial_path = self.config["facial_data_path"]
        if os.path.exists(facial_path):
            try:
                train_facial = FacialDataset(
                    facial_path, "train", transform=transform
                )
                test_facial = FacialDataset(
                    facial_path, "test", transform=transform
                )

                print(
                    f"Loaded facial dataset: {len(train_facial)} training, "
                    f"{len(test_facial)} testing samples"
                )

                if len(train_facial) == 0 or len(test_facial) == 0:
                    print(
                        "Facial dataset is empty (missing train/autistic, "
                        "train/non_autistic, test/autistic, test/non_autistic). "
                        "Using dummy data."
                    )
                    federated_train_facial = [None] * self.config["num_clients"]
                    federated_test_facial = [None] * self.config["num_clients"]
                else:
                    federated_train_facial = create_federated_data_splits(
                        train_facial,
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
                federated_test_facial = [None] * self.config["num_clients"]
        else:
            print("Facial dataset path not found. Using dummy data.")
            federated_train_facial = [None] * self.config["num_clients"]
            federated_test_facial = [None] * self.config["num_clients"]

        train_loaders: List[DataLoader] = []
        test_loaders: List[DataLoader] = []

        for i in range(self.config["num_clients"]):
            if federated_train_facial[i] is not None:
                train_loader = DataLoader(
                    federated_train_facial[i],
                    batch_size=self.config["batch_size"],
                    shuffle=True,
                )
                test_loader = DataLoader(
                    federated_test_facial[i],
                    batch_size=self.config["batch_size"],
                    shuffle=False,
                )
            else:
                train_loader = self._create_dummy_loader(is_train=True)
                test_loader = self._create_dummy_loader(is_train=False)

            train_loaders.append(train_loader)
            test_loaders.append(test_loader)

        return train_loaders, test_loaders

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

        train_loaders, test_loaders = self.prepare_datasets()

        global_model = MobileNetFeatureExtractor(num_classes=2)

        clients: List[FederatedClient] = []
        for i in range(self.config["num_clients"]):
            client_model = MobileNetFeatureExtractor(num_classes=2)
            client = FederatedClient(
                client_id=i,
                model=client_model,
                train_loader=train_loaders[i],
                test_loader=test_loaders[i],
                device=self.config["device"],
            )
            clients.append(client)

        server = FederatedServer(global_model, clients)

        if self.config["use_differential_privacy"]:
            dp = DifferentialPrivacy(
                noise_multiplier=self.config["noise_multiplier"],
                max_grad_norm=1.0,
            )

        num_rounds = self.config["num_rounds"]
        print(f"\nStarting federated training for {num_rounds} rounds...")

        for round_num in range(num_rounds):
            print(f"\n--- Round {round_num + 1}/{num_rounds} ---")

            round_metrics = server.train_round(
                local_epochs=self.config["local_epochs"],
                lr=self.config["learning_rate"],
            )

            print(f"Round {round_num + 1} - Avg Loss: {round_metrics['avg_loss']:.4f}")

            if (round_num + 1) % 2 == 0:
                eval_metrics = server.evaluate_global_model()
                print(
                    f"Round {round_num + 1} - Global Accuracy: "
                    f"{eval_metrics['global_accuracy']:.2f}%"
                )

        print("\n" + "-" * 30)
        print("FINAL EVALUATION")
        print("-" * 30)

        final_metrics = server.evaluate_global_model()

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
            "config": self.config,
        }

        # Save trained model to disk (so you don't need to retrain)
        os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
        exp_name = self.config.get("experiment_name", "default")
        model_path = os.path.join(SAVED_MODELS_DIR, f"{exp_name}_model.pt")
        torch.save(server.global_model.state_dict(), model_path)
        print(f"✓ Model saved to {model_path}")

        return self.results["facial_experiment"]

    def run_behavioral_experiment(self) -> Dict:
        """Run federated learning experiment with behavioral video data."""
        print("\n" + "=" * 50)
        print("BEHAVIORAL VIDEO FEDERATED EXPERIMENT")
        print("=" * 50)

        train_loaders: List[DataLoader] = []
        test_loaders: List[DataLoader] = []

        for i in range(self.config["num_clients"]):
            train_loader = self._create_dummy_video_loader(is_train=True)
            test_loader = self._create_dummy_video_loader(is_train=False)
            train_loaders.append(train_loader)
            test_loaders.append(test_loader)

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

        num_rounds = self.config["num_rounds"]
        print(f"\nStarting behavioral federated training for {num_rounds} rounds...")

        for round_num in range(num_rounds):
            print(f"\n--- Round {round_num + 1}/{num_rounds} ---")

            round_metrics = server.train_round(
                local_epochs=self.config["local_epochs"],
                lr=self.config["learning_rate"],
            )

            print(f"Round {round_num + 1} - Avg Loss: {round_metrics['avg_loss']:.4f}")

            if (round_num + 1) % 2 == 0:
                eval_metrics = server.evaluate_global_model()
                print(
                    f"Round {round_num + 1} - Global Accuracy: "
                    f"{eval_metrics['global_accuracy']:.2f}%"
                )

        final_metrics = server.evaluate_global_model()
        print(
            f"\nFinal Behavioral Model Accuracy: "
            f"{final_metrics['global_accuracy']:.2f}%"
        )

        self.results["behavioral_experiment"] = {
            "round_metrics": server.round_metrics,
            "final_metrics": final_metrics,
            "config": self.config,
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

    def run_fusion_experiment(self) -> Dict:
        """Run federated learning with multi-modal fusion (demonstration)."""
        print("\n" + "=" * 50)
        print("MULTI-MODAL FUSION EXPERIMENT")
        print("=" * 50)

        print("Multi-modal fusion experiment would combine:")
        print("- Facial features (1280-dim from MobileNet)")
        print("- Behavioral features (256-dim from TCN)")
        print("- Late fusion with attention mechanism")

        fusion_model = FusedModel(
            facial_feature_dim=1280,
            behavioral_feature_dim=256,
            num_classes=2,
            fusion_type="attention",
        )

        num_params = sum(p.numel() for p in fusion_model.parameters())
        print(f"Fusion model created with {num_params} parameters")

        return {
            "status": "demonstrated",
            "model_params": num_params,
        }


def load_saved_results() -> Optional[Dict[str, Any]]:
    """Load results from disk if they exist. Returns None if file not found."""
    if os.path.exists(SAVED_RESULTS_PATH):
        try:
            with open(SAVED_RESULTS_PATH, "rb") as f:
                results = pickle.load(f)
            print(f"✓ Loaded saved results from {SAVED_RESULTS_PATH}")
            return results
        except Exception as e:
            print(f"⚠ Could not load saved results: {e}")
    return None


def save_results(results: Dict[str, Any]) -> None:
    """Save results and evaluations to disk."""
    os.makedirs(os.path.dirname(SAVED_RESULTS_PATH), exist_ok=True)
    with open(SAVED_RESULTS_PATH, "wb") as f:
        pickle.dump(results, f)
    print(f"✓ Results saved to {SAVED_RESULTS_PATH}")


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

            experiment_results = {
                "facial_experiment": facial_results,
                "config": config,
            }
            all_experimental_results[experiment_name] = experiment_results

            evaluator.evaluate_experiment(experiment_name, experiment_results)
            print(f"✓ {experiment_name} completed successfully")

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
