"""
Experiment configurations and runner - orchestrates federated learning experiments.
"""

import os
import pickle
import random
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset, TensorDataset

import torchvision.transforms as transforms

from config import get_experiment_configs
from data import (
    FacialDataset,
    BehavioralVideoDataset,
    MultimodalDataset,
    multimodal_collate_fn,
)
from federated import FederatedClient, FederatedServer, create_federated_data_splits
from federated.privacy import DifferentialPrivacy
from models import MobileNetFeatureExtractor, VideoTCNModel, FusedModel, MultiModalModel
from evaluation import ComprehensiveEvaluator


_EXPERIMENTS_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAVED_RESULTS_PATH = os.path.join(_EXPERIMENTS_DIR, "saved_results.pkl")
SAVED_MODELS_DIR = os.path.join(_EXPERIMENTS_DIR, "saved_models")


def set_random_seeds(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _model_path(experiment_name: str, modality: str) -> str:
    return os.path.join(SAVED_MODELS_DIR, f"{experiment_name}_{modality}.pt")


def _save_model(model: torch.nn.Module, experiment_name: str, modality: str) -> str:
    os.makedirs(SAVED_MODELS_DIR, exist_ok=True)
    path = _model_path(experiment_name, modality)
    torch.save(model.state_dict(), path)
    print(f"Saved {modality} model to {path}")
    return path


def _load_model_state(
    model: torch.nn.Module, experiment_name: str, modality: str
) -> bool:
    path = _model_path(experiment_name, modality)
    if not os.path.exists(path):
        return False
    model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
    print(f"Loaded {modality} weights from {path}")
    return True


def _assign_dp_to_clients(
    clients: List[FederatedClient], dp: Optional[DifferentialPrivacy]
) -> None:
    for client in clients:
        client.dp_module = dp


def _create_dp_module(
    config: Dict,
    sample_count: int,
    batch_size: Optional[int] = None,
) -> Optional[DifferentialPrivacy]:
    if not config.get("use_differential_privacy"):
        return None
    effective_batch_size = int(batch_size or config.get("batch_size", 16))
    sample_rate = min(1.0, effective_batch_size / max(sample_count, 1))
    return DifferentialPrivacy(
        noise_multiplier=float(config.get("noise_multiplier", 1.1)),
        max_grad_norm=float(config.get("max_grad_norm", 1.0)),
        delta=float(config.get("dp_delta", 1e-5)),
        sample_rate=sample_rate,
    )


class CompleteFederatedTrainer:
    """Complete federated training implementation."""

    def __init__(self, config: Dict) -> None:
        self.config = config
        self.device = torch.device(config["device"])
        self.results: Dict = {}

    def _get_transforms(self) -> Tuple[Any, Any]:
        facial_train_rotation = float(
            self.config.get("facial_train_rotation_degrees", 15.0)
        )
        facial_color_jitter = float(
            self.config.get("facial_color_jitter_strength", 0.2)
        )
        facial_train_augmentation = bool(
            self.config.get("facial_train_augmentation", True)
        )

        train_transform_ops: List[Any] = [transforms.Resize((224, 224))]
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
        return transforms.Compose(train_transform_ops), eval_transform

    def prepare_facial_datasets(
        self,
    ) -> Tuple[List[DataLoader], List[DataLoader], List[DataLoader]]:
        train_transform, eval_transform = self._get_transforms()
        facial_path = self.config["facial_data_path"]

        train_loaders: List[DataLoader] = []
        val_loaders: List[DataLoader] = []
        test_loaders: List[DataLoader] = []

        if not os.path.exists(facial_path):
            print("Facial dataset path not found. Using dummy data.")
            for _ in range(self.config["num_clients"]):
                train_loaders.append(self._create_dummy_loader(is_train=True))
                val_loaders.append(self._create_dummy_loader(is_train=False))
                test_loaders.append(self._create_dummy_loader(is_train=False))
            return train_loaders, val_loaders, test_loaders

        try:
            train_facial = FacialDataset(facial_path, "train", transform=None)
            valid_facial = FacialDataset(facial_path, "valid", transform=None)
            test_facial = FacialDataset(facial_path, "test", transform=None)
            train_facial.transform = train_transform
            valid_facial.transform = eval_transform
            test_facial.transform = eval_transform

            print(
                f"Loaded facial dataset: {len(train_facial)} train, "
                f"{len(valid_facial)} valid, {len(test_facial)} test images"
            )

            if min(len(train_facial), len(valid_facial), len(test_facial)) == 0:
                raise ValueError("Facial dataset split is empty.")

            federated_train = create_federated_data_splits(
                train_facial,
                self.config["num_clients"],
                iid=self.config["iid"],
                alpha=self.config["alpha"],
            )
            federated_valid = create_federated_data_splits(
                valid_facial,
                self.config["num_clients"],
                iid=self.config["iid"],
                alpha=self.config["alpha"],
            )
            federated_test = create_federated_data_splits(
                test_facial,
                self.config["num_clients"],
                iid=self.config["iid"],
                alpha=self.config["alpha"],
            )
        except Exception as exc:
            print(f"Error loading facial dataset: {exc}. Using dummy data.")
            for _ in range(self.config["num_clients"]):
                train_loaders.append(self._create_dummy_loader(is_train=True))
                val_loaders.append(self._create_dummy_loader(is_train=False))
                test_loaders.append(self._create_dummy_loader(is_train=False))
            return train_loaders, val_loaders, test_loaders

        batch_size = int(self.config["batch_size"])
        for i in range(self.config["num_clients"]):
            train_loaders.append(
                DataLoader(federated_train[i], batch_size=batch_size, shuffle=True)
            )
            val_loaders.append(
                DataLoader(federated_valid[i], batch_size=batch_size, shuffle=False)
            )
            test_loaders.append(
                DataLoader(federated_test[i], batch_size=batch_size, shuffle=False)
            )

        return train_loaders, val_loaders, test_loaders

    def prepare_behavioral_datasets(
        self,
    ) -> Tuple[List[DataLoader], List[DataLoader], List[Subset], List[Subset]]:
        ssbd_path = self.config.get("video_data_path")
        print(f"Loading behavioral dataset from {ssbd_path}...")

        full_dataset = BehavioralVideoDataset(
            video_dir=ssbd_path,
            sequence_length=self.config["sequence_length"],
        )

        behavioral_max_videos = int(self.config.get("behavioral_max_videos", 0))
        if behavioral_max_videos > 0 and len(full_dataset) > behavioral_max_videos:
            indices = list(range(len(full_dataset)))
            random.shuffle(indices)
            full_dataset = Subset(full_dataset, indices[:behavioral_max_videos])

        train_loaders: List[DataLoader] = []
        test_loaders: List[DataLoader] = []
        train_subsets: List[Subset] = []
        test_subsets: List[Subset] = []

        if len(full_dataset) == 0:
            print("WARNING: No behavioral videos found. Using dummy video data.")
            for _ in range(self.config["num_clients"]):
                train_loaders.append(self._create_dummy_video_loader(is_train=True))
                test_loaders.append(self._create_dummy_video_loader(is_train=False))
            return train_loaders, test_loaders, train_subsets, test_subsets

        print(f"Loaded {len(full_dataset)} behavioral videos.")
        train_size = int(0.8 * len(full_dataset))
        test_size = len(full_dataset) - train_size
        generator = torch.Generator().manual_seed(int(self.config.get("random_seed", 42)))
        train_dataset, test_dataset = torch.utils.data.random_split(
            full_dataset, [train_size, test_size], generator=generator
        )

        iid = bool(self.config.get("iid", True))
        dirichlet_alpha = float(self.config.get("alpha", 0.5))
        client_train_datasets = create_federated_data_splits(
            train_dataset, self.config["num_clients"], iid=iid, alpha=dirichlet_alpha
        )
        client_test_datasets = create_federated_data_splits(
            test_dataset, self.config["num_clients"], iid=iid, alpha=dirichlet_alpha
        )

        behavioral_batch_size = int(
            self.config.get("behavioral_batch_size", self.config["batch_size"])
        )
        for train_ds, test_ds in zip(client_train_datasets, client_test_datasets):
            train_subsets.append(train_ds)
            test_subsets.append(test_ds)
            train_loaders.append(
                DataLoader(
                    train_ds,
                    batch_size=behavioral_batch_size,
                    shuffle=True,
                    num_workers=0,
                )
            )
            test_loaders.append(
                DataLoader(
                    test_ds,
                    batch_size=behavioral_batch_size,
                    shuffle=False,
                    num_workers=0,
                )
            )

        return train_loaders, test_loaders, train_subsets, test_subsets

    def prepare_fusion_datasets(
        self,
        facial_train_loaders: List[DataLoader],
        facial_test_loaders: List[DataLoader],
        behavioral_train_subsets: List[Subset],
        behavioral_test_subsets: List[Subset],
    ) -> Tuple[List[DataLoader], List[DataLoader]]:
        fusion_batch_size = int(self.config.get("fusion_batch_size", 4))
        train_loaders: List[DataLoader] = []
        test_loaders: List[DataLoader] = []

        for i in range(self.config["num_clients"]):
            facial_train_ds = facial_train_loaders[i].dataset
            facial_test_ds = facial_test_loaders[i].dataset
            if i >= len(behavioral_train_subsets) or len(behavioral_train_subsets[i]) == 0:
                raise ValueError(
                    "Fusion requires aligned facial and behavioral client datasets."
                )

            train_mm = MultimodalDataset(facial_train_ds, behavioral_train_subsets[i])
            test_mm = MultimodalDataset(facial_test_ds, behavioral_test_subsets[i])
            train_loaders.append(
                DataLoader(
                    train_mm,
                    batch_size=fusion_batch_size,
                    shuffle=True,
                    collate_fn=multimodal_collate_fn,
                    num_workers=0,
                )
            )
            test_loaders.append(
                DataLoader(
                    test_mm,
                    batch_size=fusion_batch_size,
                    shuffle=False,
                    collate_fn=multimodal_collate_fn,
                    num_workers=0,
                )
            )

        return train_loaders, test_loaders

    def _create_dummy_loader(self, is_train: bool = True) -> DataLoader:
        batch_size = self.config["batch_size"]
        num_samples = 100 if is_train else 20
        dummy_images = torch.randn(num_samples, 3, 224, 224)
        dummy_labels = torch.randint(0, 2, (num_samples,))
        dataset = TensorDataset(dummy_images, dummy_labels)
        return DataLoader(dataset, batch_size=batch_size, shuffle=is_train)

    def _create_dummy_video_loader(self, is_train: bool = True) -> DataLoader:
        num_samples = 80 if is_train else 20
        seq_len = self.config["sequence_length"]
        dummy_videos = torch.randn(num_samples, seq_len, 3, 224, 224)
        dummy_labels = torch.randint(0, 3, (num_samples,))
        dataset = TensorDataset(dummy_videos, dummy_labels)
        behavioral_batch_size = int(
            self.config.get("behavioral_batch_size", self.config["batch_size"])
        )
        return DataLoader(
            dataset,
            batch_size=behavioral_batch_size,
            shuffle=is_train,
            num_workers=0,
        )

    def _run_federated_loop(
        self,
        server: FederatedServer,
        clients: List[FederatedClient],
        dp: Optional[DifferentialPrivacy],
        num_rounds: int,
        local_epochs: int,
        eval_every_n_rounds: int = 1,
        eval_split: str = "test",
        modality_label: str = "model",
    ) -> Dict[str, Any]:
        _assign_dp_to_clients(clients, dp)

        use_early_stopping = bool(self.config.get("early_stopping_enabled", True))
        patience = int(self.config.get("early_stopping_patience", 5))
        min_delta = float(self.config.get("early_stopping_min_delta", 0.0))
        lr = float(self.config["learning_rate"])

        best_global_accuracy = float("-inf")
        best_state_dict: Optional[Dict[str, torch.Tensor]] = None
        bad_evals = 0
        best_round = 0
        stopped_round = 0
        eval_count = 0
        eval_accuracy_log: List[Dict] = []

        print(f"\nStarting federated training for {num_rounds} rounds ({modality_label})...")

        for round_num in range(num_rounds):
            print(f"\n--- {modality_label} Round {round_num + 1}/{num_rounds} ---")
            round_metrics = server.train_round(local_epochs=local_epochs, lr=lr)
            print(f"Round {round_num + 1} - Avg Loss: {round_metrics['avg_loss']:.4f}")

            should_eval = (
                (round_num + 1) % eval_every_n_rounds == 0
                or (round_num + 1) == num_rounds
            )
            if not should_eval:
                continue

            eval_count += 1
            eval_metrics = server.evaluate_global_model(
                split=eval_split, include_predictions=False
            )
            global_acc = eval_metrics["global_accuracy"]
            print(
                f"Round {round_num + 1} - Global Accuracy ({eval_split}): "
                f"{global_acc:.2f}%"
            )
            eval_accuracy_log.append(
                {
                    "round": round_num + 1,
                    "accuracy": global_acc,
                    "split": eval_split,
                }
            )

            if global_acc > (best_global_accuracy + min_delta):
                best_global_accuracy = global_acc
                best_round = round_num + 1
                best_state_dict = {
                    k: v.detach().clone()
                    for k, v in server.global_model.state_dict().items()
                }
                bad_evals = 0
            elif use_early_stopping:
                bad_evals += 1
                print(f"    [Early Stopping] patience {bad_evals}/{patience}")
                if bad_evals >= patience:
                    stopped_round = round_num + 1
                    print(
                        f"Early stopping triggered for {modality_label}. "
                        f"Best accuracy {best_global_accuracy:.2f}% at round {best_round}."
                    )
                    break

        if stopped_round == 0:
            stopped_round = num_rounds

        if best_state_dict is not None:
            server.global_model.load_state_dict(best_state_dict)

        final_metrics = server.evaluate_global_model(
            split="test", include_predictions=True
        )
        final_metrics.update(self._extended_classification_metrics(final_metrics))
        print(f"\nFinal {modality_label} Accuracy: {final_metrics['global_accuracy']:.2f}%")

        privacy_info: Dict[str, Any] = {}
        if dp is not None:
            epsilon = dp.get_epsilon(delta=float(self.config.get("dp_delta", 1e-5)))
            privacy_info = {
                "epsilon": epsilon,
                "delta": float(self.config.get("dp_delta", 1e-5)),
                "noise_multiplier": dp.noise_multiplier,
                "max_grad_norm": dp.max_grad_norm,
                "gradient_steps": dp.num_steps,
            }
            print(dp.privacy_summary(delta=float(self.config.get("dp_delta", 1e-5))))

        return {
            "round_metrics": server.round_metrics,
            "final_metrics": final_metrics,
            "eval_accuracy_log": eval_accuracy_log,
            "early_stopping": {
                "enabled": use_early_stopping,
                "metric_split": eval_split,
                "best_global_accuracy": best_global_accuracy,
                "best_round": best_round,
                "stopped_round": stopped_round,
                "num_metric_evaluations": eval_count,
                "patience": patience,
                "min_delta": min_delta,
                "eval_every_n_rounds": eval_every_n_rounds,
            },
            "privacy": privacy_info,
        }

    def _extended_classification_metrics(self, final_metrics: Dict) -> Dict[str, float]:
        """Add precision/recall/F1/AUC/client variance for reviewer tables."""
        preds = final_metrics.get("predictions", [])
        targets = final_metrics.get("targets", [])
        client_metrics = final_metrics.get("client_metrics", [])
        extended: Dict[str, float] = {}

        if preds and targets:
            try:
                from evaluation.metrics import evaluate_metrics
                import numpy as np

                y_true = np.array(targets).flatten()
                y_pred = np.array(preds).flatten()
                clf = evaluate_metrics(y_true, y_pred)
                extended["global_precision"] = float(clf["precision"] * 100)
                extended["global_recall"] = float(clf["recall"] * 100)
                extended["global_f1"] = float(clf["f1_score"] * 100)
                extended["global_auc"] = (
                    float(clf["auc"] * 100) if clf.get("auc") is not None else 0.0
                )
            except Exception:
                pass

        if client_metrics:
            import numpy as np
            client_accs = [m["accuracy"] for m in client_metrics]
            extended["client_accuracy_variance"] = float(np.var(client_accs))

        return extended

    def run_facial_experiment(self) -> Dict:
        print("\n" + "=" * 50)
        print("FACIAL IMAGE FEDERATED EXPERIMENT")
        print("=" * 50)

        exp_name = self.config["experiment_name"]
        global_model = MobileNetFeatureExtractor(num_classes=2)

        if self.config.get("skip_saved_models", True) and _load_model_state(
            global_model, exp_name, "facial"
        ):
            train_loaders, val_loaders, test_loaders = self.prepare_facial_datasets()
            clients = [
                FederatedClient(
                    client_id=i,
                    model=MobileNetFeatureExtractor(num_classes=2).to(self.device),
                    train_loader=train_loaders[i],
                    val_loader=val_loaders[i],
                    test_loader=test_loaders[i],
                    device=self.config["device"],
                )
                for i in range(self.config["num_clients"])
            ]
            for client in clients:
                client.model.load_state_dict(global_model.state_dict())
            server = FederatedServer(global_model, clients)
            final_metrics = server.evaluate_global_model(
                split="test", include_predictions=True
            )
            result = {
                "status": "loaded_from_checkpoint",
                "final_metrics": final_metrics,
                "config": self.config,
            }
            self.results["facial_experiment"] = result
            return result

        train_loaders, val_loaders, test_loaders = self.prepare_facial_datasets()
        global_model = MobileNetFeatureExtractor(num_classes=2)
        clients = [
            FederatedClient(
                client_id=i,
                model=MobileNetFeatureExtractor(num_classes=2),
                train_loader=train_loaders[i],
                val_loader=val_loaders[i],
                test_loader=test_loaders[i],
                device=self.config["device"],
            )
            for i in range(self.config["num_clients"])
        ]
        server = FederatedServer(global_model, clients)

        total_samples = sum(len(loader.dataset) for loader in train_loaders)
        dp = _create_dp_module(
            self.config,
            total_samples,
            batch_size=int(self.config["batch_size"]),
        )

        result = self._run_federated_loop(
            server=server,
            clients=clients,
            dp=dp,
            num_rounds=int(self.config["num_rounds"]),
            local_epochs=int(self.config["local_epochs"]),
            eval_every_n_rounds=int(
                self.config.get("early_stopping_eval_every_n_rounds", 1)
            ),
            eval_split=str(self.config.get("early_stopping_metric_split", "valid")),
            modality_label="facial",
        )
        result["config"] = self.config
        _save_model(server.global_model, exp_name, "facial")
        self.results["facial_experiment"] = result
        return result

    def run_behavioral_experiment(self) -> Dict:
        print("\n" + "=" * 50)
        print("BEHAVIORAL VIDEO FEDERATED EXPERIMENT")
        print("=" * 50)

        exp_name = self.config["experiment_name"]
        num_classes = int(self.config.get("behavioral_num_classes", 3))
        global_model = VideoTCNModel(
            num_classes=num_classes,
            sequence_length=self.config["sequence_length"],
        )

        if self.config.get("skip_saved_models", True) and _load_model_state(
            global_model, exp_name, "behavioral"
        ):
            train_loaders, test_loaders, _, _ = self.prepare_behavioral_datasets()
            clients = [
                FederatedClient(
                    client_id=i,
                    model=VideoTCNModel(
                        num_classes=num_classes,
                        sequence_length=self.config["sequence_length"],
                    ).to(self.device),
                    train_loader=train_loaders[i],
                    test_loader=test_loaders[i],
                    device=self.config["device"],
                )
                for i in range(self.config["num_clients"])
            ]
            for client in clients:
                client.model.load_state_dict(global_model.state_dict())
            server = FederatedServer(global_model, clients)
            final_metrics = server.evaluate_global_model(
                split="test", include_predictions=True
            )
            result = {
                "status": "loaded_from_checkpoint",
                "final_metrics": final_metrics,
                "config": self.config,
            }
            self.results["behavioral_experiment"] = result
            return result

        train_loaders, test_loaders, train_subsets, test_subsets = (
            self.prepare_behavioral_datasets()
        )
        self._behavioral_train_subsets = train_subsets
        self._behavioral_test_subsets = test_subsets

        global_model = VideoTCNModel(
            num_classes=num_classes,
            sequence_length=self.config["sequence_length"],
        )
        clients = [
            FederatedClient(
                client_id=i,
                model=VideoTCNModel(
                    num_classes=num_classes,
                    sequence_length=self.config["sequence_length"],
                ),
                train_loader=train_loaders[i],
                test_loader=test_loaders[i],
                device=self.config["device"],
            )
            for i in range(self.config["num_clients"])
        ]
        server = FederatedServer(global_model, clients)

        total_samples = sum(len(loader.dataset) for loader in train_loaders)
        dp = _create_dp_module(
            self.config,
            total_samples,
            batch_size=int(
                self.config.get("behavioral_batch_size", self.config["batch_size"])
            ),
        )

        result = self._run_federated_loop(
            server=server,
            clients=clients,
            dp=dp,
            num_rounds=int(self.config.get("behavioral_num_rounds", 50)),
            local_epochs=int(self.config.get("behavioral_local_epochs", 1)),
            eval_every_n_rounds=int(
                self.config.get("behavioral_eval_every_n_rounds", 1)
            ),
            eval_split="test",
            modality_label="behavioral",
        )
        result["config"] = self.config
        _save_model(server.global_model, exp_name, "behavioral")
        self.results["behavioral_experiment"] = result
        return result

    def run_fusion_experiment(
        self,
        facial_train_loaders: Optional[List[DataLoader]] = None,
        facial_test_loaders: Optional[List[DataLoader]] = None,
        behavioral_train_subsets: Optional[List[Subset]] = None,
        behavioral_test_subsets: Optional[List[Subset]] = None,
    ) -> Dict:
        print("\n" + "=" * 50)
        print("MULTI-MODAL FUSION FEDERATED EXPERIMENT")
        print("=" * 50)

        exp_name = self.config["experiment_name"]
        fusion_type = str(self.config.get("fusion_type", "concat"))

        if facial_train_loaders is None or facial_test_loaders is None:
            facial_train_loaders, _, facial_test_loaders = self.prepare_facial_datasets()
        if behavioral_train_subsets is None or behavioral_test_subsets is None:
            _, _, behavioral_train_subsets, behavioral_test_subsets = (
                self.prepare_behavioral_datasets()
            )

        fusion_model = MultiModalModel(
            facial_model=MobileNetFeatureExtractor(num_classes=2),
            behavioral_model=VideoTCNModel(
                num_classes=int(self.config.get("behavioral_num_classes", 3)),
                sequence_length=self.config["sequence_length"],
            ),
            fusion_model=FusedModel(
                facial_feature_dim=1280,
                behavioral_feature_dim=256,
                num_classes=2,
                fusion_type=fusion_type,
            ),
        )

        if self.config.get("skip_saved_models", True) and _load_model_state(
            fusion_model, exp_name, "fusion"
        ):
            train_loaders, test_loaders = self.prepare_fusion_datasets(
                facial_train_loaders,
                facial_test_loaders,
                behavioral_train_subsets,
                behavioral_test_subsets,
            )
            clients = [
                FederatedClient(
                    client_id=i,
                    model=MultiModalModel(
                        facial_model=MobileNetFeatureExtractor(num_classes=2),
                        behavioral_model=VideoTCNModel(
                            num_classes=int(self.config.get("behavioral_num_classes", 3)),
                            sequence_length=self.config["sequence_length"],
                        ),
                        fusion_model=FusedModel(
                            facial_feature_dim=1280,
                            behavioral_feature_dim=256,
                            num_classes=2,
                            fusion_type=fusion_type,
                        ),
                    ).to(self.device),
                    train_loader=train_loaders[i],
                    test_loader=test_loaders[i],
                    device=self.config["device"],
                )
                for i in range(self.config["num_clients"])
            ]
            for client in clients:
                client.model.load_state_dict(fusion_model.state_dict())
            server = FederatedServer(fusion_model, clients)
            final_metrics = server.evaluate_global_model(
                split="test", include_predictions=True
            )
            result = {
                "status": "loaded_from_checkpoint",
                "final_metrics": final_metrics,
                "config": self.config,
            }
            self.results["fusion_experiment"] = result
            return result

        _load_model_state(fusion_model.facial_encoder, exp_name, "facial")
        _load_model_state(fusion_model.behavioral_encoder, exp_name, "behavioral")

        if self.config.get("fusion_freeze_encoders", True):
            for param in fusion_model.facial_encoder.parameters():
                param.requires_grad = False
            for param in fusion_model.behavioral_encoder.parameters():
                param.requires_grad = False
            print("Fusion encoders frozen; training fusion head only.")

        train_loaders, test_loaders = self.prepare_fusion_datasets(
            facial_train_loaders,
            facial_test_loaders,
            behavioral_train_subsets,
            behavioral_test_subsets,
        )

        clients = [
            FederatedClient(
                client_id=i,
                model=MultiModalModel(
                    facial_model=MobileNetFeatureExtractor(num_classes=2),
                    behavioral_model=VideoTCNModel(
                        num_classes=int(self.config.get("behavioral_num_classes", 3)),
                        sequence_length=self.config["sequence_length"],
                    ),
                    fusion_model=FusedModel(
                        facial_feature_dim=1280,
                        behavioral_feature_dim=256,
                        num_classes=2,
                        fusion_type=fusion_type,
                    ),
                ),
                train_loader=train_loaders[i],
                test_loader=test_loaders[i],
                device=self.config["device"],
            )
            for i in range(self.config["num_clients"])
        ]

        for i, client in enumerate(clients):
            client.model.load_state_dict(fusion_model.state_dict())
            if self.config.get("fusion_freeze_encoders", True):
                for param in client.model.facial_encoder.parameters():
                    param.requires_grad = False
                for param in client.model.behavioral_encoder.parameters():
                    param.requires_grad = False

        server = FederatedServer(fusion_model, clients)
        total_samples = sum(len(loader.dataset) for loader in train_loaders)
        dp = _create_dp_module(
            self.config,
            total_samples,
            batch_size=int(self.config.get("fusion_batch_size", self.config["batch_size"])),
        )

        result = self._run_federated_loop(
            server=server,
            clients=clients,
            dp=dp,
            num_rounds=int(self.config.get("fusion_num_rounds", 50)),
            local_epochs=int(self.config.get("fusion_local_epochs", 1)),
            eval_every_n_rounds=int(self.config.get("fusion_eval_every_n_rounds", 1)),
            eval_split="test",
            modality_label="fusion",
        )
        result["config"] = self.config
        _save_model(server.global_model, exp_name, "fusion")
        self.results["fusion_experiment"] = result
        return result


def load_saved_results() -> Optional[Dict[str, Any]]:
    if os.path.exists(SAVED_RESULTS_PATH):
        try:
            with open(SAVED_RESULTS_PATH, "rb") as f:
                results = pickle.load(f)
            print(f"Loaded saved results from {SAVED_RESULTS_PATH}")
            return results
        except Exception as exc:
            print(f"WARNING: Could not load saved results: {exc}")
    return None


def save_results(results: Dict[str, Any]) -> None:
    os.makedirs(os.path.dirname(SAVED_RESULTS_PATH), exist_ok=True)
    with open(SAVED_RESULTS_PATH, "wb") as f:
        pickle.dump(results, f)
    print(f"Results saved to {SAVED_RESULTS_PATH}")


def _require_saved_model_checkpoints(
    configs: List[Dict],
    skip_facial: bool,
    skip_fusion: bool,
) -> None:
    """Fail fast when evaluation-only mode is missing a required checkpoint.

    This prevents an evaluation command from unexpectedly entering the training
    path when a checkpoint has been deleted or renamed.
    """
    modalities = ["behavioral"]
    if not skip_facial:
        modalities.append("facial")
    if not skip_fusion:
        modalities.append("fusion")

    missing = [
        _model_path(config["experiment_name"], modality)
        for config in configs
        for modality in modalities
        if not os.path.isfile(_model_path(config["experiment_name"], modality))
    ]
    if missing:
        paths = "\n  - ".join(missing)
        raise FileNotFoundError(
            "Evaluation-only mode requires existing checkpoints. Missing:\n"
            f"  - {paths}"
        )


def run_experiments(
    skip_training_if_saved: bool = False,
    skip_facial: bool = False,
    skip_fusion: bool = False,
    force_retrain: bool = False,
    evaluate_saved_models_only: bool = False,
    experiment_names: Optional[List[str]] = None,
    random_seed: int = 42,
) -> Dict:
    if skip_training_if_saved:
        saved = load_saved_results()
        if saved is not None:
            return saved

    seed = int(random_seed)
    set_random_seeds(seed)

    print("Federated MobileNet-TCN Framework for ASD Detection")
    print("=" * 60)
    print(f"PyTorch version: {torch.__version__}")
    print(f"Device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    configs = get_experiment_configs()
    if experiment_names:
        allowed = set(experiment_names)
        configs = [c for c in configs if c["experiment_name"] in allowed]
        if not configs:
            raise ValueError(
                f"No matching experiments for {experiment_names}. "
                "Choose from: IID_Distribution, With_Differential_Privacy"
            )
        print(f"Running selected experiments: {[c['experiment_name'] for c in configs]}")
    if force_retrain and evaluate_saved_models_only:
        raise ValueError(
            "--force-retrain and evaluation-only mode cannot be used together."
        )
    if evaluate_saved_models_only:
        _require_saved_model_checkpoints(configs, skip_facial, skip_fusion)
        for cfg in configs:
            cfg["skip_saved_models"] = True
        print("Evaluation-only mode: loading saved checkpoints; no training will run.")
    elif force_retrain:
        for cfg in configs:
            cfg["skip_saved_models"] = False

    evaluator = ComprehensiveEvaluator()
    all_experimental_results: Dict = {}

    for i, config in enumerate(configs, 1):
        config["random_seed"] = seed
        experiment_name = config["experiment_name"]

        print(f"\n{'=' * 50}")
        print(f"EXPERIMENT {i}/{len(configs)}: {experiment_name}")
        print("=" * 50)

        try:
            trainer = CompleteFederatedTrainer(config)

            facial_results: Dict = {"status": "skipped", "final_metrics": {}}
            facial_train_loaders = None
            facial_test_loaders = None
            behavioral_train_subsets = None
            behavioral_test_subsets = None

            if not skip_facial:
                facial_results = trainer.run_facial_experiment()
                facial_train_loaders, _, facial_test_loaders = (
                    trainer.prepare_facial_datasets()
                )

            behavioral_results = trainer.run_behavioral_experiment()
            behavioral_train_subsets = getattr(trainer, "_behavioral_train_subsets", None)
            behavioral_test_subsets = getattr(trainer, "_behavioral_test_subsets", None)
            if behavioral_train_subsets is None:
                _, _, behavioral_train_subsets, behavioral_test_subsets = (
                    trainer.prepare_behavioral_datasets()
                )

            fusion_results: Dict = {"status": "skipped", "final_metrics": {}}
            if not skip_fusion and facial_train_loaders is not None:
                fusion_results = trainer.run_fusion_experiment(
                    facial_train_loaders=facial_train_loaders,
                    facial_test_loaders=facial_test_loaders,
                    behavioral_train_subsets=behavioral_train_subsets,
                    behavioral_test_subsets=behavioral_test_subsets,
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

        except Exception as exc:
            print(f"ERROR in {experiment_name}: {exc}")
            import traceback

            traceback.print_exc()
            continue

    print("\n" + "=" * 70)
    print("GENERATING COMPREHENSIVE ANALYSIS")
    print("=" * 70)
    evaluator.generate_comparison_report()

    results = {
        "experimental_results": all_experimental_results,
        "evaluations": evaluator.all_results,
    }
    save_results(results)
    return results
