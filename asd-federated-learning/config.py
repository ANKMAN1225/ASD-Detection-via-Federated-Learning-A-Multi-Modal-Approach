"""
Centralized configuration for ASD Federated Learning project.
All hardcoded values are defined here for easy modification.
"""

import os
from dataclasses import dataclass
from typing import List
import torch


@dataclass
class Config:
    """Base experiment configuration."""

    # Dataset paths (relative to project root / asd-federated-learning)
    facial_data_path: str = "../autistic-children-facial-data-set"
    video_data_path: str = "../SSBD-file/ssbd2"

    # Federated learning parameters
    num_clients: int = 5
    num_rounds: int = 100
    local_epochs: int = 3
    learning_rate: float = 0.001
    batch_size: int = 4

    # Data distribution
    iid: bool = True
    alpha: float = 0.5  # Dirichlet alpha for Non-IID split

    # Model parameters
    sequence_length: int = 16
    tcn_channels: List[int] = None
    pretrained: bool = True

    # Privacy
    use_differential_privacy: bool = False
    noise_multiplier: float = 0.0
    max_grad_norm: float = 1.0

    # Training
    random_seed: int = 42
    device: str = None
    use_flower_simulation: bool = True
    flower_client_cpus: int = 1

    # Early stopping (monitors val accuracy; stops when no improvement for `patience` rounds)
    early_stopping_enabled: bool = True
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0
    early_stopping_eval_every_n_rounds: int = 1
    early_stopping_metric_split: str = "valid"

    # Facial data augmentation (helps reduce overfitting)
    facial_train_augmentation: bool = True
    facial_train_rotation_degrees: float = 15.0
    facial_color_jitter_strength: float = 0.1

    # Experiment name (set per experiment)
    experiment_name: str = "default"

    def __post_init__(self) -> None:
        if self.tcn_channels is None:
            self.tcn_channels = [64, 128, 256]
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


# Paths to try for facial dataset (list of paths - iterates correctly)
FACIAL_DATA_PATHS = [
    r"D:\WORK\VScode\Capstone\autistic-children-facial-data-set",
    "../autistic-children-facial-data-set",
]


def _resolve_facial_data_path() -> str:
    """Return first path that exists, or first in list."""
    for p in FACIAL_DATA_PATHS:
        if os.path.exists(p):
            return p
    return FACIAL_DATA_PATHS[0]


def get_experiment_configs() -> List[dict]:
    """Get different experimental configurations for IID, Non-IID, and DP experiments."""
    cfg = Config()
    facial_path = _resolve_facial_data_path()
    base_config = {
        "facial_data_path": facial_path,
        "video_data_path": cfg.video_data_path,
        "num_rounds": cfg.num_rounds,
        "local_epochs": cfg.local_epochs,
        "learning_rate": cfg.learning_rate,
        "batch_size": cfg.batch_size,
        "sequence_length": cfg.sequence_length,
        "device": cfg.device,
        "use_flower_simulation": cfg.use_flower_simulation,
        "flower_client_cpus": cfg.flower_client_cpus,
        # Early stopping settings
        "early_stopping_enabled": cfg.early_stopping_enabled,
        "early_stopping_patience": cfg.early_stopping_patience,
        "early_stopping_min_delta": cfg.early_stopping_min_delta,
        "early_stopping_eval_every_n_rounds": cfg.early_stopping_eval_every_n_rounds,
        "early_stopping_metric_split": cfg.early_stopping_metric_split,
        # Facial augmentation settings
        "facial_train_augmentation": cfg.facial_train_augmentation,
        "facial_train_rotation_degrees": cfg.facial_train_rotation_degrees,
        "facial_color_jitter_strength": cfg.facial_color_jitter_strength,
    }

    config_iid = {
        **base_config,
        "experiment_name": "IID_Distribution",
        "num_clients": 5,
        "iid": True,
        "alpha": 1.0,
        "use_differential_privacy": False,
        "noise_multiplier": 0.0,
    }

    config_privacy = {
        **base_config,
        "experiment_name": "With_Differential_Privacy",
        "num_clients": 5,
        "iid": False,
        "alpha": 0.5,
        "use_differential_privacy": True,
        "noise_multiplier": 0.1,
    }

    return [config_iid, config_privacy]
