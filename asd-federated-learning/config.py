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
    video_data_path: str = "../SSBD-file"

    # Federated learning parameters
    num_clients: int = 5
    num_rounds: int = 10
    local_epochs: int = 2
    learning_rate: float = 0.001
    batch_size: int = 16

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

    # Early stopping (for central evaluation/selection)
    early_stopping_enabled: bool = True
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0
    early_stopping_eval_every_n_rounds: int = 1
    early_stopping_metric_split: str = "valid"

    # Facial data augmentation (helps reduce overfitting)
    facial_train_augmentation: bool = True
    facial_train_rotation_degrees: float = 15.0
    facial_color_jitter_strength: float = 0.1

    # Behavioral video workload caps (to keep `--skip-facial` runs safe)
    behavioral_num_rounds: int = 1
    behavioral_local_epochs: int = 1
    behavioral_batch_size: int = 2
    behavioral_eval_every_n_rounds: int = 1
    # If <= 0, use all videos found in the dataset directory.
    behavioral_max_videos: int = 20

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
    facial_path = _resolve_facial_data_path()
    base_config = {
        "facial_data_path": facial_path,
        "video_data_path": "../SSBD-file",
        "num_rounds": 100,
        "local_epochs": 2,
        "learning_rate": 0.001,
        "batch_size": 16,
        "sequence_length": 16,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        # Early stopping settings
        "early_stopping_enabled": True,
        "early_stopping_patience": 5,
        "early_stopping_min_delta": 0.0,
        "early_stopping_eval_every_n_rounds": 1,
        "early_stopping_metric_split": "valid",
        # Facial augmentation settings
        "facial_train_augmentation": True,
        "facial_train_rotation_degrees": 15.0,
        "facial_color_jitter_strength": 0.1,

        # Behavioral caps
        "behavioral_num_rounds": 1,
        "behavioral_local_epochs": 1,
        "behavioral_batch_size": 2,
        "behavioral_eval_every_n_rounds": 1,
        "behavioral_max_videos": 20,
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
