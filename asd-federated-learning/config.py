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
    num_rounds: int = 50
    local_epochs: int = 2
    learning_rate: float = 0.001
    batch_size: int = 16

    # Data distribution
    iid: bool = True
    alpha: float = 0.5

    # Model parameters
    sequence_length: int = 16
    tcn_channels: List[int] = None
    pretrained: bool = True
    behavioral_num_classes: int = 3
    fusion_type: str = "concat"

    # Privacy (DP-SGD with RDP accountant via Opacus)
    use_differential_privacy: bool = False
    noise_multiplier: float = 1.1
    max_grad_norm: float = 1.0
    dp_delta: float = 1e-5

    # Training
    random_seed: int = 42
    device: str = None
    skip_saved_models: bool = True

    # Early stopping
    early_stopping_enabled: bool = True
    early_stopping_patience: int = 5
    early_stopping_min_delta: float = 0.0
    early_stopping_eval_every_n_rounds: int = 1
    early_stopping_metric_split: str = "valid"

    # Facial data augmentation
    facial_train_augmentation: bool = True
    facial_train_rotation_degrees: float = 15.0
    facial_color_jitter_strength: float = 0.1

    # Behavioral video settings (same round budget as facial)
    behavioral_num_rounds: int = 50
    behavioral_local_epochs: int = 1
    behavioral_batch_size: int = 2
    behavioral_eval_every_n_rounds: int = 1
    behavioral_max_videos: int = 0

    # Fusion settings
    fusion_num_rounds: int = 50
    fusion_local_epochs: int = 1
    fusion_batch_size: int = 4
    fusion_eval_every_n_rounds: int = 1
    fusion_freeze_encoders: bool = True

    experiment_name: str = "default"

    def __post_init__(self) -> None:
        if self.tcn_channels is None:
            self.tcn_channels = [64, 128, 256]
        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"


FACIAL_DATA_PATHS = [
    r"D:\WORK\VScode\Capstone\autistic-children-facial-data-set",
    "../autistic-children-facial-data-set",
]

VIDEO_DATA_PATHS = [
    r"D:\WORK\VScode\Capstone\SSBD-file\ssbd2",
    r"C:\Users\ASUS\.cache\kagglehub\datasets\shradheypathak\ssbd3\versions\1\ssbd2",
    "../SSBD-file/ssbd2",
    "../SSBD-file",
]


def _resolve_facial_data_path() -> str:
    for path in FACIAL_DATA_PATHS:
        if os.path.exists(path):
            return path
    return FACIAL_DATA_PATHS[0]


def _resolve_video_data_path() -> str:
    for path in VIDEO_DATA_PATHS:
        if not os.path.isdir(path):
            continue
        for cls in ["armFlapping", "headBanging", "spinning"]:
            if os.path.isdir(os.path.join(path, cls, cls)) or os.path.isdir(
                os.path.join(path, cls)
            ):
                return path
    return VIDEO_DATA_PATHS[0]


def get_experiment_configs() -> List[dict]:
    """Get IID and DP experimental configurations."""
    facial_path = _resolve_facial_data_path()
    video_path = _resolve_video_data_path()

    base_config = {
        "facial_data_path": facial_path,
        "video_data_path": video_path,
        "num_rounds": 50,
        "local_epochs": 2,
        "learning_rate": 0.001,
        "batch_size": 16,
        "sequence_length": 16,
        "behavioral_num_classes": 3,
        "fusion_type": "concat",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "skip_saved_models": True,
        "early_stopping_enabled": True,
        "early_stopping_patience": 5,
        "early_stopping_min_delta": 0.0,
        "early_stopping_eval_every_n_rounds": 1,
        "early_stopping_metric_split": "valid",
        "facial_train_augmentation": True,
        "facial_train_rotation_degrees": 15.0,
        "facial_color_jitter_strength": 0.1,
        "behavioral_num_rounds": 50,
        "behavioral_local_epochs": 1,
        "behavioral_batch_size": 2,
        "behavioral_eval_every_n_rounds": 1,
        "behavioral_max_videos": 0,
        "fusion_num_rounds": 50,
        "fusion_local_epochs": 1,
        "fusion_batch_size": 4,
        "fusion_eval_every_n_rounds": 1,
        "fusion_freeze_encoders": True,
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
        "noise_multiplier": 1.1,
        "max_grad_norm": 1.0,
        "dp_delta": 1e-5,
    }

    return [config_iid, config_privacy]
