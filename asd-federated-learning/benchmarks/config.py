"""Configuration object for the standalone baseline benchmark runner."""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict

import torch


@dataclass
class BenchmarkConfig:
    """Settings shared by every baseline in one benchmark invocation."""

    modality: str = "facial"
    data_dir: str = os.environ.get(
        "ASD_FACIAL_DATA_PATH", "../autistic-children-facial-data-set"
    )
    video_data_dir: str = os.environ.get("ASD_VIDEO_DATA_PATH", "../SSBD-file/ssbd2")
    num_clients: int = 5
    iid: bool = True
    dirichlet_alpha: float = 0.5
    rounds: int = 50
    local_epochs: int = 2
    batch_size: int = 16
    learning_rate: float = 0.001
    weight_decay: float = 0.0
    fedprox_mu: float = 0.01
    image_size: int = 224
    sequence_length: int = 16
    max_samples_per_class: int = 0
    train_fraction: float = 0.70
    validation_fraction: float = 0.15
    pretrained: bool = False
    class_weighted_loss: bool = True
    early_stopping_patience: int = 5
    seed: int = 42
    num_workers: int = 0
    output_dir: str = "benchmark_results"
    device: str = ""

    def __post_init__(self) -> None:
        if not self.device:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.validate()

    def validate(self) -> None:
        if self.modality not in {"facial", "video", "fusion"}:
            raise ValueError("modality must be 'facial', 'video', or 'fusion'.")
        if self.num_clients < 1:
            raise ValueError("num_clients must be at least 1.")
        if self.rounds < 1 or self.local_epochs < 1:
            raise ValueError("rounds and local_epochs must both be positive.")
        if self.batch_size < 1 or self.learning_rate <= 0:
            raise ValueError("batch_size and learning_rate must be positive.")
        if self.fedprox_mu < 0:
            raise ValueError("fedprox_mu cannot be negative.")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience cannot be negative.")
        if not 0 < self.dirichlet_alpha:
            raise ValueError("dirichlet_alpha must be positive.")
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must be between 0 and 1.")
        if not 0 < self.validation_fraction < 1 - self.train_fraction:
            raise ValueError("validation_fraction must leave a non-empty test split.")
        if self.image_size < 32 or self.sequence_length < 1:
            raise ValueError("image_size must be >= 32 and sequence_length positive.")
        if self.max_samples_per_class < 0:
            raise ValueError("max_samples_per_class cannot be negative.")
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("A CUDA device was requested but CUDA is not available.")

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir).expanduser().resolve()

    @property
    def output_path(self) -> Path:
        return Path(self.output_dir).expanduser().resolve()

    @property
    def video_data_path(self) -> Path:
        return Path(self.video_data_dir).expanduser().resolve()

    def as_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data["data_dir"] = str(self.data_path)
        data["video_data_dir"] = str(self.video_data_path)
        data["output_dir"] = str(self.output_path)
        return data
