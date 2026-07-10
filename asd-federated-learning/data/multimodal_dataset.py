"""
MultimodalDataset - pairs facial images with behavioral videos per client.

When subjects are not explicitly paired, samples are aligned by index modulo
each modality's local dataset size up to the smaller modality count.
ASD labels come from the facial branch (binary autistic / non_autistic).
"""

from typing import Tuple

import torch
from torch.utils.data import Dataset


class MultimodalDataset(Dataset):
    """Paired facial + behavioral samples for late-fusion training."""

    def __init__(self, facial_dataset: Dataset, behavioral_dataset: Dataset) -> None:
        self.facial_dataset = facial_dataset
        self.behavioral_dataset = behavioral_dataset
        self.length = min(len(self.facial_dataset), len(self.behavioral_dataset))

        if len(self.facial_dataset) == 0 or len(self.behavioral_dataset) == 0:
            raise ValueError("MultimodalDataset requires non-empty facial and behavioral subsets.")

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        facial_img, label = self.facial_dataset[idx % len(self.facial_dataset)]
        video, _ = self.behavioral_dataset[idx % len(self.behavioral_dataset)]
        return (facial_img, video), label


def multimodal_collate_fn(batch):
    """Collate multimodal batches into ((facial, video), labels)."""
    facial_imgs = torch.stack([item[0][0] for item in batch])
    videos = torch.stack([item[0][1] for item in batch])
    labels = torch.stack([item[1] for item in batch])
    return (facial_imgs, videos), labels
