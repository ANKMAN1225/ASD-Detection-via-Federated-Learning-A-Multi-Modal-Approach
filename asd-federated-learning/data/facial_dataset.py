"""
FacialDataset - PyTorch Dataset for loading facial images from the Kaggle ASD dataset.
Supports train/test/valid splits with autistic and non_autistic classes.
"""

import os
from typing import Optional, Callable

import torch
from PIL import Image
from torch.utils.data import Dataset


class FacialDataset(Dataset):
    """Dataset for facial images (Kaggle ASD dataset)."""

    def __init__(self, root_dir: str, split: str, transform: Optional[Callable] = None) -> None:
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.classes = ["autistic", "non_autistic"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        self.samples = []
        split_path = os.path.join(root_dir, split)

        for cls in self.classes:
            class_path = os.path.join(split_path, cls)
            if os.path.exists(class_path):
                for img_file in os.listdir(class_path):
                    if img_file.lower().endswith((".png", ".jpg", ".jpeg")):
                        img_path = os.path.join(class_path, img_file)
                        self.samples.append((img_path, self.class_to_idx[cls]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long)
