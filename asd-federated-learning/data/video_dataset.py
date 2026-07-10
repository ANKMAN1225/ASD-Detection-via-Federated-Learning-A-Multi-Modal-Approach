"""
BehavioralVideoDataset - PyTorch Dataset for Augmented SSBD (ssbd2).

Supports nested Kaggle layout:
    ssbd2/armFlapping/armFlapping/*.mp4
    ssbd2/headBanging/headBanging/*.mp4
    ssbd2/spinning/spinning/*.mp4

Labels: 0=armFlapping, 1=headBanging, 2=spinning
"""

import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset


class BehavioralVideoDataset(Dataset):
    """Dataset for behavioral stereotypy videos (Augmented SSBD / ssbd2)."""

    CLASSES = ["armFlapping", "headBanging", "spinning"]

    def __init__(
        self,
        video_dir: str,
        annotations_file: Optional[str] = None,
        sequence_length: int = 16,
    ) -> None:
        self.video_dir = video_dir
        self.sequence_length = sequence_length
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.CLASSES)}
        self.samples: List[Tuple[str, int]] = []

        for cls in self.CLASSES:
            candidate_paths = [
                os.path.join(video_dir, cls, cls),
                os.path.join(video_dir, cls),
            ]
            cls_dir = None
            for path in candidate_paths:
                if os.path.isdir(path):
                    cls_dir = path
                    break

            if cls_dir is None:
                print(f"  [WARN] BehavioralVideoDataset: no folder for class '{cls}' under {video_dir}")
                continue

            for video_file in sorted(os.listdir(cls_dir)):
                if video_file.lower().endswith((".mp4", ".avi", ".mov")):
                    self.samples.append(
                        (os.path.join(cls_dir, video_file), self.class_to_idx[cls])
                    )

        if len(self.samples) == 0:
            print(f"WARNING: No behavioral videos found in {video_dir}")
            if os.path.exists(video_dir):
                print(f"  Contents: {os.listdir(video_dir)}")
        else:
            counts = {cls: 0 for cls in self.CLASSES}
            for _, label in self.samples:
                counts[self.CLASSES[label]] += 1
            print(
                f"BehavioralVideoDataset: {len(self.samples)} videos "
                f"({', '.join(f'{k}={v}' for k, v in counts.items())})"
            )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        video_path, label = self.samples[idx]
        frames = self.extract_frames(video_path)
        return frames, torch.tensor(label, dtype=torch.long)

    def extract_frames(self, video_path: str) -> torch.Tensor:
        """Extract uniformly sampled frames from a video clip."""
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if frame_count <= 0:
            frame_count = 1
        indices = np.linspace(0, frame_count - 1, self.sequence_length, dtype=int)

        frames = []
        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)
        cap.release()

        if len(frames) == 0:
            frames = [np.zeros((224, 224, 3), dtype=np.uint8)]
        while len(frames) < self.sequence_length:
            frames.append(frames[-1])

        frames = np.array(frames, dtype=np.float32)
        return torch.from_numpy(frames).permute(0, 3, 1, 2) / 255.0
