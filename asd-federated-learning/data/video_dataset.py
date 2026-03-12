"""
BehavioralVideoDataset - PyTorch Dataset for loading and extracting frames from videos.
Designed for SSBD (Self-Stimulatory Behaviour Dataset) with armflapping, headbanging, spinning classes.
"""

import os
from typing import Optional

import numpy as np
import torch
import cv2
from torch.utils.data import Dataset


class BehavioralVideoDataset(Dataset):
    """Dataset for behavioral videos (SSBD-style)."""

    def __init__(
        self,
        video_dir: str,
        annotations_file: Optional[str] = None,
        sequence_length: int = 16,
    ) -> None:
        self.video_dir = video_dir
        self.sequence_length = sequence_length
        self.samples = []

        # For SSBD dataset structure
        self.classes = ["armflapping", "headbanging", "spinning"]
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

        # Load video files
        for cls in self.classes:
            class_path = os.path.join(video_dir, cls)
            if os.path.exists(class_path):
                for video_file in os.listdir(class_path):
                    if video_file.endswith(".mp4") or video_file.endswith(".avi"):
                        video_path = os.path.join(class_path, video_file)
                        self.samples.append((video_path, self.class_to_idx[cls]))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple:
        video_path, label = self.samples[idx]
        frames = self.extract_frames(video_path)
        return frames, torch.tensor(label, dtype=torch.long)

    def extract_frames(self, video_path: str) -> torch.Tensor:
        """Extract frames from video and return as tensor."""
        cap = cv2.VideoCapture(video_path)
        frames = []

        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        indices = np.linspace(0, frame_count - 1, self.sequence_length, dtype=int)

        for i in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            if ret:
                frame = cv2.resize(frame, (224, 224))
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame)

        cap.release()

        if len(frames) < self.sequence_length:
            # Pad with last frame if needed
            last_frame = frames[-1] if frames else np.zeros((224, 224, 3))
            frames.extend([last_frame] * (self.sequence_length - len(frames)))

        frames = np.array(frames)  # (T, H, W, C)
        frames = torch.from_numpy(frames).float().permute(0, 3, 1, 2) / 255.0  # (T, C, H, W)

        return frames
