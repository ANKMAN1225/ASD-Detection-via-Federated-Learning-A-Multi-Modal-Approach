"""Data loading and deterministic client partitioning for benchmarks only."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Sequence, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from .config import BenchmarkConfig


IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}
VIDEO_EXTENSIONS = {".mp4", ".avi", ".mov"}


class ImageFolderSplit(Dataset):
    """Minimal, sorted image-folder dataset with stable label assignment."""

    def __init__(self, split_dir: Path, classes: Sequence[str]) -> None:
        self.classes = list(classes)
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
        self.samples: List[Tuple[Path, int]] = []
        for class_name in self.classes:
            class_dir = split_dir / class_name
            if not class_dir.is_dir():
                raise FileNotFoundError(f"Expected class directory: {class_dir}")
            for path in sorted(class_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    self.samples.append((path, self.class_to_idx[class_name]))
        if not self.samples:
            raise ValueError(f"No supported images found in {split_dir}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[Image.Image, int]:
        path, label = self.samples[index]
        with Image.open(path) as image:
            return image.convert("RGB"), label


class VideoFolderDataset(Dataset):
    """Video dataset for the SSBD layout, implemented independently of data/."""

    CLASSES = ("armFlapping", "headBanging", "spinning")

    def __init__(self, root: Path, sequence_length: int) -> None:
        try:
            import cv2  # Imported lazily so facial benchmarks do not require OpenCV.
        except ImportError as exc:  # pragma: no cover - depends on environment
            raise RuntimeError("Video benchmarks require opencv-python.") from exc
        self.cv2 = cv2
        self.sequence_length = sequence_length
        self.classes = list(self.CLASSES)
        self.class_to_idx = {name: index for index, name in enumerate(self.classes)}
        self.samples: List[Tuple[Path, int]] = []
        for class_name in self.classes:
            nested = root / class_name / class_name
            flat = root / class_name
            class_dir = nested if nested.is_dir() else flat
            if not class_dir.is_dir():
                raise FileNotFoundError(
                    f"Expected '{class_name}' under {root} (flat or nested SSBD layout)."
                )
            for path in sorted(class_dir.iterdir()):
                if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
                    self.samples.append((path, self.class_to_idx[class_name]))
        if not self.samples:
            raise ValueError(f"No supported videos found in {root}.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[List[Image.Image], int]:
        path, label = self.samples[index]
        capture = self.cv2.VideoCapture(str(path))
        frame_count = max(1, int(capture.get(self.cv2.CAP_PROP_FRAME_COUNT)))
        frame_indices = np.linspace(0, frame_count - 1, self.sequence_length, dtype=int)
        frames: List[Image.Image] = []
        for frame_index in frame_indices:
            capture.set(self.cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if ok:
                frame = self.cv2.cvtColor(frame, self.cv2.COLOR_BGR2RGB)
                frames.append(Image.fromarray(frame))
        capture.release()
        if not frames:
            frames = [Image.new("RGB", (224, 224))]
        while len(frames) < self.sequence_length:
            frames.append(frames[-1].copy())
        return frames[: self.sequence_length], label


class IndexedTransformDataset(Dataset):
    """A fixed index view whose transform never changes the source dataset."""

    def __init__(
        self, dataset: Dataset, indices: Sequence[int], transform: Callable
    ) -> None:
        self.dataset = dataset
        self.indices = list(indices)
        self.transform = transform
        self.labels = [int(dataset.samples[i][1]) for i in self.indices]

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        data, label = self.dataset[self.indices[index]]
        if isinstance(data, list):
            data = torch.stack([self.transform(frame) for frame in data])
        else:
            data = self.transform(data)
        return data, torch.tensor(label, dtype=torch.long)


class PairedFusionDataset(Dataset):
    """Index-pair facial and video samples, matching the project's fusion rule.

    The source datasets have no subject-level correspondence.  Therefore pair
    ``i`` from each local modality and supervise it with the facial ASD label.
    This is explicit here so experiment reports do not imply clinical pairing.
    """

    def __init__(self, facial_dataset: Dataset, video_dataset: Dataset, seed: int) -> None:
        self.facial_dataset = facial_dataset
        self.video_dataset = video_dataset
        self.length = min(len(facial_dataset), len(video_dataset))
        if self.length == 0:
            raise ValueError("Fusion requires non-empty facial and video datasets.")
        rng = np.random.default_rng(seed)
        facial_labels = np.asarray(facial_dataset.labels, dtype=int)
        # Source folders are class-sorted.  A simple ``[:length]`` would
        # silently make a fusion split single-class whenever videos are fewer
        # than images.  Sample facial indices round-robin by class instead.
        groups = [rng.permutation(np.flatnonzero(facial_labels == class_id)).tolist()
                  for class_id in sorted(np.unique(facial_labels))]
        self.facial_indices: List[int] = []
        while len(self.facial_indices) < self.length:
            added = False
            for group in groups:
                if group and len(self.facial_indices) < self.length:
                    self.facial_indices.append(int(group.pop()))
                    added = True
            if not added:
                raise ValueError("Not enough facial samples to construct fusion pairs.")
        self.video_indices = rng.permutation(len(video_dataset))[: self.length].tolist()
        self.labels = [int(facial_dataset.labels[index]) for index in self.facial_indices]

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> Tuple[Tuple[torch.Tensor, torch.Tensor], torch.Tensor]:
        facial, label = self.facial_dataset[self.facial_indices[index]]
        video, _ = self.video_dataset[self.video_indices[index]]
        return (facial, video), label


@dataclass
class BenchmarkData:
    """Loaders and immutable dataset metadata used by the training engine."""

    central_train_loader: DataLoader
    client_train_loaders: List[DataLoader]
    validation_loader: DataLoader
    test_loader: DataLoader
    num_classes: int
    class_names: List[str]
    client_sample_counts: List[int]
    train_samples: int
    validation_samples: int
    test_samples: int


def _image_transform(config: BenchmarkConfig, training: bool) -> Callable:
    operations: List[Callable] = [transforms.Resize((config.image_size, config.image_size))]
    if training:
        operations.extend(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1),
            ]
        )
    operations.extend(
        [transforms.ToTensor(), transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)]
    )
    return transforms.Compose(operations)


def _stratified_split_indices(
    labels: Sequence[int], config: BenchmarkConfig
) -> Tuple[List[int], List[int], List[int]]:
    """Create reproducible train/validation/test video splits without leakage."""
    rng = np.random.default_rng(config.seed)
    grouped = {}
    for index, label in enumerate(labels):
        grouped.setdefault(int(label), []).append(index)
    train, validation, test = [], [], []
    for label_indices in grouped.values():
        shuffled = np.array(label_indices, dtype=int)
        rng.shuffle(shuffled)
        count = len(shuffled)
        train_count = int(round(count * config.train_fraction))
        validation_count = int(round(count * config.validation_fraction))
        if count >= 3:
            train_count = max(1, train_count)
            validation_count = max(1, validation_count)
            train_count = min(train_count, count - 2)
            validation_count = min(validation_count, count - train_count - 1)
        train.extend(shuffled[:train_count].tolist())
        validation.extend(shuffled[train_count : train_count + validation_count].tolist())
        test.extend(shuffled[train_count + validation_count :].tolist())
    if not train or not validation or not test:
        raise ValueError(
            "The video dataset is too small for the requested train/validation/test split. "
            "Use at least three videos per class."
        )
    return train, validation, test


def _limit_indices_per_class(
    labels: Sequence[int], maximum: int, seed: int
) -> List[int]:
    """Return a deterministic, balanced subset; zero means retain all samples."""
    if maximum == 0:
        return list(range(len(labels)))
    rng = np.random.default_rng(seed)
    selected: List[int] = []
    labels_array = np.asarray(labels)
    for class_id in np.unique(labels_array):
        indices = np.flatnonzero(labels_array == class_id)
        rng.shuffle(indices)
        selected.extend(indices[:maximum].tolist())
    return sorted(selected)


def partition_client_indices(
    labels: Sequence[int], num_clients: int, iid: bool, alpha: float, seed: int
) -> List[List[int]]:
    """Partition training indices once, preserving every sample exactly once."""
    if len(labels) < num_clients:
        raise ValueError("Training samples must be at least the number of clients.")
    rng = np.random.default_rng(seed)
    all_indices = np.arange(len(labels), dtype=int)
    partitions: List[List[int]] = [[] for _ in range(num_clients)]
    if iid:
        rng.shuffle(all_indices)
        partitions = [chunk.tolist() for chunk in np.array_split(all_indices, num_clients)]
    else:
        labels_array = np.asarray(labels)
        for class_id in np.unique(labels_array):
            class_indices = all_indices[labels_array == class_id].copy()
            rng.shuffle(class_indices)
            proportions = rng.dirichlet(np.full(num_clients, alpha))
            counts = rng.multinomial(len(class_indices), proportions)
            start = 0
            for client_id, count in enumerate(counts):
                partitions[client_id].extend(class_indices[start : start + count].tolist())
                start += count
    # A zero-sized client cannot produce a local update. Move one sample from
    # the largest client while retaining all samples and deterministic behavior.
    for client_id, partition in enumerate(partitions):
        if partition:
            continue
        donor = max(range(num_clients), key=lambda index: len(partitions[index]))
        if len(partitions[donor]) <= 1:
            raise ValueError("Unable to assign at least one sample to every client.")
        partitions[client_id].append(partitions[donor].pop())
    return partitions


def _loader(dataset: Dataset, config: BenchmarkConfig, shuffle: bool, seed: int) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=shuffle,
        num_workers=config.num_workers,
        pin_memory=config.device.startswith("cuda"),
        generator=generator,
    )


def _facial_data(config: BenchmarkConfig) -> Tuple[Dataset, Dataset, Dataset, List[str]]:
    root = config.data_path
    if not root.is_dir():
        raise FileNotFoundError(f"Facial dataset directory does not exist: {root}")
    classes = ["autistic", "non_autistic"]
    train_source = ImageFolderSplit(root / "train", classes)
    valid_source = ImageFolderSplit(root / "valid", classes)
    test_source = ImageFolderSplit(root / "test", classes)
    train_indices = _limit_indices_per_class(
        [label for _, label in train_source.samples], config.max_samples_per_class, config.seed
    )
    valid_indices = _limit_indices_per_class(
        [label for _, label in valid_source.samples], config.max_samples_per_class, config.seed + 1
    )
    test_indices = _limit_indices_per_class(
        [label for _, label in test_source.samples], config.max_samples_per_class, config.seed + 2
    )
    return (
        IndexedTransformDataset(train_source, train_indices, _image_transform(config, True)),
        IndexedTransformDataset(valid_source, valid_indices, _image_transform(config, False)),
        IndexedTransformDataset(test_source, test_indices, _image_transform(config, False)),
        classes,
    )


def _video_data(config: BenchmarkConfig) -> Tuple[Dataset, Dataset, Dataset, List[str]]:
    root = config.video_data_path if config.modality == "fusion" else config.data_path
    if not root.is_dir():
        raise FileNotFoundError(f"Video dataset directory does not exist: {root}")
    source = VideoFolderDataset(root, config.sequence_length)
    selected = _limit_indices_per_class(
        [label for _, label in source.samples], config.max_samples_per_class, config.seed
    )
    source.samples = [source.samples[index] for index in selected]
    labels = [label for _, label in source.samples]
    train_indices, valid_indices, test_indices = _stratified_split_indices(labels, config)
    return (
        IndexedTransformDataset(source, train_indices, _image_transform(config, True)),
        IndexedTransformDataset(source, valid_indices, _image_transform(config, False)),
        IndexedTransformDataset(source, test_indices, _image_transform(config, False)),
        source.classes,
    )


def build_benchmark_data(config: BenchmarkConfig) -> BenchmarkData:
    """Build a single shared split used by all selected baseline methods."""
    random.seed(config.seed)
    np.random.seed(config.seed)
    if config.modality == "fusion":
        facial_train, facial_validation, facial_test, classes = _facial_data(config)
        video_train, video_validation, video_test, _ = _video_data(config)
        train_set = PairedFusionDataset(facial_train, video_train, config.seed)
        validation_set = PairedFusionDataset(facial_validation, video_validation, config.seed + 1)
        test_set = PairedFusionDataset(facial_test, video_test, config.seed + 2)
        # Match the project's fusion procedure: partition each modality for
        # clients independently, then pair samples inside each client.
        facial_partitions = partition_client_indices(
            facial_train.labels,
            config.num_clients,
            config.iid,
            config.dirichlet_alpha,
            config.seed,
        )
        video_partitions = partition_client_indices(
            video_train.labels,
            config.num_clients,
            config.iid,
            config.dirichlet_alpha,
            config.seed + 10_000,
        )
        client_sets = []
        for client_id, (facial_indices, video_indices) in enumerate(
            zip(facial_partitions, video_partitions)
        ):
            client_facial = IndexedTransformDataset(
                facial_train.dataset,
                [facial_train.indices[index] for index in facial_indices],
                facial_train.transform,
            )
            client_video = IndexedTransformDataset(
                video_train.dataset,
                [video_train.indices[index] for index in video_indices],
                video_train.transform,
            )
            client_sets.append(
                PairedFusionDataset(client_facial, client_video, config.seed + client_id)
            )
    else:
        if config.modality == "facial":
            train_set, validation_set, test_set, classes = _facial_data(config)
        else:
            train_set, validation_set, test_set, classes = _video_data(config)
        partitions = partition_client_indices(
            train_set.labels, config.num_clients, config.iid, config.dirichlet_alpha, config.seed
        )
        client_sets = [
            IndexedTransformDataset(
                train_set.dataset,
                [train_set.indices[index] for index in indices],
                train_set.transform,
            )
            for indices in partitions
        ]
    return BenchmarkData(
        central_train_loader=_loader(train_set, config, True, config.seed),
        client_train_loaders=[
            _loader(dataset, config, True, config.seed + client_id + 1)
            for client_id, dataset in enumerate(client_sets)
        ],
        validation_loader=_loader(validation_set, config, False, config.seed + 101),
        test_loader=_loader(test_set, config, False, config.seed + 102),
        num_classes=len(classes),
        class_names=classes,
        client_sample_counts=[len(dataset) for dataset in client_sets],
        train_samples=len(train_set),
        validation_samples=len(validation_set),
        test_samples=len(test_set),
    )
