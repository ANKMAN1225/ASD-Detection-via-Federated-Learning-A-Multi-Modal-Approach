"""Baseline vision architectures, isolated from the project model package."""

from __future__ import annotations

import torch
import torch.nn as nn
from torchvision.models import (
    EfficientNet_B0_Weights,
    MobileNet_V2_Weights,
    ResNet18_Weights,
    efficientnet_b0,
    mobilenet_v2,
    resnet18,
)


class TinyVisionTransformer(nn.Module):
    """A compact ViT with 16x16 patches, suitable for small research datasets."""

    def __init__(
        self,
        num_classes: int,
        image_size: int = 224,
        patch_size: int = 16,
        embed_dim: int = 192,
        depth: int = 6,
        heads: int = 3,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if image_size % patch_size:
            raise ValueError("image_size must be divisible by patch_size for TinyViT.")
        patch_count = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.class_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.position_embedding = nn.Parameter(torch.zeros(1, patch_count + 1, embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            # Post-norm keeps PyTorch's nested-tensor fast path enabled.
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        nn.init.trunc_normal_(self.class_token, std=0.02)
        nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(inputs))

    def forward_features(self, inputs: torch.Tensor) -> torch.Tensor:
        tokens = self.patch_embed(inputs).flatten(2).transpose(1, 2)
        class_token = self.class_token.expand(inputs.shape[0], -1, -1)
        tokens = torch.cat((class_token, tokens), dim=1) + self.position_embedding
        return self.norm(self.encoder(tokens)[:, 0])


class FrameAverageClassifier(nn.Module):
    """Apply an image model per clip frame and average logits; no temporal layer."""

    def __init__(self, image_model: nn.Module) -> None:
        super().__init__()
        self.image_model = image_model

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim == 4:
            return self.image_model(inputs)
        if inputs.ndim != 5:
            raise ValueError("Expected image [B,C,H,W] or video [B,T,C,H,W] tensors.")
        batch_size, frames = inputs.shape[:2]
        logits = self.image_model(inputs.reshape(batch_size * frames, *inputs.shape[2:]))
        return logits.reshape(batch_size, frames, -1).mean(dim=1)


def build_model(
    architecture: str, num_classes: int, image_size: int, pretrained: bool
) -> nn.Module:
    """Construct a classifier with a common logits-only forward interface."""
    if architecture == "mobilenet_v2":
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        model = mobilenet_v2(weights=weights)
        model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    elif architecture == "resnet18":
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        model = resnet18(weights=weights)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
    elif architecture == "efficientnet_b0":
        weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
        model = efficientnet_b0(weights=weights)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)
    elif architecture == "tinyvit":
        model = TinyVisionTransformer(num_classes=num_classes, image_size=image_size)
    else:
        raise ValueError(f"Unsupported architecture: {architecture}")
    return FrameAverageClassifier(model)


class VisionEncoder(nn.Module):
    """Expose a pooled feature vector for a torchvision or TinyViT backbone."""

    def __init__(self, architecture: str, image_size: int, pretrained: bool) -> None:
        super().__init__()
        self.architecture = architecture
        if architecture == "mobilenet_v2":
            weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
            backbone = mobilenet_v2(weights=weights)
            self.features = backbone.features
            self.feature_dim = backbone.last_channel
            self.kind = "convolutional"
        elif architecture == "resnet18":
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            backbone = resnet18(weights=weights)
            self.features = nn.Sequential(*list(backbone.children())[:-1])
            self.feature_dim = backbone.fc.in_features
            self.kind = "convolutional"
        elif architecture == "efficientnet_b0":
            weights = EfficientNet_B0_Weights.DEFAULT if pretrained else None
            backbone = efficientnet_b0(weights=weights)
            self.features = backbone.features
            self.feature_dim = backbone.classifier[1].in_features
            self.kind = "convolutional"
        elif architecture == "tinyvit":
            self.features = TinyVisionTransformer(num_classes=2, image_size=image_size)
            self.feature_dim = self.features.head.in_features
            self.kind = "tinyvit"
        else:
            raise ValueError(f"Unsupported architecture: {architecture}")
        self.pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.kind == "tinyvit":
            return self.features.forward_features(inputs)
        return torch.flatten(self.pool(self.features(inputs)), 1)


class FusionBenchmarkModel(nn.Module):
    """Standalone late-fusion classifier for paired image/video benchmark data."""

    def __init__(
        self,
        architecture: str,
        num_classes: int,
        image_size: int,
        pretrained: bool,
        use_tcn: bool,
    ) -> None:
        super().__init__()
        self.facial_encoder = VisionEncoder(architecture, image_size, pretrained)
        self.video_encoder = VisionEncoder(architecture, image_size, pretrained)
        feature_dim = self.facial_encoder.feature_dim
        self.use_tcn = use_tcn
        if use_tcn:
            # A compact causal temporal stack over frame-level visual features.
            self.temporal = nn.Sequential(
                nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2),
                nn.Conv1d(feature_dim, feature_dim, kernel_size=3, padding=2, dilation=2),
                nn.ReLU(inplace=True),
            )
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * 2, feature_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(feature_dim, num_classes),
        )

    def forward(self, facial_inputs: torch.Tensor, video_inputs: torch.Tensor) -> torch.Tensor:
        facial_features = self.facial_encoder(facial_inputs)
        batch_size, frames = video_inputs.shape[:2]
        flat_video = video_inputs.reshape(batch_size * frames, *video_inputs.shape[2:])
        video_features = self.video_encoder(flat_video).reshape(batch_size, frames, -1)
        if self.use_tcn:
            temporal_features = self.temporal(video_features.transpose(1, 2))
            video_features = temporal_features.mean(dim=2)
        else:
            video_features = video_features.mean(dim=1)
        return self.classifier(torch.cat((facial_features, video_features), dim=1))


def build_fusion_model(
    architecture: str,
    num_classes: int,
    image_size: int,
    pretrained: bool,
    use_tcn: bool,
) -> nn.Module:
    """Build a fusion baseline with image and video inputs and facial ASD labels."""
    return FusionBenchmarkModel(
        architecture=architecture,
        num_classes=num_classes,
        image_size=image_size,
        pretrained=pretrained,
        use_tcn=use_tcn,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(parameter.numel() for parameter in model.parameters())
