"""
MobileNetFeatureExtractor - MobileNetV2-based model for facial feature extraction and classification.
Used for 2-class ASD detection from facial images.
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2


class MobileNetFeatureExtractor(nn.Module):
    """MobileNetV2 for facial feature extraction."""

    def __init__(self, num_classes: int = 2, pretrained: bool = True) -> None:
        super().__init__()
        self.backbone = mobilenet_v2(pretrained=pretrained)

        # Remove the final classifier
        self.features = self.backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = 1280

        # Add custom classifier
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(self.feature_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

    def forward(self, x: torch.Tensor) -> tuple:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        embeddings = x  # Store embeddings for fusion
        x = self.classifier(x)
        return x, embeddings
