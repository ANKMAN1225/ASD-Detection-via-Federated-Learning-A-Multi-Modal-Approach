"""
MobileNetFeatureExtractor - MobileNetV2-based model for facial feature extraction and classification.
Used for 2-class ASD detection from facial images.
"""

import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights


class MobileNetFeatureExtractor(nn.Module):
    """MobileNetV2 for facial feature extraction.

    The first 14 of 19 feature blocks are frozen (low-level ImageNet
    features that transfer well and don't need retraining on small datasets).
    Only the last 5 blocks + custom classifier are trained, reducing trainable
    parameters from ~2.2M to ~0.5M for better generalisation on ~500
    samples/client.
    """

    def __init__(self, num_classes: int = 2, pretrained: bool = True) -> None:
        super().__init__()
        weights = MobileNet_V2_Weights.DEFAULT if pretrained else None
        self.backbone = mobilenet_v2(weights=weights)

        # Remove the final classifier
        self.features = self.backbone.features
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.feature_dim = 1280

        # Freeze first 14 of 19 feature blocks.
        # Low-level edge/texture detectors transfer well from ImageNet;
        # fine-tuning them on ~500 samples only causes overfitting.
        for i, layer in enumerate(self.features):
            if i < 14:
                for param in layer.parameters():
                    param.requires_grad = False

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
