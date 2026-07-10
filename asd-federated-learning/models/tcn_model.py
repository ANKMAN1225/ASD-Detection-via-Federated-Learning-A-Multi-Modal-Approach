"""
Temporal Convolutional Network (TCN) and VideoTCNModel for behavioral video classification.
Includes Chomp1d, TemporalBlock, TemporalConvNet, VideoTCNModel, and FusedModel for multi-modal fusion.
"""

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights



class Chomp1d(nn.Module):
    """Remove padding from the end of sequence."""

    def __init__(self, chomp_size: int) -> None:
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """Individual temporal block for TCN."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1,
            self.chomp1,
            self.relu1,
            self.dropout1,
            self.conv2,
            self.chomp2,
            self.relu2,
            self.dropout2,
        )

        self.downsample = (
            nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    """Temporal Convolutional Network for behavioral video analysis."""

    def __init__(
        self,
        input_dim: int,
        num_channels: List[int],
        kernel_size: int = 3,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2**i
            in_channels = input_dim if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            layers.append(
                TemporalBlock(
                    in_channels,
                    out_channels,
                    kernel_size,
                    stride=1,
                    dilation=dilation_size,
                    padding=(kernel_size - 1) * dilation_size,
                    dropout=dropout,
                )
            )

        self.network = nn.Sequential(*layers)
        self.feature_dim = num_channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class VideoTCNModel(nn.Module):
    """Complete model for processing behavioral videos with TCN."""

    def __init__(
        self,
        num_classes: int = 3,
        sequence_length: int = 16,
        tcn_channels: List[int] = None,
        pretrained: bool = True,
    ) -> None:
        super().__init__()
        if tcn_channels is None:
            tcn_channels = [64, 128, 256]

        # Feature extractor for individual frames
        if pretrained:
            weights = MobileNet_V2_Weights.DEFAULT
            self.frame_extractor = mobilenet_v2(weights=weights).features
        else:
            self.frame_extractor = mobilenet_v2(weights=None).features

        self.frame_avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # TCN for temporal modeling
        self.tcn = TemporalConvNet(input_dim=1280, num_channels=tcn_channels)

        # Classifier
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(tcn_channels[-1], 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes),
        )

        self.feature_dim = tcn_channels[-1]

    def forward(self, x: torch.Tensor) -> tuple:
        # x shape: (batch_size, sequence_length, channels, height, width)
        batch_size, seq_len = x.shape[:2]

        # Extract features from each frame
        x = x.view(batch_size * seq_len, *x.shape[2:])  # (batch_size * seq_len, C, H, W)
        frame_features = self.frame_extractor(x)
        frame_features = self.frame_avgpool(frame_features)
        frame_features = torch.flatten(frame_features, 1)  # (batch_size * seq_len, 1280)

        # Reshape for TCN
        frame_features = frame_features.view(
            batch_size, seq_len, -1
        )  # (batch_size, seq_len, 1280)
        frame_features = frame_features.transpose(1, 2)  # (batch_size, 1280, seq_len)

        # Apply TCN
        tcn_features = self.tcn(frame_features)  # (batch_size, tcn_channels[-1], seq_len)

        # Get embeddings before classification
        embeddings = F.adaptive_avg_pool1d(tcn_features, 1).squeeze(-1)

        # Classification
        output = self.classifier(tcn_features)

        return output, embeddings


class FusedModel(nn.Module):
    """Late fusion model combining facial and behavioral features."""

    def __init__(
        self,
        facial_feature_dim: int = 1280,
        behavioral_feature_dim: int = 256,
        num_classes: int = 2,
        fusion_type: str = "concat",
    ) -> None:
        super().__init__()

        self.fusion_type = fusion_type

        if fusion_type == "concat":
            fusion_dim = facial_feature_dim + behavioral_feature_dim
        elif fusion_type == "attention":
            fusion_dim = max(facial_feature_dim, behavioral_feature_dim)
            self.facial_projection = nn.Linear(facial_feature_dim, fusion_dim)
            self.behavioral_projection = nn.Linear(behavioral_feature_dim, fusion_dim)
            self.attention = nn.MultiheadAttention(
                fusion_dim, num_heads=8, batch_first=True
            )
        else:  # average or weighted average
            fusion_dim = facial_feature_dim
            self.behavioral_projection = nn.Linear(behavioral_feature_dim, fusion_dim)

        self.fusion_classifier = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )

        if fusion_type == "weighted":
            self.weight_facial = nn.Parameter(torch.tensor(0.5))
            self.weight_behavioral = nn.Parameter(torch.tensor(0.5))

    def forward(
        self, facial_features: torch.Tensor, behavioral_features: torch.Tensor
    ) -> torch.Tensor:
        if self.fusion_type == "concat":
            fused_features = torch.cat([facial_features, behavioral_features], dim=1)
        elif self.fusion_type == "attention":
            facial_features = self.facial_projection(facial_features)
            behavioral_features = self.behavioral_projection(behavioral_features)
            combined = torch.stack([facial_features, behavioral_features], dim=1)
            attended, _ = self.attention(combined, combined, combined)
            fused_features = attended.mean(dim=1)
        elif self.fusion_type == "weighted":
            behavioral_features = self.behavioral_projection(behavioral_features)
            weights = F.softmax(
                torch.stack([self.weight_facial, self.weight_behavioral]), dim=0
            )
            fused_features = weights[0] * facial_features + weights[1] * behavioral_features
        else:  # average
            behavioral_features = self.behavioral_projection(behavioral_features)
            fused_features = (facial_features + behavioral_features) / 2

        return self.fusion_classifier(fused_features)
