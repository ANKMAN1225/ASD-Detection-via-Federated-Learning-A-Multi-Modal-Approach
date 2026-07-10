"""
MultiModalModel - late fusion of facial MobileNet and behavioral TCN encoders.
"""

import torch
import torch.nn as nn

from .mobilenet_model import MobileNetFeatureExtractor
from .tcn_model import FusedModel, VideoTCNModel


class MultiModalModel(nn.Module):
    """Federated multi-modal model with late fusion for binary ASD classification."""

    def __init__(
        self,
        facial_model: MobileNetFeatureExtractor,
        behavioral_model: VideoTCNModel,
        fusion_model: FusedModel,
    ) -> None:
        super().__init__()
        self.facial_encoder = facial_model
        self.behavioral_encoder = behavioral_model
        self.fusion = fusion_model

    def forward(
        self,
        facial_x: torch.Tensor,
        behavioral_x: torch.Tensor,
    ) -> tuple:
        _, facial_emb = self.facial_encoder(facial_x)
        _, behavioral_emb = self.behavioral_encoder(behavioral_x)
        logits = self.fusion(facial_emb, behavioral_emb)
        return logits, (facial_emb, behavioral_emb)
