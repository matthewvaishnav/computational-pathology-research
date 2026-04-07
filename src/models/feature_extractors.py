"""Feature extraction models for histopathology images."""

from typing import Optional

import torch
import torch.nn as nn
from torchvision import models


class ResNetFeatureExtractor(nn.Module):
    """
    ResNet-based feature extractor for histopathology patches.

    Uses pretrained ResNet-18 or ResNet-50 from torchvision, removing the final
    classification layer to extract features before the FC layer.

    Args:
        model_name: ResNet variant ('resnet18', 'resnet50')
        pretrained: Whether to use ImageNet pretrained weights (default: True)
        feature_dim: Output feature dimension. If None, uses native backbone dimension
                     (512 for resnet18, 2048 for resnet50). If specified, adds a
                     linear projection to match the desired dimension.
    """

    def __init__(
        self,
        model_name: str = "resnet18",
        pretrained: bool = True,
        feature_dim: Optional[int] = None,
    ):
        super().__init__()

        if model_name == "resnet18":
            self.model = models.resnet18(weights="IMAGENET1K_V1" if pretrained else None)
            self._native_feature_dim = 512
        elif model_name == "resnet50":
            self.model = models.resnet50(weights="IMAGENET1K_V1" if pretrained else None)
            self._native_feature_dim = 2048
        else:
            raise ValueError(f"Unknown model_name: {model_name}. Use 'resnet18' or 'resnet50'.")

        # Remove the final classification layer (fc)
        self.model.fc = nn.Identity()

        # Set output dimension and add projection if needed
        if feature_dim is not None:
            self._feature_dim = feature_dim
            # Add projection layer if feature_dim differs from native
            if feature_dim != self._native_feature_dim:
                self.feature_proj = nn.Linear(self._native_feature_dim, feature_dim)
            else:
                self.feature_proj = None
        else:
            self._feature_dim = self._native_feature_dim
            self.feature_proj = None

    @property
    def feature_dim(self) -> int:
        """Output feature dimension."""
        return self._feature_dim

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract features from images.

        Args:
            images: [batch, 3, H, W] - assumes H=W=96 for PCam, will adaptive average pool

        Returns:
            features: [batch, feature_dim]
        """
        # Extract features through ResNet backbone
        features = self.model(images)

        # Apply projection if needed
        if self.feature_proj is not None:
            features = self.feature_proj(features)

        return features

    def get_num_params(self) -> int:
        """Return total number of parameters."""
        return sum(p.numel() for p in self.parameters())
