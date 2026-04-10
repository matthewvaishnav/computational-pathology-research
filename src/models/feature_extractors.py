"""Feature extraction models for histopathology images."""

from typing import Optional

import torch
import torch.nn as nn
from torchvision import models


class ResNetFeatureExtractor(nn.Module):
    """
    Feature extractor for histopathology patches using torchvision models.

    Supports multiple architectures from torchvision, removing the final
    classification layer to extract features before the FC layer.

    Args:
        model_name: Model variant ('resnet18', 'resnet50', 'densenet121', 'efficientnet_b0')
        pretrained: Whether to use ImageNet pretrained weights (default: True)
        feature_dim: Output feature dimension. If None, uses native backbone dimension.
                     If specified, adds a linear projection to match the desired dimension.
    """

    def __init__(
        self,
        model_name: str = "resnet18",
        pretrained: bool = True,
        feature_dim: Optional[int] = None,
    ):
        super().__init__()

        # Model registry with native feature dimensions
        model_registry = {
            "resnet18": (models.resnet18, 512, "IMAGENET1K_V1"),
            "resnet50": (models.resnet50, 2048, "IMAGENET1K_V1"),
            "densenet121": (models.densenet121, 1024, "IMAGENET1K_V1"),
            "efficientnet_b0": (models.efficientnet_b0, 1280, "IMAGENET1K_V1"),
        }

        if model_name not in model_registry:
            raise ValueError(
                f"Unknown model_name: {model_name}. "
                f"Supported models: {list(model_registry.keys())}"
            )

        model_fn, native_dim, weights_name = model_registry[model_name]
        self.model = model_fn(weights=weights_name if pretrained else None)
        self._native_feature_dim = native_dim

        # Remove the final classification layer
        # Different architectures have different final layer names
        if model_name.startswith("resnet"):
            self.model.fc = nn.Identity()
        elif model_name.startswith("densenet"):
            self.model.classifier = nn.Identity()
        elif model_name.startswith("efficientnet"):
            self.model.classifier = nn.Identity()

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
