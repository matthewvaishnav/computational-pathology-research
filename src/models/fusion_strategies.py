"""
Multimodal fusion strategies for Multiple Instance Learning models.

This module provides fusion strategies for combining features from multiple scales
or modalities in MIL models. Two main strategies are implemented:

1. EarlyFusion: Concatenates features from all scales/modalities before attention
   computation, creating a unified representation.

2. LateFusion: Processes each scale/modality independently with separate attention
   mechanisms, then combines the outputs.

These strategies are extracted from the original attention_mil.py to reduce code
duplication across AttentionMIL, CLAM, and TransMIL models.

Example:
    >>> # Early fusion for combining multi-scale features
    >>> fusion = EarlyFusion(
    ...     feature_dim=1024,
    ...     hidden_dim=256,
    ...     num_scales=3
    ... )
    >>> scale_features = [
    ...     torch.randn(4, 100, 1024),  # Scale 1
    ...     torch.randn(4, 100, 1024),  # Scale 2
    ...     torch.randn(4, 100, 1024),  # Scale 3
    ... ]
    >>> fused = fusion(scale_features)
    >>> fused.shape
    torch.Size([4, 100, 256])
"""

from abc import ABC, abstractmethod
from typing import List, Optional

import torch
import torch.nn as nn


class FusionStrategy(ABC, nn.Module):
    """
    Abstract base class for multimodal fusion strategies.

    All fusion strategies must implement the forward method which takes a list
    of feature tensors (one per scale/modality) and returns a fused representation.

    Args:
        feature_dim: Dimension of input features for each scale/modality
        hidden_dim: Dimension of output fused features
        num_scales: Number of scales/modalities to fuse
    """

    def __init__(self, feature_dim: int, hidden_dim: int, num_scales: int):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_scales = num_scales

    @abstractmethod
    def forward(
        self, multi_scale_features: List[torch.Tensor], mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Fuse features from multiple scales/modalities.

        Args:
            multi_scale_features: List of [batch_size, num_patches, feature_dim] tensors,
                                 one per scale/modality. None values indicate missing scales.
            mask: Optional boolean mask [batch_size, num_patches], True for valid patches

        Returns:
            Fused features tensor

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclass must implement forward")


class EarlyFusion(FusionStrategy):
    """
    Early fusion strategy: concatenate and fuse features before attention.

    This strategy projects features from each scale/modality to a common dimension,
    concatenates them, and applies average pooling to create a unified representation.
    The unified representation is then used for attention computation.

    Early fusion allows the attention mechanism to consider relationships between
    features from different scales/modalities jointly.

    Args:
        feature_dim: Dimension of input features for each scale/modality
        hidden_dim: Dimension of output fused features
        num_scales: Number of scales/modalities to fuse
        dropout: Dropout rate for regularization (default: 0.1)

    Example:
        >>> fusion = EarlyFusion(feature_dim=1024, hidden_dim=256, num_scales=2)
        >>> features = [
        ...     torch.randn(4, 100, 1024),  # Scale 1
        ...     torch.randn(4, 100, 1024),  # Scale 2
        ... ]
        >>> fused = fusion(features)
        >>> fused.shape
        torch.Size([4, 100, 256])
    """

    def __init__(self, feature_dim: int, hidden_dim: int, num_scales: int, dropout: float = 0.1):
        super().__init__(feature_dim, hidden_dim, num_scales)

        # Scale-specific feature projection layers
        self.projections = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
                for _ in range(num_scales)
            ]
        )

    def forward(
        self, multi_scale_features: List[torch.Tensor], mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply early fusion to multi-scale features.

        Args:
            multi_scale_features: List of [batch_size, num_patches, feature_dim] tensors
            mask: Optional boolean mask [batch_size, num_patches]

        Returns:
            Fused features [batch_size, num_patches, hidden_dim]

        Raises:
            ValueError: If all scales are None
        """
        # Project features for each scale
        projected_features = []
        for scale_idx, scale_features in enumerate(multi_scale_features):
            if scale_features is not None:
                h = self.projections[scale_idx](scale_features)
                projected_features.append(h)

        if not projected_features:
            raise ValueError("All scales are None in multi-scale input")

        # Concatenate along feature dimension
        # [batch_size, num_patches, hidden_dim * num_valid_scales]
        h_concat = torch.cat(projected_features, dim=-1)

        # Average pooling to get back to hidden_dim
        # [batch_size, num_patches, hidden_dim]
        h = h_concat.view(
            h_concat.size(0), h_concat.size(1), len(projected_features), self.hidden_dim
        ).mean(dim=2)

        return h


class LateFusion(FusionStrategy):
    """
    Late fusion strategy: process scales independently, then concatenate outputs.

    This strategy processes each scale/modality independently (with separate attention
    mechanisms in the parent model), then concatenates the resulting representations.
    Each scale maintains its own feature projection.

    Late fusion allows each scale/modality to develop specialized representations
    before combination, which can be beneficial when scales have distinct characteristics.

    Args:
        feature_dim: Dimension of input features for each scale/modality
        hidden_dim: Dimension of output features per scale
        num_scales: Number of scales/modalities to fuse
        dropout: Dropout rate for regularization (default: 0.1)

    Note:
        The output dimension is hidden_dim * num_scales since each scale's
        representation is concatenated.

    Example:
        >>> fusion = LateFusion(feature_dim=1024, hidden_dim=256, num_scales=2)
        >>> features = [
        ...     torch.randn(4, 100, 1024),  # Scale 1
        ...     torch.randn(4, 100, 1024),  # Scale 2
        ... ]
        >>> # Late fusion returns list of projected features, one per scale
        >>> projected = fusion(features)
        >>> len(projected)
        2
        >>> projected[0].shape
        torch.Size([4, 100, 256])
    """

    def __init__(self, feature_dim: int, hidden_dim: int, num_scales: int, dropout: float = 0.1):
        super().__init__(feature_dim, hidden_dim, num_scales)

        # Scale-specific feature projection layers
        self.projections = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(feature_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout))
                for _ in range(num_scales)
            ]
        )

    def forward(
        self, multi_scale_features: List[torch.Tensor], mask: Optional[torch.Tensor] = None
    ) -> List[torch.Tensor]:
        """
        Apply late fusion to multi-scale features.

        Args:
            multi_scale_features: List of [batch_size, num_patches, feature_dim] tensors
            mask: Optional boolean mask [batch_size, num_patches] (not used in late fusion)

        Returns:
            List of projected features [batch_size, num_patches, hidden_dim], one per scale.
            None values are preserved for missing scales.

        Note:
            Unlike early fusion, late fusion returns a list of tensors rather than
            a single concatenated tensor. The parent model is responsible for
            processing each scale independently and combining the results.
        """
        projected_features = []
        for scale_idx, scale_features in enumerate(multi_scale_features):
            if scale_features is not None:
                h = self.projections[scale_idx](scale_features)
                projected_features.append(h)
            else:
                # Preserve None for missing scales
                projected_features.append(None)

        return projected_features
