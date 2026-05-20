"""
MIL Base Class for Multiple Instance Learning models.

This module provides a concrete base class that encapsulates common functionality
shared across AttentionMIL, CLAM, and TransMIL models. It reduces code duplication
by providing reusable methods for attention computation, feature aggregation, and
multimodal fusion.

The MILBase class is designed to work with the extracted fusion_strategies and
attention_mechanisms modules, providing a clean separation of concerns.

Example:
    >>> from src.models.mil.mil_base import MILBase
    >>> from src.models.components.attention_mechanisms import GatedAttention
    >>> from src.models.components.fusion_strategies import EarlyFusion
    >>>
    >>> # Create attention mechanism
    >>> attention = GatedAttention(feature_dim=256, hidden_dim=256)
    >>>
    >>> # Create fusion strategy (optional)
    >>> fusion = EarlyFusion(feature_dim=1024, hidden_dim=256, num_scales=2)
    >>>
    >>> # Create base model
    >>> model = MILBase(
    ...     feature_dim=1024,
    ...     num_classes=2,
    ...     attention=attention,
    ...     fusion=fusion
    ... )
"""

from typing import Dict, List, Optional, Union

import torch
import torch.nn as nn

from src.models.components.attention_mechanisms import AttentionMechanism
from src.models.components.fusion_strategies import FusionStrategy


class MILBase(nn.Module):
    """
    Base class for Multiple Instance Learning models.

    This class provides common functionality for MIL models including:
    - Attention weight computation
    - Feature aggregation using attention weights
    - Multimodal fusion (optional)

    The class is designed to be composable with different attention mechanisms
    and fusion strategies, allowing for flexible model architectures.

    Args:
        feature_dim: Dimension of input patch features (e.g., 1024 for ResNet50)
        num_classes: Number of output classes (2 for binary classification)
        attention: Attention mechanism to use for computing attention weights
        fusion: Optional fusion strategy for combining multi-scale/multimodal features

    Attributes:
        feature_dim: Dimension of input features
        num_classes: Number of output classes
        attention: Attention mechanism instance
        fusion: Fusion strategy instance (None if not using fusion)

    Example:
        >>> from src.models.components.attention_mechanisms import GatedAttention
        >>> attention = GatedAttention(feature_dim=256, hidden_dim=256)
        >>> model = MILBase(
        ...     feature_dim=1024,
        ...     num_classes=2,
        ...     attention=attention
        ... )
        >>> features = torch.randn(4, 100, 1024)
        >>> attention_weights = model.compute_attention(features)
        >>> attention_weights.shape
        torch.Size([4, 100])
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        attention: AttentionMechanism,
        fusion: Optional[FusionStrategy] = None,
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.attention = attention
        self.fusion = fusion

    def compute_attention(
        self, features: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute attention weights for each patch in the bag.

        This method delegates to the attention mechanism provided during initialization.
        It handles both simple attention mechanisms (that return weights directly) and
        transformer-based mechanisms (that return transformed features and weights).

        Args:
            features: Patch features [batch_size, num_patches, feature_dim]
            mask: Optional boolean mask [batch_size, num_patches]
                  True indicates valid patches, False indicates padding

        Returns:
            Attention weights [batch_size, num_patches] that sum to 1 for each slide

        Example:
            >>> model = MILBase(feature_dim=256, num_classes=2, attention=attention)
            >>> features = torch.randn(4, 100, 256)
            >>> mask = torch.ones(4, 100, dtype=torch.bool)
            >>> mask[0, 80:] = False  # First sample has only 80 valid patches
            >>> attention_weights = model.compute_attention(features, mask)
            >>> attention_weights.shape
            torch.Size([4, 100])
            >>> attention_weights[0, 80:].sum()  # Masked patches have ~0 weight
            tensor(0.)
        """
        result = self.attention(features, mask)

        # Handle different attention mechanism return types
        if isinstance(result, tuple):
            # Transformer attention returns (output, attention_weights)
            # We return the output for further processing
            return result[0]
        else:
            # Simple/Gated attention returns weights directly
            return result

    def aggregate_features(
        self, features: torch.Tensor, attention_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Aggregate patch features using attention weights to create slide representation.

        This method performs weighted sum of patch features using the attention weights.
        The result is a single feature vector per slide that represents the entire bag.

        Args:
            features: Patch features [batch_size, num_patches, feature_dim]
            attention_weights: Attention weights [batch_size, num_patches]
                              Should sum to 1 for each slide

        Returns:
            Aggregated slide representation [batch_size, feature_dim]

        Example:
            >>> model = MILBase(feature_dim=256, num_classes=2, attention=attention)
            >>> features = torch.randn(4, 100, 256)
            >>> attention_weights = torch.softmax(torch.randn(4, 100), dim=1)
            >>> aggregated = model.aggregate_features(features, attention_weights)
            >>> aggregated.shape
            torch.Size([4, 256])

        Note:
            This method assumes attention_weights has shape [batch_size, num_patches].
            If your attention mechanism returns weights with an extra dimension,
            you may need to squeeze or reshape before calling this method.
        """
        # Expand attention weights to match feature dimensions
        # [batch_size, num_patches] -> [batch_size, num_patches, 1]
        attention_weights = attention_weights.unsqueeze(-1)

        # Weighted sum: [batch_size, num_patches, feature_dim] * [batch_size, num_patches, 1]
        # -> [batch_size, num_patches, feature_dim] -> [batch_size, feature_dim]
        aggregated = torch.sum(features * attention_weights, dim=1)

        return aggregated

    def apply_fusion(
        self,
        features: Union[torch.Tensor, List[torch.Tensor]],
        mask: Optional[torch.Tensor] = None,
    ) -> Union[torch.Tensor, List[torch.Tensor]]:
        """
        Apply multimodal fusion if a fusion strategy is configured.

        This method applies the fusion strategy to combine features from multiple
        scales or modalities. If no fusion strategy is configured, it returns the
        input features unchanged.

        Args:
            features: Either a single tensor [batch_size, num_patches, feature_dim]
                     or a list of tensors (one per scale/modality)
            mask: Optional boolean mask [batch_size, num_patches]
                  True indicates valid patches, False indicates padding

        Returns:
            Fused features. The return type depends on the fusion strategy:
            - EarlyFusion: Single tensor [batch_size, num_patches, hidden_dim]
            - LateFusion: List of tensors [batch_size, num_patches, hidden_dim]
            - No fusion: Returns input unchanged

        Example:
            >>> # With early fusion
            >>> fusion = EarlyFusion(feature_dim=1024, hidden_dim=256, num_scales=2)
            >>> model = MILBase(
            ...     feature_dim=1024,
            ...     num_classes=2,
            ...     attention=attention,
            ...     fusion=fusion
            ... )
            >>> scale1 = torch.randn(4, 100, 1024)
            >>> scale2 = torch.randn(4, 100, 1024)
            >>> fused = model.apply_fusion([scale1, scale2])
            >>> fused.shape
            torch.Size([4, 100, 256])

            >>> # Without fusion
            >>> model_no_fusion = MILBase(
            ...     feature_dim=1024,
            ...     num_classes=2,
            ...     attention=attention
            ... )
            >>> features = torch.randn(4, 100, 1024)
            >>> output = model_no_fusion.apply_fusion(features)
            >>> output.shape
            torch.Size([4, 100, 1024])
        """
        if self.fusion is None:
            return features

        # If features is a single tensor, wrap it in a list for fusion
        if isinstance(features, torch.Tensor):
            features = [features]

        return self.fusion(features, mask)

    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Basic forward pass demonstrating the common MIL pipeline.

        This is a simple implementation that shows how the common methods work together.
        Subclasses will typically override this method to implement their specific
        architectures, but can still use the common methods provided by this base class.

        Args:
            features: Patch features [batch_size, num_patches, feature_dim]
            mask: Optional boolean mask [batch_size, num_patches]
            return_attention: If True, return attention weights in output dict

        Returns:
            If return_attention is False:
                Aggregated features [batch_size, feature_dim]
            If return_attention is True:
                Dictionary containing:
                - 'features': Aggregated features [batch_size, feature_dim]
                - 'attention_weights': Attention weights [batch_size, num_patches]

        Note:
            This basic implementation does not include classification layers.
            Subclasses should add their own feature extractors and classifiers.

        Example:
            >>> model = MILBase(feature_dim=256, num_classes=2, attention=attention)
            >>> features = torch.randn(4, 100, 256)
            >>> output = model(features, return_attention=True)
            >>> output['features'].shape
            torch.Size([4, 256])
            >>> output['attention_weights'].shape
            torch.Size([4, 100])
        """
        # Apply fusion if configured
        fused_features = self.apply_fusion(features, mask)

        # For early fusion, fused_features is a single tensor
        # For late fusion, it's a list - we'll just use the first one for this basic implementation
        if isinstance(fused_features, list):
            fused_features = fused_features[0] if fused_features[0] is not None else features

        # Compute attention weights
        attention_result = self.compute_attention(fused_features, mask)

        # Handle transformer attention (returns transformed features)
        # vs simple/gated attention (returns weights)
        if attention_result.dim() == 3:
            # Transformer attention: [batch_size, num_patches, feature_dim]
            # Extract attention weights by computing norm of features
            attention_weights = torch.norm(attention_result, dim=-1)
            attention_weights = torch.softmax(attention_weights, dim=1)
            features_to_aggregate = attention_result
        else:
            # Simple/Gated attention: [batch_size, num_patches]
            attention_weights = attention_result
            features_to_aggregate = fused_features

        # Aggregate features
        aggregated = self.aggregate_features(features_to_aggregate, attention_weights)

        if return_attention:
            return {
                "features": aggregated,
                "attention_weights": attention_weights,
            }
        else:
            return aggregated
