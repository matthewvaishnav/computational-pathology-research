"""
Attention-based Multiple Instance Learning (MIL) models for slide-level classification.

This module implements three state-of-the-art attention-based MIL architectures:
1. AttentionMIL: Basic attention-weighted pooling with gated attention mechanism
2. CLAM: Clustering-Constrained Attention MIL with instance-level clustering
3. TransMIL: Transformer-based MIL with multi-head self-attention

All models inherit from AttentionMILBase and work with pre-extracted patch features
stored in HDF5 format. They support variable-length bags (slides with different
numbers of patches) through masking and provide interpretable attention weights.

References:
- AttentionMIL: Ilse et al. "Attention-based Deep Multiple Instance Learning" (ICML 2018)
- CLAM: Lu et al. "Data-efficient and weakly supervised computational pathology on whole-slide images" (Nature Biomedical Engineering 2021)
- TransMIL: Shao et al. "TransMIL: Transformer based Correlated Multiple Instance Learning for Whole Slide Image Classification" (NeurIPS 2021)
"""

from abc import ABC, abstractmethod
from typing import Optional, Tuple

import torch
import torch.nn as nn

from src.models.components.attention_mechanisms import GatedAttention, SimpleAttention
from src.models.components.fusion_strategies import EarlyFusion, LateFusion
from src.models.mil.mil_base import MILBase


class AttentionMILBase(ABC, nn.Module):
    """
    Abstract base class for attention-based MIL models.

    This class defines the common interface that all attention-based MIL models
    must implement. It provides a unified API for training, inference, and
    attention weight extraction across different architectures.

    All subclasses must implement:
    - compute_attention: Calculate attention weights for patches
    - aggregate_features: Aggregate patch features using attention weights
    - forward: Complete forward pass from features to logits

    Args:
        feature_dim: Dimension of input patch features (e.g., 1024 for ResNet50)
        hidden_dim: Dimension of hidden layers in the model
        num_classes: Number of output classes (2 for binary classification)
        dropout: Dropout rate for regularization (default: 0.1)

    Example:
        >>> # Subclass must implement abstract methods
        >>> class MyAttentionMIL(AttentionMILBase):
        ...     def compute_attention(self, features, mask=None):
        ...         # Implementation here
        ...         pass
        ...     def aggregate_features(self, features, attention_weights):
        ...         # Implementation here
        ...         pass
        ...     def forward(self, features, num_patches=None, return_attention=False):
        ...         # Implementation here
        ...         pass
    """

    def __init__(self, feature_dim: int, hidden_dim: int, num_classes: int, dropout: float = 0.1):
        super().__init__()
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.dropout = dropout

    @abstractmethod
    def compute_attention(
        self, features: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute attention weights for each patch in the bag.

        Args:
            features: Patch features [batch_size, num_patches, feature_dim]
            mask: Boolean mask for valid patches [batch_size, num_patches]
                  True indicates valid patches, False indicates padding

        Returns:
            Attention weights [batch_size, num_patches] that sum to 1 for each slide

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclass must implement compute_attention")

    @abstractmethod
    def aggregate_features(
        self, features: torch.Tensor, attention_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Aggregate patch features using attention weights to create slide representation.

        Args:
            features: Patch features [batch_size, num_patches, feature_dim]
            attention_weights: Attention weights [batch_size, num_patches]

        Returns:
            Aggregated slide representation [batch_size, hidden_dim]

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclass must implement aggregate_features")

    @abstractmethod
    def forward(
        self,
        features: torch.Tensor,
        num_patches: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass from patch features to class logits.

        Args:
            features: Patch features [batch_size, num_patches, feature_dim]
            num_patches: Number of valid patches per slide [batch_size]
                        Used to create mask for variable-length bags
            return_attention: If True, return attention weights along with logits

        Returns:
            If return_attention is False:
                logits: Class logits [batch_size, num_classes]
            If return_attention is True:
                (logits, attention_weights): Tuple of logits and attention weights
                attention_weights: [batch_size, num_patches]

        Raises:
            NotImplementedError: Must be implemented by subclass
        """
        raise NotImplementedError("Subclass must implement forward")


class AttentionMIL(MILBase):
    """
    Attention-based MIL with gated attention mechanism.

    This model implements the attention-based pooling approach from Ilse et al. (2018).
    It uses a gated attention mechanism to compute importance weights for each patch,
    then aggregates patch features using these weights to create a slide-level
    representation.

    The refactored version inherits from MILBase and uses extracted components:
    - GatedAttention or SimpleAttention from attention_mechanisms.py
    - EarlyFusion or LateFusion from fusion_strategies.py (for multi-scale)

    Args:
        feature_dim: Dimension of input patch features (e.g., 1024 for ResNet50)
        hidden_dim: Dimension of hidden layers (default: 256)
        num_classes: Number of output classes (default: 2 for binary)
        dropout: Dropout rate (default: 0.1)
        gated: If True, use gated attention; if False, use simple attention (default: True)
        attention_mode: 'instance' or 'bag' level attention (default: 'instance')
        multi_scale: If True, support multi-scale features (default: False)
        num_scales: Number of scales for multi-scale features (default: 1)
        fusion_strategy: 'early' or 'late' fusion for multi-scale (default: 'early')

    Example:
        >>> model = AttentionMIL(feature_dim=1024, hidden_dim=256, num_classes=2)
        >>> features = torch.randn(4, 100, 1024)  # 4 slides, 100 patches each
        >>> num_patches = torch.tensor([100, 80, 90, 100])  # Actual patch counts
        >>> logits, attention = model(features, num_patches, return_attention=True)
        >>> logits.shape
        torch.Size([4, 2])
        >>> attention.shape
        torch.Size([4, 100])
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 2,
        dropout: float = 0.1,
        gated: bool = True,
        attention_mode: str = "instance",
        multi_scale: bool = False,
        num_scales: int = 1,
        fusion_strategy: str = "early",
    ):
        # Validate parameters
        if attention_mode not in ["instance", "bag"]:
            raise ValueError(f"attention_mode must be 'instance' or 'bag', got {attention_mode}")

        if fusion_strategy not in ["early", "late"]:
            raise ValueError(f"fusion_strategy must be 'early' or 'late', got {fusion_strategy}")

        # Create attention mechanism
        if gated:
            attention = GatedAttention(feature_dim=hidden_dim, hidden_dim=hidden_dim)
        else:
            attention = SimpleAttention(feature_dim=hidden_dim, hidden_dim=hidden_dim // 2)

        # Create fusion strategy if multi-scale
        fusion = None
        if multi_scale and num_scales > 1:
            if fusion_strategy == "early":
                fusion = EarlyFusion(
                    feature_dim=feature_dim,
                    hidden_dim=hidden_dim,
                    num_scales=num_scales,
                    dropout=dropout,
                )
            else:  # late fusion
                fusion = LateFusion(
                    feature_dim=feature_dim,
                    hidden_dim=hidden_dim,
                    num_scales=num_scales,
                    dropout=dropout,
                )

        # Initialize base class
        super().__init__(
            feature_dim=feature_dim,
            num_classes=num_classes,
            attention=attention,
            fusion=fusion,
        )

        # Store configuration
        self.hidden_dim = hidden_dim
        self.dropout = dropout
        self.gated = gated
        self.attention_mode = attention_mode
        self.multi_scale = multi_scale
        self.num_scales = num_scales
        self.fusion_strategy = fusion_strategy

        # Feature projection layer
        if multi_scale and num_scales > 1 and fusion_strategy == "late":
            # Late fusion uses scale-specific projections in the fusion strategy
            # No additional projection needed here
            self.feature_proj = None
        else:
            # Single projection for single-scale or early fusion
            self.feature_proj = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
            )

        # Classifier head
        classifier_input_dim = (
            hidden_dim * num_scales if (multi_scale and fusion_strategy == "late") else hidden_dim
        )
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(
        self,
        features: torch.Tensor,
        num_patches: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass from patch features to class logits.

        Args:
            features: Patch features [batch_size, num_patches, feature_dim] for single-scale
                     OR list of [batch_size, num_patches, feature_dim] tensors for multi-scale
            num_patches: Number of valid patches per slide [batch_size]
            return_attention: If True, return attention weights

        Returns:
            logits: Class logits [batch_size, num_classes]
            attention_weights: (optional) [batch_size, num_patches]
        """
        # Detect multi-scale input
        if isinstance(features, list):
            # Multi-scale input
            if not self.multi_scale:
                raise ValueError(
                    "Model was not initialized with multi_scale=True but received list of features"
                )

            # Get batch size and max patches from first non-None scale
            first_scale = next((f for f in features if f is not None), None)
            if first_scale is None:
                raise ValueError("All scales are None in multi-scale input")

            batch_size, max_patches, _ = first_scale.shape

            # Create mask from num_patches
            mask = None
            if num_patches is not None:
                mask = torch.arange(max_patches, device=first_scale.device).unsqueeze(
                    0
                ) < num_patches.unsqueeze(1)

            # Apply fusion strategy
            fused_features = self.apply_fusion(features, mask)

            # Handle early vs late fusion
            if self.fusion_strategy == "early":
                # Early fusion returns single tensor
                h = fused_features
                # Compute attention and aggregate
                attention_weights = self.compute_attention(h, mask)
                slide_repr = self.aggregate_features(h, attention_weights)
            else:
                # Late fusion returns list of tensors
                # Process each scale independently and concatenate
                scale_representations = []
                scale_attention_weights = []

                for scale_features in fused_features:
                    if scale_features is not None:
                        # Compute attention and aggregate for this scale
                        attn = self.compute_attention(scale_features, mask)
                        scale_repr = self.aggregate_features(scale_features, attn)
                        scale_representations.append(scale_repr)
                        scale_attention_weights.append(attn)
                    else:
                        # Handle missing scale
                        device = next(self.parameters()).device
                        scale_representations.append(
                            torch.zeros(batch_size, self.hidden_dim, device=device)
                        )

                # Concatenate scale representations
                slide_repr = torch.cat(scale_representations, dim=-1)

                # Average attention weights for visualization
                if scale_attention_weights:
                    attention_weights = torch.stack(scale_attention_weights, dim=0).mean(dim=0)
                else:
                    attention_weights = torch.zeros(batch_size, max_patches, device=device)

            # Classify
            logits = self.classifier(slide_repr)

            if return_attention:
                return logits, attention_weights
            else:
                return logits
        else:
            # Single-scale input
            batch_size, max_patches, _ = features.shape

            # Create mask from num_patches
            mask = None
            if num_patches is not None:
                mask = torch.arange(max_patches, device=features.device).unsqueeze(
                    0
                ) < num_patches.unsqueeze(1)

            # Project features
            if self.feature_proj is not None:
                h = self.feature_proj(features)
            else:
                h = features

            # Compute attention weights
            attention_weights = self.compute_attention(h, mask)

            # Aggregate features
            slide_repr = self.aggregate_features(h, attention_weights)

            # Classify
            logits = self.classifier(slide_repr)

            if return_attention:
                return logits, attention_weights
            else:
                return logits


# create_attention_model has been extracted to src/models/factory.py
# Import it from there to maintain backward compatibility
from src.models.factory import create_attention_model  # noqa: F401

# CLAM has been extracted to src/models/clam.py
# Import it from there to maintain backward compatibility
from src.models.mil.clam import CLAM  # noqa: F401

# TransMIL has been extracted to src/models/transmil.py
# Import it from there to maintain backward compatibility
from src.models.mil.transmil import TransMIL  # noqa: F401
