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
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from .attention_mechanisms import GatedAttention, SimpleAttention
from .fusion_strategies import EarlyFusion, LateFusion
from .mil_base import MILBase


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



# CLAM has been extracted to src/models/clam.py
# Import it from there to maintain backward compatibility
from .clam import CLAM  # noqa: F401

class TransMIL(AttentionMILBase):
    """
    Transformer-based Multiple Instance Learning (TransMIL).

    TransMIL uses transformer encoder layers with multi-head self-attention to model
    relationships between patches in a slide. Unlike traditional attention-based MIL
    which computes attention weights independently for each patch, TransMIL allows
    patches to attend to each other, capturing spatial and contextual relationships.

    The model uses a learnable CLS token (similar to BERT) that aggregates information
    from all patches through self-attention. The CLS token representation is then
    used for slide-level classification.

    Optional positional encoding can be added to preserve spatial information about
    patch locations within the slide.

    Multi-scale support allows processing features from multiple magnification levels:
    - Early fusion: Concatenates scale features before transformer processing
    - Late fusion: Separate transformers per scale, then concatenates CLS tokens

    Args:
        feature_dim: Dimension of input patch features
        hidden_dim: Dimension of transformer hidden layers (default: 256)
        num_classes: Number of output classes (default: 2)
        num_layers: Number of transformer encoder layers (default: 2)
        num_heads: Number of attention heads (default: 8)
        dropout: Dropout rate (default: 0.1)
        use_pos_encoding: If True, add positional encoding (default: True)
        multi_scale: If True, support multi-scale features (default: False)
        num_scales: Number of scales for multi-scale features (default: 1)
        fusion_strategy: 'early' or 'late' fusion for multi-scale (default: 'early')

    Example:
        >>> # Single-scale TransMIL
        >>> model = TransMIL(feature_dim=1024, hidden_dim=256, num_layers=2, num_heads=8)
        >>> features = torch.randn(4, 100, 1024)
        >>> num_patches = torch.tensor([100, 80, 90, 100])
        >>> logits, attention = model(features, num_patches, return_attention=True)
        >>> logits.shape
        torch.Size([4, 2])
        >>> attention.shape  # Uniform weights (transformer attention is internal)
        torch.Size([4, 100])

        >>> # Multi-scale TransMIL with late fusion
        >>> model = TransMIL(feature_dim=1024, multi_scale=True, num_scales=3, fusion_strategy='late')
        >>> features_scale1 = torch.randn(4, 100, 1024)
        >>> features_scale2 = torch.randn(4, 100, 1024)
        >>> features_scale3 = torch.randn(4, 100, 1024)
        >>> multi_scale_features = [features_scale1, features_scale2, features_scale3]
        >>> logits = model(multi_scale_features, num_patches, return_attention=False)
        >>> logits.shape
        torch.Size([4, 2])
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 2,
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
        use_pos_encoding: bool = True,
        multi_scale: bool = False,
        num_scales: int = 1,
        fusion_strategy: str = "early",
    ):
        super().__init__(feature_dim, hidden_dim, num_classes, dropout)

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )

        if fusion_strategy not in ["early", "late"]:
            raise ValueError(f"fusion_strategy must be 'early' or 'late', got {fusion_strategy}")

        self.num_layers = num_layers
        self.num_heads = num_heads
        self.use_pos_encoding = use_pos_encoding
        self.multi_scale = multi_scale
        self.num_scales = num_scales
        self.fusion_strategy = fusion_strategy

        # Feature projection layer(s)
        if multi_scale and num_scales > 1:
            # Scale-specific feature projection layers
            self.feature_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(feature_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
                    )
                    for _ in range(num_scales)
                ]
            )
        else:
            # Single feature projection layer
            self.feature_proj = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
            )

        # Learnable positional encoding (max 10000 patches)
        if use_pos_encoding:
            if multi_scale and num_scales > 1:
                # Scale-specific positional encodings
                self.pos_encoding = nn.ParameterList(
                    [
                        nn.Parameter(torch.randn(1, 10000, hidden_dim) * 0.02)
                        for _ in range(num_scales)
                    ]
                )
            else:
                # Single positional encoding
                self.pos_encoding = nn.Parameter(torch.randn(1, 10000, hidden_dim) * 0.02)

        # Learnable CLS token
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # Transformer encoder(s)
        if multi_scale and num_scales > 1 and fusion_strategy == "late":
            # Scale-specific transformers for late fusion
            self.transformer = nn.ModuleList(
                [
                    self._create_transformer(hidden_dim, num_heads, num_layers, dropout)
                    for _ in range(num_scales)
                ]
            )
        else:
            # Single transformer (for early fusion or single scale)
            self.transformer = self._create_transformer(hidden_dim, num_heads, num_layers, dropout)

        # Layer normalization
        if multi_scale and num_scales > 1 and fusion_strategy == "late":
            # Scale-specific layer norms for late fusion
            self.norm = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_scales)])
        else:
            # Single layer norm
            self.norm = nn.LayerNorm(hidden_dim)

        # Classifier head
        # For late fusion, input is concatenated CLS tokens from all scales
        classifier_input_dim = (
            hidden_dim * num_scales if (multi_scale and fusion_strategy == "late") else hidden_dim
        )
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def _create_transformer(
        self, hidden_dim: int, num_heads: int, num_layers: int, dropout: float
    ) -> nn.TransformerEncoder:
        """Create a transformer encoder."""
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        return nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    def _early_fusion_transmil(
        self,
        multi_scale_features: list,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Early fusion: concatenate features from all scales before transformer.

        Args:
            multi_scale_features: List of [batch_size, num_patches, feature_dim] tensors, one per scale
            mask: Boolean mask [batch_size, num_patches], True for valid patches

        Returns:
            cls_repr: CLS token representation [batch_size, hidden_dim]
            attention_weights: Uniform attention weights [batch_size, num_patches]
        """
        # Get batch size and max patches from first non-None scale
        first_scale = next((f for f in multi_scale_features if f is not None), None)
        batch_size, max_patches, _ = first_scale.shape

        # Project features for each scale
        projected_features = []
        for scale_idx, scale_features in enumerate(multi_scale_features):
            if scale_features is not None:
                # Use scale-specific projection if available
                if isinstance(self.feature_proj, nn.ModuleList):
                    h = self.feature_proj[scale_idx](scale_features)
                else:
                    h = self.feature_proj(scale_features)
                projected_features.append(h)

        # Concatenate along feature dimension and average to get back to hidden_dim
        # [batch_size, num_patches, hidden_dim * num_scales] -> [batch_size, num_patches, hidden_dim]
        h_concat = torch.cat(projected_features, dim=-1)
        h = h_concat.view(
            h_concat.size(0), h_concat.size(1), len(projected_features), self.hidden_dim
        ).mean(dim=2)

        # Add positional encoding
        if self.use_pos_encoding:
            # Use first positional encoding (or single one if not ModuleList)
            if isinstance(self.pos_encoding, nn.ParameterList):
                h = h + self.pos_encoding[0][:, :max_patches, :]
            else:
                h = h + self.pos_encoding[:, :max_patches, :]

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, hidden_dim]
        h = torch.cat([cls_tokens, h], dim=1)  # [batch_size, num_patches+1, hidden_dim]

        # Create attention mask for transformer
        if mask is not None:
            # Create mask: False for valid positions (CLS + valid patches), True for padding
            transformer_mask = ~mask  # Invert: True for padding
            # Prepend False for CLS token
            cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=first_scale.device)
            transformer_mask = torch.cat([cls_mask, transformer_mask], dim=1)
        else:
            transformer_mask = None

        # Apply transformer (use single transformer for early fusion)
        if isinstance(self.transformer, nn.ModuleList):
            h = self.transformer[0](h, src_key_padding_mask=transformer_mask)
        else:
            h = self.transformer(h, src_key_padding_mask=transformer_mask)

        # Extract CLS token representation
        cls_repr = h[:, 0, :]  # [batch_size, hidden_dim]

        # Apply layer normalization
        if isinstance(self.norm, nn.ModuleList):
            cls_repr = self.norm[0](cls_repr)
        else:
            cls_repr = self.norm(cls_repr)

        # Return uniform attention weights for API compatibility
        attention_weights = self.compute_attention(first_scale, mask)

        return cls_repr, attention_weights

    def _late_fusion_transmil(
        self,
        multi_scale_features: list,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Late fusion: separate transformers per scale, then concatenate CLS tokens.

        Args:
            multi_scale_features: List of [batch_size, num_patches, feature_dim] tensors, one per scale
            mask: Boolean mask [batch_size, num_patches], True for valid patches

        Returns:
            cls_repr: Concatenated CLS token representations [batch_size, hidden_dim * num_scales]
            attention_weights: Average uniform attention weights [batch_size, num_patches]
        """
        scale_cls_representations = []

        for scale_idx, scale_features in enumerate(multi_scale_features):
            if scale_features is None:
                # Handle missing scale: use zeros
                batch_size = (
                    multi_scale_features[0].size(0) if multi_scale_features[0] is not None else 1
                )
                device = next(self.parameters()).device
                scale_cls_representations.append(
                    torch.zeros(batch_size, self.hidden_dim, device=device)
                )
                continue

            batch_size, max_patches, _ = scale_features.shape

            # Project features for this scale
            if isinstance(self.feature_proj, nn.ModuleList):
                h = self.feature_proj[scale_idx](scale_features)
            else:
                h = self.feature_proj(scale_features)

            # Add scale-specific positional encoding
            if self.use_pos_encoding:
                if isinstance(self.pos_encoding, nn.ParameterList):
                    h = h + self.pos_encoding[scale_idx][:, :max_patches, :]
                else:
                    h = h + self.pos_encoding[:, :max_patches, :]

            # Prepend CLS token
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, hidden_dim]
            h = torch.cat([cls_tokens, h], dim=1)  # [batch_size, num_patches+1, hidden_dim]

            # Create attention mask for transformer
            if mask is not None:
                # Create mask: False for valid positions (CLS + valid patches), True for padding
                transformer_mask = ~mask  # Invert: True for padding
                # Prepend False for CLS token
                cls_mask = torch.zeros(
                    batch_size, 1, dtype=torch.bool, device=scale_features.device
                )
                transformer_mask = torch.cat([cls_mask, transformer_mask], dim=1)
            else:
                transformer_mask = None

            # Apply scale-specific transformer
            if isinstance(self.transformer, nn.ModuleList):
                h = self.transformer[scale_idx](h, src_key_padding_mask=transformer_mask)
            else:
                h = self.transformer(h, src_key_padding_mask=transformer_mask)

            # Extract CLS token representation
            cls_repr = h[:, 0, :]  # [batch_size, hidden_dim]

            # Apply scale-specific layer normalization
            if isinstance(self.norm, nn.ModuleList):
                cls_repr = self.norm[scale_idx](cls_repr)
            else:
                cls_repr = self.norm(cls_repr)

            scale_cls_representations.append(cls_repr)

        # Concatenate CLS representations from all scales
        cls_repr = torch.cat(scale_cls_representations, dim=-1)

        # Return uniform attention weights for API compatibility
        first_scale = next((f for f in multi_scale_features if f is not None), None)
        attention_weights = self.compute_attention(first_scale, mask)

        return cls_repr, attention_weights

    def compute_attention(
        self, features: torch.Tensor, mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Return uniform attention weights as placeholder.

        Note: TransMIL uses internal transformer attention which is not directly
        exposed. This method returns uniform weights for API compatibility.

        Args:
            features: Patch features [batch_size, num_patches, feature_dim]
            mask: Boolean mask [batch_size, num_patches]

        Returns:
            Uniform attention weights [batch_size, num_patches]
        """
        batch_size, num_patches, _ = features.shape

        # Create uniform weights
        attention_weights = torch.ones(batch_size, num_patches, device=features.device)

        # Apply mask if provided
        if mask is not None:
            attention_weights = attention_weights.masked_fill(~mask, 0.0)

        # Normalize to sum to 1
        attention_weights = attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)

        return attention_weights

    def aggregate_features(
        self, features: torch.Tensor, attention_weights: torch.Tensor
    ) -> torch.Tensor:
        """
        Not used in TransMIL (aggregation happens via CLS token).

        Included for API compatibility with AttentionMILBase.
        """
        # Project features
        h = self.feature_proj(features)

        # Simple weighted sum (not actually used in forward pass)
        aggregated = torch.bmm(attention_weights.unsqueeze(1), h).squeeze(1)

        return aggregated

    def forward(
        self,
        features: torch.Tensor,
        num_patches: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass through TransMIL model.

        Args:
            features: Patch features [batch_size, num_patches, feature_dim] for single-scale
                     OR list of [batch_size, num_patches, feature_dim] tensors for multi-scale
            num_patches: Number of valid patches per slide [batch_size]
            return_attention: If True, return uniform attention weights

        Returns:
            logits: Class logits [batch_size, num_classes]
            attention_weights: (optional) Uniform weights [batch_size, num_patches]
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
            if self.fusion_strategy == "early":
                cls_repr, attention_weights = self._early_fusion_transmil(features, mask)
            elif self.fusion_strategy == "late":
                cls_repr, attention_weights = self._late_fusion_transmil(features, mask)
            else:
                raise ValueError(f"Unknown fusion_strategy: {self.fusion_strategy}")

            # Classify
            logits = self.classifier(cls_repr)

            if return_attention:
                return logits, attention_weights
            else:
                return logits
        else:
            # Single-scale input (original behavior)
            batch_size, max_patches, _ = features.shape

            # Project features
            if isinstance(self.feature_proj, nn.ModuleList):
                h = self.feature_proj[0](features)
            else:
                h = self.feature_proj(features)  # [batch_size, num_patches, hidden_dim]

            # Add positional encoding if enabled
            if self.use_pos_encoding:
                if isinstance(self.pos_encoding, nn.ParameterList):
                    h = h + self.pos_encoding[0][:, :max_patches, :]
                else:
                    h = h + self.pos_encoding[:, :max_patches, :]

            # Prepend CLS token
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)  # [batch_size, 1, hidden_dim]
            h = torch.cat([cls_tokens, h], dim=1)  # [batch_size, num_patches+1, hidden_dim]

            # Create attention mask for transformer
            # True indicates positions that should be masked (padding)
            if num_patches is not None:
                # Create mask: False for valid positions (CLS + valid patches), True for padding
                transformer_mask = torch.arange(max_patches, device=features.device).unsqueeze(
                    0
                ) >= num_patches.unsqueeze(1)
                # Prepend False for CLS token
                cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=features.device)
                transformer_mask = torch.cat(
                    [cls_mask, transformer_mask], dim=1
                )  # [batch_size, num_patches+1]

                # Create mask for compute_attention (True for valid patches)
                patch_mask = torch.arange(max_patches, device=features.device).unsqueeze(
                    0
                ) < num_patches.unsqueeze(1)
            else:
                transformer_mask = None
                patch_mask = None

            # Apply transformer encoder
            if isinstance(self.transformer, nn.ModuleList):
                h = self.transformer[0](h, src_key_padding_mask=transformer_mask)
            else:
                h = self.transformer(
                    h, src_key_padding_mask=transformer_mask
                )  # [batch_size, num_patches+1, hidden_dim]

            # Extract CLS token representation
            cls_repr = h[:, 0, :]  # [batch_size, hidden_dim]

            # Apply layer normalization
            if isinstance(self.norm, nn.ModuleList):
                cls_repr = self.norm[0](cls_repr)
            else:
                cls_repr = self.norm(cls_repr)

            # Classify
            logits = self.classifier(cls_repr)

            if return_attention:
                # Return uniform attention weights for API compatibility
                attention_weights = self.compute_attention(features, patch_mask)
                return logits, attention_weights
            else:
                return logits


def create_attention_model(config: Dict, feature_dim: int = 1024) -> nn.Module:
    """
    Factory function to create attention-based MIL models from configuration.

    This function reads the model configuration and instantiates the appropriate
    attention-based MIL model (AttentionMIL, CLAM, or TransMIL). It also supports
    baseline pooling models (mean, max) for comparison.

    Args:
        config: Configuration dictionary with model parameters
        feature_dim: Dimension of input patch features (default: 1024)

    Returns:
        Instantiated model (AttentionMIL, CLAM, TransMIL, or baseline)

    Raises:
        ValueError: If model_type is invalid or required config is missing

    Example:
        >>> config = {
        ...     'model_type': 'attention_mil',
        ...     'hidden_dim': 256,
        ...     'num_classes': 2,
        ...     'attention_mil': {
        ...         'gated': True,
        ...         'attention_mode': 'instance'
        ...     }
        ... }
        >>> model = create_attention_model(config, feature_dim=1024)
    """
    import logging

    logger = logging.getLogger(__name__)

    model_type = config.get("model_type", "mean")
    hidden_dim = config.get("hidden_dim", 256)
    num_classes = config.get("num_classes", 2)
    dropout = config.get("dropout", 0.1)

    logger.info(
        f"Creating model: type={model_type}, feature_dim={feature_dim}, "
        f"hidden_dim={hidden_dim}, num_classes={num_classes}"
    )

    if model_type == "attention_mil":
        # AttentionMIL configuration
        attention_config = config.get("attention_mil", {})
        model = AttentionMIL(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            gated=attention_config.get("gated", True),
            attention_mode=attention_config.get("attention_mode", "instance"),
        )
        logger.info(
            f"AttentionMIL created: gated={attention_config.get('gated', True)}, "
            f"mode={attention_config.get('attention_mode', 'instance')}"
        )

    elif model_type == "clam":
        # CLAM configuration
        clam_config = config.get("clam", {})
        model = CLAM(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_clusters=clam_config.get("num_clusters", 10),
            dropout=dropout,
            multi_branch=clam_config.get("multi_branch", True),
            instance_loss_weight=clam_config.get("instance_loss_weight", 0.3),
        )
        logger.info(
            f"CLAM created: num_clusters={clam_config.get('num_clusters', 10)}, "
            f"multi_branch={clam_config.get('multi_branch', True)}"
        )

    elif model_type == "transmil":
        # TransMIL configuration
        transmil_config = config.get("transmil", {})
        model = TransMIL(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_layers=transmil_config.get("num_layers", 2),
            num_heads=transmil_config.get("num_heads", 8),
            dropout=dropout,
            use_pos_encoding=transmil_config.get("use_pos_encoding", True),
        )
        logger.info(
            f"TransMIL created: num_layers={transmil_config.get('num_layers', 2)}, "
            f"num_heads={transmil_config.get('num_heads', 8)}"
        )

    elif model_type in ["mean", "max"]:
        # Baseline pooling models
        pass

        # Create a simple pooling model
        class SimplePoolingModel(nn.Module):
            def __init__(self, feature_dim, hidden_dim, num_classes, pooling="mean"):
                super().__init__()
                self.pooling = pooling
                self.classifier = nn.Sequential(
                    nn.Linear(feature_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, num_classes),
                )

            def forward(self, features, num_patches=None, return_attention=False):
                # Apply pooling
                if self.pooling == "mean":
                    if num_patches is not None:
                        # Masked mean pooling
                        mask = torch.arange(features.size(1), device=features.device).unsqueeze(
                            0
                        ) < num_patches.unsqueeze(1)
                        mask = mask.unsqueeze(-1).float()
                        pooled = (features * mask).sum(dim=1) / (mask.sum(dim=1) + 1e-8)
                    else:
                        pooled = features.mean(dim=1)
                elif self.pooling == "max":
                    if num_patches is not None:
                        # Masked max pooling
                        mask = torch.arange(features.size(1), device=features.device).unsqueeze(
                            0
                        ) < num_patches.unsqueeze(1)
                        features_masked = features.clone()
                        features_masked[~mask] = float("-inf")
                        pooled = features_masked.max(dim=1)[0]
                    else:
                        pooled = features.max(dim=1)[0]

                logits = self.classifier(pooled)

                if return_attention:
                    # Return uniform attention for compatibility
                    attention = torch.ones(
                        features.size(0), features.size(1), device=features.device
                    )
                    if num_patches is not None:
                        mask = torch.arange(features.size(1), device=features.device).unsqueeze(
                            0
                        ) < num_patches.unsqueeze(1)
                        attention = attention.masked_fill(~mask, 0.0)
                    attention = attention / (attention.sum(dim=1, keepdim=True) + 1e-8)
                    return logits, attention
                else:
                    return logits

        model = SimplePoolingModel(feature_dim, hidden_dim, num_classes, pooling=model_type)
        logger.info(f"Baseline {model_type} pooling model created")

    else:
        raise ValueError(
            f"Invalid model_type: {model_type}. Must be one of: "
            f"attention_mil, clam, transmil, mean, max"
        )

    return model
