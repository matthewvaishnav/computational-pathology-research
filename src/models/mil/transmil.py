"""
Transformer-based Multiple Instance Learning (TransMIL) model.

This module provides the TransMIL model, which uses transformer encoder layers
with multi-head self-attention to model relationships between patches in a slide.
"""

from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

from src.models.components.fusion_strategies import EarlyFusion, FusionStrategy, LateFusion


class TransMIL(nn.Module):
    """
    Transformer-based Multiple Instance Learning (TransMIL).

    TransMIL uses transformer encoder layers with multi-head self-attention to model
    relationships between patches in a slide. Unlike traditional attention-based MIL
    which computes attention weights independently for each patch, TransMIL allows
    patches to attend to each other, capturing spatial and contextual relationships.

    The model uses a learnable CLS token (similar to BERT) that aggregates information
    from all patches through self-attention. The CLS token representation is then
    used for slide-level classification.

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
        super().__init__()

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim ({hidden_dim}) must be divisible by num_heads ({num_heads})"
            )

        if fusion_strategy not in ["early", "late"]:
            raise ValueError(f"fusion_strategy must be 'early' or 'late', got {fusion_strategy}")

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.use_pos_encoding = use_pos_encoding
        self.multi_scale = multi_scale
        self.num_scales = num_scales
        self.fusion_strategy = fusion_strategy

        # Setup fusion strategy for multi-scale
        self.fusion: Optional[FusionStrategy] = None
        if multi_scale and num_scales > 1:
            if fusion_strategy == "early":
                self.fusion = EarlyFusion(feature_dim, hidden_dim, num_scales, dropout)
            elif fusion_strategy == "late":
                self.fusion = LateFusion(feature_dim, hidden_dim, num_scales, dropout)

        # Feature projection layer(s)
        if multi_scale and num_scales > 1 and fusion_strategy == "late":
            # Scale-specific feature projection layers for late fusion
            self.feature_proj = nn.ModuleList(
                [
                    nn.Sequential(
                        nn.Linear(feature_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
                    )
                    for _ in range(num_scales)
                ]
            )
        else:
            # Single feature projection layer (for single-scale or early fusion)
            self.feature_proj = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim), nn.ReLU(), nn.Dropout(dropout)
            )

        # Learnable positional encoding (max 10000 patches)
        if use_pos_encoding:
            if multi_scale and num_scales > 1 and fusion_strategy == "late":
                # Scale-specific positional encodings for late fusion
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
            # Single transformer (for single-scale or early fusion)
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
        self, multi_scale_features: list, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Early fusion: concatenate features from all scales before transformer."""
        # Get batch size and max patches from first non-None scale
        first_scale = next((f for f in multi_scale_features if f is not None), None)
        batch_size, max_patches, _ = first_scale.shape

        # Apply fusion strategy to combine scales (already projects to hidden_dim)
        h = self.fusion(multi_scale_features, mask)

        # Add positional encoding
        if self.use_pos_encoding:
            h = h + self.pos_encoding[:, :max_patches, :]

        # Prepend CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        h = torch.cat([cls_tokens, h], dim=1)

        # Create attention mask for transformer
        if mask is not None:
            transformer_mask = ~mask
            cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=first_scale.device)
            transformer_mask = torch.cat([cls_mask, transformer_mask], dim=1)
        else:
            transformer_mask = None

        # Apply transformer
        h = self.transformer(h, src_key_padding_mask=transformer_mask)

        # Extract CLS token representation and apply layer normalization
        cls_repr = self.norm(h[:, 0, :])

        # Return uniform attention weights for API compatibility
        attention_weights = self.compute_attention(first_scale, mask)

        return cls_repr, attention_weights

    def _late_fusion_transmil(
        self, multi_scale_features: list, mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Late fusion: separate transformers per scale, then concatenate CLS tokens."""
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
            h = (
                self.feature_proj[scale_idx](scale_features)
                if isinstance(self.feature_proj, nn.ModuleList)
                else self.feature_proj(scale_features)
            )

            # Add scale-specific positional encoding
            if self.use_pos_encoding:
                if isinstance(self.pos_encoding, nn.ParameterList):
                    h = h + self.pos_encoding[scale_idx][:, :max_patches, :]
                else:
                    h = h + self.pos_encoding[:, :max_patches, :]

            # Prepend CLS token
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            h = torch.cat([cls_tokens, h], dim=1)

            # Create attention mask for transformer
            if mask is not None:
                transformer_mask = ~mask
                cls_mask = torch.zeros(
                    batch_size, 1, dtype=torch.bool, device=scale_features.device
                )
                transformer_mask = torch.cat([cls_mask, transformer_mask], dim=1)
            else:
                transformer_mask = None

            # Apply scale-specific transformer
            h = (
                self.transformer[scale_idx](h, src_key_padding_mask=transformer_mask)
                if isinstance(self.transformer, nn.ModuleList)
                else self.transformer(h, src_key_padding_mask=transformer_mask)
            )

            # Extract CLS token representation
            cls_repr = h[:, 0, :]

            # Apply scale-specific layer normalization
            cls_repr = (
                self.norm[scale_idx](cls_repr)
                if isinstance(self.norm, nn.ModuleList)
                else self.norm(cls_repr)
            )

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

        TransMIL uses internal transformer attention which is not directly exposed.
        This method returns uniform weights for API compatibility.
        """
        batch_size, num_patches, _ = features.shape
        attention_weights = torch.ones(batch_size, num_patches, device=features.device)

        if mask is not None:
            attention_weights = attention_weights.masked_fill(~mask, 0.0)

        return attention_weights / (attention_weights.sum(dim=1, keepdim=True) + 1e-8)

    def forward(
        self,
        features: Union[torch.Tensor, list],
        num_patches: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """Forward pass through TransMIL model."""
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

            return (logits, attention_weights) if return_attention else logits
        else:
            # Single-scale input
            batch_size, max_patches, _ = features.shape

            # Project features
            h = (
                self.feature_proj[0](features)
                if isinstance(self.feature_proj, nn.ModuleList)
                else self.feature_proj(features)
            )

            # Add positional encoding if enabled
            if self.use_pos_encoding:
                if isinstance(self.pos_encoding, nn.ParameterList):
                    h = h + self.pos_encoding[0][:, :max_patches, :]
                else:
                    h = h + self.pos_encoding[:, :max_patches, :]

            # Prepend CLS token
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            h = torch.cat([cls_tokens, h], dim=1)

            # Create attention mask for transformer
            if num_patches is not None:
                transformer_mask = torch.arange(max_patches, device=features.device).unsqueeze(
                    0
                ) >= num_patches.unsqueeze(1)
                cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=features.device)
                transformer_mask = torch.cat([cls_mask, transformer_mask], dim=1)

                # Create mask for compute_attention (True for valid patches)
                patch_mask = torch.arange(max_patches, device=features.device).unsqueeze(
                    0
                ) < num_patches.unsqueeze(1)
            else:
                transformer_mask = None
                patch_mask = None

            # Apply transformer encoder
            h = (
                self.transformer[0](h, src_key_padding_mask=transformer_mask)
                if isinstance(self.transformer, nn.ModuleList)
                else self.transformer(h, src_key_padding_mask=transformer_mask)
            )

            # Extract CLS token representation
            cls_repr = h[:, 0, :]

            # Apply layer normalization
            cls_repr = (
                self.norm[0](cls_repr)
                if isinstance(self.norm, nn.ModuleList)
                else self.norm(cls_repr)
            )

            # Classify
            logits = self.classifier(cls_repr)

            if return_attention:
                attention_weights = self.compute_attention(features, patch_mask)
                return logits, attention_weights
            else:
                return logits

    def get_features(
        self,
        features: torch.Tensor,
        num_patches: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Extract feature representation before classification.

        Args:
            features: Patch features [batch_size, num_patches, feature_dim]
            num_patches: Actual patch counts [batch_size] for masking padded patches

        Returns:
            cls_repr: CLS token representation [batch_size, hidden_dim]
        """
        if self.multi_scale:
            # Multi-scale input
            batch_size = features[0].shape[0] if features[0] is not None else features[1].shape[0]
            max_patches = max(f.shape[1] for f in features if f is not None)

            # Create mask from num_patches
            mask = None
            if num_patches is not None:
                first_scale = next((f for f in features if f is not None), None)
                mask = torch.arange(max_patches, device=first_scale.device).unsqueeze(
                    0
                ) < num_patches.unsqueeze(1)

            # Apply fusion strategy
            if self.fusion_strategy == "early":
                cls_repr, _ = self._early_fusion_transmil(features, mask)
            elif self.fusion_strategy == "late":
                cls_repr, _ = self._late_fusion_transmil(features, mask)
            else:
                raise ValueError(f"Unknown fusion_strategy: {self.fusion_strategy}")

            return cls_repr
        else:
            # Single-scale input
            batch_size, max_patches, _ = features.shape

            # Project features
            h = (
                self.feature_proj[0](features)
                if isinstance(self.feature_proj, nn.ModuleList)
                else self.feature_proj(features)
            )

            # Add CLS token
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            h = torch.cat([cls_tokens, h], dim=1)

            # Create attention mask for transformer
            if num_patches is not None:
                transformer_mask = torch.arange(max_patches, device=features.device).unsqueeze(
                    0
                ) >= num_patches.unsqueeze(1)
                cls_mask = torch.zeros(batch_size, 1, dtype=torch.bool, device=features.device)
                transformer_mask = torch.cat([cls_mask, transformer_mask], dim=1)
            else:
                transformer_mask = None

            # Apply transformer encoder
            h = (
                self.transformer[0](h, src_key_padding_mask=transformer_mask)
                if isinstance(self.transformer, nn.ModuleList)
                else self.transformer(h, src_key_padding_mask=transformer_mask)
            )

            # Extract CLS token representation
            cls_repr = h[:, 0, :]

            # Apply layer normalization
            cls_repr = (
                self.norm[0](cls_repr)
                if isinstance(self.norm, nn.ModuleList)
                else self.norm(cls_repr)
            )

            return cls_repr
