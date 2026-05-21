"""
Adaptive Token Pruning for TransMIL.

Reduces computational cost by pruning less important patches before transformer.
Importance scored via learned network, attention weights, or prediction confidence.

Key components:
- Importance scorer (learned MLP or attention-based)
- Top-k selection (keep most important patches)
- Integration with TransMIL branch

Architecture:
    Input: Patch features [B, N, D]
    ├─ Score importance [B, N, 1]
    ├─ Select top-k patches
    └─ Output: Pruned features [B, k, D] + mask [B, N]

Reference:
- TransnnMIL v2.0: Hierarchical + Topology (2027)
- DynamicViT: Efficient Vision Transformers with Dynamic Token Sparsification (NeurIPS 2021)
"""

from typing import Optional, Tuple

import torch
import torch.nn as nn


class ImportanceScorer(nn.Module):
    """
    Score patch importance for pruning.

    Args:
        feature_dim: Input feature dimension
        hidden_dim: Hidden dimension for MLP scorer
        scoring_method: Scoring method ('learned', 'attention', 'confidence')
        dropout: Dropout rate (default: 0.1)

    Example:
        >>> scorer = ImportanceScorer(feature_dim=1024, hidden_dim=256)
        >>> features = torch.randn(4, 100, 1024)
        >>> scores = scorer(features)  # [4, 100, 1]
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        scoring_method: str = "learned",
        dropout: float = 0.1,
    ):
        super().__init__()

        if scoring_method not in ["learned", "attention", "confidence"]:
            raise ValueError(
                f"scoring_method must be 'learned', 'attention', or 'confidence', got {scoring_method}"
            )

        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.scoring_method = scoring_method

        # Learned scorer (MLP)
        if scoring_method == "learned":
            self.scorer = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )
        # Attention-based scorer
        elif scoring_method == "attention":
            self.scorer = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.Tanh(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
        # Confidence-based scorer (requires class predictions)
        elif scoring_method == "confidence":
            self.scorer = nn.Sequential(
                nn.Linear(feature_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid(),  # Confidence in [0, 1]
            )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Score patch importance.

        Args:
            features: Patch features [batch_size, num_patches, feature_dim]

        Returns:
            scores: Importance scores [batch_size, num_patches, 1]
        """
        scores = self.scorer(features)  # [B, N, 1]
        return scores


class AdaptivePruning(nn.Module):
    """
    Adaptive token pruning module.

    Prunes less important patches to reduce computational cost.

    Args:
        feature_dim: Input feature dimension
        keep_ratio: Ratio of patches to keep (default: 0.5)
        scoring_method: Scoring method ('learned', 'attention', 'confidence')
        min_patches: Minimum number of patches to keep (default: 10)
        dropout: Dropout rate (default: 0.1)

    Example:
        >>> pruning = AdaptivePruning(feature_dim=1024, keep_ratio=0.5)
        >>> features = torch.randn(4, 100, 1024)
        >>>
        >>> pruned_features, mask, indices = pruning(features)
        >>> # pruned_features: [4, 50, 1024]
        >>> # mask: [4, 100] (True for kept patches)
        >>> # indices: [4, 50] (indices of kept patches)
    """

    def __init__(
        self,
        feature_dim: int,
        keep_ratio: float = 0.5,
        scoring_method: str = "learned",
        min_patches: int = 10,
        dropout: float = 0.1,
    ):
        super().__init__()

        if not 0 < keep_ratio <= 1:
            raise ValueError(f"keep_ratio must be in (0, 1], got {keep_ratio}")

        self.feature_dim = feature_dim
        self.keep_ratio = keep_ratio
        self.min_patches = min_patches

        # Importance scorer
        self.scorer = ImportanceScorer(
            feature_dim=feature_dim,
            hidden_dim=256,
            scoring_method=scoring_method,
            dropout=dropout,
        )

    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Prune patches based on importance scores.

        Args:
            features: Patch features [batch_size, num_patches, feature_dim]
            mask: Optional mask [batch_size, num_patches] (True for valid patches)

        Returns:
            pruned_features: Pruned features [batch_size, k, feature_dim]
            pruned_mask: Mask for pruned patches [batch_size, num_patches]
            indices: Indices of kept patches [batch_size, k]
        """
        batch_size, num_patches, feature_dim = features.shape

        # Score importance
        scores = self.scorer(features).squeeze(-1)  # [B, N]

        # Apply mask (set invalid patches to -inf)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))

        # Compute number of patches to keep
        if mask is not None:
            # Keep ratio of valid patches
            num_valid = mask.sum(dim=1)  # [B]
            k = (num_valid * self.keep_ratio).long()
            k = torch.clamp(k, min=self.min_patches, max=num_patches)
        else:
            k = max(int(num_patches * self.keep_ratio), self.min_patches)
            k = torch.full((batch_size,), k, device=features.device)

        # Select top-k patches
        # Handle variable k per batch
        max_k = k.max().item()
        _, indices = torch.topk(scores, k=max_k, dim=1, largest=True, sorted=True)

        # Create mask for kept patches
        pruned_mask = torch.zeros(batch_size, num_patches, dtype=torch.bool, device=features.device)

        # Gather pruned features
        pruned_features_list = []
        for i in range(batch_size):
            k_i = k[i].item()
            indices_i = indices[i, :k_i]  # [k_i]

            # Mark kept patches
            pruned_mask[i, indices_i] = True

            # Gather features
            features_i = features[i, indices_i]  # [k_i, D]
            pruned_features_list.append(features_i)

        # Pad to max_k
        pruned_features = torch.zeros(
            batch_size, max_k, feature_dim, device=features.device, dtype=features.dtype
        )
        for i, feats in enumerate(pruned_features_list):
            pruned_features[i, : len(feats)] = feats

        # Trim indices to max_k
        indices = indices[:, :max_k]

        return pruned_features, pruned_mask, indices

    def get_speedup(self, num_patches: int) -> float:
        """
        Estimate speedup from pruning.

        Args:
            num_patches: Original number of patches

        Returns:
            speedup: Estimated speedup factor
        """
        k = max(int(num_patches * self.keep_ratio), self.min_patches)
        # Transformer complexity: O(N^2)
        speedup = (num_patches / k) ** 2
        return speedup


class PrunedTransMIL(nn.Module):
    """
    TransMIL with adaptive pruning.

    Integrates pruning module before transformer layers.

    Args:
        feature_dim: Input feature dimension
        num_classes: Number of output classes
        keep_ratio: Ratio of patches to keep (default: 0.5)
        scoring_method: Scoring method ('learned', 'attention', 'confidence')
        num_layers: Number of transformer layers (default: 2)
        num_heads: Number of attention heads (default: 8)
        dropout: Dropout rate (default: 0.1)

    Example:
        >>> model = PrunedTransMIL(
        ...     feature_dim=1024,
        ...     num_classes=2,
        ...     keep_ratio=0.5,
        ... )
        >>>
        >>> features = torch.randn(4, 100, 1024)
        >>> logits = model(features)  # [4, 2]
    """

    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        keep_ratio: float = 0.5,
        scoring_method: str = "learned",
        num_layers: int = 2,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.keep_ratio = keep_ratio

        # Adaptive pruning
        self.pruning = AdaptivePruning(
            feature_dim=feature_dim,
            keep_ratio=keep_ratio,
            scoring_method=scoring_method,
            dropout=dropout,
        )

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=feature_dim,
            nhead=num_heads,
            dim_feedforward=feature_dim * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feature_dim // 2, num_classes),
        )

    def forward(
        self,
        features: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        return_pruning_info: bool = False,
    ) -> torch.Tensor:
        """
        Forward pass with pruning.

        Args:
            features: Patch features [batch_size, num_patches, feature_dim]
            mask: Optional mask [batch_size, num_patches]
            return_pruning_info: Return pruning info (mask, indices)

        Returns:
            logits: Class logits [batch_size, num_classes]
            (optional) pruning_info: Dict with mask and indices
        """
        # Prune patches
        pruned_features, pruned_mask, indices = self.pruning(features, mask)

        # Transformer
        # Create attention mask (True = ignore)
        attn_mask = ~pruned_mask[:, : pruned_features.shape[1]]  # [B, k]
        transformer_out = self.transformer(pruned_features, src_key_padding_mask=attn_mask)

        # Global pooling (mean over valid patches)
        pooled = (transformer_out * pruned_mask[:, : pruned_features.shape[1]].unsqueeze(-1)).sum(
            dim=1
        ) / pruned_mask[:, : pruned_features.shape[1]].sum(dim=1, keepdim=True)

        # Classification
        logits = self.classifier(pooled)

        if return_pruning_info:
            pruning_info = {
                "mask": pruned_mask,
                "indices": indices,
                "keep_ratio": pruned_mask.float().mean().item(),
            }
            return logits, pruning_info

        return logits
