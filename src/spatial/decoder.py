"""
Cross-attention transformer decoder: H&E patch features → spatial gene expression.

Architecture:
    H&E patches → positional encoding → N cross-attention layers → gene expression head

Each H&E patch attends to a learned gene query bank, producing
per-patch gene expression predictions at spatial resolution.

Primary metric: mean Pearson correlation across genes (Stahl et al. 2016 benchmark).
"""

import logging
import math
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class SpatialPositionalEncoding(nn.Module):
    """2D spatial positional encoding from (row, col) spot coordinates."""

    def __init__(self, d_model: int, max_grid: int = 256):
        super().__init__()
        self.d_model = d_model
        # Learnable row + col embeddings
        self.row_embed = nn.Embedding(max_grid, d_model // 2)
        self.col_embed = nn.Embedding(max_grid, d_model // 2)
        self.proj = nn.Linear(d_model, d_model)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: Spot coordinates [batch, n_spots, 2] (row, col) integers

        Returns:
            Positional embeddings [batch, n_spots, d_model]
        """
        rows = coords[..., 0].clamp(0, self.row_embed.num_embeddings - 1)
        cols = coords[..., 1].clamp(0, self.col_embed.num_embeddings - 1)
        pe = torch.cat([self.row_embed(rows), self.col_embed(cols)], dim=-1)
        return self.proj(pe)


class CrossAttentionLayer(nn.Module):
    """
    Cross-attention: H&E patch features attend to gene queries.

    Queries: learned gene query bank [n_genes, d_model]
    Keys/Values: H&E patch features (projected) [batch, n_patches, d_model]
    """

    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        query: torch.Tensor,
        key_value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            query: Gene queries [batch, n_genes, d_model]
            key_value: H&E patch features [batch, n_patches, d_model]
            key_padding_mask: Patch mask [batch, n_patches] (True = padding)
        """
        attended, _ = self.attn(query, key_value, key_value, key_padding_mask=key_padding_mask)
        query = self.norm1(query + self.dropout(attended))
        query = self.norm2(query + self.dropout(self.ff(query)))
        return query


class SpatialTranscriptomicsDecoder(nn.Module):
    """
    Cross-attention transformer mapping H&E features → gene expression.

    Given H&E patch embeddings from a pretrained encoder (ResNet, UNI, etc.),
    predicts the spatial transcriptome at each spot location.

    Architecture:
        patch_features → patch_proj (linear) → positional_enc + patch_proj
        gene_queries (learned) →
        N × CrossAttentionLayer(query=genes, kv=patches) →
        gene_head (linear → softplus) → expression [n_spots, n_genes]

    Training objective: 1 - mean_pearson(pred, true) + MSE + sparsity
    """

    def __init__(
        self,
        patch_feature_dim: int = 1024,
        n_genes: int = 3000,
        d_model: int = 256,
        n_heads: int = 8,
        n_layers: int = 4,
        dropout: float = 0.1,
        max_grid: int = 256,
    ):
        super().__init__()
        self.n_genes = n_genes
        self.d_model = d_model

        # Project patch features to d_model
        self.patch_proj = nn.Sequential(
            nn.Linear(patch_feature_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # Spatial positional encoding for patches
        self.spatial_pe = SpatialPositionalEncoding(d_model, max_grid)

        # Learned gene query bank: one query vector per gene
        self.gene_queries = nn.Parameter(torch.randn(1, n_genes, d_model) * 0.02)

        # Cross-attention layers
        self.layers = nn.ModuleList(
            [CrossAttentionLayer(d_model, n_heads, dropout) for _ in range(n_layers)]
        )

        # Output head: project to gene expression space
        self.gene_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
        )

        # Softplus ensures non-negative expression predictions
        self.activation = nn.Softplus()

        logger.info(
            f"SpatialTranscriptomicsDecoder: patch_dim={patch_feature_dim}, "
            f"n_genes={n_genes}, d_model={d_model}, n_layers={n_layers}"
        )

    def forward(
        self,
        patch_features: torch.Tensor,
        patch_coords: Optional[torch.Tensor] = None,
        patch_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            patch_features: H&E patch embeddings [batch, n_patches, patch_feature_dim]
            patch_coords: Patch grid coordinates [batch, n_patches, 2] (optional)
            patch_mask: Boolean mask [batch, n_patches] True=padding (optional)

        Returns:
            Predicted gene expression [batch, n_patches, n_genes]
        """
        batch_size, n_patches, _ = patch_features.shape

        # Project patches to d_model
        patch_emb = self.patch_proj(patch_features)  # [B, n_patches, d_model]

        # Add spatial positional encoding
        if patch_coords is not None:
            patch_emb = patch_emb + self.spatial_pe(patch_coords)

        # Expand gene queries to batch size
        queries = self.gene_queries.expand(batch_size, -1, -1)  # [B, n_genes, d_model]

        # Cross-attention: each patch gets its own per-gene query
        # We process per-patch by treating patch dim as batch dim
        # Reshape: [B*n_patches, 1, d_model] as query, [B, n_patches, d_model] as kv
        # More efficient: global cross-attention, then reshape
        for layer in self.layers:
            queries = layer(queries, patch_emb, key_padding_mask=patch_mask)
        # queries: [B, n_genes, d_model]

        # Gene expression head
        expr = self.gene_head(queries).squeeze(-1)  # [B, n_genes]
        expr = self.activation(expr)

        # Expand to per-patch predictions via broadcast + patch-aware weighting
        # For per-spot prediction: we need to make patch-level predictions
        # Use attention-weighted sum: predict per patch by querying with patch feats
        # Full per-spot: use patch features as queries and gene queries as memory
        # This gives [B, n_patches, n_genes]
        patch_queries = patch_emb.unsqueeze(2).expand(-1, -1, self.n_genes, -1)
        # Simpler: use the global gene prediction + patch-specific residual
        # Residual from patch features projected to gene space
        patch_residual = self.patch_proj(patch_features)  # [B, n_patches, d_model]
        # For per-patch gene predictions, use per-patch cross-attention output
        # Stack patches as individual queries
        all_queries = torch.cat(
            [
                queries.unsqueeze(1).expand(-1, n_patches, -1, -1),  # [B,N,G,D]
                patch_emb.unsqueeze(2).expand(-1, -1, self.n_genes, -1),  # [B,N,G,D]
            ],
            dim=-1,
        )
        # Project combined to expression
        per_patch_expr = self.activation(
            nn.functional.linear(
                all_queries,
                torch.cat([self.gene_head[1].weight, self.gene_head[1].weight], dim=1),
                self.gene_head[1].bias,
            ).squeeze(-1)
        )  # [B, n_patches, n_genes]

        return per_patch_expr

    def predict_global(
        self,
        patch_features: torch.Tensor,
        patch_coords: Optional[torch.Tensor] = None,
        patch_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Predict slide-level average expression (mean over patches).

        Returns: [batch, n_genes]
        """
        per_patch = self.forward(patch_features, patch_coords, patch_mask)
        if patch_mask is not None:
            valid = ~patch_mask
            per_patch = per_patch * valid.unsqueeze(-1).float()
            return per_patch.sum(1) / valid.float().sum(1, keepdim=True).clamp(min=1)
        return per_patch.mean(1)


class SpatialDecoderLoss(nn.Module):
    """
    Combined loss for spatial transcriptomics prediction.

    L = (1 - mean_pearson) + lambda_mse * MSE + lambda_sparse * L1

    The primary metric in the field is mean Pearson r across genes,
    so we directly optimize 1 - mean_pearson as the main objective.
    """

    def __init__(
        self,
        lambda_mse: float = 0.1,
        lambda_sparse: float = 0.01,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.lambda_mse = lambda_mse
        self.lambda_sparse = lambda_sparse
        self.eps = eps

    def _pearson_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        1 - mean Pearson r across genes.

        Args:
            pred: [batch, n_genes] or [batch, n_spots, n_genes]
            target: same shape as pred
        """
        # Flatten spatial dims if present
        if pred.dim() == 3:
            B, N, G = pred.shape
            pred = pred.reshape(B * N, G)
            target = target.reshape(B * N, G)

        # Per-sample Pearson r across genes
        pred_m = pred - pred.mean(dim=1, keepdim=True)
        tgt_m = target - target.mean(dim=1, keepdim=True)
        num = (pred_m * tgt_m).sum(dim=1)
        denom = pred_m.norm(dim=1) * tgt_m.norm(dim=1) + self.eps
        r = num / denom
        return 1.0 - r.mean()

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        pearson_loss = self._pearson_loss(pred, target)
        mse_loss = F.mse_loss(pred, target)
        sparse_loss = pred.abs().mean()

        total = pearson_loss + self.lambda_mse * mse_loss + self.lambda_sparse * sparse_loss
        return total, {
            "pearson_loss": pearson_loss.item(),
            "mse_loss": mse_loss.item(),
            "sparse_loss": sparse_loss.item(),
            "total_loss": total.item(),
        }
