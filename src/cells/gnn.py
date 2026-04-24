"""
Graph Attention Network for cell graph classification.

GATConv layers aggregate neighbourhood information weighted by learned attention.
CellGraphNet → patch-level embedding. TMEClassifier → slide-level TME prediction.
"""

import logging
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)

try:
    from torch_geometric.nn import GATConv, global_mean_pool, global_max_pool
    _HAS_PYG = True
except ImportError:
    _HAS_PYG = False
    logger.warning("torch-geometric not installed — using fallback mean-aggregation GNN")


# --- Fallback when torch-geometric unavailable ---

class _FallbackGATConv(nn.Module):
    """Simple mean-aggregation message passing (no attention) as PYG fallback."""

    def __init__(self, in_channels: int, out_channels: int, heads: int = 1, **kwargs):
        super().__init__()
        self.lin = nn.Linear(in_channels, out_channels * heads)
        self.out_channels = out_channels
        self.heads = heads

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        src, dst = edge_index[0], edge_index[1]
        # Mean-aggregate neighbours
        agg = torch.zeros(N, x.size(1), device=x.device)
        count = torch.zeros(N, 1, device=x.device)
        agg.scatter_add_(0, dst.unsqueeze(1).expand(-1, x.size(1)), x[src])
        count.scatter_add_(0, dst.unsqueeze(1), torch.ones(dst.size(0), 1, device=x.device))
        agg = agg / (count + 1e-8)
        return self.lin(x + agg).view(N, self.heads * self.out_channels)


def _global_mean_pool_fallback(x: torch.Tensor, batch: Optional[torch.Tensor]) -> torch.Tensor:
    if batch is None:
        return x.mean(0, keepdim=True)
    B = int(batch.max().item()) + 1
    out = torch.zeros(B, x.size(1), device=x.device)
    count = torch.zeros(B, 1, device=x.device)
    out.scatter_add_(0, batch.unsqueeze(1).expand(-1, x.size(1)), x)
    count.scatter_add_(0, batch.unsqueeze(1), torch.ones(batch.size(0), 1, device=x.device))
    return out / (count + 1e-8)


# --- Main GNN ---

class CellGraphNet(nn.Module):
    """
    Graph Attention Network: CellGraph → fixed-size embedding.

    Architecture: 3× GATConv → global mean+max pool → MLP → embed_dim
    """

    def __init__(
        self,
        node_dim: int,
        edge_dim: int = 4,
        hidden_dim: int = 128,
        embed_dim: int = 256,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.embed_dim = embed_dim

        if _HAS_PYG:
            self.conv1 = GATConv(node_dim, hidden_dim, heads=heads, dropout=dropout, concat=True)
            self.conv2 = GATConv(hidden_dim * heads, hidden_dim, heads=heads, dropout=dropout, concat=True)
            self.conv3 = GATConv(hidden_dim * heads, embed_dim, heads=1, dropout=dropout, concat=False)
        else:
            self.conv1 = _FallbackGATConv(node_dim, hidden_dim, heads=heads)
            self.conv2 = _FallbackGATConv(hidden_dim * heads, hidden_dim, heads=heads)
            self.conv3 = _FallbackGATConv(hidden_dim * heads, embed_dim, heads=1)

        self.norm1 = nn.LayerNorm(hidden_dim * heads)
        self.norm2 = nn.LayerNorm(hidden_dim * heads)

        # Global pool: concat mean + max → 2 * embed_dim → embed_dim
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
        )
        self.dropout = dropout

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.norm1(self.conv1(x, edge_index)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.norm2(self.conv2(x, edge_index)))
        x = self.conv3(x, edge_index)

        if _HAS_PYG:
            g_mean = global_mean_pool(x, batch)
            g_max = global_max_pool(x, batch)
        else:
            g_mean = _global_mean_pool_fallback(x, batch)
            g_max = _global_mean_pool_fallback(x, batch)  # simplified

        return self.mlp(torch.cat([g_mean, g_max], dim=1))


class TMEClassifier(nn.Module):
    """
    Full TME classification model: CellGraph → immune phenotype + TIL density.

    Outputs:
      - phenotype_logits: (B, num_phenotypes) — inflamed / excluded / desert
      - til_density: (B, 1) — normalised TIL fraction in [0, 1]
    """

    def __init__(
        self,
        node_dim: int,
        num_phenotypes: int = 3,
        hidden_dim: int = 128,
        embed_dim: int = 256,
        heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = CellGraphNet(
            node_dim=node_dim,
            hidden_dim=hidden_dim,
            embed_dim=embed_dim,
            heads=heads,
            dropout=dropout,
        )
        self.phenotype_head = nn.Linear(embed_dim, num_phenotypes)
        self.til_head = nn.Sequential(nn.Linear(embed_dim, 64), nn.ReLU(), nn.Linear(64, 1), nn.Sigmoid())

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> dict:
        emb = self.encoder(x, edge_index, batch)
        return {
            "embedding": emb,
            "phenotype_logits": self.phenotype_head(emb),
            "til_density": self.til_head(emb).squeeze(-1),
        }
