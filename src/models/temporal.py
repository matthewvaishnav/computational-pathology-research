"""
Cross-slide temporal reasoning for disease progression analysis.

This module implements temporal attention mechanisms that analyze relationships
across multiple slides from the same patient over time, incorporating temporal
distances and extracting progression features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import math


class TemporalAttention(nn.Module):
    """
    Temporal attention mechanism with positional encoding for temporal distances.
    
    Implements self-attention over a sequence of slide embeddings with
    learnable positional encodings that capture temporal relationships.
    
    Args:
        embed_dim: Dimension of slide embeddings (default: 256)
        num_heads: Number of attention heads (default: 8)
        num_layers: Number of transformer layers (default: 2)
        max_temporal_distance: Maximum temporal distance in days (default: 365*5)
        dropout: Dropout rate (default: 0.1)
    
    Example:
        >>> temporal_attn = TemporalAttention(embed_dim=256, num_heads=8)
        >>> slide_embeddings = torch.randn(16, 5, 256)  # [batch, num_slides, embed_dim]
        >>> timestamps = torch.randn(16, 5)  # [batch, num_slides] in days
        >>> output = temporal_attn(slide_embeddings, timestamps)
        >>> output.shape
        torch.Size([16, 5, 256])
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        max_temporal_distance: float = 365 * 5,  # 5 years in days
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.max_temporal_distance = max_temporal_distance
        
        # Temporal positional encoding
        # We'll use a learnable embedding for discretized temporal distances
        self.num_temporal_bins = 100
        self.temporal_embedding = nn.Embedding(self.num_temporal_bins, embed_dim)
        
        # Transformer encoder for temporal reasoning
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)
        
    def _compute_temporal_encoding(
        self,
        timestamps: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute temporal positional encodings from timestamps.
        
        Args:
            timestamps: Timestamps [batch_size, num_slides] (in days or arbitrary units)
            
        Returns:
            Temporal encodings [batch_size, num_slides, embed_dim]
        """
        batch_size, num_slides = timestamps.shape
        
        # Normalize timestamps to relative positions (0 to max_temporal_distance)
        # First, make timestamps relative to the first slide
        relative_times = timestamps - timestamps[:, :1]  # [B, num_slides]
        
        # Clip to max temporal distance
        relative_times = torch.clamp(relative_times, 0, self.max_temporal_distance)
        
        # Discretize into bins
        temporal_bins = (relative_times / self.max_temporal_distance * (self.num_temporal_bins - 1))
        temporal_bins = temporal_bins.long()  # [B, num_slides]
        
        # Get temporal embeddings
        temporal_enc = self.temporal_embedding(temporal_bins)  # [B, num_slides, embed_dim]
        
        return temporal_enc
    
    def forward(
        self,
        slide_embeddings: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply temporal attention to slide sequence.
        
        Args:
            slide_embeddings: Slide embeddings [batch_size, num_slides, embed_dim]
            timestamps: Optional timestamps [batch_size, num_slides] for temporal encoding
            mask: Optional mask [batch_size, num_slides] where True indicates valid slides
            
        Returns:
            Temporally attended embeddings [batch_size, num_slides, embed_dim]
        """
        batch_size, num_slides, embed_dim = slide_embeddings.shape
        
        # Add temporal positional encoding if timestamps provided
        if timestamps is not None:
            temporal_enc = self._compute_temporal_encoding(timestamps)
            x = slide_embeddings + temporal_enc
        else:
            # Use standard positional encoding based on sequence position
            positions = torch.arange(num_slides, device=slide_embeddings.device)
            positions = positions.unsqueeze(0).expand(batch_size, -1)  # [B, num_slides]
            
            # Create sinusoidal positional encoding
            pos_enc = self._sinusoidal_encoding(positions, embed_dim)
            x = slide_embeddings + pos_enc
        
        # Apply transformer encoder
        if mask is not None:
            # Invert mask for transformer (True = ignore)
            attn_mask = ~mask
        else:
            attn_mask = None
        
        x = self.transformer(x, src_key_padding_mask=attn_mask)
        x = self.norm(x)
        
        return x
    
    def _sinusoidal_encoding(
        self,
        positions: torch.Tensor,
        embed_dim: int
    ) -> torch.Tensor:
        """
        Create sinusoidal positional encoding.
        
        Args:
            positions: Position indices [batch_size, num_positions]
            embed_dim: Embedding dimension
            
        Returns:
            Positional encodings [batch_size, num_positions, embed_dim]
        """
        batch_size, num_positions = positions.shape
        device = positions.device
        
        # Compute frequencies
        freqs = 1.0 / (10000 ** (2 * torch.arange(embed_dim // 2, device=device).float() / embed_dim))
        
        # Compute angles
        angles = positions.unsqueeze(-1).float() * freqs.unsqueeze(0).unsqueeze(0)
        
        # Apply sin and cos
        sin_enc = torch.sin(angles)
        cos_enc = torch.cos(angles)
        
        # Interleave sin and cos
        pos_enc = torch.stack([sin_enc, cos_enc], dim=-1)
        pos_enc = pos_enc.view(batch_size, num_positions, embed_dim)
        
        return pos_enc


class CrossSlideTemporalReasoner(nn.Module):
    """
    Complete temporal reasoning module for cross-slide analysis.
    
    Integrates temporal attention with progression feature extraction and
    temporal pooling to produce sequence-level representations for disease
    progression analysis.
    
    Args:
        embed_dim: Dimension of slide embeddings (default: 256)
        num_heads: Number of attention heads (default: 8)
        num_layers: Number of transformer layers (default: 2)
        max_temporal_distance: Maximum temporal distance in days (default: 365*5)
        dropout: Dropout rate (default: 0.1)
        pooling: Temporal pooling strategy ('mean', 'max', 'last', 'attention') (default: 'attention')
    
    Example:
        >>> reasoner = CrossSlideTemporalReasoner(embed_dim=256)
        >>> slide_embeddings = torch.randn(16, 5, 256)  # [batch, num_slides, embed_dim]
        >>> timestamps = torch.randn(16, 5)
        >>> sequence_emb, progression_features = reasoner(slide_embeddings, timestamps)
        >>> sequence_emb.shape, progression_features.shape
        (torch.Size([16, 256]), torch.Size([16, 128]))
    """
    
    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 2,
        max_temporal_distance: float = 365 * 5,
        dropout: float = 0.1,
        pooling: str = 'attention'
    ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.pooling = pooling
        
        # Temporal attention mechanism
        self.temporal_attention = TemporalAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            max_temporal_distance=max_temporal_distance,
            dropout=dropout
        )
        
        # Progression feature extraction
        # Compute differences between consecutive slides
        self.progression_proj = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim // 2)
        )
        
        # Attention-based pooling
        if pooling == 'attention':
            self.attention_pool = nn.Sequential(
                nn.Linear(embed_dim, 1),
                nn.Softmax(dim=1)
            )
        
        # Final projection for sequence-level representation
        self.sequence_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim)
        )
        
    def forward(
        self,
        slide_embeddings: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Apply temporal reasoning to slide sequence.
        
        Args:
            slide_embeddings: Slide embeddings [batch_size, num_slides, embed_dim]
            timestamps: Optional timestamps [batch_size, num_slides]
            mask: Optional mask [batch_size, num_slides] where True indicates valid slides
            
        Returns:
            Tuple of:
                - Sequence-level embedding [batch_size, embed_dim]
                - Progression features [batch_size, embed_dim // 2]
        """
        batch_size, num_slides, embed_dim = slide_embeddings.shape
        
        # Apply temporal attention
        attended_slides = self.temporal_attention(
            slide_embeddings,
            timestamps=timestamps,
            mask=mask
        )  # [B, num_slides, embed_dim]
        
        # Extract progression features (differences between consecutive slides)
        if num_slides > 1:
            # Compute pairwise differences
            slide_diffs = attended_slides[:, 1:, :] - attended_slides[:, :-1, :]  # [B, num_slides-1, embed_dim]
            
            # Concatenate with original embeddings for context
            slide_pairs = torch.cat([
                attended_slides[:, :-1, :],
                slide_diffs
            ], dim=-1)  # [B, num_slides-1, embed_dim*2]
            
            # Project to progression features
            progression_per_pair = self.progression_proj(slide_pairs)  # [B, num_slides-1, embed_dim//2]
            
            # Pool progression features
            if mask is not None:
                # Mask for pairs (valid if both slides in pair are valid)
                pair_mask = mask[:, :-1] & mask[:, 1:]  # [B, num_slides-1]
                masked_progression = progression_per_pair * pair_mask.unsqueeze(-1).float()
                progression_features = masked_progression.sum(dim=1) / (pair_mask.sum(dim=1, keepdim=True).float() + 1e-9)
            else:
                progression_features = progression_per_pair.mean(dim=1)  # [B, embed_dim//2]
        else:
            # Single slide: no progression features
            progression_features = torch.zeros(
                batch_size, embed_dim // 2,
                device=slide_embeddings.device
            )
        
        # Temporal pooling for sequence-level representation
        if self.pooling == 'attention':
            # Attention-weighted pooling
            attn_weights = self.attention_pool(attended_slides)  # [B, num_slides, 1]
            
            if mask is not None:
                # Mask out invalid slides
                attn_weights = attn_weights.masked_fill(~mask.unsqueeze(-1), 0.0)
                # Renormalize
                attn_weights = attn_weights / (attn_weights.sum(dim=1, keepdim=True) + 1e-9)
            
            sequence_emb = (attended_slides * attn_weights).sum(dim=1)  # [B, embed_dim]
        
        elif self.pooling == 'mean':
            # Mean pooling
            if mask is not None:
                masked_slides = attended_slides * mask.unsqueeze(-1).float()
                sequence_emb = masked_slides.sum(dim=1) / (mask.sum(dim=1, keepdim=True).float() + 1e-9)
            else:
                sequence_emb = attended_slides.mean(dim=1)
        
        elif self.pooling == 'max':
            # Max pooling
            if mask is not None:
                masked_slides = attended_slides.masked_fill(~mask.unsqueeze(-1), float('-inf'))
                sequence_emb = masked_slides.max(dim=1)[0]
            else:
                sequence_emb = attended_slides.max(dim=1)[0]
        
        elif self.pooling == 'last':
            # Use last valid slide
            if mask is not None:
                # Find last valid slide for each batch
                last_indices = mask.sum(dim=1) - 1  # [B]
                last_indices = last_indices.clamp(min=0)
                sequence_emb = attended_slides[torch.arange(batch_size), last_indices]
            else:
                sequence_emb = attended_slides[:, -1, :]
        
        else:
            raise ValueError(f"Unknown pooling strategy: {self.pooling}")
        
        # Final projection
        sequence_emb = self.sequence_proj(sequence_emb)  # [B, embed_dim]
        
        return sequence_emb, progression_features
