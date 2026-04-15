"""
Baseline model implementations for comparison with MultimodalFusionModel.

This module provides baseline architectures that isolate specific aspects
of the multimodal fusion model for ablation studies and performance comparison.

Baselines:
1. SingleModalityModel - Tests individual modality performance
2. LateFusionModel - Simple concatenation without cross-attention
3. AttentionBaseline - Self-attention only, no cross-modal interaction
"""

from typing import Dict, Optional

import torch
import torch.nn as nn

from .encoders import ClinicalTextEncoder, GenomicEncoder, WSIEncoder


class SingleModalityModel(nn.Module):
    """
    Baseline that uses only one modality with the full encoding pipeline.

    This baseline isolates the performance contribution of each individual
    modality (WSI, genomic, or clinical text) without any multimodal fusion.
    It demonstrates what level of performance is achievable with each
    modality alone and serves as a lower bound for fusion performance.

    Args:
        modality: Which modality to use ('wsi', 'genomic', or 'clinical')
        config: Configuration dict matching MultimodalFusionModel format
        embed_dim: Common embedding dimension (default: 256)

    Example:
        >>> model = SingleModalityModel(modality='wsi', embed_dim=256)
        >>> batch = {'wsi_features': torch.randn(16, 100, 1024)}
        >>> output = model(batch)
        >>> output.shape
        torch.Size([16, 256])
    """

    SUPPORTED_MODALITIES = ["wsi", "genomic", "clinical"]

    def __init__(self, modality: str, config: Optional[Dict] = None, embed_dim: int = 256):
        super().__init__()

        if modality not in self.SUPPORTED_MODALITIES:
            raise ValueError(f"Modality must be one of {self.SUPPORTED_MODALITIES}, got {modality}")

        self.modality = modality
        self.embed_dim = embed_dim
        config = config or {}

        # Initialize the appropriate encoder based on modality
        if modality == "wsi":
            wsi_config = config.get("wsi_config", {})
            self.encoder = WSIEncoder(
                input_dim=wsi_config.get("input_dim", 1024),
                hidden_dim=wsi_config.get("hidden_dim", 512),
                output_dim=embed_dim,
                num_heads=wsi_config.get("num_heads", 8),
                num_layers=wsi_config.get("num_layers", 2),
                dropout=wsi_config.get("dropout", 0.1),
                pooling=wsi_config.get("pooling", "attention"),
            )

        elif modality == "genomic":
            genomic_config = config.get("genomic_config", {})
            self.encoder = GenomicEncoder(
                input_dim=genomic_config.get("input_dim", 2000),
                hidden_dims=genomic_config.get("hidden_dims", [1024, 512]),
                output_dim=embed_dim,
                dropout=genomic_config.get("dropout", 0.3),
                use_batch_norm=genomic_config.get("use_batch_norm", True),
            )

        elif modality == "clinical":
            clinical_config = config.get("clinical_config", {})
            self.encoder = ClinicalTextEncoder(
                vocab_size=clinical_config.get("vocab_size", 30000),
                embed_dim=clinical_config.get("embed_dim", 256),
                hidden_dim=clinical_config.get("hidden_dim", 512),
                output_dim=embed_dim,
                num_heads=clinical_config.get("num_heads", 8),
                num_layers=clinical_config.get("num_layers", 3),
                max_seq_length=clinical_config.get("max_seq_length", 512),
                dropout=clinical_config.get("dropout", 0.1),
                pooling=clinical_config.get("pooling", "mean"),
            )

        # Projection to ensure consistent output dimension
        self.output_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim))

    def forward(self, batch: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        """
        Forward pass through single modality encoder.

        Args:
            batch: Dictionary containing the modality-specific data:
                - 'wsi_features': [batch_size, num_patches, feature_dim] or None
                - 'genomic': [batch_size, num_genes] or None
                - 'clinical_text': [batch_size, seq_len] or None

        Returns:
            Modality embedding [batch_size, embed_dim]
        """
        if self.modality == "wsi":
            wsi_features = batch.get("wsi_features")
            if wsi_features is None:
                raise ValueError(
                    "WSI features required for SingleModalityModel with modality='wsi'"
                )
            embedding = self.encoder(wsi_features, mask=batch.get("wsi_mask"))

        elif self.modality == "genomic":
            genomic = batch.get("genomic")
            if genomic is None:
                raise ValueError(
                    "Genomic data required for SingleModalityModel with modality='genomic'"
                )
            embedding = self.encoder(genomic)

        elif self.modality == "clinical":
            clinical_text = batch.get("clinical_text")
            if clinical_text is None:
                raise ValueError(
                    "Clinical text required for SingleModalityModel with modality='clinical'"
                )
            embedding = self.encoder(clinical_text, attention_mask=batch.get("clinical_mask"))

        # Project to consistent output dimension
        embedding = self.output_proj(embedding)
        return embedding

    def get_embedding_dim(self) -> int:
        """Return the dimension of the embedding."""
        return self.embed_dim


class LateFusionModel(nn.Module):
    """
    Late fusion baseline that concatenates modality embeddings without cross-attention.

    This baseline tests whether the multimodal performance comes from simply
    combining modality-specific representations, without the overhead of
    cross-modal attention mechanisms. It concatenates the outputs of
    independent modality encoders and projects them to a unified space.

    This baseline addresses: "Does cross-attention provide any benefit over
    simple concatenation of modality embeddings?"

    Args:
        config: Configuration dict matching MultimodalFusionModel format
        embed_dim: Common embedding dimension (default: 256)
        dropout: Dropout rate (default: 0.1)

    Example:
        >>> model = LateFusionModel(embed_dim=256)
        >>> batch = {
        ...     'wsi_features': torch.randn(16, 100, 1024),
        ...     'genomic': torch.randn(16, 2000),
        ...     'clinical_text': torch.randint(0, 30000, (16, 128))
        ... }
        >>> output = model(batch)
        >>> output.shape
        torch.Size([16, 256])
    """

    def __init__(self, config: Optional[Dict] = None, embed_dim: int = 256, dropout: float = 0.1):
        super().__init__()

        self.embed_dim = embed_dim
        config = config or {}

        # Initialize modality encoders
        wsi_config = config.get("wsi_config", {})
        self.wsi_encoder = WSIEncoder(
            input_dim=wsi_config.get("input_dim", 1024),
            hidden_dim=wsi_config.get("hidden_dim", 512),
            output_dim=embed_dim,
            num_heads=wsi_config.get("num_heads", 8),
            num_layers=wsi_config.get("num_layers", 2),
            dropout=wsi_config.get("dropout", dropout),
            pooling=wsi_config.get("pooling", "attention"),
        )

        genomic_config = config.get("genomic_config", {})
        self.genomic_encoder = GenomicEncoder(
            input_dim=genomic_config.get("input_dim", 2000),
            hidden_dims=genomic_config.get("hidden_dims", [1024, 512]),
            output_dim=embed_dim,
            dropout=genomic_config.get("dropout", dropout * 1.5),
            use_batch_norm=genomic_config.get("use_batch_norm", True),
        )

        clinical_config = config.get("clinical_config", {})
        self.clinical_encoder = ClinicalTextEncoder(
            vocab_size=clinical_config.get("vocab_size", 30000),
            embed_dim=clinical_config.get("embed_dim", 256),
            hidden_dim=clinical_config.get("hidden_dim", 512),
            output_dim=embed_dim,
            num_heads=clinical_config.get("num_heads", 8),
            num_layers=clinical_config.get("num_layers", 3),
            max_seq_length=clinical_config.get("max_seq_length", 512),
            dropout=clinical_config.get("dropout", dropout),
            pooling=clinical_config.get("pooling", "mean"),
        )

        # Modality list for reference
        self.modalities = ["wsi", "genomic", "clinical"]

        # Fusion: concatenate and project
        fusion_input_dim = embed_dim * len(self.modalities)
        self.fusion_proj = nn.Sequential(
            nn.Linear(fusion_input_dim, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def forward(self, batch: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        """
        Forward pass through late fusion model.

        Args:
            batch: Dictionary containing modality data:
                - 'wsi_features': [batch_size, num_patches, feature_dim] or None
                - 'genomic': [batch_size, num_genes] or None
                - 'clinical_text': [batch_size, seq_len] or None

        Returns:
            Fused embedding [batch_size, embed_dim]
        """
        device = next(self.parameters()).device
        batch_size = self._get_batch_size(batch)

        embeddings = []

        # Encode each modality and collect
        if batch.get("wsi_features") is not None:
            wsi_emb = self.wsi_encoder(batch["wsi_features"], mask=batch.get("wsi_mask"))
            embeddings.append(wsi_emb)
        else:
            # Use zeros for missing modality
            embeddings.append(torch.zeros(batch_size, self.embed_dim, device=device))

        if batch.get("genomic") is not None:
            genomic_emb = self.genomic_encoder(batch["genomic"])
            embeddings.append(genomic_emb)
        else:
            embeddings.append(torch.zeros(batch_size, self.embed_dim, device=device))

        if batch.get("clinical_text") is not None:
            clinical_emb = self.clinical_encoder(
                batch["clinical_text"], attention_mask=batch.get("clinical_mask")
            )
            embeddings.append(clinical_emb)
        else:
            embeddings.append(torch.zeros(batch_size, self.embed_dim, device=device))

        # Concatenate all embeddings
        concatenated = torch.cat(embeddings, dim=-1)

        # Project to final embedding
        fused = self.fusion_proj(concatenated)
        return fused

    def _get_batch_size(self, batch: Dict[str, Optional[torch.Tensor]]) -> int:
        """Extract batch size from the first available modality."""
        for key in ["wsi_features", "genomic", "clinical_text"]:
            if batch.get(key) is not None:
                return batch[key].shape[0]
        raise ValueError("At least one modality must be provided")

    def get_embedding_dim(self) -> int:
        """Return the dimension of the fused embedding."""
        return self.embed_dim


class AttentionBaseline(nn.Module):
    """
    Self-attention baseline without cross-modal attention mechanisms.

    This baseline tests whether the performance of MultimodalFusionModel
    comes from the self-attention mechanisms within each modality encoder
    rather than the cross-modal attention between modalities. It uses
    standard self-attention aggregation for each modality but does not
    include any cross-modal interaction layers.

    This baseline addresses: "Is the benefit from self-attention within
    modalities rather than cross-modal attention?"

    Args:
        config: Configuration dict matching MultimodalFusionModel format
        embed_dim: Common embedding dimension (default: 256)
        dropout: Dropout rate (default: 0.1)

    Example:
        >>> model = AttentionBaseline(embed_dim=256)
        >>> batch = {
        ...     'wsi_features': torch.randn(16, 100, 1024),
        ...     'genomic': torch.randn(16, 2000),
        ...     'clinical_text': torch.randint(0, 30000, (16, 128))
        ... }
        >>> output = model(batch)
        >>> output.shape
        torch.Size([16, 256])
    """

    def __init__(self, config: Optional[Dict] = None, embed_dim: int = 256, dropout: float = 0.1):
        super().__init__()

        self.embed_dim = embed_dim
        config = config or {}

        # Initialize modality encoders (these already use self-attention internally)
        wsi_config = config.get("wsi_config", {})
        self.wsi_encoder = WSIEncoder(
            input_dim=wsi_config.get("input_dim", 1024),
            hidden_dim=wsi_config.get("hidden_dim", 512),
            output_dim=embed_dim,
            num_heads=wsi_config.get("num_heads", 8),
            num_layers=wsi_config.get("num_layers", 2),
            dropout=wsi_config.get("dropout", dropout),
            pooling=wsi_config.get("pooling", "attention"),
        )

        genomic_config = config.get("genomic_config", {})
        self.genomic_encoder = GenomicEncoder(
            input_dim=genomic_config.get("input_dim", 2000),
            hidden_dims=genomic_config.get("hidden_dims", [1024, 512]),
            output_dim=embed_dim,
            dropout=genomic_config.get("dropout", dropout * 1.5),
            use_batch_norm=genomic_config.get("use_batch_norm", True),
        )

        clinical_config = config.get("clinical_config", {})
        self.clinical_encoder = ClinicalTextEncoder(
            vocab_size=clinical_config.get("vocab_size", 30000),
            embed_dim=clinical_config.get("embed_dim", 256),
            hidden_dim=clinical_config.get("hidden_dim", 512),
            output_dim=embed_dim,
            num_heads=clinical_config.get("num_heads", 8),
            num_layers=clinical_config.get("num_layers", 3),
            max_seq_length=clinical_config.get("max_seq_length", 512),
            dropout=clinical_config.get("dropout", dropout),
            pooling=clinical_config.get("pooling", "mean"),
        )

        # Modality list
        self.modalities = ["wsi", "genomic", "clinical"]

        # Simple weighted sum fusion (learnable weights per modality)
        self.fusion_weights = nn.Parameter(torch.ones(len(self.modalities)))

        # Output projection
        self.output_proj = nn.Sequential(nn.Linear(embed_dim, embed_dim), nn.LayerNorm(embed_dim))

    def forward(self, batch: Dict[str, Optional[torch.Tensor]]) -> torch.Tensor:
        """
        Forward pass through attention baseline model.

        Args:
            batch: Dictionary containing modality data:
                - 'wsi_features': [batch_size, num_patches, feature_dim] or None
                - 'genomic': [batch_size, num_genes] or None
                - 'clinical_text': [batch_size, seq_len] or None

        Returns:
            Fused embedding [batch_size, embed_dim]
        """
        device = next(self.parameters()).device
        batch_size = self._get_batch_size(batch)

        embeddings = []
        valid_mask = []

        # Encode each modality
        if batch.get("wsi_features") is not None:
            wsi_emb = self.wsi_encoder(batch["wsi_features"], mask=batch.get("wsi_mask"))
            embeddings.append(wsi_emb)
            valid_mask.append(True)
        else:
            embeddings.append(torch.zeros(batch_size, self.embed_dim, device=device))
            valid_mask.append(False)

        if batch.get("genomic") is not None:
            genomic_emb = self.genomic_encoder(batch["genomic"])
            embeddings.append(genomic_emb)
            valid_mask.append(True)
        else:
            embeddings.append(torch.zeros(batch_size, self.embed_dim, device=device))
            valid_mask.append(False)

        if batch.get("clinical_text") is not None:
            clinical_emb = self.clinical_encoder(
                batch["clinical_text"], attention_mask=batch.get("clinical_mask")
            )
            embeddings.append(clinical_emb)
            valid_mask.append(True)
        else:
            embeddings.append(torch.zeros(batch_size, self.embed_dim, device=device))
            valid_mask.append(False)

        # Stack embeddings: [batch_size, num_modalities, embed_dim]
        stacked = torch.stack(embeddings, dim=1)

        # Normalize fusion weights with softmax
        weights = torch.softmax(self.fusion_weights, dim=0)

        # Apply weights (zero out invalid modalities)
        weight_mask = torch.tensor(valid_mask, dtype=torch.float, device=device)
        weights = weights * weight_mask
        weights = weights / (weights.sum() + 1e-8)

        # Weighted sum: [batch_size, num_modalities, embed_dim] -> [batch_size, embed_dim]
        fused = (stacked * weights.view(1, -1, 1)).sum(dim=1)

        # Output projection
        fused = self.output_proj(fused)
        return fused

    def _get_batch_size(self, batch: Dict[str, Optional[torch.Tensor]]) -> int:
        """Extract batch size from the first available modality."""
        for key in ["wsi_features", "genomic", "clinical_text"]:
            if batch.get(key) is not None:
                return batch[key].shape[0]
        raise ValueError("At least one modality must be provided")

    def get_embedding_dim(self) -> int:
        """Return the dimension of the fused embedding."""
        return self.embed_dim


def get_baseline_model(baseline_type: str, config: Optional[Dict] = None, **kwargs) -> nn.Module:
    """
    Factory function to create baseline models.

    Args:
        baseline_type: Type of baseline ('single_modality', 'late_fusion', 'attention')
        config: Configuration dict
        **kwargs: Additional arguments (modality for single_modality, embed_dim, etc.)

    Returns:
        Baseline model instance

    Example:
        >>> model = get_baseline_model('late_fusion', embed_dim=256)
        >>> single_wsi = get_baseline_model('single_modality', modality='wsi')
    """
    if baseline_type == "single_modality":
        modality = kwargs.get("modality", "wsi")
        return SingleModalityModel(
            modality=modality, config=config, embed_dim=kwargs.get("embed_dim", 256)
        )
    elif baseline_type == "late_fusion":
        return LateFusionModel(
            config=config,
            embed_dim=kwargs.get("embed_dim", 256),
            dropout=kwargs.get("dropout", 0.1),
        )
    elif baseline_type == "attention":
        return AttentionBaseline(
            config=config,
            embed_dim=kwargs.get("embed_dim", 256),
            dropout=kwargs.get("dropout", 0.1),
        )
    else:
        raise ValueError(f"Unknown baseline type: {baseline_type}")
