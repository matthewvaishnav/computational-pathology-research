"""
Complete multimodal fusion model integrating all encoders and fusion mechanisms.

This module provides the end-to-end multimodal architecture that combines
WSI, genomic, and clinical text data through attention-based fusion.
"""

from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn as nn

from .encoders import ClinicalTextEncoder, GenomicEncoder, WSIEncoder
from .fusion import MultiModalFusionLayer


class MultimodalFusionModel(nn.Module):
    """
    Complete multimodal fusion model for computational pathology.

    Integrates modality-specific encoders (WSI, genomic, clinical text) with
    cross-modal attention fusion. Handles missing modalities gracefully through
    masking and produces a unified multimodal representation.

    Args:
        wsi_config: Configuration dict for WSI encoder
        genomic_config: Configuration dict for genomic encoder
        clinical_config: Configuration dict for clinical text encoder
        fusion_config: Configuration dict for fusion layer
        embed_dim: Common embedding dimension for all modalities (default: 256)
        dropout: Dropout rate (default: 0.1)

    Example:
        >>> model = MultimodalFusionModel(embed_dim=256)
        >>> batch = {
        ...     'wsi_features': torch.randn(16, 100, 1024),
        ...     'genomic': torch.randn(16, 2000),
        ...     'clinical_text': torch.randint(0, 30000, (16, 128))
        ... }
        >>> output = model(batch)
        >>> output.shape
        torch.Size([16, 256])
    """

    def __init__(
        self,
        wsi_config: Optional[Dict] = None,
        genomic_config: Optional[Dict] = None,
        clinical_config: Optional[Dict] = None,
        fusion_config: Optional[Dict] = None,
        embed_dim: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embed_dim = embed_dim

        # Default configurations
        if wsi_config is None:
            wsi_config = {
                "input_dim": 1024,
                "hidden_dim": 512,
                "output_dim": embed_dim,
                "num_heads": 8,
                "num_layers": 2,
                "dropout": dropout,
                "pooling": "attention",
            }

        if genomic_config is None:
            genomic_config = {
                "input_dim": 2000,
                "hidden_dims": [1024, 512],
                "output_dim": embed_dim,
                "dropout": dropout * 1.5,  # Higher dropout for genomic data
                "use_batch_norm": True,
            }

        if clinical_config is None:
            clinical_config = {
                "vocab_size": 30000,
                "embed_dim": 256,
                "hidden_dim": 512,
                "output_dim": embed_dim,
                "num_heads": 8,
                "num_layers": 3,
                "max_seq_length": 512,
                "dropout": dropout,
                "pooling": "mean",
            }

        if fusion_config is None:
            fusion_config = {
                "embed_dim": embed_dim,
                "num_heads": 8,
                "dropout": dropout,
                "modalities": ["wsi", "genomic", "clinical"],
            }

        # Initialize modality-specific encoders
        self.wsi_encoder = WSIEncoder(**wsi_config)
        self.genomic_encoder = GenomicEncoder(**genomic_config)
        self.clinical_encoder = ClinicalTextEncoder(**clinical_config)

        # Validate fusion config modalities match expected modalities
        self.modalities = ["wsi", "genomic", "clinical"]
        if "modalities" in fusion_config:
            if set(fusion_config["modalities"]) != set(self.modalities):
                raise ValueError(
                    f"Fusion config modalities {fusion_config['modalities']} "
                    f"do not match expected modalities {self.modalities}"
                )

        # Initialize fusion layer
        self.fusion_layer = MultiModalFusionLayer(**fusion_config)

    def forward(
        self, batch: Dict[str, Optional[torch.Tensor]], return_modality_embeddings: bool = False
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:
        """
        Forward pass through multimodal fusion model.

        Args:
            batch: Dictionary containing:
                - 'wsi_features': WSI patch features [batch_size, num_patches, feature_dim] or None
                - 'wsi_mask': Optional mask for WSI patches [batch_size, num_patches] (True = valid)
                - 'genomic': Genomic features [batch_size, num_genes] or None
                - 'genomic_mask': Optional mask for genomic samples [batch_size] (True = valid)
                - 'clinical_text': Clinical text token IDs [batch_size, seq_len] or None
                - 'clinical_mask': Optional mask for clinical text [batch_size, seq_len] (True = valid)
            return_modality_embeddings: If True, return individual modality embeddings

        Returns:
            If return_modality_embeddings is False:
                Fused embedding [batch_size, embed_dim]
            If return_modality_embeddings is True:
                Tuple of (fused_embedding [batch_size, embed_dim],
                         modality_embeddings_dict {str: Optional[Tensor]})
        """
        device = next(self.parameters()).device
        batch_size = self._get_batch_size(batch)

        # Encode each modality
        modality_embeddings = {}
        modality_masks = {}

        # WSI encoding
        if batch.get("wsi_features") is not None:
            wsi_emb = self.wsi_encoder(batch["wsi_features"], mask=batch.get("wsi_mask"))
            modality_embeddings["wsi"] = wsi_emb
            if batch.get("wsi_mask") is not None:
                # Mask is [batch_size, num_patches], reduce to [batch_size]
                modality_masks["wsi"] = batch["wsi_mask"].any(dim=1).to(device)
            else:
                modality_masks["wsi"] = torch.ones(batch_size, dtype=torch.bool, device=device)
        else:
            modality_embeddings["wsi"] = None
            modality_masks["wsi"] = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Genomic encoding
        if batch.get("genomic") is not None:
            genomic_emb = self.genomic_encoder(batch["genomic"])
            modality_embeddings["genomic"] = genomic_emb
            if batch.get("genomic_mask") is not None:
                # Mask is [batch_size], use directly
                modality_masks["genomic"] = batch["genomic_mask"].to(device)
            else:
                modality_masks["genomic"] = torch.ones(batch_size, dtype=torch.bool, device=device)
        else:
            modality_embeddings["genomic"] = None
            modality_masks["genomic"] = torch.zeros(batch_size, dtype=torch.bool, device=device)

        # Clinical text encoding
        if batch.get("clinical_text") is not None:
            clinical_emb = self.clinical_encoder(
                batch["clinical_text"], attention_mask=batch.get("clinical_mask")
            )
            modality_embeddings["clinical"] = clinical_emb
            if batch.get("clinical_mask") is not None:
                # Mask can be [batch_size, seq_len] or [batch_size], standardize to [batch_size]
                if batch["clinical_mask"].dim() > 1:
                    modality_masks["clinical"] = batch["clinical_mask"].any(dim=1).to(device)
                else:
                    modality_masks["clinical"] = batch["clinical_mask"].to(device)
            else:
                modality_masks["clinical"] = torch.ones(batch_size, dtype=torch.bool, device=device)
        else:
            modality_embeddings["clinical"] = None
            modality_masks["clinical"] = torch.zeros(batch_size, dtype=torch.bool, device=device)

        if all(embedding is None for embedding in modality_embeddings.values()):
            fused_embedding = torch.zeros(batch_size, self.embed_dim, device=device)
            if return_modality_embeddings:
                return fused_embedding, modality_embeddings
            return fused_embedding

        # Fuse modalities
        fused_embedding = self.fusion_layer(modality_embeddings, modality_masks)

        if return_modality_embeddings:
            return fused_embedding, modality_embeddings
        else:
            return fused_embedding

    def _get_batch_size(self, batch: Dict[str, Optional[torch.Tensor]]) -> int:
        """Extract batch size from the first available modality."""
        for key in ["wsi_features", "genomic", "clinical_text"]:
            if batch.get(key) is not None:
                return batch[key].shape[0]
        for key in ["wsi_mask", "genomic_mask", "clinical_mask"]:
            if batch.get(key) is not None:
                return batch[key].shape[0]
        if batch.get("patient_ids") is not None:
            return len(batch["patient_ids"])
        raise ValueError("At least one modality must be provided")

    def get_embedding_dim(self) -> int:
        """Return the dimension of the fused embedding."""
        return self.embed_dim
