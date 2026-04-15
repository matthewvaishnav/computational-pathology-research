"""
Risk factor analysis for disease development prediction.

This module provides risk scoring for disease development based on imaging features
and clinical risk factors, supporting early detection and preventive interventions.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from .patient_context import ClinicalMetadata, SmokingStatus
from .taxonomy import DiseaseTaxonomy

logger = logging.getLogger(__name__)


class RiskAnalyzer(nn.Module):
    """
    Risk analyzer for disease development prediction.

    Calculates risk scores (0.0-1.0) for disease development with multiple time horizons
    (1-year, 5-year, 10-year). Incorporates imaging features and clinical risk factors
    (smoking, family history, age, previous diagnoses) to detect pre-disease anomalies
    and generate risk scores for each disease state in the taxonomy.

    The risk analyzer uses separate prediction heads for each time horizon, allowing
    the model to learn different patterns for short-term vs long-term risk.

    Args:
        taxonomy: DiseaseTaxonomy instance defining disease classification scheme
        input_dim: Dimension of input embeddings (default: 256)
        hidden_dim: Dimension of hidden layers (default: 128)
        dropout: Dropout rate (default: 0.3)
        time_horizons: List of time horizons in years (default: [1, 5, 10])

    Example:
        >>> taxonomy = DiseaseTaxonomy(config_dict={
        ...     'name': 'Cancer Risk',
        ...     'diseases': [
        ...         {'id': 'benign', 'name': 'Benign', 'parent': None, 'children': []},
        ...         {'id': 'malignant', 'name': 'Malignant', 'parent': None, 'children': []},
        ...     ]
        ... })
        >>> analyzer = RiskAnalyzer(taxonomy, input_dim=256)
        >>>
        >>> # Imaging features from WSI
        >>> imaging_features = torch.randn(16, 256)
        >>>
        >>> # Clinical metadata
        >>> metadata = [ClinicalMetadata(age=65, smoking_status=SmokingStatus.FORMER)] * 16
        >>>
        >>> # Calculate risk scores
        >>> risk_output = analyzer(imaging_features, clinical_metadata=metadata)
        >>> risk_output['risk_scores'].shape  # [16, num_diseases, num_time_horizons]
    """

    def __init__(
        self,
        taxonomy: DiseaseTaxonomy,
        input_dim: int = 256,
        hidden_dim: int = 128,
        dropout: float = 0.3,
        time_horizons: List[int] = None,
    ):
        super().__init__()

        if not isinstance(taxonomy, DiseaseTaxonomy):
            raise TypeError(f"taxonomy must be DiseaseTaxonomy instance, got {type(taxonomy)}")

        if time_horizons is None:
            time_horizons = [1, 5, 10]

        self.taxonomy = taxonomy
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_diseases = taxonomy.get_num_classes()
        self.time_horizons = sorted(time_horizons)
        self.num_time_horizons = len(time_horizons)

        # Map disease IDs to indices
        self.disease_ids = taxonomy.disease_ids
        self.id_to_idx = {disease_id: idx for idx, disease_id in enumerate(self.disease_ids)}
        self.idx_to_id = {idx: disease_id for disease_id, idx in self.id_to_idx.items()}

        # Clinical risk factor encoder
        self.risk_factor_encoder = ClinicalRiskFactorEncoder(
            embed_dim=hidden_dim,
            dropout=dropout,
        )

        # Combined feature dimension (imaging + clinical risk factors)
        combined_dim = input_dim + hidden_dim

        # Shared feature processing
        self.feature_processor = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Separate prediction heads for each time horizon
        self.time_horizon_heads = nn.ModuleDict()
        for horizon in self.time_horizons:
            self.time_horizon_heads[f"horizon_{horizon}y"] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.num_diseases),
            )

        # Anomaly detection head for pre-disease detection
        self.anomaly_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, self.num_diseases),
        )

        logger.info(
            f"Initialized RiskAnalyzer with taxonomy '{taxonomy.name}' "
            f"({self.num_diseases} diseases, {self.num_time_horizons} time horizons)"
        )

    def forward(
        self,
        imaging_features: torch.Tensor,
        clinical_metadata: Optional[List[ClinicalMetadata]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Calculate risk scores for disease development.

        Args:
            imaging_features: Input imaging embeddings [batch_size, input_dim]
            clinical_metadata: Optional list of ClinicalMetadata instances (one per sample)

        Returns:
            Dictionary containing:
                - 'risk_scores': Risk scores [batch_size, num_diseases, num_time_horizons]
                                Values in [0, 1], higher = greater risk
                - 'anomaly_scores': Pre-disease anomaly scores [batch_size, num_diseases]
                                   Values in [0, 1], higher = more anomalous
                - 'primary_risk_disease': Disease with highest risk [batch_size, num_time_horizons]
                - 'max_risk_scores': Maximum risk score per sample [batch_size, num_time_horizons]

        Raises:
            ValueError: If imaging_features have incorrect shape
        """
        if imaging_features.dim() != 2:
            raise ValueError(
                f"Expected 2D imaging_features [batch_size, input_dim], got shape {imaging_features.shape}"
            )

        if imaging_features.shape[1] != self.input_dim:
            raise ValueError(
                f"Expected input_dim={self.input_dim}, got {imaging_features.shape[1]}"
            )

        batch_size = imaging_features.shape[0]
        device = imaging_features.device

        # Encode clinical risk factors
        if clinical_metadata is not None and len(clinical_metadata) > 0:
            if len(clinical_metadata) != batch_size:
                raise ValueError(
                    f"Clinical metadata batch size ({len(clinical_metadata)}) "
                    f"doesn't match imaging features ({batch_size})"
                )
            risk_factor_embeddings = self.risk_factor_encoder(clinical_metadata, device=device)
        else:
            # Use zero embeddings if no clinical metadata provided
            risk_factor_embeddings = torch.zeros(
                batch_size, self.hidden_dim, dtype=imaging_features.dtype, device=device
            )

        # Combine imaging features and clinical risk factors
        combined_features = torch.cat([imaging_features, risk_factor_embeddings], dim=-1)

        # Process combined features
        processed_features = self.feature_processor(combined_features)

        # Calculate risk scores for each time horizon
        risk_scores_list = []
        for horizon in self.time_horizons:
            horizon_key = f"horizon_{horizon}y"
            logits = self.time_horizon_heads[horizon_key](processed_features)
            # Apply sigmoid to get risk scores in [0, 1]
            risk_scores = torch.sigmoid(logits)
            risk_scores_list.append(risk_scores)

        # Stack risk scores: [batch_size, num_diseases, num_time_horizons]
        risk_scores = torch.stack(risk_scores_list, dim=-1)

        # Calculate anomaly scores for pre-disease detection
        anomaly_logits = self.anomaly_detector(processed_features)
        anomaly_scores = torch.sigmoid(anomaly_logits)

        # Identify primary risk disease (highest risk) for each time horizon
        max_risk_scores, primary_risk_disease = torch.max(risk_scores, dim=1)

        return {
            "risk_scores": risk_scores,
            "anomaly_scores": anomaly_scores,
            "primary_risk_disease": primary_risk_disease,
            "max_risk_scores": max_risk_scores,
        }

    def get_risk_by_disease_id(
        self,
        imaging_features: torch.Tensor,
        clinical_metadata: Optional[List[ClinicalMetadata]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Get risk scores organized by disease ID.

        Args:
            imaging_features: Input imaging embeddings [batch_size, input_dim]
            clinical_metadata: Optional list of ClinicalMetadata instances

        Returns:
            Dictionary mapping disease IDs to risk score tensors
            [batch_size, num_time_horizons]
        """
        output = self.forward(imaging_features, clinical_metadata)
        risk_scores = output["risk_scores"]

        # Organize by disease ID
        risk_by_disease = {}
        for disease_id, idx in self.id_to_idx.items():
            risk_by_disease[disease_id] = risk_scores[:, idx, :]

        return risk_by_disease

    def get_high_risk_cases(
        self,
        imaging_features: torch.Tensor,
        clinical_metadata: Optional[List[ClinicalMetadata]] = None,
        threshold: float = 0.5,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Identify high-risk cases exceeding threshold.

        Args:
            imaging_features: Input imaging embeddings [batch_size, input_dim]
            clinical_metadata: Optional list of ClinicalMetadata instances
            threshold: Risk threshold for flagging (default: 0.5)

        Returns:
            Tuple of:
                - Boolean mask [batch_size] indicating high-risk cases
                - Dictionary with details about flagged cases
        """
        output = self.forward(imaging_features, clinical_metadata)
        risk_scores = output["risk_scores"]
        max_risk_scores = output["max_risk_scores"]

        # Flag cases where any disease/time horizon exceeds threshold
        high_risk_mask = (risk_scores > threshold).any(dim=(1, 2))

        # Get details for flagged cases
        flagged_details = {
            "risk_scores": risk_scores[high_risk_mask],
            "anomaly_scores": output["anomaly_scores"][high_risk_mask],
            "primary_risk_disease": output["primary_risk_disease"][high_risk_mask],
            "max_risk_scores": max_risk_scores[high_risk_mask],
        }

        return high_risk_mask, flagged_details

    def get_time_horizon_names(self) -> List[str]:
        """
        Get human-readable time horizon names.

        Returns:
            List of time horizon names (e.g., ["1-year", "5-year", "10-year"])
        """
        return [f"{horizon}-year" for horizon in self.time_horizons]

    def __repr__(self) -> str:
        """String representation of risk analyzer."""
        return (
            f"RiskAnalyzer(\n"
            f"  taxonomy='{self.taxonomy.name}',\n"
            f"  num_diseases={self.num_diseases},\n"
            f"  time_horizons={self.time_horizons},\n"
            f"  input_dim={self.input_dim},\n"
            f"  hidden_dim={self.hidden_dim}\n"
            f")"
        )


class ClinicalRiskFactorEncoder(nn.Module):
    """
    Encoder for clinical risk factors into vector representations.

    Focuses on risk-relevant clinical factors: smoking status, family history,
    age, and previous diagnoses. Uses learned embeddings and projections to
    create risk factor representations.

    Args:
        embed_dim: Output embedding dimension (default: 128)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(self, embed_dim: int = 128, dropout: float = 0.1):
        super().__init__()

        self.embed_dim = embed_dim

        # Embedding dimensions for categorical fields
        categorical_embed_dim = 16

        # Learned embeddings for smoking status (major risk factor)
        self.smoking_embedding = nn.Embedding(len(SmokingStatus), categorical_embed_dim)

        # Age projection (age is a key risk factor)
        self.age_projection = nn.Sequential(
            nn.Linear(1, categorical_embed_dim),
            nn.LayerNorm(categorical_embed_dim),
            nn.GELU(),
        )

        # Family history projection (count-based)
        self.family_history_projection = nn.Sequential(
            nn.Linear(1, categorical_embed_dim),
            nn.LayerNorm(categorical_embed_dim),
            nn.GELU(),
        )

        # Combine all risk factors
        total_features = categorical_embed_dim * 3  # smoking, age, family history
        self.risk_combiner = nn.Sequential(
            nn.Linear(total_features, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Enum to index mapping
        self.smoking_to_idx = {status: idx for idx, status in enumerate(SmokingStatus)}

    def forward(
        self, clinical_metadata: List[ClinicalMetadata], device: torch.device
    ) -> torch.Tensor:
        """
        Encode clinical risk factors into embeddings.

        Args:
            clinical_metadata: List of ClinicalMetadata instances
            device: Device to create tensors on

        Returns:
            Risk factor embeddings [batch_size, embed_dim]
        """
        # Convert metadata to tensors
        smoking_indices = torch.tensor(
            [self.smoking_to_idx[m.smoking_status] for m in clinical_metadata],
            dtype=torch.long,
            device=device,
        )

        # Age (normalized to 0-1 range, assuming max age 100)
        ages = torch.tensor(
            [m.age / 100.0 if m.age is not None else 0.5 for m in clinical_metadata],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(-1)

        # Family history count (normalized, capped at 5)
        fh_counts = torch.tensor(
            [min(len(m.family_history) / 5.0, 1.0) for m in clinical_metadata],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(-1)

        # Embed categorical features
        smoking_emb = self.smoking_embedding(smoking_indices)

        # Project numerical features
        age_emb = self.age_projection(ages)
        fh_emb = self.family_history_projection(fh_counts)

        # Concatenate all risk factors
        all_risk_factors = torch.cat([smoking_emb, age_emb, fh_emb], dim=-1)

        # Combine into final risk factor embedding
        risk_embeddings = self.risk_combiner(all_risk_factors)

        return risk_embeddings
