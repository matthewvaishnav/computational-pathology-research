"""
Temporal progression modeling for disease state prediction.

This module provides models for predicting future disease states based on current
imaging, patient history, and risk factors. It extends existing temporal models
to incorporate treatment effects and patient-specific progression patterns.
"""

import logging
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.temporal import CrossSlideTemporalReasoner
from .patient_context import ClinicalMetadata

logger = logging.getLogger(__name__)


class TemporalProgressionModel(nn.Module):
    """
    Temporal progression model for predicting future disease states.

    Predicts future disease states based on current imaging, patient history,
    and risk factors. Outputs progression probabilities for multiple time horizons
    (3 months, 6 months, 1 year, 5 years). Incorporates treatment effects and learns
    patient-specific progression patterns from multiple scans.

    The model extends existing temporal reasoning infrastructure (CrossSlideTemporalReasoner)
    and integrates with clinical metadata and risk factors.

    Args:
        taxonomy: DiseaseTaxonomy instance defining disease classification scheme
        input_dim: Dimension of input embeddings (default: 256)
        hidden_dim: Dimension of hidden layers (default: 128)
        num_temporal_heads: Number of attention heads for temporal reasoning (default: 8)
        num_temporal_layers: Number of transformer layers (default: 2)
        dropout: Dropout rate (default: 0.3)
        time_horizons: List of time horizons in months (default: [3, 6, 12, 60])

    Example:
        >>> from .taxonomy import DiseaseTaxonomy
        >>> taxonomy = DiseaseTaxonomy(config_dict={
        ...     'name': 'Cancer Grading',
        ...     'diseases': [
        ...         {'id': 'benign', 'name': 'Benign', 'parent': None, 'children': []},
        ...         {'id': 'grade_1', 'name': 'Grade 1', 'parent': None, 'children': []},
        ...     ]
        ... })
        >>> model = TemporalProgressionModel(taxonomy, input_dim=256)
        >>>
        >>> # Single scan prediction
        >>> current_features = torch.randn(16, 256)
        >>> output = model(current_features)
        >>> output['progression_probabilities'].shape  # [16, num_diseases, num_horizons]
        >>>
        >>> # Multi-scan prediction with history
        >>> scan_sequence = torch.randn(16, 5, 256)  # [batch, num_scans, embed_dim]
        >>> timestamps = torch.tensor([[0, 30, 60, 90, 120]] * 16, dtype=torch.float32)
        >>> output = model(scan_sequence, timestamps=timestamps)
    """

    def __init__(
        self,
        taxonomy,
        input_dim: int = 256,
        hidden_dim: int = 128,
        num_temporal_heads: int = 8,
        num_temporal_layers: int = 2,
        dropout: float = 0.3,
        time_horizons: List[int] = None,
    ):
        super().__init__()

        if time_horizons is None:
            time_horizons = [3, 6, 12, 60]  # 3mo, 6mo, 1yr, 5yr

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

        # Temporal reasoning module for multi-scan sequences
        self.temporal_reasoner = CrossSlideTemporalReasoner(
            embed_dim=input_dim,
            num_heads=num_temporal_heads,
            num_layers=num_temporal_layers,
            max_temporal_distance=365 * 5,  # 5 years in days
            dropout=dropout,
            pooling="attention",
        )

        # Clinical metadata encoder
        self.clinical_encoder = ClinicalMetadataEncoder(
            embed_dim=hidden_dim,
            dropout=dropout,
        )

        # Treatment effect encoder
        self.treatment_encoder = TreatmentEffectEncoder(
            embed_dim=hidden_dim,
            dropout=dropout,
        )

        # Combined feature dimension
        # imaging (input_dim) + progression (input_dim//2) + clinical (hidden_dim) + treatment (hidden_dim)
        combined_dim = input_dim + (input_dim // 2) + hidden_dim + hidden_dim

        # Feature fusion
        self.feature_fusion = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim * 2),
            nn.LayerNorm(hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
        )

        # Separate prediction heads for each time horizon
        self.progression_heads = nn.ModuleDict()
        for horizon in self.time_horizons:
            self.progression_heads[f"horizon_{horizon}mo"] = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, self.num_diseases),
            )

        # Rapid progression detector
        self.rapid_progression_detector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

        logger.info(
            f"Initialized TemporalProgressionModel with taxonomy '{taxonomy.name}' "
            f"({self.num_diseases} diseases, {self.num_time_horizons} time horizons)"
        )

    def forward(
        self,
        imaging_features: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        clinical_metadata: Optional[List[ClinicalMetadata]] = None,
        treatment_history: Optional[List[Dict]] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict future disease state progression.

        Args:
            imaging_features: Input imaging embeddings
                             Single scan: [batch_size, input_dim]
                             Multi-scan: [batch_size, num_scans, input_dim]
            timestamps: Optional timestamps for multi-scan sequences [batch_size, num_scans]
                       (in days relative to first scan)
            clinical_metadata: Optional list of ClinicalMetadata instances (one per sample)
            treatment_history: Optional list of treatment history dicts (one per sample)
                              Each dict contains: {'treatments': List[str], 'dates': List[int]}
            mask: Optional mask for multi-scan sequences [batch_size, num_scans]
                 where True indicates valid scans

        Returns:
            Dictionary containing:
                - 'progression_probabilities': Progression probabilities
                                               [batch_size, num_diseases, num_time_horizons]
                                               Values in [0, 1], sum to 1.0 per horizon
                - 'rapid_progression_risk': Risk of rapid progression [batch_size]
                                           Values in [0, 1], higher = greater risk
                - 'primary_progression_state': Most likely future disease state per horizon
                                              [batch_size, num_time_horizons]
                - 'progression_confidence': Confidence in primary progression state
                                           [batch_size, num_time_horizons]

        Raises:
            ValueError: If imaging_features have incorrect shape
        """
        device = imaging_features.device
        is_multi_scan = imaging_features.dim() == 3

        if is_multi_scan:
            # Multi-scan sequence: [batch_size, num_scans, input_dim]
            batch_size, num_scans, embed_dim = imaging_features.shape

            if embed_dim != self.input_dim:
                raise ValueError(f"Expected input_dim={self.input_dim}, got {embed_dim}")

            # Apply temporal reasoning
            sequence_emb, progression_features = self.temporal_reasoner(
                imaging_features, timestamps=timestamps, mask=mask
            )
            # sequence_emb: [batch_size, input_dim]
            # progression_features: [batch_size, input_dim // 2]

        else:
            # Single scan: [batch_size, input_dim]
            if imaging_features.dim() != 2:
                raise ValueError(
                    f"Expected 2D or 3D imaging_features, got shape {imaging_features.shape}"
                )

            if imaging_features.shape[1] != self.input_dim:
                raise ValueError(
                    f"Expected input_dim={self.input_dim}, got {imaging_features.shape[1]}"
                )

            batch_size = imaging_features.shape[0]
            sequence_emb = imaging_features
            # No progression features for single scan
            progression_features = torch.zeros(
                batch_size, self.input_dim // 2, dtype=imaging_features.dtype, device=device
            )

        # Encode clinical metadata
        if clinical_metadata is not None and len(clinical_metadata) > 0:
            if len(clinical_metadata) != batch_size:
                raise ValueError(
                    f"Clinical metadata batch size ({len(clinical_metadata)}) "
                    f"doesn't match imaging features ({batch_size})"
                )
            clinical_embeddings = self.clinical_encoder(clinical_metadata, device=device)
        else:
            clinical_embeddings = torch.zeros(
                batch_size, self.hidden_dim, dtype=imaging_features.dtype, device=device
            )

        # Encode treatment history
        if treatment_history is not None and len(treatment_history) > 0:
            if len(treatment_history) != batch_size:
                raise ValueError(
                    f"Treatment history batch size ({len(treatment_history)}) "
                    f"doesn't match imaging features ({batch_size})"
                )
            treatment_embeddings = self.treatment_encoder(treatment_history, device=device)
        else:
            treatment_embeddings = torch.zeros(
                batch_size, self.hidden_dim, dtype=imaging_features.dtype, device=device
            )

        # Combine all features
        combined_features = torch.cat(
            [sequence_emb, progression_features, clinical_embeddings, treatment_embeddings],
            dim=-1,
        )

        # Fuse features
        fused_features = self.feature_fusion(combined_features)

        # Predict progression for each time horizon
        progression_probs_list = []
        for horizon in self.time_horizons:
            horizon_key = f"horizon_{horizon}mo"
            logits = self.progression_heads[horizon_key](fused_features)
            # Apply softmax to get probability distribution
            probs = F.softmax(logits, dim=1)
            progression_probs_list.append(probs)

        # Stack progression probabilities: [batch_size, num_diseases, num_time_horizons]
        progression_probabilities = torch.stack(progression_probs_list, dim=-1)

        # Detect rapid progression risk
        rapid_progression_logits = self.rapid_progression_detector(fused_features)
        rapid_progression_risk = torch.sigmoid(rapid_progression_logits).squeeze(-1)

        # Identify primary progression state (highest probability) for each horizon
        progression_confidence, primary_progression_state = torch.max(
            progression_probabilities, dim=1
        )

        return {
            "progression_probabilities": progression_probabilities,
            "rapid_progression_risk": rapid_progression_risk,
            "primary_progression_state": primary_progression_state,
            "progression_confidence": progression_confidence,
        }

    def predict_with_treatment(
        self,
        imaging_features: torch.Tensor,
        current_disease_state: str,
        proposed_treatment: str,
        timestamps: Optional[torch.Tensor] = None,
        clinical_metadata: Optional[List[ClinicalMetadata]] = None,
        treatment_history: Optional[List[Dict]] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Predict progression with proposed treatment incorporated.

        Args:
            imaging_features: Input imaging embeddings
            current_disease_state: Current disease state ID
            proposed_treatment: Proposed treatment type
            timestamps: Optional timestamps for multi-scan sequences
            clinical_metadata: Optional clinical metadata
            treatment_history: Optional treatment history
            mask: Optional mask for multi-scan sequences

        Returns:
            Dictionary with progression predictions incorporating treatment effects
        """
        batch_size = (
            imaging_features.shape[0] if imaging_features.dim() == 2 else imaging_features.shape[0]
        )

        # Create augmented treatment history with proposed treatment
        if treatment_history is None:
            treatment_history = [{"treatments": [], "dates": []} for _ in range(batch_size)]
        else:
            treatment_history = [th.copy() for th in treatment_history]

        # Add proposed treatment (assume it starts now, day 0)
        for th in treatment_history:
            th["treatments"] = th.get("treatments", []) + [proposed_treatment]
            th["dates"] = th.get("dates", []) + [0]

        # Run forward pass with augmented treatment history
        return self.forward(
            imaging_features,
            timestamps=timestamps,
            clinical_metadata=clinical_metadata,
            treatment_history=treatment_history,
            mask=mask,
        )

    def identify_rapid_progressors(
        self,
        imaging_features: torch.Tensor,
        timestamps: Optional[torch.Tensor] = None,
        clinical_metadata: Optional[List[ClinicalMetadata]] = None,
        treatment_history: Optional[List[Dict]] = None,
        mask: Optional[torch.Tensor] = None,
        threshold: float = 0.7,
        urgent_threshold: float = 0.85,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Identify patients at high risk for rapid progression.

        Flags cases requiring urgent intervention based on rapid progression risk scores
        and predicted disease state changes. Uses two thresholds: one for high-risk
        flagging and one for urgent intervention.

        Args:
            imaging_features: Input imaging embeddings
            timestamps: Optional timestamps for multi-scan sequences
            clinical_metadata: Optional clinical metadata
            treatment_history: Optional treatment history
            mask: Optional mask for multi-scan sequences
            threshold: Risk threshold for high-risk flagging (default: 0.7)
            urgent_threshold: Risk threshold for urgent intervention (default: 0.85)

        Returns:
            Tuple of:
                - Boolean mask [batch_size] indicating high-risk cases
                - Dictionary with details about flagged cases including:
                  - 'rapid_progression_risk': Risk scores for flagged cases
                  - 'progression_probabilities': Progression probabilities
                  - 'primary_progression_state': Most likely future states
                  - 'progression_confidence': Confidence scores
                  - 'urgent_intervention_needed': Boolean mask for urgent cases
                  - 'recommendations': List of clinical recommendations
        """
        output = self.forward(
            imaging_features,
            timestamps=timestamps,
            clinical_metadata=clinical_metadata,
            treatment_history=treatment_history,
            mask=mask,
        )

        rapid_progression_risk = output["rapid_progression_risk"]

        # Flag cases exceeding threshold
        high_risk_mask = rapid_progression_risk > threshold

        # Flag cases requiring urgent intervention
        urgent_mask = rapid_progression_risk > urgent_threshold

        # Generate recommendations for flagged cases
        recommendations = []
        for i in range(rapid_progression_risk.shape[0]):
            if high_risk_mask[i]:
                risk_score = rapid_progression_risk[i].item()
                if urgent_mask[i]:
                    recommendations.append(
                        f"URGENT: Very high rapid progression risk ({risk_score:.2f}). "
                        "Immediate clinical intervention recommended."
                    )
                else:
                    recommendations.append(
                        f"High rapid progression risk ({risk_score:.2f}). "
                        "Close monitoring and early intervention recommended."
                    )
            else:
                recommendations.append(None)

        # Get details for flagged cases
        flagged_details = {
            "rapid_progression_risk": rapid_progression_risk[high_risk_mask],
            "progression_probabilities": output["progression_probabilities"][high_risk_mask],
            "primary_progression_state": output["primary_progression_state"][high_risk_mask],
            "progression_confidence": output["progression_confidence"][high_risk_mask],
            "urgent_intervention_needed": urgent_mask[high_risk_mask],
            "recommendations": [r for r in recommendations if r is not None],
        }

        return high_risk_mask, flagged_details

    def get_time_horizon_names(self) -> List[str]:
        """
        Get human-readable time horizon names.

        Returns:
            List of time horizon names (e.g., ["3-month", "6-month", "1-year", "5-year"])
        """
        names = []
        for horizon in self.time_horizons:
            if horizon < 12:
                names.append(f"{horizon}-month")
            elif horizon == 12:
                names.append("1-year")
            else:
                years = horizon // 12
                names.append(f"{years}-year")
        return names

    def __repr__(self) -> str:
        """String representation of temporal progression model."""
        return (
            f"TemporalProgressionModel(\n"
            f"  taxonomy='{self.taxonomy.name}',\n"
            f"  num_diseases={self.num_diseases},\n"
            f"  time_horizons={self.time_horizons},\n"
            f"  input_dim={self.input_dim},\n"
            f"  hidden_dim={self.hidden_dim}\n"
            f")"
        )


class ClinicalMetadataEncoder(nn.Module):
    """
    Encoder for clinical metadata into vector representations.

    Encodes clinical factors relevant to disease progression: age, smoking status,
    family history, comorbidities, and other risk factors.

    Args:
        embed_dim: Output embedding dimension (default: 128)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(self, embed_dim: int = 128, dropout: float = 0.1):
        super().__init__()

        self.embed_dim = embed_dim

        # Import SmokingStatus enum
        from .patient_context import SmokingStatus

        self.SmokingStatus = SmokingStatus

        # Embedding dimensions for categorical fields
        categorical_embed_dim = 16

        # Learned embeddings for smoking status
        self.smoking_embedding = nn.Embedding(len(SmokingStatus), categorical_embed_dim)

        # Age projection
        self.age_projection = nn.Sequential(
            nn.Linear(1, categorical_embed_dim),
            nn.LayerNorm(categorical_embed_dim),
            nn.GELU(),
        )

        # Family history projection
        self.family_history_projection = nn.Sequential(
            nn.Linear(1, categorical_embed_dim),
            nn.LayerNorm(categorical_embed_dim),
            nn.GELU(),
        )

        # Combine all clinical factors
        total_features = categorical_embed_dim * 3
        self.clinical_combiner = nn.Sequential(
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
        Encode clinical metadata into embeddings.

        Args:
            clinical_metadata: List of ClinicalMetadata instances
            device: Device to create tensors on

        Returns:
            Clinical embeddings [batch_size, embed_dim]
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

        # Concatenate all clinical factors
        all_clinical_factors = torch.cat([smoking_emb, age_emb, fh_emb], dim=-1)

        # Combine into final clinical embedding
        clinical_embeddings = self.clinical_combiner(all_clinical_factors)

        return clinical_embeddings


class TreatmentEffectEncoder(nn.Module):
    """
    Encoder for treatment history and effects.

    Encodes treatment types, timing, and sequences to model treatment effects
    on disease progression.

    Args:
        embed_dim: Output embedding dimension (default: 128)
        num_treatment_types: Number of treatment types to embed (default: 50)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(self, embed_dim: int = 128, num_treatment_types: int = 50, dropout: float = 0.1):
        super().__init__()

        self.embed_dim = embed_dim
        self.num_treatment_types = num_treatment_types

        # Treatment type vocabulary (simple hash-based mapping)
        self.treatment_vocab = {}
        self.next_treatment_id = 0

        # Learned embeddings for treatment types
        self.treatment_embedding = nn.Embedding(num_treatment_types, embed_dim)

        # Temporal encoding for treatment timing
        self.temporal_projection = nn.Sequential(
            nn.Linear(1, embed_dim // 4),
            nn.LayerNorm(embed_dim // 4),
            nn.GELU(),
        )

        # Treatment sequence encoder
        self.sequence_encoder = nn.Sequential(
            nn.Linear(embed_dim + embed_dim // 4, embed_dim),
            nn.LayerNorm(embed_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim, embed_dim),
            nn.LayerNorm(embed_dim),
        )

    def _get_treatment_id(self, treatment_name: str) -> int:
        """
        Get or create treatment ID for treatment name.

        Args:
            treatment_name: Treatment name string

        Returns:
            Treatment ID (integer)
        """
        if treatment_name not in self.treatment_vocab:
            if self.next_treatment_id >= self.num_treatment_types:
                # Use hash-based mapping for overflow
                return hash(treatment_name) % self.num_treatment_types
            self.treatment_vocab[treatment_name] = self.next_treatment_id
            self.next_treatment_id += 1

        return self.treatment_vocab[treatment_name]

    def forward(self, treatment_history: List[Dict], device: torch.device) -> torch.Tensor:
        """
        Encode treatment history into embeddings.

        Args:
            treatment_history: List of treatment history dicts (one per sample)
                              Each dict contains:
                              - 'treatments': List[str] - treatment names
                              - 'dates': List[int] - treatment dates (days)
            device: Device to create tensors on

        Returns:
            Treatment embeddings [batch_size, embed_dim]
        """
        batch_size = len(treatment_history)
        treatment_embeddings_list = []

        for th in treatment_history:
            treatments = th.get("treatments", [])
            dates = th.get("dates", [])

            if len(treatments) == 0:
                # No treatment history - use zero embedding
                treatment_emb = torch.zeros(self.embed_dim, device=device)
            else:
                # Encode each treatment
                treatment_ids = [self._get_treatment_id(t) for t in treatments]
                treatment_ids_tensor = torch.tensor(treatment_ids, dtype=torch.long, device=device)

                # Get treatment embeddings
                treatment_embs = self.treatment_embedding(
                    treatment_ids_tensor
                )  # [num_treatments, embed_dim]

                # Encode temporal information
                dates_tensor = torch.tensor(dates, dtype=torch.float32, device=device).unsqueeze(
                    -1
                )  # [num_treatments, 1]
                # Normalize dates (assuming max 5 years = 1825 days)
                dates_normalized = dates_tensor / 1825.0
                temporal_embs = self.temporal_projection(
                    dates_normalized
                )  # [num_treatments, embed_dim//4]

                # Combine treatment and temporal embeddings
                combined = torch.cat(
                    [treatment_embs, temporal_embs], dim=-1
                )  # [num_treatments, embed_dim + embed_dim//4]

                # Encode sequence
                sequence_embs = self.sequence_encoder(combined)  # [num_treatments, embed_dim]

                # Pool across treatments (mean pooling)
                treatment_emb = sequence_embs.mean(dim=0)  # [embed_dim]

            treatment_embeddings_list.append(treatment_emb)

        # Stack into batch
        treatment_embeddings = torch.stack(
            treatment_embeddings_list, dim=0
        )  # [batch_size, embed_dim]

        return treatment_embeddings
