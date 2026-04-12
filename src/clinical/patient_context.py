"""
Patient context integration for clinical workflow.

This module provides data structures and integration logic for combining
imaging features with clinical metadata to create multimodal patient
representations for improved diagnostic predictions.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# Enums for categorical clinical metadata fields
class SmokingStatus(str, Enum):
    """Smoking status categories."""

    NEVER = "never"
    FORMER = "former"
    CURRENT = "current"
    UNKNOWN = "unknown"


class AlcoholConsumption(str, Enum):
    """Alcohol consumption categories."""

    NONE = "none"
    LIGHT = "light"  # 1-7 drinks/week
    MODERATE = "moderate"  # 8-14 drinks/week
    HEAVY = "heavy"  # >14 drinks/week
    UNKNOWN = "unknown"


class ExerciseFrequency(str, Enum):
    """Exercise frequency categories."""

    NONE = "none"
    LIGHT = "light"  # <1 hour/week
    MODERATE = "moderate"  # 1-3 hours/week
    ACTIVE = "active"  # >3 hours/week
    UNKNOWN = "unknown"


class Sex(str, Enum):
    """Biological sex categories."""

    MALE = "male"
    FEMALE = "female"
    OTHER = "other"
    UNKNOWN = "unknown"


@dataclass
class ClinicalMetadata:
    """
    Structured clinical metadata for patient context.

    This dataclass stores patient clinical information including lifestyle factors,
    demographics, and medical history. It supports validation, serialization, and
    integration with multimodal fusion for diagnostic predictions.

    Required fields: None (all fields are optional to handle incomplete data)
    Optional fields: All fields have defaults to handle missing data gracefully

    Attributes:
        smoking_status: Patient smoking status (never/former/current/unknown)
        alcohol_consumption: Alcohol consumption level (none/light/moderate/heavy/unknown)
        medications: List of current medications (generic names or codes)
        exercise_frequency: Exercise frequency (none/light/moderate/active/unknown)
        age: Patient age in years (None if unknown)
        sex: Biological sex (male/female/other/unknown)
        family_history: List of diseases in family history (e.g., ["breast_cancer", "diabetes"])
        metadata: Additional metadata fields (flexible dict for extensibility)

    Example:
        >>> metadata = ClinicalMetadata(
        ...     smoking_status=SmokingStatus.FORMER,
        ...     alcohol_consumption=AlcoholConsumption.LIGHT,
        ...     medications=["metformin", "lisinopril"],
        ...     exercise_frequency=ExerciseFrequency.MODERATE,
        ...     age=65,
        ...     sex=Sex.MALE,
        ...     family_history=["diabetes", "hypertension"]
        ... )
        >>> metadata.validate()
        >>> metadata.to_dict()
        >>> metadata.to_json("patient_metadata.json")
    """

    smoking_status: SmokingStatus = SmokingStatus.UNKNOWN
    alcohol_consumption: AlcoholConsumption = AlcoholConsumption.UNKNOWN
    medications: List[str] = field(default_factory=list)
    exercise_frequency: ExerciseFrequency = ExerciseFrequency.UNKNOWN
    age: Optional[int] = None
    sex: Sex = Sex.UNKNOWN
    family_history: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        """
        Validate clinical metadata fields.

        Raises:
            ValueError: If any field contains invalid values
        """
        errors = []

        # Validate smoking_status
        if not isinstance(self.smoking_status, SmokingStatus):
            try:
                self.smoking_status = SmokingStatus(self.smoking_status)
            except ValueError:
                errors.append(
                    f"Invalid smoking_status: {self.smoking_status}. "
                    f"Must be one of {[s.value for s in SmokingStatus]}"
                )

        # Validate alcohol_consumption
        if not isinstance(self.alcohol_consumption, AlcoholConsumption):
            try:
                self.alcohol_consumption = AlcoholConsumption(self.alcohol_consumption)
            except ValueError:
                errors.append(
                    f"Invalid alcohol_consumption: {self.alcohol_consumption}. "
                    f"Must be one of {[a.value for a in AlcoholConsumption]}"
                )

        # Validate exercise_frequency
        if not isinstance(self.exercise_frequency, ExerciseFrequency):
            try:
                self.exercise_frequency = ExerciseFrequency(self.exercise_frequency)
            except ValueError:
                errors.append(
                    f"Invalid exercise_frequency: {self.exercise_frequency}. "
                    f"Must be one of {[e.value for e in ExerciseFrequency]}"
                )

        # Validate sex
        if not isinstance(self.sex, Sex):
            try:
                self.sex = Sex(self.sex)
            except ValueError:
                errors.append(
                    f"Invalid sex: {self.sex}. " f"Must be one of {[s.value for s in Sex]}"
                )

        # Validate age
        if self.age is not None:
            if not isinstance(self.age, int):
                errors.append(f"Age must be an integer, got {type(self.age)}")
            elif self.age < 0 or self.age > 150:
                errors.append(f"Age must be between 0 and 150, got {self.age}")

        # Validate medications
        if not isinstance(self.medications, list):
            errors.append(f"Medications must be a list, got {type(self.medications)}")
        elif not all(isinstance(med, str) for med in self.medications):
            errors.append("All medications must be strings")

        # Validate family_history
        if not isinstance(self.family_history, list):
            errors.append(f"Family history must be a list, got {type(self.family_history)}")
        elif not all(isinstance(disease, str) for disease in self.family_history):
            errors.append("All family history entries must be strings")

        # Validate metadata
        if not isinstance(self.metadata, dict):
            errors.append(f"Metadata must be a dict, got {type(self.metadata)}")

        if errors:
            error_msg = "Clinical metadata validation failed:\n" + "\n".join(
                f"  - {error}" for error in errors
            )
            raise ValueError(error_msg)

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert clinical metadata to dictionary format.

        Returns:
            Dictionary representation with enum values converted to strings
        """
        data = asdict(self)

        # Convert enums to string values
        data["smoking_status"] = self.smoking_status.value
        data["alcohol_consumption"] = self.alcohol_consumption.value
        data["exercise_frequency"] = self.exercise_frequency.value
        data["sex"] = self.sex.value

        return data

    def to_json(self, output_path: Union[str, Path]) -> None:
        """
        Save clinical metadata to JSON file.

        Args:
            output_path: Output file path
        """
        output_path = Path(output_path)
        data = self.to_dict()

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved clinical metadata to {output_path}")

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ClinicalMetadata":
        """
        Create ClinicalMetadata from dictionary.

        Args:
            data: Dictionary containing clinical metadata fields

        Returns:
            ClinicalMetadata instance
        """
        # Convert string values to enums
        if "smoking_status" in data and isinstance(data["smoking_status"], str):
            data["smoking_status"] = SmokingStatus(data["smoking_status"])

        if "alcohol_consumption" in data and isinstance(data["alcohol_consumption"], str):
            data["alcohol_consumption"] = AlcoholConsumption(data["alcohol_consumption"])

        if "exercise_frequency" in data and isinstance(data["exercise_frequency"], str):
            data["exercise_frequency"] = ExerciseFrequency(data["exercise_frequency"])

        if "sex" in data and isinstance(data["sex"], str):
            data["sex"] = Sex(data["sex"])

        return cls(**data)

    @classmethod
    def from_json(cls, input_path: Union[str, Path]) -> "ClinicalMetadata":
        """
        Load clinical metadata from JSON file.

        Args:
            input_path: Input file path

        Returns:
            ClinicalMetadata instance
        """
        input_path = Path(input_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Clinical metadata file not found: {input_path}")

        with open(input_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        return cls.from_dict(data)

    def is_complete(self) -> bool:
        """
        Check if all clinical metadata fields are populated (not unknown/None).

        Returns:
            True if all fields have known values
        """
        return (
            self.smoking_status != SmokingStatus.UNKNOWN
            and self.alcohol_consumption != AlcoholConsumption.UNKNOWN
            and self.exercise_frequency != ExerciseFrequency.UNKNOWN
            and self.sex != Sex.UNKNOWN
            and self.age is not None
        )

    def get_missing_fields(self) -> List[str]:
        """
        Get list of missing or unknown clinical metadata fields.

        Returns:
            List of field names that are missing or unknown
        """
        missing = []

        if self.smoking_status == SmokingStatus.UNKNOWN:
            missing.append("smoking_status")
        if self.alcohol_consumption == AlcoholConsumption.UNKNOWN:
            missing.append("alcohol_consumption")
        if self.exercise_frequency == ExerciseFrequency.UNKNOWN:
            missing.append("exercise_frequency")
        if self.sex == Sex.UNKNOWN:
            missing.append("sex")
        if self.age is None:
            missing.append("age")

        return missing

    def __repr__(self) -> str:
        """String representation of clinical metadata."""
        missing = self.get_missing_fields()
        complete_str = "complete" if not missing else f"missing {len(missing)} fields"
        return (
            f"ClinicalMetadata(age={self.age}, sex={self.sex.value}, "
            f"smoking={self.smoking_status.value}, {complete_str})"
        )


class PatientContextIntegrator(nn.Module):
    """
    Patient context integrator combining imaging features with clinical metadata.

    This module extends multimodal fusion to incorporate structured clinical metadata
    alongside imaging features (WSI, genomic, clinical text). It encodes clinical
    metadata into vector representations and integrates them through cross-modal
    attention fusion.

    The integrator handles missing/incomplete metadata gracefully using masking,
    allowing predictions even when some clinical information is unavailable.

    Args:
        embed_dim: Dimension of embeddings (default: 256)
        num_heads: Number of attention heads for fusion (default: 8)
        dropout: Dropout rate (default: 0.1)
        modalities: List of modality names including 'clinical_metadata' (default: ['wsi', 'clinical_metadata'])

    Example:
        >>> integrator = PatientContextIntegrator(embed_dim=256)
        >>>
        >>> # Create clinical metadata
        >>> metadata = ClinicalMetadata(
        ...     age=65,
        ...     sex=Sex.MALE,
        ...     smoking_status=SmokingStatus.FORMER
        ... )
        >>>
        >>> # Create imaging features
        >>> wsi_features = torch.randn(16, 256)
        >>>
        >>> # Generate multimodal patient representation
        >>> patient_repr = integrator(
        ...     imaging_features={'wsi': wsi_features},
        ...     clinical_metadata=[metadata] * 16
        ... )
        >>> patient_repr.shape
        torch.Size([16, 256])
    """

    def __init__(
        self,
        embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
        modalities: List[str] = None,
    ):
        super().__init__()

        if modalities is None:
            modalities = ["wsi", "clinical_metadata"]

        self.embed_dim = embed_dim
        self.modalities = modalities

        # Clinical metadata encoder
        self.metadata_encoder = ClinicalMetadataEncoder(embed_dim=embed_dim, dropout=dropout)

        # Import MultiModalFusionLayer dynamically to avoid circular imports
        from src.models.fusion import MultiModalFusionLayer

        # Multimodal fusion layer
        self.fusion = MultiModalFusionLayer(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            modalities=modalities,
        )

        logger.info(f"Initialized PatientContextIntegrator with modalities: {modalities}")

    def forward(
        self,
        imaging_features: Dict[str, torch.Tensor],
        clinical_metadata: Optional[List[ClinicalMetadata]] = None,
    ) -> torch.Tensor:
        """
        Generate multimodal patient representations.

        Args:
            imaging_features: Dict mapping modality names to feature tensors
                             [batch_size, embed_dim]. Can include 'wsi', 'genomic', 'clinical_text'
            clinical_metadata: Optional list of ClinicalMetadata instances (one per sample)
                              If None or empty, clinical metadata modality is masked out

        Returns:
            Fused patient representation [batch_size, embed_dim]

        Raises:
            ValueError: If batch sizes don't match or no modalities are available
        """
        # Get batch size from first available imaging feature
        try:
            first_feature = next(iter(imaging_features.values()))
            batch_size = first_feature.shape[0]
            device = first_feature.device
        except StopIteration:
            raise ValueError("At least one imaging feature must be provided")

        # Validate batch sizes
        for modality, features in imaging_features.items():
            if features.shape[0] != batch_size:
                raise ValueError(
                    f"Batch size mismatch: {modality} has {features.shape[0]} samples, "
                    f"expected {batch_size}"
                )

        # Encode clinical metadata if provided
        embeddings = imaging_features.copy()
        modality_masks = {}

        # Initialize masks for all modalities in the fusion layer
        for mod in self.modalities:
            if mod == "clinical_metadata":
                # Handle clinical metadata separately
                continue
            elif mod in imaging_features:
                # Imaging modality is present
                modality_masks[mod] = torch.ones(batch_size, dtype=torch.bool, device=device)
            else:
                # Imaging modality is missing
                modality_masks[mod] = torch.zeros(batch_size, dtype=torch.bool, device=device)
                embeddings[mod] = None

        if clinical_metadata is not None and len(clinical_metadata) > 0:
            if len(clinical_metadata) != batch_size:
                raise ValueError(
                    f"Clinical metadata batch size ({len(clinical_metadata)}) "
                    f"doesn't match imaging features ({batch_size})"
                )

            # Encode clinical metadata
            metadata_embeddings, metadata_mask = self.metadata_encoder(
                clinical_metadata, device=device
            )
            embeddings["clinical_metadata"] = metadata_embeddings
            modality_masks["clinical_metadata"] = metadata_mask
        else:
            # No clinical metadata provided - mask it out
            embeddings["clinical_metadata"] = None
            modality_masks["clinical_metadata"] = torch.zeros(
                batch_size, dtype=torch.bool, device=device
            )

        # Fuse all modalities
        fused = self.fusion(embeddings, modality_masks=modality_masks)

        return fused

    def get_metadata_availability(
        self, clinical_metadata: List[ClinicalMetadata]
    ) -> Dict[str, float]:
        """
        Get availability statistics for clinical metadata fields.

        Args:
            clinical_metadata: List of ClinicalMetadata instances

        Returns:
            Dictionary mapping field names to availability percentages (0-100)
        """
        if not clinical_metadata:
            return {}

        total = len(clinical_metadata)
        availability = {
            "smoking_status": 0,
            "alcohol_consumption": 0,
            "exercise_frequency": 0,
            "sex": 0,
            "age": 0,
            "medications": 0,
            "family_history": 0,
        }

        for metadata in clinical_metadata:
            if metadata.smoking_status != SmokingStatus.UNKNOWN:
                availability["smoking_status"] += 1
            if metadata.alcohol_consumption != AlcoholConsumption.UNKNOWN:
                availability["alcohol_consumption"] += 1
            if metadata.exercise_frequency != ExerciseFrequency.UNKNOWN:
                availability["exercise_frequency"] += 1
            if metadata.sex != Sex.UNKNOWN:
                availability["sex"] += 1
            if metadata.age is not None:
                availability["age"] += 1
            if metadata.medications:
                availability["medications"] += 1
            if metadata.family_history:
                availability["family_history"] += 1

        # Convert to percentages
        for field_name in availability:
            availability[field_name] = (availability[field_name] / total) * 100

        return availability


class ClinicalMetadataEncoder(nn.Module):
    """
    Encoder for clinical metadata into vector representations.

    Converts structured clinical metadata (categorical and numerical fields)
    into fixed-dimensional embeddings suitable for multimodal fusion.

    Uses learned embeddings for categorical fields and linear projections
    for numerical fields, combined through a feedforward network.

    Args:
        embed_dim: Output embedding dimension (default: 256)
        dropout: Dropout rate (default: 0.1)
    """

    def __init__(self, embed_dim: int = 256, dropout: float = 0.1):
        super().__init__()

        self.embed_dim = embed_dim

        # Embedding dimensions for categorical fields
        categorical_embed_dim = 16

        # Learned embeddings for categorical fields
        self.smoking_embedding = nn.Embedding(len(SmokingStatus), categorical_embed_dim)
        self.alcohol_embedding = nn.Embedding(len(AlcoholConsumption), categorical_embed_dim)
        self.exercise_embedding = nn.Embedding(len(ExerciseFrequency), categorical_embed_dim)
        self.sex_embedding = nn.Embedding(len(Sex), categorical_embed_dim)

        # Age normalization and projection
        self.age_projection = nn.Sequential(
            nn.Linear(1, categorical_embed_dim),
            nn.LayerNorm(categorical_embed_dim),
            nn.GELU(),
        )

        # Medication and family history encoding (count-based for simplicity)
        self.medication_projection = nn.Sequential(
            nn.Linear(1, categorical_embed_dim),
            nn.LayerNorm(categorical_embed_dim),
            nn.GELU(),
        )
        self.family_history_projection = nn.Sequential(
            nn.Linear(1, categorical_embed_dim),
            nn.LayerNorm(categorical_embed_dim),
            nn.GELU(),
        )

        # Combine all features
        total_features = categorical_embed_dim * 7  # 7 feature groups
        self.feature_combiner = nn.Sequential(
            nn.Linear(total_features, embed_dim * 2),
            nn.LayerNorm(embed_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LayerNorm(embed_dim),
        )

        # Enum to index mappings
        self.smoking_to_idx = {status: idx for idx, status in enumerate(SmokingStatus)}
        self.alcohol_to_idx = {level: idx for idx, level in enumerate(AlcoholConsumption)}
        self.exercise_to_idx = {freq: idx for idx, freq in enumerate(ExerciseFrequency)}
        self.sex_to_idx = {sex: idx for idx, sex in enumerate(Sex)}

    def forward(
        self, clinical_metadata: List[ClinicalMetadata], device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Encode clinical metadata into embeddings.

        Args:
            clinical_metadata: List of ClinicalMetadata instances
            device: Device to create tensors on

        Returns:
            Tuple of:
                - Embeddings [batch_size, embed_dim]
                - Mask [batch_size] indicating which samples have any metadata (True = has data)
        """
        # Convert metadata to tensors
        smoking_indices = torch.tensor(
            [self.smoking_to_idx[m.smoking_status] for m in clinical_metadata],
            dtype=torch.long,
            device=device,
        )
        alcohol_indices = torch.tensor(
            [self.alcohol_to_idx[m.alcohol_consumption] for m in clinical_metadata],
            dtype=torch.long,
            device=device,
        )
        exercise_indices = torch.tensor(
            [self.exercise_to_idx[m.exercise_frequency] for m in clinical_metadata],
            dtype=torch.long,
            device=device,
        )
        sex_indices = torch.tensor(
            [self.sex_to_idx[m.sex] for m in clinical_metadata],
            dtype=torch.long,
            device=device,
        )

        # Age (normalized to 0-1 range, assuming max age 100)
        ages = torch.tensor(
            [m.age / 100.0 if m.age is not None else 0.0 for m in clinical_metadata],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(-1)

        # Medication count (normalized)
        med_counts = torch.tensor(
            [min(len(m.medications) / 10.0, 1.0) for m in clinical_metadata],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(-1)

        # Family history count (normalized)
        fh_counts = torch.tensor(
            [min(len(m.family_history) / 5.0, 1.0) for m in clinical_metadata],
            dtype=torch.float32,
            device=device,
        ).unsqueeze(-1)

        # Embed categorical features
        smoking_emb = self.smoking_embedding(smoking_indices)
        alcohol_emb = self.alcohol_embedding(alcohol_indices)
        exercise_emb = self.exercise_embedding(exercise_indices)
        sex_emb = self.sex_embedding(sex_indices)

        # Project numerical features
        age_emb = self.age_projection(ages)
        med_emb = self.medication_projection(med_counts)
        fh_emb = self.family_history_projection(fh_counts)

        # Concatenate all features
        all_features = torch.cat(
            [smoking_emb, alcohol_emb, exercise_emb, sex_emb, age_emb, med_emb, fh_emb],
            dim=-1,
        )

        # Combine into final embedding
        embeddings = self.feature_combiner(all_features)

        # Create mask: True if metadata has any non-unknown/non-None values
        # False only if all fields are unknown/None (completely empty)
        mask = torch.tensor(
            [len(m.get_missing_fields()) < 5 for m in clinical_metadata],
            dtype=torch.bool,
            device=device,
        )

        return embeddings, mask
