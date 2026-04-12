"""
Unit tests for patient context integration.

Tests clinical metadata data structures, validation, serialization,
and multimodal patient representation generation.
"""

import tempfile
from pathlib import Path

import pytest
import torch

from src.clinical.patient_context import (
    AlcoholConsumption,
    ClinicalMetadata,
    ClinicalMetadataEncoder,
    ExerciseFrequency,
    PatientContextIntegrator,
    Sex,
    SmokingStatus,
)


class TestClinicalMetadata:
    """Tests for ClinicalMetadata dataclass."""

    def test_default_initialization(self):
        """Test that ClinicalMetadata can be created with defaults."""
        metadata = ClinicalMetadata()

        assert metadata.smoking_status == SmokingStatus.UNKNOWN
        assert metadata.alcohol_consumption == AlcoholConsumption.UNKNOWN
        assert metadata.exercise_frequency == ExerciseFrequency.UNKNOWN
        assert metadata.sex == Sex.UNKNOWN
        assert metadata.age is None
        assert metadata.medications == []
        assert metadata.family_history == []
        assert metadata.metadata == {}

    def test_full_initialization(self):
        """Test ClinicalMetadata with all fields populated."""
        metadata = ClinicalMetadata(
            smoking_status=SmokingStatus.FORMER,
            alcohol_consumption=AlcoholConsumption.LIGHT,
            medications=["metformin", "lisinopril"],
            exercise_frequency=ExerciseFrequency.MODERATE,
            age=65,
            sex=Sex.MALE,
            family_history=["diabetes", "hypertension"],
            metadata={"custom_field": "value"},
        )

        assert metadata.smoking_status == SmokingStatus.FORMER
        assert metadata.alcohol_consumption == AlcoholConsumption.LIGHT
        assert metadata.medications == ["metformin", "lisinopril"]
        assert metadata.exercise_frequency == ExerciseFrequency.MODERATE
        assert metadata.age == 65
        assert metadata.sex == Sex.MALE
        assert metadata.family_history == ["diabetes", "hypertension"]
        assert metadata.metadata == {"custom_field": "value"}

    def test_validation_success(self):
        """Test that valid metadata passes validation."""
        metadata = ClinicalMetadata(
            smoking_status=SmokingStatus.NEVER,
            alcohol_consumption=AlcoholConsumption.NONE,
            age=45,
            sex=Sex.FEMALE,
        )

        # Should not raise
        metadata.validate()

    def test_validation_invalid_age(self):
        """Test that invalid age raises validation error."""
        metadata = ClinicalMetadata(age=200)

        with pytest.raises(ValueError, match="Age must be between 0 and 150"):
            metadata.validate()

    def test_validation_negative_age(self):
        """Test that negative age raises validation error."""
        metadata = ClinicalMetadata(age=-5)

        with pytest.raises(ValueError, match="Age must be between 0 and 150"):
            metadata.validate()

    def test_validation_invalid_medications(self):
        """Test that invalid medications type raises validation error."""
        metadata = ClinicalMetadata(medications="not_a_list")

        with pytest.raises(ValueError, match="Medications must be a list"):
            metadata.validate()

    def test_validation_invalid_family_history(self):
        """Test that invalid family history type raises validation error."""
        metadata = ClinicalMetadata(family_history="not_a_list")

        with pytest.raises(ValueError, match="Family history must be a list"):
            metadata.validate()

    def test_to_dict(self):
        """Test conversion to dictionary."""
        metadata = ClinicalMetadata(
            smoking_status=SmokingStatus.CURRENT,
            age=50,
            sex=Sex.MALE,
            medications=["aspirin"],
        )

        data = metadata.to_dict()

        assert data["smoking_status"] == "current"
        assert data["age"] == 50
        assert data["sex"] == "male"
        assert data["medications"] == ["aspirin"]
        assert isinstance(data, dict)

    def test_from_dict(self):
        """Test creation from dictionary."""
        data = {
            "smoking_status": "former",
            "alcohol_consumption": "moderate",
            "exercise_frequency": "active",
            "sex": "female",
            "age": 60,
            "medications": ["metformin"],
            "family_history": ["diabetes"],
            "metadata": {},
        }

        metadata = ClinicalMetadata.from_dict(data)

        assert metadata.smoking_status == SmokingStatus.FORMER
        assert metadata.alcohol_consumption == AlcoholConsumption.MODERATE
        assert metadata.exercise_frequency == ExerciseFrequency.ACTIVE
        assert metadata.sex == Sex.FEMALE
        assert metadata.age == 60
        assert metadata.medications == ["metformin"]
        assert metadata.family_history == ["diabetes"]

    def test_json_serialization(self):
        """Test JSON serialization and deserialization."""
        metadata = ClinicalMetadata(
            smoking_status=SmokingStatus.NEVER,
            age=40,
            sex=Sex.FEMALE,
            medications=["vitamin_d"],
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            json_path = Path(tmpdir) / "metadata.json"

            # Save to JSON
            metadata.to_json(json_path)
            assert json_path.exists()

            # Load from JSON
            loaded = ClinicalMetadata.from_json(json_path)

            assert loaded.smoking_status == metadata.smoking_status
            assert loaded.age == metadata.age
            assert loaded.sex == metadata.sex
            assert loaded.medications == metadata.medications

    def test_is_complete(self):
        """Test completeness checking."""
        # Complete metadata
        complete = ClinicalMetadata(
            smoking_status=SmokingStatus.NEVER,
            alcohol_consumption=AlcoholConsumption.NONE,
            exercise_frequency=ExerciseFrequency.MODERATE,
            sex=Sex.MALE,
            age=50,
        )
        assert complete.is_complete()

        # Incomplete metadata (missing age)
        incomplete = ClinicalMetadata(
            smoking_status=SmokingStatus.NEVER,
            alcohol_consumption=AlcoholConsumption.NONE,
            exercise_frequency=ExerciseFrequency.MODERATE,
            sex=Sex.MALE,
        )
        assert not incomplete.is_complete()

    def test_get_missing_fields(self):
        """Test missing fields detection."""
        metadata = ClinicalMetadata(
            smoking_status=SmokingStatus.FORMER,
            age=55,
            # Other fields are unknown/None
        )

        missing = metadata.get_missing_fields()

        assert "alcohol_consumption" in missing
        assert "exercise_frequency" in missing
        assert "sex" in missing
        assert "smoking_status" not in missing
        assert "age" not in missing


class TestClinicalMetadataEncoder:
    """Tests for ClinicalMetadataEncoder."""

    def test_encoder_initialization(self):
        """Test encoder can be initialized."""
        encoder = ClinicalMetadataEncoder(embed_dim=256)
        assert encoder.embed_dim == 256

    def test_encode_single_metadata(self):
        """Test encoding single clinical metadata."""
        encoder = ClinicalMetadataEncoder(embed_dim=256)
        metadata = ClinicalMetadata(
            smoking_status=SmokingStatus.FORMER,
            age=65,
            sex=Sex.MALE,
        )

        embeddings, mask = encoder([metadata], device=torch.device("cpu"))

        assert embeddings.shape == (1, 256)
        assert mask.shape == (1,)
        assert mask.dtype == torch.bool

    def test_encode_batch_metadata(self):
        """Test encoding batch of clinical metadata."""
        encoder = ClinicalMetadataEncoder(embed_dim=256)
        metadata_list = [
            ClinicalMetadata(smoking_status=SmokingStatus.NEVER, age=45, sex=Sex.FEMALE),
            ClinicalMetadata(smoking_status=SmokingStatus.CURRENT, age=55, sex=Sex.MALE),
            ClinicalMetadata(smoking_status=SmokingStatus.FORMER, age=60, sex=Sex.FEMALE),
        ]

        embeddings, mask = encoder(metadata_list, device=torch.device("cpu"))

        assert embeddings.shape == (3, 256)
        assert mask.shape == (3,)
        assert mask.all()  # All should have some metadata

    def test_encode_incomplete_metadata(self):
        """Test encoding metadata with missing fields."""
        encoder = ClinicalMetadataEncoder(embed_dim=256)
        metadata = ClinicalMetadata()  # All defaults (unknown/None)

        embeddings, mask = encoder([metadata], device=torch.device("cpu"))

        assert embeddings.shape == (1, 256)
        assert mask.shape == (1,)
        # Mask should be False for completely empty metadata
        assert not mask[0]


class TestPatientContextIntegrator:
    """Tests for PatientContextIntegrator."""

    def test_integrator_initialization(self):
        """Test integrator can be initialized."""
        integrator = PatientContextIntegrator(embed_dim=256)
        assert integrator.embed_dim == 256
        assert "clinical_metadata" in integrator.modalities

    def test_integrate_wsi_with_metadata(self):
        """Test integration of WSI features with clinical metadata."""
        integrator = PatientContextIntegrator(embed_dim=256)

        # Create WSI features
        wsi_features = torch.randn(4, 256)

        # Create clinical metadata
        metadata_list = [
            ClinicalMetadata(smoking_status=SmokingStatus.NEVER, age=45, sex=Sex.FEMALE),
            ClinicalMetadata(smoking_status=SmokingStatus.CURRENT, age=55, sex=Sex.MALE),
            ClinicalMetadata(smoking_status=SmokingStatus.FORMER, age=60, sex=Sex.FEMALE),
            ClinicalMetadata(smoking_status=SmokingStatus.NEVER, age=50, sex=Sex.MALE),
        ]

        # Generate patient representations
        patient_repr = integrator(
            imaging_features={"wsi": wsi_features},
            clinical_metadata=metadata_list,
        )

        assert patient_repr.shape == (4, 256)
        assert not torch.isnan(patient_repr).any()

    def test_integrate_wsi_without_metadata(self):
        """Test integration with only WSI features (no metadata)."""
        integrator = PatientContextIntegrator(embed_dim=256)

        # Create WSI features
        wsi_features = torch.randn(4, 256)

        # Generate patient representations without metadata
        patient_repr = integrator(
            imaging_features={"wsi": wsi_features},
            clinical_metadata=None,
        )

        assert patient_repr.shape == (4, 256)
        assert not torch.isnan(patient_repr).any()

    def test_integrate_multiple_modalities(self):
        """Test integration with multiple imaging modalities and metadata."""
        integrator = PatientContextIntegrator(
            embed_dim=256,
            modalities=["wsi", "genomic", "clinical_metadata"],
        )

        # Create imaging features
        wsi_features = torch.randn(2, 256)
        genomic_features = torch.randn(2, 256)

        # Create clinical metadata
        metadata_list = [
            ClinicalMetadata(smoking_status=SmokingStatus.NEVER, age=45),
            ClinicalMetadata(smoking_status=SmokingStatus.CURRENT, age=55),
        ]

        # Generate patient representations
        patient_repr = integrator(
            imaging_features={"wsi": wsi_features, "genomic": genomic_features},
            clinical_metadata=metadata_list,
        )

        assert patient_repr.shape == (2, 256)
        assert not torch.isnan(patient_repr).any()

    def test_integrate_missing_modality(self):
        """Test integration with missing imaging modality."""
        integrator = PatientContextIntegrator(
            embed_dim=256,
            modalities=["wsi", "genomic", "clinical_metadata"],
        )

        # Only provide WSI features (genomic missing)
        wsi_features = torch.randn(2, 256)

        metadata_list = [
            ClinicalMetadata(age=45),
            ClinicalMetadata(age=55),
        ]

        # Should handle missing genomic modality gracefully
        patient_repr = integrator(
            imaging_features={"wsi": wsi_features},
            clinical_metadata=metadata_list,
        )

        assert patient_repr.shape == (2, 256)
        assert not torch.isnan(patient_repr).any()

    def test_batch_size_mismatch_error(self):
        """Test that batch size mismatch raises error."""
        integrator = PatientContextIntegrator(embed_dim=256)

        wsi_features = torch.randn(4, 256)
        metadata_list = [ClinicalMetadata(age=45)]  # Only 1 metadata for 4 samples

        with pytest.raises(ValueError, match="batch size"):
            integrator(
                imaging_features={"wsi": wsi_features},
                clinical_metadata=metadata_list,
            )

    def test_no_imaging_features_error(self):
        """Test that missing imaging features raises error."""
        integrator = PatientContextIntegrator(embed_dim=256)

        with pytest.raises(ValueError, match="At least one imaging feature"):
            integrator(
                imaging_features={},
                clinical_metadata=[ClinicalMetadata()],
            )

    def test_get_metadata_availability(self):
        """Test metadata availability statistics."""
        integrator = PatientContextIntegrator(embed_dim=256)

        metadata_list = [
            ClinicalMetadata(
                smoking_status=SmokingStatus.NEVER,
                age=45,
                sex=Sex.FEMALE,
                medications=["aspirin"],
            ),
            ClinicalMetadata(
                smoking_status=SmokingStatus.CURRENT,
                age=55,
                # sex unknown, no medications
            ),
            ClinicalMetadata(
                # All unknown/None
            ),
        ]

        availability = integrator.get_metadata_availability(metadata_list)

        assert availability["smoking_status"] == pytest.approx(66.67, rel=0.1)
        assert availability["age"] == pytest.approx(66.67, rel=0.1)
        assert availability["sex"] == pytest.approx(33.33, rel=0.1)
        assert availability["medications"] == pytest.approx(33.33, rel=0.1)


class TestMultimodalFusionWithMissingModalities:
    """Tests for multimodal fusion with missing modalities."""

    def test_fusion_with_all_modalities(self):
        """Test fusion when all modalities are present."""
        integrator = PatientContextIntegrator(
            embed_dim=256,
            modalities=["wsi", "genomic", "clinical_metadata"],
        )

        wsi_features = torch.randn(2, 256)
        genomic_features = torch.randn(2, 256)
        metadata_list = [
            ClinicalMetadata(age=45, sex=Sex.FEMALE),
            ClinicalMetadata(age=55, sex=Sex.MALE),
        ]

        patient_repr = integrator(
            imaging_features={"wsi": wsi_features, "genomic": genomic_features},
            clinical_metadata=metadata_list,
        )

        assert patient_repr.shape == (2, 256)
        assert not torch.isnan(patient_repr).any()

    def test_fusion_with_only_wsi(self):
        """Test fusion with only WSI modality."""
        integrator = PatientContextIntegrator(
            embed_dim=256,
            modalities=["wsi", "genomic", "clinical_metadata"],
        )

        wsi_features = torch.randn(2, 256)

        patient_repr = integrator(
            imaging_features={"wsi": wsi_features},
            clinical_metadata=None,
        )

        assert patient_repr.shape == (2, 256)
        assert not torch.isnan(patient_repr).any()

    def test_fusion_with_incomplete_metadata(self):
        """Test fusion with incomplete clinical metadata."""
        integrator = PatientContextIntegrator(embed_dim=256)

        wsi_features = torch.randn(3, 256)
        metadata_list = [
            ClinicalMetadata(age=45),  # Only age
            ClinicalMetadata(smoking_status=SmokingStatus.NEVER),  # Only smoking
            ClinicalMetadata(),  # All unknown
        ]

        patient_repr = integrator(
            imaging_features={"wsi": wsi_features},
            clinical_metadata=metadata_list,
        )

        assert patient_repr.shape == (3, 256)
        assert not torch.isnan(patient_repr).any()
