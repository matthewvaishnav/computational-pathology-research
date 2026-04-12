"""
Unit tests for clinical decision threshold system.

Tests threshold configuration, validation, and flagging logic.
"""

import tempfile
from pathlib import Path

import pytest
import torch

from src.clinical.thresholds import ClinicalThresholdSystem, ThresholdConfig


class TestThresholdConfig:
    """Test suite for ThresholdConfig dataclass."""

    def test_initialization_defaults(self):
        """Test ThresholdConfig initialization with defaults."""
        config = ThresholdConfig(disease_id="malignant")

        assert config.disease_id == "malignant"
        assert config.risk_threshold == 0.5
        assert config.confidence_threshold == 0.7
        assert config.anomaly_threshold == 0.6
        assert config.time_horizon_thresholds is None

    def test_initialization_custom_values(self):
        """Test ThresholdConfig with custom values."""
        config = ThresholdConfig(
            disease_id="grade_2",
            risk_threshold=0.3,
            confidence_threshold=0.8,
            anomaly_threshold=0.5,
            time_horizon_thresholds={"1-year": 0.4, "5-year": 0.3},
        )

        assert config.disease_id == "grade_2"
        assert config.risk_threshold == 0.3
        assert config.confidence_threshold == 0.8
        assert config.anomaly_threshold == 0.5
        assert config.time_horizon_thresholds == {"1-year": 0.4, "5-year": 0.3}

    def test_validate_valid_config(self):
        """Test validation of valid configuration."""
        config = ThresholdConfig(
            disease_id="benign",
            risk_threshold=0.5,
            confidence_threshold=0.7,
            anomaly_threshold=0.6,
        )

        # Should not raise
        config.validate()

    def test_validate_invalid_risk_threshold(self):
        """Test validation fails for invalid risk threshold."""
        config = ThresholdConfig(disease_id="test", risk_threshold=1.5)

        with pytest.raises(ValueError, match="risk_threshold"):
            config.validate()

        config = ThresholdConfig(disease_id="test", risk_threshold=-0.1)

        with pytest.raises(ValueError, match="risk_threshold"):
            config.validate()

    def test_validate_invalid_confidence_threshold(self):
        """Test validation fails for invalid confidence threshold."""
        config = ThresholdConfig(disease_id="test", confidence_threshold=1.2)

        with pytest.raises(ValueError, match="confidence_threshold"):
            config.validate()

    def test_validate_invalid_anomaly_threshold(self):
        """Test validation fails for invalid anomaly threshold."""
        config = ThresholdConfig(disease_id="test", anomaly_threshold=-0.5)

        with pytest.raises(ValueError, match="anomaly_threshold"):
            config.validate()

    def test_validate_invalid_time_horizon_thresholds(self):
        """Test validation fails for invalid time horizon thresholds."""
        config = ThresholdConfig(
            disease_id="test",
            time_horizon_thresholds={"1-year": 0.5, "5-year": 1.5},
        )

        with pytest.raises(ValueError, match="time_horizon_thresholds"):
            config.validate()

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = ThresholdConfig(
            disease_id="malignant",
            risk_threshold=0.3,
            confidence_threshold=0.8,
        )

        config_dict = config.to_dict()

        assert config_dict["disease_id"] == "malignant"
        assert config_dict["risk_threshold"] == 0.3
        assert config_dict["confidence_threshold"] == 0.8

    def test_from_dict(self):
        """Test creation from dictionary."""
        config_dict = {
            "disease_id": "grade_1",
            "risk_threshold": 0.4,
            "confidence_threshold": 0.75,
            "anomaly_threshold": 0.55,
        }

        config = ThresholdConfig.from_dict(config_dict)

        assert config.disease_id == "grade_1"
        assert config.risk_threshold == 0.4
        assert config.confidence_threshold == 0.75
        assert config.anomaly_threshold == 0.55


class TestClinicalThresholdSystem:
    """Test suite for ClinicalThresholdSystem class."""

    def test_initialization_defaults(self):
        """Test initialization with default values."""
        system = ClinicalThresholdSystem()

        assert system.default_risk_threshold == 0.5
        assert system.default_confidence_threshold == 0.7
        assert system.default_anomaly_threshold == 0.6
        assert len(system.thresholds) == 0

    def test_initialization_custom_defaults(self):
        """Test initialization with custom default values."""
        system = ClinicalThresholdSystem(
            default_risk_threshold=0.4,
            default_confidence_threshold=0.8,
            default_anomaly_threshold=0.55,
        )

        assert system.default_risk_threshold == 0.4
        assert system.default_confidence_threshold == 0.8
        assert system.default_anomaly_threshold == 0.55

    def test_add_threshold(self):
        """Test adding threshold configuration."""
        system = ClinicalThresholdSystem()

        config = ThresholdConfig(disease_id="malignant", risk_threshold=0.3)
        system.add_threshold(config)

        assert "malignant" in system.thresholds
        assert system.thresholds["malignant"].risk_threshold == 0.3

    def test_add_threshold_invalid(self):
        """Test adding invalid threshold configuration."""
        system = ClinicalThresholdSystem()

        config = ThresholdConfig(disease_id="test", risk_threshold=1.5)

        with pytest.raises(ValueError):
            system.add_threshold(config)

    def test_get_threshold_configured(self):
        """Test getting configured threshold."""
        system = ClinicalThresholdSystem()

        config = ThresholdConfig(disease_id="malignant", risk_threshold=0.3)
        system.add_threshold(config)

        retrieved = system.get_threshold("malignant")

        assert retrieved.disease_id == "malignant"
        assert retrieved.risk_threshold == 0.3

    def test_get_threshold_default(self):
        """Test getting default threshold for unconfigured disease."""
        system = ClinicalThresholdSystem(default_risk_threshold=0.4)

        threshold = system.get_threshold("unknown_disease")

        assert threshold.disease_id == "unknown_disease"
        assert threshold.risk_threshold == 0.4

    def test_evaluate_risk_scores_2d(self):
        """Test evaluating 2D risk scores."""
        system = ClinicalThresholdSystem()

        # Add disease-specific thresholds
        system.add_threshold(ThresholdConfig(disease_id="benign", risk_threshold=0.7))
        system.add_threshold(ThresholdConfig(disease_id="malignant", risk_threshold=0.3))

        # Create risk scores [batch_size=3, num_diseases=2]
        risk_scores = torch.tensor(
            [
                [0.2, 0.5],  # malignant exceeds threshold (0.5 > 0.3)
                [0.5, 0.2],  # neither exceeds
                [0.8, 0.4],  # benign exceeds threshold (0.8 > 0.7)
            ]
        )

        disease_ids = ["benign", "malignant"]
        flags = system.evaluate_risk_scores(risk_scores, disease_ids)

        assert flags.shape == (3,)
        assert flags[0].item() is True  # malignant exceeds
        assert flags[1].item() is False  # neither exceeds
        assert flags[2].item() is True  # benign exceeds

    def test_evaluate_risk_scores_3d(self):
        """Test evaluating 3D risk scores with time horizons."""
        system = ClinicalThresholdSystem()

        system.add_threshold(ThresholdConfig(disease_id="disease_1", risk_threshold=0.5))
        system.add_threshold(ThresholdConfig(disease_id="disease_2", risk_threshold=0.5))

        # Create risk scores [batch_size=2, num_diseases=2, num_time_horizons=3]
        risk_scores = torch.tensor(
            [
                [[0.3, 0.4, 0.6], [0.2, 0.3, 0.4]],  # disease_1 exceeds at horizon 2
                [[0.4, 0.4, 0.4], [0.4, 0.4, 0.4]],  # neither exceeds
            ]
        )

        disease_ids = ["disease_1", "disease_2"]
        flags = system.evaluate_risk_scores(risk_scores, disease_ids)

        assert flags.shape == (2,)
        assert flags[0].item() is True  # exceeds at one time horizon
        assert flags[1].item() is False  # never exceeds

    def test_evaluate_risk_scores_specific_time_horizon(self):
        """Test evaluating risk scores for specific time horizon."""
        system = ClinicalThresholdSystem()

        system.add_threshold(ThresholdConfig(disease_id="disease_1", risk_threshold=0.5))

        # Create risk scores [batch_size=2, num_diseases=1, num_time_horizons=3]
        risk_scores = torch.tensor(
            [
                [[0.3, 0.6, 0.4]],  # exceeds at horizon 1
                [[0.4, 0.4, 0.4]],  # never exceeds
            ]
        )

        disease_ids = ["disease_1"]

        # Evaluate only time horizon 1
        flags = system.evaluate_risk_scores(risk_scores, disease_ids, time_horizon_idx=1)

        assert flags[0].item() is True
        assert flags[1].item() is False

    def test_evaluate_anomaly_scores(self):
        """Test evaluating anomaly scores."""
        system = ClinicalThresholdSystem()

        system.add_threshold(ThresholdConfig(disease_id="disease_1", anomaly_threshold=0.6))
        system.add_threshold(ThresholdConfig(disease_id="disease_2", anomaly_threshold=0.7))

        # Create anomaly scores [batch_size=3, num_diseases=2]
        anomaly_scores = torch.tensor(
            [
                [0.5, 0.8],  # disease_2 exceeds (0.8 > 0.7)
                [0.7, 0.6],  # disease_1 exceeds (0.7 > 0.6)
                [0.5, 0.5],  # neither exceeds
            ]
        )

        disease_ids = ["disease_1", "disease_2"]
        flags = system.evaluate_anomaly_scores(anomaly_scores, disease_ids)

        assert flags.shape == (3,)
        assert flags[0].item() is True
        assert flags[1].item() is True
        assert flags[2].item() is False

    def test_evaluate_confidence(self):
        """Test evaluating confidence scores."""
        system = ClinicalThresholdSystem()

        system.add_threshold(ThresholdConfig(disease_id="disease_1", confidence_threshold=0.7))
        system.add_threshold(ThresholdConfig(disease_id="disease_2", confidence_threshold=0.8))

        # Create confidence scores [batch_size=3]
        confidence_scores = torch.tensor([0.6, 0.75, 0.85])

        primary_disease_ids = ["disease_1", "disease_2", "disease_1"]
        flags = system.evaluate_confidence(confidence_scores, primary_disease_ids)

        assert flags.shape == (3,)
        assert flags[0].item() is True  # 0.6 < 0.7
        assert flags[1].item() is True  # 0.75 < 0.8
        assert flags[2].item() is False  # 0.85 >= 0.7

    def test_get_flagged_details(self):
        """Test getting comprehensive flagging details."""
        system = ClinicalThresholdSystem()

        system.add_threshold(ThresholdConfig(disease_id="disease_1", risk_threshold=0.5))

        risk_scores = torch.tensor([[0.6, 0.4], [0.3, 0.3]])
        anomaly_scores = torch.tensor([[0.7, 0.5], [0.5, 0.5]])
        confidence_scores = torch.tensor([0.6, 0.8])
        disease_ids = ["disease_1", "disease_2"]
        primary_disease_ids = ["disease_1", "disease_2"]

        details = system.get_flagged_details(
            risk_scores,
            anomaly_scores,
            confidence_scores,
            disease_ids,
            primary_disease_ids,
        )

        assert "risk_flags" in details
        assert "anomaly_flags" in details
        assert "confidence_flags" in details
        assert "any_flag" in details
        assert "all_flags" in details

        # Check shapes
        assert details["risk_flags"].shape == (2,)
        assert details["any_flag"].shape == (2,)

    def test_load_from_dict(self):
        """Test loading configuration from dictionary."""
        config_dict = {
            "default_risk_threshold": 0.4,
            "default_confidence_threshold": 0.75,
            "thresholds": [
                {
                    "disease_id": "malignant",
                    "risk_threshold": 0.3,
                    "confidence_threshold": 0.8,
                },
                {
                    "disease_id": "benign",
                    "risk_threshold": 0.6,
                },
            ],
        }

        system = ClinicalThresholdSystem(config_dict=config_dict)

        assert system.default_risk_threshold == 0.4
        assert system.default_confidence_threshold == 0.75
        assert len(system.thresholds) == 2
        assert "malignant" in system.thresholds
        assert system.thresholds["malignant"].risk_threshold == 0.3

    def test_save_and_load_yaml(self):
        """Test saving and loading configuration from YAML file."""
        system = ClinicalThresholdSystem(default_risk_threshold=0.4)

        system.add_threshold(ThresholdConfig(disease_id="malignant", risk_threshold=0.3))
        system.add_threshold(ThresholdConfig(disease_id="benign", risk_threshold=0.6))

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "thresholds.yaml"

            # Save
            system.save_to_file(config_path, format="yaml")

            # Load
            loaded_system = ClinicalThresholdSystem(config_path=config_path)

            assert loaded_system.default_risk_threshold == 0.4
            assert len(loaded_system.thresholds) == 2
            assert "malignant" in loaded_system.thresholds
            assert loaded_system.thresholds["malignant"].risk_threshold == 0.3

    def test_save_and_load_json(self):
        """Test saving and loading configuration from JSON file."""
        system = ClinicalThresholdSystem(default_confidence_threshold=0.8)

        system.add_threshold(ThresholdConfig(disease_id="grade_1", confidence_threshold=0.85))

        with tempfile.TemporaryDirectory() as tmpdir:
            config_path = Path(tmpdir) / "thresholds.json"

            # Save
            system.save_to_file(config_path, format="json")

            # Load
            loaded_system = ClinicalThresholdSystem(config_path=config_path)

            assert loaded_system.default_confidence_threshold == 0.8
            assert len(loaded_system.thresholds) == 1
            assert "grade_1" in loaded_system.thresholds

    def test_repr(self):
        """Test string representation."""
        system = ClinicalThresholdSystem(default_risk_threshold=0.4)

        system.add_threshold(ThresholdConfig(disease_id="malignant", risk_threshold=0.3))

        repr_str = repr(system)

        assert "ClinicalThresholdSystem" in repr_str
        assert "num_disease_thresholds=1" in repr_str
        assert "default_risk_threshold=0.4" in repr_str


class TestThresholdSystemIntegration:
    """Integration tests for threshold system."""

    def test_end_to_end_threshold_evaluation(self):
        """Test complete threshold evaluation workflow."""
        # Create threshold system
        system = ClinicalThresholdSystem()

        system.add_threshold(
            ThresholdConfig(
                disease_id="benign",
                risk_threshold=0.7,
                confidence_threshold=0.8,
                anomaly_threshold=0.6,
            )
        )
        system.add_threshold(
            ThresholdConfig(
                disease_id="malignant",
                risk_threshold=0.3,
                confidence_threshold=0.75,
                anomaly_threshold=0.5,
            )
        )

        # Create sample data
        batch_size = 5
        risk_scores = torch.rand(batch_size, 2)  # [batch, diseases]
        anomaly_scores = torch.rand(batch_size, 2)
        confidence_scores = torch.rand(batch_size)

        disease_ids = ["benign", "malignant"]
        primary_disease_ids = ["benign", "malignant", "benign", "malignant", "benign"]

        # Evaluate all thresholds
        details = system.get_flagged_details(
            risk_scores,
            anomaly_scores,
            confidence_scores,
            disease_ids,
            primary_disease_ids,
        )

        # Verify all flags are boolean tensors
        assert details["risk_flags"].dtype == torch.bool
        assert details["anomaly_flags"].dtype == torch.bool
        assert details["confidence_flags"].dtype == torch.bool
        assert details["any_flag"].dtype == torch.bool
        assert details["all_flags"].dtype == torch.bool

        # Verify logical relationships
        # any_flag should be True if any individual flag is True
        for i in range(batch_size):
            if details["any_flag"][i]:
                assert (
                    details["risk_flags"][i]
                    or details["anomaly_flags"][i]
                    or details["confidence_flags"][i]
                )
