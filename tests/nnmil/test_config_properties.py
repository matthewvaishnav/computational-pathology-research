"""
Property-based tests for nnMIL configuration management.

Tests universal correctness properties for nnMILConfig including
configuration loading, validation, inheritance, logging, and persistence.
"""

import logging
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
import yaml

from hypothesis import given, settings
from hypothesis import strategies as st
from src.core.config.nnmil_config import nnMILConfig


class TestnnMILConfigProperties:
    """Property tests for nnMILConfig."""

    @given(
        feature_dim=st.integers(min_value=1, max_value=4096),
        hidden_dim=st.integers(min_value=1, max_value=1024),
        num_classes=st.integers(min_value=1, max_value=100),
        dropout=st.floats(min_value=0.0, max_value=1.0),
        batch_size=st.integers(min_value=1, max_value=128),
        learning_rate=st.floats(min_value=1e-6, max_value=1e-1),
        bag_length=st.integers(min_value=1, max_value=10000),
    )
    @settings(max_examples=50)
    def test_property_28_configuration_loading(
        self, feature_dim, hidden_dim, num_classes, dropout, batch_size, learning_rate, bag_length
    ):
        """
        Property 28: Configuration Loading

        Any valid configuration parameters should be loadable and produce
        a valid nnMILConfig instance with correct parameter values.

        Validates: Requirements 9.1
        """
        # Create configuration with valid parameters
        config = nnMILConfig(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=dropout,
            batch_size=batch_size,
            learning_rate=learning_rate,
            bag_length=bag_length,
        )

        # Property: All parameters should be loaded correctly
        assert config.feature_dim == feature_dim
        assert config.hidden_dim == hidden_dim
        assert config.num_classes == num_classes
        assert config.dropout == dropout
        assert config.batch_size == batch_size
        assert config.learning_rate == learning_rate
        assert config.bag_length == bag_length

        # Property: Configuration should be valid (no exceptions during creation)
        assert isinstance(config, nnMILConfig)

    @given(
        feature_dim=st.integers(max_value=0),  # Invalid: non-positive
        hidden_dim=st.integers(max_value=0),  # Invalid: non-positive
        dropout=st.floats(min_value=1.1, max_value=2.0),  # Invalid: > 1.0
        batch_size=st.integers(max_value=0),  # Invalid: non-positive
        learning_rate=st.floats(max_value=0.0),  # Invalid: non-positive
        bag_length=st.integers(max_value=0),  # Invalid: non-positive
    )
    @settings(max_examples=30)
    def test_property_29_configuration_validation(
        self, feature_dim, hidden_dim, dropout, batch_size, learning_rate, bag_length
    ):
        """
        Property 29: Configuration Validation

        Invalid configuration parameters should be rejected with
        appropriate ValueError exceptions.

        Validates: Requirements 9.3
        """
        # Test each invalid parameter individually to isolate validation

        # Test invalid feature_dim
        if feature_dim <= 0:
            with pytest.raises(ValueError, match="feature_dim must be positive"):
                nnMILConfig(feature_dim=feature_dim)

        # Test invalid hidden_dim
        if hidden_dim <= 0:
            with pytest.raises(ValueError, match="hidden_dim must be positive"):
                nnMILConfig(hidden_dim=hidden_dim)

        # Test invalid dropout
        if not (0 <= dropout <= 1):
            with pytest.raises(ValueError, match="dropout must be in"):
                nnMILConfig(dropout=dropout)

        # Test invalid batch_size
        if batch_size <= 0:
            with pytest.raises(ValueError, match="batch_size must be positive"):
                nnMILConfig(batch_size=batch_size)

        # Test invalid learning_rate
        if learning_rate <= 0:
            with pytest.raises(ValueError, match="learning_rate must be positive"):
                nnMILConfig(learning_rate=learning_rate)

        # Test invalid bag_length
        if bag_length <= 0:
            with pytest.raises(ValueError, match="bag_length must be positive"):
                nnMILConfig(bag_length=bag_length)

    @given(
        base_hidden_dim=st.integers(min_value=64, max_value=512),
        base_batch_size=st.integers(min_value=8, max_value=64),
        override_hidden_dim=st.integers(min_value=128, max_value=1024),
        override_learning_rate=st.floats(min_value=1e-5, max_value=1e-2),
    )
    @settings(max_examples=30)
    def test_property_30_configuration_inheritance(
        self, base_hidden_dim, base_batch_size, override_hidden_dim, override_learning_rate
    ):
        """
        Property 30: Configuration Inheritance

        Child configurations should inherit parameters from base configurations
        and correctly override specified parameters.

        Validates: Requirements 9.4
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create base configuration file
            base_config = {
                "hidden_dim": base_hidden_dim,
                "batch_size": base_batch_size,
                "learning_rate": 3e-4,  # Default
                "task_type": "classification",
            }

            base_path = temp_path / "base.yaml"
            with open(base_path, "w") as f:
                yaml.dump(base_config, f)

            # Create child configuration with inheritance
            child_config = {
                "inherit_from": "base.yaml",
                "hidden_dim": override_hidden_dim,  # Override
                "learning_rate": override_learning_rate,  # Override
                # batch_size should be inherited
            }

            child_path = temp_path / "child.yaml"
            with open(child_path, "w") as f:
                yaml.dump(child_config, f)

            # Load child configuration
            config = nnMILConfig.from_yaml(child_path)

            # Property: Overridden parameters should use child values
            assert config.hidden_dim == override_hidden_dim
            assert config.learning_rate == override_learning_rate

            # Property: Non-overridden parameters should use base values
            assert config.batch_size == base_batch_size
            assert config.task_type == "classification"

    @given(
        feature_dim=st.integers(min_value=256, max_value=2048),
        num_classes=st.integers(min_value=2, max_value=10),
        task_type=st.sampled_from(["classification", "regression", "survival"]),
    )
    @settings(max_examples=30)
    def test_property_31_configuration_logging(self, feature_dim, num_classes, task_type):
        """
        Property 31: Configuration Logging

        Configuration logging should capture all important parameters
        without raising exceptions.

        Validates: Requirements 9.5
        """
        config = nnMILConfig(feature_dim=feature_dim, num_classes=num_classes, task_type=task_type)

        # Create a logger to capture output
        logger = logging.getLogger("test_config_logging")
        logger.setLevel(logging.INFO)

        # Property: Logging should not raise exceptions
        try:
            config.log_config(logger)
            logging_successful = True
        except Exception:
            logging_successful = False

        assert logging_successful, "Configuration logging should not raise exceptions"

        # Property: Configuration should have all required methods
        assert hasattr(config, "log_config")
        assert callable(config.log_config)

        # Property: Configuration should provide parameter access methods
        model_params = config.get_model_params()
        training_params = config.get_training_params()
        inference_params = config.get_inference_params()

        assert isinstance(model_params, dict)
        assert isinstance(training_params, dict)
        assert isinstance(inference_params, dict)

        # Property: Model parameters should contain expected keys
        expected_model_keys = {"feature_dim", "hidden_dim", "num_classes", "dropout"}
        assert expected_model_keys.issubset(model_params.keys())

        # Property: Parameter values should match configuration
        assert model_params["feature_dim"] == feature_dim
        assert model_params["num_classes"] == num_classes

    @given(
        hidden_dim=st.integers(min_value=128, max_value=512),
        batch_size=st.integers(min_value=16, max_value=64),
        learning_rate=st.floats(min_value=1e-5, max_value=1e-2),
        task_type=st.sampled_from(["classification", "regression", "survival"]),
    )
    @settings(max_examples=30)
    def test_property_32_configuration_persistence(
        self, hidden_dim, batch_size, learning_rate, task_type
    ):
        """
        Property 32: Configuration Persistence

        Configurations should be saveable to YAML and loadable with
        identical parameter values (round-trip preservation).

        Validates: Requirements 9.6
        """
        # Create original configuration
        original_config = nnMILConfig(
            hidden_dim=hidden_dim,
            batch_size=batch_size,
            learning_rate=learning_rate,
            task_type=task_type,
        )

        with tempfile.TemporaryDirectory() as temp_dir:
            config_path = Path(temp_dir) / "test_config.yaml"

            # Save configuration
            original_config.to_yaml(config_path)

            # Property: File should be created
            assert config_path.exists()

            # Load configuration
            loaded_config = nnMILConfig.from_yaml(config_path)

            # Property: Round-trip preservation - key parameters should match
            assert loaded_config.hidden_dim == original_config.hidden_dim
            assert loaded_config.batch_size == original_config.batch_size
            assert abs(loaded_config.learning_rate - original_config.learning_rate) < 1e-10
            assert loaded_config.task_type == original_config.task_type

            # Property: Configuration should be valid after loading
            assert isinstance(loaded_config, nnMILConfig)

            # Property: to_dict should work for both configurations
            original_dict = original_config.to_dict()
            loaded_dict = loaded_config.to_dict()

            assert isinstance(original_dict, dict)
            assert isinstance(loaded_dict, dict)

            # Property: Key parameters should match in dictionaries
            key_params = ["hidden_dim", "batch_size", "learning_rate", "task_type"]
            for param in key_params:
                assert original_dict[param] == loaded_dict[param]


class TestnnMILConfigDatasetFingerprinting:
    """Property tests for dataset fingerprinting functionality."""

    @given(
        median_patches=st.integers(min_value=50, max_value=5000),
        task_type=st.sampled_from(["classification", "regression", "survival"]),
    )
    @settings(max_examples=20)
    def test_property_33_dataset_fingerprinting(self, median_patches, task_type):
        """
        Property 33: Dataset Fingerprinting

        Dataset fingerprinting should derive reasonable configuration
        parameters based on dataset characteristics.

        Validates: Requirements 9.1, 9.2
        """
        # Mock dataset analysis to return controlled fingerprint
        mock_fingerprint = {
            "dataset_path": "/mock/dataset",
            "task_type": task_type,
            "median_patches": median_patches,
            "iqr_patches": (median_patches // 2, median_patches * 2),
            "num_slides": 1000,
            "class_distribution": [0.6, 0.4] if task_type == "classification" else None,
            "target_range": (0.0, 1.0) if task_type == "regression" else None,
            "event_rate": 0.3 if task_type == "survival" else None,
        }

        with patch.object(
            nnMILConfig, "_extract_dataset_fingerprint", return_value=mock_fingerprint
        ):
            config = nnMILConfig.from_dataset(dataset_path="/mock/dataset", task_type=task_type)

        # Property: bag_length should be derived from median_patches
        expected_bag_length = max(100, min(10000, median_patches // 2))
        assert config.bag_length == expected_bag_length

        # Property: task_type should match input
        assert config.task_type == task_type

        # Property: sampler_type should match task_type
        if task_type == "classification":
            assert config.sampler_type == "balanced"
        elif task_type == "regression":
            assert config.sampler_type == "regression"
        elif task_type == "survival":
            assert config.sampler_type == "survival"

        # Property: learning_rate should be task-appropriate
        if task_type == "classification":
            assert config.learning_rate == 3e-4
        else:  # regression or survival
            assert config.learning_rate == 1e-4

        # Property: dataset_fingerprint should be stored
        assert config.dataset_fingerprint == mock_fingerprint


class TestnnMILConfigEdgeCases:
    """Test edge cases for nnMILConfig."""

    def test_config_with_minimal_parameters(self):
        """Test configuration with only required parameters."""
        config = nnMILConfig()  # Use all defaults

        # Should not raise exceptions
        assert isinstance(config, nnMILConfig)
        assert config.feature_dim > 0
        assert config.hidden_dim > 0
        assert config.num_classes > 0

    def test_config_task_type_dependent_parameters(self):
        """Test that task-specific parameters are set correctly."""
        # Classification task
        config_cls = nnMILConfig(task_type="classification")
        assert config_cls.sampler_type == "balanced"
        assert config_cls.learning_rate == 3e-4

        # Regression task
        config_reg = nnMILConfig(task_type="regression")
        assert config_reg.sampler_type == "regression"
        assert config_reg.learning_rate == 1e-4

        # Survival task
        config_surv = nnMILConfig(task_type="survival")
        assert config_surv.sampler_type == "survival"
        assert config_surv.learning_rate == 1e-4

    def test_config_dependent_parameter_derivation(self):
        """Test that dependent parameters are derived correctly."""
        config = nnMILConfig(bag_length=1000)

        # window_size should default to bag_length
        assert config.window_size == config.bag_length

        # stride should default to window_size // 4
        assert config.stride == config.window_size // 4

    def test_config_invalid_fusion_type(self):
        """Test validation of fusion_type parameter."""
        with pytest.raises(ValueError, match="fusion_type must be"):
            nnMILConfig(fusion_type="invalid")

    def test_config_invalid_sampler_type(self):
        """Test validation of sampler_type parameter."""
        with pytest.raises(ValueError, match="sampler_type must be"):
            nnMILConfig(sampler_type="invalid")

    def test_config_invalid_task_type(self):
        """Test validation of task_type parameter."""
        with pytest.raises(ValueError, match="task_type must be"):
            nnMILConfig(task_type="invalid")

    def test_config_yaml_inheritance_missing_base(self):
        """Test error handling when base config file is missing."""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Create child config referencing non-existent base
            child_config = {"inherit_from": "nonexistent.yaml", "hidden_dim": 256}

            child_path = temp_path / "child.yaml"
            with open(child_path, "w") as f:
                yaml.dump(child_config, f)

            # Should raise FileNotFoundError
            with pytest.raises(FileNotFoundError, match="Base configuration not found"):
                nnMILConfig.from_yaml(child_path)

    def test_config_yaml_missing_file(self):
        """Test error handling when config file is missing."""
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            nnMILConfig.from_yaml("nonexistent.yaml")
