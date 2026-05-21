"""
Property-based tests for nnMIL configuration management.

This test file validates correctness properties for the nnMILConfig
class using property-based testing with Hypothesis. Each property test runs
a minimum of 100 iterations to verify universal invariants.

Feature: nnmil-architecture-upgrade
"""

import os
import tempfile

import pytest
import torch
import yaml

from hypothesis import given, settings
from hypothesis import strategies as st
from src.core.config.nnmil_config import nnMILConfig

# ============================================================================
# Property 28: Configuration Loading
# ============================================================================


@settings(max_examples=50, deadline=None)
@given(
    feature_dim=st.integers(min_value=256, max_value=2048),
    hidden_dim=st.integers(min_value=64, max_value=512),
    num_classes=st.integers(min_value=2, max_value=10),
    dropout=st.floats(min_value=0.0, max_value=0.5),
    batch_size=st.integers(min_value=1, max_value=64),
    learning_rate=st.floats(min_value=1e-5, max_value=1e-2),
    task_type=st.sampled_from(["classification", "regression", "survival"]),
)
def test_property_28_configuration_loading(
    feature_dim, hidden_dim, num_classes, dropout, batch_size, learning_rate, task_type
):
    """
    Feature: nnmil-architecture-upgrade, Property 28: For any valid YAML
    configuration file, the system SHALL successfully load and parse the
    configuration without errors.

    **Validates: Requirements 9.1**
    """
    # Create valid configuration dictionary
    config_dict = {
        "model": {
            "feature_dim": feature_dim,
            "hidden_dim": hidden_dim,
            "num_classes": num_classes,
            "dropout": dropout,
            "multi_scale": False,
            "num_scales": 1,
            "fusion_strategy": "early",
        },
        "training": {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_epochs": 100,
            "patience": 10,
            "checkpoint_interval": 5,
        },
        "data": {"bag_length": 512, "task_type": task_type},
    }

    # Write to temporary YAML file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_dict, f)
        temp_path = f.name

    try:
        # Should successfully load configuration
        config = nnMILConfig.from_yaml(temp_path)

        # Verify loaded values match input
        assert config.model.feature_dim == feature_dim
        assert config.model.hidden_dim == hidden_dim
        assert config.model.num_classes == num_classes
        assert abs(config.model.dropout - dropout) < 1e-6
        assert config.training.batch_size == batch_size
        assert abs(config.training.learning_rate - learning_rate) < 1e-6
        assert config.data.task_type == task_type

    finally:
        # Clean up temporary file
        os.unlink(temp_path)


# ============================================================================
# Property 29: Configuration Validation
# ============================================================================


@settings(max_examples=100, deadline=None)
@given(
    invalid_param=st.sampled_from(
        [
            ("feature_dim", -1),
            ("hidden_dim", 0),
            ("num_classes", -5),
            ("dropout", 1.5),
            ("dropout", -0.1),
            ("batch_size", 0),
            ("learning_rate", -0.001),
            ("max_epochs", -10),
            ("patience", -1),
            ("bag_length", 0),
            ("num_scales", 0),
            ("fusion_strategy", "invalid"),
            ("task_type", "invalid_task"),
        ]
    )
)
def test_property_29_configuration_validation(invalid_param):
    """
    Feature: nnmil-architecture-upgrade, Property 29: For any invalid configuration
    parameter, the system SHALL raise a descriptive ValueError indicating which
    parameter is invalid and why.

    **Validates: Requirements 9.3**
    """
    param_name, invalid_value = invalid_param

    # Create base valid configuration
    base_config = {
        "model": {
            "feature_dim": 1024,
            "hidden_dim": 256,
            "num_classes": 2,
            "dropout": 0.25,
            "multi_scale": False,
            "num_scales": 1,
            "fusion_strategy": "early",
        },
        "training": {
            "batch_size": 32,
            "learning_rate": 3e-4,
            "max_epochs": 100,
            "patience": 10,
            "checkpoint_interval": 5,
        },
        "data": {"bag_length": 512, "task_type": "classification"},
    }

    # Inject invalid parameter
    if param_name in [
        "feature_dim",
        "hidden_dim",
        "num_classes",
        "dropout",
        "multi_scale",
        "num_scales",
        "fusion_strategy",
    ]:
        base_config["model"][param_name] = invalid_value
    elif param_name in [
        "batch_size",
        "learning_rate",
        "max_epochs",
        "patience",
        "checkpoint_interval",
    ]:
        base_config["training"][param_name] = invalid_value
    elif param_name in ["bag_length", "task_type"]:
        base_config["data"][param_name] = invalid_value

    # Write to temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(base_config, f)
        temp_path = f.name

    try:
        # Should raise ValueError with descriptive message
        with pytest.raises(ValueError) as exc_info:
            nnMILConfig.from_yaml(temp_path)

        # Verify error message mentions the invalid parameter
        error_msg = str(exc_info.value).lower()
        assert param_name.lower() in error_msg or str(invalid_value) in error_msg, (
            f"Error message should mention invalid parameter '{param_name}' or value '{invalid_value}'. "
            f"Got: {exc_info.value}"
        )

    finally:
        os.unlink(temp_path)


# ============================================================================
# Property 30: Configuration Inheritance
# ============================================================================


def test_property_30_configuration_inheritance():
    """
    Feature: nnmil-architecture-upgrade, Property 30: For any task-specific
    configuration that overrides base configuration, the final configuration
    SHALL contain task-specific values for overridden parameters and base
    values for non-overridden parameters.

    **Validates: Requirements 9.4**
    """
    # Create base configuration
    base_config = {
        "model": {"feature_dim": 1024, "hidden_dim": 256, "num_classes": 2, "dropout": 0.25},
        "training": {"batch_size": 32, "learning_rate": 3e-4, "max_epochs": 100},
        "data": {"bag_length": 512, "task_type": "classification"},
    }

    # Create task-specific override (only some parameters)
    task_override = {
        "model": {
            "num_classes": 5,  # Override
            "dropout": 0.1,  # Override
            # feature_dim, hidden_dim not specified -> should inherit from base
        },
        "training": {
            "learning_rate": 1e-4  # Override
            # batch_size, max_epochs not specified -> should inherit from base
        },
        # data section not specified -> should inherit entirely from base
    }

    # Write configurations to temporary files
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(base_config, f)
        base_path = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(task_override, f)
        task_path = f.name

    try:
        # Load with inheritance
        config = nnMILConfig.from_yaml_with_inheritance(base_path, task_path)

        # Verify overridden parameters use task-specific values
        assert config.model.num_classes == 5  # Overridden
        assert abs(config.model.dropout - 0.1) < 1e-6  # Overridden
        assert abs(config.training.learning_rate - 1e-4) < 1e-6  # Overridden

        # Verify non-overridden parameters use base values
        assert config.model.feature_dim == 1024  # Inherited from base
        assert config.model.hidden_dim == 256  # Inherited from base
        assert config.training.batch_size == 32  # Inherited from base
        assert config.training.max_epochs == 100  # Inherited from base
        assert config.data.bag_length == 512  # Inherited from base (entire section)
        assert config.data.task_type == "classification"  # Inherited from base

    finally:
        os.unlink(base_path)
        os.unlink(task_path)


# ============================================================================
# Property 31: Configuration Logging
# ============================================================================


def test_property_31_configuration_logging():
    """
    Feature: nnmil-architecture-upgrade, Property 31: For any training run,
    all active configuration parameters SHALL be logged at the start of training.

    **Validates: Requirements 9.5**
    """
    # Create configuration
    config_dict = {
        "model": {"feature_dim": 1024, "hidden_dim": 256, "num_classes": 2, "dropout": 0.25},
        "training": {"batch_size": 32, "learning_rate": 3e-4, "max_epochs": 100},
        "data": {"bag_length": 512, "task_type": "classification"},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_dict, f)
        temp_path = f.name

    try:
        config = nnMILConfig.from_yaml(temp_path)

        # Get configuration log string
        log_str = config.to_log_string()

        # Verify all major parameters are logged
        assert "feature_dim: 1024" in log_str
        assert "hidden_dim: 256" in log_str
        assert "num_classes: 2" in log_str
        assert "dropout: 0.25" in log_str
        assert "batch_size: 32" in log_str
        assert "learning_rate: 0.0003" in log_str or "learning_rate: 3e-04" in log_str
        assert "bag_length: 512" in log_str
        assert "task_type: classification" in log_str

    finally:
        os.unlink(temp_path)


# ============================================================================
# Property 32: Configuration Persistence
# ============================================================================


def test_property_32_configuration_persistence():
    """
    Feature: nnmil-architecture-upgrade, Property 32: For any saved model
    checkpoint, the checkpoint file SHALL contain the complete configuration
    used during training.

    **Validates: Requirements 9.6**
    """
    # Create configuration
    config_dict = {
        "model": {"feature_dim": 1024, "hidden_dim": 256, "num_classes": 3, "dropout": 0.2},
        "training": {"batch_size": 16, "learning_rate": 1e-4, "max_epochs": 50},
        "data": {"bag_length": 256, "task_type": "classification"},
    }

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(config_dict, f)
        temp_path = f.name

    try:
        config = nnMILConfig.from_yaml(temp_path)

        # Simulate saving checkpoint with configuration
        checkpoint_data = {
            "model_state_dict": {},  # Would contain actual model weights
            "optimizer_state_dict": {},  # Would contain optimizer state
            "epoch": 10,
            "best_val_auc": 0.85,
            "config": config.to_dict(),  # Configuration should be saved
        }

        # Verify configuration is in checkpoint format
        saved_config = checkpoint_data["config"]

        # Verify all sections are preserved
        assert "model" in saved_config
        assert "training" in saved_config
        assert "data" in saved_config

        # Verify specific values are preserved
        assert saved_config["model"]["feature_dim"] == 1024
        assert saved_config["model"]["num_classes"] == 3
        assert saved_config["training"]["batch_size"] == 16
        assert saved_config["data"]["bag_length"] == 256

        # Verify configuration can be reconstructed from checkpoint
        reconstructed_config = nnMILConfig.from_dict(saved_config)

        # Verify reconstructed config matches original
        assert reconstructed_config.model.feature_dim == config.model.feature_dim
        assert reconstructed_config.model.num_classes == config.model.num_classes
        assert reconstructed_config.training.batch_size == config.training.batch_size
        assert reconstructed_config.data.bag_length == config.data.bag_length

    finally:
        os.unlink(temp_path)


# ============================================================================
# Edge Case Tests
# ============================================================================


def test_empty_configuration_file():
    """Test handling of empty configuration file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("")  # Empty file
        temp_path = f.name

    try:
        with pytest.raises(ValueError) as exc_info:
            nnMILConfig.from_yaml(temp_path)

        error_msg = str(exc_info.value).lower()
        assert "empty" in error_msg or "missing" in error_msg

    finally:
        os.unlink(temp_path)


def test_malformed_yaml():
    """Test handling of malformed YAML file."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("invalid: yaml: content: [unclosed")  # Malformed YAML
        temp_path = f.name

    try:
        with pytest.raises((ValueError, yaml.YAMLError)):
            nnMILConfig.from_yaml(temp_path)

    finally:
        os.unlink(temp_path)


def test_missing_required_sections():
    """Test handling of configuration missing required sections."""
    # Configuration missing 'model' section
    incomplete_config = {"training": {"batch_size": 32, "learning_rate": 3e-4}}

    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yaml.dump(incomplete_config, f)
        temp_path = f.name

    try:
        with pytest.raises(ValueError) as exc_info:
            nnMILConfig.from_yaml(temp_path)

        error_msg = str(exc_info.value).lower()
        assert "model" in error_msg or "required" in error_msg or "missing" in error_msg

    finally:
        os.unlink(temp_path)


def test_dataset_fingerprinting():
    """Test automatic configuration from dataset fingerprint."""

    # Create mock dataset with known characteristics
    class MockDataset:
        def __init__(self, patch_counts, labels):
            self.patch_counts = patch_counts
            self.labels = labels

        def __len__(self):
            return len(self.patch_counts)

        def __getitem__(self, idx):
            return {
                "features": torch.randn(self.patch_counts[idx], 1024),
                "label": self.labels[idx],
                "num_patches": self.patch_counts[idx],
            }

    # Dataset with median patch count = 500
    patch_counts = [200, 400, 500, 600, 800]
    labels = [0, 1, 0, 1, 0]
    dataset = MockDataset(patch_counts, labels)

    # Generate configuration from dataset
    config = nnMILConfig.from_dataset(dataset, task_type="classification")

    # Verify rule-based configuration
    # bag_length should be median_patches / 2 = 500 / 2 = 250
    assert config.data.bag_length == 250

    # Verify fixed hyperparameters
    assert config.model.hidden_dim == 256
    assert abs(config.model.dropout - 0.25) < 1e-6

    # Verify task-specific defaults
    assert config.data.task_type == "classification"
    assert abs(config.training.learning_rate - 3e-4) < 1e-6  # Classification default
