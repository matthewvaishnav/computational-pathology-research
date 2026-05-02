"""
Unit tests for Training Task Executor.

Tests task specification validation, configuration translation for each framework,
and configuration difference logging.

Requirements: 2.1, 2.8
"""

from pathlib import Path

import pytest

from experiments.benchmark_system.models import (
    FrameworkEnvironment,
    TaskSpecification,
)
from experiments.benchmark_system.task_executor import (
    EquivalenceReport,
    TrainingConfig,
    TrainingTaskExecutor,
)


@pytest.fixture
def sample_task_spec():
    """Create a sample task specification for testing."""
    return TaskSpecification(
        dataset_name="PatchCamelyon",
        data_root=Path("/data/pcam"),
        train_split=0.8,
        val_split=0.1,
        test_split=0.1,
        model_architecture="resnet18_transformer",
        feature_dim=512,
        num_classes=2,
        num_epochs=10,
        batch_size=32,
        learning_rate=1e-4,
        weight_decay=1e-5,
        optimizer="AdamW",
        random_seed=42,
        augmentation_config={
            "flip": True,
            "rotate": 0.2,
            "normalize": True,
        },
        metrics=["accuracy", "auc", "f1"],
    )


@pytest.fixture
def executor():
    """Create a TrainingTaskExecutor instance."""
    return TrainingTaskExecutor()


class TestTaskSpecificationValidation:
    """Test task specification validation (Requirement 2.1)."""
    
    def test_valid_task_specification(self, sample_task_spec):
        """Test that valid task specification is accepted."""
        # Should not raise any exceptions
        assert sample_task_spec.dataset_name == "PatchCamelyon"
        assert sample_task_spec.num_epochs == 10
        assert sample_task_spec.random_seed == 42
    
    def test_invalid_train_split_too_low(self):
        """Test that train_split below 0 is rejected."""
        with pytest.raises(ValueError, match="train_split must be in"):
            TaskSpecification(
                dataset_name="PatchCamelyon",
                data_root=Path("/data/pcam"),
                train_split=-0.1,  # Invalid
                val_split=0.1,
                test_split=0.1,
                model_architecture="resnet18_transformer",
            )
    
    def test_invalid_train_split_too_high(self):
        """Test that train_split above 1 is rejected."""
        with pytest.raises(ValueError, match="train_split must be in"):
            TaskSpecification(
                dataset_name="PatchCamelyon",
                data_root=Path("/data/pcam"),
                train_split=1.5,  # Invalid
                val_split=0.1,
                test_split=0.1,
                model_architecture="resnet18_transformer",
            )
    
    def test_invalid_splits_dont_sum_to_one(self):
        """Test that splits not summing to 1.0 are rejected."""
        with pytest.raises(ValueError, match="Splits must sum to 1.0"):
            TaskSpecification(
                dataset_name="PatchCamelyon",
                data_root=Path("/data/pcam"),
                train_split=0.5,
                val_split=0.3,
                test_split=0.3,  # Sum = 1.1, invalid
                model_architecture="resnet18_transformer",
            )
    
    def test_invalid_num_epochs_zero(self):
        """Test that num_epochs of 0 is rejected."""
        with pytest.raises(ValueError, match="num_epochs must be positive"):
            TaskSpecification(
                dataset_name="PatchCamelyon",
                data_root=Path("/data/pcam"),
                train_split=0.8,
                val_split=0.1,
                test_split=0.1,
                model_architecture="resnet18_transformer",
                num_epochs=0,  # Invalid
            )
    
    def test_invalid_num_epochs_negative(self):
        """Test that negative num_epochs is rejected."""
        with pytest.raises(ValueError, match="num_epochs must be positive"):
            TaskSpecification(
                dataset_name="PatchCamelyon",
                data_root=Path("/data/pcam"),
                train_split=0.8,
                val_split=0.1,
                test_split=0.1,
                model_architecture="resnet18_transformer",
                num_epochs=-5,  # Invalid
            )
    
    def test_invalid_batch_size_zero(self):
        """Test that batch_size of 0 is rejected."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            TaskSpecification(
                dataset_name="PatchCamelyon",
                data_root=Path("/data/pcam"),
                train_split=0.8,
                val_split=0.1,
                test_split=0.1,
                model_architecture="resnet18_transformer",
                batch_size=0,  # Invalid
            )
    
    def test_invalid_learning_rate_zero(self):
        """Test that learning_rate of 0 is rejected."""
        with pytest.raises(ValueError, match="learning_rate must be positive"):
            TaskSpecification(
                dataset_name="PatchCamelyon",
                data_root=Path("/data/pcam"),
                train_split=0.8,
                val_split=0.1,
                test_split=0.1,
                model_architecture="resnet18_transformer",
                learning_rate=0.0,  # Invalid
            )


class TestConfigurationTranslation:
    """Test configuration translation for each framework (Requirement 2.1)."""
    
    def test_configure_histocore(self, executor, sample_task_spec):
        """Test configuration translation for HistoCore."""
        config = executor.configure_task(sample_task_spec, "HistoCore")
        
        assert config.framework_name == "HistoCore"
        assert config.task_spec == sample_task_spec
        assert config.random_seed == 42
        assert config.data_split_hash is not None
        
        # Check framework-specific config structure
        assert "model" in config.config_dict
        assert "training" in config.config_dict
        assert config.config_dict["model"]["architecture"] == "resnet18_transformer"
        assert config.config_dict["training"]["optimizer"]["name"] == "AdamW"
    
    def test_configure_pathml(self, executor, sample_task_spec):
        """Test configuration translation for PathML."""
        config = executor.configure_task(sample_task_spec, "PathML")
        
        assert config.framework_name == "PathML"
        assert config.task_spec == sample_task_spec
        assert config.random_seed == 42
        
        # Check framework-specific config structure
        assert "model_config" in config.config_dict
        assert "train_config" in config.config_dict
        assert config.config_dict["model_config"]["name"] == "resnet18"
        assert config.config_dict["train_config"]["optimizer"] == "adamw"
    
    def test_configure_clam(self, executor, sample_task_spec):
        """Test configuration translation for CLAM."""
        config = executor.configure_task(sample_task_spec, "CLAM")
        
        assert config.framework_name == "CLAM"
        assert config.task_spec == sample_task_spec
        assert config.random_seed == 42
        
        # Check framework-specific config structure
        assert config.config_dict["model_type"] == "resnet18"
        assert config.config_dict["opt"] == "adam"  # CLAM may not support AdamW
        assert config.config_dict["model_size"] == "small"  # 512 feature_dim
    
    def test_configure_pytorch(self, executor, sample_task_spec):
        """Test configuration translation for baseline PyTorch."""
        config = executor.configure_task(sample_task_spec, "PyTorch")
        
        assert config.framework_name == "PyTorch"
        assert config.task_spec == sample_task_spec
        assert config.random_seed == 42
        
        # Check framework-specific config structure
        assert config.config_dict["model"] == "resnet18"
        assert config.config_dict["optimizer"]["type"] == "AdamW"
        assert config.config_dict["epochs"] == 10
    
    def test_configure_unsupported_framework(self, executor, sample_task_spec):
        """Test that unsupported framework raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported framework"):
            executor.configure_task(sample_task_spec, "UnsupportedFramework")
    
    def test_optimizer_translation_adam(self, executor):
        """Test optimizer translation for Adam."""
        task_spec = TaskSpecification(
            dataset_name="PatchCamelyon",
            data_root=Path("/data/pcam"),
            model_architecture="resnet18_transformer",
            optimizer="Adam",
        )
        
        histocore_config = executor.configure_task(task_spec, "HistoCore")
        pathml_config = executor.configure_task(task_spec, "PathML")
        
        assert histocore_config.config_dict["training"]["optimizer"]["name"] == "Adam"
        assert pathml_config.config_dict["train_config"]["optimizer"] == "adam"
    
    def test_optimizer_translation_sgd(self, executor):
        """Test optimizer translation for SGD."""
        task_spec = TaskSpecification(
            dataset_name="PatchCamelyon",
            data_root=Path("/data/pcam"),
            model_architecture="resnet18_transformer",
            optimizer="SGD",
        )
        
        histocore_config = executor.configure_task(task_spec, "HistoCore")
        clam_config = executor.configure_task(task_spec, "CLAM")
        
        assert histocore_config.config_dict["training"]["optimizer"]["name"] == "SGD"
        assert clam_config.config_dict["opt"] == "sgd"
    
    def test_architecture_translation_vit(self, executor):
        """Test architecture translation for ViT."""
        task_spec = TaskSpecification(
            dataset_name="PatchCamelyon",
            data_root=Path("/data/pcam"),
            model_architecture="vit",
        )
        
        histocore_config = executor.configure_task(task_spec, "HistoCore")
        pytorch_config = executor.configure_task(task_spec, "PyTorch")
        
        assert histocore_config.config_dict["model"]["architecture"] == "vit_base"
        assert pytorch_config.config_dict["model"] == "vit_b_16"
    
    def test_data_split_hash_consistency(self, executor, sample_task_spec):
        """Test that data split hash is consistent for same task spec."""
        config1 = executor.configure_task(sample_task_spec, "HistoCore")
        config2 = executor.configure_task(sample_task_spec, "PathML")
        
        # Same task spec should produce same hash
        assert config1.data_split_hash == config2.data_split_hash
    
    def test_data_split_hash_differs_for_different_specs(self, executor):
        """Test that data split hash differs for different task specs."""
        task_spec1 = TaskSpecification(
            dataset_name="PatchCamelyon",
            data_root=Path("/data/pcam"),
            model_architecture="resnet18_transformer",
            random_seed=42,
        )
        task_spec2 = TaskSpecification(
            dataset_name="PatchCamelyon",
            data_root=Path("/data/pcam"),
            model_architecture="resnet18_transformer",
            random_seed=123,  # Different seed
        )
        
        config1 = executor.configure_task(task_spec1, "HistoCore")
        config2 = executor.configure_task(task_spec2, "HistoCore")
        
        # Different seeds should produce different hashes
        assert config1.data_split_hash != config2.data_split_hash


class TestEquivalenceValidation:
    """Test configuration equivalence validation (Requirement 2.8)."""
    
    def test_validate_equivalence_identical_configs(self, executor, sample_task_spec):
        """Test that identical configurations are reported as equivalent."""
        configs = [
            executor.configure_task(sample_task_spec, "HistoCore"),
            executor.configure_task(sample_task_spec, "PathML"),
            executor.configure_task(sample_task_spec, "CLAM"),
            executor.configure_task(sample_task_spec, "PyTorch"),
        ]
        
        report = executor.validate_equivalence(configs)
        
        assert report.is_equivalent
        assert len(report.differences) == 0
        assert len(report.random_seeds) == 4
        assert len(report.data_splits) == 4
    
    def test_validate_equivalence_single_config(self, executor, sample_task_spec):
        """Test that single configuration is reported as equivalent (trivially)."""
        config = executor.configure_task(sample_task_spec, "HistoCore")
        
        report = executor.validate_equivalence([config])
        
        assert report.is_equivalent
        assert len(report.warnings) == 1
        assert "only one configuration" in report.warnings[0].lower()
    
    def test_validate_equivalence_detects_seed_mismatch(self, executor):
        """Test that random seed mismatch is detected."""
        task_spec1 = TaskSpecification(
            dataset_name="PatchCamelyon",
            data_root=Path("/data/pcam"),
            model_architecture="resnet18_transformer",
            random_seed=42,
        )
        task_spec2 = TaskSpecification(
            dataset_name="PatchCamelyon",
            data_root=Path("/data/pcam"),
            model_architecture="resnet18_transformer",
            random_seed=123,  # Different seed
        )
        
        config1 = executor.configure_task(task_spec1, "HistoCore")
        config2 = executor.configure_task(task_spec2, "PathML")
        
        report = executor.validate_equivalence([config1, config2])
        
        assert not report.is_equivalent
        assert any("seed" in diff.lower() for diff in report.differences)
    
    def test_validate_equivalence_detects_split_mismatch(self, executor):
        """Test that data split mismatch is detected."""
        task_spec1 = TaskSpecification(
            dataset_name="PatchCamelyon",
            data_root=Path("/data/pcam"),
            model_architecture="resnet18_transformer",
            train_split=0.8,
            val_split=0.1,
            test_split=0.1,
        )
        task_spec2 = TaskSpecification(
            dataset_name="PatchCamelyon",
            data_root=Path("/data/pcam"),
            model_architecture="resnet18_transformer",
            train_split=0.7,  # Different split
            val_split=0.2,
            test_split=0.1,
        )
        
        config1 = executor.configure_task(task_spec1, "HistoCore")
        config2 = executor.configure_task(task_spec2, "PathML")
        
        report = executor.validate_equivalence([config1, config2])
        
        assert not report.is_equivalent
        assert any("split" in diff.lower() for diff in report.differences)
    
    def test_validate_equivalence_detects_learning_rate_mismatch(self, executor):
        """Test that learning rate mismatch is detected."""
        task_spec1 = TaskSpecification(
            dataset_name="PatchCamelyon",
            data_root=Path("/data/pcam"),
            model_architecture="resnet18_transformer",
            learning_rate=1e-4,
        )
        task_spec2 = TaskSpecification(
            dataset_name="PatchCamelyon",
            data_root=Path("/data/pcam"),
            model_architecture="resnet18_transformer",
            learning_rate=1e-3,  # Different learning rate
        )
        
        config1 = executor.configure_task(task_spec1, "HistoCore")
        config2 = executor.configure_task(task_spec2, "PathML")
        
        report = executor.validate_equivalence([config1, config2])
        
        assert not report.is_equivalent
        assert any("learning rate" in diff.lower() for diff in report.differences)
    
    def test_validate_equivalence_detects_batch_size_mismatch(self, executor):
        """Test that batch size mismatch is detected."""
        task_spec1 = TaskSpecification(
            dataset_name="PatchCamelyon",
            data_root=Path("/data/pcam"),
            model_architecture="resnet18_transformer",
            batch_size=32,
        )
        task_spec2 = TaskSpecification(
            dataset_name="PatchCamelyon",
            data_root=Path("/data/pcam"),
            model_architecture="resnet18_transformer",
            batch_size=64,  # Different batch size
        )
        
        config1 = executor.configure_task(task_spec1, "HistoCore")
        config2 = executor.configure_task(task_spec2, "PathML")
        
        report = executor.validate_equivalence([config1, config2])
        
        assert not report.is_equivalent
        assert any("batch size" in diff.lower() for diff in report.differences)
    
    def test_validate_equivalence_detects_epoch_mismatch(self, executor):
        """Test that number of epochs mismatch is detected."""
        task_spec1 = TaskSpecification(
            dataset_name="PatchCamelyon",
            data_root=Path("/data/pcam"),
            model_architecture="resnet18_transformer",
            num_epochs=10,
        )
        task_spec2 = TaskSpecification(
            dataset_name="PatchCamelyon",
            data_root=Path("/data/pcam"),
            model_architecture="resnet18_transformer",
            num_epochs=20,  # Different epochs
        )
        
        config1 = executor.configure_task(task_spec1, "HistoCore")
        config2 = executor.configure_task(task_spec2, "PathML")
        
        report = executor.validate_equivalence([config1, config2])
        
        assert not report.is_equivalent
        assert any("epochs" in diff.lower() for diff in report.differences)
    
    def test_validate_equivalence_detects_feature_dim_mismatch(self, executor):
        """Test that feature dimension mismatch is detected."""
        task_spec1 = TaskSpecification(
            dataset_name="PatchCamelyon",
            data_root=Path("/data/pcam"),
            model_architecture="resnet18_transformer",
            feature_dim=512,
        )
        task_spec2 = TaskSpecification(
            dataset_name="PatchCamelyon",
            data_root=Path("/data/pcam"),
            model_architecture="resnet18_transformer",
            feature_dim=1024,  # Different feature dim
        )
        
        config1 = executor.configure_task(task_spec1, "HistoCore")
        config2 = executor.configure_task(task_spec2, "PathML")
        
        report = executor.validate_equivalence([config1, config2])
        
        assert not report.is_equivalent
        assert any("feature dimension" in diff.lower() for diff in report.differences)
    
    def test_validate_equivalence_detects_augmentation_mismatch(self, executor):
        """Test that augmentation config mismatch is detected."""
        task_spec1 = TaskSpecification(
            dataset_name="PatchCamelyon",
            data_root=Path("/data/pcam"),
            model_architecture="resnet18_transformer",
            augmentation_config={"flip": True},
        )
        task_spec2 = TaskSpecification(
            dataset_name="PatchCamelyon",
            data_root=Path("/data/pcam"),
            model_architecture="resnet18_transformer",
            augmentation_config={"flip": False},  # Different augmentation
        )
        
        config1 = executor.configure_task(task_spec1, "HistoCore")
        config2 = executor.configure_task(task_spec2, "PathML")
        
        report = executor.validate_equivalence([config1, config2])
        
        assert not report.is_equivalent
        assert any("augmentation" in diff.lower() for diff in report.differences)
    
    def test_validate_equivalence_allows_architecture_translation(self, executor):
        """Test that framework-specific architecture names are allowed (warnings only)."""
        task_spec = TaskSpecification(
            dataset_name="PatchCamelyon",
            data_root=Path("/data/pcam"),
            model_architecture="resnet18_transformer",
        )
        
        configs = [
            executor.configure_task(task_spec, "HistoCore"),
            executor.configure_task(task_spec, "PathML"),
        ]
        
        report = executor.validate_equivalence(configs)
        
        # Should be equivalent (architecture translation is acceptable)
        assert report.is_equivalent
        # May have warnings about architecture names
        # (this is acceptable for framework-specific translations)


class TestConfigurationDifferenceLogging:
    """Test configuration difference logging (Requirement 2.8)."""
    
    def test_logs_configuration_differences(self, executor, caplog):
        """Test that configuration differences are logged."""
        import logging
        caplog.set_level(logging.WARNING)
        
        task_spec1 = TaskSpecification(
            dataset_name="PatchCamelyon",
            data_root=Path("/data/pcam"),
            model_architecture="resnet18_transformer",
            learning_rate=1e-4,
        )
        task_spec2 = TaskSpecification(
            dataset_name="PatchCamelyon",
            data_root=Path("/data/pcam"),
            model_architecture="resnet18_transformer",
            learning_rate=1e-3,  # Different
        )
        
        config1 = executor.configure_task(task_spec1, "HistoCore")
        config2 = executor.configure_task(task_spec2, "PathML")
        
        report = executor.validate_equivalence([config1, config2])
        
        # Should log warning about differences
        assert any("Configuration differences detected" in record.message 
                   for record in caplog.records)
        assert any("learning rate" in record.message.lower() 
                   for record in caplog.records)
    
    def test_logs_success_for_equivalent_configs(self, executor, sample_task_spec, caplog):
        """Test that success is logged for equivalent configurations."""
        import logging
        caplog.set_level(logging.INFO)
        
        configs = [
            executor.configure_task(sample_task_spec, "HistoCore"),
            executor.configure_task(sample_task_spec, "PathML"),
        ]
        
        report = executor.validate_equivalence(configs)
        
        # Should log success message
        assert any("equivalent" in record.message.lower() and "fair comparison" in record.message.lower()
                   for record in caplog.records)
