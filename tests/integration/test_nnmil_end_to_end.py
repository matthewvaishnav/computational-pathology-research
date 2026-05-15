"""
End-to-end integration tests for nnMIL architecture upgrade.

This module tests the complete nnMIL pipeline from data loading through
training to inference, validating all major components work together.
"""

import tempfile
from pathlib import Path
from typing import Dict, List, Tuple

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Import nnMIL components
from src.config.nnmil_config import nnMILConfig
from src.data.bag_samplers import FixedLengthBagSampler
from src.data.batch_samplers import BalancedBatchSampler
from src.data.data_models import Bag, TrainingBatch, InferenceOutput
from src.inference.sliding_window import SlidingWindowInference
from src.inference.uncertainty import UncertaintyEstimator
from src.models import nnMIL, FoundationModelAdapter
from src.training import nnMILTrainer, UnifiedTrainer


class TestnnMILEndToEnd:
    """End-to-end integration tests for nnMIL."""

    @pytest.fixture
    def synthetic_dataset(self) -> List[Bag]:
        """Create synthetic dataset for testing."""
        bags = []

        for i in range(20):  # 20 bags total
            # Variable number of patches (50-200)
            num_patches = torch.randint(50, 201, (1,)).item()

            # Random features (1024-dim for UNI compatibility)
            features = torch.randn(num_patches, 1024)

            # Binary classification labels
            label = i % 2

            bag = Bag(
                features=features,
                label=label,
                num_patches=num_patches,
                slide_id=f"slide_{i:03d}",
                metadata={"patient_id": f"P{i:03d}", "magnification": "20x"},
            )
            bags.append(bag)

        return bags

    @pytest.fixture
    def nnmil_config(self) -> nnMILConfig:
        """Create test configuration."""
        return nnMILConfig(
            feature_dim=1024,
            hidden_dim=128,  # Smaller for faster testing
            num_classes=2,
            dropout=0.1,
            batch_size=4,  # Small batch for testing
            learning_rate=1e-3,
            num_epochs=3,  # Few epochs for testing
            patience=5,
            bag_length=64,  # Small bag length for testing
            task_type="classification",
            sampler_type="balanced",
            enable_uncertainty=True,
            num_mc_samples=5,  # Few samples for testing
        )

    def create_dataloader(
        self, bags: List[Bag], config: nnMILConfig, shuffle: bool = True
    ) -> DataLoader:
        """Create DataLoader from bags."""
        # Convert bags to tensors for DataLoader
        features_list = []
        labels_list = []

        # Use bag sampler to create fixed-length bags
        bag_sampler = FixedLengthBagSampler(bag_length=config.bag_length, mode="train")

        for bag in bags:
            sampled_features, mask = bag_sampler.sample_bag(bag.features, bag.num_patches)
            features_list.append(sampled_features)
            labels_list.append(torch.tensor(bag.label, dtype=torch.long))

        # Stack into tensors
        features_tensor = torch.stack(features_list)  # [N, bag_length, feature_dim]
        labels_tensor = torch.stack(labels_list)  # [N]

        # Create dataset and dataloader
        dataset = TensorDataset(features_tensor, labels_tensor)
        dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=shuffle)

        return dataloader

    def test_nnmil_model_creation(self, nnmil_config: nnMILConfig):
        """Test nnMIL model creation and basic forward pass."""
        model = nnMIL(
            feature_dim=nnmil_config.feature_dim,
            hidden_dim=nnmil_config.hidden_dim,
            num_classes=nnmil_config.num_classes,
            dropout=nnmil_config.dropout,
        )

        # Test forward pass
        batch_size = 2
        bag_length = 64
        features = torch.randn(batch_size, bag_length, nnmil_config.feature_dim)
        masks = torch.ones(batch_size, bag_length, dtype=torch.bool)

        with torch.no_grad():
            logits = model(features, masks)

        assert logits.shape == (batch_size, nnmil_config.num_classes)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_foundation_model_adapter(self, nnmil_config: nnMILConfig):
        """Test foundation model adapter with different input dimensions."""
        adapter = FoundationModelAdapter(target_dim=nnmil_config.hidden_dim)

        # Test different foundation model dimensions
        test_dims = [512, 768, 1024, 2048]  # CONCH, Phikon, UNI, ResNet50

        for input_dim in test_dims:
            features = torch.randn(2, 100, input_dim)
            adapted_features = adapter(features)

            assert adapted_features.shape == (2, 100, nnmil_config.hidden_dim)
            assert not torch.isnan(adapted_features).any()

    def test_training_pipeline(self, synthetic_dataset: List[Bag], nnmil_config: nnMILConfig):
        """Test complete training pipeline."""
        # Split dataset
        train_bags = synthetic_dataset[:16]
        val_bags = synthetic_dataset[16:]

        # Create dataloaders
        train_loader = self.create_dataloader(train_bags, nnmil_config, shuffle=True)
        val_loader = self.create_dataloader(val_bags, nnmil_config, shuffle=False)

        # Create model
        model = nnMIL(
            feature_dim=nnmil_config.feature_dim,
            hidden_dim=nnmil_config.hidden_dim,
            num_classes=nnmil_config.num_classes,
            dropout=nnmil_config.dropout,
        )

        # Create trainer
        trainer = nnMILTrainer(model, nnmil_config, device="cpu")

        # Train model
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_dir = Path(temp_dir) / "checkpoints"

            history = trainer.train(
                train_loader=train_loader, val_loader=val_loader, checkpoint_dir=checkpoint_dir
            )

            # Validate training history
            assert "train_loss" in history
            assert "val_loss" in history
            assert "val_metric" in history
            assert len(history["train_loss"]) == nnmil_config.num_epochs

            # Check that loss decreased (learning occurred)
            initial_loss = history["train_loss"][0]
            final_loss = history["train_loss"][-1]
            assert final_loss < initial_loss, "Model should learn and reduce loss"

            # Check checkpoint was saved
            checkpoint_files = list(checkpoint_dir.glob("*.pth"))
            assert len(checkpoint_files) > 0, "Checkpoints should be saved"

    def test_sliding_window_inference(self, nnmil_config: nnMILConfig):
        """Test sliding window inference with large bags."""
        model = nnMIL(
            feature_dim=nnmil_config.feature_dim,
            hidden_dim=nnmil_config.hidden_dim,
            num_classes=nnmil_config.num_classes,
            dropout=nnmil_config.dropout,
        )

        # Create sliding window inference
        inference = SlidingWindowInference(
            model=model, window_size=nnmil_config.bag_length, device="cpu"
        )

        # Test with large bag (requires sliding window)
        large_bag = Bag(
            features=torch.randn(200, nnmil_config.feature_dim),  # Larger than window
            label=1,
            num_patches=200,
            slide_id="large_slide",
        )

        output = inference(large_bag)

        # Validate output
        assert isinstance(output, InferenceOutput)
        assert output.logits.shape == (nnmil_config.num_classes,)
        assert output.probabilities.shape == (nnmil_config.num_classes,)
        assert output.attention_weights.shape[0] == 200  # Original bag size
        assert not torch.isnan(output.logits).any()
        assert not torch.isnan(output.epistemic_uncertainty).any()
        assert not torch.isnan(output.aleatoric_uncertainty).any()

    def test_uncertainty_estimation(self, nnmil_config: nnMILConfig):
        """Test uncertainty estimation with Monte Carlo dropout."""
        model = nnMIL(
            feature_dim=nnmil_config.feature_dim,
            hidden_dim=nnmil_config.hidden_dim,
            num_classes=nnmil_config.num_classes,
            dropout=nnmil_config.dropout,
        )

        # Create uncertainty estimator
        estimator = UncertaintyEstimator(
            model=model, num_samples=nnmil_config.num_mc_samples, device="cpu"
        )

        # Test bag
        test_bag = Bag(
            features=torch.randn(100, nnmil_config.feature_dim),
            label=0,
            num_patches=100,
            slide_id="test_slide",
        )

        output = estimator(test_bag)

        # Validate uncertainty output
        assert isinstance(output, InferenceOutput)
        assert output.epistemic_uncertainty.shape == ()  # Scalar
        assert output.aleatoric_uncertainty.shape == ()  # Scalar
        assert output.total_uncertainty.shape == ()  # Scalar

        # Uncertainties should be non-negative
        assert output.epistemic_uncertainty >= 0
        assert output.aleatoric_uncertainty >= 0
        assert output.total_uncertainty >= 0

    def test_unified_trainer_nnmil(self, synthetic_dataset: List[Bag], nnmil_config: nnMILConfig):
        """Test unified trainer with nnMIL model."""
        # Create dataloaders
        train_bags = synthetic_dataset[:16]
        val_bags = synthetic_dataset[16:]

        train_loader = self.create_dataloader(train_bags, nnmil_config, shuffle=True)
        val_loader = self.create_dataloader(val_bags, nnmil_config, shuffle=False)

        # Create unified trainer
        trainer = UnifiedTrainer(config=nnmil_config, model_type="nnmil", device="cpu")

        # Train model
        with tempfile.TemporaryDirectory() as temp_dir:
            result = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                checkpoint_dir=Path(temp_dir) / "checkpoints",
            )

            # Validate result
            assert result["model_type"] == "nnmil"
            assert "training_history" in result
            assert "training_stats" in result
            assert "model_info" in result

            # Check model info
            model_info = result["model_info"]
            assert model_info["multi_scale"] == nnmil_config.multi_scale
            assert model_info["uncertainty_enabled"] == nnmil_config.enable_uncertainty

    def test_unified_trainer_transmil(self, synthetic_dataset: List[Bag]):
        """Test unified trainer with TransMIL model (backward compatibility)."""
        # Create TransMIL-style config
        transmil_config = {
            "feature_dim": 1024,
            "hidden_dim": 128,
            "num_classes": 2,
            "dropout": 0.1,
            "batch_size": 4,
            "learning_rate": 2e-4,  # TransMIL default
            "num_epochs": 2,
            "bag_length": 64,
            "task_type": "classification",
        }

        # Create dataloaders
        train_bags = synthetic_dataset[:16]
        val_bags = synthetic_dataset[16:]

        config = nnMILConfig(**transmil_config)
        train_loader = self.create_dataloader(train_bags, config, shuffle=True)
        val_loader = self.create_dataloader(val_bags, config, shuffle=False)

        # Create unified trainer with TransMIL
        trainer = UnifiedTrainer(config=transmil_config, model_type="transmil", device="cpu")

        # Train model
        with tempfile.TemporaryDirectory() as temp_dir:
            result = trainer.train(
                train_loader=train_loader,
                val_loader=val_loader,
                checkpoint_dir=Path(temp_dir) / "checkpoints",
            )

            # Validate result
            assert result["model_type"] == "transmil"
            assert "training_history" in result

            # Check model info (TransMIL-specific)
            model_info = result["model_info"]
            assert model_info["multi_scale"] == False
            assert model_info["uncertainty_enabled"] == False

    def test_config_system(self):
        """Test configuration system with YAML loading and inheritance."""
        # Test automatic configuration from dataset fingerprint
        config = nnMILConfig.from_dataset(
            dataset_path="dummy_path", task_type="classification"  # Won't exist, will use defaults
        )

        assert config.task_type == "classification"
        assert config.feature_dim == 1024  # Default
        assert config.bag_length > 0
        assert config.sampler_type == "balanced"

        # Test configuration validation
        with pytest.raises(ValueError):
            nnMILConfig(feature_dim=-1)  # Invalid feature_dim

        with pytest.raises(ValueError):
            nnMILConfig(batch_size=0)  # Invalid batch_size

    def test_multi_scale_processing(self, nnmil_config: nnMILConfig):
        """Test multi-scale feature processing."""
        # Enable multi-scale
        config = nnMILConfig(**nnmil_config.to_dict(), multi_scale=True, fusion_type="early")

        model = nnMIL(
            feature_dim=config.feature_dim,
            hidden_dim=config.hidden_dim,
            num_classes=config.num_classes,
            dropout=config.dropout,
            multi_scale=config.multi_scale,
            fusion_type=config.fusion_type,
        )

        # Test with multi-scale input (2 scales)
        batch_size = 2
        bag_length = 64

        # Scale 1: 20x magnification
        features_20x = torch.randn(batch_size, bag_length, config.feature_dim)

        # Scale 2: 40x magnification
        features_40x = torch.randn(batch_size, bag_length, config.feature_dim)

        # Combine scales (early fusion)
        multi_scale_features = torch.cat([features_20x, features_40x], dim=2)  # [B, N, 2*D]

        masks = torch.ones(batch_size, bag_length, dtype=torch.bool)

        with torch.no_grad():
            logits = model(multi_scale_features, masks)

        assert logits.shape == (batch_size, config.num_classes)
        assert not torch.isnan(logits).any()

    def test_performance_benchmarking(self, nnmil_config: nnMILConfig):
        """Test performance benchmarking utilities."""
        # Create foundation model adapter
        adapter = FoundationModelAdapter(target_dim=nnmil_config.hidden_dim)

        # Run benchmark (small scale for testing)
        results = adapter.benchmark_projections(
            batch_size=2, num_patches=50, num_iterations=5, device="cpu"
        )

        # Validate benchmark results
        assert isinstance(results, dict)

        for dim_str, metrics in results.items():
            assert "model_name" in metrics
            assert "avg_time_ms" in metrics
            assert "throughput_bags_per_sec" in metrics
            assert metrics["avg_time_ms"] > 0
            assert metrics["throughput_bags_per_sec"] > 0

    def test_error_handling(self, nnmil_config: nnMILConfig):
        """Test error handling and edge cases."""
        # Test invalid bag dimensions
        with pytest.raises(ValueError):
            Bag(
                features=torch.randn(100),  # 1D instead of 2D
                label=0,
                num_patches=100,
                slide_id="test",
            )

        # Test invalid batch dimensions
        with pytest.raises(ValueError):
            TrainingBatch(
                features=torch.randn(4, 100),  # 2D instead of 3D
                labels=torch.tensor([0, 1, 0, 1]),
                masks=torch.ones(4, 100, dtype=torch.bool),
                num_patches=torch.tensor([100, 100, 100, 100]),
                slide_ids=["s1", "s2", "s3", "s4"],
            )

        # Test model with mismatched dimensions
        model = nnMIL(
            feature_dim=512,  # Different from config
            hidden_dim=nnmil_config.hidden_dim,
            num_classes=nnmil_config.num_classes,
        )

        # Should handle dimension mismatch gracefully with adapter
        features = torch.randn(2, 64, 1024)  # 1024-dim input
        masks = torch.ones(2, 64, dtype=torch.bool)

        # This should work with foundation model adapter
        adapter = FoundationModelAdapter(target_dim=512)
        adapted_features = adapter(features)

        with torch.no_grad():
            logits = model(adapted_features, masks)

        assert logits.shape == (2, nnmil_config.num_classes)


# Additional integration test for complete workflow
def test_complete_nnmil_workflow():
    """Test complete nnMIL workflow from config to inference."""
    # 1. Create configuration
    config = nnMILConfig(
        feature_dim=1024,
        hidden_dim=256,
        num_classes=3,  # Multi-class
        batch_size=8,
        num_epochs=2,
        bag_length=128,
        task_type="classification",
        enable_uncertainty=True,
    )

    # 2. Create synthetic multi-class dataset
    bags = []
    for i in range(24):  # 24 bags for 3 classes
        num_patches = torch.randint(80, 200, (1,)).item()
        features = torch.randn(num_patches, 1024)
        label = i % 3  # 3 classes

        bag = Bag(
            features=features, label=label, num_patches=num_patches, slide_id=f"slide_{i:03d}"
        )
        bags.append(bag)

    # 3. Create model with foundation adapter
    adapter = FoundationModelAdapter(target_dim=config.hidden_dim)
    model = nnMIL(
        feature_dim=config.hidden_dim,  # Use adapted dimension
        hidden_dim=config.hidden_dim,
        num_classes=config.num_classes,
        dropout=config.dropout,
    )

    # 4. Create training pipeline
    trainer = nnMILTrainer(model, config, device="cpu")

    # 5. Prepare data
    bag_sampler = FixedLengthBagSampler(config.bag_length, mode="train")

    features_list = []
    labels_list = []

    for bag in bags:
        # Adapt features first
        adapted_features = adapter(bag.features)

        # Then sample fixed-length bag
        sampled_features, mask = bag_sampler.sample_bag(adapted_features, bag.num_patches)
        features_list.append(sampled_features)
        labels_list.append(torch.tensor(bag.label, dtype=torch.long))

    # Create dataloader
    features_tensor = torch.stack(features_list)
    labels_tensor = torch.stack(labels_list)
    dataset = TensorDataset(features_tensor, labels_tensor)

    train_loader = DataLoader(dataset[:20], batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(dataset[20:], batch_size=config.batch_size, shuffle=False)

    # 6. Train model
    with tempfile.TemporaryDirectory() as temp_dir:
        history = trainer.train(
            train_loader=train_loader, val_loader=val_loader, checkpoint_dir=Path(temp_dir)
        )

        # 7. Test inference with uncertainty
        estimator = UncertaintyEstimator(model, num_samples=10, device="cpu")

        test_bag = bags[0]
        adapted_test_features = adapter(test_bag.features)

        # Create test bag with adapted features
        adapted_bag = Bag(
            features=adapted_test_features,
            label=test_bag.label,
            num_patches=test_bag.num_patches,
            slide_id=test_bag.slide_id,
        )

        output = estimator(adapted_bag)

        # 8. Validate complete workflow
        assert isinstance(output, InferenceOutput)
        assert output.logits.shape == (config.num_classes,)
        assert output.probabilities.shape == (config.num_classes,)
        assert torch.allclose(output.probabilities.sum(), torch.tensor(1.0), atol=1e-6)
        assert output.epistemic_uncertainty >= 0
        assert output.aleatoric_uncertainty >= 0
        assert output.total_uncertainty >= 0

        # Check training worked
        assert len(history["train_loss"]) == config.num_epochs
        assert history["train_loss"][-1] < history["train_loss"][0]  # Loss decreased


if __name__ == "__main__":
    # Run tests if executed directly
    pytest.main([__file__, "-v"])
