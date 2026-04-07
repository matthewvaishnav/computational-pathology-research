"""
Regression tests for the no-validation training path fix.

Verifies that training actually occurs when validation data is absent,
instead of the previous bug where the training loop was just `pass`.
"""

import sys
from pathlib import Path

import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.models import ClassificationHead, MultimodalFusionModel
from src.training import SupervisedTrainer


class TestNoValidationTraining:
    """Test that training works without validation data."""

    def _create_dummy_multimodal_dataloader(self, batch_size=4, num_samples=16):
        """Create a dummy multimodal dataloader for testing."""
        # Create dummy multimodal data
        data = []
        for i in range(num_samples):
            sample = {
                "wsi_features": torch.randn(10, 1024),  # 10 patches, 1024 dim
                "wsi_mask": torch.ones(10, dtype=torch.bool),
                "genomic": torch.randn(2000),
                "clinical_text": torch.randint(0, 30000, (128,)),
                "clinical_mask": torch.ones(128, dtype=torch.bool),
                "label": torch.tensor(i % 2),  # Binary classification
            }
            data.append(sample)

        # Simple collate function
        def collate_fn(batch):
            return {
                "wsi_features": torch.stack([b["wsi_features"] for b in batch]),
                "wsi_mask": torch.stack([b["wsi_mask"] for b in batch]),
                "genomic": torch.stack([b["genomic"] for b in batch]),
                "clinical_text": torch.stack([b["clinical_text"] for b in batch]),
                "clinical_mask": torch.stack([b["clinical_mask"] for b in batch]),
                "label": torch.stack([b["label"] for b in batch]),
            }

        from torch.utils.data import DataLoader as TorchDataLoader

        return TorchDataLoader(data, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)

    def test_trainer_fit_accepts_none_val_loader(self):
        """Verify trainer.fit() accepts val_loader=None."""
        model = MultimodalFusionModel(embed_dim=64)
        task_head = ClassificationHead(input_dim=64, num_classes=2)

        trainer = SupervisedTrainer(
            model=model,
            task_head=task_head,
            num_classes=2,
            device="cpu",
            learning_rate=1e-3,
            checkpoint_dir=None,  # No checkpointing needed for test
            log_dir=None,
        )

        train_loader = self._create_dummy_multimodal_dataloader()

        # Should not raise error when val_loader is None
        history = trainer.fit(
            train_loader=train_loader,
            num_epochs=2,
            val_loader=None,
        )

        # Verify training history was populated
        assert len(history["train_loss"]) == 2
        assert len(history["train_acc"]) == 2
        assert len(history["val_loss"]) == 2  # Filled with 0.0 when no validation
        assert len(history["val_acc"]) == 2

    def test_training_runs_without_validation(self):
        """Verify that actual training happens when val_loader=None."""
        model = MultimodalFusionModel(embed_dim=64)
        task_head = ClassificationHead(input_dim=64, num_classes=2)

        trainer = SupervisedTrainer(
            model=model,
            task_head=task_head,
            num_classes=2,
            device="cpu",
            learning_rate=1e-3,
            checkpoint_dir=None,
            log_dir=None,
        )

        train_loader = self._create_dummy_multimodal_dataloader()

        # Get initial model parameters to compare later
        initial_param = next(trainer.model.parameters()).clone().detach()

        # Train without validation
        history = trainer.fit(
            train_loader=train_loader,
            num_epochs=2,
            val_loader=None,
        )

        # Get final model parameters
        final_param = next(trainer.model.parameters()).clone().detach()

        # Verify model parameters changed (training actually happened)
        assert not torch.equal(
            initial_param, final_param
        ), "Model parameters did not change - training did not occur!"

        # Verify training loss was computed
        assert all(loss > 0 for loss in history["train_loss"]), "Training loss should be positive"

        # Verify validation metrics are 0.0 (placeholder values)
        assert all(
            v == 0.0 for v in history["val_loss"]
        ), "Val loss should be 0.0 when no validation"
        assert all(v == 0.0 for v in history["val_acc"]), "Val acc should be 0.0 when no validation"

    def test_training_with_validation_still_works(self):
        """Verify training with validation still works after the fix."""
        model = MultimodalFusionModel(embed_dim=64)
        task_head = ClassificationHead(input_dim=64, num_classes=2)

        trainer = SupervisedTrainer(
            model=model,
            task_head=task_head,
            num_classes=2,
            device="cpu",
            learning_rate=1e-3,
            checkpoint_dir=None,
            log_dir=None,
        )

        train_loader = self._create_dummy_multimodal_dataloader()
        val_loader = self._create_dummy_multimodal_dataloader()

        # Get initial model parameters
        initial_param = next(trainer.model.parameters()).clone().detach()

        # Train with validation
        history = trainer.fit(
            train_loader=train_loader,
            num_epochs=2,
            val_loader=val_loader,
        )

        # Get final model parameters
        final_param = next(trainer.model.parameters()).clone().detach()

        # Verify model parameters changed (training actually happened)
        assert not torch.equal(initial_param, final_param), "Model parameters did not change!"

        # Verify validation metrics are actual computed values (not 0.0 placeholders)
        assert any(
            v > 0 for v in history["val_loss"]
        ), "Val loss should be computed when validation is used"
        assert any(
            v > 0 for v in history["val_acc"]
        ), "Val acc should be computed when validation is used"

    def test_no_validation_disables_early_stopping(self):
        """Verify that early stopping is disabled when no validation."""
        model = MultimodalFusionModel(embed_dim=64)
        task_head = ClassificationHead(input_dim=64, num_classes=2)

        trainer = SupervisedTrainer(
            model=model,
            task_head=task_head,
            num_classes=2,
            device="cpu",
            learning_rate=1e-3,
            early_stopping_patience=1,  # Very short patience
            checkpoint_dir=None,
            log_dir=None,
        )

        train_loader = self._create_dummy_multimodal_dataloader()

        # Train without validation - should complete all epochs (no early stopping)
        history = trainer.fit(
            train_loader=train_loader,
            num_epochs=5,
            val_loader=None,
        )

        # Should complete all 5 epochs (no early stopping without validation)
        assert (
            len(history["train_loss"]) == 5
        ), f"Expected 5 epochs, got {len(history['train_loss'])}"

    def test_fit_signature_requires_num_epochs_before_val_loader(self):
        """Verify fit() signature: num_epochs is required, val_loader is optional."""
        train_loader = self._create_dummy_multimodal_dataloader()
        val_loader = self._create_dummy_multimodal_dataloader()

        # Test 1: num_epochs as positional arg, val_loader omitted (no validation)
        model1 = MultimodalFusionModel(embed_dim=64)
        task_head1 = ClassificationHead(input_dim=64, num_classes=2)
        trainer1 = SupervisedTrainer(
            model=model1,
            task_head=task_head1,
            num_classes=2,
            device="cpu",
            checkpoint_dir=None,
            log_dir=None,
        )
        history1 = trainer1.fit(train_loader, 1, val_loader=None)
        assert (
            len(history1["train_loss"]) == 1
        ), f"Expected 1 epoch, got {len(history1['train_loss'])}"

        # Test 2: all keyword args (new recommended pattern)
        model2 = MultimodalFusionModel(embed_dim=64)
        task_head2 = ClassificationHead(input_dim=64, num_classes=2)
        trainer2 = SupervisedTrainer(
            model=model2,
            task_head=task_head2,
            num_classes=2,
            device="cpu",
            checkpoint_dir=None,
            log_dir=None,
        )
        history2 = trainer2.fit(
            train_loader=train_loader,
            num_epochs=1,
            val_loader=None,
        )
        assert len(history2["train_loss"]) == 1

        # Test 3: with validation provided
        model3 = MultimodalFusionModel(embed_dim=64)
        task_head3 = ClassificationHead(input_dim=64, num_classes=2)
        trainer3 = SupervisedTrainer(
            model=model3,
            task_head=task_head3,
            num_classes=2,
            device="cpu",
            checkpoint_dir=None,
            log_dir=None,
        )
        history3 = trainer3.fit(train_loader, 1, val_loader=val_loader)
        assert len(history3["train_loss"]) == 1
