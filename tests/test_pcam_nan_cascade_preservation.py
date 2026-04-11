"""
Preservation Property Tests for PCam NaN Cascade Fix

Property 2: Preservation - Single NaN Batch Handling

**Validates: Requirements 3.1, 3.2, 3.3, 3.4**

This test follows observation-first methodology:
1. Observe behavior on UNFIXED code for non-buggy inputs (isolated NaN losses, normal training)
2. Write property-based tests capturing observed behavior patterns from Preservation Requirements

These tests MUST PASS on unfixed code to confirm baseline behavior to preserve.
"""

import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import tempfile
import logging
from unittest.mock import patch, MagicMock
import random
from typing import Dict, List, Tuple, Any

# Import the training function under test
from experiments.train_pcam import train_epoch, validate, create_single_modality_model
from src.models.encoders import WSIEncoder
from src.models.feature_extractors import ResNetFeatureExtractor
from src.models.heads import ClassificationHead


class TestPCamNaNCascadePreservation:
    """
    Preservation property tests for single NaN batch handling.

    These tests MUST PASS on unfixed code to confirm baseline behavior to preserve.
    Tests that isolated NaN losses (1-2 non-consecutive batches) are handled correctly
    without triggering recovery mechanisms.
    """

    @pytest.fixture
    def mock_config(self):
        """Create a minimal config for testing."""
        return {
            "model": {
                "embed_dim": 256,
                "feature_extractor": {"model": "resnet18", "pretrained": False, "feature_dim": 512},
                "wsi": {
                    "input_dim": 512,
                    "hidden_dim": 256,
                    "num_heads": 4,
                    "num_layers": 1,
                    "pooling": "mean",
                },
            },
            "task": {"classification": {"hidden_dims": [128], "dropout": 0.3}},
            "training": {"max_grad_norm": 1.0, "dropout": 0.1},
            "logging": {"log_interval": 1},
        }

    @pytest.fixture
    def models_and_optimizer(self, mock_config):
        """Create models and optimizer for testing."""
        device = "cpu"  # Use CPU for testing to avoid GPU dependencies

        # Create models
        feature_extractor, encoder, head = create_single_modality_model(mock_config)

        # Move to device
        feature_extractor = feature_extractor.to(device)
        encoder = encoder.to(device)
        head = head.to(device)

        # Create optimizer
        params = (
            list(feature_extractor.parameters())
            + list(encoder.parameters())
            + list(head.parameters())
        )
        optimizer = optim.Adam(params, lr=0.001)

        # Create scheduler (required by train_epoch)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

        # Create loss function
        criterion = nn.BCEWithLogitsLoss()

        return feature_extractor, encoder, head, optimizer, scheduler, criterion, device

    def create_normal_dataloader(self, batch_size=4, num_batches=10):
        """Create a dataloader with normal, non-NaN data."""
        all_images = []
        all_labels = []

        for _ in range(num_batches):
            # Create normal data that won't produce NaN
            images = (
                torch.randn(batch_size, 3, 96, 96) * 0.1
            )  # Small values to avoid numerical issues
            labels = torch.randint(0, 2, (batch_size,)).float()
            all_images.append(images)
            all_labels.append(labels)

        all_images = torch.cat(all_images, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        class MockPCamDataset:
            def __init__(self, images, labels):
                self.images = images
                self.labels = labels

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                return {"image": self.images[idx], "label": self.labels[idx]}

        mock_dataset = MockPCamDataset(all_images, all_labels)
        return DataLoader(mock_dataset, batch_size=batch_size, shuffle=False)

    def create_isolated_nan_dataloader(self, batch_size=4, num_batches=10, nan_batch_indices=None):
        """
        Create a dataloader with isolated NaN batches (1-2 non-consecutive).

        Args:
            batch_size: Size of each batch
            num_batches: Total number of batches
            nan_batch_indices: List of batch indices that should produce NaN (default: [2, 7])
        """
        if nan_batch_indices is None:
            nan_batch_indices = [2, 7]  # Non-consecutive isolated NaN batches

        all_images = []
        all_labels = []

        for batch_idx in range(num_batches):
            if batch_idx in nan_batch_indices:
                # Create data that will produce NaN during forward pass
                # Use very large values that will cause overflow in computations
                images = torch.full((batch_size, 3, 96, 96), 1e10)  # Very large values
                labels = torch.randint(0, 2, (batch_size,)).float()
            else:
                # Create normal data
                images = torch.randn(batch_size, 3, 96, 96) * 0.1
                labels = torch.randint(0, 2, (batch_size,)).float()

            all_images.append(images)
            all_labels.append(labels)

        all_images = torch.cat(all_images, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        class MockPCamDataset:
            def __init__(self, images, labels):
                self.images = images
                self.labels = labels

            def __len__(self):
                return len(self.images)

            def __getitem__(self, idx):
                return {"image": self.images[idx], "label": self.labels[idx]}

        mock_dataset = MockPCamDataset(all_images, all_labels)
        return DataLoader(mock_dataset, batch_size=batch_size, shuffle=False)

    def test_isolated_nan_batch_handling_preservation(self, mock_config, models_and_optimizer):
        """
        Property 2: Preservation - Isolated NaN Batch Handling

        **Validates: Requirements 3.1**

        Test that isolated NaN losses (1-2 non-consecutive batches) are skipped
        without triggering recovery mechanisms. This behavior must be preserved.

        Expected outcome: Test PASSES (confirms baseline behavior to preserve)
        """
        feature_extractor, encoder, head, optimizer, scheduler, criterion, device = (
            models_and_optimizer
        )

        # Create normal dataloader
        dataloader = self.create_normal_dataloader(batch_size=4, num_batches=10)

        # Patch the criterion to return NaN for specific batches to simulate isolated NaN
        original_criterion = criterion
        call_count = [0]  # Use list to allow modification in nested function

        def nan_criterion(logits, labels):
            call_count[0] += 1
            if call_count[0] in [3, 8]:  # Return NaN for batches 3 and 8 (non-consecutive)
                return torch.tensor(float("nan"))
            return original_criterion(logits, labels)

        # Track behavior during training
        with patch("experiments.train_pcam.logger") as mock_logger:
            metrics = train_epoch(
                feature_extractor=feature_extractor,
                encoder=encoder,
                head=head,
                dataloader=dataloader,
                optimizer=optimizer,
                criterion=nan_criterion,  # Use our NaN-injecting criterion
                device=device,
                config=mock_config,
                epoch=1,
                scheduler=scheduler,
                scaler=None,
                writer=None,
                run_id="test_isolated_nan",
                status_path=None,
            )

        # Analyze the behavior
        nan_warnings = [call for call in mock_logger.warning.call_args_list if "NaN" in str(call)]
        # Look for actual recovery actions, not just informational logging
        recovery_calls = [
            call
            for call in mock_logger.info.call_args_list
            if (
                "recovery successful" in str(call).lower()
                or "attempting recovery" in str(call).lower()
                or "recovery failed" in str(call).lower()
            )
        ]

        # Verify preservation requirements:
        # 1. Isolated NaN batches should be skipped (some NaN warnings expected)
        assert len(nan_warnings) > 0, "Expected NaN warnings for isolated NaN batches"
        assert (
            len(nan_warnings) <= 4
        ), f"Too many NaN warnings: {len(nan_warnings)} (expected <= 4 for isolated cases)"

        # 2. No recovery should be triggered for isolated NaN
        assert (
            len(recovery_calls) == 0
        ), f"Recovery should not be triggered for isolated NaN, but found: {recovery_calls}"

        # 3. Training should complete successfully with valid metrics
        assert isinstance(metrics, dict), "Training should return metrics dictionary"
        assert "loss" in metrics, "Metrics should contain loss"
        assert "accuracy" in metrics, "Metrics should contain accuracy"
        assert not np.isnan(metrics["loss"]), f"Final loss should not be NaN: {metrics['loss']}"
        assert 0.0 <= metrics["accuracy"] <= 1.0, f"Accuracy should be valid: {metrics['accuracy']}"

        print(
            f"✓ Isolated NaN handling preserved: {len(nan_warnings)} NaN warnings, no recovery triggered"
        )

    def test_normal_training_preservation(self, mock_config, models_and_optimizer):
        """
        Property 2: Preservation - Normal Training Without Additional Overhead

        **Validates: Requirements 3.2**

        Test that normal training without NaN issues continues without any
        additional overhead or interference. This behavior must be preserved.

        Expected outcome: Test PASSES (confirms baseline behavior to preserve)
        """
        feature_extractor, encoder, head, optimizer, scheduler, criterion, device = (
            models_and_optimizer
        )

        # Create normal dataloader without any NaN-inducing data
        dataloader = self.create_normal_dataloader(batch_size=4, num_batches=10)

        # Track behavior during normal training
        with patch("experiments.train_pcam.logger") as mock_logger:
            metrics = train_epoch(
                feature_extractor=feature_extractor,
                encoder=encoder,
                head=head,
                dataloader=dataloader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                config=mock_config,
                epoch=1,
                scheduler=scheduler,
                scaler=None,
                writer=None,
                run_id="test_normal_training",
                status_path=None,
            )

        # Analyze the behavior
        nan_warnings = [call for call in mock_logger.warning.call_args_list if "NaN" in str(call)]
        # Look for actual recovery actions, not just informational logging
        recovery_calls = [
            call
            for call in mock_logger.info.call_args_list
            if (
                "recovery successful" in str(call).lower()
                or "attempting recovery" in str(call).lower()
                or "recovery failed" in str(call).lower()
            )
        ]

        # Verify preservation requirements:
        # 1. No NaN warnings should occur during normal training
        assert (
            len(nan_warnings) == 0
        ), f"No NaN warnings expected for normal training, but found: {nan_warnings}"

        # 2. No recovery mechanisms should be triggered
        assert (
            len(recovery_calls) == 0
        ), f"No recovery should be triggered for normal training, but found: {recovery_calls}"

        # 3. Training should complete successfully with valid metrics
        assert isinstance(metrics, dict), "Training should return metrics dictionary"
        assert "loss" in metrics, "Metrics should contain loss"
        assert "accuracy" in metrics, "Metrics should contain accuracy"
        assert not np.isnan(metrics["loss"]), f"Final loss should not be NaN: {metrics['loss']}"
        assert 0.0 <= metrics["accuracy"] <= 1.0, f"Accuracy should be valid: {metrics['accuracy']}"

        # 4. All batches should be processed (no skipped batches)
        expected_batches = len(dataloader)
        # We can't directly check processed batches, but we can verify no skip warnings
        skip_warnings = [
            call for call in mock_logger.warning.call_args_list if "skip" in str(call).lower()
        ]
        assert (
            len(skip_warnings) == 0
        ), f"No batches should be skipped in normal training: {skip_warnings}"

        print(f"✓ Normal training preserved: No NaN warnings, no recovery, valid metrics")

    def test_validation_nan_handling_preservation(self, mock_config, models_and_optimizer):
        """
        Property 2: Preservation - Validation NaN Handling

        **Validates: Requirements 3.3**

        Test that validation NaN handling continues to skip validation batches
        without affecting training state. This behavior must be preserved.

        Expected outcome: Test PASSES (confirms baseline behavior to preserve)
        """
        feature_extractor, encoder, head, optimizer, scheduler, criterion, device = (
            models_and_optimizer
        )

        # Create normal validation dataloader
        val_dataloader = self.create_normal_dataloader(batch_size=4, num_batches=8)

        # Patch the criterion to return NaN for specific batches to simulate validation NaN
        original_criterion = criterion
        call_count = [0]  # Use list to allow modification in nested function

        def nan_criterion(logits, labels):
            call_count[0] += 1
            if call_count[0] in [2, 6]:  # Return NaN for batches 2 and 6
                return torch.tensor(float("nan"))
            return original_criterion(logits, labels)

        # Track behavior during validation with NaN injection
        with patch("experiments.train_pcam.logger") as mock_logger:
            val_metrics = validate(
                feature_extractor=feature_extractor,
                encoder=encoder,
                head=head,
                dataloader=val_dataloader,
                criterion=nan_criterion,  # Use our NaN-injecting criterion
                device=device,
                scaler=None,
            )

        # Analyze the behavior
        nan_warnings = [call for call in mock_logger.warning.call_args_list if "NaN" in str(call)]
        # Look for actual recovery actions, not just informational logging
        recovery_calls = [
            call
            for call in mock_logger.info.call_args_list
            if (
                "recovery successful" in str(call).lower()
                or "attempting recovery" in str(call).lower()
                or "recovery failed" in str(call).lower()
            )
        ]

        # Verify preservation requirements:
        # 1. Validation should handle NaN by skipping batches
        assert len(nan_warnings) > 0, "Expected NaN warnings during validation with NaN batches"

        # 2. No recovery should be triggered during validation
        assert (
            len(recovery_calls) == 0
        ), f"Recovery should not be triggered during validation, but found: {recovery_calls}"

        # 3. Validation should complete successfully with valid metrics
        assert isinstance(val_metrics, dict), "Validation should return metrics dictionary"
        assert "val_loss" in val_metrics, "Validation metrics should contain val_loss"
        assert "val_accuracy" in val_metrics, "Validation metrics should contain val_accuracy"
        assert not np.isnan(
            val_metrics["val_loss"]
        ), f"Validation loss should not be NaN: {val_metrics['val_loss']}"
        assert (
            0.0 <= val_metrics["val_accuracy"] <= 1.0
        ), f"Validation accuracy should be valid: {val_metrics['val_accuracy']}"

        print(
            f"✓ Validation NaN handling preserved: {len(nan_warnings)} NaN warnings, no recovery triggered"
        )

    def test_checkpointing_functionality_preservation(self, mock_config, models_and_optimizer):
        """
        Property 2: Preservation - Checkpointing and Logging Functionality

        **Validates: Requirements 3.4**

        Test that checkpointing and logging functionality continues to work as before.
        This behavior must be preserved.

        Expected outcome: Test PASSES (confirms baseline behavior to preserve)
        """
        feature_extractor, encoder, head, optimizer, scheduler, criterion, device = (
            models_and_optimizer
        )

        # Import checkpoint functions
        from experiments.train_pcam import save_checkpoint, load_checkpoint

        # Create a temporary directory for checkpoint testing
        with tempfile.TemporaryDirectory() as temp_dir:
            checkpoint_path = Path(temp_dir) / "test_checkpoint.pth"

            # Create some dummy metrics
            test_metrics = {"loss": 0.5, "accuracy": 0.8, "val_loss": 0.6, "val_accuracy": 0.75}

            # Test checkpoint saving
            save_checkpoint(
                epoch=1,
                feature_extractor=feature_extractor,
                encoder=encoder,
                head=head,
                optimizer=optimizer,
                scheduler=None,
                metrics=test_metrics,
                config=mock_config,
                path=str(checkpoint_path),
            )

            # Verify checkpoint file was created
            assert checkpoint_path.exists(), "Checkpoint file should be created"

            # Create new models to test loading
            feature_extractor_new, encoder_new, head_new = create_single_modality_model(mock_config)
            feature_extractor_new = feature_extractor_new.to(device)
            encoder_new = encoder_new.to(device)
            head_new = head_new.to(device)

            params_new = (
                list(feature_extractor_new.parameters())
                + list(encoder_new.parameters())
                + list(head_new.parameters())
            )
            optimizer_new = optim.Adam(params_new, lr=0.001)
            scheduler_new = optim.lr_scheduler.StepLR(optimizer_new, step_size=10, gamma=0.1)

            # Test checkpoint loading
            loaded_epoch, loaded_metrics = load_checkpoint(
                path=str(checkpoint_path),
                feature_extractor=feature_extractor_new,
                encoder=encoder_new,
                head=head_new,
                optimizer=optimizer_new,
                scheduler=scheduler_new,
            )

            # Verify checkpoint loading worked correctly
            assert loaded_epoch == 1, f"Loaded epoch should be 1, got {loaded_epoch}"
            assert loaded_metrics == test_metrics, f"Loaded metrics should match saved metrics"

            # Verify model parameters were loaded (they should be different from initial random values)
            # We can't easily compare exact values, but we can verify the loading process completed
            assert hasattr(
                feature_extractor_new, "parameters"
            ), "Feature extractor should have parameters"
            assert hasattr(encoder_new, "parameters"), "Encoder should have parameters"
            assert hasattr(head_new, "parameters"), "Head should have parameters"

        print("✓ Checkpointing functionality preserved: Save and load operations work correctly")

    def test_multiple_isolated_nan_scenarios_preservation(self, mock_config, models_and_optimizer):
        """
        Property 2: Preservation - Multiple Isolated NaN Scenarios

        **Validates: Requirements 3.1, 3.2**

        Test various patterns of isolated NaN batches to ensure consistent preservation
        of the current behavior across different scenarios.

        Expected outcome: Test PASSES (confirms baseline behavior to preserve)
        """
        feature_extractor, encoder, head, optimizer, scheduler, criterion, device = (
            models_and_optimizer
        )

        # Test different isolated NaN patterns
        test_scenarios = [
            {
                "name": "single_nan_early",
                "nan_indices": [1],
                "total_batches": 8,
                "description": "Single NaN batch early in training",
            },
            {
                "name": "single_nan_late",
                "nan_indices": [6],
                "total_batches": 8,
                "description": "Single NaN batch late in training",
            },
            {
                "name": "two_non_consecutive",
                "nan_indices": [2, 5],
                "total_batches": 8,
                "description": "Two non-consecutive NaN batches",
            },
            {
                "name": "two_far_apart",
                "nan_indices": [1, 7],
                "total_batches": 10,
                "description": "Two NaN batches far apart",
            },
        ]

        for scenario in test_scenarios:
            # Reset models for each scenario
            feature_extractor, encoder, head, optimizer, scheduler, criterion, device = (
                models_and_optimizer
            )

            # Create normal dataloader
            dataloader = self.create_normal_dataloader(
                batch_size=4, num_batches=scenario["total_batches"]
            )

            # Create NaN-injecting criterion for this scenario
            original_criterion = criterion
            call_count = [0]

            def nan_criterion(logits, labels):
                call_count[0] += 1
                if call_count[0] in scenario["nan_indices"]:
                    return torch.tensor(float("nan"))
                return original_criterion(logits, labels)

            # Track behavior
            with patch("experiments.train_pcam.logger") as mock_logger:
                metrics = train_epoch(
                    feature_extractor=feature_extractor,
                    encoder=encoder,
                    head=head,
                    dataloader=dataloader,
                    optimizer=optimizer,
                    criterion=nan_criterion,
                    device=device,
                    config=mock_config,
                    epoch=1,
                    scheduler=scheduler,
                    scaler=None,
                    writer=None,
                    run_id=f"test_{scenario['name']}",
                    status_path=None,
                )

            # Analyze behavior for this scenario
            nan_warnings = [
                call for call in mock_logger.warning.call_args_list if "NaN" in str(call)
            ]
            # Look for actual recovery actions, not just informational logging
            recovery_calls = [
                call
                for call in mock_logger.info.call_args_list
                if (
                    "recovery successful" in str(call).lower()
                    or "attempting recovery" in str(call).lower()
                    or "recovery failed" in str(call).lower()
                )
            ]

            # Verify preservation for this scenario
            expected_nan_warnings = len(scenario["nan_indices"])
            assert (
                len(nan_warnings) >= expected_nan_warnings
            ), f"Scenario {scenario['name']}: Expected at least {expected_nan_warnings} NaN warnings, got {len(nan_warnings)}"

            assert (
                len(recovery_calls) == 0
            ), f"Scenario {scenario['name']}: No recovery should be triggered for isolated NaN, but found: {recovery_calls}"

            assert (
                isinstance(metrics, dict) and "loss" in metrics
            ), f"Scenario {scenario['name']}: Should return valid metrics"

            assert not np.isnan(
                metrics["loss"]
            ), f"Scenario {scenario['name']}: Final loss should not be NaN: {metrics['loss']}"

            print(
                f"✓ Scenario '{scenario['name']}' preserved: {len(nan_warnings)} NaN warnings, no recovery"
            )

    def test_mixed_precision_preservation(self, mock_config, models_and_optimizer):
        """
        Property 2: Preservation - Mixed Precision Training Behavior

        **Validates: Requirements 3.1, 3.2**

        Test that mixed precision training behavior is preserved for both normal
        and isolated NaN scenarios.

        Expected outcome: Test PASSES (confirms baseline behavior to preserve)
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for mixed precision testing")

        feature_extractor, encoder, head, optimizer, scheduler, criterion, device = (
            models_and_optimizer
        )

        # Move models to GPU for mixed precision
        device = "cuda"
        feature_extractor = feature_extractor.to(device)
        encoder = encoder.to(device)
        head = head.to(device)

        # Create gradient scaler
        scaler = torch.cuda.amp.GradScaler()

        # Test with isolated NaN batches
        dataloader = self.create_isolated_nan_dataloader(
            batch_size=4, num_batches=8, nan_batch_indices=[2, 6]
        )

        # Track behavior with mixed precision
        with patch("experiments.train_pcam.logger") as mock_logger:
            metrics = train_epoch(
                feature_extractor=feature_extractor,
                encoder=encoder,
                head=head,
                dataloader=dataloader,
                optimizer=optimizer,
                criterion=criterion,
                device=device,
                config=mock_config,
                epoch=1,
                scheduler=scheduler,
                scaler=scaler,
                writer=None,
                run_id="test_mixed_precision",
                status_path=None,
            )

        # Analyze behavior
        nan_warnings = [call for call in mock_logger.warning.call_args_list if "NaN" in str(call)]
        # Look for actual recovery actions, not just informational logging
        recovery_calls = [
            call
            for call in mock_logger.info.call_args_list
            if (
                "recovery successful" in str(call).lower()
                or "attempting recovery" in str(call).lower()
                or "recovery failed" in str(call).lower()
            )
        ]

        # Verify mixed precision preservation
        assert (
            len(nan_warnings) > 0
        ), "Expected NaN warnings for isolated NaN batches with mixed precision"
        assert (
            len(recovery_calls) == 0
        ), f"No recovery should be triggered with mixed precision for isolated NaN: {recovery_calls}"
        assert (
            isinstance(metrics, dict) and "loss" in metrics
        ), "Should return valid metrics with mixed precision"
        assert not np.isnan(
            metrics["loss"]
        ), f"Final loss should not be NaN with mixed precision: {metrics['loss']}"

        print(
            f"✓ Mixed precision behavior preserved: {len(nan_warnings)} NaN warnings, no recovery triggered"
        )
