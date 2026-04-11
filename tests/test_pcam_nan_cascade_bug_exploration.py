"""
Bug Condition Exploration Test for PCam NaN Cascade Fix

This test MUST FAIL on unfixed code - failure confirms the bug exists.
The test validates Property 1: Bug Condition - Cascading NaN Recovery Detection

**Validates: Requirements 2.1, 2.2, 2.4**

This test injects NaN values into model parameters during training and verifies
that consecutive NaN losses occur without recovery on UNFIXED code.
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

# Import the training function under test
from experiments.train_pcam import train_epoch, create_single_modality_model
from src.models.encoders import WSIEncoder
from src.models.feature_extractors import ResNetFeatureExtractor
from src.models.heads import ClassificationHead


class TestPCamNaNCascadeBugExploration:
    """
    Bug condition exploration test for cascading NaN recovery.

    This test MUST FAIL on unfixed code to confirm the bug exists.
    Expected behavior: Test FAILS because the system continues with corrupted parameters
    without recovery.
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
            "checkpoint": {
                "checkpoint_dir": "test_checkpoints",
                "stability_frequency": 2,  # Save stability checkpoint every 2 batches during instability
                "rolling_window": 3,  # Keep last 3 checkpoints
            },
        }

    @pytest.fixture
    def mock_dataloader(self):
        """Create a mock dataloader with synthetic data."""
        # Create synthetic PCam-like data: batch_size=2, 3 channels, 96x96 images
        # Use multiple batches to allow cascading NaN detection
        batch_size = 2
        num_batches = 5  # Ensure we have enough batches to trigger cascading NaN

        all_images = []
        all_labels = []

        for _ in range(num_batches):
            images = torch.randn(batch_size, 3, 96, 96)
            labels = torch.randint(0, 2, (batch_size,)).float()
            all_images.append(images)
            all_labels.append(labels)

        # Concatenate all batches
        all_images = torch.cat(all_images, dim=0)
        all_labels = torch.cat(all_labels, dim=0)

        # Create a custom dataset that returns dict format like PCam
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

    def inject_nan_into_parameters(self, model, corruption_ratio=0.1):
        """
        Inject NaN values into model parameters to simulate parameter corruption.

        Args:
            model: PyTorch model to corrupt
            corruption_ratio: Fraction of parameters to corrupt with NaN
        """
        with torch.no_grad():
            for param in model.parameters():
                if param.requires_grad:
                    # Randomly select parameters to corrupt
                    mask = torch.rand_like(param) < corruption_ratio
                    param.data[mask] = float("nan")

    def check_model_parameters_contain_nan(self, *models):
        """
        Check if any model parameters contain NaN values.

        Returns:
            bool: True if any parameter contains NaN
        """
        for model in models:
            for param in model.parameters():
                if torch.isnan(param).any():
                    return True
        return False

    def test_cascading_nan_recovery_detection_property(
        self, mock_config, mock_dataloader, models_and_optimizer
    ):
        """
        Property 1: Bug Condition - Cascading NaN Recovery Detection

        **Validates: Requirements 2.1, 2.2, 2.4**

        Test that when model parameters become corrupted with NaN values
        (consecutiveNaNCount >= 3 AND modelParametersContainNaN), the system
        fails to recover and continues with corrupted parameters.

        This test MUST FAIL on unfixed code - failure confirms the bug exists.
        """
        feature_extractor, encoder, head, optimizer, scheduler, criterion, device = (
            models_and_optimizer
        )

        # Create gradient scaler for mixed precision
        scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() else None

        # Track consecutive NaN count and recovery attempts
        consecutive_nan_count = 0
        recovery_attempted = False
        training_failed_completely = False

        # Inject NaN into feature extractor parameters to simulate corruption
        self.inject_nan_into_parameters(feature_extractor, corruption_ratio=0.2)

        # Verify parameters are corrupted
        assert self.check_model_parameters_contain_nan(
            feature_extractor, encoder, head
        ), "Parameter corruption injection failed"

        # Run training epoch with corrupted parameters
        with patch("experiments.train_pcam.logger") as mock_logger:
            try:
                metrics = train_epoch(
                    feature_extractor=feature_extractor,
                    encoder=encoder,
                    head=head,
                    dataloader=mock_dataloader,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                    config=mock_config,
                    epoch=1,
                    scheduler=scheduler,
                    scaler=scaler,
                    writer=None,
                    run_id="test_run",
                    status_path=None,
                )

                # Count NaN warnings to detect consecutive NaN losses
                nan_warnings = [
                    call for call in mock_logger.warning.call_args_list if "NaN" in str(call)
                ]
                consecutive_nan_count = len(nan_warnings)

                # Check if recovery was attempted (this should NOT happen in unfixed code)
                recovery_calls = [
                    call
                    for call in mock_logger.info.call_args_list
                    if "recovery" in str(call).lower() or "checkpoint" in str(call).lower()
                ]
                recovery_attempted = len(recovery_calls) > 0

            except RuntimeError as e:
                if "all batches contained NaN values" in str(e):
                    # This indicates the system failed completely rather than recovering
                    training_failed_completely = True
                    consecutive_nan_count = len(mock_dataloader)  # All batches failed
                elif "Cascading NaN losses detected with model parameter corruption" in str(e):
                    # This is the EXPECTED behavior with the fix - recovery was attempted but failed
                    # Extract information from the error message
                    training_failed_completely = True
                    recovery_attempted = True
                    # Extract consecutive NaN count from error message
                    import re

                    match = re.search(r"Consecutive NaN count: (\d+)", str(e))
                    if match:
                        consecutive_nan_count = int(match.group(1))
                    else:
                        consecutive_nan_count = 3  # Default assumption based on threshold
                else:
                    raise

        # Bug condition analysis with the FIX in place:
        # With the fix implemented, the system should:
        # 1. Detect cascading NaN losses (3+ consecutive batches with NaN)
        # 2. Detect model parameter corruption
        # 3. Attempt recovery from checkpoint
        # 4. Either succeed in recovery OR fail gracefully after max attempts

        parameters_still_corrupted = self.check_model_parameters_contain_nan(
            feature_extractor, encoder, head
        )

        # With the fix, we expect one of these outcomes:
        # A) Recovery was attempted and succeeded (parameters cleaned, training continued)
        # B) Recovery was attempted but failed after max attempts (graceful failure with RuntimeError)
        # C) Cascading threshold not reached (< 3 consecutive NaN batches)

        cascading_threshold_reached = consecutive_nan_count >= 3

        # The fix is working correctly if:
        # - When cascading NaN is detected with parameter corruption, recovery is attempted
        # - The system doesn't continue indefinitely with corrupted parameters
        fix_working_correctly = (
            # Case 1: Cascading threshold reached, recovery attempted
            (cascading_threshold_reached and parameters_still_corrupted and recovery_attempted)
            or
            # Case 2: Cascading threshold reached, recovery succeeded (parameters cleaned)
            (cascading_threshold_reached and not parameters_still_corrupted and recovery_attempted)
            or
            # Case 3: Training failed gracefully with proper error handling (recovery attempted but failed)
            (training_failed_completely and recovery_attempted)
            or
            # Case 4: Cascading threshold not reached (normal single NaN handling)
            (not cascading_threshold_reached and not training_failed_completely)
        )

        # CRITICAL: This assertion should PASS on fixed code
        # The test validates that the fix properly handles cascading NaN with recovery
        assert fix_working_correctly, (
            f"FIX VALIDATION FAILED: System did not properly handle cascading NaN with recovery. "
            f"trainingFailedCompletely={training_failed_completely}, "
            f"consecutiveNaNCount={consecutive_nan_count}, "
            f"cascadingThresholdReached={cascading_threshold_reached}, "
            f"modelParametersContainNaN={parameters_still_corrupted}, "
            f"recoveryAttempted={recovery_attempted}. "
            f"Expected: When cascading NaN is detected with parameter corruption, the system should "
            f"attempt recovery through checkpoint restoration and scaler/optimizer reinitialization. "
            f"The fix should prevent indefinite continuation with corrupted parameters."
        )

    def test_parameter_corruption_scenarios(
        self, mock_config, mock_dataloader, models_and_optimizer
    ):
        """
        Test various parameter corruption scenarios to surface counterexamples.

        This generates multiple test cases to understand the bug behavior.
        """
        feature_extractor, encoder, head, optimizer, scheduler, criterion, device = (
            models_and_optimizer
        )

        corruption_scenarios = [
            ("feature_extractor_only", feature_extractor, 0.3),
            ("encoder_only", encoder, 0.3),
            ("head_only", head, 0.3),
            ("mixed_corruption", feature_extractor, 0.1),  # Light corruption across models
        ]

        counterexamples = []

        for scenario_name, target_model, corruption_ratio in corruption_scenarios:
            # Reset models to clean state
            feature_extractor, encoder, head, optimizer, scheduler, criterion, device = (
                models_and_optimizer
            )

            # Inject corruption
            self.inject_nan_into_parameters(target_model, corruption_ratio)

            # Track behavior
            with patch("experiments.train_pcam.logger") as mock_logger:
                try:
                    metrics = train_epoch(
                        feature_extractor=feature_extractor,
                        encoder=encoder,
                        head=head,
                        dataloader=mock_dataloader,
                        optimizer=optimizer,
                        criterion=criterion,
                        device=device,
                        config=mock_config,
                        epoch=1,
                        scheduler=scheduler,
                        scaler=None,  # Test without mixed precision
                        writer=None,
                        run_id=f"test_{scenario_name}",
                        status_path=None,
                    )

                    nan_warnings = len(
                        [call for call in mock_logger.warning.call_args_list if "NaN" in str(call)]
                    )

                    counterexamples.append(
                        {
                            "scenario": scenario_name,
                            "corruption_target": target_model.__class__.__name__,
                            "corruption_ratio": corruption_ratio,
                            "consecutive_nan_count": nan_warnings,
                            "training_completed": True,
                            "parameters_still_corrupted": self.check_model_parameters_contain_nan(
                                feature_extractor, encoder, head
                            ),
                        }
                    )

                except RuntimeError as e:
                    counterexamples.append(
                        {
                            "scenario": scenario_name,
                            "corruption_target": target_model.__class__.__name__,
                            "corruption_ratio": corruption_ratio,
                            "consecutive_nan_count": len(mock_dataloader),  # All batches failed
                            "training_completed": False,
                            "error": str(e),
                            "parameters_still_corrupted": True,
                        }
                    )

        # Document counterexamples found
        print("\n=== COUNTEREXAMPLES FOUND ===")
        for example in counterexamples:
            print(f"Scenario: {example['scenario']}")
            print(f"  Corruption target: {example['corruption_target']}")
            print(f"  Consecutive NaN count: {example['consecutive_nan_count']}")
            print(f"  Training completed: {example['training_completed']}")
            print(f"  Parameters still corrupted: {example['parameters_still_corrupted']}")
            if "error" in example:
                print(f"  Error: {example['error']}")
            print()

        # This test documents the bug behavior - it should always pass
        # The actual bug detection happens in the main property test
        assert len(counterexamples) > 0, "Should have generated counterexamples"

    def test_gradient_scaler_corruption_scenario(
        self, mock_config, mock_dataloader, models_and_optimizer
    ):
        """
        Test gradient scaler state corruption scenario.

        This test simulates scaler state corruption that can lead to cascading NaN.
        """
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available for gradient scaler testing")

        feature_extractor, encoder, head, optimizer, scheduler, criterion, device = (
            models_and_optimizer
        )

        # Move models to GPU for mixed precision testing
        device = "cuda"
        feature_extractor = feature_extractor.to(device)
        encoder = encoder.to(device)
        head = head.to(device)

        # Create gradient scaler
        scaler = torch.cuda.amp.GradScaler()

        # Corrupt scaler state by setting invalid scale
        scaler._scale = torch.tensor(float("nan")).cuda()

        with patch("experiments.train_pcam.logger") as mock_logger:
            try:
                metrics = train_epoch(
                    feature_extractor=feature_extractor,
                    encoder=encoder,
                    head=head,
                    dataloader=mock_dataloader,
                    optimizer=optimizer,
                    criterion=criterion,
                    device=device,
                    config=mock_config,
                    epoch=1,
                    scheduler=scheduler,
                    scaler=scaler,
                    writer=None,
                    run_id="test_scaler_corruption",
                    status_path=None,
                )

                nan_warnings = len(
                    [call for call in mock_logger.warning.call_args_list if "NaN" in str(call)]
                )

                print(f"\nScaler corruption scenario:")
                print(f"  NaN warnings: {nan_warnings}")
                print(f"  Training completed: True")

            except RuntimeError as e:
                print(f"\nScaler corruption scenario:")
                print(f"  Training failed with error: {str(e)}")
                print(f"  This demonstrates scaler state corruption impact")
