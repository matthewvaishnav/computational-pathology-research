"""
Property-based tests for uncertainty quantification.

This test file validates correctness properties for the UncertaintyEstimator
class using property-based testing with Hypothesis. Each property test runs
a minimum of 100 iterations to verify universal invariants.

Feature: nnmil-architecture-upgrade
"""

import pytest
import torch
import numpy as np
from hypothesis import given, settings, strategies as st
from typing import Dict, List, Tuple

from src.inference.uncertainty import UncertaintyEstimator

# ============================================================================
# Property 20: Uncertainty Output Shape
# ============================================================================


@settings(max_examples=100, deadline=None)
@given(
    batch_size=st.integers(min_value=1, max_value=16),
    num_windows=st.integers(min_value=2, max_value=20),
    num_classes=st.integers(min_value=2, max_value=10),
    task_type=st.sampled_from(["classification", "regression", "survival"]),
)
def test_property_20_uncertainty_output_shape(batch_size, num_windows, num_classes, task_type):
    """
    Feature: nnmil-architecture-upgrade, Property 20: For any batch of size B
    processed with uncertainty estimation, the uncertainty output SHALL have
    shape [B, 2] containing [epistemic, aleatoric] uncertainties.

    **Validates: Requirements 6.3**
    """
    estimator = UncertaintyEstimator(task_type=task_type)

    # Create predictions from multiple windows
    if task_type == "regression":
        # Regression: predictions are scalars
        predictions = torch.randn(batch_size, num_windows, 1)
    else:
        # Classification/survival: predictions are logits
        predictions = torch.randn(batch_size, num_windows, num_classes)

    # Compute uncertainties
    uncertainties = estimator.compute_uncertainty(predictions)

    # Verify output structure
    assert "epistemic" in uncertainties
    assert "aleatoric" in uncertainties
    assert "total" in uncertainties

    # Verify shapes
    assert uncertainties["epistemic"].shape == (batch_size,), (
        f"Expected epistemic uncertainty shape ({batch_size},), "
        f"got {uncertainties['epistemic'].shape}"
    )
    assert uncertainties["aleatoric"].shape == (batch_size,), (
        f"Expected aleatoric uncertainty shape ({batch_size},), "
        f"got {uncertainties['aleatoric'].shape}"
    )
    assert uncertainties["total"].shape == (batch_size,), (
        f"Expected total uncertainty shape ({batch_size},), " f"got {uncertainties['total'].shape}"
    )


# ============================================================================
# Property 21: Dropout Activation for Uncertainty
# ============================================================================


def test_property_21_dropout_activation_for_uncertainty():
    """
    Feature: nnmil-architecture-upgrade, Property 21: For any inference request
    with uncertainty estimation enabled, dropout layers SHALL remain active
    during forward passes.

    **Validates: Requirements 6.1, 6.4**

    Note: This property is tested indirectly through the sliding window inference
    which enables dropout during uncertainty estimation. The actual dropout
    activation is handled by the model's training mode.
    """
    # This is a design property that's enforced by the sliding window inference
    # implementation. We test that the uncertainty estimator can handle
    # predictions from multiple forward passes (which would come from MC dropout).

    estimator = UncertaintyEstimator(task_type="classification")

    # Simulate predictions from multiple MC dropout passes
    # These would have different values due to dropout randomness
    batch_size, num_mc_samples, num_classes = 4, 10, 3

    # Create slightly different predictions (simulating MC dropout effect)
    base_logits = torch.randn(batch_size, 1, num_classes)
    noise = torch.randn(batch_size, num_mc_samples, num_classes) * 0.1
    mc_predictions = base_logits + noise

    uncertainties = estimator.compute_uncertainty(mc_predictions)

    # Verify that epistemic uncertainty is non-zero (indicating variance across samples)
    assert torch.all(
        uncertainties["epistemic"] >= 0
    ), "Epistemic uncertainty should be non-negative"

    # For MC dropout, we expect some epistemic uncertainty
    # (unless all predictions are identical, which is very unlikely)
    mean_epistemic = uncertainties["epistemic"].mean()
    assert mean_epistemic > 1e-6, "Expected non-zero epistemic uncertainty from MC dropout"


# ============================================================================
# Property 22: Uncertainty Normalization
# ============================================================================


@settings(max_examples=100, deadline=None)
@given(
    batch_size=st.integers(min_value=1, max_value=16),
    num_windows=st.integers(min_value=2, max_value=20),
    num_classes=st.integers(min_value=2, max_value=10),
    task_type=st.sampled_from(["classification", "regression"]),
)
def test_property_22_uncertainty_normalization(batch_size, num_windows, num_classes, task_type):
    """
    Feature: nnmil-architecture-upgrade, Property 22: For any computed uncertainty
    values (epistemic, aleatoric), the values SHALL be in the range [0, 1].

    **Validates: Requirements 6.5**
    """
    estimator = UncertaintyEstimator(task_type=task_type)

    # Create predictions with high variance to test normalization
    if task_type == "regression":
        # High variance regression predictions
        predictions = torch.randn(batch_size, num_windows, 1) * 10
    else:
        # High variance classification logits
        predictions = torch.randn(batch_size, num_windows, num_classes) * 5

    uncertainties = estimator.compute_uncertainty(predictions)

    # Check epistemic uncertainty range
    epistemic = uncertainties["epistemic"]
    assert torch.all(epistemic >= 0), "Epistemic uncertainty should be >= 0"
    assert torch.all(
        epistemic <= 1
    ), f"Epistemic uncertainty should be <= 1, got max {epistemic.max()}"

    # Check aleatoric uncertainty range
    aleatoric = uncertainties["aleatoric"]
    assert torch.all(aleatoric >= 0), "Aleatoric uncertainty should be >= 0"
    assert torch.all(
        aleatoric <= 1
    ), f"Aleatoric uncertainty should be <= 1, got max {aleatoric.max()}"

    # Check total uncertainty range
    total = uncertainties["total"]
    assert torch.all(total >= 0), "Total uncertainty should be >= 0"
    assert torch.all(total <= 1), f"Total uncertainty should be <= 1, got max {total.max()}"


# ============================================================================
# Property 23: Combined Uncertainty Formula
# ============================================================================


@settings(max_examples=100, deadline=None)
@given(
    batch_size=st.integers(min_value=1, max_value=16),
    num_windows=st.integers(min_value=2, max_value=20),
    num_classes=st.integers(min_value=2, max_value=10),
    task_type=st.sampled_from(["classification", "regression"]),
)
def test_property_23_combined_uncertainty_formula(batch_size, num_windows, num_classes, task_type):
    """
    Feature: nnmil-architecture-upgrade, Property 23: For any epistemic uncertainty E
    and aleatoric uncertainty A, the combined uncertainty SHALL equal sqrt(E² + A²).

    **Validates: Requirements 6.6**
    """
    estimator = UncertaintyEstimator(task_type=task_type)

    # Create predictions
    if task_type == "regression":
        predictions = torch.randn(batch_size, num_windows, 1)
    else:
        predictions = torch.randn(batch_size, num_windows, num_classes)

    uncertainties = estimator.compute_uncertainty(predictions)

    epistemic = uncertainties["epistemic"]
    aleatoric = uncertainties["aleatoric"]
    total = uncertainties["total"]

    # Compute expected total uncertainty using the formula
    expected_total = torch.sqrt(epistemic**2 + aleatoric**2)

    # Verify the formula holds (with small tolerance for floating point errors)
    torch.testing.assert_close(
        total,
        expected_total,
        rtol=1e-5,
        atol=1e-6,
        msg="Total uncertainty should equal sqrt(epistemic² + aleatoric²)",
    )


# ============================================================================
# Edge Case Tests
# ============================================================================


def test_single_prediction_uncertainty():
    """Test uncertainty computation with single prediction (no epistemic uncertainty)."""
    estimator = UncertaintyEstimator(task_type="classification")

    # Single prediction (no variance)
    predictions = torch.randn(2, 1, 3)  # batch=2, windows=1, classes=3

    uncertainties = estimator.compute_uncertainty(predictions)

    # Epistemic uncertainty should be zero (no variance across windows)
    assert torch.allclose(uncertainties["epistemic"], torch.zeros(2), atol=1e-6)

    # Aleatoric uncertainty should be non-zero (entropy of predictions)
    assert torch.all(uncertainties["aleatoric"] >= 0)

    # Total should equal aleatoric (since epistemic is zero)
    torch.testing.assert_close(
        uncertainties["total"], uncertainties["aleatoric"], rtol=1e-5, atol=1e-6
    )


def test_identical_predictions_uncertainty():
    """Test uncertainty with identical predictions across windows."""
    estimator = UncertaintyEstimator(task_type="classification")

    # Identical predictions across windows
    base_pred = torch.randn(2, 1, 3)
    predictions = base_pred.repeat(1, 5, 1)  # Repeat across 5 windows

    uncertainties = estimator.compute_uncertainty(predictions)

    # Epistemic uncertainty should be very small (identical predictions)
    assert torch.all(uncertainties["epistemic"] < 1e-5)

    # Aleatoric uncertainty should be non-zero (entropy)
    assert torch.all(uncertainties["aleatoric"] >= 0)


def test_high_confidence_predictions():
    """Test uncertainty with high-confidence predictions."""
    estimator = UncertaintyEstimator(task_type="classification")

    # High confidence predictions (one class has very high logit)
    predictions = torch.zeros(2, 5, 3)
    predictions[:, :, 0] = 10.0  # Very high confidence for class 0
    predictions[:, :, 1:] = -10.0  # Very low for other classes

    uncertainties = estimator.compute_uncertainty(predictions)

    # Aleatoric uncertainty should be low (high confidence)
    assert torch.all(uncertainties["aleatoric"] < 0.1)

    # All uncertainties should be normalized
    assert torch.all(uncertainties["epistemic"] <= 1)
    assert torch.all(uncertainties["aleatoric"] <= 1)
    assert torch.all(uncertainties["total"] <= 1)


def test_regression_uncertainty():
    """Test uncertainty computation for regression tasks."""
    estimator = UncertaintyEstimator(task_type="regression")

    # Regression predictions with variance
    predictions = torch.randn(3, 8, 1) * 2  # batch=3, windows=8, output=1

    uncertainties = estimator.compute_uncertainty(predictions)

    # All uncertainty types should be computed
    assert uncertainties["epistemic"].shape == (3,)
    assert uncertainties["aleatoric"].shape == (3,)
    assert uncertainties["total"].shape == (3,)

    # All should be non-negative and normalized
    for unc_type in ["epistemic", "aleatoric", "total"]:
        unc = uncertainties[unc_type]
        assert torch.all(unc >= 0), f"{unc_type} should be non-negative"
        assert torch.all(unc <= 1), f"{unc_type} should be <= 1"


def test_survival_uncertainty():
    """Test uncertainty computation for survival tasks."""
    estimator = UncertaintyEstimator(task_type="survival")

    # Survival predictions (risk scores)
    predictions = torch.randn(2, 6, 1)  # batch=2, windows=6, risk_score=1

    uncertainties = estimator.compute_uncertainty(predictions)

    # Should handle survival task
    assert uncertainties["epistemic"].shape == (2,)
    assert uncertainties["aleatoric"].shape == (2,)
    assert uncertainties["total"].shape == (2,)

    # Verify formula still holds
    expected_total = torch.sqrt(uncertainties["epistemic"] ** 2 + uncertainties["aleatoric"] ** 2)
    torch.testing.assert_close(uncertainties["total"], expected_total, rtol=1e-5)
