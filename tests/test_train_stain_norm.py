"""
Unit tests for stain normalization training utilities.

Tests the loss functions and morphology preservation metrics.
"""

import sys
from pathlib import Path

import pytest
import torch

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.train_stain_norm import (
    ColorConsistencyLoss,
    MorphologyPreservationMetrics,
    PerceptualLoss,
)


class TestPerceptualLoss:
    """Tests for PerceptualLoss."""

    def test_perceptual_loss_forward(self):
        """Test that perceptual loss computes without errors."""
        loss_fn = PerceptualLoss()

        # Create dummy images
        input_img = torch.randn(2, 3, 224, 224)
        target_img = torch.randn(2, 3, 224, 224)

        # Compute loss
        loss = loss_fn(input_img, target_img)

        # Check output
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss.item() >= 0  # Loss should be non-negative

    def test_perceptual_loss_identical_images(self):
        """Test that identical images have low perceptual loss."""
        loss_fn = PerceptualLoss()

        # Create identical images
        img = torch.randn(2, 3, 224, 224)

        # Compute loss
        loss = loss_fn(img, img)

        # Loss should be very small for identical images
        assert loss.item() < 0.01


class TestColorConsistencyLoss:
    """Tests for ColorConsistencyLoss."""

    def test_color_loss_with_reference(self):
        """Test color consistency loss with reference image."""
        loss_fn = ColorConsistencyLoss()

        normalized = torch.randn(2, 3, 64, 64)
        reference = torch.randn(2, 3, 64, 64)

        loss = loss_fn(normalized, reference=reference)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_color_loss_with_target_stats(self):
        """Test color consistency loss with target statistics."""
        loss_fn = ColorConsistencyLoss()

        normalized = torch.randn(2, 3, 64, 64)
        target_mean = torch.tensor([0.5, 0.5, 0.5])
        target_std = torch.tensor([0.2, 0.2, 0.2])

        loss = loss_fn(normalized, target_mean=target_mean, target_std=target_std)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_color_loss_batch_consistency(self):
        """Test color consistency loss for batch consistency."""
        loss_fn = ColorConsistencyLoss()

        normalized = torch.randn(4, 3, 64, 64)

        loss = loss_fn(normalized)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0


class TestMorphologyPreservationMetrics:
    """Tests for MorphologyPreservationMetrics."""

    def test_ssim_identical_images(self):
        """Test SSIM for identical images (should be 1.0)."""
        img = torch.randn(2, 3, 128, 128)

        ssim = MorphologyPreservationMetrics.compute_ssim(img, img)

        assert isinstance(ssim, torch.Tensor)
        assert 0.99 <= ssim.item() <= 1.0  # Should be very close to 1

    def test_ssim_different_images(self):
        """Test SSIM for different images (should be < 1.0)."""
        img1 = torch.randn(2, 3, 128, 128)
        img2 = torch.randn(2, 3, 128, 128)

        ssim = MorphologyPreservationMetrics.compute_ssim(img1, img2)

        assert isinstance(ssim, torch.Tensor)
        assert 0.0 <= ssim.item() < 1.0

    def test_edge_preservation_identical_images(self):
        """Test edge preservation for identical images (should be 1.0)."""
        img = torch.randn(2, 3, 128, 128)

        edge_score = MorphologyPreservationMetrics.compute_edge_preservation(img, img)

        assert isinstance(edge_score, torch.Tensor)
        assert 0.99 <= edge_score.item() <= 1.01  # Allow small floating point error

    def test_edge_preservation_different_images(self):
        """Test edge preservation for different images."""
        img1 = torch.randn(2, 3, 128, 128)
        img2 = torch.randn(2, 3, 128, 128)

        edge_score = MorphologyPreservationMetrics.compute_edge_preservation(img1, img2)

        assert isinstance(edge_score, torch.Tensor)
        assert -1.0 <= edge_score.item() <= 1.0  # Cosine similarity range

    def test_compute_all_metrics(self):
        """Test computing all morphology metrics at once."""
        img1 = torch.randn(2, 3, 128, 128)
        img2 = torch.randn(2, 3, 128, 128)

        metrics = MorphologyPreservationMetrics.compute_all_metrics(img1, img2)

        assert isinstance(metrics, dict)
        assert "ssim" in metrics
        assert "edge_preservation" in metrics
        assert isinstance(metrics["ssim"], float)
        assert isinstance(metrics["edge_preservation"], float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
