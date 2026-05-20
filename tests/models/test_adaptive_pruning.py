"""
Unit tests for adaptive token pruning.

Tests:
- ImportanceScorer: scoring methods
- AdaptivePruning: top-k selection, masking
- PrunedTransMIL: end-to-end forward pass
"""

import pytest
import torch
import torch.nn as nn

from src.models.transnnmil.adaptive_pruning import AdaptivePruning, ImportanceScorer, PrunedTransMIL


class TestImportanceScorer:
    """Test importance scoring methods."""

    @pytest.mark.parametrize("scoring_method", ["learned", "attention", "confidence"])
    def test_scoring_methods(self, scoring_method):
        """Test different scoring methods."""
        scorer = ImportanceScorer(
            feature_dim=512,
            hidden_dim=256,
            scoring_method=scoring_method,
        )

        features = torch.randn(4, 50, 512)
        scores = scorer(features)

        # Check shape
        assert scores.shape == (4, 50, 1)

        # Check finite
        assert torch.isfinite(scores).all()

        # Check confidence in [0, 1]
        if scoring_method == "confidence":
            assert (scores >= 0).all()
            assert (scores <= 1).all()

    def test_gradient_flow(self):
        """Test gradient flow through scorer."""
        scorer = ImportanceScorer(feature_dim=512, hidden_dim=256)

        features = torch.randn(2, 30, 512, requires_grad=True)
        scores = scorer(features)
        loss = scores.sum()
        loss.backward()

        # Check gradients exist
        assert features.grad is not None
        assert torch.isfinite(features.grad).all()

    def test_invalid_scoring_method(self):
        """Test invalid scoring method raises error."""
        with pytest.raises(ValueError, match="scoring_method must be"):
            ImportanceScorer(feature_dim=512, scoring_method="invalid")


class TestAdaptivePruning:
    """Test adaptive pruning module."""

    def test_basic_pruning(self):
        """Test basic pruning operation."""
        pruning = AdaptivePruning(feature_dim=512, keep_ratio=0.5)

        features = torch.randn(4, 100, 512)
        pruned_features, mask, indices = pruning(features)

        # Check shapes
        assert pruned_features.shape[0] == 4
        assert pruned_features.shape[2] == 512
        assert mask.shape == (4, 100)
        assert indices.shape[0] == 4

        # Check keep ratio
        kept_ratio = mask.float().mean().item()
        assert 0.4 < kept_ratio < 0.6  # Approximately 50%

    def test_pruning_with_mask(self):
        """Test pruning with input mask."""
        pruning = AdaptivePruning(feature_dim=512, keep_ratio=0.5)

        features = torch.randn(4, 100, 512)
        mask = torch.rand(4, 100) > 0.2  # 80% valid patches

        pruned_features, pruned_mask, indices = pruning(features, mask)

        # Check shapes
        assert pruned_features.shape[0] == 4
        assert pruned_mask.shape == (4, 100)

        # Check pruned patches are subset of valid patches
        assert (pruned_mask & ~mask).sum() == 0  # No invalid patches kept

    def test_different_keep_ratios(self):
        """Test different keep ratios."""
        features = torch.randn(2, 100, 512)

        for keep_ratio in [0.25, 0.5, 0.75]:
            pruning = AdaptivePruning(feature_dim=512, keep_ratio=keep_ratio)
            pruned_features, mask, indices = pruning(features)

            kept_ratio = mask.float().mean().item()
            # Allow some tolerance
            assert abs(kept_ratio - keep_ratio) < 0.15

    def test_min_patches(self):
        """Test minimum patches constraint."""
        pruning = AdaptivePruning(feature_dim=512, keep_ratio=0.01, min_patches=20)

        features = torch.randn(2, 100, 512)
        pruned_features, mask, indices = pruning(features)

        # Should keep at least min_patches
        num_kept = mask.sum(dim=1)
        assert (num_kept >= 20).all()

    def test_speedup_estimation(self):
        """Test speedup estimation."""
        pruning = AdaptivePruning(feature_dim=512, keep_ratio=0.5)

        speedup = pruning.get_speedup(num_patches=100)

        # With 50% pruning, speedup should be ~4x (quadratic complexity)
        assert 3.5 < speedup < 4.5

    def test_invalid_keep_ratio(self):
        """Test invalid keep ratio raises error."""
        with pytest.raises(ValueError, match="keep_ratio must be"):
            AdaptivePruning(feature_dim=512, keep_ratio=0.0)

        with pytest.raises(ValueError, match="keep_ratio must be"):
            AdaptivePruning(feature_dim=512, keep_ratio=1.5)

    def test_deterministic_pruning(self):
        """Test deterministic pruning with same input."""
        torch.manual_seed(42)

        pruning = AdaptivePruning(feature_dim=512, keep_ratio=0.5)
        pruning.eval()

        features = torch.randn(2, 50, 512)

        # Two forward passes
        with torch.no_grad():
            _, mask1, indices1 = pruning(features)
            _, mask2, indices2 = pruning(features)

        # Should be identical
        assert torch.equal(mask1, mask2)
        assert torch.equal(indices1, indices2)


class TestPrunedTransMIL:
    """Test PrunedTransMIL model."""

    def test_forward_pass(self):
        """Test forward pass."""
        model = PrunedTransMIL(
            feature_dim=512,
            num_classes=2,
            keep_ratio=0.5,
            num_layers=2,
            num_heads=8,
        )

        features = torch.randn(4, 100, 512)
        logits = model(features)

        # Check shape
        assert logits.shape == (4, 2)

        # Check finite
        assert torch.isfinite(logits).all()

    def test_with_mask(self):
        """Test forward pass with mask."""
        model = PrunedTransMIL(
            feature_dim=512,
            num_classes=2,
            keep_ratio=0.5,
        )

        features = torch.randn(4, 100, 512)
        mask = torch.rand(4, 100) > 0.2

        logits = model(features, mask)

        assert logits.shape == (4, 2)
        assert torch.isfinite(logits).all()

    def test_return_pruning_info(self):
        """Test returning pruning info."""
        model = PrunedTransMIL(
            feature_dim=512,
            num_classes=2,
            keep_ratio=0.5,
        )

        features = torch.randn(4, 100, 512)
        logits, pruning_info = model(features, return_pruning_info=True)

        # Check logits
        assert logits.shape == (4, 2)

        # Check pruning info
        assert "mask" in pruning_info
        assert "indices" in pruning_info
        assert "keep_ratio" in pruning_info

        assert pruning_info["mask"].shape == (4, 100)
        assert 0 < pruning_info["keep_ratio"] < 1

    @pytest.mark.parametrize("scoring_method", ["learned", "attention", "confidence"])
    def test_scoring_methods(self, scoring_method):
        """Test different scoring methods."""
        model = PrunedTransMIL(
            feature_dim=512,
            num_classes=2,
            keep_ratio=0.5,
            scoring_method=scoring_method,
        )

        features = torch.randn(2, 50, 512)
        logits = model(features)

        assert logits.shape == (2, 2)
        assert torch.isfinite(logits).all()

    def test_gradient_flow(self):
        """Test gradient flow through model."""
        model = PrunedTransMIL(
            feature_dim=512,
            num_classes=2,
            keep_ratio=0.5,
        )

        features = torch.randn(2, 50, 512, requires_grad=True)
        logits = model(features)
        loss = logits.sum()
        loss.backward()

        # Check gradients exist
        assert features.grad is not None
        assert torch.isfinite(features.grad).all()

    def test_variable_bag_sizes(self):
        """Test with variable bag sizes (via masking)."""
        model = PrunedTransMIL(
            feature_dim=512,
            num_classes=2,
            keep_ratio=0.5,
        )

        features = torch.randn(4, 100, 512)

        # Variable sizes: 50, 75, 100, 60
        mask = torch.zeros(4, 100, dtype=torch.bool)
        mask[0, :50] = True
        mask[1, :75] = True
        mask[2, :100] = True
        mask[3, :60] = True

        logits = model(features, mask)

        assert logits.shape == (4, 2)
        assert torch.isfinite(logits).all()

    def test_different_keep_ratios(self):
        """Test different keep ratios."""
        features = torch.randn(2, 100, 512)

        for keep_ratio in [0.25, 0.5, 0.75]:
            model = PrunedTransMIL(
                feature_dim=512,
                num_classes=2,
                keep_ratio=keep_ratio,
            )

            logits = model(features)
            assert logits.shape == (2, 2)
            assert torch.isfinite(logits).all()


class TestPruningIntegration:
    """Integration tests for pruning."""

    def test_pruning_reduces_computation(self):
        """Test that pruning reduces computation."""
        # Count FLOPs (approximate via forward pass time)
        import time

        features = torch.randn(4, 200, 512)

        # Without pruning (keep_ratio=1.0)
        model_full = PrunedTransMIL(
            feature_dim=512,
            num_classes=2,
            keep_ratio=1.0,
            num_layers=2,
        )
        model_full.eval()

        with torch.no_grad():
            start = time.time()
            for _ in range(10):
                _ = model_full(features)
            time_full = time.time() - start

        # With pruning (keep_ratio=0.5)
        model_pruned = PrunedTransMIL(
            feature_dim=512,
            num_classes=2,
            keep_ratio=0.5,
            num_layers=2,
        )
        model_pruned.eval()

        with torch.no_grad():
            start = time.time()
            for _ in range(10):
                _ = model_pruned(features)
            time_pruned = time.time() - start

        # Pruned should be faster (allow some tolerance)
        speedup = time_full / time_pruned
        assert speedup > 1.2  # At least 20% faster

    def test_pruning_preserves_accuracy(self):
        """Test that pruning doesn't break model."""
        # Simple sanity check: model can overfit small dataset
        model = PrunedTransMIL(
            feature_dim=128,
            num_classes=2,
            keep_ratio=0.5,
        )

        # Small synthetic dataset
        features = torch.randn(8, 50, 128)
        labels = torch.randint(0, 2, (8,))

        # Train for a few steps
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        initial_loss = None
        for _ in range(20):
            optimizer.zero_grad()
            logits = model(features)
            loss = criterion(logits, labels)

            if initial_loss is None:
                initial_loss = loss.item()

            loss.backward()
            optimizer.step()

        final_loss = loss.item()

        # Loss should decrease
        assert final_loss < initial_loss
