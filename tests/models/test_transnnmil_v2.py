"""
Unit tests for TransnnMIL v2.0.

Tests:
- TransnnMILv2: three-branch architecture
- TransnnMILv2TwoBranch: two-branch ablations
"""

import pytest
import torch
import torch.nn as nn

from src.models.transnnmil_v2 import TransnnMILv2, TransnnMILv2TwoBranch


class TestTransnnMILv2:
    """Test TransnnMIL v2.0 three-branch model."""

    def test_forward_pass(self):
        """Test forward pass."""
        model = TransnnMILv2(
            feature_dim=512,
            num_classes=2,
            num_regions=8,
            k_neighbors=4,
            gnn_type="gat",
        )

        features = torch.randn(2, 50, 512)
        coords = torch.rand(2, 50, 2)

        logits = model(features, coords)

        # Check shape
        assert logits.shape == (2, 2)

        # Check finite
        assert torch.isfinite(logits).all()

    def test_with_mask(self):
        """Test forward pass with mask."""
        model = TransnnMILv2(
            feature_dim=512,
            num_classes=2,
            num_regions=8,
            k_neighbors=4,
        )

        features = torch.randn(2, 50, 512)
        coords = torch.rand(2, 50, 2)
        mask = torch.rand(2, 50) > 0.2

        logits = model(features, coords, mask)

        assert logits.shape == (2, 2)
        assert torch.isfinite(logits).all()

    def test_with_pruning(self):
        """Test with adaptive pruning enabled."""
        model = TransnnMILv2(
            feature_dim=512,
            num_classes=2,
            use_pruning=True,
            keep_ratio=0.5,
        )

        features = torch.randn(2, 50, 512)
        coords = torch.rand(2, 50, 2)

        logits = model(features, coords)

        assert logits.shape == (2, 2)
        assert torch.isfinite(logits).all()

    @pytest.mark.parametrize("gnn_type", ["gat", "sage", "gin"])
    def test_gnn_types(self, gnn_type):
        """Test different GNN types."""
        model = TransnnMILv2(
            feature_dim=512,
            num_classes=2,
            gnn_type=gnn_type,
        )

        features = torch.randn(2, 30, 512)
        coords = torch.rand(2, 30, 2)

        logits = model(features, coords)

        assert logits.shape == (2, 2)
        assert torch.isfinite(logits).all()

    def test_gradient_flow(self):
        """Test gradient flow through model."""
        model = TransnnMILv2(
            feature_dim=256,
            num_classes=2,
            num_regions=4,
            k_neighbors=4,
        )

        features = torch.randn(2, 30, 256, requires_grad=True)
        coords = torch.rand(2, 30, 2)

        logits = model(features, coords)
        loss = logits.sum()
        loss.backward()

        # Check gradients exist
        assert features.grad is not None
        assert torch.isfinite(features.grad).all()

    def test_variable_bag_sizes(self):
        """Test with variable bag sizes (via masking)."""
        model = TransnnMILv2(
            feature_dim=512,
            num_classes=2,
        )

        features = torch.randn(4, 100, 512)
        coords = torch.rand(4, 100, 2)

        # Variable sizes: 50, 75, 100, 60
        mask = torch.zeros(4, 100, dtype=torch.bool)
        mask[0, :50] = True
        mask[1, :75] = True
        mask[2, :100] = True
        mask[3, :60] = True

        logits = model(features, coords, mask)

        assert logits.shape == (4, 2)
        assert torch.isfinite(logits).all()


class TestTransnnMILv2TwoBranch:
    """Test TransnnMIL v2.0 two-branch ablations."""

    @pytest.mark.parametrize("branches", ["AB", "AC", "BC"])
    def test_two_branch_variants(self, branches):
        """Test different two-branch combinations."""
        model = TransnnMILv2TwoBranch(
            feature_dim=512,
            num_classes=2,
            branches=branches,
            num_regions=8,
            k_neighbors=4,
        )

        features = torch.randn(2, 50, 512)
        coords = torch.rand(2, 50, 2)

        logits = model(features, coords)

        # Check shape
        assert logits.shape == (2, 2)

        # Check finite
        assert torch.isfinite(logits).all()

    def test_invalid_branches(self):
        """Test invalid branches raises error."""
        with pytest.raises(ValueError, match="branches must be"):
            TransnnMILv2TwoBranch(
                feature_dim=512,
                num_classes=2,
                branches="XY",
            )

    def test_gradient_flow(self):
        """Test gradient flow through two-branch model."""
        model = TransnnMILv2TwoBranch(
            feature_dim=256,
            num_classes=2,
            branches="AB",
        )

        features = torch.randn(2, 30, 256, requires_grad=True)
        coords = torch.rand(2, 30, 2)

        logits = model(features, coords)
        loss = logits.sum()
        loss.backward()

        # Check gradients exist
        assert features.grad is not None
        assert torch.isfinite(features.grad).all()


class TestTransnnMILv2Integration:
    """Integration tests for TransnnMIL v2.0."""

    def test_model_can_overfit(self):
        """Test that model can overfit small dataset."""
        model = TransnnMILv2(
            feature_dim=128,
            num_classes=2,
            num_regions=4,
            k_neighbors=4,
        )

        # Small synthetic dataset
        features = torch.randn(8, 30, 128)
        coords = torch.rand(8, 30, 2)
        labels = torch.randint(0, 2, (8,))

        # Train for a few steps
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        initial_loss = None
        for _ in range(20):
            optimizer.zero_grad()
            logits = model(features, coords)
            loss = criterion(logits, labels)

            if initial_loss is None:
                initial_loss = loss.item()

            loss.backward()
            optimizer.step()

        final_loss = loss.item()

        # Loss should decrease
        assert final_loss < initial_loss

    def test_three_vs_two_branch(self):
        """Test three-branch vs two-branch models."""
        # Three-branch
        model_3 = TransnnMILv2(
            feature_dim=256,
            num_classes=2,
            num_regions=4,
            k_neighbors=4,
        )

        # Two-branch
        model_2 = TransnnMILv2TwoBranch(
            feature_dim=256,
            num_classes=2,
            branches="AB",
            num_regions=4,
        )

        features = torch.randn(2, 30, 256)
        coords = torch.rand(2, 30, 2)

        # Both should work
        logits_3 = model_3(features, coords)
        logits_2 = model_2(features, coords)

        assert logits_3.shape == (2, 2)
        assert logits_2.shape == (2, 2)
        assert torch.isfinite(logits_3).all()
        assert torch.isfinite(logits_2).all()

    def test_parameter_count(self):
        """Test parameter count is reasonable."""
        model = TransnnMILv2(
            feature_dim=512,
            num_classes=2,
            num_regions=8,
            k_neighbors=4,
        )

        num_params = sum(p.numel() for p in model.parameters())

        # Should have reasonable number of parameters
        assert 1e6 < num_params < 50e6  # Between 1M and 50M
