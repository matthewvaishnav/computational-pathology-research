"""Regression tests for non-degenerate TransnnMIL branch fusion."""

import torch

from src.models.transnnmil.branch_token_fusion import BranchTokenFusion
from src.models.transnnmil.transnnmil_branch_token import TransnnMILBranchToken


def test_branch_token_fusion_uses_both_inputs():
    torch.manual_seed(7)
    fusion = BranchTokenFusion(embed_dim=32, num_heads=4, dropout=0.0).eval()

    branch_a = torch.randn(3, 1, 32)
    branch_b = torch.randn(3, 1, 32)

    baseline, weights = fusion(branch_a, branch_b, branch_b)
    changed_a, _ = fusion(branch_a + 0.5, branch_b, branch_b)
    changed_b, _ = fusion(branch_a, branch_b + 0.5, branch_b + 0.5)

    assert baseline.shape == (3, 1, 32)
    assert weights.shape == (3, 1, 2)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(3, 1), atol=1e-6)
    assert not torch.allclose(baseline, changed_a), "Fusion is still independent of Branch A"
    assert not torch.allclose(baseline, changed_b), "Fusion is independent of Branch B"


def test_branch_token_fusion_backpropagates_to_both_inputs():
    torch.manual_seed(11)
    fusion = BranchTokenFusion(embed_dim=32, num_heads=4, dropout=0.0)

    branch_a = torch.randn(2, 1, 32, requires_grad=True)
    branch_b = torch.randn(2, 1, 32, requires_grad=True)
    fused, _ = fusion(branch_a, branch_b, branch_b)
    fused.square().mean().backward()

    assert branch_a.grad is not None and branch_a.grad.abs().sum() > 0
    assert branch_b.grad is not None and branch_b.grad.abs().sum() > 0


def test_corrected_transnnmil_forward_shape():
    torch.manual_seed(13)
    model = TransnnMILBranchToken(
        feature_dim=64,
        hidden_dim=32,
        num_classes=6,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        use_pos_encoding=False,
    ).eval()

    features = torch.randn(2, 12, 64)
    num_patches = torch.tensor([12, 9])
    logits = model(features, num_patches=num_patches)

    assert logits.shape == (2, 6)
    assert torch.isfinite(logits).all()
