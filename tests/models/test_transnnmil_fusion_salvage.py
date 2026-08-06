"""Focused regression tests for the salvaged TransnnMIL fusion work."""

from __future__ import annotations

import io

import pytest
import torch
from torch import nn

from src.models.transnnmil.branch_token_fusion import (
    BranchAttentionFusion,
    BranchConcatFusion,
    BranchGateFusion,
)
from src.models.transnnmil.transnnmil_branch_token import (
    TransnnMILBranchAttentionExperimental,
    TransnnMILConcatExperimental,
    TransnnMILGateExperimental,
)


def test_historical_single_key_attention_is_query_invariant() -> None:
    torch.manual_seed(3)
    attention = nn.MultiheadAttention(32, 4, dropout=0.0, batch_first=True).eval()
    query_a = torch.randn(2, 1, 32)
    query_b = torch.randn(2, 1, 32)
    key_value = torch.randn(2, 1, 32)

    output_a, weights_a = attention(query_a, key_value, key_value)
    output_b, weights_b = attention(query_b, key_value, key_value)

    assert torch.allclose(weights_a, torch.ones_like(weights_a))
    assert torch.allclose(weights_b, torch.ones_like(weights_b))
    assert torch.allclose(output_a, output_b, atol=1e-7, rtol=0.0)


@pytest.mark.parametrize(
    "fusion_class",
    [BranchAttentionFusion, BranchConcatFusion, BranchGateFusion],
)
def test_fusion_responds_to_both_branches(fusion_class) -> None:
    kwargs = {"embed_dim": 32, "dropout": 0.0}
    if fusion_class is BranchAttentionFusion:
        kwargs["num_heads"] = 4
    fusion = fusion_class(**kwargs).eval()

    branch_a = torch.randn(3, 32)
    branch_b = torch.randn(3, 32)
    baseline, _ = fusion(torch.stack([branch_a, branch_b], dim=1))
    changed_a, _ = fusion(torch.stack([branch_a + 0.5, branch_b], dim=1))
    changed_b, _ = fusion(torch.stack([branch_a, branch_b - 0.5], dim=1))

    assert baseline.shape == (3, 32)
    assert not torch.allclose(baseline, changed_a)
    assert not torch.allclose(baseline, changed_b)


def test_attention_fusion_backpropagates_to_both_branch_tokens() -> None:
    torch.manual_seed(11)
    fusion = BranchAttentionFusion(embed_dim=32, num_heads=4, dropout=0.0)
    tokens = torch.randn(2, 2, 32, requires_grad=True)
    fused, details = fusion(tokens)
    fused.square().mean().backward()

    assert set(details) == {"branch_pool_weights", "self_attention_weights"}
    assert torch.all(tokens.grad.norm(dim=-1) > 1e-8)


@pytest.mark.parametrize(
    "model_class",
    [
        TransnnMILBranchAttentionExperimental,
        TransnnMILConcatExperimental,
        TransnnMILGateExperimental,
    ],
)
def test_model_checkpoint_round_trip(model_class) -> None:
    torch.manual_seed(13)
    kwargs = {
        "feature_dim": 64,
        "hidden_dim": 32,
        "num_classes": 6,
        "num_layers": 1,
        "num_heads": 4,
        "dropout": 0.0,
        "use_pos_encoding": False,
    }
    model = model_class(**kwargs).eval()
    features = torch.randn(2, 12, 64)
    num_patches = torch.tensor([12, 9])
    logits_before = model(features, num_patches=num_patches)

    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)

    restored = model_class(**kwargs).eval()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    logits_after = restored(features, num_patches=num_patches)

    assert logits_before.shape == (2, 6)
    assert torch.allclose(logits_before, logits_after, atol=1e-7, rtol=0.0)


def test_topology_is_rejected_for_experimental_variants() -> None:
    with pytest.raises(ValueError, match="enable_topology=False"):
        TransnnMILBranchAttentionExperimental(
            feature_dim=64,
            hidden_dim=32,
            num_classes=6,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
            enable_topology=True,
        )
