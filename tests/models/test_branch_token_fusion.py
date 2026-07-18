"""Regression tests for the historical TransnnMIL defect and experimental controls."""

from __future__ import annotations

import io

import pytest
import torch
from torch import nn

from src.models.factory import create_attention_model
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


def test_historical_single_key_cross_attention_is_query_invariant():
    """Lock down the exact defect motivating the experimental variants."""
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


def test_branch_attention_uses_both_inputs_and_returns_honest_details():
    torch.manual_seed(7)
    fusion = BranchAttentionFusion(embed_dim=32, num_heads=4, dropout=0.0).eval()
    branch_a = torch.randn(3, 32)
    branch_b = torch.randn(3, 32)
    tokens = torch.stack([branch_a, branch_b], dim=1)

    baseline, details = fusion(tokens)
    changed_a, _ = fusion(torch.stack([torch.randn_like(branch_a), branch_b], dim=1))
    changed_b, _ = fusion(torch.stack([branch_a, torch.randn_like(branch_b)], dim=1))

    assert baseline.shape == (3, 32)
    assert set(details) == {"branch_pool_weights", "self_attention_weights"}
    assert details["branch_pool_weights"].shape == (3, 2)
    assert details["self_attention_weights"].shape == (3, 4, 2, 2)
    assert torch.allclose(details["branch_pool_weights"].sum(dim=1), torch.ones(3), atol=1e-6)
    assert not torch.allclose(baseline, changed_a)
    assert not torch.allclose(baseline, changed_b)


def test_branch_identity_embeddings_break_permutation_invariance():
    torch.manual_seed(8)
    fusion = BranchAttentionFusion(embed_dim=32, num_heads=4, dropout=0.0).eval()
    tokens = torch.randn(4, 2, 32)
    original, _ = fusion(tokens)
    swapped, _ = fusion(tokens.flip(1))
    assert not torch.allclose(original, swapped)


def test_branch_attention_backpropagates_material_gradients_to_both_inputs():
    torch.manual_seed(11)
    fusion = BranchAttentionFusion(embed_dim=32, num_heads=4, dropout=0.0)
    tokens = torch.randn(2, 2, 32, requires_grad=True)
    fused, _ = fusion(tokens)
    fused.square().mean().backward()

    gradient_norms = tokens.grad.norm(dim=-1)
    assert torch.all(gradient_norms > 1e-8)


@pytest.mark.parametrize(
    "fusion_class",
    [BranchAttentionFusion, BranchConcatFusion, BranchGateFusion],
)
def test_fusion_controls_have_expected_shape(fusion_class):
    kwargs = {"embed_dim": 32, "dropout": 0.0}
    if fusion_class is BranchAttentionFusion:
        kwargs["num_heads"] = 4
    fusion = fusion_class(**kwargs).eval()
    fused, _ = fusion(torch.randn(3, 2, 32))
    assert fused.shape == (3, 32)
    assert torch.isfinite(fused).all()


@pytest.mark.parametrize(
    "model_type,expected_class",
    [
        ("transnnmil_branch_attention_experimental", TransnnMILBranchAttentionExperimental),
        ("transnnmil_concat_experimental", TransnnMILConcatExperimental),
        ("transnnmil_gate_experimental", TransnnMILGateExperimental),
    ],
)
def test_factory_builds_experimental_variants(model_type, expected_class):
    config = {
        "model_type": model_type,
        "hidden_dim": 32,
        "num_classes": 6,
        "dropout": 0.0,
        "transnnmil": {"num_layers": 1, "num_heads": 4, "use_pos_encoding": False},
    }
    model = create_attention_model(config, feature_dim=64)
    assert isinstance(model, expected_class)


def test_topology_is_explicitly_rejected():
    with pytest.raises(ValueError, match="disable enable_topology"):
        TransnnMILBranchAttentionExperimental(
            feature_dim=64,
            hidden_dim=32,
            num_classes=6,
            num_layers=1,
            num_heads=4,
            dropout=0.0,
            enable_topology=True,
        )


@pytest.mark.parametrize(
    "model_class",
    [
        TransnnMILBranchAttentionExperimental,
        TransnnMILConcatExperimental,
        TransnnMILGateExperimental,
    ],
)
def test_forward_and_checkpoint_round_trip(model_class):
    torch.manual_seed(13)
    model = model_class(
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
    logits_before = model(features, num_patches=num_patches)

    checkpoint = io.BytesIO()
    torch.save(model.state_dict(), checkpoint)
    checkpoint.seek(0)

    restored = model_class(
        feature_dim=64,
        hidden_dim=32,
        num_classes=6,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        use_pos_encoding=False,
    ).eval()
    restored.load_state_dict(torch.load(checkpoint, weights_only=True))
    logits_after = restored(features, num_patches=num_patches)

    assert logits_before.shape == (2, 6)
    assert torch.isfinite(logits_before).all()
    assert torch.allclose(logits_before, logits_after, atol=1e-7, rtol=0.0)


def test_attention_variant_exposes_separate_branch_pool_and_patch_attention():
    model = TransnnMILBranchAttentionExperimental(
        feature_dim=64,
        hidden_dim=32,
        num_classes=6,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        use_pos_encoding=False,
    ).eval()
    features = torch.randn(2, 10, 64)
    logits, patch_attention, details = model(
        features,
        num_patches=torch.tensor([10, 8]),
        return_attention=True,
        return_fusion_details=True,
    )
    assert logits.shape == (2, 6)
    assert patch_attention is not None
    assert details["branch_pool_weights"].shape == (2, 2)
