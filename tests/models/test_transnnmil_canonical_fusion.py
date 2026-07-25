"""Regression tests for the repaired canonical TransnnMIL implementation."""

from __future__ import annotations

import pytest
import torch
from torch import nn

from src.models.transnnmil.transnnmil import TOPOLOGY_AVAILABLE, TransnnMIL


def _model() -> TransnnMIL:
    torch.manual_seed(17)
    model = TransnnMIL(
        feature_dim=64,
        hidden_dim=32,
        num_classes=3,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        use_pos_encoding=False,
    ).eval()
    model.proj_a = nn.Identity()
    model.proj_b = nn.Identity()
    return model


def test_canonical_fusion_responds_to_each_branch() -> None:
    model = _model()
    features = torch.randn(2, 8, 64)
    counts = torch.tensor([8, 6])

    branch_a = torch.randn(2, 512)
    branch_b = torch.randn(2, 512)
    model.branch_a.get_features = lambda *_args, **_kwargs: branch_a
    model.branch_b.get_features = lambda *_args, **_kwargs: branch_b
    baseline = model._fused_features(features, counts, None)

    model.branch_a.get_features = lambda *_args, **_kwargs: branch_a + 0.5
    changed_a = model._fused_features(features, counts, None)

    model.branch_a.get_features = lambda *_args, **_kwargs: branch_a
    model.branch_b.get_features = lambda *_args, **_kwargs: branch_b - 0.5
    changed_b = model._fused_features(features, counts, None)

    assert baseline.shape == (2, 512)
    assert not torch.allclose(baseline, changed_a)
    assert not torch.allclose(baseline, changed_b)


def test_canonical_fusion_backpropagates_to_both_branch_tokens() -> None:
    model = _model()
    features = torch.randn(2, 8, 64)
    counts = torch.tensor([8, 7])
    branch_a = torch.randn(2, 512, requires_grad=True)
    branch_b = torch.randn(2, 512, requires_grad=True)
    model.branch_a.get_features = lambda *_args, **_kwargs: branch_a
    model.branch_b.get_features = lambda *_args, **_kwargs: branch_b

    fused = model._fused_features(features, counts, None)
    fused.square().mean().backward()

    assert branch_a.grad is not None
    assert branch_b.grad is not None
    assert torch.all(branch_a.grad.norm(dim=1) > 1e-8)
    assert torch.all(branch_b.grad.norm(dim=1) > 1e-8)


def test_gate_parameter_is_not_used_as_fusion_weight() -> None:
    model = _model()
    features = torch.randn(2, 8, 64)
    counts = torch.tensor([8, 8])
    branch_a = torch.randn(2, 512)
    branch_b = torch.randn(2, 512)
    model.branch_a.get_features = lambda *_args, **_kwargs: branch_a
    model.branch_b.get_features = lambda *_args, **_kwargs: branch_b

    before = model._fused_features(features, counts, None)
    with torch.no_grad():
        model.gate_param.fill_(20.0)
    after = model._fused_features(features, counts, None)

    assert torch.allclose(before, after, atol=1e-7, rtol=0.0)


@pytest.mark.skipif(not TOPOLOGY_AVAILABLE, reason="torch-geometric is unavailable")
def test_topology_projection_is_registered() -> None:
    model = TransnnMIL(
        feature_dim=64,
        hidden_dim=32,
        num_classes=3,
        num_layers=1,
        num_heads=4,
        dropout=0.0,
        enable_topology=True,
        region_hidden_dim=32,
    )
    parameter_names = set(dict(model.named_parameters()))
    assert "proj_c.weight" in parameter_names
    assert "proj_c.bias" in parameter_names
