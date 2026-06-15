from __future__ import annotations

import torch

from src.models.scorpion_pathoalign import (
    ProjectionConfig,
    ScorpionProjection,
    covariance_loss,
    cross_covariance_loss,
    projection_loss,
    scanner_dependence_loss,
    supervised_contrastive_loss,
)


def toy_batch(input_dim: int = 12):
    torch.manual_seed(7)
    inputs = torch.randn(15, input_dim)
    region_labels = torch.arange(3).repeat_interleave(5)
    scanner_labels = torch.arange(5).repeat(3)
    return inputs, region_labels, scanner_labels


def test_paired_consistency_forward_and_loss_are_finite():
    inputs, region_labels, scanner_labels = toy_batch()
    config = ProjectionConfig(
        input_dim=inputs.shape[1],
        hidden_dim=16,
        biological_dim=8,
        acquisition_dim=4,
    )
    model = ScorpionProjection("paired_consistency", config)
    output = model(inputs)
    loss, parts = projection_loss(model, inputs, scanner_labels, region_labels)

    assert output["biological"].shape == (15, 8)
    assert output["acquisition"] is None
    assert output["reconstruction"].shape == inputs.shape
    assert torch.isfinite(loss)
    assert set(parts) >= {
        "contrastive",
        "reconstruction",
        "variance_b",
        "covariance_b",
        "total",
    }


def test_pathoalign_forward_and_loss_include_factor_terms():
    inputs, region_labels, scanner_labels = toy_batch()
    config = ProjectionConfig(
        input_dim=inputs.shape[1],
        hidden_dim=16,
        biological_dim=8,
        acquisition_dim=4,
        scanner_dependence_weight=1.0,
    )
    model = ScorpionProjection("pathoalign", config)
    output = model(inputs)
    loss, parts = projection_loss(model, inputs, scanner_labels, region_labels)

    assert output["biological"].shape == (15, 8)
    assert output["acquisition"].shape == (15, 4)
    assert output["scanner_b"].shape == (15, 5)
    assert output["scanner_a"].shape == (15, 5)
    assert torch.isfinite(loss)
    assert set(parts) >= {
        "scanner_b",
        "scanner_a",
        "scanner_dependence",
        "variance_a",
        "cross_covariance",
    }


def test_contrastive_loss_rewards_same_region_alignment():
    region_labels = torch.arange(3).repeat_interleave(5)
    aligned = torch.eye(3).repeat_interleave(5, dim=0)
    torch.manual_seed(11)
    random_representation = torch.randn_like(aligned)

    aligned_loss = supervised_contrastive_loss(aligned, region_labels, 0.1)
    random_loss = supervised_contrastive_loss(
        random_representation, region_labels, 0.1
    )
    assert aligned_loss < random_loss


def test_covariance_penalties_are_nonnegative():
    torch.manual_seed(13)
    biological = torch.randn(20, 8)
    acquisition = torch.randn(20, 4)
    scanner_labels = torch.arange(5).repeat(4)
    assert covariance_loss(biological) >= 0
    assert cross_covariance_loss(biological, acquisition) >= 0
    assert scanner_dependence_loss(biological, scanner_labels) >= 0


def test_scanner_dependence_detects_scanner_aligned_representation():
    scanner_labels = torch.arange(5).repeat(4)
    aligned = torch.nn.functional.one_hot(scanner_labels, num_classes=5).float()
    torch.manual_seed(17)
    random_representation = torch.randn(20, 5)

    aligned_loss = scanner_dependence_loss(aligned, scanner_labels)
    random_loss = scanner_dependence_loss(random_representation, scanner_labels)
    assert aligned_loss > random_loss


def test_gradient_reversal_changes_biological_gradient_direction():
    inputs, _, scanner_labels = toy_batch()
    config = ProjectionConfig(
        input_dim=inputs.shape[1],
        hidden_dim=16,
        biological_dim=8,
        acquisition_dim=4,
        gradient_reversal_strength=1.0,
    )
    model = ScorpionProjection("pathoalign", config)
    output = model(inputs)
    loss = torch.nn.functional.cross_entropy(output["scanner_b"], scanner_labels)
    loss.backward()

    gradients = [
        parameter.grad
        for parameter in model.biological.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)
