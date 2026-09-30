import torch
import torch.nn.functional as F

from experiments.paired_acquisition import run_pa_nf_v3_group_transport_development as v3
from experiments.paired_acquisition.run_pa_nf_v3r_orthogonal_repair import (
    ExactOrthogonalTransportModel,
    FRESH_REPAIR_SEEDS,
)


def test_parameter_budget_and_fresh_seeds():
    config = v3.ExperimentConfig()
    candidate = ExactOrthogonalTransportModel(config, "group_transport")
    control = ExactOrthogonalTransportModel(config, "nonclosed_transport_control")
    assert v3.parameter_count(candidate) == 4815
    assert v3.parameter_count(control) == 4815
    assert FRESH_REPAIR_SEEDS == (4201, 4202, 4203)


def test_group_operator_identity_inverse_composition():
    config = v3.ExperimentConfig()
    v3.set_deterministic_seed(4201)
    model = ExactOrthogonalTransportModel(config, "group_transport")
    x = torch.randn(32, config.feature_dim)
    a = torch.randn(32, config.acquisition_dim) * 0.2
    b = torch.randn(32, config.acquisition_dim) * 0.2
    z = torch.zeros_like(a)
    assert F.mse_loss(model.apply_operator(x, z), x).item() < 1e-8
    assert F.mse_loss(model.apply_operator(model.apply_operator(x, a), -a), x).item() < 1e-8
    assert F.mse_loss(
        model.apply_operator(model.apply_operator(x, a), b),
        model.apply_operator(x, a + b),
    ).item() < 1e-8


def test_basis_stays_orthogonal_under_optimizer_steps():
    config = v3.ExperimentConfig()
    v3.set_deterministic_seed(4201)
    model = ExactOrthogonalTransportModel(config, "group_transport")
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay)
    for _ in range(10):
        optimizer.zero_grad(set_to_none=True)
        loss = model.basis_raw.square().mean()
        loss.backward()
        optimizer.step()
    q = model.basis_matrix().detach()
    eye = torch.eye(config.feature_dim)
    assert (q.T @ q - eye).abs().max().item() < 1e-5
