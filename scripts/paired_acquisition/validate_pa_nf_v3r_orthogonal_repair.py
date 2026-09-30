#!/usr/bin/env python3
"""No-outcome validator for the PA-NF v3r orthogonal-basis repair."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from experiments.paired_acquisition import run_pa_nf_v3_group_transport_development as v3
from experiments.paired_acquisition.run_pa_nf_v3r_orthogonal_repair import (
    ExactOrthogonalTransportModel,
    FRESH_REPAIR_SEEDS,
)


def invariant_metrics(model: ExactOrthogonalTransportModel, device: torch.device):
    model.eval()
    with torch.no_grad():
        q = model.basis_matrix()
        eye = torch.eye(q.shape[0], device=device, dtype=q.dtype)
        ortho = float((q.T @ q - eye).abs().max().cpu())
        g = torch.Generator(device=device)
        g.manual_seed(20260930)
        x = torch.randn(64, model.feature_dim, generator=g, device=device)
        a = torch.randn(64, model.acquisition_dim, generator=g, device=device) * 0.2
        b = torch.randn(64, model.acquisition_dim, generator=g, device=device) * 0.2
        z = torch.zeros_like(a)
        ident = float(F.mse_loss(model.apply_operator(x, z), x).cpu())
        inv = float(
            F.mse_loss(
                model.apply_operator(model.apply_operator(x, a), -a), x
            ).cpu()
        )
        comp = float(
            F.mse_loss(
                model.apply_operator(model.apply_operator(x, a), b),
                model.apply_operator(x, a + b),
            ).cpu()
        )
    return ortho, ident, inv, comp


def main() -> None:
    config = v3.ExperimentConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v3.set_deterministic_seed(4201)
    candidate = ExactOrthogonalTransportModel(config, "group_transport").to(device)
    control = ExactOrthogonalTransportModel(config, "nonclosed_transport_control").to(device)

    cp = v3.parameter_count(candidate)
    xp = v3.parameter_count(control)
    if cp != xp or cp != 4815:
        raise SystemExit(f"Parameter mismatch: candidate={cp}, control={xp}, expected=4815")

    initial = invariant_metrics(candidate, device)
    if initial[0] >= 1e-5 or max(initial[1:]) >= 1e-8:
        raise SystemExit(f"Initial exact-basis invariant failure: {initial}")

    # Stress the raw parameter with the same optimizer family and weight decay.
    # This intentionally tries to collapse basis_raw; Q must remain orthogonal.
    candidate.train()
    optimizer = torch.optim.AdamW(
        candidate.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    for _ in range(50):
        optimizer.zero_grad(set_to_none=True)
        loss = candidate.basis_raw.square().mean()
        loss.backward()
        optimizer.step()

    stressed = invariant_metrics(candidate, device)
    if stressed[0] >= 1e-5 or max(stressed[1:]) >= 1e-8:
        raise SystemExit(f"Post-optimizer exact-basis invariant failure: {stressed}")

    print("PA-NF V3R ORTHOGONAL REPAIR VALIDATION PASSED")
    print(f"Candidate/control parameters: {cp}")
    print(f"Fresh repaired development seeds: {list(FRESH_REPAIR_SEEDS)}")
    print("Prior v3 seeds 4101-4103: burned diagnostic only")
    print("Basis: Q = matrix_exp(A - A^T)")
    print(f"Initial W^T W max error: {initial[0]:.3e}")
    print(f"Initial zero-theta identity MSE: {initial[1]:.3e}")
    print(f"Initial inverse MSE: {initial[2]:.3e}")
    print(f"Initial composition MSE: {initial[3]:.3e}")
    print(f"Post-optimizer W^T W max error: {stressed[0]:.3e}")
    print(f"Post-optimizer zero-theta identity MSE: {stressed[1]:.3e}")
    print(f"Post-optimizer inverse MSE: {stressed[2]:.3e}")
    print(f"Post-optimizer composition MSE: {stressed[3]:.3e}")
    print("Generator, losses, metrics, control and promotion gates unchanged")
    print("No repaired development or confirmation outcomes were computed")


if __name__ == "__main__":
    main()
