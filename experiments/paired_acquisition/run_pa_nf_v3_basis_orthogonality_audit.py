#!/usr/bin/env python3
"""Post-outcome audit of the learned PA-NF v3 orthogonal basis.

Retrains exactly the frozen linear_biology/4101 candidate and checks whether the
parametrized basis remains orthogonal after optimization, plus direct zero-theta
identity and inverse-roundtrip consistency. Diagnostic only; no hyperparameters,
objectives, gates, or model definitions are changed.
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.paired_acquisition.run_pa_nf_v3_group_transport_development import (
    ExperimentConfig,
    TransportModel,
    make_dataset,
    set_deterministic_seed,
    train_model,
)

OUTPUT = Path("results/pa_nf_v3_basis_orthogonality_audit_20260930.json")


def main() -> None:
    config = ExperimentConfig()
    renderer = "linear_biology"
    seed = 4101
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = make_dataset(config, renderer)
    set_deterministic_seed(seed)
    model = TransportModel(config, "group_transport").to(device)
    train_model(model, dataset, config, device)
    model.eval()

    x = torch.as_tensor(dataset.observations[:128], dtype=torch.float32, device=device)
    with torch.no_grad():
        theta = model.encode_acquisition(x)
        zero = torch.zeros_like(theta)
        w = model.basis.weight
        eye = torch.eye(w.shape[0], dtype=w.dtype, device=w.device)

        wt_w = w.T @ w
        w_wt = w @ w.T
        wt_w_max = float((wt_w - eye).abs().max().cpu())
        w_wt_max = float((w_wt - eye).abs().max().cpu())
        wt_w_mse = float(F.mse_loss(wt_w, eye).cpu())
        w_wt_mse = float(F.mse_loss(w_wt, eye).cpu())

        zero_out = model.apply_operator(x, zero)
        inverse_then_forward = model.apply_operator(model.apply_operator(x, -theta), theta)
        forward_then_inverse = model.apply_operator(model.apply_operator(x, theta), -theta)

        # Pure basis analysis/reconstruction with no acquisition scaling.
        centered = x - model.center
        analyzed = model.basis(centered)
        reconstructed = F.linear(analyzed, model.basis.weight.T) + model.center

        payload = {
            "schema_version": "pa-nf-v3-basis-orthogonality-audit/v1",
            "audit_type": "post-outcome_deterministic_rerun_of_frozen_seed",
            "changes_candidate_or_control": False,
            "renderer": renderer,
            "seed": seed,
            "device": str(device),
            "basis_wt_w_max_abs_error": wt_w_max,
            "basis_w_wt_max_abs_error": w_wt_max,
            "basis_wt_w_mse": wt_w_mse,
            "basis_w_wt_mse": w_wt_mse,
            "zero_theta_identity_mse": float(F.mse_loss(zero_out, x).cpu()),
            "pure_basis_reconstruction_mse": float(F.mse_loss(reconstructed, x).cpu()),
            "inverse_then_forward_mse": float(F.mse_loss(inverse_then_forward, x).cpu()),
            "forward_then_inverse_mse": float(F.mse_loss(forward_then_inverse, x).cpu()),
            "basis_frobenius_norm": float(torch.linalg.matrix_norm(w).cpu()),
            "basis_singular_value_min": float(torch.linalg.svdvals(w).min().cpu()),
            "basis_singular_value_max": float(torch.linalg.svdvals(w).max().cpu()),
        }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    print(f"Artifacts: {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
