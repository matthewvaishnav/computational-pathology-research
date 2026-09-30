#!/usr/bin/env python3
"""No-outcome validator for frozen PA-NF v3r synthetic confirmation."""

from __future__ import annotations

import json
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
import torch.nn.functional as F

from experiments.paired_acquisition import run_pa_nf_v3_group_transport_development as v3
from experiments.paired_acquisition.run_pa_nf_v3r_confirmation import (
    CONFIRMATION_BOOTSTRAP_SEED,
    CONFIRMATION_DATASET_SEED,
    CONFIRMATION_SEEDS,
)
from experiments.paired_acquisition.run_pa_nf_v3r_orthogonal_repair import ExactOrthogonalTransportModel

SPEC = REPO_ROOT / "experiments" / "paired_acquisition" / "pa_nf_v3r_confirmation_spec_20260930.json"


def main() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if spec.get("status") != "frozen_before_confirmation_outcomes":
        raise SystemExit("Confirmation spec not frozen")
    if tuple(spec.get("confirmation_model_seeds", [])) != CONFIRMATION_SEEDS:
        raise SystemExit("Confirmation model seeds mismatch")
    if int(spec.get("confirmation_dataset_seed")) != CONFIRMATION_DATASET_SEED:
        raise SystemExit("Confirmation dataset seed mismatch")
    if int(spec.get("confirmation_bootstrap_seed")) != CONFIRMATION_BOOTSTRAP_SEED:
        raise SystemExit("Confirmation bootstrap seed mismatch")

    config = replace(v3.ExperimentConfig(), dataset_seed=CONFIRMATION_DATASET_SEED, bootstrap_seed=CONFIRMATION_BOOTSTRAP_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v3.set_deterministic_seed(CONFIRMATION_SEEDS[0])
    candidate = ExactOrthogonalTransportModel(config, "group_transport").to(device)
    control = ExactOrthogonalTransportModel(config, "nonclosed_transport_control").to(device)
    cp = v3.parameter_count(candidate)
    xp = v3.parameter_count(control)
    if cp != xp or cp != 4815:
        raise SystemExit(f"Parameter mismatch candidate={cp} control={xp}")

    with torch.no_grad():
        q = candidate.basis_matrix()
        eye = torch.eye(q.shape[0], device=device, dtype=q.dtype)
        ortho = float((q.T @ q - eye).abs().max().cpu())
        x = torch.randn(32, config.feature_dim, device=device)
        z = torch.zeros(32, config.acquisition_dim, device=device)
        ident = float(F.mse_loss(candidate.apply_operator(x, z), x).cpu())
    if ortho >= 1e-5 or ident >= 1e-8:
        raise SystemExit(f"Basis invariant failure ortho={ortho} ident={ident}")

    print("PA-NF V3R CONFIRMATION VALIDATION PASSED")
    print(f"Candidate/control parameters: {cp}")
    print(f"Confirmation model seeds: {list(CONFIRMATION_SEEDS)}")
    print(f"Confirmation dataset seed: {CONFIRMATION_DATASET_SEED}")
    print(f"Confirmation bootstrap seed: {CONFIRMATION_BOOTSTRAP_SEED}")
    print("Architecture, losses, control, epochs, sample sizes and gates unchanged from v3r development")
    print("No confirmation outcomes were computed")


if __name__ == "__main__":
    main()
