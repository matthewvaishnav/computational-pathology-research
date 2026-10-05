#!/usr/bin/env python3
"""Validate frozen PA-NF v5 confirmation without computing confirmation outcomes."""

from __future__ import annotations

import os
import sys
from dataclasses import asdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from experiments.paired_acquisition import run_pa_nf_v4_reference_gauge_factorization as v4
from experiments.paired_acquisition import run_pa_nf_v5_decoupled_reference_gauge as v5
from experiments.paired_acquisition import run_pa_nf_v5_confirmation as confirm


def main() -> None:
    dev = asdict(v5.frozen_config())
    conf = asdict(confirm.frozen_confirmation_config())

    allowed = {"dataset_seed", "bootstrap_seed"}
    changed = {k for k in dev if dev[k] != conf[k]}
    if changed != allowed:
        raise SystemExit(f"Unexpected config differences from v5 development: {sorted(changed)}")
    if confirm.CONFIRMATION_MODEL_SEEDS != (4601, 4602, 4603, 4604, 4605):
        raise SystemExit("Confirmation model seed lock mismatch")
    if set(confirm.CONFIRMATION_MODEL_SEEDS) & set(v5.FROZEN_MODEL_SEEDS):
        raise SystemExit("Development and confirmation model seeds overlap")
    if conf["dataset_seed"] == dev["dataset_seed"]:
        raise SystemExit("Development and confirmation dataset seeds overlap")

    torch.manual_seed(777)
    cfg = confirm.frozen_confirmation_config()
    candidate = v4.ReferenceGaugeModel(cfg, "inverse_transport")
    control = v4.ReferenceGaugeModel(cfg, "no_inverse_transport_control")
    if v4.parameter_count(candidate) != v4.parameter_count(control):
        raise SystemExit("Candidate/control parameter counts differ")

    x = torch.randn(32, cfg.feature_dim)
    max_roundtrip = 0.0
    with torch.no_grad():
        for model in (candidate, control):
            for scanner in v4.ALL_SCANNERS:
                y = model.apply_operator(x, scanner)
                xr = model.invert_operator(y, scanner)
                max_roundtrip = max(max_roundtrip, float(torch.mean((xr - x) ** 2)))
    if max_roundtrip >= 1e-8:
        raise SystemExit(f"Initial inverse roundtrip invariant failed: {max_roundtrip:.3e}")

    observations = torch.randn(24, len(v4.ALL_SCANNERS), cfg.feature_dim).numpy()
    pairs = [(0, 1), (1, 2), (5, 0)]
    before = v5._operator_only_transport_gain(candidate, observations, pairs, torch.device("cpu"))
    with torch.no_grad():
        for p in candidate.encoder.parameters():
            p.add_(10.0 * torch.randn_like(p))
        for p in candidate.decoder.parameters():
            p.add_(10.0 * torch.randn_like(p))
    after = v5._operator_only_transport_gain(candidate, observations, pairs, torch.device("cpu"))
    if abs(before - after) > 1e-12:
        raise SystemExit("Operator-only transport changed after encoder/decoder perturbation")

    print("PA-NF V5 CONFIRMATION VALIDATION PASSED")
    print(f"Candidate/control parameters: {v4.parameter_count(candidate)}")
    print(f"Development dataset seed burned: {v5.FROZEN_DATASET_SEED}")
    print(f"Confirmation dataset seed: {confirm.CONFIRMATION_DATASET_SEED}")
    print(f"Development model seeds burned: {list(v5.FROZEN_MODEL_SEEDS)}")
    print(f"Confirmation model seeds: {list(confirm.CONFIRMATION_MODEL_SEEDS)}")
    print(f"Confirmation bootstrap seed: {confirm.CONFIRMATION_BOOTSTRAP_SEED}")
    print(f"Max initial inverse roundtrip MSE: {max_roundtrip:.3e}")
    print("Architecture, losses, control, calibration, sample sizes, renderers, metrics and gates unchanged")
    print("Operator-only transport invariant to encoder/decoder perturbation")
    if torch.cuda.is_available():
        print(f"CUDA CUBLAS_WORKSPACE_CONFIG currently: {os.environ.get('CUBLAS_WORKSPACE_CONFIG')!r}")
    print("This validator computes no development or confirmation outcomes")


if __name__ == "__main__":
    main()
