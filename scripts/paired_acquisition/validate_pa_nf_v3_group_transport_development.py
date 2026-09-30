#!/usr/bin/env python3
"""Validate frozen PA-NF v3 group-transport development wiring without outcomes."""

from __future__ import annotations

import inspect
import json
import sys
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.paired_acquisition import (  # noqa: E402
    run_pa_nf_v3_group_transport_development as v3,
)

SPEC_PATH = (
    REPOSITORY_ROOT
    / "experiments"
    / "paired_acquisition"
    / "pa_nf_v3_group_transport_dev_spec_20260930.json"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise SystemExit("PA-NF V3 DEVELOPMENT VALIDATION FAILED: {}".format(message))


def main() -> None:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    require(spec["status"] == "frozen_before_v3_outcome_inspection", "spec not frozen")
    require(spec["schema_version"] == v3.SCHEMA_VERSION, "schema mismatch")
    require(spec["new_synthetic_benchmark"]["reuse_of_v2_r1_to_r5_generator"] is False, "v2 generator reused")
    require(spec["new_synthetic_benchmark"]["model_seeds"] == list(v3.DEFAULT_SMOKE_SEEDS), "seed mismatch")
    require(spec["new_synthetic_benchmark"]["heldout_scanner"] == v3.HELDOUT_COMPOSED_SCANNER, "heldout scanner mismatch")

    config = v3.ExperimentConfig()
    candidate = v3.TransportModel(config, v3.MODEL_FAMILIES[0])
    control = v3.TransportModel(config, v3.MODEL_FAMILIES[1])
    candidate_count = v3.parameter_count(candidate)
    control_count = v3.parameter_count(control)
    require(candidate_count == control_count, "candidate/control parameter mismatch")

    dataset = v3.make_dataset(config, v3.RENDERERS[0])
    require(not (dataset.scanner_ids[dataset.train_indices] == v3.HELDOUT_COMPOSED_SCANNER).any(), "heldout scanner leaked into training")
    expected_train = config.train_identities * len(v3.TRAIN_SCANNERS)
    require(len(dataset.train_indices) == expected_train, "unexpected train count")

    torch.manual_seed(9917)
    x = torch.randn(17, config.feature_dim)
    zero = torch.zeros(17, config.acquisition_dim)
    a = torch.randn(17, config.acquisition_dim) * 0.2
    b = torch.randn(17, config.acquisition_dim) * 0.2
    with torch.no_grad():
        identity_error = torch.max(torch.abs(candidate.apply_operator(x, zero) - x)).item()
        inverse_error = torch.max(
            torch.abs(candidate.apply_operator(candidate.apply_operator(x, a), -a) - x)
        ).item()
        sequential = candidate.apply_operator(candidate.apply_operator(x, a), b)
        composed = candidate.apply_operator(x, a + b)
        composition_mse = torch.mean((sequential - composed) ** 2).item()
        control_sequential = control.apply_operator(control.apply_operator(x, a), b)
        control_composed = control.apply_operator(x, a + b)
        control_composition_mse = torch.mean((control_sequential - control_composed) ** 2).item()

    require(identity_error < 1e-5, "candidate identity law failed")
    require(inverse_error < 1e-5, "candidate inverse law failed")
    require(composition_mse < 1e-10, "candidate composition law failed")
    require(control_composition_mse > composition_mse * 100.0 + 1e-12, "control did not break closure")

    source = inspect.getsource(v3)
    forbidden = [
        "run_pa_nf_v2_crossed_intervention_development",
        "shortcut_conflict",
        "scorpion",
    ]
    for token in forbidden:
        require(token not in source.lower(), "forbidden prior-outcome dependency: {}".format(token))

    print("PA-NF V3 DEVELOPMENT VALIDATION PASSED")
    print("Candidate/control parameters: {}".format(candidate_count))
    print("Fresh smoke seeds: {}".format(list(v3.DEFAULT_SMOKE_SEEDS)))
    print("New synthetic generator: true")
    print("Held-out scanner 5 absent from training: true")
    print("Candidate operator laws: identity + inverse + additive composition")
    print("Matched control: same parameters/losses, deliberately nonclosed coordinate law")
    print("Candidate identity max error: {:.3e}".format(identity_error))
    print("Candidate inverse max error: {:.3e}".format(inverse_error))
    print("Candidate composition MSE: {:.3e}".format(composition_mse))
    print("Control composition MSE: {:.3e}".format(control_composition_mse))
    print("No v3 development or confirmation outcomes were computed")


if __name__ == "__main__":
    main()
