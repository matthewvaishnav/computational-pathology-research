#!/usr/bin/env python3
"""Validate PA-NF v2 development wiring without training or reading outcomes."""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path

import torch

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPOSITORY_ROOT))

from experiments.paired_acquisition import (  # noqa: E402
    run_pa_nf_v2_crossed_intervention_development as experiment,
)
from experiments.paired_acquisition import (  # noqa: E402
    run_synthetic_crossed_factor_identifiability as base,
)


SPEC_PATH = REPOSITORY_ROOT / (
    "experiments/paired_acquisition/"
    "pa_nf_v2_crossed_intervention_dev_spec_20260929.json"
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def main() -> None:
    spec = json.loads(SPEC_PATH.read_text(encoding="utf-8"))
    config = experiment.ExperimentConfig()

    require(spec["schema_version"] == experiment.SCHEMA_VERSION, "Schema mismatch")
    require(spec["status"] == "development_only_not_confirmation", "Status mismatch")
    require(
        spec["architectures"]["candidate"] == "pa_nf_v2_crossed_intervention",
        "Candidate name mismatch",
    )
    require(
        spec["architectures"]["primary_control"] == "v2_control_no_cross_cycle",
        "Control name mismatch",
    )
    require(
        bool(spec["architectures"]["identical_parameterization_required"]),
        "Spec must require identical parameterization",
    )

    losses = spec["losses"]
    expected = {
        "self_reconstruction_weight": config.self_reconstruction_weight,
        "crossed_reconstruction_weight_candidate": config.crossed_reconstruction_weight,
        "crossed_reconstruction_weight_control": 0.0,
        "cycle_weight_candidate": config.cycle_weight,
        "cycle_weight_control": 0.0,
        "biological_consistency_weight": config.biological_consistency_weight,
        "acquisition_prototype_weight": config.acquisition_prototype_weight,
        "biological_variance_weight": config.biological_variance_weight,
        "prototype_center_weight": config.prototype_center_weight,
        "prototype_separation_weight": config.prototype_separation_weight,
    }
    for key, value in expected.items():
        require(float(losses[key]) == float(value), "Loss mismatch: {}".format(key))

    grid = spec["development_grid"]
    require(tuple(grid["full_model_seeds"]) == experiment.DEFAULT_FULL_SEEDS, "Full seeds mismatch")
    require(tuple(grid["smoke_model_seeds"]) == experiment.DEFAULT_SMOKE_SEEDS, "Smoke seeds mismatch")
    require(int(grid["full_identities"]) == config.identities, "Full identities mismatch")
    require(int(grid["scanners"]) == config.scanners, "Scanner count mismatch")
    require(int(grid["full_epochs"]) == config.epochs, "Full epochs mismatch")
    require(int(grid["dataset_seed"]) == config.dataset_seed, "Dataset seed mismatch")

    candidate = experiment.build_model(config, torch.device("cpu"))
    control = experiment.build_model(config, torch.device("cpu"))
    require(
        experiment.parameter_count(candidate) == experiment.parameter_count(control),
        "Candidate/control parameter count mismatch",
    )
    require(
        experiment.family_weights("pa_nf_v2_crossed_intervention", config)
        == (config.crossed_reconstruction_weight, config.cycle_weight),
        "Candidate weights mismatch",
    )
    require(
        experiment.family_weights("v2_control_no_cross_cycle", config) == (0.0, 0.0),
        "Control weights mismatch",
    )

    forward_parameters = list(inspect.signature(candidate.forward).parameters)
    require(
        forward_parameters == ["inputs"],
        "Inference must require image/features only, got {}".format(forward_parameters),
    )

    smoke_config = experiment.replace(
        config,
        identities=int(grid["smoke_identities"]),
        epochs=int(grid["smoke_epochs"]),
        bootstrap_replicates=1000,
    )
    dataset = base.make_synthetic_dataset(experiment.to_base_config(smoke_config), "linear")
    source, target = experiment.build_crossed_pairs(dataset)
    train_set = set(int(index) for index in dataset.train_indices.tolist())
    expected_pairs = (
        smoke_config.identities
        * (smoke_config.scanners - 1)
        * (smoke_config.scanners - 2)
    )
    require(len(source) == expected_pairs, "Unexpected crossed-pair count")
    require(len(source) == len(target), "Crossed-pair vector length mismatch")
    for left, right in zip(source.tolist(), target.tolist()):
        require(left in train_set and right in train_set, "Held-out combination leaked into training pair")
        require(
            dataset.identity_ids[left] == dataset.identity_ids[right],
            "Crossed pair changed biological identity",
        )
        require(
            dataset.scanner_ids[left] != dataset.scanner_ids[right],
            "Crossed pair failed to change scanner",
        )

    forbidden = spec["isolation"]["forbidden_for_tuning"]
    require(len(forbidden) >= 1, "Frozen-outcome isolation list is empty")

    print("PA-NF V2 DEVELOPMENT VALIDATION PASSED")
    print("Candidate/control parameters: {}".format(experiment.parameter_count(candidate)))
    print("Smoke crossed pairs: {}".format(len(source)))
    print("Inference inputs: image/features only")
    print("Frozen 2026-09-29 SCORPION outcomes are not read by this validator or runner")
    print("No development or confirmation outcomes were computed")


if __name__ == "__main__":
    main()
