#!/usr/bin/env python3
"""Validate PA-NF v2 R4 wiring without computing development outcomes."""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from experiments.paired_acquisition import (
    run_pa_nf_v2_crossed_intervention_development as r1,
)
from experiments.paired_acquisition import (
    run_pa_nf_v2_crossed_intervention_development_r4 as r4,
)
from experiments.paired_acquisition import (
    run_synthetic_crossed_factor_identifiability as base,
)


def main() -> None:
    spec_path = REPO_ROOT / "experiments/paired_acquisition/pa_nf_v2_crossed_intervention_dev_r4_spec_20260929.json"
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    if spec["schema_version"] != r4.SCHEMA_VERSION:
        raise SystemExit("R4 schema mismatch")
    if tuple(spec["development_grid"]["smoke_model_seeds"]) != r4.DEFAULT_SMOKE_SEEDS:
        raise SystemExit("R4 seed mismatch")
    if spec["r4_intervention_operator"]["name"] != "donor-invariant acquisition pooling":
        raise SystemExit("R4 intervention mismatch")
    if spec["losses"]["biological_contrastive_weight"] != 0.0:
        raise SystemExit("R4 must not inherit the failed R3 contrastive grid")

    config = r4.ExperimentConfig(identities=64, epochs=1, bootstrap_replicates=10)
    candidate = r1.build_model(config, torch.device("cpu"))
    control = r1.build_model(config, torch.device("cpu"))
    if r1.parameter_count(candidate) != r1.parameter_count(control):
        raise SystemExit("Candidate/control parameter mismatch")
    if r1.parameter_count(candidate) != 158432:
        raise SystemExit("Unexpected R4 parameter count")

    dataset = base.make_synthetic_dataset(r1.to_base_config(config), "linear")
    observations = torch.as_tensor(dataset.observations, dtype=torch.float32)
    scanner_ids = torch.as_tensor(dataset.scanner_ids, dtype=torch.long)
    train = torch.as_tensor(dataset.train_indices, dtype=torch.long)
    target = torch.as_tensor(dataset.test_indices, dtype=torch.long)
    with torch.no_grad():
        acquisition = candidate.encode_acquisition(observations)
        pooled = r4.pooled_acquisition_for_queries(
            acquisition, scanner_ids, train, target, config.scanners
        )
    if pooled.shape != (config.identities, config.acquisition_dim):
        raise SystemExit("Unexpected pooled acquisition shape")

    print("PA-NF V2 DEVELOPMENT R4 VALIDATION PASSED")
    print("Candidate/control parameters: {}".format(r1.parameter_count(candidate)))
    print("Fresh smoke seeds: {}".format(list(r4.DEFAULT_SMOKE_SEEDS)))
    print("Crossed acquisition source: leave-query-out same-scanner training pool")
    print("Individual donor acquisition code in crossed intervention: false")
    print("R3 biological contrastive weight carried forward: 0.0")
    print("Frozen 2026-09-29 SCORPION outcomes remain excluded from development")
    print("No R4 development or confirmation outcomes were computed")


if __name__ == "__main__":
    main()
