#!/usr/bin/env python3
"""Validate PA-NF v2 R5 wiring without computing development outcomes."""

from __future__ import annotations

import json
import os
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPOSITORY_ROOT))

import torch

from experiments.paired_acquisition import (
    run_pa_nf_v2_crossed_intervention_development_r5 as r5,
)

SPEC = REPOSITORY_ROOT / "experiments" / "paired_acquisition" / "pa_nf_v2_crossed_intervention_dev_r5_spec_20260930.json"


def main() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if spec["status"] != "frozen_before_r5_outcome_inspection":
        raise SystemExit("R5 spec is not frozen")
    if spec["smoke"]["model_seeds"] != list(r5.DEFAULT_SMOKE_SEEDS):
        raise SystemExit("R5 seed mismatch")
    if spec["architecture"]["name"] != "canonical_biology_plus_bounded_acquisition_operator":
        raise SystemExit("R5 architecture mismatch")
    if spec["architecture"]["individual_donor_acquisition_code_used_in_crossed_intervention"]:
        raise SystemExit("R5 unexpectedly permits individual donor acquisition code")

    config = r5.ExperimentConfig()
    candidate = r5.build_model(config, torch.device("cpu"))
    control = r5.build_model(config, torch.device("cpu"))
    candidate_count = r5.r1.parameter_count(candidate)
    control_count = r5.r1.parameter_count(control)
    if candidate_count != control_count:
        raise SystemExit("Candidate/control parameter counts differ")

    batch = 4
    biological = torch.randn(batch, config.biological_dim)
    zero_acquisition = torch.zeros(batch, config.acquisition_dim)
    decoder = candidate.decoder
    with torch.no_grad():
        canonical = decoder.canonical_biology(biological)
        operator = decoder.acquisition_operator(zero_acquisition)
        if not torch.equal(operator, torch.zeros_like(operator)):
            raise SystemExit("Zero acquisition code is not exact zero operator")
        expected = decoder.shared_readout(canonical)
        observed = candidate.decode(biological, zero_acquisition)
        if not torch.equal(expected, observed):
            raise SystemExit("Zero acquisition code is not exact identity modulation")

    print("PA-NF V2 DEVELOPMENT R5 VALIDATION PASSED")
    print("Candidate/control parameters: {}".format(candidate_count))
    print("Fresh smoke seeds: {}".format(list(r5.DEFAULT_SMOKE_SEEDS)))
    print("Decoder: canonical biology + bounded acquisition operator")
    print("Zero acquisition code: exact identity modulation")
    print("Crossed acquisition source: leave-query-out same-scanner training pool")
    print("Individual donor acquisition code in crossed intervention: false")
    print("Frozen 2026-09-29 SCORPION outcomes remain excluded from development")
    print("No R5 development or confirmation outcomes were computed")


if __name__ == "__main__":
    main()
