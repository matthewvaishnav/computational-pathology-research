#!/usr/bin/env python3
from __future__ import annotations

import json
import os
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_ROOT) not in os.sys.path:
    os.sys.path.insert(0, str(REPOSITORY_ROOT))

import torch

from experiments.paired_acquisition import run_pa_nf_v2_crossed_intervention_development as r1
from experiments.paired_acquisition import run_pa_nf_v2_crossed_intervention_development_r3 as r3

SPEC = REPOSITORY_ROOT / "experiments/paired_acquisition/pa_nf_v2_crossed_intervention_dev_r3_spec_20260929.json"


def main() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    if spec["status"] != "development_only_not_confirmation":
        raise SystemExit("R3 spec status mismatch")
    if tuple(spec["biological_contrastive"]["weight_grid"]) != r3.CONTRASTIVE_WEIGHT_GRID:
        raise SystemExit("R3 weight grid mismatch")
    if tuple(spec["development_grid"]["model_seeds"]) != r3.DEFAULT_SMOKE_SEEDS:
        raise SystemExit("R3 seed mismatch")
    if set(r3.DEFAULT_SMOKE_SEEDS) & set(range(3101, 3111)):
        raise SystemExit("R3 seeds overlap R1")
    if set(r3.DEFAULT_SMOKE_SEEDS) & set(range(3201, 3211)):
        raise SystemExit("R3 seeds overlap R2")

    config = r3.ExperimentConfig(identities=64, epochs=1, bootstrap_replicates=10)
    device = torch.device("cpu")
    candidate = r1.build_model(config, device)
    control = r1.build_model(config, device)
    if r1.parameter_count(candidate) != r1.parameter_count(control):
        raise SystemExit("Candidate/control parameter count mismatch")

    dataset = r3.base.make_synthetic_dataset(r1.to_base_config(config), "linear")
    train = torch.as_tensor(dataset.train_indices, dtype=torch.long)
    observations = torch.as_tensor(dataset.observations, dtype=torch.float32)
    identities = torch.as_tensor(dataset.identity_ids, dtype=torch.long).index_select(0, train)
    biological = candidate.encode_biological(observations).index_select(0, train)
    loss = r3.biological_contrastive_loss(
        biological, identities, config.biological_contrastive_temperature
    )
    if not torch.isfinite(loss):
        raise SystemExit("Contrastive loss is non-finite")

    print("PA-NF V2 DEVELOPMENT R3 VALIDATION PASSED")
    print("Candidate/control parameters: {}".format(r1.parameter_count(candidate)))
    print("Contrastive weight grid: {}".format(list(r3.CONTRASTIVE_WEIGHT_GRID)))
    print("Fresh smoke seeds: {}".format(list(r3.DEFAULT_SMOKE_SEEDS)))
    print("Selection rule: lowest weight passing every unchanged R2 promotion gate")
    print("Frozen 2026-09-29 SCORPION outcomes remain excluded from development")
    print("No R3 development or confirmation outcomes were computed")


if __name__ == "__main__":
    main()
