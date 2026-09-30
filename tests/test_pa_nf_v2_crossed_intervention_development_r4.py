from __future__ import annotations

import json
from pathlib import Path

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


def test_r4_parameter_counts_equal() -> None:
    config = r4.ExperimentConfig(identities=64, epochs=1, bootstrap_replicates=10)
    a = r1.build_model(config, torch.device("cpu"))
    b = r1.build_model(config, torch.device("cpu"))
    assert r1.parameter_count(a) == r1.parameter_count(b) == 158432


def test_pool_excludes_training_query_sample() -> None:
    acquisition = torch.tensor(
        [[1.0], [10.0], [3.0], [30.0], [5.0], [50.0]], dtype=torch.float32
    )
    scanner_ids = torch.tensor([0, 1, 0, 1, 0, 1], dtype=torch.long)
    train = torch.arange(6, dtype=torch.long)
    query = torch.tensor([0, 3], dtype=torch.long)
    pooled = r4.pooled_acquisition_for_queries(acquisition, scanner_ids, train, query, 2)
    # query 0 (scanner 0) is excluded: mean(3,5)=4
    # query 3 (scanner 1) is excluded: mean(10,50)=30
    assert torch.allclose(pooled, torch.tensor([[4.0], [30.0]]))


def test_heldout_query_uses_training_scanner_pool_without_self_contribution() -> None:
    config = r4.ExperimentConfig(identities=64, epochs=1, bootstrap_replicates=10)
    dataset = base.make_synthetic_dataset(r1.to_base_config(config), "linear")
    model = r1.build_model(config, torch.device("cpu"))
    observations = torch.as_tensor(dataset.observations, dtype=torch.float32)
    scanner_ids = torch.as_tensor(dataset.scanner_ids, dtype=torch.long)
    train = torch.as_tensor(dataset.train_indices, dtype=torch.long)
    target = torch.as_tensor(dataset.test_indices, dtype=torch.long)
    with torch.no_grad():
        acquisition = model.encode_acquisition(observations)
        pooled = r4.pooled_acquisition_for_queries(
            acquisition, scanner_ids, train, target, config.scanners
        )
    assert pooled.shape == (config.identities, config.acquisition_dim)
    train_set = set(dataset.train_indices.tolist())
    assert all(int(index) not in train_set for index in dataset.test_indices.tolist())


def test_r4_spec_retains_r2_gates_and_drops_r3_contrastive_tuning() -> None:
    root = Path(__file__).resolve().parents[1]
    spec = json.loads(
        (root / "experiments/paired_acquisition/pa_nf_v2_crossed_intervention_dev_r4_spec_20260929.json").read_text(
            encoding="utf-8"
        )
    )
    assert spec["schema_version"] == r4.SCHEMA_VERSION
    assert spec["losses"]["biological_contrastive_weight"] == 0.0
    assert spec["development_grid"]["no_weight_grid"] is True
    assert spec["development_promotion_gate"][
        "candidate_mean_biology_retention_delta_greater_than_control"
    ] is True
    assert spec["r4_intervention_operator"]["scanner_identity_required_at_model_inference"] is False
