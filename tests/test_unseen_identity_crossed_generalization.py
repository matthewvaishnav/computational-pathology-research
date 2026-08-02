from __future__ import annotations

import numpy as np

from experiments.paired_acquisition import (
    run_crossed_target_scanner_prototype_factorization as parent,
)
from experiments.paired_acquisition import (
    run_unseen_identity_crossed_generalization as experiment,
)


def _config() -> experiment.ExperimentConfig:
    return experiment.ExperimentConfig(
        identities=12,
        test_identities=6,
        scanners=3,
        biological_latent_dim=3,
        acquisition_latent_dim=2,
        observation_dim=8,
        nonlinear_hidden_dim=16,
        pa_nf_biological_dim=8,
        pa_nf_acquisition_dim=4,
        pa_nf_hidden_dim=16,
        prototype_biological_dim=6,
        prototype_acquisition_dim=3,
        prototype_hidden_dim=12,
        epochs=1,
        bootstrap_replicates=20,
        dataset_seed=4301,
    )


def test_identity_split_has_zero_overlap_and_all_scanners() -> None:
    config = _config()
    dataset = experiment.make_unseen_identity_dataset(config, "linear")

    train_identity_ids = np.unique(
        dataset.identity_ids[dataset.train_indices]
    )
    test_identity_ids = np.unique(
        dataset.identity_ids[dataset.test_indices]
    )

    assert len(train_identity_ids) == config.identities
    assert len(test_identity_ids) == config.test_identities
    assert np.intersect1d(train_identity_ids, test_identity_ids).size == 0

    for identity in train_identity_ids:
        scanners = dataset.scanner_ids[dataset.identity_ids == identity]
        assert sorted(scanners.tolist()) == list(range(config.scanners))

    for identity in test_identity_ids:
        scanners = dataset.scanner_ids[dataset.identity_ids == identity]
        assert sorted(scanners.tolist()) == list(range(config.scanners))


def test_parent_training_pairs_never_touch_unseen_identities() -> None:
    config = _config()
    dataset = experiment.make_unseen_identity_dataset(config, "linear")
    source, target = parent.build_crossed_pairs(dataset)

    assert len(source) == config.identities * config.scanners * (
        config.scanners - 1
    )
    assert set(source.tolist()).issubset(set(dataset.train_indices.tolist()))
    assert set(target.tolist()).issubset(set(dataset.train_indices.tolist()))
    assert np.all(dataset.identity_ids[source] == dataset.identity_ids[target])
    assert np.all(dataset.scanner_ids[source] != dataset.scanner_ids[target])


def test_all_ordered_test_pairs_cover_every_transfer() -> None:
    config = _config()
    dataset = experiment.make_unseen_identity_dataset(config, "nonlinear")
    pairs = experiment.build_all_ordered_test_pairs(
        dataset,
        config.scanners,
    )

    expected = config.test_identities * config.scanners * (
        config.scanners - 1
    )
    assert len(pairs["source"]) == expected
    assert len(
        set(
            zip(
                pairs["source_scanner"].tolist(),
                pairs["target_scanner"].tolist(),
            )
        )
    ) == config.scanners * (config.scanners - 1)

    source = pairs["source"]
    target = pairs["target"]
    donor = pairs["donor"]
    assert set(source.tolist()).issubset(set(dataset.test_indices.tolist()))
    assert set(target.tolist()).issubset(set(dataset.test_indices.tolist()))
    assert set(donor.tolist()).issubset(set(dataset.test_indices.tolist()))
    assert np.all(dataset.identity_ids[source] == dataset.identity_ids[target])
    assert np.all(dataset.identity_ids[donor] != dataset.identity_ids[target])
    assert np.all(dataset.scanner_ids[source] != dataset.scanner_ids[target])
    assert np.all(dataset.scanner_ids[donor] == dataset.scanner_ids[target])


def test_cross_scanner_retrieval_recovers_unseen_identity() -> None:
    identities = np.repeat(np.arange(5), 3)
    scanners = np.tile(np.arange(3), 5)
    features = np.eye(5, dtype=np.float64)[identities]

    score = experiment.cross_scanner_identity_retrieval_top1(
        features,
        identities,
        scanners,
    )

    assert score == 1.0


def _passing_metrics() -> dict:
    return {
        "biology_retention_delta_ci_025": 0.1,
        "acquisition_transfer_delta_ci_025": 0.1,
        "two_axis_identity_success_rate": 0.75,
        "biological_to_biological_r2": 0.80,
        "biological_to_acquisition_r2": 0.10,
        "acquisition_to_acquisition_r2": 0.80,
        "acquisition_to_biological_r2": 0.10,
        "combined_to_joint_factors_r2": 0.80,
    }


def test_factorization_thresholds_are_unchanged() -> None:
    metrics = _passing_metrics()
    gates = experiment.make_gates(metrics)
    assert gates["crossed_factorization_success"] is True

    metrics["biological_to_acquisition_r2"] = 0.100001
    gates = experiment.make_gates(metrics)
    assert gates["factor_allocation_success"] is False
    assert gates["crossed_factorization_success"] is False


def _run(dataset_seed: int, renderer: str, model: str, passed: bool) -> dict:
    return {
        "dataset_seed": dataset_seed,
        "renderer": renderer,
        "model_family": model,
        "evaluation": {
            "gates": {
                "crossed_factorization_success": passed,
            }
        },
    }


def test_generalization_gate_requires_every_condition_and_seed() -> None:
    runs = []
    for dataset_seed in (4301, 5301):
        for renderer in ("linear", "nonlinear"):
            runs.extend(
                [
                    _run(dataset_seed, renderer, "oracle_supervised", True),
                    _run(dataset_seed, renderer, "pa_nf", False),
                    _run(
                        dataset_seed,
                        renderer,
                        "prototype_reconstruction",
                        False,
                    ),
                    _run(
                        dataset_seed,
                        renderer,
                        "crossed_target_prototype",
                        True,
                    ),
                ]
            )

    result = experiment.validate_controls(runs)
    assert result["unseen_identity_generalization_gate_open"] is True
    assert result["crossed_objective_incremental_value"] is True

    runs.append(_run(4301, "linear", "pa_nf", True))
    result = experiment.validate_controls(runs)
    assert result["unseen_identity_generalization_gate_open"] is False
