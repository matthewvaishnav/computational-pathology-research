from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from experiments.paired_acquisition import (
    run_unseen_identity_crossed_generalization as unseen,
)
from experiments.paired_acquisition import (
    run_unseen_identity_representation_geometry as experiment,
)


def _dataset() -> object:
    config = unseen.ExperimentConfig(
        identities=10,
        test_identities=4,
        scanners=3,
        biological_latent_dim=3,
        acquisition_latent_dim=2,
        observation_dim=8,
        nonlinear_hidden_dim=12,
        epochs=2,
        bootstrap_replicates=10,
        dataset_seed=4301,
    )
    return unseen.make_unseen_identity_dataset(config, "linear")


def _split():
    dataset = _dataset()
    return dataset, experiment.make_probe_identity_split(dataset, 1234, 0.20)


def _reference_payload() -> dict:
    runs = []
    for dataset_seed in experiment.DATASET_SEEDS:
        for renderer in experiment.RENDERERS:
            for model_seed in experiment.MODEL_SEEDS:
                runs.append(
                    {
                        "dataset_seed": dataset_seed,
                        "renderer": renderer,
                        "model_family": "crossed_target_prototype",
                        "model_seed": model_seed,
                        "evaluation": {
                            "gates": {"two_axis_counterfactual_success": True},
                            "metrics": {
                                name: 1.0 for name in experiment.REFERENCE_METRICS
                            },
                        },
                    }
                )
    return {
        "schema_version": experiment.REFERENCE_SCHEMA_VERSION,
        "dataset_seeds": list(experiment.DATASET_SEEDS),
        "model_seeds": list(experiment.MODEL_SEEDS),
        "renderers": list(experiment.RENDERERS),
        "control_validation": {
            "unseen_identity_generalization_gate_open": False
        },
        "runs": runs,
    }


def test_probe_identity_partitions_are_disjoint() -> None:
    _, split = _split()
    groups = [
        set(split.probe_training_identities.tolist()),
        set(split.probe_validation_identities.tolist()),
        set(split.unseen_test_identities.tolist()),
    ]
    assert not groups[0] & groups[1]
    assert not groups[0] & groups[2]
    assert not groups[1] & groups[2]


def test_observations_from_one_identity_cannot_cross_probe_boundaries() -> None:
    dataset, split = _split()
    boundaries = [
        split.probe_training_indices,
        split.probe_validation_indices,
        split.unseen_test_indices,
    ]
    for identity in np.unique(dataset.identity_ids):
        memberships = [
            bool(np.any(dataset.identity_ids[indices] == identity))
            for indices in boundaries
        ]
        assert sum(memberships) == 1


def test_scaler_is_fit_only_on_probe_training_rows() -> None:
    _, split = _split()
    row_count = max(
        split.probe_training_indices.max(),
        split.probe_validation_indices.max(),
        split.unseen_test_indices.max(),
    ) + 1
    values = np.full((row_count, 2), 10_000.0)
    values[split.probe_training_indices] = np.arange(
        len(split.probe_training_indices) * 2
    ).reshape(-1, 2)
    scaler = experiment.fit_training_scaler(values, split.probe_training_indices)
    assert np.allclose(scaler.mean_, values[split.probe_training_indices].mean(axis=0))
    assert not np.allclose(scaler.mean_, values.mean(axis=0))


def test_reference_run_matching_uses_all_four_coordinates() -> None:
    payload = _reference_payload()
    found = experiment.find_reference_run(
        payload["runs"], 4301, "nonlinear", "crossed_target_prototype", 2202
    )
    assert experiment.reference_run_key(found) == (
        4301,
        "nonlinear",
        "crossed_target_prototype",
        2202,
    )
    with pytest.raises(experiment.ExperimentError):
        experiment.find_reference_run(
            payload["runs"], 4301, "nonlinear", "oracle_supervised", 2202
        )


@pytest.mark.parametrize("failure", ["open_gate", "missing_run"])
def test_reference_verification_rejects_open_gate_or_missing_runs(failure: str) -> None:
    payload = _reference_payload()
    if failure == "open_gate":
        payload["control_validation"][
            "unseen_identity_generalization_gate_open"
        ] = True
    else:
        payload["runs"].pop()
    with pytest.raises(experiment.ExperimentError):
        experiment.verify_reference_payload(payload)


def test_nonlinear_probe_is_deterministic_under_same_seed() -> None:
    dataset, split = _split()
    rng = np.random.default_rng(99)
    features = rng.normal(size=(len(dataset.identity_ids), 5)).astype(np.float32)
    targets = rng.normal(size=(len(dataset.identity_ids), 3)).astype(np.float32)
    config = experiment.ProbeConfig(
        hidden_width=8,
        hidden_layers=1,
        maximum_epochs=12,
        early_stopping_patience=4,
    )
    first = experiment.nonlinear_regression_probe(
        features, targets, split, 8675309, config
    )
    second = experiment.nonlinear_regression_probe(
        features, targets, split, 8675309, config
    )
    assert first == second


def test_scanner_chance_level_is_reciprocal_of_class_count() -> None:
    assert experiment.scanner_chance_level(np.array([0, 1, 2, 3, 4] * 2)) == 0.2


def test_per_dimension_r2_has_expected_latent_length() -> None:
    targets = np.arange(24, dtype=np.float64).reshape(6, 4)
    scores = experiment._per_dimension_r2(targets, targets.copy())
    assert scores == [1.0, 1.0, 1.0, 1.0]


def test_acquisition_prototypes_have_zero_within_scanner_donor_variance() -> None:
    scanners = np.tile(np.arange(3), 5)
    prototypes = np.array([[1.0, 0.0], [0.0, 1.0], [-1.0, -1.0]])
    acquisition = prototypes[scanners]
    variance = unseen.acquisition_within_scanner_variance(acquisition, scanners)
    assert variance == 0.0
    assert experiment.verify_scanner_prototype_invariance(
        "crossed_target_prototype", variance
    )
    with pytest.raises(experiment.ExperimentError):
        experiment.verify_scanner_prototype_invariance(
            "crossed_target_prototype", 1e-4
        )


def _aggregate_run(dataset_seed: int, renderer: str, family: str, supported: bool) -> dict:
    flags = {
        "reference_replication_passed": True,
        "ridge_biology_recovery": True,
        "nonlinear_biology_recovery": True,
        "nonlinear_materially_improves_over_ridge": False,
        "linear_scanner_exclusion": True,
        "nonlinear_scanner_exclusion": True,
        "acquisition_biology_exclusion": True,
        "cross_scanner_identity_retrieval_success": True,
        "independent_decoder_generalization_success": True,
        "nonlinear_transferable_representation_supported": supported,
        "decoder_dependent_representation_suspected": False,
        "hidden_scanner_leakage_detected": False,
    }
    return {
        "dataset_seed": dataset_seed,
        "renderer": renderer,
        "model_family": family,
        "diagnostic": {"interpretation_flags": flags},
    }


def test_aggregate_interpretation_fails_closed_when_one_crossed_run_fails() -> None:
    runs = []
    for dataset_seed in experiment.DATASET_SEEDS:
        for renderer in experiment.RENDERERS:
            for _model_seed in experiment.MODEL_SEEDS:
                runs.append(
                    _aggregate_run(
                        dataset_seed, renderer, "crossed_target_prototype", True
                    )
                )
                runs.append(
                    _aggregate_run(dataset_seed, renderer, "oracle_supervised", True)
                )
    runs[0] = copy.deepcopy(runs[0])
    runs[0]["diagnostic"]["interpretation_flags"][
        "nonlinear_transferable_representation_supported"
    ] = False
    result = experiment.aggregate_interpretation(runs)
    assert not result[
        "all_crossed_target_runs_support_nonlinear_transferable_geometry"
    ]
    assert result["status"] == "complete_mixed_representation_geometry"


def test_output_directory_cannot_be_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "already-exists"
    output.mkdir()
    with pytest.raises(experiment.ExperimentError):
        experiment.ensure_new_output_root(output)


def test_summary_csv_supports_heterogeneous_metrics(tmp_path: Path) -> None:
    rows = [{"dataset_seed": 4301, "metric_a": 1.0}, {"renderer": "linear", "metric_b": 2.0}]
    path = tmp_path / "summary.csv"
    experiment.parent.atomic_csv(
        path, experiment.parent.summary_csv_fieldnames(rows), rows
    )
    text = path.read_text(encoding="utf-8")
    assert "metric_a" in text.splitlines()[0]
    assert "metric_b" in text.splitlines()[0]
