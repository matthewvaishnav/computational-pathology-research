from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pytest

from experiments.paired_acquisition import (
    run_calibrated_unseen_identity_representation_geometry_v2 as diagnostic,
)
from experiments.paired_acquisition import (
    run_representation_geometry_instrument_calibration as calibration_v1,
)
from experiments.paired_acquisition import (
    run_unseen_identity_representation_geometry as geometry_v1,
)


REPOSITORY = Path(__file__).resolve().parents[1]
PRIMARY = REPOSITORY / "results/unseen_identity_crossed_generalization_smoke_20260802T101904/unseen_identity_generalization_result.json"
FAILED = REPOSITORY / "results/unseen_identity_representation_geometry_smoke_20260802T131914/unseen_identity_representation_geometry_result.json"
V1 = REPOSITORY / "results/representation_geometry_instrument_calibration_20260802T135823/representation_geometry_instrument_calibration_result.json"
V2 = REPOSITORY / "results/residual_probe_nonlinear_capacity_calibration_v2_20260802T164702/residual_probe_nonlinear_capacity_calibration_v2_result.json"


def test_all_frozen_artifacts_and_internal_hashes_verify() -> None:
    verified = diagnostic.verify_frozen_artifacts(PRIMARY, FAILED, V1, V2)
    assert verified["hashes"] == {
        "primary_reference": diagnostic.PRIMARY_SHA256,
        "failed_geometry": diagnostic.FAILED_GEOMETRY_SHA256,
        "v1_calibration": diagnostic.V1_SHA256,
        "v2_calibration": diagnostic.V2_SHA256,
    }


def test_v2_composite_adjudication_must_be_complete(monkeypatch) -> None:
    original = diagnostic.base.json.loads

    def altered(text: str):
        payload = original(text)
        if payload.get("schema_version") == diagnostic.calibration_v2.SCHEMA_VERSION:
            payload["instrument_family_adjudication"] = "incomplete"
        return payload

    monkeypatch.setattr(diagnostic.base.json, "loads", altered)
    with pytest.raises(diagnostic.ExperimentError, match="adjudication"):
        diagnostic.verify_frozen_artifacts(PRIMARY, FAILED, V1, V2)


@pytest.mark.parametrize("family", ["oracle_supervised", "pa_nf", "prototype_reconstruction"])
def test_only_crossed_target_factorizer_may_be_built(family: str) -> None:
    with pytest.raises(diagnostic.ExperimentError):
        diagnostic.validate_model_family(family)


def test_exactly_eight_crossed_target_fits_are_scheduled() -> None:
    schedule = diagnostic.scheduled_factorizer_runs()
    assert len(schedule) == 8
    assert {row[2] for row in schedule} == {"crossed_target_prototype"}


def test_reference_matching_uses_all_four_keys() -> None:
    runs = [
        {
            "dataset_seed": 4301,
            "renderer": "linear",
            "model_family": "crossed_target_prototype",
            "model_seed": 2201,
        }
    ]
    assert diagnostic.matching_reference_run(
        runs, 4301, "linear", "crossed_target_prototype", 2201
    ) is runs[0]
    with pytest.raises(Exception):
        diagnostic.matching_reference_run(
            runs, 4301, "linear", "crossed_target_prototype", 2202
        )


def test_replication_failure_fails_closed() -> None:
    reference = {"evaluation": {"metrics": {name: 1.0 for name in diagnostic.REFERENCE_METRICS}}}
    observed = {name: 1.0 for name in diagnostic.REFERENCE_METRICS}
    observed[diagnostic.REFERENCE_METRICS[0]] = 1.1
    assert not diagnostic.compare_replication(reference, observed)["passed"]


def test_failed_diagnostic_split_hashes_reproduce() -> None:
    failed = diagnostic.base.json.loads(FAILED.read_text(encoding="utf-8"))
    split = failed["dataset_manifest"]["4301:linear"]["identity_split"]
    assert diagnostic.verify_probe_split(split, split)["passed"]


def _small_split() -> geometry_v1.IdentitySplit:
    return geometry_v1.IdentitySplit(
        probe_training_identities=np.array([0, 1]),
        probe_validation_identities=np.array([2]),
        unseen_test_identities=np.array([3, 4]),
        probe_training_indices=np.array([0, 1, 2, 3]),
        probe_validation_indices=np.array([4, 5]),
        unseen_test_indices=np.array([6, 7, 8, 9]),
        split_seed=1,
    )


def test_probe_scaler_uses_only_probe_training_rows() -> None:
    values = np.arange(20, dtype=float).reshape(10, 2)
    split = _small_split()
    scaler = geometry_v1.fit_training_scaler(values, split.probe_training_indices)
    assert np.allclose(scaler.mean_, values[split.probe_training_indices].mean(axis=0))


def test_residual_probe_epoch_zero_equals_ridge() -> None:
    rng = np.random.default_rng(3)
    features = rng.normal(size=(10, 3)).astype(np.float32)
    targets = rng.normal(size=(10, 2)).astype(np.float32)
    config = calibration_v1.ResidualConfig(maximum_epochs=1, early_stopping_patience=1)
    _, fit = calibration_v1.residual_probe_result(features, targets, _small_split(), 9, config)
    assert fit.epoch_zero_max_abs_difference <= calibration_v1.TIGHT_NUMERICAL_TOLERANCE


def _repeat(r2: float, epoch_zero: bool = False, ridge: float = 0.2):
    return {
        "selected_epoch_zero": epoch_zero,
        "unseen_test": {"residual_r2": r2, "ridge_r2": ridge},
    }


def test_both_residual_seeds_are_required_for_stable_recovery() -> None:
    assert diagnostic.stable_residual_recovery([_repeat(0.8), _repeat(0.9)])
    assert not diagnostic.stable_residual_recovery([_repeat(0.9)])
    assert not diagnostic.stable_residual_recovery([_repeat(0.9), _repeat(0.79)])


def _scanner_result(observed=(0.5, 0.5, 0.5), null=(0.2, 0.2, 0.2)):
    return {
        "chance_level": 0.2,
        "observed_balanced_accuracy_median": float(np.median(observed)),
        "repeats": [
            {
                "observed": {"balanced_accuracy": x},
                "permutation_null": {"balanced_accuracy": y},
            }
            for x, y in zip(observed, null)
        ],
    }


def test_scanner_leakage_requires_every_paired_observed_null_condition() -> None:
    assert diagnostic.scanner_interpretation(_scanner_result(), True)[
        "hidden_scanner_leakage_detected"
    ]
    assert not diagnostic.scanner_interpretation(
        _scanner_result(null=(0.2, 0.6, 0.2)), True
    )["hidden_scanner_leakage_detected"]


def test_one_scanner_initialization_cannot_declare_leakage() -> None:
    result = _scanner_result(observed=(0.9, 0.2, 0.2))
    assert not diagnostic.scanner_interpretation(result, True)[
        "hidden_scanner_leakage_detected"
    ]


def test_acquisition_exclusion_uses_maximum_across_seeds() -> None:
    assert diagnostic.acquisition_biology_exclusion([_repeat(0.0), _repeat(0.1)])
    assert not diagnostic.acquisition_biology_exclusion([_repeat(0.0), _repeat(0.1001)])


def test_retrieval_requires_worst_scanner_pair() -> None:
    assert not diagnostic.retrieval_success(
        {
            "unseen_identity_retrieval_top1": 1.0,
            "worst_scanner_pair_identity_retrieval_top1": 0.89,
        }
    )


def test_independent_decoder_separation_requires_every_negative() -> None:
    primary, negatives = 0.4, [0.5, 0.6, 0.8, 1.0]
    assert all(primary <= 0.8 * value for value in negatives)
    negatives[0] = 0.49
    assert not all(primary <= 0.8 * value for value in negatives)


def _ordered_input_fixture():
    primary = diagnostic.base.json.loads(PRIMARY.read_text(encoding="utf-8"))
    config = diagnostic.unseen.ExperimentConfig(**primary["config"])
    dataset = diagnostic.unseen.make_unseen_identity_dataset(config, "linear")
    biological = np.arange(
        len(dataset.identity_ids) * config.prototype_biological_dim, dtype=np.float32
    ).reshape(len(dataset.identity_ids), config.prototype_biological_dim)
    return dataset, biological, diagnostic.ordered_decoder_inputs(biological, dataset)


def test_independent_decoder_positive_input_uses_true_target_scanner() -> None:
    dataset, biological, ordered = _ordered_input_fixture()
    width = biological.shape[1]
    assert np.array_equal(
        ordered["primary"][:, width:],
        dataset.acquisition_latents[ordered["target_indices"]],
    )


def test_wrong_scanner_control_changes_scanner_only() -> None:
    dataset, biological, ordered = _ordered_input_fixture()
    width = biological.shape[1]
    assert np.array_equal(ordered["wrong_scanner"][:, :width], ordered["primary"][:, :width])
    assert np.array_equal(
        ordered["wrong_scanner"][:, width:],
        dataset.acquisition_latents[ordered["wrong_scanner_indices"]],
    )
    assert np.all(
        dataset.scanner_ids[ordered["wrong_scanner_indices"]]
        != dataset.scanner_ids[ordered["target_indices"]]
    )


def test_permuted_biology_control_permutes_identity_not_observations() -> None:
    dataset, biological, ordered = _ordered_input_fixture()
    donors = ordered["permuted_biology_donor_indices"]
    sources = ordered["source_indices"]
    assert np.all(dataset.identity_ids[donors] != dataset.identity_ids[sources])
    assert np.all(dataset.scanner_ids[donors] == dataset.scanner_ids[sources])
    assert np.array_equal(ordered["truth"], dataset.observations[ordered["target_indices"]])


def test_true_factor_positive_control_is_imported() -> None:
    v1 = diagnostic.base.json.loads(V1.read_text(encoding="utf-8"))
    imported = diagnostic.inherited_true_factor_control(v1, 4301, "linear")
    assert imported["source"] == "immutable_v1_calibration"
    assert imported["true_factor_normalized_mse"] > 0


def _status_run(**flags):
    defaults = {
        "reference_replication_passed": True,
        "calibrated_transferable_geometry_supported": False,
        "calibrated_decoder_dependent_geometry_suspected": False,
        "hidden_scanner_leakage_detected": False,
        "calibrated_geometry_unresolved": True,
    }
    defaults.update(flags)
    return {
        "interpretation_flags": defaults,
        "probe_split_verification": {"passed": True},
    }


def test_mixed_evidence_produces_mixed_not_failed(monkeypatch) -> None:
    monkeypatch.setattr(diagnostic, "scheduled_factorizer_runs", lambda: [None, None])
    result = diagnostic.aggregate_interpretation(
        [
            _status_run(calibrated_transferable_geometry_supported=True),
            _status_run(),
        ]
    )
    assert result["status"] == "complete_calibrated_mixed_representation_geometry"


def test_poor_scientific_performance_is_not_execution_failure(monkeypatch) -> None:
    monkeypatch.setattr(diagnostic, "scheduled_factorizer_runs", lambda: [None])
    result = diagnostic.aggregate_interpretation([_status_run()])
    assert result["execution_valid"]
    assert result["status"] != "calibrated_representation_diagnostic_failed"


def test_previous_artifact_hashes_can_be_checked_before_and_after() -> None:
    before = diagnostic.verify_frozen_artifacts(PRIMARY, FAILED, V1, V2)["hashes"]
    after = diagnostic.verify_frozen_artifacts(PRIMARY, FAILED, V1, V2)["hashes"]
    assert after == before


def test_existing_output_directories_cannot_be_overwritten(tmp_path: Path) -> None:
    with pytest.raises(diagnostic.ExperimentError, match="overwrite"):
        diagnostic.ensure_new_output_root(tmp_path)


def test_summary_csv_supports_heterogeneous_metrics(tmp_path: Path) -> None:
    rows = [{"a": 1}, {"b": 2, "a": 3}]
    path = tmp_path / "summary.csv"
    diagnostic.parent.atomic_csv(
        path, diagnostic.parent.summary_csv_fieldnames(rows), rows
    )
    with path.open(newline="", encoding="utf-8") as handle:
        parsed = list(csv.DictReader(handle))
    assert list(parsed[0]) == ["a", "b"]


def test_linear_solution_preservation_applies_when_epoch_zero_selected() -> None:
    assert diagnostic.linear_solution_preserved([_repeat(0.2, True, 0.2)])
    assert not diagnostic.linear_solution_preserved([_repeat(0.19, True, 0.2)])


def test_no_forbidden_path_forward_field_in_runner() -> None:
    source = Path(diagnostic.__file__).read_text(encoding="utf-8")
    assert "path_forward_gate_open" not in source
