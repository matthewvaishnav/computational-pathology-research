from __future__ import annotations

import inspect
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.paired_acquisition import (
    run_representation_geometry_instrument_calibration as v1,
)
from experiments.paired_acquisition import (
    run_residual_probe_nonlinear_capacity_calibration_v2 as experiment,
)


V1_RESULT = Path(
    r"C:\Users\matth\computational-pathology-research\results"
    r"\representation_geometry_instrument_calibration_20260802T135823"
    r"\representation_geometry_instrument_calibration_result.json"
)


def _small_dataset(control: str, seed: int = 9901):
    return experiment.make_capacity_dataset(
        control,
        seed,
        train_identities=64,
        test_identities=32,
        scanners=3,
    )


def _short_config(maximum_epochs: int = 0) -> v1.ResidualConfig:
    return v1.ResidualConfig(
        hidden_width=32,
        hidden_layers=2,
        maximum_epochs=maximum_epochs,
        early_stopping_patience=30,
    )


def test_v1_residual_probe_implementation_is_imported_unchanged() -> None:
    assert experiment.v1.fit_residual_regressor is v1.fit_residual_regressor
    assert experiment.v1.ResidualConfig is v1.ResidualConfig
    source = inspect.getsource(experiment.run_probe)
    assert "v1.residual_probe_result" in source


def test_epoch_zero_exactly_equals_ridge() -> None:
    dataset = _small_dataset("ridge_preservation")
    result = experiment.run_probe(dataset, 1, _short_config())
    assert result["selected_epoch_zero"]
    assert result["epoch_zero_max_abs_difference"] == 0.0
    assert result["unseen_test"]["residual_r2"] == result["unseen_test"]["ridge_r2"]


def test_teacher_hidden_width_is_no_greater_than_student_width() -> None:
    assert experiment.TEACHER_HIDDEN_WIDTH <= 16
    assert experiment.TEACHER_HIDDEN_WIDTH <= v1.ResidualConfig().hidden_width


def test_teacher_parameters_remain_frozen() -> None:
    teacher = experiment.FrozenResidualTeacher()
    before = experiment.teacher_parameter_sha256(teacher)
    optimizer_parameters = [
        parameter for parameter in teacher.parameters() if parameter.requires_grad
    ]
    assert optimizer_parameters == []
    _ = teacher(torch.randn(5, experiment.INPUT_DIM))
    assert experiment.teacher_parameter_sha256(teacher) == before


def test_teacher_targets_are_deterministic() -> None:
    first = _small_dataset("teacher_in_hypothesis_class", 9902)
    second = _small_dataset("teacher_in_hypothesis_class", 9902)
    assert np.array_equal(first.features, second.features)
    assert np.array_equal(first.targets, second.targets)
    assert first.metadata["teacher_parameter_sha256"] == second.metadata[
        "teacher_parameter_sha256"
    ]


def test_test_identities_are_absent_from_generation_calibration_and_selection() -> None:
    dataset = _small_dataset("teacher_in_hypothesis_class", 9903)
    split = dataset.split
    assert not set(split.unseen_test_identities) & set(split.probe_training_identities)
    assert not set(split.unseen_test_identities) & set(split.probe_validation_identities)
    config = _short_config(maximum_epochs=4)
    first = v1.fit_residual_regressor(
        dataset.features, dataset.targets, split, 2, config
    )
    altered = dataset.targets.copy()
    altered[split.unseen_test_indices] += 1000.0
    second = v1.fit_residual_regressor(
        dataset.features, altered, split, 2, config
    )
    assert first.selected_epoch == second.selected_epoch
    assert first.history == second.history


def test_scanner_repeats_cannot_cross_identity_boundaries() -> None:
    dataset = _small_dataset("analytic_interaction", 9904)
    for identity in np.unique(dataset.identity_ids):
        rows = np.flatnonzero(dataset.identity_ids == identity)
        assert len(rows) == 3
        assert np.all(dataset.features[rows] == dataset.features[rows[0]])
        assert np.all(dataset.targets[rows] == dataset.targets[rows[0]])
        memberships = sum(
            np.any(dataset.identity_ids[indices] == identity)
            for indices in (
                dataset.split.probe_training_indices,
                dataset.split.probe_validation_indices,
                dataset.split.unseen_test_indices,
            )
        )
        assert memberships == 1


def test_teacher_in_class_residual_is_nonlinear() -> None:
    teacher = experiment.FrozenResidualTeacher()
    x = torch.linspace(-2.0, 2.0, 64).reshape(8, 8)
    zero = torch.zeros_like(x)
    with torch.no_grad():
        curvature = teacher(x) + teacher(-x) - 2.0 * teacher(zero)
    assert float(curvature.abs().max()) > 1e-4


def test_teacher_in_class_control_can_improve_beyond_ridge_reduced_fixture() -> None:
    dataset = experiment.make_capacity_dataset(
        "teacher_in_hypothesis_class",
        experiment.GENERATION_SEEDS["teacher_in_hypothesis_class"],
        train_identities=160,
        test_identities=80,
        scanners=2,
    )
    result = experiment.run_probe(
        dataset,
        experiment.PROBE_SEEDS[0],
        replace(v1.ResidualConfig(), maximum_epochs=400, early_stopping_patience=50),
    )
    assert result["selected_epoch"] > 0
    assert result["validation"]["residual_minus_ridge_mse"] < 0.0
    assert result["unseen_test"]["residual_minus_ridge_r2"] > 0.0


def test_analytic_interaction_targets_contain_intended_products() -> None:
    features = np.arange(24, dtype=np.float64).reshape(3, 8) / 10.0
    interactions = experiment.analytic_interaction_values(features)
    assert interactions.shape == (3, 8)
    for output, (left, right) in enumerate(experiment.INTERACTION_PAIRS):
        assert np.array_equal(
            interactions[:, output], features[:, left] * features[:, right]
        )


def test_target_permutation_occurs_by_identity_not_observation() -> None:
    dataset = _small_dataset("permuted_target", 9905)
    permutation = dataset.metadata["identity_permutation"]
    assert len(permutation) == len(np.unique(dataset.identity_ids))
    for identity in np.unique(dataset.identity_ids):
        rows = np.flatnonzero(dataset.identity_ids == identity)
        assert np.all(dataset.targets[rows] == dataset.targets[rows[0]])


def test_inherited_v1_hashes_are_verified() -> None:
    inherited = experiment.verify_inherited_v1(V1_RESULT)
    assert inherited["source_file_sha256"] == experiment.V1_FILE_SHA256
    assert inherited["source_internal_sha256"] == experiment.V1_INTERNAL_SHA256
    assert inherited["primary_reference_sha256"] == experiment.PRIMARY_SHA256
    assert inherited["failed_geometry_sha256"] == experiment.GEOMETRY_SHA256


def test_inherited_scanner_decoder_and_oracle_results_are_present() -> None:
    inherited = experiment.verify_inherited_v1(V1_RESULT)
    assert inherited["scanner_calibration_passed"]
    assert inherited["true_factor_decoder_calibration_passed"]
    assert inherited["oracle_ridge_positive_representations_preserved"]
    assert inherited["oracle_run_count"] == 8


def test_no_factorizer_training_function_is_called(monkeypatch: pytest.MonkeyPatch) -> None:
    def forbidden(*_args, **_kwargs):
        raise AssertionError("factorizer training must not be called")

    monkeypatch.setattr(v1.parent, "build_model", forbidden)
    monkeypatch.setattr(v1.parent, "train_model", forbidden)
    dataset = _small_dataset("ridge_preservation", 9906)
    experiment.run_probe(dataset, 3, _short_config())


def test_no_crossed_target_or_oracle_fit_is_executed() -> None:
    source = inspect.getsource(experiment.run_calibration)
    assert "build_model(" not in source
    assert "train_model(" not in source
    assert "oracle_supervised" not in source
    assert "crossed_target_prototype" not in source


def test_output_directories_cannot_be_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(experiment.ExperimentError):
        experiment.ensure_new_output_root(output)


def test_heterogeneous_summary_csv_fields_are_supported(tmp_path: Path) -> None:
    rows = [{"control": "a", "metric_a": 1.0}, {"control": "b", "metric_b": 2.0}]
    path = tmp_path / "summary.csv"
    experiment.csv_support.atomic_csv(
        path, experiment.csv_support.summary_csv_fieldnames(rows), rows
    )
    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert "metric_a" in header and "metric_b" in header


@pytest.mark.parametrize(
    ("teacher_passed", "interaction_passed", "expected"),
    [
        (False, True, "teacher_residual_control_failed"),
        (True, False, "analytic_interaction_control_failed"),
    ],
)
def test_aggregate_status_fails_closed_when_either_nonlinear_control_fails(
    teacher_passed: bool,
    interaction_passed: bool,
    expected: str,
) -> None:
    flags = {
        "ridge_preservation_passed": True,
        "teacher_in_class_passed": teacher_passed,
        "analytic_interaction_passed": interaction_passed,
        "negative_control_passed": True,
    }
    result = experiment.aggregate_status(flags, True, True)
    assert result["status"] == expected
    assert result["instrument_family_adjudication"] != (
        "complete_instrument_families_calibrated_v2"
    )
