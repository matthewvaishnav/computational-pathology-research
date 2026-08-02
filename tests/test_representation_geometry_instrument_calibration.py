from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.paired_acquisition import (
    run_representation_geometry_instrument_calibration as experiment,
)
from experiments.paired_acquisition import (
    run_unseen_identity_crossed_generalization as unseen,
)
from experiments.paired_acquisition import (
    run_unseen_identity_representation_geometry as geometry,
)


PRIMARY = Path(
    r"C:\Users\matth\computational-pathology-research\results"
    r"\unseen_identity_crossed_generalization_smoke_20260802T101904"
    r"\unseen_identity_generalization_result.json"
)
FAILED = Path(
    r"C:\Users\matth\computational-pathology-research\results"
    r"\unseen_identity_representation_geometry_smoke_20260802T131914"
    r"\unseen_identity_representation_geometry_result.json"
)


def _identity_dataset(case: str = "identity_map") -> experiment.CalibrationDataset:
    return experiment.make_regression_calibration_dataset(
        case,
        seed={
            "identity_map": 8101,
            "invertible_affine": 8102,
            "permuted_target": 8103,
        }[case],
        training_identities=20,
        test_identities=10,
        scanners=3,
        latent_dim=4,
    )


def _epoch_zero_config() -> experiment.ResidualConfig:
    return experiment.ResidualConfig(
        hidden_width=8,
        hidden_layers=1,
        maximum_epochs=0,
        early_stopping_patience=2,
    )


def _small_synthetic(renderer: str = "linear"):
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
    dataset = unseen.make_unseen_identity_dataset(config, renderer)
    split = geometry.make_probe_identity_split(dataset, 1234, 0.20)
    return dataset, split


def test_residual_probe_epoch_zero_exactly_equals_ridge() -> None:
    data = _identity_dataset()
    fit = experiment.fit_residual_regressor(
        data.features,
        data.biological_targets,
        data.split,
        seed=1,
        config=_epoch_zero_config(),
    )
    test = data.split.unseen_test_indices
    assert fit.epoch_zero_max_abs_difference == 0.0
    assert np.array_equal(
        fit.predict(data.features[test]), fit.ridge_predict(data.features[test])
    )


def test_zero_initialized_residual_output_is_zero() -> None:
    torch.manual_seed(2)
    model = experiment.ZeroInitializedResidualMLP(4, 3, 8, 2)
    output = model(torch.randn(7, 4))
    assert torch.equal(output, torch.zeros_like(output))


def test_validation_can_select_epoch_zero() -> None:
    assert experiment.select_best_epoch([0.10, 0.11, 0.12], 1e-7) == 0
    data = _identity_dataset()
    config = replace(_epoch_zero_config(), maximum_epochs=3, learning_rate=0.0)
    result, _ = experiment.residual_probe_result(
        data.features, data.biological_targets, data.split, 3, config
    )
    assert result["selected_epoch_zero"]


def test_unseen_test_data_cannot_affect_checkpoint_selection() -> None:
    data = _identity_dataset()
    config = replace(_epoch_zero_config(), maximum_epochs=4)
    first = experiment.fit_residual_regressor(
        data.features, data.biological_targets, data.split, 4, config
    )
    altered_targets = data.biological_targets.copy()
    altered_targets[data.split.unseen_test_indices] += 1000.0
    second = experiment.fit_residual_regressor(
        data.features, altered_targets, data.split, 4, config
    )
    assert first.selected_epoch == second.selected_epoch
    assert first.history == second.history


def test_identity_boundaries_remain_disjoint() -> None:
    data = _identity_dataset()
    groups = [
        set(data.split.probe_training_identities.tolist()),
        set(data.split.probe_validation_identities.tolist()),
        set(data.split.unseen_test_identities.tolist()),
    ]
    assert not groups[0] & groups[1]
    assert not groups[0] & groups[2]
    assert not groups[1] & groups[2]
    for identity in np.unique(data.identity_ids):
        memberships = sum(
            np.any(data.identity_ids[indices] == identity)
            for indices in (
                data.split.probe_training_indices,
                data.split.probe_validation_indices,
                data.split.unseen_test_indices,
            )
        )
        assert memberships == 1


def test_affine_positive_control_is_recovered() -> None:
    data = _identity_dataset("invertible_affine")
    result, _ = experiment.residual_probe_result(
        data.features,
        data.biological_targets,
        data.split,
        seed=5,
        config=_epoch_zero_config(),
    )
    assert result["unseen_test"]["ridge_r2"] > 0.95
    assert result["unseen_test"]["residual_r2"] == result["unseen_test"]["ridge_r2"]


def test_permuted_target_negative_control_is_rejected() -> None:
    data = _identity_dataset("permuted_target")
    result, _ = experiment.residual_probe_result(
        data.features,
        data.biological_targets,
        data.split,
        seed=6,
        config=_epoch_zero_config(),
    )
    assert result["unseen_test"]["ridge_r2"] < 0.80
    assert result["unseen_test"]["residual_r2"] < 0.80


def _scanner_probe_config() -> geometry.ProbeConfig:
    return geometry.ProbeConfig(
        hidden_width=12,
        hidden_layers=1,
        maximum_epochs=300,
        early_stopping_patience=30,
    )


def test_scanner_free_negative_control_stays_near_chance() -> None:
    data = experiment.make_scanner_control_features(
        seed=8201, training_identities=24, test_identities=12, scanners=3
    )
    result = experiment.repeated_scanner_probe(
        data["scanner_free"],
        data["scanner_ids"],
        data["identity_ids"],
        data["split"],
        seeds=(7, 8, 9),
        probe_config=_scanner_probe_config(),
        include_permutation_null=False,
    )
    assert result["observed_balanced_accuracy_max"] <= result["chance_level"] + 0.10


def test_scanner_positive_control_is_detected() -> None:
    data = experiment.make_scanner_control_features(
        seed=8202, training_identities=24, test_identities=12, scanners=3
    )
    result = experiment.repeated_scanner_probe(
        data["scanner_positive"],
        data["scanner_ids"],
        data["identity_ids"],
        data["split"],
        seeds=(10,),
        probe_config=_scanner_probe_config(),
        include_permutation_null=False,
    )
    assert result["observed_balanced_accuracy_min"] >= 0.90
    assert result["observed_macro_f1_min"] >= 0.90


def test_paired_permutation_null_is_deterministic() -> None:
    data = experiment.make_scanner_control_features(
        seed=8203, training_identities=12, test_identities=6, scanners=3
    )
    first = experiment.identity_aware_permuted_scanner_labels(
        data["scanner_ids"], data["identity_ids"], data["split"], 11
    )
    second = experiment.identity_aware_permuted_scanner_labels(
        data["scanner_ids"], data["identity_ids"], data["split"], 11
    )
    assert np.array_equal(first, second)
    assert not np.array_equal(
        first[data["split"].probe_training_indices],
        data["scanner_ids"][data["split"].probe_training_indices],
    )


def test_residual_decoder_epoch_zero_equals_linear_baseline() -> None:
    dataset, split = _small_synthetic()
    inputs = experiment.true_factor_decoder_inputs(dataset, "correct")
    fit = experiment.fit_residual_regressor(
        inputs,
        dataset.observations,
        split,
        seed=12,
        config=experiment.DecoderConfig(maximum_epochs=0),
    )
    test = split.unseen_test_indices
    assert fit.epoch_zero_max_abs_difference == 0.0
    assert np.array_equal(fit.predict(inputs[test]), fit.ridge_predict(inputs[test]))


def test_true_factor_decoder_uses_correct_target_scanner() -> None:
    dataset, _ = _small_synthetic()
    correct = experiment.true_factor_decoder_inputs(dataset, "correct")
    biological_dim = dataset.biological_latents.shape[1]
    assert np.array_equal(correct[:, :biological_dim], dataset.biological_latents)
    assert np.array_equal(correct[:, biological_dim:], dataset.acquisition_latents)


def test_wrong_scanner_control_changes_acquisition_but_retains_biology() -> None:
    dataset, _ = _small_synthetic()
    correct = experiment.true_factor_decoder_inputs(dataset, "correct")
    wrong = experiment.true_factor_decoder_inputs(dataset, "wrong_scanner")
    biological_dim = dataset.biological_latents.shape[1]
    assert np.array_equal(correct[:, :biological_dim], wrong[:, :biological_dim])
    assert not np.array_equal(correct[:, biological_dim:], wrong[:, biological_dim:])


def test_frozen_input_hashes_are_verified_and_unchanged(tmp_path: Path) -> None:
    before = (experiment._sha256_file(PRIMARY), experiment._sha256_file(FAILED))
    verified = experiment.verify_frozen_inputs(PRIMARY, FAILED)
    after = (experiment._sha256_file(PRIMARY), experiment._sha256_file(FAILED))
    assert before == after
    assert verified["primary_sha256"] == experiment.PRIMARY_SHA256
    corrupted = tmp_path / "failed.json"
    corrupted.write_text("{}", encoding="utf-8")
    with pytest.raises(experiment.ExperimentError):
        experiment.verify_frozen_inputs(PRIMARY, corrupted)


def test_oracle_ridge_positive_nonlinear_negative_fails_preservation() -> None:
    assert not experiment.ridge_positive_preserved(0.85, 0.79)
    result = experiment.aggregate_status(True, True, True, False, True)
    assert result["status"] == "oracle_representation_calibration_failed"


def test_output_directory_cannot_be_overwritten(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()
    with pytest.raises(experiment.ExperimentError):
        experiment.ensure_new_output_root(output)


def test_heterogeneous_summary_csv_fields_are_supported(tmp_path: Path) -> None:
    rows = [{"section": "a", "metric_a": 1.0}, {"section": "b", "metric_b": 2.0}]
    path = tmp_path / "summary.csv"
    experiment.parent.atomic_csv(
        path, experiment.parent.summary_csv_fieldnames(rows), rows
    )
    header = path.read_text(encoding="utf-8").splitlines()[0]
    assert "metric_a" in header and "metric_b" in header


@pytest.mark.parametrize(
    ("flags", "expected"),
    [
        ((False, True, True, True, True), "regression_probe_calibration_failed"),
        ((True, False, True, True, True), "scanner_probe_calibration_failed"),
        ((True, True, False, True, True), "decoder_calibration_failed"),
        ((True, True, True, False, True), "oracle_representation_calibration_failed"),
        ((True, True, True, True, False), "instrument_calibration_failed"),
    ],
)
def test_aggregate_status_fails_closed_if_any_instrument_family_fails(
    flags: tuple[bool, bool, bool, bool, bool], expected: str
) -> None:
    assert experiment.aggregate_status(*flags)["status"] == expected
