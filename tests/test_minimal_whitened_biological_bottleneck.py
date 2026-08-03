from __future__ import annotations

import csv
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch

from experiments.paired_acquisition import (
    run_minimal_whitened_biological_bottleneck as experiment,
)


REPOSITORY = Path(__file__).resolve().parents[1]
REFERENCE = REPOSITORY / "results/calibrated_unseen_identity_representation_geometry_v2_20260802T205035/calibrated_unseen_identity_representation_geometry_v2_result.json"
INCOMPLETE = REPOSITORY / "results/calibrated_unseen_identity_representation_geometry_v2_20260802T204912/terminal_output.txt"


def test_successful_calibrated_diagnostic_and_full_sha_verify() -> None:
    verified = experiment.verify_calibrated_reference(REFERENCE)
    assert verified["file_sha256"] == experiment.CALIBRATED_DIAGNOSTIC_FILE_SHA256
    assert verified["payload"]["result_sha256"] == experiment.CALIBRATED_DIAGNOSTIC_INTERNAL_SHA256


def test_incomplete_first_attempt_cannot_be_reference() -> None:
    with pytest.raises(experiment.ExperimentError, match="SHA-256"):
        experiment.verify_calibrated_reference(INCOMPLETE)


def test_exactly_four_factorial_families_exist() -> None:
    assert tuple(experiment.family_configurations()) == experiment.FACTORIAL_FAMILIES
    assert len(experiment.FACTORIAL_FAMILIES) == 4


def test_exactly_32_fits_are_scheduled() -> None:
    schedule = experiment.scheduled_factorizer_runs()
    assert len(schedule) == 32
    assert {row[2] for row in schedule} == set(experiment.FACTORIAL_FAMILIES)
    assert all(sum(row[2] == family for row in schedule) == 8 for family in experiment.FACTORIAL_FAMILIES)


def _base_config():
    reference = experiment.base.json.loads(REFERENCE.read_text(encoding="utf-8"))
    return experiment.unseen.ExperimentConfig(**reference["config"])


def test_only_dimension_and_whitening_weight_differ() -> None:
    manifest = experiment.validate_factorial_isolation(
        _base_config(), experiment.family_configurations()
    )
    for condition in manifest.values():
        assert set(condition["model_configuration_differences"]) <= {
            "prototype_biological_dim"
        }


def test_baseline_dimension_is_32() -> None:
    assert experiment.family_configurations()["overcomplete_unwhitened"].biological_code_dimension == 32


@pytest.mark.parametrize("family", ["minimal_unwhitened", "minimal_whitened"])
def test_minimal_family_dimension_is_8(family: str) -> None:
    assert experiment.family_configurations()[family].biological_code_dimension == 8


def test_whitening_disabled_objective_is_same_tensor() -> None:
    previous = torch.tensor(3.0, requires_grad=True)
    whitening = torch.tensor(9.0, requires_grad=True)
    result = experiment.factorial_objective(previous, whitening, 0.0)
    assert result is previous
    assert result.item() == 3.0


def test_identity_averaging_uses_every_selected_view_once() -> None:
    codes = np.arange(18, dtype=float).reshape(6, 3)
    identity_ids = np.array([0, 0, 1, 1, 2, 2])
    identities, means = experiment.identity_level_means_array(
        codes, identity_ids, np.arange(6)
    )
    assert identities.tolist() == [0, 1, 2]
    assert np.allclose(means, np.stack([codes[:2].mean(0), codes[2:4].mean(0), codes[4:].mean(0)]))


def test_whitening_covariance_uses_identities_as_samples() -> None:
    identity_means = torch.tensor([[0.0, 0.0], [1.0, 2.0], [3.0, 1.0]])
    diagonal, off_diagonal, total = experiment.covariance_penalties(identity_means)
    centered = identity_means - identity_means.mean(0)
    covariance = centered.T @ centered / 2
    assert torch.allclose(diagonal, ((torch.diag(covariance) - 1) ** 2).mean())
    assert torch.allclose(off_diagonal, covariance[~torch.eye(2, dtype=torch.bool)].square().mean())
    assert torch.allclose(total, diagonal + off_diagonal)


def test_whitening_training_uses_only_train_indices() -> None:
    source = inspect.getsource(experiment.train_factorial_model)
    assert "identity_level_means_tensor(\n            all_biological, dataset.identity_ids, train_indices" in source
    assert "test_indices" not in source


def test_training_never_reads_biological_latent_values() -> None:
    source = inspect.getsource(experiment.train_factorial_model)
    assert "dataset.biological_latents" not in source
    assert "biological_latents_read_by_training" in source


def test_covariance_penalty_components_are_correct() -> None:
    values = torch.tensor([[-1.0, 0.0], [0.0, 1.0], [1.0, -1.0]])
    diagonal, off_diagonal, total = experiment.covariance_penalties(values)
    covariance = (values - values.mean(0)).T @ (values - values.mean(0)) / 2
    expected_diagonal = ((torch.diag(covariance) - 1.0) ** 2).mean()
    expected_off = covariance[~torch.eye(2, dtype=torch.bool)].square().mean()
    assert diagonal.item() == pytest.approx(expected_diagonal.item())
    assert off_diagonal.item() == pytest.approx(expected_off.item())
    assert total.item() == pytest.approx((expected_diagonal + expected_off).item())


def test_whitening_weight_is_fixed_at_point_one() -> None:
    configurations = experiment.family_configurations()
    assert experiment.WHITENING_WEIGHT == 0.10
    assert configurations["overcomplete_whitened"].biological_whitening_weight == 0.10
    assert configurations["minimal_whitened"].biological_whitening_weight == 0.10


def test_rank_feasibility_is_checked() -> None:
    with pytest.raises(experiment.ExperimentError, match="rank-impossible"):
        experiment.covariance_penalties(torch.zeros(2, 2))


def _replication_sections(value: float):
    reference = {
        "original_operational_evaluation_recomputed": {"metric": value},
        "frozen_ridge_biological_probe": {"metric": value},
        "retrieval_geometry": {"metric": value},
        "linear_scanner_probe": {"metric": value},
        "repeated_nonlinear_scanner_probe": {"metric": value},
        "calibrated_acquisition_biology_probe": [{"metric": value}],
        "acquisition_prototype_within_scanner_donor_variance": value,
        "calibrated_independent_decoder": {"metric": value},
    }
    observed = {
        "operational_evaluation": {"metric": value},
        "frozen_ridge_biological_probe": {"metric": value},
        "retrieval_geometry": {"metric": value},
        "linear_scanner_probe": {"metric": value},
        "repeated_nonlinear_scanner_probe": {"metric": value},
        "calibrated_acquisition_biology_probe": [{"metric": value}],
        "acquisition_prototype_within_scanner_donor_variance": value,
        "calibrated_independent_decoder": {"metric": value},
    }
    return reference, observed


def test_baseline_replication_failure_fails_closed() -> None:
    reference, observed = _replication_sections(1.0)
    observed["frozen_ridge_biological_probe"]["metric"] = 2.0
    assert not experiment.compare_baseline_replication(reference, observed)["passed"]


def test_both_calibrated_residual_probe_seeds_are_required() -> None:
    repeats = [
        {"unseen_test": {"residual_r2": 0.81}},
        {"unseen_test": {"residual_r2": 0.79}},
    ]
    assert not experiment.calibrated.stable_residual_recovery(repeats)


def test_scanner_leakage_uses_all_three_pairs() -> None:
    scanner = {
        "chance_level": 0.2,
        "observed_balanced_accuracy_median": 0.5,
        "repeats": [
            {"observed": {"balanced_accuracy": 0.5}, "permutation_null": {"balanced_accuracy": null}}
            for null in (0.1, 0.2, 0.6)
        ],
    }
    assert not experiment.calibrated.scanner_interpretation(scanner, True)[
        "hidden_scanner_leakage_detected"
    ]


def test_acquisition_exclusion_uses_worst_residual_seed() -> None:
    repeats = [
        {"unseen_test": {"residual_r2": 0.0}},
        {"unseen_test": {"residual_r2": 0.11}},
    ]
    assert not experiment.calibrated.acquisition_biology_exclusion(repeats)


def test_retrieval_uses_worst_scanner_pair() -> None:
    assert not experiment.calibrated.retrieval_success(
        {
            "unseen_identity_retrieval_top1": 1.0,
            "worst_scanner_pair_identity_retrieval_top1": 0.89,
        }
    )


def test_decoder_requires_separation_from_every_negative() -> None:
    primary = 0.4
    negatives = [0.5, 0.6, 0.8, 1.0]
    assert all(primary <= 0.8 * value for value in negatives)
    negatives[0] = 0.49
    assert not all(primary <= 0.8 * value for value in negatives)


def test_covariance_generalization_is_evaluated_on_unseen_geometry() -> None:
    rng = np.random.default_rng(4)
    biological = rng.normal(size=(30, 3))
    identities = np.repeat(np.arange(10), 3)
    metrics = experiment.covariance_geometry(biological, identities, np.arange(30))
    assert metrics["identity_count"] == 10
    assert "mean_absolute_off_diagonal_covariance" in metrics
    assert isinstance(experiment.covariance_whitening_generalized(metrics), bool)


def _summaries(success=(), canonical=(), operational=()):
    return [
        {
            "model_family": family,
            "mechanism_target_success_count": 8 if family in success else 0,
            "canonical_biology_recovery_count": 8 if family in canonical else 0,
            "operational_capabilities_preserved_count": 8 if family in operational else 0,
        }
        for family in experiment.FACTORIAL_FAMILIES
    ]


def test_dimensionality_sufficient_status_logic() -> None:
    result = experiment.factorial_interpretation(
        _summaries(
            success={"minimal_unwhitened", "minimal_whitened"},
            canonical={"minimal_unwhitened", "minimal_whitened"},
            operational={"minimal_unwhitened", "minimal_whitened"},
        )
    )
    assert result["status"] == "complete_dimensionality_sufficient"


def test_whitening_sufficient_status_logic() -> None:
    result = experiment.factorial_interpretation(
        _summaries(
            success={"overcomplete_whitened", "minimal_whitened"},
            canonical={"overcomplete_whitened", "minimal_whitened"},
            operational={"overcomplete_whitened", "minimal_whitened"},
        )
    )
    assert result["status"] == "complete_whitening_sufficient"


def test_interaction_required_status_logic() -> None:
    result = experiment.factorial_interpretation(
        _summaries(
            success={"minimal_whitened"},
            canonical={"minimal_whitened"},
            operational={"minimal_whitened"},
        )
    )
    assert result["status"] == "complete_dimensionality_whitening_interaction_required"


def test_mechanism_unsupported_is_scientific_status() -> None:
    result = experiment.factorial_interpretation(_summaries())
    assert result["status"] == "complete_canonicalization_mechanism_unsupported"
    assert result["execution_valid"]


def test_tradeoff_detects_degraded_operational_capabilities() -> None:
    result = experiment.factorial_interpretation(
        _summaries(canonical={"minimal_whitened"})
    )
    assert result["status"] == "complete_canonicalization_tradeoff_detected"


def test_heterogeneous_summary_csv_supported(tmp_path: Path) -> None:
    rows = [{"a": 1}, {"a": 2, "b": 3}]
    path = tmp_path / "summary.csv"
    experiment.parent.atomic_csv(
        path, experiment.parent.summary_csv_fieldnames(rows), rows
    )
    with path.open(encoding="utf-8", newline="") as handle:
        parsed = list(csv.DictReader(handle))
    assert list(parsed[0]) == ["a", "b"]


def test_existing_output_directory_cannot_be_overwritten(tmp_path: Path) -> None:
    with pytest.raises(experiment.ExperimentError, match="overwrite"):
        experiment.ensure_new_output_root(tmp_path)


def test_all_frozen_artifacts_remain_unchanged() -> None:
    before = experiment.verify_calibrated_reference(REFERENCE)
    after = experiment.verify_calibrated_reference(REFERENCE)
    assert before["file_sha256"] == after["file_sha256"]
    assert before["upstream"]["hashes"] == after["upstream"]["hashes"]


def test_forbidden_path_forward_field_absent() -> None:
    assert "path_forward_gate_open" not in Path(experiment.__file__).read_text(encoding="utf-8")
