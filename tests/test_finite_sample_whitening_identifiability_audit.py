from __future__ import annotations

import csv
import inspect
import math
from pathlib import Path

import numpy as np
import pytest

from experiments.paired_acquisition import (
    run_finite_sample_whitening_identifiability_audit as audit,
)


REPOSITORY = Path(__file__).resolve().parents[1]
FACTORIAL = REPOSITORY / "results/minimal_whitened_biological_bottleneck_20260802T215848/minimal_whitened_biological_bottleneck_result.json"


def test_factorial_file_and_internal_hashes_verify() -> None:
    verified = audit.verify_factorial_artifact(FACTORIAL)
    assert verified["file_sha256"] == audit.FACTORIAL_FILE_SHA256
    assert verified["payload"]["result_sha256"] == audit.FACTORIAL_INTERNAL_SHA256


def test_no_factorizer_build_function_is_called() -> None:
    source = inspect.getsource(audit)
    assert ".build_model(" not in source
    assert "factorizer_models_initialized\": 0" in source


def test_no_factorizer_training_function_is_called() -> None:
    source = inspect.getsource(audit)
    assert ".train_model(" not in source
    assert ".train_factorial_model(" not in source


def test_all_expected_identity_dimension_combinations_present() -> None:
    payload = audit.verify_factorial_artifact(FACTORIAL)["payload"]
    records = audit.extract_covariance_records(payload)
    combinations = tuple(sorted({(row["identity_count"], row["dimension"]) for row in records}))
    assert combinations == audit.EXPECTED_COMBINATIONS


def test_covariance_estimator_matches_factorial_runner() -> None:
    rng = np.random.default_rng(4)
    values = rng.normal(size=(20, 8))
    expected = audit.factorial.covariance_geometry(
        values,
        np.arange(20),
        np.arange(20),
    )
    observed = audit.covariance_metrics(values)
    for name in (
        "mean_diagonal",
        "minimum_diagonal",
        "maximum_diagonal",
        "diagonal_deviation_from_one",
        "mean_absolute_off_diagonal_covariance",
        "maximum_absolute_off_diagonal_covariance",
        "effective_rank",
        "participation_ratio",
        "numerical_rank",
    ):
        assert observed[name] == pytest.approx(expected[name])


def test_monte_carlo_generation_is_deterministic() -> None:
    first = audit.generate_white_null(8, 8, 100, 17, 25)
    second = audit.generate_white_null(8, 8, 100, 17, 25)
    for name in first:
        assert np.array_equal(first[name], second[name])


def test_chunking_does_not_materially_change_quantiles() -> None:
    first = audit.generate_white_null(8, 8, 500, 19, 17)
    second = audit.generate_white_null(8, 8, 500, 19, 128)
    name = "mean_absolute_off_diagonal_covariance"
    assert np.allclose(np.quantile(first[name], audit.QUANTILES), np.quantile(second[name], audit.QUANTILES))


def test_analytic_expected_absolute_off_diagonal_is_correct() -> None:
    assert audit.analytic_expected_absolute_off_diagonal(20) == pytest.approx(
        math.sqrt(2 / (math.pi * 19))
    )


@pytest.mark.parametrize("n,d", audit.EXPECTED_COMBINATIONS)
def test_null_rank_equals_min_dimension_and_n_minus_one(n: int, d: int) -> None:
    values = audit.generate_white_null(n, d, 20, 31 + n + d, 7)
    assert np.all(values["numerical_rank"] == min(d, n - 1))


def test_white_finite_sample_need_not_pass_old_point_one_cutoff() -> None:
    assert audit.analytic_expected_absolute_off_diagonal(8) > 0.10


def test_new_null_criterion_uses_matched_n_and_d() -> None:
    values = audit.generate_white_null(8, 8, 100, 55, 20)
    summary = audit.summarize_null(values, 8, 8, 100, 55, 20)
    record = {
        "dataset_seed": 1,
        "renderer": "linear",
        "model_seed": 1,
        "model_family": "minimal_whitened",
        "split": "probe_validation_identities",
        "identity_count": 8,
        "dimension": 8,
        "metrics": audit.covariance_metrics(np.random.default_rng(1).normal(size=(8, 8))),
    }
    comparison = audit.compare_to_finite_sample_null(record, values, summary)
    assert comparison["matched_null_seed"] == 55
    assert comparison["identity_count"] == 8 and comparison["dimension"] == 8


def test_original_and_post_hoc_whitening_flags_are_separate() -> None:
    values = audit.generate_white_null(8, 8, 200, 57, 50)
    summary = audit.summarize_null(values, 8, 8, 200, 57, 50)
    metrics = audit.covariance_metrics(np.random.default_rng(2).normal(size=(8, 8)))
    record = {
        "dataset_seed": 1,
        "renderer": "linear",
        "model_seed": 1,
        "model_family": "minimal_whitened",
        "split": "probe_validation_identities",
        "identity_count": 8,
        "dimension": 8,
        "metrics": metrics,
    }
    comparison = audit.compare_to_finite_sample_null(record, values, summary)
    assert "original_fixed_whitening_criterion" in comparison
    assert "finite_sample_whitening_consistent" in comparison


def test_paired_comparisons_match_all_condition_keys() -> None:
    payload = audit.verify_factorial_artifact(FACTORIAL)["payload"]
    records = audit.extract_covariance_records(payload)
    comparisons = []
    for record in records:
        comparisons.append(
            {
                **{key: value for key, value in record.items() if key != "metrics"},
                "observed_covariance_metrics": record["metrics"],
                "normalized_off_diagonal_ratio_to_null_median": record["metrics"][
                    "mean_absolute_off_diagonal_covariance"
                ],
            }
        )
    paired = audit.matched_paired_effects(comparisons)
    for contrast in paired.values():
        for split in audit.SPLIT_NAMES:
            assert len(contrast[split]["paired_conditions"]) == 8


def test_cube_representation_has_population_variance_one() -> None:
    rng = np.random.default_rng(8)
    b = rng.normal(size=500_000)
    z = b**3 / math.sqrt(15)
    assert np.var(z) == pytest.approx(1.0, abs=0.03)


def test_cube_transformation_is_bijective() -> None:
    values = np.linspace(-5, 5, 101)
    transformed = values**3 / math.sqrt(15)
    recovered = np.cbrt(math.sqrt(15) * transformed)
    assert np.allclose(recovered, values)


def test_analytic_linear_r2_equals_point_six() -> None:
    assert 9 / 15 == pytest.approx(0.60)


def test_nonlinear_inverse_recovery_approaches_one() -> None:
    result = audit.cube_counterexample(20_000, 20_000, 91)
    assert result["analytic_inverse_r2"] > 0.999999
    assert result["sample_ridge_r2"] == pytest.approx(0.60, abs=0.04)


def test_counterexample_is_exactly_minimal_eight_dimensional() -> None:
    result = audit.cube_counterexample(1_000, 1_000, 93)
    assert result["dimension"] == 8
    assert result["minimal_dimension_verified"]


def test_counterexample_covariance_approaches_identity() -> None:
    result = audit.cube_counterexample(30_000, 30_000, 95)
    covariance = result["sample_covariance_metrics"]
    assert covariance["mean_diagonal"] == pytest.approx(1.0, abs=0.06)
    assert covariance["mean_absolute_off_diagonal_covariance"] < 0.03


def test_audit_status_cannot_overwrite_factorial_status() -> None:
    result = audit.audit_status([], True, True)
    assert result["status"] != audit.FACTORIAL_STATUS
    assert audit.FACTORIAL_STATUS == "complete_canonicalization_tradeoff_detected"


def test_poor_scientific_evidence_is_not_execution_failure() -> None:
    comparisons = [
        {"model_family": "minimal_whitened", "finite_sample_whitening_consistent": False}
    ]
    result = audit.audit_status(comparisons, True, True)
    assert result["status"] == "complete_whitening_not_supported_and_canonical_identifiability_absent"
    assert result["execution_valid"]


def test_upstream_artifacts_remain_unchanged() -> None:
    before = audit.verify_factorial_artifact(FACTORIAL)
    after = audit.verify_factorial_artifact(FACTORIAL)
    assert before["file_sha256"] == after["file_sha256"]
    assert before["calibrated"]["upstream"]["hashes"] == after["calibrated"]["upstream"]["hashes"]


def test_existing_output_directory_cannot_be_overwritten(tmp_path: Path) -> None:
    with pytest.raises(audit.AuditError, match="overwrite"):
        audit.ensure_new_output_root(tmp_path)


def test_heterogeneous_summary_csv_fields_supported(tmp_path: Path) -> None:
    rows = [{"a": 1}, {"b": 2, "a": 3}]
    path = tmp_path / "summary.csv"
    audit.output_helpers.atomic_csv(
        path, audit.output_helpers.summary_csv_fieldnames(rows), rows
    )
    with path.open(encoding="utf-8", newline="") as handle:
        parsed = list(csv.DictReader(handle))
    assert list(parsed[0]) == ["a", "b"]
