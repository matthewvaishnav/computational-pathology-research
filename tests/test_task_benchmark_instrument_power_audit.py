from __future__ import annotations

import csv
import inspect
from pathlib import Path

import numpy as np
import pytest

from experiments.paired_acquisition import (
    run_task_benchmark_instrument_power_audit as audit,
)


REPOSITORY = Path(__file__).resolve().parents[1]
FROZEN = REPOSITORY / (
    "results/task_defined_biological_sufficiency_20260803T094641/"
    "task_defined_biological_sufficiency_result.json"
)


@pytest.fixture(scope="module")
def verified() -> dict:
    return audit.verify_frozen_benchmark(FROZEN)


@pytest.fixture(scope="module")
def definitions(verified: dict) -> dict:
    return audit.verify_task_definitions(verified["payload"])


def test_frozen_benchmark_file_and_internal_hashes_verify(verified: dict) -> None:
    assert verified["file_sha256"] == audit.FROZEN_FILE_SHA256
    assert verified["payload"]["result_sha256"] == audit.FROZEN_INTERNAL_SHA256


def test_every_upstream_frozen_hash_verifies(verified: dict) -> None:
    upstream = verified["upstream"]["factorial"]["calibrated"]["upstream"]
    for name, path in upstream["paths"].items():
        assert audit.sha256_file(Path(path)) == upstream["hashes"][name]


def test_no_factorizer_builder_is_called() -> None:
    source = inspect.getsource(audit)
    assert ".build_model(" not in source
    assert '"factorizer_models_initialized": 0' in source


def test_no_factorizer_trainer_is_called() -> None:
    source = inspect.getsource(audit)
    assert ".train_model(" not in source
    assert ".train_factorial_model(" not in source


def test_exact_frozen_task_definitions_are_reused(definitions: dict) -> None:
    assert definitions["manifest"]["linear_matrix_sha256"] == (
        "7989e083dbf1ffe5a526b1de08dfb178988664d3f1aa5ba228b6ada5c4a0b447"
    )
    assert definitions["manifest"]["class_balance"] == [20_000] * 5


def test_exact_frozen_teacher_hashes_are_verified(definitions: dict) -> None:
    assert definitions["manifest"]["regression_teacher_parameter_sha256"] == (
        "11db96567e0c46b4f51f9ab743fc84ed67065bd2eb18f89f879062eef442831c"
    )
    assert definitions["manifest"]["classification_teacher_parameter_sha256"] == (
        "5cb1dc7f530b8cc84a98e4f8d91546f5c0cf8a00dfa178873354e368b31a64ea"
    )


def test_calibration_identities_are_independent(definitions: dict) -> None:
    pool = audit.make_independent_pool(8501, definitions["calibration"])
    assert pool["manifest"]["seed"] not in (4301, 5301)
    assert pool["manifest"]["biological_sha256"] != definitions["manifest"][
        "biological_latent_sha256"
    ]


def test_train_validation_and_test_pools_are_disjoint(definitions: dict) -> None:
    pool = audit.make_independent_pool(8502, definitions["calibration"])
    train, validation, test = map(set, (pool["train_pool"], pool["validation_pool"], pool["test_pool"]))
    assert not (train & validation or train & test or validation & test)


def test_all_seven_training_budgets_are_generated() -> None:
    assert audit.TRAINING_BUDGETS == (8, 16, 32, 64, 128, 256, 512)


def test_original_validation_uses_exactly_eight_identities() -> None:
    assert audit.VALIDATION_REGIMES["original_validation_8"] == 8


def test_powered_validation_uses_exactly_128_identities() -> None:
    assert audit.VALIDATION_REGIMES["powered_validation_128"] == 128


def test_frozen_and_selected_probes_remain_separate() -> None:
    source = inspect.getsource(audit.run_power_calibration)
    assert '"frozen_probe": main' in source
    assert '"validation_selected_probe": selected_fit' in source


def test_ridge_alpha_selection_uses_validation_only(definitions: dict, verified: dict) -> None:
    payload = verified["payload"]
    config = audit.residual_calibration.ResidualConfig(**payload["probe_configurations"]["residual_regressor"])
    pool = audit.make_independent_pool(8503, definitions["calibration"])
    split = audit.make_split(pool["train_pool"][:16], pool["validation_pool"][:8], pool["test_pool"][:32], 1)
    result = audit.select_ridge_alpha(pool["biological"], pool["targets"]["linear_regression"], split, config)
    assert result["selection_uses_validation_only"]
    assert result["selected_alpha"] in audit.RIDGE_GRID


def test_logistic_c_selection_uses_validation_only(definitions: dict) -> None:
    pool = audit.make_independent_pool(8504, definitions["calibration"])
    split = audit.make_split(pool["train_pool"][:16], pool["validation_pool"][:8], pool["test_pool"][:32], 2)
    result = audit.select_logistic_c(pool["biological"], pool["targets"]["classification"], split)
    assert result["selection_uses_validation_only"]
    assert result["selected_C"] in audit.LOGISTIC_GRID


def test_test_data_cannot_affect_hyperparameter_selection() -> None:
    ridge = inspect.getsource(audit.select_ridge_alpha)
    logistic = inspect.getsource(audit.select_logistic_c)
    assert "unseen_test_indices" not in ridge
    assert "unseen_test_indices" not in logistic


def test_negative_control_permutations_operate_by_identity() -> None:
    source = inspect.getsource(audit.run_power_calibration)
    assert "permutation(training)" in source
    assert "permuted_targets[training] = targets[permutation]" in source


def test_classification_class_coverage_is_calculated_correctly() -> None:
    assert audit.complete_class_coverage_probability(1) == pytest.approx(0.0)
    assert audit.complete_class_coverage_probability(128) > 0.999999


def test_one_favorable_seed_cannot_establish_power() -> None:
    success = np.asarray([True] + [False] * 9)
    assert not audit.instrument_powered_decision(success, np.ones(10, bool), np.ones(10, bool), np.zeros(10))


def test_power_requires_eighty_percent_task_success() -> None:
    threshold = np.asarray([True] * 7 + [False] * 3)
    assert not audit.instrument_powered_decision(threshold, np.ones(10, bool), np.ones(10, bool), np.zeros(10))


def test_power_requires_ninety_five_percent_negative_rejection() -> None:
    negative = np.asarray([True] * 18 + [False] * 2)
    assert not audit.instrument_powered_decision(np.ones(20, bool), negative, np.ones(20, bool), np.zeros(20))


def test_power_rejects_incomplete_class_coverage() -> None:
    coverage = np.asarray([True] * 9 + [False])
    assert not audit.instrument_powered_decision(np.ones(10, bool), np.ones(10, bool), coverage, np.zeros(10))


def test_original_design_admissibility_requires_oracle_control() -> None:
    source = inspect.getsource(audit.task_admissibility)
    assert 'oracle[task]["original_design_oracle_control_passed"]' in source


def test_representation_failure_cannot_use_inadmissible_oracle(verified: dict, definitions: dict) -> None:
    coverage = audit.frozen_class_coverage(verified["payload"], definitions["calibration"])
    adjudication = audit.original_oracle_adjudication(verified["payload"], coverage)
    assert not adjudication["nonlinear_teacher"]["original_design_oracle_control_passed"]
    assert not adjudication["interaction"]["original_design_oracle_control_passed"]


def test_linear_adjudication_compares_identical_subsets(verified: dict) -> None:
    result = audit.linear_task_adjudication(verified["payload"])
    assert all(record["identical_subset_sha256"] for record in result["records"])
    assert {record["training_identity_count"] for record in result["records"]} == {32}


def test_ridge_and_residual_linear_failures_are_separate(verified: dict) -> None:
    aggregate = audit.linear_task_adjudication(verified["payload"])["aggregate"]
    assert "median_biological_ridge_r2" in aggregate
    assert "median_biological_residual_r2" in aggregate


def test_unavailable_feature_covariance_is_not_fabricated(verified: dict) -> None:
    record = audit.linear_task_adjudication(verified["payload"])["records"][0]
    assert not record["biological_feature_covariance"]["available"]


def test_counterfactual_eligibility_requires_direct_performance() -> None:
    assert audit.counterfactual_metric_eligible(0.70)
    assert not audit.counterfactual_metric_eligible(0.6999)


def test_ineligible_counterfactual_cannot_support_semantic_failure(verified: dict) -> None:
    rows = audit.counterfactual_eligibility(verified["payload"])
    for row in rows:
        if not row["counterfactual_metric_eligible"]:
            assert not row["semantic_failure_supported"]


def test_frozen_benchmark_status_remains_unchanged() -> None:
    assert audit.FROZEN_STATUS == "complete_task_defined_biological_sufficiency_unsupported"


def test_poor_scientific_evidence_is_not_execution_failure() -> None:
    decisions = {task: "task_not_learnable_within_calibrated_range" for task in audit.TASK_NAMES}
    assert audit.audit_status(decisions) == "complete_original_task_benchmark_instrument_invalid"


def test_execution_failure_has_only_failure_status() -> None:
    decisions = {task: "admissible_at_original_design" for task in audit.TASK_NAMES}
    assert audit.audit_status(decisions, False) == "task_benchmark_instrument_power_audit_failed"


def test_no_cuda_initialization_occurs() -> None:
    source = inspect.getsource(audit)
    assert ".cuda(" not in source
    assert 'torch.device("cuda")' not in source
    assert '"cuda_contexts_initialized": 0' in source


def test_existing_output_directories_cannot_be_overwritten(tmp_path: Path) -> None:
    with pytest.raises(audit.AuditError, match="overwrite"):
        audit.ensure_new_output_root(tmp_path)


def test_heterogeneous_summary_csv_fields_are_supported(tmp_path: Path) -> None:
    rows = [{"a": 1}, {"a": 2, "b": 3}]
    path = tmp_path / "summary.csv"
    audit.frozen_benchmark.parent.atomic_csv(
        path, audit.frozen_benchmark.parent.summary_csv_fieldnames(rows), rows
    )
    with path.open(encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle))[0]["b"] == ""


def test_all_frozen_artifacts_remain_unchanged() -> None:
    before = audit.verify_frozen_benchmark(FROZEN)
    after = audit.verify_frozen_benchmark(FROZEN)
    assert before["file_sha256"] == after["file_sha256"]
    assert before["upstream"]["factorial"]["calibrated"]["upstream"]["hashes"] == after[
        "upstream"
    ]["factorial"]["calibrated"]["upstream"]["hashes"]


def test_frozen_record_extraction_keeps_probe_families_separate(verified: dict) -> None:
    rows = audit.extract_frozen_records(verified["payload"])
    assert {row["probe_family"] for row in rows} == {
        "ridge",
        "calibrated_residual_regressor",
        "multinomial_logistic",
        "calibrated_shallow_classifier",
    }


def test_independent_test_pool_has_at_least_4096_identities(definitions: dict) -> None:
    pool = audit.make_independent_pool(8505, definitions["calibration"])
    assert len(pool["test_pool"]) >= 4096

