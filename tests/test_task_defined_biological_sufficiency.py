from __future__ import annotations

import csv
import inspect
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from experiments.paired_acquisition import (
    run_task_defined_biological_sufficiency as benchmark,
)
from experiments.paired_acquisition import (
    run_unseen_identity_crossed_generalization as unseen,
)
from experiments.paired_acquisition import (
    run_unseen_identity_representation_geometry as geometry,
)


REPOSITORY = Path(__file__).resolve().parents[1]
AUDIT = REPOSITORY / (
    "results/finite_sample_whitening_identifiability_audit_20260802T222234/"
    "finite_sample_whitening_identifiability_audit_result.json"
)


@pytest.fixture(scope="module")
def frozen() -> dict:
    return benchmark.verify_identifiability_audit(AUDIT)


@pytest.fixture(scope="module")
def calibration() -> dict:
    return benchmark.build_task_calibration()


@pytest.fixture(scope="module")
def dataset_and_split(frozen: dict) -> tuple:
    config = unseen.ExperimentConfig(**frozen["factorial"]["calibrated"]["payload"]["config"])
    dataset = unseen.make_unseen_identity_dataset(replace(config, dataset_seed=4301), "linear")
    probe_config = frozen["factorial"]["calibrated"]["upstream"]["payloads"][
        "v1_calibration"
    ]["scanner_probe_config"]
    split = geometry.make_probe_identity_split(
        dataset, 4301 + 700_000, probe_config["validation_fraction"]
    )
    return dataset, split


def test_audit_and_all_upstream_hashes_verify(frozen: dict) -> None:
    assert frozen["file_sha256"] == benchmark.AUDIT_FILE_SHA256
    assert frozen["payload"]["result_sha256"] == benchmark.AUDIT_INTERNAL_SHA256
    assert frozen["factorial"]["file_sha256"] == frozen["payload"][
        "frozen_factorial_artifact"
    ]["file_sha256_after"]
    assert frozen["factorial"]["calibrated"]["upstream"]["hashes"]


def test_exactly_eight_crossed_target_fits_are_scheduled() -> None:
    schedule = benchmark.scheduled_factorizer_runs()
    assert len(schedule) == 8
    assert {row[2] for row in schedule} == {"crossed_target_prototype"}


def test_no_other_factorizer_family_can_be_built() -> None:
    benchmark.validate_model_family("crossed_target_prototype")
    with pytest.raises(benchmark.BenchmarkError, match="Only crossed_target_prototype"):
        benchmark.validate_model_family("oracle")


def test_exact_factorizer_replication_is_required() -> None:
    source = inspect.getsource(benchmark.run_experiment)
    assert "compare_replication" in source
    assert "Factorizer reference replication failed closed" in source


def test_biological_task_labels_are_evaluation_only() -> None:
    source = inspect.getsource(benchmark.run_experiment)
    training_call = source.index("parent.train_model")
    assert "labels" not in source[training_call : source.index("unseen.evaluate_model", training_call)]
    assert '"factorizer_training_reads_task_labels": False' in source


def test_task_calibration_is_independent(calibration: dict) -> None:
    manifest = benchmark.calibration_manifest(calibration)
    assert manifest["sample_count"] == 100_000
    assert manifest["dimension"] == 8
    assert manifest["independent_from_experimental_identities"]


def test_teacher_parameters_are_deterministic(calibration: dict) -> None:
    second = benchmark.build_task_calibration()
    assert calibration["regression_teacher"].parameter_sha256 == second[
        "regression_teacher"
    ].parameter_sha256
    assert calibration["classification_teacher"].parameter_sha256 == second[
        "classification_teacher"
    ].parameter_sha256


def test_teacher_parameters_are_read_only(calibration: dict) -> None:
    parameters = (
        *calibration["regression_teacher"].weights,
        *calibration["regression_teacher"].biases,
    )
    assert all(not parameter.flags.writeable for parameter in parameters)


def test_class_thresholds_come_from_calibration(calibration: dict) -> None:
    scores = calibration["classification_teacher"](calibration["biological"]).reshape(-1)
    assert np.allclose(calibration["class_thresholds"], np.quantile(scores, (0.2, 0.4, 0.6, 0.8)))
    assert calibration["class_balance"] == [20_000] * 5


def test_label_budgets_count_identities(dataset_and_split: tuple) -> None:
    _, split = dataset_and_split
    subsets = benchmark.nested_identity_subsets(split.probe_training_identities, 8101)
    assert [len(subsets[value]) for value in benchmark.LABEL_BUDGETS] == [8, 16, 32]


def test_labeled_subsets_are_nested(dataset_and_split: tuple) -> None:
    _, split = dataset_and_split
    subsets = benchmark.nested_identity_subsets(split.probe_training_identities, 8102)
    assert set(subsets[8]) < set(subsets[16]) < set(subsets[32])


def test_scanner_repeats_do_not_cross_partitions(dataset_and_split: tuple) -> None:
    dataset, split = dataset_and_split
    partitions = [
        set(split.probe_training_identities),
        set(split.probe_validation_identities),
        set(split.unseen_test_identities),
    ]
    assert not (partitions[0] & partitions[1] or partitions[0] & partitions[2] or partitions[1] & partitions[2])
    for identities in partitions:
        assert set(np.unique(dataset.identity_ids[np.isin(dataset.identity_ids, list(identities))])) == identities


def test_balanced_assignment_is_target_independent(dataset_and_split: tuple) -> None:
    dataset, split = dataset_and_split
    indices, manifest = benchmark.balanced_view_indices(
        dataset, split.probe_training_identities[:16], 41
    )
    counts = np.bincount(dataset.scanner_ids[indices], minlength=5)
    assert counts.max() - counts.min() <= 1
    assert manifest["target_independent"]


def test_confounding_assigns_scanner_equal_to_class(
    dataset_and_split: tuple, calibration: dict
) -> None:
    dataset, split = dataset_and_split
    labels = benchmark.labels_by_identity(dataset, calibration, 4301)["classification"]
    indices, manifest = benchmark.class_assigned_view_indices(
        dataset, split.probe_training_identities, labels, 0
    )
    assert np.array_equal(dataset.scanner_ids[indices], labels[indices])
    assert manifest["scanner_equals_class"]


def test_anti_confounding_shifts_scanner_and_preserves_class(
    dataset_and_split: tuple, calibration: dict
) -> None:
    dataset, split = dataset_and_split
    labels = benchmark.labels_by_identity(dataset, calibration, 4301)["classification"]
    indices, manifest = benchmark.class_assigned_view_indices(
        dataset, split.unseen_test_identities, labels, 2
    )
    assert np.array_equal(dataset.scanner_ids[indices], (labels[indices] + 2) % 5)
    assert manifest["class_preserved"]


def test_all_scalers_use_only_labeled_training_indices() -> None:
    regression = inspect.getsource(benchmark.fit_regression_probes)
    classification = inspect.getsource(benchmark.fit_classification_probes)
    assert "residual_probe_result(\n            features, targets, split" in regression
    assert "input_scaler_fit_index_sha256" in regression
    assert "fit_training_scaler(features, split.probe_training_indices)" in classification


def test_oracle_latent_is_evaluation_only() -> None:
    source = inspect.getsource(benchmark.run_experiment)
    assert '"oracle_latent_is_evaluation_only": True' in source
    assert "oracle_biological_latent" not in inspect.getsource(benchmark.counterfactual_biological_codes)


def test_acquisition_code_has_no_identity_specific_donor_variation(dataset_and_split: tuple) -> None:
    dataset, _ = dataset_and_split
    prototypes = np.arange(40, dtype=np.float32).reshape(5, 8)
    acquisition = prototypes[dataset.scanner_ids]
    for scanner in range(5):
        assert np.unique(acquisition[dataset.scanner_ids == scanner], axis=0).shape[0] == 1


def test_identity_permutation_is_identity_level(dataset_and_split: tuple) -> None:
    dataset, split = dataset_and_split
    features = np.column_stack((dataset.identity_ids, dataset.scanner_ids)).astype(np.float32)
    permuted, manifest = benchmark.identity_permuted_features(features, dataset, split, 29)
    assert manifest["permutation_unit"] == "identity"
    for identity, donor in manifest["mapping"].items():
        rows = dataset.identity_ids == identity
        assert np.all(permuted[rows, 0] == donor)
        assert np.array_equal(permuted[rows, 1], dataset.scanner_ids[rows])


def test_regression_uses_both_calibrated_residual_seeds() -> None:
    assert benchmark.RESIDUAL_SEEDS == (7203, 7204)
    assert "for seed in RESIDUAL_SEEDS" in inspect.getsource(benchmark.fit_regression_probes)


def test_classification_uses_all_fixed_classifier_seeds() -> None:
    assert benchmark.CLASSIFIER_SEEDS == (7301, 7302, 7303)
    assert "for seed in CLASSIFIER_SEEDS" in inspect.getsource(benchmark.fit_classification_probes)


def test_worst_scanner_regression_metric_is_minimum() -> None:
    identity = np.repeat(np.arange(4), 5)
    scanner = np.tile(np.arange(5), 4)
    dataset = SimpleNamespace(identity_ids=identity, scanner_ids=scanner)
    truth = np.column_stack((identity, identity + 1.0))
    prediction = truth.copy()
    prediction[scanner == 4] = 0.0
    result = benchmark.regression_prediction_metrics(truth, prediction, np.arange(20), dataset)
    assert result["worst_scanner_r2"] == min(row["r2"] for row in result["per_scanner"])


def test_label_efficiency_integration_uses_identity_budgets() -> None:
    assert benchmark.label_efficiency_area((8, 16, 32), (0.2, 0.5, 0.8)) == pytest.approx(0.5)


def _regression_record(score: float, worst: float | None = None) -> dict:
    return {
        "probes": {
            "residual_repeats": [
                {"metrics": {"r2": score, "worst_scanner_r2": score if worst is None else worst}},
                {"metrics": {"r2": score, "worst_scanner_r2": score if worst is None else worst}},
            ]
        }
    }


def test_regression_sufficiency_requires_both_subset_records() -> None:
    assert benchmark.regression_task_sufficient(
        [_regression_record(0.9), _regression_record(0.9)],
        [_regression_record(0.95), _regression_record(0.95)],
    )
    assert not benchmark.regression_task_sufficient(
        [_regression_record(0.9)], [_regression_record(0.95), _regression_record(0.95)]
    )


def test_broad_sufficiency_requires_all_four_tasks() -> None:
    source = inspect.getsource(benchmark.run_experiment)
    assert "all(task_sufficiency.values())" in source


def test_acquisition_exclusion_requires_every_task_to_fail() -> None:
    source = inspect.getsource(benchmark.run_experiment)
    assert "acquisition_excluded = all(acquisition_failures.values())" in source


def test_one_favorable_task_cannot_establish_broad_sufficiency() -> None:
    source = inspect.getsource(benchmark.run_experiment)
    assert "any(task_sufficiency.values())" not in source[source.index("broad = bool(") : source.index("acquisition_excluded")]


def test_confounding_robustness_uses_anti_confounded_performance() -> None:
    source = inspect.getsource(benchmark.run_experiment)
    assert "anti_accuracy >= 0.70" in source
    assert "balanced_accuracy - anti_accuracy <= 0.10" in source


def test_shortcut_uses_validation_to_anti_confounded_drop() -> None:
    source = inspect.getsource(benchmark.run_experiment)
    assert "validation_accuracy - anti_accuracy >= 0.20" in source


def test_counterfactual_decode_uses_requested_target_scanner() -> None:
    source = inspect.getsource(benchmark.counterfactual_biological_codes)
    assert "model.acquisition_from_scanner(requested_scanner)" in source
    assert '"counterfactual_decode_uses_requested_target_scanner": True' in source


def test_counterfactual_reencoding_occurs_after_decode() -> None:
    source = inspect.getsource(benchmark.counterfactual_biological_codes)
    assert source.index("generated = model.decode") < source.index("model.encode_biological(generated)")


def test_counterfactual_preservation_uses_worst_pair() -> None:
    source = inspect.getsource(benchmark.evaluate_counterfactual_tasks)
    assert "worst_scanner_pair_r2" in source
    assert "worst_scanner_pair_balanced_accuracy" in source


def test_poor_scientific_performance_is_not_execution_failure() -> None:
    empty_flags = {
        "task_defined_representation_supported": False,
        "broad_biological_task_sufficiency": False,
        "scanner_confounding_robust": False,
    }
    outcome = benchmark.aggregate_status([{"interpretation_flags": empty_flags}] * 8)
    assert outcome["status"] == "complete_task_defined_biological_sufficiency_unsupported"
    assert outcome["execution_valid"]


def test_previous_statuses_remain_immutable() -> None:
    assert benchmark.AUDIT_STATUS == "complete_partial_finite_sample_whitening_support"
    assert benchmark.AUDIT_STATUS not in {
        "complete_task_defined_biological_representation_supported",
        "complete_mixed_task_defined_biological_sufficiency",
    }


def test_existing_output_directory_cannot_be_overwritten(tmp_path: Path) -> None:
    with pytest.raises(benchmark.BenchmarkError, match="overwrite"):
        benchmark.ensure_new_output_root(tmp_path)


def test_heterogeneous_summary_csv_fields_are_supported(tmp_path: Path) -> None:
    rows = [{"a": 1}, {"a": 2, "b": 3}]
    path = tmp_path / "summary.csv"
    benchmark.parent.atomic_csv(path, benchmark.parent.summary_csv_fieldnames(rows), rows)
    with path.open(encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle))[0]["b"] == ""


def test_all_frozen_artifacts_remain_unchanged() -> None:
    before = benchmark.verify_identifiability_audit(AUDIT)
    after = benchmark.verify_identifiability_audit(AUDIT)
    assert before["file_sha256"] == after["file_sha256"]
    assert before["factorial"]["calibrated"]["upstream"]["hashes"] == after[
        "factorial"
    ]["calibrated"]["upstream"]["hashes"]


def test_calibration_linear_matrix_is_full_rank_with_frozen_singular_values(calibration: dict) -> None:
    singular = np.linalg.svd(calibration["linear_matrix"], compute_uv=False)
    assert np.linalg.matrix_rank(calibration["linear_matrix"]) == 4
    assert np.allclose(singular, np.linspace(1.2, 0.8, 4))


def test_task_labels_are_constant_across_scanner_views(
    dataset_and_split: tuple, calibration: dict
) -> None:
    dataset, _ = dataset_and_split
    labels = benchmark.labels_by_identity(dataset, calibration, 4301)
    for values in labels.values():
        for identity in np.unique(dataset.identity_ids):
            assert np.unique(values[dataset.identity_ids == identity], axis=0).shape[0] == 1


def test_empty_confounded_validation_scanner_slice_stays_finite() -> None:
    dataset = SimpleNamespace(identity_ids=np.arange(4), scanner_ids=np.zeros(4, dtype=int))
    truth = np.asarray([0, 1, 2, 3])
    probabilities = np.eye(5)[truth]
    result = benchmark.classification_metrics(truth, probabilities, np.arange(4), dataset)
    assert result["per_scanner"][4]["sample_count"] == 0
    assert np.isfinite(result["worst_scanner_balanced_accuracy"])


def test_no_path_forward_gate_is_created() -> None:
    assert "path_forward_gate_open" not in inspect.getsource(benchmark)


def test_canonical_coordinates_are_not_a_success_requirement() -> None:
    source = inspect.getsource(benchmark.run_experiment)
    assert '"canonical_generator_coordinates_not_a_success_requirement": True' in source
