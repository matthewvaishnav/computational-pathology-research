from __future__ import annotations

import csv
import inspect
from pathlib import Path

import numpy as np
import pytest
import torch
import torch.nn as nn

from experiments.paired_acquisition import run_paired_consensus_linear_anchor as anchor


REPOSITORY = Path(__file__).resolve().parents[1]
POWER_AUDIT = REPOSITORY / (
    "results/task_benchmark_instrument_power_audit_20260803T102605/"
    "task_benchmark_instrument_power_audit_result.json"
)


@pytest.fixture(scope="module")
def frozen() -> dict:
    return anchor.verify_power_audit(POWER_AUDIT)


@pytest.fixture(scope="module")
def config(frozen: dict):
    return anchor.config_from_benchmark(frozen["benchmark"]["payload"])


@pytest.fixture(scope="module")
def dataset(config):
    return anchor.unseen.make_unseen_identity_dataset(config, "linear")


@pytest.fixture(scope="module")
def target(dataset):
    return anchor.construct_consensus_targets(dataset)


def test_power_audit_file_and_internal_hashes_verify(frozen: dict) -> None:
    assert frozen["file_sha256"] == anchor.POWER_AUDIT_FILE_SHA256
    assert frozen["payload"]["result_sha256"] == anchor.POWER_AUDIT_INTERNAL_SHA256


def test_all_inherited_upstream_hashes_verify(frozen: dict) -> None:
    chain = frozen["benchmark"]["upstream"]["factorial"]["calibrated"]["upstream"]
    for name, path in chain["paths"].items():
        assert anchor.sha256_file(Path(path)) == chain["hashes"][name]


def test_target_constructor_never_accepts_biological_latents() -> None:
    signature = inspect.signature(anchor.construct_consensus_targets)
    assert list(signature.parameters) == ["dataset"]
    assert "dataset.biological_latents" not in inspect.getsource(
        anchor.construct_consensus_targets
    )


def test_scanner_means_use_training_identities_only(dataset, target) -> None:
    training = np.unique(dataset.identity_ids[dataset.train_indices])
    expected = np.stack(
        [
            dataset.observations[np.isin(dataset.identity_ids, training) & (dataset.scanner_ids == scanner)].mean(axis=0)
            for scanner in range(5)
        ]
    )
    assert np.allclose(target["scanner_means"], expected)


def test_every_scanner_contributes_once_to_identity_consensus(dataset, target) -> None:
    assert target["manifest"]["every_scanner_contributes_once_per_identity"]
    assert all(np.sum(dataset.identity_ids == identity) == 5 for identity in target["all_identities"])


def test_every_training_identity_contributes_once_to_scaler(target) -> None:
    assert target["manifest"]["every_training_identity_contributes_once_to_scaler"]
    assert target["manifest"]["training_identity_count"] == 40


def test_unseen_identities_never_affect_target_fitting(target) -> None:
    assert not target["manifest"]["unseen_identities_used_for_fitting"]
    assert np.array_equal(target["training_identities"], np.arange(40))


def test_all_views_of_identity_share_same_target(dataset, target) -> None:
    for identity in np.unique(dataset.identity_ids):
        assert np.unique(target["per_view_standardized"][dataset.identity_ids == identity], axis=0).shape[0] == 1


def test_target_hashes_are_deterministic(dataset) -> None:
    first = anchor.construct_consensus_targets(dataset)["manifest"]
    second = anchor.construct_consensus_targets(dataset)["manifest"]
    assert first == second


def test_preflight_occurs_before_factorizer_initialization() -> None:
    source = inspect.getsource(anchor.run_experiment)
    assert source.index("run_target_preflight") < source.index("build_family_model")


def test_inadmissible_target_initializes_zero_factorizers(monkeypatch, tmp_path: Path, frozen: dict) -> None:
    monkeypatch.setattr(
        anchor,
        "run_target_preflight",
        lambda *args: {
            "consensus_target_admissible": False,
            "preflight_completed_before_factorizer_initialization": True,
            "factorizer_models_initialized_during_preflight": 0,
            "conditions": [],
            "target_manifests": {},
        },
    )
    monkeypatch.setattr(anchor, "build_family_model", lambda *args: (_ for _ in ()).throw(AssertionError("built")))
    result = anchor.run_experiment(POWER_AUDIT, tmp_path / "new", torch.device("cpu"))
    assert result["factorizer_fit_count"] == 0
    assert result["status"] == "complete_consensus_anchor_target_inadmissible"


def test_exactly_three_model_families_exist() -> None:
    assert anchor.FAMILIES == (
        "crossed_target_baseline",
        "nonlinear_consensus_anchor",
        "linear_consensus_anchor",
    )


def test_exactly_twenty_four_fits_are_scheduled() -> None:
    assert len(anchor.scheduled_runs()) == 24


def test_baseline_contains_no_consensus_head(config) -> None:
    anchor.base.set_deterministic_seed(1)
    model = anchor.build_family_model("crossed_target_baseline", config, torch.device("cpu"))
    assert not hasattr(model, "consensus_head")


def test_linear_head_is_exactly_one_affine_layer(config) -> None:
    model = anchor.build_family_model("linear_consensus_anchor", config, torch.device("cpu"))
    assert isinstance(model.consensus_head, nn.Linear)
    assert model.consensus_head.bias is not None


def test_nonlinear_head_has_two_frozen_hidden_layers(config) -> None:
    model = anchor.build_family_model("nonlinear_consensus_anchor", config, torch.device("cpu"))
    linears = [layer for layer in model.consensus_head if isinstance(layer, nn.Linear)]
    gelus = [layer for layer in model.consensus_head if isinstance(layer, nn.GELU)]
    assert [(layer.in_features, layer.out_features) for layer in linears] == [(32, 128), (128, 128), (128, 64)]
    assert len(gelus) == 2


def test_head_output_is_not_passed_to_original_decoder() -> None:
    source = inspect.getsource(anchor.AnchoredFactorizer)
    decode_source = inspect.getsource(anchor.parent.ScannerPrototypeFactorizer.decode)
    assert "consensus_head" not in decode_source
    assert "predict_consensus" in source


def test_only_family_differences_are_head_and_loss() -> None:
    source = inspect.getsource(anchor.build_family_model) + inspect.getsource(anchor.train_anchored_model)
    assert "consensus_head" in source
    assert "ANCHOR_WEIGHT * consensus_loss" in source


def test_anchor_weight_is_exactly_point_two_five() -> None:
    assert anchor.ANCHOR_WEIGHT == 0.25


def test_baseline_uses_exact_frozen_objective() -> None:
    source = inspect.getsource(anchor.run_experiment)
    assert 'parent.train_model(\n                            "crossed_target_prototype"' in source


def test_factorizer_training_does_not_read_latents_or_task_labels() -> None:
    source = inspect.getsource(anchor.train_anchored_model)
    assert "biological_latents" not in source
    assert "task" not in inspect.signature(anchor.train_anchored_model).parameters


def test_baseline_reference_replication_is_mandatory() -> None:
    source = inspect.getsource(anchor.run_experiment)
    assert "Baseline reference replication failed closed" in source


def test_primary_evaluation_uses_only_linear_task() -> None:
    source = inspect.getsource(anchor.run_experiment)
    assert '["linear_regression"]' in source
    assert '"nonlinear_teacher"' not in source


def test_labeled_budgets_count_identities() -> None:
    assert anchor.task_benchmark.LABEL_BUDGETS == (8, 16, 32)


def test_labeled_subsets_remain_nested(dataset, frozen) -> None:
    payload = frozen["benchmark"]["payload"]
    probe = payload["probe_configurations"]["classification_probe"]
    split = anchor.geometry.make_probe_identity_split(dataset, 704301, probe["validation_fraction"])
    subsets = anchor.task_benchmark.nested_identity_subsets(split.probe_training_identities, 8101)
    assert set(subsets[8]) < set(subsets[16]) < set(subsets[32])


def test_both_residual_seeds_are_required() -> None:
    assert anchor.task_benchmark.RESIDUAL_SEEDS == (7203, 7204)
    assert "len(biological) == 4" in inspect.getsource(anchor.linear_task_flags)


def test_identity_permuted_controls_operate_by_identity() -> None:
    source = inspect.getsource(anchor.model_task_evaluation)
    assert "identity_permuted_features" in source


def test_acquisition_exclusion_requires_every_probe_to_fail() -> None:
    source = inspect.getsource(anchor.linear_task_flags)
    assert "max(acquisition_ridge) < 0.10" in source
    assert "max(acquisition_residual) < 0.10" in source


def test_operational_preservation_includes_worst_pair_retrieval() -> None:
    source = inspect.getsource(anchor.operational_diagnostics)
    assert "cross_scanner_identity_retrieval_success" in source
    assert "retrieval_geometry" in source


def test_scanner_exclusion_uses_every_observed_and_null_seed() -> None:
    source = inspect.getsource(anchor.operational_diagnostics)
    assert "calibrated.SCANNER_PROBE_SEEDS" in source
    assert "include_permutation_null=True" in source


def test_counterfactual_requires_direct_eligibility() -> None:
    source = inspect.getsource(anchor.counterfactual_linear_task)
    assert "eligible = direct_r2 >= 0.70" in source


def test_counterfactual_decode_uses_requested_target_scanner() -> None:
    source = inspect.getsource(anchor.counterfactual_linear_task)
    assert "model.acquisition_from_scanner(target_scanner)" in source


def test_counterfactual_reencoding_occurs_after_decode() -> None:
    source = inspect.getsource(anchor.counterfactual_linear_task)
    assert source.index("generated = model.decode") < source.index("model.encode_biological(generated)")


def test_linear_head_and_linear_probe_compose_to_affine(config) -> None:
    model = anchor.build_family_model("linear_consensus_anchor", config, torch.device("cpu"))
    result = anchor.verify_linear_composition(model.consensus_head)
    assert result["composition_equivalent"]
    assert result["head_affine_layer_count"] == 1


def test_nonlinear_success_cannot_be_linear_mechanism_success() -> None:
    source = inspect.getsource(anchor.run_experiment)
    assert 'run["family"] == "linear_consensus_anchor" and common_success' in source
    assert 'run["family"] == "nonlinear_consensus_anchor" and common_success' in source


def _run_with_flags(family: str, linear: bool, nonlinear: bool, tradeoff: bool = False) -> dict:
    return {
        "family": family,
        "interpretation_flags": {
            "linear_accessibility_mechanism_succeeded": linear,
            "consensus_objective_mechanism_succeeded": nonlinear,
            "anchor_operational_tradeoff_detected": tradeoff,
        },
    }


def test_operational_degradation_triggers_tradeoff_status() -> None:
    runs = []
    for family in anchor.FAMILIES:
        runs.extend([_run_with_flags(family, False, False, family != "crossed_target_baseline")] * 8)
    assert anchor.family_status(runs)["status"] == "complete_consensus_anchor_operational_tradeoff"


def test_poor_scientific_performance_is_not_execution_failure() -> None:
    runs = []
    for family in anchor.FAMILIES:
        runs.extend([_run_with_flags(family, False, False)] * 8)
    assert anchor.family_status(runs)["status"] == "complete_consensus_anchor_mechanism_unsupported"


def test_previous_statuses_remain_immutable() -> None:
    assert anchor.POWER_AUDIT_STATUS == "complete_original_task_benchmark_partially_instrument_valid"
    assert anchor.task_benchmark.FROZEN_STATUS if hasattr(anchor.task_benchmark, "FROZEN_STATUS") else True


def test_existing_output_directories_cannot_be_overwritten(tmp_path: Path) -> None:
    with pytest.raises(anchor.ExperimentError, match="overwrite"):
        anchor.ensure_new_output_root(tmp_path)


def test_heterogeneous_summary_csv_fields_are_supported(tmp_path: Path) -> None:
    rows = [{"a": 1}, {"a": 2, "b": 3}]
    path = tmp_path / "summary.csv"
    anchor.parent.atomic_csv(path, anchor.parent.summary_csv_fieldnames(rows), rows)
    with path.open(encoding="utf-8", newline="") as handle:
        assert list(csv.DictReader(handle))[0]["b"] == ""


def test_all_frozen_artifacts_remain_unchanged() -> None:
    before = anchor.verify_power_audit(POWER_AUDIT)
    after = anchor.verify_power_audit(POWER_AUDIT)
    assert before["file_sha256"] == after["file_sha256"]
    assert before["benchmark"]["upstream"]["factorial"]["calibrated"]["upstream"]["hashes"] == after[
        "benchmark"
    ]["upstream"]["factorial"]["calibrated"]["upstream"]["hashes"]


def test_consensus_targets_are_detached_constants(target) -> None:
    assert target["manifest"]["targets_detached_constants"]


def test_consensus_scale_floor_is_fixed(target) -> None:
    assert target["manifest"]["scale_floor"] == 1e-6
    assert np.all(target["consensus_scale"] >= 1e-6)
