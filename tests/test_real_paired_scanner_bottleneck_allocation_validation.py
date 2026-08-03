from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from experiments.paired_acquisition import (
    run_real_paired_scanner_bottleneck_allocation_validation as audit,
)


REPOSITORY = Path(__file__).resolve().parents[1]
SYNTHETIC = (
    REPOSITORY
    / "results/biological_bottleneck_capacity_allocation_factorial_20260803T150254"
    / "biological_bottleneck_capacity_allocation_factorial_result.json"
)


@pytest.fixture(scope="session")
def verified_synthetic():
    return audit.verify_synthetic_factorial(SYNTHETIC)


@pytest.fixture(scope="session")
def readiness():
    return audit.audit_readiness(REPOSITORY)


def tiny_frame() -> pd.DataFrame:
    rows = []
    for split, regions in (("train", range(4)), ("val", range(4, 6)), ("test", range(6, 8))):
        for region in regions:
            for scanner in ("a", "b"):
                rows.append(
                    {
                        "slide_id": f"slide_{region}",
                        "region_id": f"region_{region}",
                        "scanner_id": scanner,
                        "path": f"{region}_{scanner}.npy",
                        "split": split,
                        "fold": "0",
                        "category_name": "x" if region % 2 == 0 else "y",
                    }
                )
    return pd.DataFrame(rows)


def fake_run(family: str, fold: int, seed: int, values: dict[str, float], broken=False):
    category_available = values.get("category") is not None
    return {
        "dataset": "d",
        "fold": fold,
        "seed": seed,
        "family": family,
        "broken_pair_control": broken,
        "layer1": {
            "biological_scanner_probe": {"linear": {"balanced_accuracy_median": values["scanner"]}},
            "acquisition_scanner_probe": {"linear": {"balanced_accuracy_median": values["acq_scanner"]}},
            "paired_region_preservation": {
                "worst_ordered_scanner_pair_top1": values["worst"],
                "overall_top1": values["overall"],
                "similarity_margin": values.get("margin", 0.5),
            },
            "biological_category_accessibility": {
                "available": category_available,
                **({"linear": {"balanced_accuracy_median": values["category"]}} if category_available else {}),
            },
            "acquisition_category_leakage": {
                "available": category_available,
                **({"linear": {"balanced_accuracy_median": values["acq_category"]}} if category_available else {}),
            },
        },
    }


def test_final_synthetic_file_hash_verifies(verified_synthetic):
    assert verified_synthetic["file_sha256"] == audit.SYNTHETIC_FILE_SHA256


def test_final_synthetic_internal_hash_verifies(verified_synthetic):
    assert verified_synthetic["internal_sha256"] == audit.SYNTHETIC_INTERNAL_SHA256


def test_final_synthetic_manifest_hash_verifies(verified_synthetic):
    assert verified_synthetic["manifest_internal_sha256"] == audit.SYNTHETIC_MANIFEST_INTERNAL_SHA256


def test_inherited_synthetic_chain_verifies(verified_synthetic):
    assert len(verified_synthetic["inherited_artifacts"]) == 11
    assert all(item["verified_file_sha256"] for item in verified_synthetic["inherited_artifacts"].values())


def test_prior_synthetic_status_is_immutable(verified_synthetic):
    assert verified_synthetic["status"] == "complete_capacity_gain_with_scanner_tradeoff"


def test_no_synthetic_dataset_generator_is_called():
    source = inspect.getsource(audit.run_experiment)
    assert "generate_dataset(" not in source
    assert '"training_uses_synthetic_generator": False' in source


def test_no_pixel_or_wsi_model_is_constructed():
    source = inspect.getsource(audit)
    assert "pixel_or_wsi_model_constructed\": False" in source
    assert "openslide" not in source.lower()


def test_readiness_occurs_before_model_initialization(monkeypatch, tmp_path):
    calls = []
    monkeypatch.setattr(audit, "verify_synthetic_factorial", lambda _: {"path": "x", "file_sha256": "h", "manifest_path": "m", "manifest_file_sha256": "mh", "inherited_artifacts": {}})
    monkeypatch.setattr(audit, "audit_readiness", lambda _: {"schema_version": "x", "datasets": {}, "ready_datasets": [], "common_feature_backbone": None, "claim_scope": {}, "readiness_sha256": "r"})
    monkeypatch.setattr(audit, "build_factorizer", lambda *args, **kwargs: calls.append("model"))
    monkeypatch.setattr(audit, "verify_inputs_unchanged", lambda _: None)
    result = audit.run_experiment(tmp_path / "synthetic.json", tmp_path, tmp_path / "out", torch.device("cpu"))
    assert result["factorizer_fit_count"] == 0
    assert calls == []


def test_absent_feature_is_not_replaced_by_guessed_path(tmp_path):
    spec = dict(audit.DATASET_SPECS["scorpion"])
    result = audit.audit_dataset(tmp_path, "missing", spec)
    assert result["readiness_state"] == "feature_artifact_missing"


def test_feature_rows_align_with_metadata(readiness):
    assert all(all(item["row_alignment_by_fold"]) for item in readiness["datasets"].values())


def test_paired_regions_cannot_cross_folds():
    frame = tiny_frame()
    frame.loc[frame.region_id == "region_0", "split"] = ["train", "test"]
    assert not audit.fold_integrity(frame, "slide_id")["passed"]


def test_specimens_cannot_cross_folds():
    frame = tiny_frame()
    frame.loc[frame.region_id == "region_1", "slide_id"] = "slide_0"
    frame.loc[frame.region_id == "region_1", "split"] = "test"
    assert audit.fold_integrity(frame, "slide_id")["specimens_crossing_splits"] == 1


def test_category_claims_require_labels():
    frame = tiny_frame()
    train, _, test = audit.split_indices(frame)
    result = audit.category_probe_battery(np.random.default_rng(1).normal(size=(len(frame), 4)), frame, None, train, test)
    assert not result["available"]


def test_held_out_category_absent_from_training_is_scored_not_fitted():
    rng = np.random.default_rng(14)
    features = rng.normal(size=(12, 4))
    labels = np.asarray(["a"] * 5 + ["b"] * 5 + ["c"] * 2)
    result = audit.probe(
        features,
        labels,
        np.arange(10),
        np.arange(10, 12),
        nonlinear=False,
        seeds=(8401,),
    )
    assert result["classes_absent_from_training"] == ["c"]
    assert result["runs"][0]["per_class_recall"]["c"] == 0.0


def test_decoder_evaluation_requires_verified_swap_artifacts(readiness):
    assert all(not item["layer2_ready"] for item in readiness["datasets"].values())
    assert all(item["layer2_unavailable_reasons"] for item in readiness["datasets"].values())


def test_exactly_two_architecture_families_exist():
    assert audit.FAMILIES == ("real_b32_reference", "real_b64_parameter_matched")


def test_no_consensus_loss_or_auxiliary_head_exists():
    source = inspect.getsource(audit._objective)
    assert "consensus" not in source
    assert "auxiliary" not in source


def test_labels_absent_from_factorizer_training():
    source = inspect.getsource(audit.train_factorizer)
    assert "category_column" not in source
    assert "category_name" not in source


@pytest.mark.parametrize(
    "d,b,h,a,s,expected",
    [(768, 32, 128, 8, 5, 225224), (768, 64, 124, 8, 5, 225684)],
)
def test_dataset_parameter_formula(d, b, h, a, s, expected):
    assert audit.parameter_formula(d, b, h, a, s) == expected


@pytest.mark.parametrize("b,h", [(32, 128), (64, 124)])
def test_actual_pytorch_count_matches_formula(b, h):
    model = audit.build_factorizer(768, 5, b, h, torch.device("cpu"))
    assert sum(parameter.numel() for parameter in model.parameters()) == audit.parameter_formula(768, b, h, 8, 5)


def test_parameter_match_is_below_half_percent():
    assert audit.select_matched_hidden_width(768, 5)["relative_difference"] < 0.005


def test_hidden_width_is_frozen_before_training():
    result = audit.select_matched_hidden_width(768, 5)
    assert result["selected_before_training"]
    assert result["real_b64_parameter_matched"]["hidden_width"] == 124


def test_frozen_folds_are_reused(readiness):
    assert all(len(item["manifest_files"]) == 5 for item in readiness["datasets"].values())


def test_exactly_five_model_seeds_scheduled():
    assert audit.MODEL_SEEDS == (2201, 2202, 2203, 2204, 2205)


def test_seed_averaging_occurs_inside_fold_first():
    runs = []
    base = {"category": 0.6, "scanner": 0.2, "worst": 0.8, "overall": 0.9, "acq_category": 0.2, "acq_scanner": 1.0}
    for fold in audit.FOLDS:
        for seed in audit.MODEL_SEEDS:
            for family in audit.FAMILIES:
                runs.append(fake_run(family, fold, seed, base))
    rows = audit.average_seeds_within_fold(runs, "d")
    assert len(rows) == 5
    assert all(row["seed_averaged_before_inference"] for row in rows)


def test_scanner_views_not_used_as_independent_inference_units():
    interval = audit.deterministic_fold_then_specimen_bootstrap([1, 2, 3, 4, 5], 7)
    assert "folds resampled" in interval["clustering"]


def test_feature_scaler_uses_training_fold_only():
    assert '"feature_scaler_fit_split": "train"' in inspect.getsource(audit.train_factorizer)


def test_checkpoint_selection_uses_validation_only():
    source = inspect.getsource(audit.train_factorizer)
    assert '"test_used_for_checkpoint_selection": False' in source
    assert "best_validation_loss" in source


def test_broken_pair_control_preserves_scanner_counts():
    frame = tiny_frame()
    train, _, _ = audit.split_indices(frame)
    _, _, report = audit.build_pair_indices(frame, train, broken=True, seed=11)
    assert report["scanner_counts_preserved"]


def test_broken_pair_control_stays_in_training_fold():
    frame = tiny_frame()
    train, _, _ = audit.split_indices(frame)
    _, _, report = audit.build_pair_indices(frame, train, broken=True, seed=11)
    assert report["indices_within_requested_split"]


def test_broken_pair_control_deranges_regions():
    frame = tiny_frame()
    train, _, _ = audit.split_indices(frame)
    _, _, report = audit.build_pair_indices(frame, train, broken=True, seed=11)
    assert report["same_region_assignment_count"] == 0


def test_linear_and_nonlinear_scanner_probes_use_paired_nulls():
    source = inspect.getsource(audit.scanner_probe_battery)
    assert "paired_permutation_labels" in source
    assert "nonlinear" in source


def test_retrieval_reports_worst_ordered_pair():
    frame = tiny_frame()
    features = np.eye(len(frame), dtype=float)
    # Make paired scanner views identical.
    for _, group in frame.groupby("region_id"):
        features[group.index[1]] = features[group.index[0]]
    _, _, test = audit.split_indices(frame)
    result = audit.retrieval_metrics(features, frame, test)
    assert result["worst_ordered_scanner_pair_top1"] == 1.0


def test_category_leakage_measured_in_acquisition_branch():
    assert "acquisition_category_leakage" in inspect.getsource(audit.evaluate_representation)


def test_pca_fits_training_representations_only():
    source = inspect.getsource(audit.spectral_diagnostics)
    assert "PCA().fit(biological[train])" in source


def test_test_fold_does_not_select_pca_components():
    assert '"test_used_to_select_components": False' in inspect.getsource(audit.spectral_diagnostics)


def test_historical_baselines_are_contextual():
    source = inspect.getsource(audit.simple_baseline)
    assert '"comparative": False' in source


def test_primary_contrasts_pair_fold_and_seed():
    source = inspect.getsource(audit.average_seeds_within_fold)
    assert 'run["fold"] == fold' in source
    assert "len(MODEL_SEEDS)" in source


def test_material_effect_margin_is_fixed():
    assert audit.MATERIAL_MARGIN == 0.02


def test_scanner_tradeoff_logic_detects_scanner_increase():
    contrast = {"metrics": {
        "category_balanced_accuracy": {"mean_difference": 0.03, "positive_fold_count": 5},
        "biological_scanner_balanced_accuracy": {"mean_difference": 0.03},
        "worst_pair_region_retrieval": {"mean_difference": 0.0},
        "overall_region_retrieval": {"mean_difference": 0.0},
        "acquisition_category_leakage": {"mean_difference": 0.0},
    }}
    result = audit.classify_dataset(
        {"feature_space_ready": True, "category_labels_available": True},
        contrast,
        {"paired_supervision_demonstrated": True},
    )
    assert result["real_scanner_tradeoff_detected"]


def test_one_ready_dataset_cannot_create_cross_dataset_status():
    assert audit.top_level_status({"only": {"conclusion": "b64_allocation_supported"}}) == "complete_single_dataset_b64_allocation_supported"


def test_poor_scientific_performance_is_not_execution_failure():
    status = audit.top_level_status({"a": {"conclusion": "b64_allocation_unsupported"}})
    assert status == "complete_real_paired_scanner_allocation_unsupported"
    assert status != "real_paired_scanner_validation_failed"


def test_output_directory_cannot_be_overwritten(tmp_path):
    output = tmp_path / "exists"
    output.mkdir()
    with pytest.raises(audit.ExperimentError, match="already exists"):
        audit.run_experiment(SYNTHETIC, REPOSITORY, output, torch.device("cpu"))


def test_heterogeneous_csv_fields_are_supported():
    assert audit.heterogeneous_fieldnames([{"a": 1}, {"b": 2, "a": 3}]) == ["a", "b"]


def test_immutable_input_verification_detects_changes(tmp_path):
    path = tmp_path / "input.bin"
    path.write_bytes(b"one")
    expected = audit.sha256_file(path)
    path.write_bytes(b"two")
    with pytest.raises(audit.ExperimentError, match="changed"):
        audit.verify_inputs_unchanged({str(path): expected})


def test_common_dinov2_backbone_is_verified(readiness):
    assert readiness["common_feature_backbone_verified"]
    assert readiness["common_feature_backbone"] == "dinov2_base"


def test_scorpion_is_feature_only_ready(readiness):
    item = readiness["datasets"]["scorpion"]
    assert item["readiness_state"] == "feature_only_ready"
    assert not item["category_labels_available"]


def test_canine_is_feature_only_ready_with_categories(readiness):
    item = readiness["datasets"]["canine_scc"]
    assert item["readiness_state"] == "feature_only_ready"
    assert item["category_count"] == 7


def test_readiness_initializes_zero_models(readiness):
    assert readiness["model_initializations_during_readiness"] == 0


def test_pixel_space_is_prohibited_for_each_dataset(readiness):
    assert all(item["pixel_space_prohibited"] for item in readiness["datasets"].values())


def test_all_feature_arrays_are_finite(readiness):
    assert all(item["all_numerical_arrays_finite"] for item in readiness["datasets"].values())


def test_readiness_hash_is_canonical(readiness):
    copy = dict(readiness)
    embedded = copy.pop("readiness_sha256")
    assert audit.canonical_hash(copy) == embedded


def test_training_grid_has_100_primary_fits_when_both_ready(readiness):
    expected = len(readiness["ready_datasets"]) * len(audit.FOLDS) * len(audit.MODEL_SEEDS) * len(audit.FAMILIES)
    assert expected == 100


def test_broken_control_grid_is_small_and_predeclared():
    assert audit.BROKEN_PAIR_FOLDS == (0,)
    assert audit.BROKEN_PAIR_SEEDS == (2201,)


def test_no_high_budget_or_routed_consensus_family():
    assert all("high" not in family and "routed" not in family for family in audit.FAMILIES)


def test_no_previous_result_is_written_by_runner():
    source = inspect.getsource(audit.write_outputs)
    assert "biological_bottleneck_capacity_allocation_factorial_result.json" not in source
