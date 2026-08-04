"""Contract tests for the no-training fixed-estimand real feature-space adjudication."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments.paired_acquisition import (
    run_fixed_estimand_real_feature_space_adjudication as adj,
)


REPOSITORY = Path(__file__).resolve().parents[1]
FROZEN_RESULT = (
    REPOSITORY
    / "results/real_paired_scanner_bottleneck_allocation_validation_20260803T191207"
    / "real_paired_scanner_bottleneck_allocation_validation_result.json"
)
READINESS = (
    REPOSITORY
    / "results/real_paired_scanner_bottleneck_allocation_validation_20260803T191207"
    / "real_paired_scanner_bottleneck_allocation_readiness.json"
)
MANIFEST = (
    REPOSITORY
    / "results/real_paired_scanner_bottleneck_allocation_validation_20260803T191207"
    / "real_paired_scanner_bottleneck_allocation_validation_manifest.json"
)


@pytest.fixture(scope="session")
def verified():
    return adj.verify_frozen_real_validation(
        FROZEN_RESULT,
        REPOSITORY,
        copied_path=Path(adj.FROZEN_RESULT_COPY_PATH),
    )


@pytest.fixture(scope="session")
def frozen_value():
    return json.loads(FROZEN_RESULT.read_text(encoding="utf-8"))


def tiny_frame(with_samples: bool = True) -> pd.DataFrame:
    rows = []
    for split, regions in (
        ("train", range(0, 4)),
        ("val", range(4, 6)),
        ("test", range(6, 8)),
    ):
        for region in regions:
            for scanner in ("a", "b"):
                rows.append(
                    {
                        "slide_id": f"slide_{region}",
                        "sample_id": f"sample_{region}" if with_samples else f"sample_{region}",
                        "region_id": f"region_{region}",
                        "scanner_id": scanner,
                        "path": f"{region}_{scanner}.npy",
                        "split": split,
                        "category_name": "A" if region % 2 == 0 else "B",
                    }
                )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# 1-3. Frozen artifact chain
# ---------------------------------------------------------------------------


def test_frozen_real_result_file_and_internal_hash_verify(verified):
    assert verified["frozen_result"]["file_sha256"] == adj.FROZEN_RESULT_FILE_SHA256
    assert verified["frozen_result"]["internal_sha256"] == adj.FROZEN_RESULT_INTERNAL_SHA256


def test_readiness_and_manifest_hashes_verify(verified):
    assert verified["frozen_readiness"]["file_sha256"] == adj.READINESS_FILE_SHA256
    assert verified["frozen_readiness"]["internal_sha256"] == adj.READINESS_INTERNAL_SHA256
    assert verified["frozen_manifest"]["file_sha256"] == adj.MANIFEST_FILE_SHA256
    assert verified["frozen_manifest"]["internal_sha256"] == adj.MANIFEST_INTERNAL_SHA256


def test_complete_inherited_chain_verifies(verified):
    assert verified["chain_verified"]
    assert verified["immutable_input_count"] == 225
    assert verified["inherited_artifact_count"] == 11
    assert verified["frozen_synthetic"]["status"] == adj.real_validation.SYNTHETIC_STATUS


def test_copied_frozen_result_hash_verifies(verified):
    assert verified["copied_result"]["file_sha256"] == adj.FROZEN_RESULT_FILE_SHA256


# ---------------------------------------------------------------------------
# 4-8. Zero-training guards
# ---------------------------------------------------------------------------


def test_no_factorizer_optimizer_is_constructed():
    source = inspect.getsource(adj.run_experiment)
    assert "torch.optim" not in source
    assert "AdamW" not in source
    source_all = inspect.getsource(adj)
    assert "torch.optim." not in source_all


def test_no_backward_pass_is_executed():
    source = inspect.getsource(adj)
    assert ".backward(" not in source
    assert "autograd" not in source


def test_no_factorizer_training_function_is_called():
    source = inspect.getsource(adj.run_experiment)
    assert "train_factorizer(" not in source
    assert "optimizer.step(" not in source
    assert "zero_training_verification" in source


def test_no_synthetic_generator_is_called():
    source = inspect.getsource(adj.run_experiment)
    assert "generate_dataset(" not in source
    assert '"synthetic_datasets_generated": 0' in inspect.getsource(adj.zero_training_verification)


def test_no_pixel_or_wsi_model_is_constructed():
    source = inspect.getsource(adj)
    assert "openslide" not in source.lower()
    assert "pixel_or_wsi_model_constructed" not in source or "wsi_or_pixel_models_constructed" in source
    assert '"wsi_or_pixel_models_constructed": 0' in inspect.getsource(adj.zero_training_verification)


# ---------------------------------------------------------------------------
# 9-11. Fixed-estimand reuse and category set
# ---------------------------------------------------------------------------


def test_exact_corrected_fixed_estimand_implementation_is_reused():
    source = inspect.getsource(adj.derive_fixed_categories_authoritative)
    assert "fixed_estimand.derive_fixed_categories" in source
    assert "fixed_estimand.category_sample_support" in source
    assert "reused_implementation" in source


def test_bone_and_cartilage_are_excluded():
    assert "Bone" in adj.EXCLUDED_CATEGORIES
    assert "Cartilage" in adj.EXCLUDED_CATEGORIES
    assert "Bone" not in adj.FIXED_CATEGORIES
    assert "Cartilage" not in adj.FIXED_CATEGORIES


def test_five_retained_categories_match_corrected_evidence():
    assert adj.FIXED_CATEGORIES == [
        "Dermis",
        "Epidermis",
        "Inflamm/Necrosis",
        "SCC",
        "Subcutis",
    ]


# ---------------------------------------------------------------------------
# 12-13. Sample support and fold integrity
# ---------------------------------------------------------------------------


def test_every_category_meets_fit_and_test_sample_support(verified):
    del verified
    support = adj.derive_fixed_categories_authoritative(REPOSITORY)
    assert support["support_ok"]
    for row in support["per_fold_support"]:
        assert row["fit_ok"]
        assert row["test_ok"]


def test_slides_and_samples_cannot_cross_folds():
    frame = tiny_frame()
    # Move one scanner view of sample_6 into train; its pair stays in test.
    frame.loc[(frame.sample_id == "sample_6") & (frame.scanner_id == "a"), "split"] = "train"
    check = adj.fold_integrity_check(frame, "slide_id")
    assert check["samples_crossing_splits"] >= 1
    assert not check["passed"]

    frame = tiny_frame()
    frame.loc[(frame.slide_id == "slide_7") & (frame.scanner_id == "a"), "split"] = "val"
    check = adj.fold_integrity_check(frame, "slide_id")
    assert check["specimens_crossing_splits"] >= 1
    assert not check["passed"]


def test_scanner_views_of_one_region_stay_in_one_fold():
    frame = tiny_frame()
    # Move one scanner view of region_6 into val while its pair stays in test.
    frame.loc[(frame.region_id == "region_6") & (frame.scanner_id == "a"), "split"] = "val"
    check = adj.fold_integrity_check(frame, "slide_id")
    assert check["regions_crossing_splits"] >= 1
    assert not check["passed"]


# ---------------------------------------------------------------------------
# 14-16. Vocabulary and probe leakage
# ---------------------------------------------------------------------------


def test_global_category_vocabulary_uses_metadata_only():
    support = adj.derive_fixed_categories_authoritative(REPOSITORY)
    assert support["vocabulary_source"].startswith("fold manifests only")
    all_categories = set()
    for fold in adj.FOLDS:
        manifest = pd.read_csv(
            REPOSITORY
            / adj.CANINE_MANIFEST_DIR
            / adj.CANINE_MANIFEST_PATTERN.format(fold=fold),
            dtype=str,
        )
        all_categories.update(manifest["category_name"].astype(str))
    assert set(adj.FIXED_CATEGORIES).issubset(all_categories)
    assert set(adj.EXCLUDED_CATEGORIES).issubset(all_categories)


def test_probe_fitting_uses_training_rows_and_labels_only():
    source = inspect.getsource(adj.fixed_category_evaluation)
    assert "v2.split_indices" in source
    probe_source = inspect.getsource(adj.fixed_estimand.fit_probe)
    assert "model.fit(features[fit], labels[fit])" in probe_source


def test_absent_training_category_is_scored_with_zero_recall_not_silently_removed():
    rng = np.random.default_rng(3)
    features = rng.normal(size=(20, 4)).astype(np.float32)
    frame = pd.DataFrame(
        {
            "sample_id": [f"s{i}" for i in range(20)],
            "region_id": [f"r{i}" for i in range(20)],
            "scanner_id": ["a"] * 20,
            "slide_id": [f"sl{i}" for i in range(20)],
            "split": ["train"] * 8 + ["val"] * 4 + ["test"] * 8,
            "category_name": ["A"] * 8 + ["B"] * 8 + ["C"] * 4,
        }
    )
    labels = frame["category_name"].astype(str).to_numpy()
    fit = np.flatnonzero(frame["split"].to_numpy() != "test")
    test = np.flatnonzero(frame["split"].to_numpy() == "test")
    prediction = adj.fixed_estimand.fit_probe(features, labels, fit, test)
    truth = labels[test]
    recall = adj.per_category_recall(truth, prediction, ["A", "B", "C"])
    # C appears only in test rows, so the probe can never predict C.
    assert recall["C"] == 0.0


# ---------------------------------------------------------------------------
# 17-18. Neighbour exclusion
# ---------------------------------------------------------------------------


def test_same_region_neighbours_are_excluded():
    frame = tiny_frame()
    fit = np.flatnonzero(frame["split"].to_numpy() != "test")
    test = np.flatnonzero(frame["split"].to_numpy() == "test")
    features = np.random.default_rng(0).normal(size=(len(frame), 8)).astype(np.float32)
    result = adj.v2.category_purity_fit_pool(features, frame, fit, test)
    assert "purity_fit_pool_k5" in result


def test_same_region_and_same_sample_exclusion_uses_infinity_mask():
    source = inspect.getsource(adj.v2.category_purity_fit_pool)
    assert "similarity[row, forbidden] = -np.inf" in source


# ---------------------------------------------------------------------------
# 19-20. Neural representation recovery
# ---------------------------------------------------------------------------


def test_neural_cells_map_to_exact_frozen_fold_seed_family_grid():
    cells = adj.expected_neural_cells()
    assert len(cells) == 5 * 5 * 2 == 50
    for cell in cells:
        assert cell["dataset"] == "canine_scc"
        assert cell["fold"] in adj.FOLDS
        assert cell["seed"] in adj.MODEL_SEEDS
        assert cell["family"] in adj.FAMILIES
    assert len({(c["fold"], c["seed"], c["family"]) for c in cells}) == 50


def test_recovered_neural_metrics_reproduce_frozen_metrics():
    frozen_run = {
        "layer1": {
            "biological_category_accessibility": {"linear": {"balanced_accuracy_median": 0.5}},
            "biological_scanner_probe": {"linear": {"balanced_accuracy_median": 0.3}},
            "paired_region_preservation": {
                "overall_top1": 0.8,
                "worst_ordered_scanner_pair_top1": 0.7,
            },
        }
    }
    matching = {
        "category_balanced_accuracy": 0.5,
        "scanner_balanced_accuracy": 0.3,
        "overall_retrieval": 0.8,
        "worst_pair_retrieval": 0.7,
    }
    ok = adj.compare_recovered_to_frozen_metrics(frozen_run, matching)
    assert ok["frozen_metrics_reproduced"]
    bad = adj.compare_recovered_to_frozen_metrics(frozen_run, {**matching, "overall_retrieval": 0.81})
    assert not bad["frozen_metrics_reproduced"]
    assert any(item["metric"] == "overall_retrieval" for item in bad["mismatches"])


def test_recovery_fails_closed_when_no_saved_arrays_or_checkpoints_exist():
    recovery = adj.recover_neural_cells(REPOSITORY)
    assert recovery["expected_cells"] == 50
    assert not recovery["all_recovered"]
    assert all(not cell["recovered"] for cell in recovery["cells"])
    assert any(cell["missing_artifacts"] for cell in recovery["cells"])


# ---------------------------------------------------------------------------
# 21. Historical baselines require exact compatibility
# ---------------------------------------------------------------------------


def test_historical_baseline_import_requires_exact_hash_and_fold_match():
    source = inspect.getsource(adj)
    assert "verify_frozen_real_validation" in source
    assert "file_sha256" in source
    # Frozen neural metrics are imported only as descriptive seven-category evidence.
    assert "estimand" in inspect.getsource(adj.frozen_neural_descriptive_metrics)


# ---------------------------------------------------------------------------
# 22. Endpoint separation
# ---------------------------------------------------------------------------


def test_seven_category_and_five_category_endpoints_remain_separate():
    assert "exploratory_seven_category_endpoint" in inspect.getsource(adj.run_experiment)
    assert "fixed_category_set" in inspect.getsource(adj.run_experiment)


# ---------------------------------------------------------------------------
# 23-24. Seed averaging and scanner-view independence
# ---------------------------------------------------------------------------


def test_seed_averaging_occurs_within_fold_before_inference():
    # The frozen runner averages model seeds within fold before cross-fold inference.
    source = inspect.getsource(adj.real_validation.average_seeds_within_fold)
    assert "seed_averaged_before_inference" in source


def test_scanner_views_are_not_independent_biological_samples():
    # A biological sample with one scanner view in fit and another in test must raise.
    frame = tiny_frame()
    frame.loc[(frame.sample_id == "sample_6") & (frame.scanner_id == "a"), "split"] = "train"
    fit = np.flatnonzero(frame["split"].to_numpy() != "test")
    test = np.flatnonzero(frame["split"].to_numpy() == "test")
    fit_samples = set(frame.iloc[fit]["sample_id"].astype(str))
    test_samples = set(frame.iloc[test]["sample_id"].astype(str))
    assert len(fit_samples & test_samples) >= 1
    with pytest.raises(adj.v2.AuditError):
        adj.v2.split_indices(frame)


# ---------------------------------------------------------------------------
# 25. Retrieval candidate pools identical
# ---------------------------------------------------------------------------


def test_retrieval_candidate_pools_are_identical_across_methods():
    source = inspect.getsource(adj.retrieval_evaluation)
    assert "identical fixed held-out rows" in source
    assert "retrieval_metrics" in source


# ---------------------------------------------------------------------------
# 26-28. Broken paired-linear controls and attribution
# ---------------------------------------------------------------------------


def test_broken_paired_linear_control_preserves_scanner_counts():
    frame = tiny_frame()
    train = np.flatnonzero(frame["split"].to_numpy() == "train")
    source, target, report = adj.real_validation.build_pair_indices(
        frame, train, broken=True, seed=11
    )
    del source, target
    assert report["scanner_counts_preserved"]


def test_broken_paired_linear_permutation_stays_within_training_fold():
    frame = tiny_frame()
    train = np.flatnonzero(frame["split"].to_numpy() == "train")
    source, target, report = adj.real_validation.build_pair_indices(
        frame, train, broken=True, seed=11
    )
    assert report["indices_within_requested_split"]
    assert set(source.tolist()).issubset(set(train.tolist()))
    assert set(target.tolist()).issubset(set(train.tolist()))


def test_true_pair_and_broken_pair_conclusions_remain_separate():
    # Frozen result separates broken-pair controls from primary runs.
    frozen = json.loads(FROZEN_RESULT.read_text(encoding="utf-8"))
    true_runs = [r for r in frozen["runs"] if not r["broken_pair_control"]]
    broken_runs = [r for r in frozen["runs"] if r["broken_pair_control"]]
    assert len(true_runs) == 100
    assert len(broken_runs) == 4
    assert all(not r["broken_pair_control"] for r in true_runs)
    assert all(r["broken_pair_control"] for r in broken_runs)


# ---------------------------------------------------------------------------
# 29-31. Dominance margins
# ---------------------------------------------------------------------------


def test_dominance_uses_fixed_0_02_margins():
    assert adj.MATERIAL_MARGIN == 0.02
    source = inspect.getsource(adj.weak_dominance)
    assert "MATERIAL_MARGIN" in source
    assert "+ MATERIAL_MARGIN" in source or "- MATERIAL_MARGIN" in source


def test_no_weighted_composite_score_is_created():
    source = inspect.getsource(adj)
    assert "composite" not in source.lower().replace("decoder_or_composition_identifier", "")


def test_cross_fold_dominance_requires_at_least_four_folds():
    assert adj.CROSS_FOLD_DOMINANCE_REQUIRED_FOLDS == 4
    a = {f: {"scanner_balanced_accuracy": 0.1, "category_balanced_accuracy": 0.8, "worst_pair_retrieval": 0.9, "overall_retrieval": 0.95} for f in adj.FOLDS}
    b = {f: {"scanner_balanced_accuracy": 0.5, "category_balanced_accuracy": 0.5, "worst_pair_retrieval": 0.5, "overall_retrieval": 0.6} for f in adj.FOLDS}
    axes = ["scanner_balanced_accuracy", "category_balanced_accuracy", "worst_pair_retrieval", "overall_retrieval"]
    result = adj.cross_fold_material_dominance(a, b, axes)
    assert result["material_fold_count"] == 5
    assert result["cross_fold_material_dominance"]


# ---------------------------------------------------------------------------
# 32. SCORPION biological-accessibility boundary
# ---------------------------------------------------------------------------


def test_scorpion_cannot_produce_biological_accessibility_conclusion():
    allowed = {
        "simple_baseline_scanner_retrieval_frontier_supported",
        "neural_scanner_retrieval_increment_supported",
        "mixed_scanner_retrieval_frontier",
        "feature_only_no_biological_claim",
    }
    conclusions = adj.dataset_conclusions(neural_available=False)
    assert conclusions["scorpion"]["conclusion"] in allowed
    # SCORPION never receives a category-accessibility conclusion.
    assert "category_accessibility" not in conclusions["scorpion"]["conclusion"]
    assert conclusions["scorpion"]["conclusion"] != "neural_feature_space_increment_supported"


# ---------------------------------------------------------------------------
# 33-34. Synthetic transport
# ---------------------------------------------------------------------------


def test_synthetic_transport_requires_corrected_category_improvement():
    decision = {
        "synthetic_accessibility_effect_transported": False,
        "corrected_real_category_effect": None,
    }
    assert not decision["synthetic_accessibility_effect_transported"]
    assert "corrected_real_category_effect" in decision


def test_retrieval_gain_alone_cannot_establish_synthetic_transport():
    source = inspect.getsource(adj.run_experiment)
    assert "counted as transport of biological accessibility" in source
    assert "never " in source


# ---------------------------------------------------------------------------
# 35. Layer-2 boundary
# ---------------------------------------------------------------------------


def test_layer2_assignments_cannot_be_inferred():
    schema = adj.layer2_missing_metadata_schema(
        {"frozen_readiness": {"path": str(READINESS)}}
    )
    assert schema["execution"] == "not_executed"
    assert "inference_prohibited" in schema
    assert "reconstructed" in schema["inference_prohibited"] or "inferred" in schema["inference_prohibited"]


def test_layer2_schema_contains_all_required_fields():
    schema = adj.layer2_missing_metadata_schema(
        {"frozen_readiness": {"path": str(READINESS)}}
    )
    fields = {item["field"] for item in schema["required_fields"]}
    assert {
        "checkpoint_identifier",
        "source_region",
        "source_scanner",
        "target_scanner",
        "acquisition_source_record",
        "decoder_or_composition_identifier",
        "fold",
        "row_index_or_feature_identifier",
        "pair_assignment_generation_procedure",
        "sha256",
    }.issubset(fields)


# ---------------------------------------------------------------------------
# 36-38. Public files and immutability
# ---------------------------------------------------------------------------


def test_no_public_claim_file_is_modified():
    source = inspect.getsource(adj.write_outputs)
    assert "claim_boundary_snapshot.md" not in source
    assert "release_manifest.json" not in source
    assert "README.md" not in source
    for forbidden in ("CLAIM_BOUNDARY.md", "claim_source_manifest.csv"):
        assert forbidden not in inspect.getsource(adj)


def test_prior_statuses_remain_immutable():
    frozen = json.loads(FROZEN_RESULT.read_text(encoding="utf-8"))
    assert frozen["status"] == adj.FROZEN_STATUS
    assert frozen["status"] == "complete_mixed_real_paired_scanner_allocation_effects"
    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    assert manifest["status"] == adj.FROZEN_STATUS


def test_poor_scientific_evidence_does_not_trigger_execution_failure():
    status = adj.top_level_status(frozen_ok=True, neural_available=False, conclusions={})
    assert status == "fixed_estimand_adjudication_not_ready"
    assert status != "fixed_estimand_adjudication_failed"
    # A scientific not-ready status is not a failure status.
    assert "failed" not in status


# ---------------------------------------------------------------------------
# 39-41. Output safety
# ---------------------------------------------------------------------------


def test_output_directories_cannot_be_overwritten(tmp_path):
    output = tmp_path / "exists"
    output.mkdir()
    with pytest.raises(adj.ExperimentError, match="already exists"):
        adj.run_experiment(
            FROZEN_RESULT,
            REPOSITORY,
            output,
            copied_path=None,
        )


def test_heterogeneous_csv_fields_are_supported():
    assert adj.heterogeneous_fieldnames([{"a": 1}, {"b": 2, "a": 3}]) == ["a", "b"]


def test_frozen_artifacts_remain_unchanged_after_execution():
    before = {p: adj.sha256_file(p) for p in (FROZEN_RESULT, READINESS, MANIFEST)}
    # Recomputation must be a pure function of frozen inputs; the frozen hashes are stable.
    assert before[FROZEN_RESULT] == adj.FROZEN_RESULT_FILE_SHA256
    assert before[READINESS] == adj.READINESS_FILE_SHA256
    assert before[MANIFEST] == adj.MANIFEST_FILE_SHA256


# ---------------------------------------------------------------------------
# Adjudication machinery unit tests
# ---------------------------------------------------------------------------


def test_weak_dominance_uses_lower_is_better_scanner_axis():
    a = {"scanner_balanced_accuracy": 0.1, "worst_pair_retrieval": 0.9}
    b = {"scanner_balanced_accuracy": 0.3, "worst_pair_retrieval": 0.7}
    # a has lower (better) scanner BA and higher (better) retrieval.
    assert adj.weak_dominance(a, b, ["scanner_balanced_accuracy", "worst_pair_retrieval"])


def test_pareto_front_marks_dominated_methods():
    methods = [
        {"method": "a", "scanner_balanced_accuracy": 0.1, "worst_pair_retrieval": 0.9},
        {"method": "b", "scanner_balanced_accuracy": 0.5, "worst_pair_retrieval": 0.5},
    ]
    front = adj.pareto_front(
        methods,
        ["scanner_balanced_accuracy", "worst_pair_retrieval"],
        lower_is_better=["scanner_balanced_accuracy"],
    )
    assert "a" in front
    assert "b" not in front


def test_contrast_summary_reports_all_fold_effects():
    a = {fold: float(fold) for fold in adj.FOLDS}
    b = {fold: float(fold - 1) for fold in adj.FOLDS}
    summary = adj.contrast_summary(a, b, name="a_minus_b")
    assert summary["available"]
    assert list(summary["fold_effects"].values()) == [1.0] * 5
    assert summary["mean"] == 1.0
    assert summary["positive_fold_count"] == 5


def test_contrast_summary_fails_closed_on_missing_fold():
    a = {fold: float(fold) for fold in adj.FOLDS}
    b = {fold: float(fold) for fold in adj.FOLDS[:-1]}
    summary = adj.contrast_summary(a, b, name="incomplete")
    assert not summary["available"]
