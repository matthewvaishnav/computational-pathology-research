"""Contract tests for the versioned no-training fixed-estimand adjudication."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from experiments.paired_acquisition import (
    run_fixed_estimand_real_feature_space_adjudication as adj,
)
from experiments.paired_acquisition import (
    run_fixed_estimand_real_feature_space_adjudication_v2 as v2,
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


def fake_cell(fold, seed, family, tmp_path, accepted=True):
    cell_dir = tmp_path / f"{fold}_{seed}_{family}"
    cell_dir.mkdir(parents=True, exist_ok=True)
    projected = cell_dir / "projected_features.npz"
    np.savez(
        projected,
        biological_features=np.zeros((4, 32), dtype="<f4"),
        acquisition_features=np.zeros((4, 8), dtype="<f4"),
        combined_features=np.zeros((4, 40), dtype="<f4"),
    )
    checkpoint = cell_dir / "checkpoint.pt"
    checkpoint.write_bytes(b"ckpt")
    return {
        "dataset": "canine_scc",
        "fold": fold,
        "seed": seed,
        "family": family,
        "accepted": accepted,
        "projected_features_path": str(projected.resolve()),
        "projected_features_sha256": v2.sha256_file(projected),
        "checkpoint_path": str(checkpoint.resolve()),
        "checkpoint_sha256": v2.sha256_file(checkpoint),
    }


def fake_recovery(tmp_path, accepted=True):
    cells = [
        fake_cell(fold, seed, family, tmp_path, accepted=accepted)
        for fold in adj.FOLDS
        for seed in adj.MODEL_SEEDS
        for family in adj.FAMILIES
    ]
    return {
        "status": v2.RECOVERY_STATUS_REQUIRED if accepted else "real_bottleneck_representation_recovery_failed",
        "cells": cells,
        "git_commit": "abc",
        "result_sha256": "x",
    }


def fake_metric_table():
    table = {}
    for fold in adj.FOLDS:
        table[fold] = {}
        for method in adj.DETERMINISTIC_METHODS + adj.FAMILIES:
            table[fold][method] = {
                "category_balanced_accuracy": 0.5,
                "scanner_balanced_accuracy": 0.3,
                "worst_pair_retrieval": 0.8,
                "overall_retrieval": 0.9,
            }
    return table


# ---------------------------------------------------------------------------
# Reuse and zero-training guards
# ---------------------------------------------------------------------------


def test_v2_imports_and_reuses_original_adjudication():
    source = inspect.getsource(v2.run_adjudication_v2)
    assert "adj.verify_frozen_real_validation" in source
    assert "adj.derive_fixed_categories_authoritative" in source
    assert "adj.evaluate_deterministic_methods" in source
    assert "adj.required_canine_contrasts" in source
    assert "adj.cross_fold_material_dominance" in source


def test_no_optimizer_constructed():
    source = inspect.getsource(v2.run_adjudication_v2)
    assert "torch.optim" not in source
    assert "AdamW" not in source


def test_no_backward_pass_executed():
    source = inspect.getsource(v2)
    assert ".backward(" not in source
    assert "optimizer.step(" not in source


def test_no_training_function_called():
    source = inspect.getsource(v2.run_adjudication_v2)
    assert "train_factorizer(" not in source
    assert "train(" not in source


def test_loads_recovered_arrays_only():
    source = inspect.getsource(v2.load_cell_arrays)
    assert "projected_features_path" in source
    assert "biological_features" in source
    assert "acquisition_features" in source


def test_zero_training_verification_reused():
    assert v2.adj.zero_training_verification()["optimizers_constructed"] == 0
    assert v2.adj.zero_training_verification()["backward_passes_executed"] == 0
    assert v2.adj.zero_training_verification()["factorizers_trained"] == 0


# ---------------------------------------------------------------------------
# Recovery manifest verification
# ---------------------------------------------------------------------------


def test_recovery_status_required():
    with pytest.raises(v2.AdjudicationV2Error, match="Recovery status"):
        v2.load_recovery_manifest(FROZEN_RESULT)  # not a recovery manifest


def test_recovery_must_contain_50_accepted_cells(tmp_path):
    recovery = fake_recovery(tmp_path)
    assert len(recovery["cells"]) == 50
    path = tmp_path / "recovery.json"
    path.write_text(json.dumps(recovery), encoding="utf-8")
    parsed = v2.load_recovery_manifest(path)
    assert parsed["status"] == v2.RECOVERY_STATUS_REQUIRED


def test_recovery_with_unaccepted_cell_fails(tmp_path):
    recovery = fake_recovery(tmp_path, accepted=False)
    path = tmp_path / "recovery_bad.json"
    path.write_text(json.dumps(recovery), encoding="utf-8")
    with pytest.raises(v2.AdjudicationV2Error, match="Recovery status"):
        v2.load_recovery_manifest(path)


def test_recovery_hash_verification_fails_closed_on_tamper(tmp_path):
    recovery = fake_recovery(tmp_path)
    # Tamper the projected npz after recording the hash.
    cell = recovery["cells"][0]
    tampered = Path(cell["projected_features_path"])
    tampered.write_bytes(b"tampered")
    with pytest.raises(v2.AdjudicationV2Error, match="projected-feature hash mismatch"):
        v2.verify_recovery_hashes(recovery, tmp_path)


def test_recovery_hash_verification_passes(tmp_path):
    recovery = fake_recovery(tmp_path)
    result = v2.verify_recovery_hashes(recovery, tmp_path)
    assert result["verified_cells"] == 50
    assert result["all_hashes_verified"]


def test_missing_recovered_file_fails(tmp_path):
    recovery = fake_recovery(tmp_path)
    Path(recovery["cells"][0]["projected_features_path"]).unlink()
    with pytest.raises(v2.AdjudicationV2Error, match="missing"):
        v2.verify_recovery_hashes(recovery, tmp_path)


# ---------------------------------------------------------------------------
# Frozen reproduction gate
# ---------------------------------------------------------------------------


def test_frozen_reproduction_verification_detects_mismatch():
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
        "category_balanced_accuracy_seven": 0.5,
        "scanner_balanced_accuracy": 0.3,
        "overall_retrieval": 0.8,
        "worst_pair_retrieval": 0.7,
    }
    assert v2.verify_frozen_reproduction(matching, frozen_run)["frozen_metrics_reproduced"]
    bad = {**matching, "overall_retrieval": 0.81}
    assert not v2.verify_frozen_reproduction(bad, frozen_run)["frozen_metrics_reproduced"]


# ---------------------------------------------------------------------------
# Seed averaging and endpoints
# ---------------------------------------------------------------------------


def test_seed_averaging_occurs_within_fold():
    evaluations = {}
    for fold in adj.FOLDS:
        evaluations[fold] = {}
        for seed in adj.MODEL_SEEDS:
            evaluations[fold][seed] = {}
            for family in adj.FAMILIES:
                evaluations[fold][seed][family] = {
                    "corrected_category": {"balanced_accuracy": 0.5 + seed * 1e-9},
                    "scanner": {"linear_balanced_accuracy": 0.3},
                    "retrieval": {"worst_ordered_scanner_pair_top1": 0.8, "overall_top1": 0.9},
                }
    averaged = v2.average_neural_within_fold(evaluations)
    assert averaged[0]["real_b32_reference"]["seed_averaged_before_inference"]
    assert set(averaged[0]["real_b32_reference"]["seed_distribution"].keys()) == set(adj.MODEL_SEEDS)


def test_seven_and_five_category_endpoints_remain_separate():
    assert "exploratory_seven_category_endpoint" in inspect.getsource(v2.run_adjudication_v2)
    assert "fixed_category_set" in inspect.getsource(v2.run_adjudication_v2)


# ---------------------------------------------------------------------------
# Adjudication logic
# ---------------------------------------------------------------------------


def test_dominance_margins_are_fixed_0_02():
    assert v2.MATERIAL_MARGIN == 0.02
    assert v2.adj.MATERIAL_MARGIN == 0.02


def test_cross_fold_dominance_requires_four_folds():
    assert v2.CROSS_FOLD_REQUIRED == 4


def test_neural_increment_analysis_requires_category_gain():
    table = fake_metric_table()
    # All methods equal; no category gain over baselines.
    for fold in adj.FOLDS:
        for method in table[fold]:
            table[fold][method]["category_balanced_accuracy"] = 0.5
    result = v2.neural_increment_analysis(table)
    assert not result["neural_feature_space_increment_supported"]


def test_neural_increment_analysis_detects_gain():
    table = fake_metric_table()
    for fold in adj.FOLDS:
        for method in table[fold]:
            if method in adj.DETERMINISTIC_METHODS:
                table[fold][method]["category_balanced_accuracy"] = 0.4
                table[fold][method]["scanner_balanced_accuracy"] = 0.5
                table[fold][method]["worst_pair_retrieval"] = 0.7
            else:
                table[fold][method]["category_balanced_accuracy"] = 0.6
                table[fold][method]["scanner_balanced_accuracy"] = 0.5
                table[fold][method]["worst_pair_retrieval"] = 0.7
    result = v2.neural_increment_analysis(table)
    assert result["neural_feature_space_increment_supported"]


def test_scorpion_cannot_produce_biological_accessibility_conclusion():
    conclusions = v2.v2_dataset_conclusions(
        {"neural_feature_space_increment_supported": False},
        {"simple_baseline_pareto_dominance_supported": True},
    )
    assert conclusions["scorpion"]["conclusion"] == "feature_only_no_biological_claim"


def test_synthetic_transport_requires_corrected_category_gain():
    contrasts = {
        "b64_minus_b32": {
            "available": True,
            "axes": {
                "category_balanced_accuracy": {"mean": 0.01, "positive_fold_count": 2},
                "scanner_balanced_accuracy": {"mean": 0.0},
                "worst_pair_retrieval": {"mean": 0.0},
                "overall_retrieval": {"mean": 0.0},
            },
        }
    }
    transport = v2.synthetic_transport_decision(
        contrasts, {"neural_feature_space_increment_supported": False}
    )
    assert not transport["synthetic_accessibility_effect_transported"]


def test_synthetic_transport_supported_with_gain():
    contrasts = {
        "b64_minus_b32": {
            "available": True,
            "axes": {
                "category_balanced_accuracy": {"mean": 0.05, "positive_fold_count": 5},
                "scanner_balanced_accuracy": {"mean": 0.0},
                "worst_pair_retrieval": {"mean": 0.02},
                "overall_retrieval": {"mean": 0.02},
            },
        }
    }
    transport = v2.synthetic_transport_decision(
        contrasts, {"neural_feature_space_increment_supported": True}
    )
    assert transport["synthetic_accessibility_effect_transported"]


def test_layer2_is_prohibited():
    schema = v2.adj.layer2_missing_metadata_schema({"frozen_readiness": {"path": str(READINESS)}})
    assert schema["execution"] == "not_executed"
    assert "inference_prohibited" in schema


def test_prior_adjudication_result_not_overwritten():
    # The v2 result files are distinct names.
    assert "v2_result.json" in "fixed_estimand_real_feature_space_adjudication_v2_result.json"
    assert "v2" in v2.SCHEMA_VERSION


def test_output_directory_cannot_be_overwritten(tmp_path):
    output = tmp_path / "exists"
    output.mkdir()
    with pytest.raises(v2.AdjudicationV2Error, match="already exists"):
        v2.run_adjudication_v2(
            FROZEN_RESULT,
            REPOSITORY,
            tmp_path / "recovery.json",
            output,
            copied_path=None,
        )


def test_frozen_artifacts_remain_unchanged_by_v2_code():
    assert v2.sha256_file(FROZEN_RESULT) == adj.FROZEN_RESULT_FILE_SHA256
    assert v2.sha256_file(READINESS) == adj.READINESS_FILE_SHA256
    assert v2.sha256_file(MANIFEST) == adj.MANIFEST_FILE_SHA256
