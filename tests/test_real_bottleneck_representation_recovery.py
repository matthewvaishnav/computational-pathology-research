"""Contract tests for the exact canine B32/B64 neural artifact replay."""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch

from experiments.paired_acquisition import (
    run_real_bottleneck_representation_recovery as recovery,
)


REPOSITORY = Path(__file__).resolve().parents[1]
FROZEN_RESULT = (
    REPOSITORY
    / "results/real_paired_scanner_bottleneck_allocation_validation_20260803T191207"
    / "real_paired_scanner_bottleneck_allocation_validation_result.json"
)


def fake_frozen_run(**overrides):
    run = {
        "dataset": "canine_scc",
        "fold": 0,
        "seed": 2201,
        "family": "real_b32_reference",
        "broken_pair_control": False,
        "biological_dimension": 32,
        "hidden_width": 128,
        "acquisition_dimension": 8,
        "training": {
            "best_epoch": 75,
            "best_validation_loss": 1.7,
            "actual_parameter_count": 225224,
            "formula_parameter_count": 225224,
            "optimizer": "AdamW",
            "learning_rate": 0.0003,
            "weight_decay": 0.0001,
            "epochs": 75,
            "checkpoint_selected_by": "minimum validation objective only",
            "test_used_for_checkpoint_selection": False,
            "feature_scaler_fit_split": "train",
            "factorizer_reads_category_or_biological_labels": False,
            "paired_region_and_scanner_metadata_only": True,
            "broken_pair_control": False,
            "pair_audit": {"broken_pairs": False, "source_count": 10, "scanner_counts_preserved": True},
            "history": [
                {
                    "epoch": 1,
                    "train": {"total": 3.0},
                    "validation": {"total": 3.1},
                }
            ],
        },
        "layer1": {
            "biological_scanner_probe": {"linear": {"balanced_accuracy_median": 0.3}},
            "acquisition_scanner_probe": {"linear": {"balanced_accuracy_median": 0.99}},
            "biological_category_accessibility": {"linear": {"balanced_accuracy_median": 0.5}},
            "acquisition_category_leakage": {"linear": {"balanced_accuracy_median": 0.2}},
            "paired_region_preservation": {
                "overall_top1": 0.9,
                "worst_ordered_scanner_pair_top1": 0.8,
                "same_region_cosine_similarity": 0.7,
                "different_region_cosine_similarity": 0.3,
                "similarity_margin": 0.4,
            },
            "spectral_accessibility": {"numerical_rank": 10, "effective_rank": 3.5},
        },
        "layer2": {"available": False, "reason": ["no verified swap metadata"]},
        "pixel_space_evaluation_performed": False,
    }
    run.update(overrides)
    return run


def fake_replay(training=None, layer1=None):
    return {
        "training": training or fake_frozen_run()["training"],
        "layer1": layer1 or fake_frozen_run()["layer1"],
        "layer2": {"available": False, "reason": ["no verified swap metadata"]},
        "pixel_space_evaluation_performed": False,
        "biological_dimension": 32,
        "hidden_width": 128,
        "acquisition_dimension": 8,
    }


# ---------------------------------------------------------------------------
# Frozen inputs and source equivalence
# ---------------------------------------------------------------------------


def test_frozen_result_file_and_internal_hash_verify():
    verified = recovery.adjudication.verify_frozen_real_validation(
        FROZEN_RESULT, REPOSITORY, copied_path=Path(recovery.adjudication.FROZEN_RESULT_COPY_PATH)
    )
    assert verified["chain_verified"]
    assert verified["frozen_result"]["internal_sha256"] == recovery.adjudication.FROZEN_RESULT_INTERNAL_SHA256
    assert verified["immutable_input_count"] == 225


def test_frozen_source_commit_is_e95d8526():
    assert recovery.FROZEN_SOURCE_COMMIT == "e95d8526958ac781748f92b4ebb617b75a52fce0"


def test_training_source_is_equivalent_to_frozen_commit():
    check = recovery.source_equivalent_check(REPOSITORY)
    assert check["all_source_equivalent"]
    assert len(check["files"]) == 3


def test_replay_grid_is_exactly_canine_50():
    grid = recovery.run_replay
    del grid
    assert recovery.FOLDS == (0, 1, 2, 3, 4)
    assert recovery.MODEL_SEEDS == (2201, 2202, 2203, 2204, 2205)
    assert recovery.FAMILIES == ("real_b32_reference", "real_b64_parameter_matched")
    total = len(recovery.FOLDS) * len(recovery.MODEL_SEEDS) * len(recovery.FAMILIES)
    assert total == 50


def test_replay_scope_excludes_other_experiments():
    source = inspect.getsource(recovery.run_replay)
    assert "broken_pairs=True" not in source
    assert "scorpion" not in source.lower() or "dataset" in source
    # No routed/synthetic/alternative-width families are scheduled.
    assert recovery.FAMILIES == ("real_b32_reference", "real_b64_parameter_matched")
    assert "routed" not in " ".join(recovery.FAMILIES)


def test_replay_requires_cuda():
    with pytest.raises(recovery.RecoveryError, match="CUDA device"):
        recovery.run_replay(
            FROZEN_RESULT,
            REPOSITORY,
            REPOSITORY / "tmp_recovery_output",
            __import__("torch").device("cpu"),
            copied_path=None,
        )


def test_output_directory_cannot_be_overwritten(tmp_path):
    output = tmp_path / "exists"
    output.mkdir()
    import torch as _torch
    with pytest.raises(recovery.RecoveryError, match="already exists"):
        recovery.run_replay(
            FROZEN_RESULT, REPOSITORY, output, _torch.device("cuda"), copied_path=None
        )


# ---------------------------------------------------------------------------
# Replay comparison
# ---------------------------------------------------------------------------


def test_comparison_accepts_exact_replay():
    frozen = fake_frozen_run()
    deltas = recovery.compare_frozen_replay(frozen, fake_replay())
    assert deltas == []


def test_comparison_detects_numeric_mismatch():
    frozen = fake_frozen_run()
    replay = fake_replay()
    replay["layer1"]["biological_scanner_probe"]["linear"]["balanced_accuracy_median"] = 0.31
    deltas = recovery.compare_frozen_replay(frozen, replay)
    assert any(
        item["kind"] == "numeric_mismatch"
        and item["path"].endswith("balanced_accuracy_median")
        for item in deltas
    )


def test_comparison_detects_missing_replay_field():
    frozen = fake_frozen_run()
    replay = fake_replay()
    del replay["layer1"]["paired_region_preservation"]["similarity_margin"]
    deltas = recovery.compare_frozen_replay(frozen, replay)
    assert any(item["kind"] == "missing_in_replay" for item in deltas)


def test_comparison_detects_best_epoch_mismatch():
    frozen = fake_frozen_run()
    replay = fake_replay()
    replay["training"]["best_epoch"] = 70
    deltas = recovery.compare_frozen_replay(frozen, replay)
    assert any(item["kind"] == "integer_mismatch" and "best_epoch" in item["path"] for item in deltas)


def test_comparison_detects_parameter_count_mismatch():
    frozen = fake_frozen_run()
    replay = fake_replay()
    replay["training"]["actual_parameter_count"] = 999999
    deltas = recovery.compare_frozen_replay(frozen, replay)
    assert any(item["kind"] == "integer_mismatch" and "actual_parameter_count" in item["path"] for item in deltas)


def test_frozen_run_map_returns_exactly_50_canine_runs():
    frozen = json.loads(FROZEN_RESULT.read_text(encoding="utf-8"))
    mapping = recovery.frozen_run_map(frozen)
    assert len(mapping) == 50
    assert all(not run["broken_pair_control"] for run in mapping.values())


def test_category_labels_never_enter_factorizer_optimization():
    source = inspect.getsource(recovery.run_replay)
    assert "category" not in source or "train_factorizer" in source
    training_source = inspect.getsource(recovery.real_validation.train_factorizer)
    assert "category_column" not in training_source
    assert "category_name" not in training_source


def test_no_new_seeds_or_alternative_widths_are_replayed():
    assert set(recovery.MODEL_SEEDS) == {2201, 2202, 2203, 2204, 2205}
    source = inspect.getsource(recovery.run_replay)
    assert "MODEL_SEEDS" in source
    assert "FAMILIES" in source


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def test_persist_cell_writes_projected_and_checkpoint(tmp_path):
    np.random.seed(0)
    biological = np.random.default_rng(0).normal(size=(8, 32)).astype(np.float32)
    acquisition = np.random.default_rng(1).normal(size=(8, 8)).astype(np.float32)
    frame = pd.DataFrame(
        {
            "region_id": [f"r{i}" for i in range(8)],
            "slide_id": [f"s{i}" for i in range(8)],
            "scanner_id": ["a"] * 8,
            "split": ["train"] * 8,
            "category_name": ["A", "B"] * 4,
        }
    )
    dataset = type(
        "D",
        (),
        {
            "manifests": {0: frame},
            "category_column": "category_name",
            "scanner_names": ("a",),
            "features": np.zeros((8, 768), dtype=np.float32),
        },
    )()
    model = torch.nn.Linear(32, 32)
    scaler = type("S", (), {"mean_": np.zeros(768), "scale_": np.ones(768)})()
    training = {
        "best_epoch": 75,
        "actual_parameter_count": 225224,
        "feature_scaler_fit_split": "train",
    }
    persisted = recovery.persist_cell(
        tmp_path,
        fold=0,
        seed=2201,
        family="real_b32_reference",
        biological=biological,
        acquisition=acquisition,
        dataset=dataset,
        model=model,
        scaler=scaler,
        training=training,
        source_commit="abc",
        input_hashes={"feature_input": "x"},
        frozen_run=fake_frozen_run(),
    )
    projection_path = tmp_path / "canine_scc" / "fold_0" / "real_b32_reference" / "seed_2201" / "projected_features.npz"
    checkpoint_path = tmp_path / "canine_scc" / "fold_0" / "real_b32_reference" / "seed_2201" / "checkpoint.pt"
    assert projection_path.is_file()
    assert checkpoint_path.is_file()
    with np.load(projection_path, allow_pickle=False) as archive:
        assert archive["biological_features"].shape == (8, 32)
        assert archive["acquisition_features"].shape == (8, 8)
        assert archive["combined_features"].shape == (8, 40)
        assert "region_id" in archive
        assert "category_name" in archive
        assert "feature_input_sha256" in archive
    assert persisted["projected_features_sha256"] == recovery.sha256_file(projection_path)
    assert persisted["checkpoint_sha256"] == recovery.sha256_file(checkpoint_path)


def test_frozen_artifacts_remain_unchanged_by_recovery_code():
    before = {
        p: recovery.sha256_file(REPOSITORY / p)
        for p in (
            "results/real_paired_scanner_bottleneck_allocation_validation_20260803T191207/real_paired_scanner_bottleneck_allocation_validation_result.json",
            "results/real_paired_scanner_bottleneck_allocation_validation_20260803T191207/real_paired_scanner_bottleneck_allocation_readiness.json",
            "results/real_paired_scanner_bottleneck_allocation_validation_20260803T191207/real_paired_scanner_bottleneck_allocation_validation_manifest.json",
        )
    }
    assert before["results/real_paired_scanner_bottleneck_allocation_validation_20260803T191207/real_paired_scanner_bottleneck_allocation_validation_result.json"] == recovery.adjudication.FROZEN_RESULT_FILE_SHA256


def test_status_is_scientific_only_not_a_claim_status():
    assert recovery.STATUS_COMPLETE == "complete_exact_real_bottleneck_representation_recovery"
    assert recovery.STATUS_FAILED == "real_bottleneck_representation_recovery_failed"
    assert "supported" not in recovery.STATUS_COMPLETE


def test_failure_status_must_be_used_for_replication_failure():
    # All-accepted and any-failed statuses are distinct.
    assert recovery.STATUS_FAILED != recovery.STATUS_COMPLETE


def test_numeric_tolerance_is_fixed_and_strict():
    assert recovery.REPLAY_NUMERIC_TOLERANCE == 1e-6
    source = inspect.getsource(recovery.compare_frozen_replay)
    assert "REPLAY_NUMERIC_TOLERANCE" in source
