#!/usr/bin/env python3
"""Exact deterministic replay of the frozen canine B32/B64 neural grid.

This runner performs a provenance-remediation replay: the frozen real
paired-scanner run persisted no per-cell projected representations or
checkpoints, which left the fixed-estimand adjudication at
``fixed_estimand_adjudication_not_ready``. This runner recovers exactly the 50
primary canine cells (5 folds x 5 seeds x 2 families) so that a versioned,
no-training adjudication can later be executed.

This is **not** a new experiment. It reuses the exact frozen training
implementation from commit e95d8526, the exact frozen feature arrays, metadata,
folds, preprocessing, architecture definitions, parameter counts, losses,
optimizer, learning rate, weight decay, epochs, checkpoint-selection policy,
deterministic settings, seeds, and probe configurations. Every replayed cell is
compared against its frozen per-run record under a strict deterministic
tolerance fixed before execution.

Scope: canine SCC only. SCORPION, broken-pair controls, routed consensus,
synthetic models, alternative widths, alternative hidden sizes, and new seeds
are never replayed. No WSI or pixel model is constructed.

If any required cell fails deterministic replication the runner preserves all
partial outputs and returns ``real_bottleneck_representation_recovery_failed``;
Phase C is then not run.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch

from experiments.paired_acquisition import (
    run_fixed_estimand_real_feature_space_adjudication as adjudication,
)
from experiments.paired_acquisition import (
    run_real_paired_scanner_bottleneck_allocation_validation as real_validation,
)


SCHEMA_VERSION = "real-bottleneck-representation-recovery/v1"
STATUS_COMPLETE = "complete_exact_real_bottleneck_representation_recovery"
STATUS_FAILED = "real_bottleneck_representation_recovery_failed"

FROZEN_SOURCE_COMMIT = "e95d8526958ac781748f92b4ebb617b75a52fce0"
SOURCE_FILES = (
    "experiments/paired_acquisition/run_real_paired_scanner_bottleneck_allocation_validation.py",
    "experiments/paired_acquisition/run_crossed_target_scanner_prototype_factorization.py",
    "experiments/paired_acquisition/run_synthetic_crossed_factor_identifiability.py",
)

FOLDS = real_validation.FOLDS
MODEL_SEEDS = real_validation.MODEL_SEEDS
FAMILIES = real_validation.FAMILIES
ACQUISITION_DIM = real_validation.ACQUISITION_DIM

# Fixed before any replay: strict deterministic replication tolerance. The frozen
# run and the replay use identical code, data, seeds, and device, so a genuine
# replication must agree far below this bound. It is never widened after a
# mismatch.
REPLAY_NUMERIC_TOLERANCE = 1e-6
PROJECTION_BATCH_SIZE = 512

CLAIM_SCOPE = {
    "replay_is_provenance_remediation": True,
    "replay_is_not_a_new_architecture_experiment": True,
    "uses_frozen_training_implementation_from_commit": FROZEN_SOURCE_COMMIT,
    "replays_only_canine_primary_grid": True,
    "pixel_space_reconstruction_claimed": False,
    "category_labels_enter_factorizer_optimization": False,
    "swap_or_layer2_claims_made": False,
}


class RecoveryError(RuntimeError):
    """A structural or execution failure distinct from a replication failure."""


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def hash_string_array(values: Iterable[Any]) -> str:
    normalized = [
        "\x1f".join(map(str, value))
        if isinstance(value, (tuple, list, np.ndarray))
        else str(value)
        for value in values
    ]
    return hashlib.sha256("\n".join(normalized).encode("utf-8")).hexdigest()


def canonical_hash(value: Mapping[str, Any]) -> str:
    return adjudication.canonical_hash(value)


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    adjudication.atomic_json(path, value)


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=adjudication.heterogeneous_fieldnames(rows)
            )
            writer.writeheader()
            writer.writerows(dict(row) for row in rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def git_commit(repository_root: Path) -> str | None:
    completed = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.stdout.strip() if completed.returncode == 0 else None


def git_diff_quiet(repository_root: Path, commit: str, path: str) -> bool:
    completed = subprocess.run(
        ["git", "diff", "--quiet", commit, "--", path],
        cwd=repository_root,
        capture_output=True,
        text=True,
        check=False,
    )
    return completed.returncode == 0


def git_blob_sha256(repository_root: Path, commit: str, path: str) -> str:
    completed = subprocess.run(
        ["git", "show", f"{commit}:{path}"],
        cwd=repository_root,
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RecoveryError(f"Could not read source blob at {commit}:{path}")
    return hashlib.sha256(completed.stdout).hexdigest()


def source_equivalent_check(repository_root: Path) -> dict[str, Any]:
    """Verify the working training implementation equals the frozen commit's."""
    records: dict[str, Any] = {}
    for relative in SOURCE_FILES:
        path = repository_root / relative
        if not path.is_file():
            raise RecoveryError(f"Source file missing: {path}")
        working_hash = sha256_file(path)
        frozen_blob = git_blob_sha256(repository_root, FROZEN_SOURCE_COMMIT, relative)
        working_normalized = git_diff_quiet(repository_root, FROZEN_SOURCE_COMMIT, relative)
        if working_hash != frozen_blob and not working_normalized:
            raise RecoveryError(
                f"Training source is not equivalent to {FROZEN_SOURCE_COMMIT}: {relative}"
            )
        records[relative] = {
            "working_file_sha256": working_hash,
            "frozen_commit": FROZEN_SOURCE_COMMIT,
            "frozen_blob_sha256": frozen_blob,
            "source_equivalent": True,
        }
    return {
        "frozen_source_commit": FROZEN_SOURCE_COMMIT,
        "files": records,
        "all_source_equivalent": True,
    }


# ---------------------------------------------------------------------------
# Frozen per-run records and replay comparison
# ---------------------------------------------------------------------------

_MISSING = object()


def compare_tree(
    frozen: Any,
    replay: Any,
    path: str,
    deltas: list[dict[str, Any]],
    *,
    tolerance: float,
) -> None:
    if isinstance(frozen, Mapping) and isinstance(replay, Mapping):
        for key, frozen_value in frozen.items():
            compare_tree(
                frozen_value,
                replay.get(key, _MISSING),
                f"{path}.{key}",
                deltas,
                tolerance=tolerance,
            )
        return
    if isinstance(frozen, (list, tuple)):
        if not isinstance(replay, (list, tuple)) or len(frozen) != len(replay):
            deltas.append(
                {
                    "path": path,
                    "kind": "length_mismatch",
                    "frozen": len(frozen),
                    "replay": len(replay) if isinstance(replay, (list, tuple)) else None,
                }
            )
            return
        for index, (frozen_value, replay_value) in enumerate(zip(frozen, replay)):
            compare_tree(
                frozen_value,
                replay_value,
                f"{path}[{index}]",
                deltas,
                tolerance=tolerance,
            )
        return
    if replay is _MISSING:
        deltas.append({"path": path, "kind": "missing_in_replay", "frozen": frozen})
        return
    if isinstance(frozen, bool):
        if replay != frozen:
            deltas.append({"path": path, "kind": "boolean_mismatch", "frozen": frozen, "replay": replay})
        return
    if isinstance(frozen, (int, np.integer)) and not isinstance(frozen, bool):
        if isinstance(replay, bool):
            deltas.append({"path": path, "kind": "type_mismatch", "frozen": frozen, "replay": replay})
        elif int(replay) != int(frozen):
            deltas.append({"path": path, "kind": "integer_mismatch", "frozen": int(frozen), "replay": int(replay)})
        return
    if isinstance(frozen, (float, np.floating)):
        if isinstance(replay, bool) or not isinstance(replay, (int, float, np.integer, np.floating)):
            deltas.append({"path": path, "kind": "type_mismatch", "frozen": frozen, "replay": replay})
        elif abs(float(frozen) - float(replay)) > tolerance:
            deltas.append(
                {
                    "path": path,
                    "kind": "numeric_mismatch",
                    "frozen": float(frozen),
                    "replay": float(replay),
                    "absolute_difference": float(abs(float(frozen) - float(replay))),
                }
            )
        return
    if frozen is None:
        if replay is not None:
            deltas.append({"path": path, "kind": "none_mismatch", "frozen": None, "replay": replay})
        return
    if isinstance(frozen, str):
        if str(replay) != frozen:
            deltas.append({"path": path, "kind": "string_mismatch", "frozen": frozen, "replay": replay})
        return
    if frozen != replay:
        deltas.append({"path": path, "kind": "value_mismatch", "frozen": frozen, "replay": replay})


def compare_frozen_replay(
    frozen_run: Mapping[str, Any],
    replay: Mapping[str, Any],
) -> list[dict[str, Any]]:
    deltas: list[dict[str, Any]] = []
    for section, frozen_value in (
        ("training", frozen_run["training"]),
        ("layer1", frozen_run["layer1"]),
        ("layer2", frozen_run["layer2"]),
        ("pixel_space_evaluation_performed", frozen_run["pixel_space_evaluation_performed"]),
        ("biological_dimension", frozen_run["biological_dimension"]),
        ("hidden_width", frozen_run["hidden_width"]),
        ("acquisition_dimension", frozen_run["acquisition_dimension"]),
    ):
        compare_tree(
            frozen_value,
            replay.get(section, _MISSING),
            section,
            deltas,
            tolerance=REPLAY_NUMERIC_TOLERANCE,
        )
    return deltas


def frozen_run_map(result: Mapping[str, Any]) -> dict[tuple[int, int, str], dict[str, Any]]:
    mapping: dict[tuple[int, int, str], dict[str, Any]] = {}
    for run in result.get("runs", []):
        if (
            run.get("dataset") == "canine_scc"
            and not run.get("broken_pair_control")
            and run.get("family") in FAMILIES
        ):
            mapping[(run["fold"], run["seed"], run["family"])] = run
    return mapping


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def persist_cell(
    output_root: Path,
    *,
    fold: int,
    seed: int,
    family: str,
    biological: np.ndarray,
    acquisition: np.ndarray,
    dataset: Any,
    model: torch.nn.Module,
    scaler: Any,
    training: Mapping[str, Any],
    source_commit: str,
    input_hashes: Mapping[str, str],
    frozen_run: Mapping[str, Any],
) -> dict[str, Any]:
    cell_dir = (
        output_root
        / "canine_scc"
        / f"fold_{fold}"
        / family
        / f"seed_{seed}"
    )
    cell_dir.mkdir(parents=True, exist_ok=True)
    frame = dataset.manifests[fold]
    category = dataset.category_column
    combined = np.concatenate([biological, acquisition], axis=1).astype(np.float32)
    row_index = np.arange(len(biological), dtype=np.int64)
    hashes = {
        "feature_input_sha256": input_hashes["feature_input"],
        "row_order_sha256": hash_string_array(
            frame["region_id"].astype(str) + "|" + frame["scanner_id"].astype(str)
        ),
        "region_order_sha256": hash_string_array(frame["region_id"].astype(str)),
        "slide_order_sha256": hash_string_array(frame["slide_id"].astype(str)),
        "scanner_order_sha256": hash_string_array(frame["scanner_id"].astype(str)),
        "category_order_sha256": hash_string_array(
            frame[category].astype(str) if category else ["<none>"] * len(frame)
        ),
    }

    def unicode_array(values: Any, width: int = 128) -> np.ndarray:
        return np.asarray(values, dtype=f"U{width}")

    archive: dict[str, Any] = {
        "biological_features": np.ascontiguousarray(biological, dtype="<f4"),
        "acquisition_features": np.ascontiguousarray(acquisition, dtype="<f4"),
        "combined_features": np.ascontiguousarray(combined, dtype="<f4"),
        "row_index": row_index,
        "region_id": unicode_array(frame["region_id"].astype(str).to_numpy()),
        "slide_id": unicode_array(frame["slide_id"].astype(str).to_numpy()),
        "scanner_id": unicode_array(frame["scanner_id"].astype(str).to_numpy()),
        "split": unicode_array(frame["split"].astype(str).to_numpy()),
        "fold": np.full(len(biological), fold, dtype=np.int64),
        "family": unicode_array([family] * len(biological), width=64),
        "seed": np.full(len(biological), seed, dtype=np.int64),
        **hashes,
    }
    if category:
        archive["category_name"] = unicode_array(
            frame[category].astype(str).to_numpy(), width=128
        )
    projection_path = cell_dir / "projected_features.npz"
    np.savez(projection_path, **archive)

    scaler_state = {
        "mean_": np.asarray(scaler.mean_, dtype="<f4"),
        "scale_": np.asarray(scaler.scale_, dtype="<f4"),
    }

    def to_cpu(value: Any) -> Any:
        if hasattr(value, "cpu"):
            return value.cpu()
        return value

    checkpoint: dict[str, Any] = {
        "model_state_dict": {key: to_cpu(value) for key, value in model.state_dict().items()},
        "architecture": {
            "input_dim": int(dataset.features.shape[1]),
            "biological_dim": int(frozen_run["biological_dimension"]),
            "acquisition_dim": int(ACQUISITION_DIM),
            "hidden_dim": int(frozen_run["hidden_width"]),
            "scanners": len(dataset.scanner_names),
            "family": family,
            "fold": fold,
            "seed": seed,
        },
        "selected_epoch": training["best_epoch"],
        "parameter_count": training["actual_parameter_count"],
        "source_commit": source_commit,
        "input_hashes": dict(input_hashes),
        "inference_config": {
            "feature_scaler_fit_split": "train",
            "scaler": scaler_state,
            "projection_batch_size": PROJECTION_BATCH_SIZE,
            "deterministic_algorithms": True,
        },
        "cubLas_workspace_config": ":4096:8",
    }
    checkpoint_path = cell_dir / "checkpoint.pt"
    torch.save(checkpoint, checkpoint_path)
    return {
        "projected_features_path": str(projection_path.resolve()),
        "projected_features_sha256": sha256_file(projection_path),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "representation_hashes": hashes,
    }


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def replay_cell(
    dataset: Any,
    fold: int,
    seed: int,
    family: int,
    parameter_match: Mapping[str, Any],
    device: torch.device,
    frozen_run: Mapping[str, Any],
) -> dict[str, Any]:
    started = time.time()
    model, scaler, training = real_validation.train_factorizer(
        dataset,
        fold,
        family,
        seed,
        parameter_match,
        device,
        broken_pairs=False,
    )
    biological, acquisition = real_validation.project_factorizer(model, scaler, dataset, fold, device)
    layer1 = real_validation.evaluate_representation(biological, acquisition, dataset, fold)
    layer2 = {
        "available": False,
        "reason": ["no verified swap/pair-assignment metadata with source-region and target-scanner provenance"],
    }
    replay = {
        "training": training,
        "layer1": layer1,
        "layer2": layer2,
        "pixel_space_evaluation_performed": False,
        "biological_dimension": int(frozen_run["biological_dimension"]),
        "hidden_width": int(frozen_run["hidden_width"]),
        "acquisition_dimension": int(ACQUISITION_DIM),
        "runtime_seconds": time.time() - started,
    }
    deltas = compare_frozen_replay(frozen_run, replay)
    accepted = not deltas
    return {
        "replay": replay,
        "model": model,
        "scaler": scaler,
        "biological": biological,
        "acquisition": acquisition,
        "deltas": deltas,
        "accepted": accepted,
    }


def run_replay(
    frozen_result_path: Path,
    repository_root: Path,
    output_root: Path,
    device: torch.device,
    copied_path: Path | None = None,
) -> dict[str, Any]:
    if output_root.exists():
        raise RecoveryError(f"Output directory already exists: {output_root}")
    if device.type != "cuda":
        raise RecoveryError("Exact replay requires a CUDA device matching the frozen run.")

    frozen_verification = adjudication.verify_frozen_real_validation(
        frozen_result_path, repository_root, copied_path=copied_path
    )
    source_check = source_equivalent_check(repository_root)

    result_dir = frozen_result_path.resolve().parent
    readiness_path = result_dir / "real_paired_scanner_bottleneck_allocation_readiness.json"
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    parameter_match = readiness["datasets"]["canine_scc"]["parameter_match"]

    frozen_value = json.loads(frozen_result_path.read_text(encoding="utf-8"))
    runs_by_cell = frozen_run_map(frozen_value)
    if len(runs_by_cell) != 50:
        raise RecoveryError(f"Expected 50 frozen canine primary runs, found {len(runs_by_cell)}.")

    feature_path = repository_root / adjudication.CANINE_FEATURE_PATH
    feature_input_hash = sha256_file(feature_path)
    input_hashes = {
        "feature_input": feature_input_hash,
        "frozen_result": frozen_verification["frozen_result"]["file_sha256"],
        "frozen_readiness": frozen_verification["frozen_readiness"]["file_sha256"],
        "frozen_manifest": frozen_verification["frozen_manifest"]["file_sha256"],
    }

    dataset = real_validation.load_dataset(repository_root, "canine_scc")
    source_commit = git_commit(repository_root)
    cells: list[dict[str, Any]] = []
    fit_count = 0
    for fold in FOLDS:
        for seed in MODEL_SEEDS:
            for family in FAMILIES:
                frozen_run = runs_by_cell[(fold, seed, family)]
                outcome = replay_cell(
                    dataset,
                    fold,
                    seed,
                    family,
                    parameter_match,
                    device,
                    frozen_run,
                )
                persisted = persist_cell(
                    output_root,
                    fold=fold,
                    seed=seed,
                    family=family,
                    biological=outcome["biological"],
                    acquisition=outcome["acquisition"],
                    dataset=dataset,
                    model=outcome["model"],
                    scaler=outcome["scaler"],
                    training=outcome["replay"]["training"],
                    source_commit=str(source_commit),
                    input_hashes=input_hashes,
                    frozen_run=frozen_run,
                )
                fit_count += 1
                cells.append(
                    {
                        "dataset": "canine_scc",
                        "fold": fold,
                        "seed": seed,
                        "family": family,
                        "accepted": outcome["accepted"],
                        "deltas": outcome["deltas"],
                        "parameter_count_expected": int(parameter_match[family]["formula_parameter_count"]),
                        "parameter_count_actual": int(outcome["replay"]["training"]["actual_parameter_count"]),
                        "parameter_count_matches": int(outcome["replay"]["training"]["actual_parameter_count"])
                        == int(parameter_match[family]["formula_parameter_count"]),
                        "selected_checkpoint_epoch": int(outcome["replay"]["training"]["best_epoch"]),
                        "frozen_selected_checkpoint_epoch": int(frozen_run["training"]["best_epoch"]),
                        "best_validation_loss_replay": float(outcome["replay"]["training"]["best_validation_loss"]),
                        "best_validation_loss_frozen": float(frozen_run["training"]["best_validation_loss"]),
                        **persisted,
                    }
                )
                print(
                    f"completed replay fit {fit_count}: fold={fold} seed={seed} family={family} accepted={outcome['accepted']}",
                    flush=True,
                )
                del outcome["model"]
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    accepted_cells = [cell for cell in cells if cell["accepted"]]
    all_accepted = len(accepted_cells) == len(cells) == 50
    status = STATUS_COMPLETE if all_accepted else STATUS_FAILED
    return {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": CLAIM_SCOPE,
        "git_commit": source_commit,
        "device": str(device),
        "source_check": source_check,
        "frozen_verification": frozen_verification,
        "frozen_source_commit": FROZEN_SOURCE_COMMIT,
        "replay_grid": {
            "dataset": "canine_scc",
            "backbone": "dinov2_base",
            "folds": list(FOLDS),
            "model_seeds": list(MODEL_SEEDS),
            "families": list(FAMILIES),
            "total_cells": len(cells),
        },
        "environment": {
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "cuda_device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
            "cuda_device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "cudnn_version": torch.backends.cudnn.version() if torch.cuda.is_available() else None,
            "deterministic_algorithms": True,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "replay_settings": {
            "epochs": real_validation.EPOCHS,
            "learning_rate": real_validation.LEARNING_RATE,
            "weight_decay": real_validation.WEIGHT_DECAY,
            "optimizer": "AdamW",
            "checkpoint_selection": "minimum validation objective only",
            "deterministic_seed_handling": "set_deterministic_seed per fit",
            "numeric_tolerance": REPLAY_NUMERIC_TOLERANCE,
            "projection_batch_size": PROJECTION_BATCH_SIZE,
            "frozen_device": "cuda",
        },
        "input_hashes": input_hashes,
        "parameter_match": parameter_match,
        "replay_fit_count": fit_count,
        "cells": cells,
        "accepted_cell_count": len(accepted_cells),
        "total_cell_count": len(cells),
        "all_cells_replicated": all_accepted,
        "status": status,
        "failure_reasons": [] if all_accepted else [
            f"cell replication failed: dataset=canine_scc fold={cell['fold']} seed={cell['seed']} family={cell['family']}"
            for cell in cells
            if not cell["accepted"]
        ],
    }


def summary_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for cell in result.get("cells", []):
        rows.append(
            {
                "row_type": "cell",
                "dataset": cell["dataset"],
                "fold": cell["fold"],
                "seed": cell["seed"],
                "family": cell["family"],
                "accepted": cell["accepted"],
                "parameter_count_expected": cell["parameter_count_expected"],
                "parameter_count_actual": cell["parameter_count_actual"],
                "selected_checkpoint_epoch": cell["selected_checkpoint_epoch"],
                "projected_features_sha256": cell["projected_features_sha256"],
                "checkpoint_sha256": cell["checkpoint_sha256"],
            }
        )
    rows.append({"row_type": "top_level", "status": result["status"]})
    return rows


def write_outputs(output_root: Path, result: Mapping[str, Any]) -> None:
    result["result_sha256"] = canonical_hash(result)
    result_path = output_root / "real_bottleneck_representation_recovery_result.json"
    summary_path = output_root / "real_bottleneck_representation_recovery_summary.csv"
    manifest_path = output_root / "real_bottleneck_representation_recovery_manifest.json"
    atomic_json(result_path, result)
    atomic_csv(summary_path, summary_rows(result))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": CLAIM_SCOPE,
        "git_commit": result["git_commit"],
        "status": result["status"],
        "canonical_internal_result_hash": result["result_sha256"],
        "replay_grid": result["replay_grid"],
        "accepted_cell_count": result["accepted_cell_count"],
        "total_cell_count": result["total_cell_count"],
        "source_check": result["source_check"],
        "frozen_verification": result["frozen_verification"],
        "input_hashes": result["input_hashes"],
        "artifacts": {
            "result": result_path.name,
            "summary": summary_path.name,
            "manifest": manifest_path.name,
        },
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    atomic_json(manifest_path, manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-validation-result", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument(
        "--copied-result",
        type=Path,
        default=None,
        help="Optional copied frozen result path to verify alongside the repository path.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    copied = (
        Path(adjudication.FROZEN_RESULT_COPY_PATH)
        if args.copied_result is None
        else args.copied_result
    )
    started = time.time()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RecoveryError("CUDA was requested but is unavailable.")
    result = run_replay(
        args.real_validation_result.resolve(),
        args.repository_root.resolve(),
        args.output_root.resolve(),
        device,
        copied_path=copied,
    )
    write_outputs(args.output_root.resolve(), result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "replay_fit_count": result["replay_fit_count"],
                "accepted_cell_count": result["accepted_cell_count"],
                "total_cell_count": result["total_cell_count"],
                "result_sha256": result["result_sha256"],
                "output_root": str(args.output_root.resolve()),
                "runtime_seconds": time.time() - started,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
