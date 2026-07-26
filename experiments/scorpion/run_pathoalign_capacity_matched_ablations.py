#!/usr/bin/env python3
"""Run the frozen, resumable SCORPION objective-ablation campaign.

Full mode executes the preregistered seven variants, five slide-blocked folds,
five seeds, and 75-epoch schedule (175 fits). Smoke mode executes all seven
variants for fold 0 and seed 801 for one epoch and marks every output as
ineligible for scientific evidence.

Every cell uses a deterministic identity, an append-only status ledger, unique
attempt directories, artifact hashes, and fail-closed resume validation. The
runner never imports historical result tables.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
REPOSITORY_IMPORT_ROOT = Path(__file__).resolve().parents[2]
if str(REPOSITORY_IMPORT_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_IMPORT_ROOT))

import numpy as np
import pandas as pd
import torch

from experiments.scorpion import run_pathoalign_crossfold as crossfold
from experiments.scorpion.run_pathoalign_projection import ExperimentError, train_one
from scripts.scorpion import analyze_pathoalign_crossfold as metrics
from src.models.scorpion_pathoalign import ScorpionProjection

FOLDS = tuple(range(5))
SEEDS = tuple(range(801, 806))
SMOKE_FOLDS = (0,)
SMOKE_SEEDS = (801,)
SCANNERS = ("AT2", "B300", "DP200", "GT450", "P1000")
LEDGER_STATUSES = {"pending", "running", "completed", "failed", "invalid"}
SCHEMA_VERSION = "scorpion-capacity-matched-ablations/v1"

FROZEN_SCHEDULE = {
    "epochs": 75,
    "region_batch_size": 32,
    "learning_rate": 3e-4,
    "weight_decay": 1e-4,
}

VARIANTS: dict[str, dict[str, float | str]] = {
    "paired_reference": {
        "method": "paired_consistency",
        "scanner_adversary_weight": 0.0,
        "scanner_acquisition_weight": 0.0,
        "scanner_dependence_weight": 0.0,
        "cross_covariance_weight": 0.0,
        "gradient_reversal_strength": 0.0,
    },
    "two_branch_no_scanner_objectives": {
        "method": "pathoalign",
        "scanner_adversary_weight": 0.0,
        "scanner_acquisition_weight": 0.0,
        "scanner_dependence_weight": 0.0,
        "cross_covariance_weight": 0.0,
        "gradient_reversal_strength": 0.0,
    },
    "pathoalign_dep20": {
        "method": "pathoalign",
        "scanner_adversary_weight": 0.5,
        "scanner_acquisition_weight": 0.5,
        "scanner_dependence_weight": 20.0,
        "cross_covariance_weight": 0.05,
        "gradient_reversal_strength": 1.0,
    },
    "no_adversary": {
        "method": "pathoalign",
        "scanner_adversary_weight": 0.0,
        "scanner_acquisition_weight": 0.5,
        "scanner_dependence_weight": 20.0,
        "cross_covariance_weight": 0.05,
        "gradient_reversal_strength": 0.0,
    },
    "no_acquisition_classifier": {
        "method": "pathoalign",
        "scanner_adversary_weight": 0.5,
        "scanner_acquisition_weight": 0.0,
        "scanner_dependence_weight": 20.0,
        "cross_covariance_weight": 0.05,
        "gradient_reversal_strength": 1.0,
    },
    "no_scanner_dependence": {
        "method": "pathoalign",
        "scanner_adversary_weight": 0.5,
        "scanner_acquisition_weight": 0.5,
        "scanner_dependence_weight": 0.0,
        "cross_covariance_weight": 0.05,
        "gradient_reversal_strength": 1.0,
    },
    "no_cross_covariance": {
        "method": "pathoalign",
        "scanner_adversary_weight": 0.5,
        "scanner_acquisition_weight": 0.5,
        "scanner_dependence_weight": 20.0,
        "cross_covariance_weight": 0.0,
        "gradient_reversal_strength": 1.0,
    },
}

VARIANT_NOTES = {
    "paired_reference": (
        "Historical one-branch paired-consistency comparator. It is intentionally "
        "not capacity matched; capacity claims must use two_branch_no_scanner_objectives."
    ),
    "two_branch_no_scanner_objectives": (
        "Two-branch capacity control with scanner adversary, acquisition classifier, "
        "scanner-dependence penalty, cross-covariance penalty, and gradient reversal disabled."
    ),
    "pathoalign_dep20": "Full registered two-branch factorization objective.",
    "no_adversary": "Scanner adversary and gradient reversal disabled.",
    "no_acquisition_classifier": "Acquisition-branch scanner-classification loss disabled.",
    "no_scanner_dependence": "Direct scanner-dependence penalty disabled.",
    "no_cross_covariance": "Biological/acquisition cross-covariance penalty disabled.",
}

FROZEN_FEATURE = {
    "sha256": "dbbd75887b8674921e388e4a0d09635a658ab078e934b67326af3fcc69483e71",
    "size_bytes": 6_869_432,
    "sample_count": 2_400,
    "feature_dim": 768,
    "slide_count": 48,
    "region_count": 480,
    "originating_extraction_commit": "0ab4ca0413fee44e83a2bf280554134691dd39a4",
    "model_revision": "f9e44c814b77203eaa57a6bdbbd535f21ede1415",
}

FROZEN_MANIFESTS = {
    0: {
        "sha256": "4ba23cd72346fb2e55766ad0aac8bc0c1cdcfcbbbfc0274fb2e6556ef114743c",
        "size_bytes": 202_470,
    },
    1: {
        "sha256": "3624c2024650444da8a27ce2a2d95833bdbe46b9bf4acc759db9acd82e68822b",
        "size_bytes": 202_470,
    },
    2: {
        "sha256": "a55374e7eb88c19c532d1f8ab1ccfd9f25cfa8a3ab076e02a20963b85f64a0ea",
        "size_bytes": 202_570,
    },
    3: {
        "sha256": "fdfcfa3d1d37685958e3ba1339cf117fd5707df13f404c27c92594037a975411",
        "size_bytes": 202_620,
    },
    4: {
        "sha256": "5231379f03e2510c5d851dd09c8d4d5ab3af21a3defe35dbe0371d655a9a79ae",
        "size_bytes": 202_520,
    },
}
MANIFEST_ORIGIN_COMMIT = "426274d398a5830ec79c676d64d145c58edc758f"


@dataclass(frozen=True)
class Cell:
    fold: int
    variant: str
    seed: int
    run_id: str
    config_hash: str


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def canonical_json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def canonical_hash(value: Any) -> str:
    return hashlib.sha256(canonical_json(value).encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    if temporary.exists():
        raise ExperimentError(f"Refusing to overwrite temporary file: {temporary}")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ExperimentError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    if temporary.exists():
        raise ExperimentError(f"Refusing to overwrite temporary file: {temporary}")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def git_output(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.stdout.strip()


def repository_root() -> Path:
    return Path(git_output(Path.cwd(), "rev-parse", "--show-toplevel")).resolve()


def repository_relative(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(root).as_posix()
    except ValueError as exc:
        raise ExperimentError(f"Required path is outside the repository: {resolved}") from exc


def is_tracked(root: Path, relative_path: str) -> bool:
    completed = subprocess.run(
        ["git", "ls-files", "--error-unmatch", "--", relative_path],
        cwd=root,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    return completed.returncode == 0


def source_state(root: Path) -> dict[str, Any]:
    status = git_output(root, "status", "--porcelain", "--untracked-files=no")
    return {
        "commit": git_output(root, "rev-parse", "HEAD"),
        "tree": git_output(root, "rev-parse", "HEAD^{tree}"),
        "tracked_worktree_clean": not bool(status),
    }


def source_file_hashes(root: Path) -> dict[str, str]:
    paths = (
        "experiments/scorpion/pathoalign_capacity_matched_analysis_spec.json",
        "experiments/scorpion/run_pathoalign_capacity_matched_ablations.py",
        "experiments/scorpion/run_pathoalign_crossfold.py",
        "experiments/scorpion/run_pathoalign_projection.py",
        "scripts/scorpion/analyze_pathoalign_capacity_matched_ablations.py",
        "scripts/scorpion/analyze_pathoalign_crossfold.py",
        "src/models/scorpion_pathoalign.py",
    )
    return {path: sha256_file(root / path) for path in paths}


def package_versions() -> dict[str, str]:
    versions: dict[str, str] = {}
    for distribution in importlib.metadata.distributions():
        name = distribution.metadata.get("Name")
        if name:
            versions[str(name).lower()] = distribution.version
    return dict(sorted(versions.items()))


def capture_environment() -> dict[str, Any]:
    gpu_rows: list[dict[str, str]] = []
    try:
        output = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,driver_version,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
        ).stdout
        for line in output.splitlines():
            index, name, driver, memory = [item.strip() for item in line.split(",", 3)]
            gpu_rows.append(
                {
                    "index": index,
                    "name": name,
                    "driver_version": driver,
                    "memory_mib": memory,
                }
            )
    except (FileNotFoundError, subprocess.CalledProcessError):
        gpu_rows = []
    return {
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "torch_cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "cuda_available": torch.cuda.is_available(),
        "gpus": gpu_rows,
        "packages": package_versions(),
    }


def _validate_frozen_file(path: Path, expected: dict[str, Any]) -> None:
    if not path.is_file():
        raise ExperimentError(f"Missing frozen input: {path}")
    observed_size = path.stat().st_size
    observed_hash = sha256_file(path)
    if observed_size != expected["size_bytes"] or observed_hash != expected["sha256"]:
        raise ExperimentError(
            "Frozen input mismatch: "
            f"path={path}, expected_size={expected['size_bytes']}, "
            f"observed_size={observed_size}, expected_sha256={expected['sha256']}, "
            f"observed_sha256={observed_hash}"
        )


def validate_inputs(
    base_features_path: Path,
    manifests_dir: Path,
    root: Path,
) -> tuple[np.ndarray, pd.DataFrame, dict[str, Any], dict[int, tuple[np.ndarray, pd.DataFrame]]]:
    _validate_frozen_file(base_features_path, FROZEN_FEATURE)
    base_features, base_frame, source_metadata = crossfold.load_archive(base_features_path)
    if list(base_features.shape) != [
        FROZEN_FEATURE["sample_count"],
        FROZEN_FEATURE["feature_dim"],
    ]:
        raise ExperimentError(f"Unexpected frozen feature shape: {base_features.shape}")
    if int(base_frame["slide_id"].nunique()) != FROZEN_FEATURE["slide_count"]:
        raise ExperimentError("Frozen feature archive does not contain 48 slides.")
    if int(base_frame["region_id"].nunique()) != FROZEN_FEATURE["region_count"]:
        raise ExperimentError("Frozen feature archive does not contain 480 regions.")
    if set(base_frame["scanner_id"]) != set(SCANNERS):
        raise ExperimentError("Frozen feature archive has unexpected scanner labels.")
    if base_frame["scanner_id"].isna().any():
        raise ExperimentError("Frozen feature archive has missing scanner labels.")
    if base_frame["path"].map(lambda value: Path(str(value)).is_absolute()).any():
        raise ExperimentError("Frozen feature rows contain absolute machine-local image paths.")
    if base_frame["path"].str.contains("canine", case=False, na=False).any():
        raise ExperimentError("Canine paths entered the SCORPION feature archive.")
    group_valid = base_frame.groupby("region_id")["scanner_id"].apply(
        lambda values: len(values) == 5 and set(values) == set(SCANNERS)
    )
    if not bool(group_valid.all()):
        raise ExperimentError("Every SCORPION region must contain one view from each scanner.")
    if source_metadata.get("model") != "dinov2_base":
        raise ExperimentError("Frozen feature archive is not the registered DINOv2-Base archive.")
    if source_metadata.get("model_revision") != FROZEN_FEATURE["model_revision"]:
        raise ExperimentError("Frozen DINOv2 model revision mismatch.")

    fold_data: dict[int, tuple[np.ndarray, pd.DataFrame]] = {}
    test_slide_assignments: list[str] = []
    inventory_rows: list[dict[str, Any]] = [
        {
            "role": "base_feature_archive",
            "repository_relative_path": repository_relative(base_features_path, root),
            "size_bytes": base_features_path.stat().st_size,
            "sha256": sha256_file(base_features_path),
            "sample_count": len(base_frame),
            "feature_dimension": base_features.shape[1],
            "fold_coverage": list(FOLDS),
            "slide_coverage": int(base_frame["slide_id"].nunique()),
            "biological_sample_definition": "original slide_id",
            "biological_sample_coverage": int(base_frame["slide_id"].nunique()),
            "scanner_coverage": sorted(base_frame["scanner_id"].unique()),
            "tracked": is_tracked(root, repository_relative(base_features_path, root)),
            "intentionally_external": True,
            "originating_experiment_or_commit": FROZEN_FEATURE["originating_extraction_commit"],
        }
    ]
    for fold in FOLDS:
        manifest_path = manifests_dir / f"fold_{fold}_manifest.csv"
        _validate_frozen_file(manifest_path, FROZEN_MANIFESTS[fold])
        aligned_features, frame = crossfold.align_fold(
            base_features,
            base_frame,
            manifest_path,
        )
        fit_indices, test_indices = crossfold.validate_fold(frame, fold)
        if frame.duplicated(list(crossfold.KEY_COLUMNS)).any():
            raise ExperimentError(f"Duplicate sample identity in fold {fold}.")
        if frame["scanner_id"].isna().any() or set(frame["scanner_id"]) != set(SCANNERS):
            raise ExperimentError(f"Missing or unexpected scanner labels in fold {fold}.")
        fit_slides = set(frame.iloc[fit_indices]["slide_id"])
        test_slides = set(frame.iloc[test_indices]["slide_id"])
        if fit_slides & test_slides:
            raise ExperimentError(f"Biological-sample leakage in fold {fold}.")
        test_slide_assignments.extend(sorted(test_slides))
        fold_data[fold] = (aligned_features, frame)
        relative_path = repository_relative(manifest_path, root)
        inventory_rows.append(
            {
                "role": f"fold_{fold}_split_manifest",
                "repository_relative_path": relative_path,
                "size_bytes": manifest_path.stat().st_size,
                "sha256": sha256_file(manifest_path),
                "sample_count": len(frame),
                "feature_dimension": base_features.shape[1],
                "fold_coverage": [fold],
                "slide_coverage": int(frame["slide_id"].nunique()),
                "fit_slide_count": len(fit_slides),
                "test_slide_count": len(test_slides),
                "biological_sample_definition": "original slide_id",
                "biological_sample_coverage": int(frame["slide_id"].nunique()),
                "scanner_coverage": sorted(frame["scanner_id"].unique()),
                "tracked": is_tracked(root, relative_path),
                "intentionally_external": True,
                "originating_experiment_or_commit": MANIFEST_ORIGIN_COMMIT,
            }
        )
    counts = pd.Series(test_slide_assignments).value_counts()
    if len(counts) != 48 or not bool((counts == 1).all()):
        raise ExperimentError("Every original slide must enter the test set exactly once.")

    inventory = {
        "schema_version": SCHEMA_VERSION,
        "status": "valid",
        "inputs": inventory_rows,
        "checks": {
            "duplicate_sample_identities": 0,
            "train_test_slide_overlap": 0,
            "test_slides_covered_exactly_once": True,
            "missing_scanner_labels": 0,
            "non_finite_feature_values": 0,
            "feature_dimensions_consistent": True,
            "machine_local_paths_used_for_training": False,
            "canine_evidence_used": False,
            "historical_result_tables_used": False,
        },
        "source_metadata": {
            "model": source_metadata["model"],
            "model_source": source_metadata["model_source"],
            "model_revision": source_metadata["model_revision"],
            "feature_dim": int(source_metadata["feature_dim"]),
            "n_images": int(source_metadata["n_images"]),
            "extraction_torch_version": source_metadata["torch_version"],
            "absolute_extraction_paths_copied_to_new_records": False,
        },
    }
    return base_features, base_frame, inventory, fold_data


def parameter_inventory(input_dim: int) -> list[dict[str, Any]]:
    rows = []
    full_count: int | None = None
    counts: dict[str, tuple[int, int]] = {}
    for variant_name, variant in VARIANTS.items():
        config = crossfold.config_for(input_dim, variant)
        model = ScorpionProjection(str(variant["method"]), config)
        total = sum(parameter.numel() for parameter in model.parameters())
        trainable = sum(
            parameter.numel() for parameter in model.parameters() if parameter.requires_grad
        )
        counts[variant_name] = (trainable, total)
        if variant_name == "pathoalign_dep20":
            full_count = total
    if full_count is None:
        raise ExperimentError("Full registered variant is missing.")
    for variant_name in VARIANTS:
        trainable, total = counts[variant_name]
        rows.append(
            {
                "variant": variant_name,
                "trainable_parameter_count": trainable,
                "total_parameter_count": total,
                "intended_removed_objective_or_branch": VARIANT_NOTES[variant_name],
                "capacity_matched_to_pathoalign_dep20": total == full_count,
                "parameter_count_difference_from_pathoalign_dep20": total - full_count,
            }
        )
    return rows


def build_design(
    *,
    root: Path,
    base_features_path: Path,
    manifests_dir: Path,
    smoke: bool,
    device: str,
    source: dict[str, Any],
    inventory: dict[str, Any],
    parameters: list[dict[str, Any]],
) -> dict[str, Any]:
    mode = "smoke" if smoke else "full"
    executed_folds = list(SMOKE_FOLDS if smoke else FOLDS)
    executed_seeds = list(SMOKE_SEEDS if smoke else SEEDS)
    design = {
        "schema_version": SCHEMA_VERSION,
        "campaign_mode": mode,
        "evidence_eligible": not smoke,
        "smoke_outputs_must_not_be_promoted": smoke,
        "source": {**source, "files": source_file_hashes(root)},
        "inputs": {row["repository_relative_path"]: row["sha256"] for row in inventory["inputs"]},
        "base_features": repository_relative(base_features_path, root),
        "manifests_directory": repository_relative(manifests_dir, root),
        "frozen_full_design": {
            "variants": VARIANTS,
            "folds": list(FOLDS),
            "seeds": list(SEEDS),
            **FROZEN_SCHEDULE,
            "expected_fit_count": 175,
        },
        "executed_design": {
            "variants": list(VARIANTS),
            "folds": executed_folds,
            "seeds": executed_seeds,
            "epochs": 1 if smoke else FROZEN_SCHEDULE["epochs"],
            "expected_fit_count": len(VARIANTS) * len(executed_folds) * len(executed_seeds),
        },
        "parameter_inventory": parameters,
        "device": device,
        "checkpoint_selection": "none; fixed final epoch only",
        "preprocessing_fit_scope": "all non-test slides in the registered fold",
        "test_labels_used_during_training_or_model_selection": False,
        "historical_metrics_imported": False,
        "claim_boundary": (
            "This campaign may support objective-level evidence under the registered "
            "SCORPION protocol. It cannot establish pure biological factors, complete "
            "scanner invariance, universal objective necessity, or clinical utility."
        ),
    }
    design["campaign_hash"] = canonical_hash(design)
    return design


def cells_for_design(design: dict[str, Any]) -> list[Cell]:
    cells: list[Cell] = []
    config_hash = str(design["campaign_hash"])
    source_commit = str(design["source"]["commit"])
    executed = design["executed_design"]
    for fold in executed["folds"]:
        for seed in executed["seeds"]:
            for variant in executed["variants"]:
                identity = {
                    "variant": variant,
                    "fold": int(fold),
                    "seed": int(seed),
                    "configuration": config_hash,
                    "source_commit": source_commit,
                }
                suffix = canonical_hash(identity)[:16]
                cells.append(
                    Cell(
                        fold=int(fold),
                        variant=str(variant),
                        seed=int(seed),
                        run_id=f"fold{fold}-{variant}-seed{seed}-{suffix}",
                        config_hash=config_hash,
                    )
                )
    if len({cell.run_id for cell in cells}) != len(cells):
        raise ExperimentError("Duplicate deterministic run identities.")
    return cells


def append_event(
    out_dir: Path,
    cell: Cell,
    status: str,
    *,
    attempt: int,
    message: str | None = None,
    manifest_sha256: str | None = None,
    details: dict[str, Any] | None = None,
) -> None:
    if status not in LEDGER_STATUSES:
        raise ExperimentError(f"Invalid ledger status: {status}")
    event = {
        "timestamp_utc": utc_now(),
        "run_id": cell.run_id,
        "variant": cell.variant,
        "fold": cell.fold,
        "seed": cell.seed,
        "config_hash": cell.config_hash,
        "status": status,
        "attempt": attempt,
        "message": message,
        "manifest_sha256": manifest_sha256,
    }
    if details:
        event.update(details)
    ledger_path = out_dir / "run_ledger.jsonl"
    with ledger_path.open("a", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(event, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def load_events(out_dir: Path) -> list[dict[str, Any]]:
    path = out_dir / "run_ledger.jsonl"
    if not path.is_file():
        return []
    events = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        try:
            event = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ExperimentError(f"Corrupt ledger JSON at line {line_number}.") from exc
        if event.get("status") not in LEDGER_STATUSES:
            raise ExperimentError(f"Invalid ledger status at line {line_number}.")
        events.append(event)
    return events


def latest_events(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    for event in events:
        latest[str(event["run_id"])] = event
    return latest


def cell_root(out_dir: Path, cell: Cell) -> Path:
    return (
        out_dir / "cells" / f"fold_{cell.fold}" / cell.variant / f"seed_{cell.seed}" / cell.run_id
    )


def ensure_immutable_json(path: Path, value: Any) -> None:
    if path.exists():
        try:
            observed = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ExperimentError(f"Corrupt immutable JSON: {path}") from exc
        if observed != value:
            raise ExperimentError(f"Immutable campaign record mismatch: {path}")
        return
    atomic_json(path, value)


def initialize_ledger(
    out_dir: Path,
    cells: list[Cell],
    design: dict[str, Any],
) -> None:
    events = load_events(out_dir)
    known = {str(event["run_id"]) for event in events}
    expected = {cell.run_id for cell in cells}
    unexpected = sorted(known - expected)
    if unexpected:
        raise ExperimentError(f"Ledger contains unexpected run identities: {unexpected[:5]}")
    for cell in cells:
        if cell.run_id not in known:
            append_event(
                out_dir,
                cell,
                "pending",
                attempt=0,
                details={
                    "source_commit": design["source"]["commit"],
                    "input_hashes": design["inputs"],
                },
            )


def prepare_fold(
    out_dir: Path,
    fold: int,
    features: np.ndarray,
    frame: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[np.ndarray], str]:
    fit_indices, test_indices = crossfold.validate_fold(frame, fold)
    transformed, mean, std = crossfold.standardize(features, fit_indices)
    groups = crossfold.region_groups(frame, fit_indices)
    path = out_dir / "fold_context" / f"fold_{fold}_standardization.npz"
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        try:
            with np.load(path, allow_pickle=False) as archive:
                observed_mean = np.asarray(archive["mean"], dtype=np.float32)
                observed_std = np.asarray(archive["std"], dtype=np.float32)
        except (KeyError, OSError, ValueError) as exc:
            raise ExperimentError(f"Corrupt existing standardization: {path}") from exc
        if not np.array_equal(observed_mean, mean) or not np.array_equal(observed_std, std):
            raise ExperimentError(f"Existing standardization mismatch: {path}")
    else:
        crossfold.atomic_npz(path, {"mean": mean, "std": std})
    return transformed, fit_indices, test_indices, groups, sha256_file(path)


def enrich_projection_metadata(
    path: Path,
    *,
    cell: Cell,
    design: dict[str, Any],
    attempt: int,
    parameter_row: dict[str, Any],
) -> None:
    try:
        with np.load(path, allow_pickle=False) as archive:
            arrays = {name: archive[name] for name in archive.files}
        metadata = json.loads(str(arrays["metadata_json"].item()))
    except (KeyError, OSError, ValueError, json.JSONDecodeError) as exc:
        raise ExperimentError(f"Unreadable projected archive: {path}") from exc
    metadata.update(
        {
            "campaign_schema_version": SCHEMA_VERSION,
            "campaign_mode": design["campaign_mode"],
            "evidence_eligible": design["evidence_eligible"],
            "run_id": cell.run_id,
            "variant": cell.variant,
            "fold": cell.fold,
            "seed": cell.seed,
            "attempt": attempt,
            "config_hash": cell.config_hash,
            "source_commit": design["source"]["commit"],
            "parameter_counts": {
                "trainable": parameter_row["trainable_parameter_count"],
                "total": parameter_row["total_parameter_count"],
            },
        }
    )
    text = json.dumps(metadata, sort_keys=True)
    arrays["metadata_json"] = np.asarray(text, dtype=f"<U{len(text)}")
    crossfold.atomic_npz(path, arrays)


def evaluate_projection(
    projected_path: Path,
    *,
    cell: Cell,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    biological, acquisition, frame, metadata = metrics.load_projected(projected_path)
    fit_indices, test_indices = metrics.split_indices(frame)
    if metadata.get("run_id") != cell.run_id or metadata.get("variant") != cell.variant:
        raise ExperimentError("Projected archive identity metadata mismatch.")
    biological_probe, biological_slide_probe = metrics.scanner_probe(
        biological,
        frame,
        fit_indices,
        test_indices,
    )
    paired, biological_slide_paired = metrics.paired_slide_metrics(
        biological,
        frame,
        test_indices,
    )
    merged = biological_slide_paired.merge(
        biological_slide_probe,
        on="slide_id",
        validate="one_to_one",
    )
    acquisition_applicable = acquisition is not None
    acquisition_probe_value: float | None = None
    acquisition_retrieval_value: float | None = None
    acquisition_rank_value: float | None = None
    cross_covariance_value: float | None = None
    if acquisition_applicable:
        acquisition_probe, acquisition_slide_probe = metrics.scanner_probe(
            acquisition,
            frame,
            fit_indices,
            test_indices,
        )
        acquisition_paired, _ = metrics.paired_slide_metrics(
            acquisition,
            frame,
            test_indices,
        )
        acquisition_probe_value = float(acquisition_probe["balanced_accuracy"])
        acquisition_retrieval_value = float(acquisition_paired["retrieval_top1_average"])
        acquisition_rank_value = metrics.effective_rank(acquisition[test_indices])
        cross_covariance_value = metrics.cross_covariance_rms(
            biological,
            acquisition,
            test_indices,
        )
        acquisition_slide_probe = acquisition_slide_probe.rename(
            columns={
                "scanner_probe_accuracy": "acquisition_scanner_probe_accuracy",
            }
        )
        merged = merged.merge(acquisition_slide_probe, on="slide_id", validate="one_to_one")
    else:
        merged["acquisition_scanner_probe_accuracy"] = None
    test_features = biological[test_indices]
    values = {
        "fold": cell.fold,
        "variant": cell.variant,
        "seed": cell.seed,
        "run_id": cell.run_id,
        "scanner_probe_balanced_accuracy": float(biological_probe["balanced_accuracy"]),
        **paired,
        "biological_effective_rank": metrics.effective_rank(test_features),
        "biological_nonzero_variance_fraction": float(np.mean(test_features.var(axis=0) > 1e-12)),
        "acquisition_metrics_applicable": acquisition_applicable,
        "acquisition_scanner_probe_balanced_accuracy": acquisition_probe_value,
        "acquisition_retrieval_top1_average": acquisition_retrieval_value,
        "acquisition_effective_rank": acquisition_rank_value,
        "cross_covariance_rms": cross_covariance_value,
        "n_fit_slides": int(frame.iloc[fit_indices]["slide_id"].nunique()),
        "n_test_slides": int(frame.iloc[test_indices]["slide_id"].nunique()),
    }
    required = [
        "scanner_probe_balanced_accuracy",
        "pair_cosine_average",
        "pair_cosine_worst",
        "retrieval_top1_average",
        "retrieval_top1_worst",
        "biological_effective_rank",
        "biological_nonzero_variance_fraction",
    ]
    if not all(math.isfinite(float(values[name])) for name in required):
        raise ExperimentError("Missing or non-finite biological evaluation metric.")
    if acquisition_applicable:
        acquisition_fields = (
            "acquisition_scanner_probe_balanced_accuracy",
            "acquisition_retrieval_top1_average",
            "acquisition_effective_rank",
            "cross_covariance_rms",
        )
        if not all(math.isfinite(float(values[name])) for name in acquisition_fields):
            raise ExperimentError("Missing or non-finite acquisition evaluation metric.")
    slide_rows = []
    for row in merged.to_dict("records"):
        slide_rows.append(
            {
                "fold": cell.fold,
                "variant": cell.variant,
                "seed": cell.seed,
                "run_id": cell.run_id,
                **row,
            }
        )
    return values, slide_rows


def _validate_no_absolute_paths(value: Any, *, context: str) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _validate_no_absolute_paths(item, context=f"{context}.{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            _validate_no_absolute_paths(item, context=f"{context}[{index}]")
    elif isinstance(value, str):
        normalized = value.replace("\\", "/")
        if (
            normalized.startswith("/")
            or (len(normalized) >= 3 and normalized[1:3] == ":/")
            or "/Users/" in normalized
            or "/home/" in normalized
        ):
            raise ExperimentError(f"Machine-local absolute path in {context}: {value}")


def validate_cell(
    out_dir: Path,
    cell: Cell,
    design: dict[str, Any],
    *,
    expected_manifest_hash: str | None,
) -> dict[str, Any]:
    root = cell_root(out_dir, cell)
    manifest_path = root / "cell_manifest.json"
    if not manifest_path.is_file():
        raise ExperimentError(f"Missing cell manifest: {manifest_path}")
    if expected_manifest_hash and sha256_file(manifest_path) != expected_manifest_hash:
        raise ExperimentError(f"Cell manifest hash mismatch: {manifest_path}")
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExperimentError(f"Corrupt cell manifest: {manifest_path}") from exc
    expected_identity = {
        "run_id": cell.run_id,
        "variant": cell.variant,
        "fold": cell.fold,
        "seed": cell.seed,
        "config_hash": cell.config_hash,
    }
    for key, expected in expected_identity.items():
        if manifest.get(key) != expected:
            raise ExperimentError(f"Cell manifest {key} mismatch: {manifest_path}")
    if manifest.get("status") != "valid":
        raise ExperimentError(f"Cell manifest is not valid: {manifest_path}")
    if manifest.get("campaign_mode") != design["campaign_mode"]:
        raise ExperimentError("Cell campaign mode mismatch.")
    if bool(manifest.get("evidence_eligible")) != bool(design["evidence_eligible"]):
        raise ExperimentError("Cell evidence-eligibility mismatch.")
    if design["campaign_mode"] == "smoke" and manifest.get("evidence_eligible"):
        raise ExperimentError("Smoke output cannot be evidence eligible.")
    _validate_no_absolute_paths(manifest, context="cell_manifest")

    attempt = int(manifest["attempt"])
    attempt_dir = root / "attempts" / f"attempt_{attempt:03d}"
    attempt_manifest_path = attempt_dir / "attempt_manifest.json"
    if not attempt_manifest_path.is_file():
        raise ExperimentError(f"Missing attempt manifest: {attempt_manifest_path}")
    if sha256_file(attempt_manifest_path) != manifest.get("attempt_manifest_sha256"):
        raise ExperimentError(f"Attempt manifest hash mismatch: {attempt_manifest_path}")
    for record in manifest["artifacts"]:
        relative_path = Path(str(record["path"]))
        if relative_path.is_absolute() or ".." in relative_path.parts:
            raise ExperimentError("Unsafe cell artifact path.")
        artifact_path = root / relative_path
        if not artifact_path.is_file():
            raise ExperimentError(f"Missing cell artifact: {artifact_path}")
        if artifact_path.stat().st_size != int(record["size_bytes"]):
            raise ExperimentError(f"Cell artifact size mismatch: {artifact_path}")
        if sha256_file(artifact_path) != record["sha256"]:
            raise ExperimentError(f"Cell artifact hash mismatch: {artifact_path}")

    history_path = attempt_dir / "training_history.csv"
    history = pd.read_csv(history_path)
    if history.empty or int(history.iloc[-1]["epoch"]) != int(manifest["executed_epochs"]):
        raise ExperimentError("Training history does not reach the fixed final epoch.")
    numeric_history = history.select_dtypes(include=[np.number]).to_numpy()
    if not np.isfinite(numeric_history).all():
        raise ExperimentError("Training history contains non-finite values.")
    if not bool(manifest.get("finite_gradients")):
        raise ExperimentError("Cell does not certify finite gradients.")

    checkpoint_path = attempt_dir / "checkpoint.pt"
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except (OSError, RuntimeError, ValueError) as exc:
        raise ExperimentError(f"Unreadable checkpoint: {checkpoint_path}") from exc
    if checkpoint.get("method") != VARIANTS[cell.variant]["method"]:
        raise ExperimentError("Checkpoint method mismatch.")
    if int(checkpoint.get("seed")) != cell.seed:
        raise ExperimentError("Checkpoint seed mismatch.")
    if int(checkpoint.get("epochs")) != int(manifest["executed_epochs"]):
        raise ExperimentError("Checkpoint epoch mismatch.")
    config = crossfold.config_for(
        FROZEN_FEATURE["feature_dim"],
        VARIANTS[cell.variant],
    )
    model = ScorpionProjection(str(VARIANTS[cell.variant]["method"]), config)
    expected_state = model.state_dict()
    observed_state = checkpoint.get("state_dict", {})
    if set(expected_state) != set(observed_state):
        raise ExperimentError("Checkpoint state-dict keys mismatch.")
    for name, tensor in expected_state.items():
        if tuple(tensor.shape) != tuple(observed_state[name].shape):
            raise ExperimentError(f"Checkpoint tensor shape mismatch: {name}")

    projected_path = attempt_dir / "projected_features.npz"
    biological, acquisition, frame, metadata = metrics.load_projected(projected_path)
    if metadata.get("run_id") != cell.run_id:
        raise ExperimentError("Projected archive run identity mismatch.")
    if bool(metadata.get("evidence_eligible")) != bool(design["evidence_eligible"]):
        raise ExperimentError("Projected archive evidence-eligibility mismatch.")
    if len(frame) != FROZEN_FEATURE["sample_count"] or biological.shape[1] != 256:
        raise ExperimentError("Projected archive shape mismatch.")
    expects_acquisition = VARIANTS[cell.variant]["method"] == "pathoalign"
    if expects_acquisition != (acquisition is not None):
        raise ExperimentError("Projected acquisition-branch presence mismatch.")

    metric_path = attempt_dir / "metrics.json"
    try:
        metric_row = json.loads(metric_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ExperimentError(f"Corrupt metric artifact: {metric_path}") from exc
    if metric_row.get("run_id") != cell.run_id:
        raise ExperimentError("Metric row identity mismatch.")
    return manifest


def build_artifact_records(root: Path, paths: list[Path]) -> list[dict[str, Any]]:
    records = []
    for path in paths:
        records.append(
            {
                "path": path.relative_to(root).as_posix(),
                "size_bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
        )
    return records


def next_attempt(root: Path) -> int:
    attempts = root / "attempts"
    if not attempts.exists():
        return 1
    numbers = []
    for path in attempts.iterdir():
        if path.is_dir() and path.name.startswith("attempt_"):
            try:
                numbers.append(int(path.name.removeprefix("attempt_")))
            except ValueError as exc:
                raise ExperimentError(f"Unexpected attempt directory: {path}") from exc
    return max(numbers, default=0) + 1


def execute_cell(
    *,
    out_dir: Path,
    cell: Cell,
    design: dict[str, Any],
    fold_data: tuple[np.ndarray, pd.DataFrame],
    device: torch.device,
    parameter_row: dict[str, Any],
) -> None:
    root = cell_root(out_dir, cell)
    root.mkdir(parents=True, exist_ok=True)
    attempt = next_attempt(root)
    attempt_dir = root / "attempts" / f"attempt_{attempt:03d}"
    attempt_dir.mkdir(parents=True, exist_ok=False)
    append_event(
        out_dir,
        cell,
        "running",
        attempt=attempt,
        details={
            "source_commit": design["source"]["commit"],
            "input_hashes": design["inputs"],
            "started_at_utc": utc_now(),
        },
    )
    context = {
        "schema_version": SCHEMA_VERSION,
        "run_id": cell.run_id,
        "variant": cell.variant,
        "fold": cell.fold,
        "seed": cell.seed,
        "attempt": attempt,
        "config_hash": cell.config_hash,
        "source_commit": design["source"]["commit"],
        "campaign_mode": design["campaign_mode"],
        "evidence_eligible": design["evidence_eligible"],
    }
    atomic_json(attempt_dir / "attempt_context.json", context)
    started_at = utc_now()
    started = time.perf_counter()
    try:
        features, frame = fold_data
        transformed, fit_indices, _, groups, standardization_hash = prepare_fold(
            out_dir,
            cell.fold,
            features,
            frame,
        )
        config = crossfold.config_for(features.shape[1], VARIANTS[cell.variant])
        executed_epochs = int(design["executed_design"]["epochs"])
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
        result = train_one(
            method=str(VARIANTS[cell.variant]["method"]),
            seed=cell.seed,
            features=transformed,
            frame=frame,
            train_indices=fit_indices,
            development_indices=np.arange(len(frame), dtype=np.int64),
            groups=groups,
            config=config,
            device=device,
            epochs=executed_epochs,
            region_batch_size=FROZEN_SCHEDULE["region_batch_size"],
            learning_rate=FROZEN_SCHEDULE["learning_rate"],
            weight_decay=FROZEN_SCHEDULE["weight_decay"],
            run_dir=attempt_dir,
            strict_determinism=True,
        )
        projected_path = attempt_dir / "projected_features.npz"
        crossfold.mark_frozen_test_projection(projected_path, cell.fold)
        enrich_projection_metadata(
            projected_path,
            cell=cell,
            design=design,
            attempt=attempt,
            parameter_row=parameter_row,
        )
        metric_row, slide_rows = evaluate_projection(projected_path, cell=cell)
        atomic_json(attempt_dir / "metrics.json", metric_row)
        atomic_csv(attempt_dir / "slide_metrics.csv", slide_rows)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
            peak_memory_bytes = int(torch.cuda.max_memory_allocated(device))
        else:
            peak_memory_bytes = None
        runtime_seconds = time.perf_counter() - started
        artifact_paths = [
            attempt_dir / "attempt_context.json",
            attempt_dir / "training_history.csv",
            attempt_dir / "checkpoint.pt",
            projected_path,
            attempt_dir / "metrics.json",
            attempt_dir / "slide_metrics.csv",
        ]
        manifest = {
            **context,
            "status": "valid",
            "started_at_utc": started_at,
            "finished_at_utc": utc_now(),
            "executed_epochs": executed_epochs,
            "runtime_seconds": runtime_seconds,
            "peak_gpu_memory_bytes": peak_memory_bytes,
            "finite_gradients": bool(result["finite_gradients"]),
            "max_preclip_gradient_norm": float(result["max_preclip_gradient_norm"]),
            "final_training_loss": float(result["final_training_loss"]),
            "standardization_sha256": standardization_hash,
            "parameter_counts": {
                "trainable": parameter_row["trainable_parameter_count"],
                "total": parameter_row["total_parameter_count"],
            },
            "input_hashes": design["inputs"],
            "artifacts": build_artifact_records(root, artifact_paths),
        }
        _validate_no_absolute_paths(manifest, context="new_cell_manifest")
        attempt_manifest_path = attempt_dir / "attempt_manifest.json"
        atomic_json(attempt_manifest_path, manifest)
        manifest["attempt_manifest_sha256"] = sha256_file(attempt_manifest_path)
        cell_manifest_path = root / "cell_manifest.json"
        if cell_manifest_path.exists():
            raise ExperimentError(f"Refusing to overwrite completed cell: {cell_manifest_path}")
        atomic_json(cell_manifest_path, manifest)
        manifest_hash = sha256_file(cell_manifest_path)
        validate_cell(
            out_dir,
            cell,
            design,
            expected_manifest_hash=manifest_hash,
        )
        append_event(
            out_dir,
            cell,
            "completed",
            attempt=attempt,
            manifest_sha256=manifest_hash,
            details={
                "source_commit": design["source"]["commit"],
                "input_hashes": design["inputs"],
                "output_hashes": {
                    record["path"]: record["sha256"] for record in manifest["artifacts"]
                },
                "started_at_utc": manifest["started_at_utc"],
                "finished_at_utc": manifest["finished_at_utc"],
                "runtime_seconds": manifest["runtime_seconds"],
            },
        )
    except Exception as exc:
        failure = {
            **context,
            "status": "failed",
            "started_at_utc": started_at,
            "finished_at_utc": utc_now(),
            "runtime_seconds": time.perf_counter() - started,
            "failure_type": type(exc).__name__,
            "failure_message": str(exc),
        }
        atomic_json(attempt_dir / "failure.json", failure)
        append_event(
            out_dir,
            cell,
            "failed",
            attempt=attempt,
            message=f"{type(exc).__name__}: {exc}",
            details={
                "source_commit": design["source"]["commit"],
                "input_hashes": design["inputs"],
                "started_at_utc": failure["started_at_utc"],
                "finished_at_utc": failure["finished_at_utc"],
                "runtime_seconds": failure["runtime_seconds"],
            },
        )
        raise


def rebuild_summaries(
    out_dir: Path,
    cells: list[Cell],
    design: dict[str, Any],
) -> dict[str, Any]:
    events = load_events(out_dir)
    latest = latest_events(events)
    state_rows = []
    metric_rows = []
    matrix_rows = []
    completed = 0
    for cell in cells:
        event = latest.get(cell.run_id)
        status = str(event["status"]) if event else "pending"
        state_rows.append(
            {
                "run_id": cell.run_id,
                "variant": cell.variant,
                "fold": cell.fold,
                "seed": cell.seed,
                "status": status,
                "attempt": int(event["attempt"]) if event else 0,
            }
        )
        matrix_rows.append(
            {
                "variant": cell.variant,
                "fold": cell.fold,
                "seed": cell.seed,
                "run_id": cell.run_id,
                "status": status,
            }
        )
        if status == "completed":
            manifest = validate_cell(
                out_dir,
                cell,
                design,
                expected_manifest_hash=event.get("manifest_sha256"),
            )
            attempt = int(manifest["attempt"])
            path = cell_root(out_dir, cell) / "attempts" / f"attempt_{attempt:03d}" / "metrics.json"
            metric_rows.append(json.loads(path.read_text(encoding="utf-8")))
            completed += 1
    expected = len(cells)
    counts = pd.Series([row["status"] for row in state_rows]).value_counts().to_dict()
    next_cell = next(
        (row["run_id"] for row in state_rows if row["status"] != "completed"),
        None,
    )
    summary = {
        "schema_version": SCHEMA_VERSION,
        "campaign_mode": design["campaign_mode"],
        "evidence_eligible": design["evidence_eligible"],
        "campaign_hash": design["campaign_hash"],
        "source_commit": design["source"]["commit"],
        "expected_run_count": expected,
        "completed_run_count": completed,
        "status_counts": counts,
        "next_cell": next_cell,
        "status": "complete" if completed == expected else "incomplete",
        "work_dir": repository_relative(out_dir, repository_root()),
    }
    atomic_json(out_dir / "campaign_state.json", state_rows)
    atomic_csv(out_dir / "completeness_matrix.csv", matrix_rows)
    if metric_rows:
        ordered_metrics = sorted(
            metric_rows,
            key=lambda row: (row["variant"], int(row["fold"]), int(row["seed"])),
        )
        atomic_csv(out_dir / "run_metrics.csv", ordered_metrics)
    atomic_json(out_dir / "campaign_summary.json", summary)
    return summary


def audit_existing_state(
    out_dir: Path,
    cells: list[Cell],
    design: dict[str, Any],
) -> None:
    events = load_events(out_dir)
    latest = latest_events(events)
    for cell in cells:
        event = latest.get(cell.run_id)
        if not event or event["status"] == "pending":
            continue
        if event["status"] == "completed":
            try:
                validate_cell(
                    out_dir,
                    cell,
                    design,
                    expected_manifest_hash=event.get("manifest_sha256"),
                )
            except ExperimentError as exc:
                append_event(
                    out_dir,
                    cell,
                    "invalid",
                    attempt=int(event["attempt"]),
                    message=str(exc),
                    details={
                        "source_commit": design["source"]["commit"],
                        "input_hashes": design.get("inputs"),
                    },
                )
                raise
            continue
        if event["status"] == "running":
            append_event(
                out_dir,
                cell,
                "invalid",
                attempt=int(event["attempt"]),
                message="Interrupted running attempt requires manual inspection.",
                details={
                    "source_commit": design["source"]["commit"],
                    "input_hashes": design.get("inputs"),
                },
            )
            raise ExperimentError(
                f"Interrupted running attempt detected for {cell.run_id}; fail-closed."
            )
        if event["status"] == "invalid":
            raise ExperimentError(
                f"Invalid existing cell requires manual inspection: {cell.run_id}"
            )


def run_campaign(args: argparse.Namespace) -> dict[str, Any]:
    root = repository_root()
    base_features_path = args.base_features.resolve()
    manifests_dir = args.manifests_dir.resolve()
    out_dir = args.out_dir.resolve()
    repository_relative(out_dir, root)
    source = source_state(root)
    if not args.smoke and not source["tracked_worktree_clean"]:
        raise ExperimentError("Full evidence execution requires a clean tracked worktree.")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise ExperimentError("CUDA requested but unavailable.")
    observed_schedule = {
        "epochs": args.epochs,
        "region_batch_size": args.region_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
    }
    if observed_schedule != FROZEN_SCHEDULE:
        raise ExperimentError(
            "Capacity-matched ablation schedule is frozen: "
            f"expected={FROZEN_SCHEDULE}, observed={observed_schedule}"
        )
    if args.max_new_runs <= 0:
        raise ExperimentError("--max-new-runs must be positive.")

    _, _, inventory, fold_data = validate_inputs(
        base_features_path,
        manifests_dir,
        root,
    )
    parameters = parameter_inventory(FROZEN_FEATURE["feature_dim"])
    design = build_design(
        root=root,
        base_features_path=base_features_path,
        manifests_dir=manifests_dir,
        smoke=args.smoke,
        device=args.device,
        source=source,
        inventory=inventory,
        parameters=parameters,
    )
    cells = cells_for_design(design)
    if not args.smoke and len(cells) != 175:
        raise ExperimentError(f"Expected a 175-cell full grid, observed {len(cells)}.")
    if args.smoke and len(cells) != 7:
        raise ExperimentError(f"Expected a seven-cell smoke grid, observed {len(cells)}.")

    if out_dir.exists() and not (out_dir / "campaign_design.json").is_file():
        if any(out_dir.iterdir()):
            raise ExperimentError(
                f"Output directory is nonempty without a campaign design: {out_dir}"
            )
    out_dir.mkdir(parents=True, exist_ok=True)
    ensure_immutable_json(out_dir / "campaign_design.json", design)
    ensure_immutable_json(out_dir / "input_inventory.json", inventory)
    ensure_immutable_json(out_dir / "environment.json", capture_environment())
    initialize_ledger(out_dir, cells, design)
    audit_existing_state(out_dir, cells, design)
    summary = rebuild_summaries(out_dir, cells, design)
    if args.validate_only:
        return summary

    latest = latest_events(load_events(out_dir))
    new_runs = 0
    parameter_lookup = {row["variant"]: row for row in parameters}
    device = torch.device(args.device)
    for cell in cells:
        event = latest.get(cell.run_id)
        if event and event["status"] == "completed":
            continue
        if new_runs >= args.max_new_runs:
            break
        execute_cell(
            out_dir=out_dir,
            cell=cell,
            design=design,
            fold_data=fold_data[cell.fold],
            device=device,
            parameter_row=parameter_lookup[cell.variant],
        )
        new_runs += 1
        summary = rebuild_summaries(out_dir, cells, design)
        latest = latest_events(load_events(out_dir))
    return rebuild_summaries(out_dir, cells, design)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-features", type=Path, required=True)
    parser.add_argument("--manifests-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--region-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run one epoch for fold 0/seed 801 across all variants; never evidence eligible",
    )
    parser.add_argument(
        "--max-new-runs",
        type=int,
        default=175,
        help="maximum new cells attempted in this invocation",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="validate inputs, ledger, and completed artifacts without training",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = run_campaign(args)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (ExperimentError, OSError, RuntimeError, ValueError) as exc:
        print(f"SCORPION CAPACITY-MATCHED ABLATIONS FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
