#!/usr/bin/env python3
"""Forward-valid biological-bottleneck allocation test on frozen pathology features.

The runner has a deliberately hard readiness boundary.  It audits concrete
SCORPION and canine-SCC DINOv2 archives and their five frozen manifests before
constructing any neural module.  Pixel data are never opened.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import os
import subprocess
import tempfile
import time
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.special import softmax
from sklearn.decomposition import PCA
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler

from experiments.paired_acquisition import (
    run_biological_bottleneck_capacity_allocation_factorial as synthetic_factorial,
)
from experiments.paired_acquisition import (
    run_crossed_target_scanner_prototype_factorization as prototype,
)
from experiments.paired_acquisition import (
    run_synthetic_crossed_factor_identifiability as canonical,
)


SCHEMA_VERSION = "real-paired-scanner-bottleneck-allocation-validation/v1"
READINESS_SCHEMA_VERSION = "real-paired-scanner-bottleneck-allocation-readiness/v1"
SYNTHETIC_FILE_SHA256 = "674bd8e891747ddf5cd6f215e2ce3115b633ec298c5b2e8e849f20adedf4f056"
SYNTHETIC_INTERNAL_SHA256 = "7fc2e764331ac384f78699255487a01be9d2f94100ef2482a6a42322584c2c37"
SYNTHETIC_MANIFEST_INTERNAL_SHA256 = "4f866c370052d58107b0e2b9b84bde11b7ce6fadb1248d1f46580b6cf1926a26"
SYNTHETIC_STATUS = "complete_capacity_gain_with_scanner_tradeoff"

FOLDS = tuple(range(5))
MODEL_SEEDS = (2201, 2202, 2203, 2204, 2205)
PROBE_SEEDS = (8401, 8402, 8403)
PCA_COMPONENTS = (4, 8, 16, 24, 32, 48, 64)
FAMILIES = ("real_b32_reference", "real_b64_parameter_matched")
BROKEN_PAIR_FOLDS = (0,)
BROKEN_PAIR_SEEDS = (2201,)
BOOTSTRAP_SEED = 8601
BOOTSTRAP_REPLICATES = 10_000
MATERIAL_MARGIN = 0.02
EPOCHS = 75
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
ACQUISITION_DIM = 8

DATASET_SPECS: dict[str, dict[str, Any]] = {
    "scorpion": {
        "dataset_identifier": "SCORPION",
        "feature_path": Path("results/scorpion/features/fold_0_dinov2_base.npz"),
        "manifest_dir": Path("data/scorpion/splits"),
        "manifest_pattern": "fold_{}_manifest.csv",
        "legacy_root": Path("results/scorpion/pathoalign_dinov2_crossfold"),
        "category_column": None,
        "specimen_column": "slide_id",
    },
    "canine_scc": {
        "dataset_identifier": "external_multiscanner_caninescc",
        "feature_path": Path(
            "results/external_multiscanner_caninescc/features/fold_0_dinov2_base.npz"
        ),
        "manifest_dir": Path(
            "data/external_multiscanner_caninescc/patch_manifests/splits"
        ),
        "manifest_pattern": "fold_{}_patch_manifest.csv",
        "legacy_root": Path(
            "results/external_multiscanner_caninescc/pathoalign_dinov2_crossfold"
        ),
        "category_column": "category_name",
        "specimen_column": "slide_id",
    },
}

CLAIM_SCOPE = {
    "synthetic_status_unchanged": SYNTHETIC_STATUS,
    "new_forward_valid_real_feature_test": True,
    "factorizer_training_uses_biological_or_category_labels": False,
    "frozen_features_are_raw_histology": False,
    "pixel_space_reconstruction_claimed": False,
    "category_preservation_is_clinical_validation": False,
    "universal_scanner_generalization_claimed": False,
    "scanner_site_stain_cohort_endpoint_generalization_are_distinct": True,
    "clinical_deployment_or_patient_care_claimed": False,
    "pixel_space_prohibited_without_registration_and_qc": True,
}


class ExperimentError(RuntimeError):
    """A structural or execution failure, distinct from a poor scientific result."""


@dataclass(frozen=True)
class LoadedDataset:
    name: str
    features: np.ndarray
    archive_frame: pd.DataFrame
    manifests: tuple[pd.DataFrame, ...]
    scanner_names: tuple[str, ...]
    category_column: str | None
    specimen_column: str


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(value: Mapping[str, Any]) -> str:
    return canonical.sha256_bytes(canonical.canonical_json_bytes(dict(value)))


def atomic_json(path: Path, value: Mapping[str, Any]) -> None:
    canonical.atomic_json(path, dict(value))


def heterogeneous_fieldnames(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    fields: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            key = str(key)
            if key not in seen:
                fields.append(key)
                seen.add(key)
    return fields


def atomic_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    os.close(fd)
    temporary = Path(temporary_name)
    try:
        with temporary.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=heterogeneous_fieldnames(rows))
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


def verify_synthetic_factorial(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ExperimentError(f"Frozen synthetic factorial missing: {path}")
    file_hash = sha256_file(path)
    if file_hash != SYNTHETIC_FILE_SHA256:
        raise ExperimentError(f"Frozen synthetic factorial file hash mismatch: {file_hash}")
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("result_sha256") != SYNTHETIC_INTERNAL_SHA256:
        raise ExperimentError("Frozen synthetic factorial internal hash mismatch.")
    if result.get("status") != SYNTHETIC_STATUS:
        raise ExperimentError("Frozen synthetic factorial status mismatch.")
    canonical_copy = copy.deepcopy(result)
    embedded = canonical_copy.pop("result_sha256")
    if canonical_hash(canonical_copy) != embedded:
        raise ExperimentError("Frozen synthetic factorial canonical hash is invalid.")
    frozen: dict[str, Any] = {}
    for name, item in result.get("frozen_artifacts", {}).items():
        source = Path(item["path"])
        observed = sha256_file(source) if source.is_file() else None
        if observed != item.get("file_sha256_before") or observed != item.get(
            "file_sha256_after"
        ):
            raise ExperimentError(f"Inherited frozen artifact failed: {name}")
        frozen[name] = {**item, "verified_file_sha256": observed}
    manifest_path = path.with_name(
        "biological_bottleneck_capacity_allocation_factorial_manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_copy = copy.deepcopy(manifest)
    manifest_hash = manifest_copy.pop("manifest_sha256")
    if manifest_hash != SYNTHETIC_MANIFEST_INTERNAL_SHA256:
        raise ExperimentError("Frozen synthetic manifest internal hash mismatch.")
    if canonical_hash(manifest_copy) != manifest_hash:
        raise ExperimentError("Frozen synthetic manifest canonical hash is invalid.")
    return {
        "path": str(path.resolve()),
        "file_sha256": file_hash,
        "internal_sha256": embedded,
        "manifest_path": str(manifest_path.resolve()),
        "manifest_file_sha256": sha256_file(manifest_path),
        "manifest_internal_sha256": manifest_hash,
        "status": result["status"],
        "inherited_artifacts": frozen,
    }


def archive_frame(arrays: Mapping[str, np.ndarray]) -> pd.DataFrame:
    required = ("slide_id", "region_id", "scanner_id", "path", "split", "fold")
    missing = [key for key in required if key not in arrays]
    if missing:
        raise ExperimentError(f"Feature archive metadata missing: {missing}")
    return pd.DataFrame({key: arrays[key].astype(str) for key in required})


def row_keys(frame: pd.DataFrame) -> list[tuple[str, str, str, str]]:
    return list(
        zip(
            frame["slide_id"].astype(str),
            frame["region_id"].astype(str),
            frame["scanner_id"].astype(str),
            frame["path"].astype(str),
        )
    )


def hash_string_array(values: Iterable[Any]) -> str:
    normalized = [
        "\x1f".join(map(str, value))
        if isinstance(value, (tuple, list, np.ndarray))
        else str(value)
        for value in values
    ]
    return hashlib.sha256("\n".join(normalized).encode("utf-8")).hexdigest()


def fold_integrity(frame: pd.DataFrame, specimen_column: str) -> dict[str, Any]:
    required_splits = {"train", "val", "test"}
    present = set(frame["split"].astype(str))
    region_crosses = int((frame.groupby("region_id")["split"].nunique() > 1).sum())
    specimen_crosses = int(
        (frame.groupby(specimen_column)["split"].nunique() > 1).sum()
    )
    return {
        "required_splits_present": required_splits.issubset(present),
        "paired_regions_crossing_splits": region_crosses,
        "specimens_crossing_splits": specimen_crosses,
        "passed": bool(
            required_splits.issubset(present)
            and region_crosses == 0
            and specimen_crosses == 0
        ),
    }


def discover_legacy_artifacts(root: Path) -> dict[str, Any]:
    branch_archives = sorted(root.glob("fold_*/runs/*/projected_features.npz")) if root.is_dir() else []
    decoder_weights = sorted(root.glob("fold_*/runs/*/checkpoint.pt")) if root.is_dir() else []
    swap_metadata = (
        sorted(root.rglob("*swap*"))
        + sorted(root.rglob("*pair_assignment*"))
        + sorted(root.rglob("*source_target*"))
        if root.is_dir()
        else []
    )
    swap_metadata = [path for path in swap_metadata if path.is_file()]
    return {
        "root": str(root.resolve()),
        "branch_embedding_archives": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in branch_archives
        ],
        "decoder_or_composition_weights": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in decoder_weights
        ],
        "swap_or_pair_assignment_metadata": [
            {"path": str(path.resolve()), "sha256": sha256_file(path)}
            for path in sorted(set(swap_metadata))
        ],
        "source_region_mapping_verified": False,
        "target_scanner_mapping_verified": False,
        "fold_consistent_swap_provenance_verified": False,
    }


def parameter_formula(input_dim: int, biological_dim: int, hidden_dim: int, acquisition_dim: int, scanners: int) -> int:
    return int(
        hidden_dim**2
        + (2 * input_dim + 2 * biological_dim + 2 * acquisition_dim + 9)
        * hidden_dim
        + biological_dim
        + scanners * acquisition_dim
        + input_dim
    )


def select_matched_hidden_width(input_dim: int, scanners: int) -> dict[str, Any]:
    reference = parameter_formula(input_dim, 32, 128, ACQUISITION_DIM, scanners)
    candidates = []
    for hidden in range(16, 513):
        count = parameter_formula(input_dim, 64, hidden, ACQUISITION_DIM, scanners)
        candidates.append((abs(count - reference), hidden, count))
    difference, hidden, candidate = min(candidates)
    relative = difference / reference
    if relative >= 0.005:
        raise ExperimentError(
            f"No B64 integer hidden width matches parameter budget below 0.5%: {relative}"
        )
    return {
        "formula": "H^2 + (2D + 2B + 2A + 9)H + B + SA + D",
        "input_dimension": input_dim,
        "scanner_count": scanners,
        "acquisition_dimension": ACQUISITION_DIM,
        "real_b32_reference": {
            "biological_dimension": 32,
            "hidden_width": 128,
            "formula_parameter_count": reference,
        },
        "real_b64_parameter_matched": {
            "biological_dimension": 64,
            "hidden_width": hidden,
            "formula_parameter_count": candidate,
        },
        "absolute_difference": difference,
        "relative_difference": relative,
        "selected_before_training": True,
    }


def audit_dataset(repository_root: Path, name: str, spec: Mapping[str, Any]) -> dict[str, Any]:
    feature_path = repository_root / spec["feature_path"]
    manifest_paths = tuple(
        repository_root / spec["manifest_dir"] / spec["manifest_pattern"].format(fold)
        for fold in FOLDS
    )
    discovered = {
        "dataset_name": name,
        "dataset_identifier": spec["dataset_identifier"],
        "feature_path": str(feature_path.resolve()),
        "manifest_paths": [str(path.resolve()) for path in manifest_paths],
    }
    if not feature_path.is_file():
        return {**discovered, "readiness_state": "feature_artifact_missing", "feature_space_ready": False}
    if not all(path.is_file() for path in manifest_paths):
        return {**discovered, "readiness_state": "paired_metadata_incomplete", "feature_space_ready": False}
    try:
        with np.load(feature_path, allow_pickle=False) as archive:
            if "features" not in archive:
                raise ExperimentError("Feature array key is absent.")
            features = np.asarray(archive["features"])
            frame = archive_frame(archive)
            metadata = json.loads(str(archive["metadata_json"].item())) if "metadata_json" in archive else {}
        if features.ndim != 2 or len(features) != len(frame):
            raise ExperimentError("Feature shape and row metadata do not align.")
        if not np.isfinite(features).all():
            raise ExperimentError("Feature array contains non-finite values.")
        manifests = [pd.read_csv(path) for path in manifest_paths]
        archive_keys = row_keys(frame)
        alignment = []
        fold_checks = []
        for fold, manifest in enumerate(manifests):
            aligned = row_keys(manifest) == archive_keys
            alignment.append(aligned)
            check = fold_integrity(manifest, str(spec["specimen_column"]))
            fold_checks.append({"fold": fold, **check})
        if not all(alignment):
            raise ExperimentError("Manifest row order does not match the feature archive.")
        scanners_per_region = frame.groupby("region_id")["scanner_id"].nunique()
        stable_pairs = bool((scanners_per_region >= 2).all())
        category_column = spec["category_column"]
        category_available = bool(
            category_column
            and all(category_column in manifest.columns for manifest in manifests)
            and all(manifest[category_column].notna().all() for manifest in manifests)
        )
        categories = (
            sorted(manifests[0][category_column].astype(str).unique())
            if category_available
            else []
        )
        legacy = discover_legacy_artifacts(repository_root / spec["legacy_root"])
        layer2 = bool(
            legacy["branch_embedding_archives"]
            and legacy["decoder_or_composition_weights"]
            and legacy["swap_or_pair_assignment_metadata"]
            and legacy["source_region_mapping_verified"]
            and legacy["target_scanner_mapping_verified"]
            and legacy["fold_consistent_swap_provenance_verified"]
        )
        fold_pass = all(item["passed"] for item in fold_checks)
        feature_ready = bool(stable_pairs and fold_pass)
        state = (
            "feature_and_decoder_ready"
            if feature_ready and layer2
            else "feature_only_ready"
            if feature_ready
            else "fold_integrity_failed"
            if not fold_pass
            else "dataset_not_ready"
        )
        scanner_names = sorted(frame["scanner_id"].astype(str).unique())
        split_counts = []
        fold_identity_hashes = []
        for fold, manifest in enumerate(manifests):
            for split, group in manifest.groupby("split"):
                split_counts.append(
                    {
                        "fold": fold,
                        "split": str(split),
                        "rows": int(len(group)),
                        "regions": int(group["region_id"].nunique()),
                        "slides_or_specimens": int(group[str(spec["specimen_column"])].nunique()),
                    }
                )
                fold_identity_hashes.append(
                    {
                        "fold": fold,
                        "split": str(split),
                        "row_index_sha256": hash_string_array(group.index),
                        "region_id_sha256": hash_string_array(group["region_id"]),
                        "slide_or_specimen_id_sha256": hash_string_array(
                            group[str(spec["specimen_column"])]
                        ),
                    }
                )
        parameter_match = select_matched_hidden_width(
            int(features.shape[1]), len(scanner_names)
        )
        return {
            **discovered,
            "readiness_state": state,
            "feature_space_ready": feature_ready,
            "layer1_ready": feature_ready,
            "layer2_ready": layer2,
            "pixel_space_ready": False,
            "pixel_space_prohibited": True,
            "feature_backbone": metadata.get("model"),
            "feature_model_revision": metadata.get("model_revision"),
            "feature_sha256": sha256_file(feature_path),
            "feature_dtype": str(features.dtype),
            "feature_shape": list(features.shape),
            "feature_dimension": int(features.shape[1]),
            "all_numerical_arrays_finite": True,
            "scanner_identifiers": scanner_names,
            "scanner_count": len(scanner_names),
            "paired_region_count": int(frame["region_id"].nunique()),
            "regions_per_scanner": {
                str(scanner): int(group["region_id"].nunique())
                for scanner, group in frame.groupby("scanner_id")
            },
            "slide_or_specimen_count": int(frame[str(spec["specimen_column"])].nunique()),
            "regions_have_at_least_two_scanners": stable_pairs,
            "category_column": category_column,
            "category_labels_available": category_available,
            "categories": categories,
            "category_count": len(categories),
            "row_alignment_by_fold": alignment,
            "fold_integrity": fold_checks,
            "split_counts": split_counts,
            "fold_identity_hashes": fold_identity_hashes,
            "excluded_rows": [],
            "row_order_manifest_sha256": hash_string_array(archive_keys),
            "pair_group_manifest_sha256": hash_string_array(frame["region_id"]),
            "scanner_manifest_sha256": hash_string_array(frame["scanner_id"]),
            "slide_or_specimen_manifest_sha256": hash_string_array(frame[str(spec["specimen_column"])]),
            "manifest_files": [
                {"path": str(path.resolve()), "sha256": sha256_file(path)}
                for path in manifest_paths
            ],
            "legacy_layer2_discovery": legacy,
            "layer2_unavailable_reasons": []
            if layer2
            else [
                "no verified swap/pair-assignment metadata with source-region and target-scanner provenance"
            ],
            "parameter_match": parameter_match,
        }
    except ExperimentError as exc:
        return {
            **discovered,
            "readiness_state": "dataset_not_ready",
            "feature_space_ready": False,
            "failure_reason": str(exc),
        }


def audit_readiness(repository_root: Path) -> dict[str, Any]:
    datasets = {
        name: audit_dataset(repository_root, name, spec)
        for name, spec in DATASET_SPECS.items()
    }
    ready = [name for name, item in datasets.items() if item.get("feature_space_ready")]
    backbones = {
        str(item.get("feature_backbone", "")).lower()
        for item in datasets.values()
        if item.get("feature_space_ready")
    }
    common = len(backbones) == 1 and bool(backbones) and "dinov2" in next(iter(backbones))
    if len(ready) > 1 and not common:
        for name in ready:
            datasets[name]["feature_space_ready"] = False
            datasets[name]["readiness_state"] = "dataset_not_ready"
            datasets[name]["failure_reason"] = "No exact common frozen feature backbone."
        ready = []
    readiness = {
        "schema_version": READINESS_SCHEMA_VERSION,
        "claim_scope": CLAIM_SCOPE,
        "repository_root": str(repository_root.resolve()),
        "audit_precedes_model_initialization": True,
        "model_initializations_during_readiness": 0,
        "datasets": datasets,
        "ready_datasets": ready,
        "common_feature_backbone_verified": common,
        "common_feature_backbone": next(iter(backbones)) if common else None,
        "readiness_allows_training": bool(ready),
    }
    readiness["readiness_sha256"] = canonical_hash(readiness)
    return readiness


def load_dataset(repository_root: Path, name: str) -> LoadedDataset:
    spec = DATASET_SPECS[name]
    feature_path = repository_root / spec["feature_path"]
    with np.load(feature_path, allow_pickle=False) as archive:
        features = np.asarray(archive["features"], dtype=np.float32)
        frame = archive_frame(archive)
    manifests = tuple(
        pd.read_csv(
            repository_root
            / spec["manifest_dir"]
            / spec["manifest_pattern"].format(fold)
        )
        for fold in FOLDS
    )
    category = spec["category_column"]
    if category:
        for index, manifest in enumerate(manifests):
            frame_category = manifest[category].astype(str).to_numpy()
            if index == 0:
                frame[category] = frame_category
            elif not np.array_equal(frame[category].to_numpy(), frame_category):
                raise ExperimentError(f"Category row alignment differs in {name}, fold {index}.")
    return LoadedDataset(
        name=name,
        features=features,
        archive_frame=frame,
        manifests=manifests,
        scanner_names=tuple(sorted(frame["scanner_id"].unique())),
        category_column=category,
        specimen_column=str(spec["specimen_column"]),
    )


def split_indices(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return tuple(
        np.flatnonzero(frame["split"].astype(str).to_numpy() == split).astype(np.int64)
        for split in ("train", "val", "test")
    )  # type: ignore[return-value]


def deterministic_derangement(length: int, seed: int) -> np.ndarray:
    if length < 2:
        raise ExperimentError("Broken-pair derangement requires at least two regions.")
    rng = np.random.default_rng(seed)
    for _ in range(10_000):
        candidate = rng.permutation(length)
        if np.all(candidate != np.arange(length)):
            return candidate
    return np.roll(np.arange(length), 1)


def build_pair_indices(
    frame: pd.DataFrame,
    indices: np.ndarray,
    *,
    broken: bool,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    subset = frame.iloc[indices].copy()
    # Explicit row ids keep the pairing provenance exact.
    subset["_row_index"] = indices
    table = subset.pivot(index="region_id", columns="scanner_id", values="_row_index")
    table = table.dropna(axis=0)
    scanners = sorted(table.columns.astype(str))
    regions = table.index.astype(str).tolist()
    sources: list[int] = []
    targets: list[int] = []
    assignments: list[tuple[str, str]] = []
    for offset, source_scanner in enumerate(scanners):
        target_scanner = scanners[(offset + 1) % len(scanners)]
        target_order = np.arange(len(regions))
        if broken:
            target_order = deterministic_derangement(
                len(regions), seed + 101 * (offset + 1)
            )
        for region_index, target_region_index in enumerate(target_order):
            sources.append(int(table.iloc[region_index][source_scanner]))
            targets.append(int(table.iloc[int(target_region_index)][target_scanner]))
            assignments.append((regions[region_index], regions[int(target_region_index)]))
    source = np.asarray(sources, dtype=np.int64)
    target = np.asarray(targets, dtype=np.int64)
    audit = {
        "broken_pairs": broken,
        "source_count": len(source),
        "scanner_counts_preserved": sorted(frame.iloc[source]["scanner_id"].value_counts().tolist())
        == sorted(frame.iloc[target]["scanner_id"].value_counts().tolist()),
        "indices_within_requested_split": bool(
            set(source.tolist()).issubset(set(indices.tolist()))
            and set(target.tolist()).issubset(set(indices.tolist()))
        ),
        "same_region_assignment_count": sum(left == right for left, right in assignments),
        "assignment_sha256": hash_string_array(f"{left}->{right}" for left, right in assignments),
    }
    if broken and audit["same_region_assignment_count"]:
        raise ExperimentError("Broken-pair control retained a same-region assignment.")
    return source, target, audit


def family_dimensions(parameter_match: Mapping[str, Any], family: str) -> tuple[int, int]:
    item = parameter_match[family]
    return int(item["biological_dimension"]), int(item["hidden_width"])


def build_factorizer(
    input_dim: int,
    scanners: int,
    biological_dim: int,
    hidden_dim: int,
    device: torch.device,
) -> prototype.ScannerPrototypeFactorizer:
    return prototype.ScannerPrototypeFactorizer(
        input_dim=input_dim,
        biological_dim=biological_dim,
        acquisition_dim=ACQUISITION_DIM,
        hidden_dim=hidden_dim,
        scanners=scanners,
    ).to(device)


def _objective(
    model: prototype.ScannerPrototypeFactorizer,
    observations: torch.Tensor,
    scanner_ids: torch.Tensor,
    indices: torch.Tensor,
    source: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    selected = observations.index_select(0, indices)
    selected_scanners = scanner_ids.index_select(0, indices)
    output = model(selected, selected_scanners)
    self_reconstruction = F.mse_loss(output["reconstruction"], selected)
    biological_source = model.encode_biological(observations.index_select(0, source))
    biological_target = model.encode_biological(observations.index_select(0, target))
    crossed = model.decode(
        biological_source,
        model.acquisition_from_scanner(scanner_ids.index_select(0, target)),
    )
    crossed_reconstruction = F.mse_loss(crossed, observations.index_select(0, target))
    consistency = F.mse_loss(biological_source, biological_target)
    variance = prototype.biological_variance_floor(output["biological"])
    center, separation = prototype.prototype_regularization(model.scanner_prototypes.weight)
    loss = (
        self_reconstruction
        + crossed_reconstruction
        + consistency
        + 0.05 * variance
        + 0.01 * center
        + 0.01 * separation
    )
    metrics = {
        "total": float(loss.detach().cpu()),
        "self_feature_reconstruction": float(self_reconstruction.detach().cpu()),
        "crossed_target_feature_reconstruction": float(crossed_reconstruction.detach().cpu()),
        "same_region_biological_consistency": float(consistency.detach().cpu()),
        "biological_variance_floor": float(variance.detach().cpu()),
        "prototype_centering": float(center.detach().cpu()),
        "prototype_separation": float(separation.detach().cpu()),
    }
    return loss, metrics


def train_factorizer(
    dataset: LoadedDataset,
    fold: int,
    family: str,
    seed: int,
    parameter_match: Mapping[str, Any],
    device: torch.device,
    *,
    broken_pairs: bool,
) -> tuple[prototype.ScannerPrototypeFactorizer, StandardScaler, dict[str, Any]]:
    canonical.set_deterministic_seed(seed)
    frame = dataset.manifests[fold]
    train, validation, _ = split_indices(frame)
    scaler = StandardScaler().fit(dataset.features[train])
    standardized = scaler.transform(dataset.features).astype(np.float32)
    scanner_encoder = {name: index for index, name in enumerate(dataset.scanner_names)}
    scanner_ids_np = frame["scanner_id"].astype(str).map(scanner_encoder).to_numpy(np.int64, copy=True)
    train_source, train_target, pair_audit = build_pair_indices(
        frame, train, broken=broken_pairs, seed=seed + fold * 1000
    )
    val_source, val_target, _ = build_pair_indices(
        frame, validation, broken=False, seed=seed + fold * 1000 + 1
    )
    biological_dim, hidden_dim = family_dimensions(parameter_match, family)
    model = build_factorizer(
        dataset.features.shape[1], len(dataset.scanner_names), biological_dim, hidden_dim, device
    )
    observations = torch.as_tensor(standardized, device=device)
    scanner_ids = torch.as_tensor(scanner_ids_np, dtype=torch.long, device=device)
    train_tensor = torch.as_tensor(train, dtype=torch.long, device=device)
    val_tensor = torch.as_tensor(validation, dtype=torch.long, device=device)
    train_source_tensor = torch.as_tensor(train_source, dtype=torch.long, device=device)
    train_target_tensor = torch.as_tensor(train_target, dtype=torch.long, device=device)
    val_source_tensor = torch.as_tensor(val_source, dtype=torch.long, device=device)
    val_target_tensor = torch.as_tensor(val_target, dtype=torch.long, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    history: list[dict[str, Any]] = []
    best_loss = math.inf
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    for epoch in range(1, EPOCHS + 1):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss, train_metrics = _objective(
            model,
            observations,
            scanner_ids,
            train_tensor,
            train_source_tensor,
            train_target_tensor,
        )
        if not torch.isfinite(loss):
            raise ExperimentError(f"Non-finite training loss: {dataset.name}/{fold}/{family}/{seed}")
        loss.backward()
        if any(
            parameter.grad is not None and not torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ):
            raise ExperimentError("Non-finite factorizer gradient.")
        optimizer.step()
        model.eval()
        with torch.no_grad():
            val_loss, val_metrics = _objective(
                model,
                observations,
                scanner_ids,
                val_tensor,
                val_source_tensor,
                val_target_tensor,
            )
        current = float(val_loss.cpu())
        if current < best_loss:
            best_loss = current
            best_epoch = epoch
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        if epoch == 1 or epoch == EPOCHS or epoch % 5 == 0:
            history.append({"epoch": epoch, "train": train_metrics, "validation": val_metrics})
    if best_state is None:
        raise ExperimentError("No finite validation checkpoint selected.")
    model.load_state_dict(best_state)
    actual_count = sum(parameter.numel() for parameter in model.parameters())
    expected_count = int(parameter_match[family]["formula_parameter_count"])
    if actual_count != expected_count:
        raise ExperimentError(
            f"PyTorch/formula parameter mismatch for {dataset.name}/{family}: {actual_count} != {expected_count}"
        )
    return model, scaler, {
        "epochs": EPOCHS,
        "optimizer": "AdamW",
        "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY,
        "checkpoint_selected_by": "minimum validation objective only",
        "best_epoch": best_epoch,
        "best_validation_loss": best_loss,
        "test_used_for_checkpoint_selection": False,
        "feature_scaler_fit_split": "train",
        "factorizer_reads_category_or_biological_labels": False,
        "paired_region_and_scanner_metadata_only": True,
        "broken_pair_control": broken_pairs,
        "pair_audit": pair_audit,
        "actual_parameter_count": actual_count,
        "formula_parameter_count": expected_count,
        "history": history,
    }


def project_factorizer(
    model: prototype.ScannerPrototypeFactorizer,
    scaler: StandardScaler,
    dataset: LoadedDataset,
    fold: int,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    frame = dataset.manifests[fold]
    scanner_encoder = {name: index for index, name in enumerate(dataset.scanner_names)}
    scanner_ids = frame["scanner_id"].astype(str).map(scanner_encoder).to_numpy(np.int64, copy=True)
    inputs = scaler.transform(dataset.features).astype(np.float32)
    biological: list[np.ndarray] = []
    acquisition: list[np.ndarray] = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(inputs), 512):
            batch = torch.as_tensor(inputs[start : start + 512], device=device)
            scanners = torch.as_tensor(scanner_ids[start : start + 512], dtype=torch.long, device=device)
            biological.append(model.encode_biological(batch).cpu().numpy())
            acquisition.append(model.acquisition_from_scanner(scanners).cpu().numpy())
    return np.concatenate(biological), np.concatenate(acquisition)


def paired_permutation_labels(frame: pd.DataFrame, indices: np.ndarray, seed: int) -> np.ndarray:
    labels = frame.iloc[indices]["scanner_id"].astype(str).to_numpy().copy()
    rng = np.random.default_rng(seed)
    region_values = frame.iloc[indices]["region_id"].astype(str).to_numpy()
    for region in sorted(set(region_values)):
        positions = np.flatnonzero(region_values == region)
        labels[positions] = labels[positions][rng.permutation(len(positions))]
    return labels


def _probabilities(model: Any, features: np.ndarray, class_count: int) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return np.asarray(model.predict_proba(features))
    decision = np.asarray(model.decision_function(features))
    if decision.ndim == 1:
        decision = np.column_stack([-decision, decision])
    probabilities = softmax(decision, axis=1)
    if probabilities.shape[1] != class_count:
        raise ExperimentError("Probe probability shape mismatch.")
    return probabilities


def classification_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    probabilities: np.ndarray,
    classes: Sequence[str],
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "balanced_accuracy": float(balanced_accuracy_score(truth, prediction)),
        "macro_f1": float(f1_score(truth, prediction, average="macro", zero_division=0)),
        "per_class_recall": {},
    }
    for index, label in enumerate(classes):
        mask = truth == index
        result["per_class_recall"][str(label)] = float(np.mean(prediction[mask] == index)) if mask.any() else None
    try:
        result["auroc"] = float(
            roc_auc_score(truth, probabilities, multi_class="ovr", average="macro")
        )
    except ValueError:
        result["auroc"] = None
    return result


def probe(
    features: np.ndarray,
    labels: Sequence[Any],
    train: np.ndarray,
    test: np.ndarray,
    *,
    nonlinear: bool,
    seeds: Sequence[int],
) -> dict[str, Any]:
    encoder = LabelEncoder().fit(np.asarray(labels, dtype=str)[train])
    all_labels = encoder.transform(np.asarray(labels, dtype=str))
    rows = []
    for seed in seeds:
        scaler = StandardScaler().fit(features[train])
        x_train = scaler.transform(features[train])
        x_test = scaler.transform(features[test])
        if nonlinear:
            model = MLPClassifier(
                hidden_layer_sizes=(32,),
                activation="relu",
                alpha=1e-3,
                max_iter=80,
                random_state=seed,
                early_stopping=False,
            )
        else:
            model = LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=1000,
                random_state=seed,
            )
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=ConvergenceWarning)
            model.fit(x_train, all_labels[train])
        prediction = model.predict(x_test)
        probabilities = _probabilities(model, x_test, len(encoder.classes_))
        rows.append(
            {
                "seed": seed,
                **classification_metrics(
                    all_labels[test], prediction, probabilities, encoder.classes_
                ),
                "predictions": prediction,
                "truth": all_labels[test],
            }
        )
    for row in rows:
        row["predictions"] = row["predictions"].tolist()
        row["truth"] = row["truth"].tolist()
    return {
        "configuration": "MLP(32,relu,alpha=1e-3,max_iter=80)"
        if nonlinear
        else "multinomial_logistic(C=1,class_weight=balanced)",
        "seeds": list(seeds),
        "runs": rows,
        "balanced_accuracy_median": float(np.median([row["balanced_accuracy"] for row in rows])),
        "balanced_accuracy_range": [
            float(min(row["balanced_accuracy"] for row in rows)),
            float(max(row["balanced_accuracy"] for row in rows)),
        ],
        "macro_f1_median": float(np.median([row["macro_f1"] for row in rows])),
        "fit_split": "train",
    }


def scanner_probe_battery(
    features: np.ndarray,
    frame: pd.DataFrame,
    train: np.ndarray,
    test: np.ndarray,
) -> dict[str, Any]:
    labels = frame["scanner_id"].astype(str).to_numpy()
    linear = probe(features, labels, train, test, nonlinear=False, seeds=PROBE_SEEDS)
    nonlinear = probe(features, labels, train, test, nonlinear=True, seeds=PROBE_SEEDS)
    primary = linear["runs"][0]
    primary_truth = np.asarray(primary["truth"])
    primary_prediction = np.asarray(primary["predictions"])
    test_slides = frame.iloc[test]["slide_id"].astype(str).to_numpy()
    slide_scores = {}
    for slide in sorted(set(test_slides)):
        mask = test_slides == slide
        slide_scores[slide] = float(
            balanced_accuracy_score(primary_truth[mask], primary_prediction[mask])
        )
    null_rows = []
    for seed in PROBE_SEEDS:
        permuted = labels.copy()
        permuted[train] = paired_permutation_labels(frame, train, seed)
        permuted[test] = paired_permutation_labels(frame, test, seed + 10_000)
        null = probe(features, permuted, train, test, nonlinear=False, seeds=(seed,))
        null_rows.append(
            {"seed": seed, "balanced_accuracy": null["balanced_accuracy_median"]}
        )
    return {
        "chance_level": 1.0 / frame["scanner_id"].nunique(),
        "linear": linear,
        "nonlinear": nonlinear,
        "linear_confusion_matrix": confusion_matrix(
            primary_truth, primary_prediction
        ).tolist(),
        "slide_balanced_accuracy": slide_scores,
        "paired_identity_aware_permutation_null": {
            "preserves_region_blocks": True,
            "runs": null_rows,
            "median": float(np.median([row["balanced_accuracy"] for row in null_rows])),
            "range": [
                float(min(row["balanced_accuracy"] for row in null_rows)),
                float(max(row["balanced_accuracy"] for row in null_rows)),
            ],
        },
    }


def category_probe_battery(
    features: np.ndarray,
    frame: pd.DataFrame,
    category_column: str | None,
    train: np.ndarray,
    test: np.ndarray,
) -> dict[str, Any]:
    if not category_column:
        return {"available": False, "reason": "dataset has no validated category labels"}
    labels = frame[category_column].astype(str).to_numpy()
    linear = probe(features, labels, train, test, nonlinear=False, seeds=PROBE_SEEDS)
    nonlinear = probe(features, labels, train, test, nonlinear=True, seeds=PROBE_SEEDS)
    primary = linear["runs"][0]
    prediction = np.asarray(primary["predictions"])
    truth = np.asarray(primary["truth"])
    scanner = frame.iloc[test]["scanner_id"].astype(str).to_numpy()
    slide = frame.iloc[test]["slide_id"].astype(str).to_numpy()
    scanner_scores = {
        name: float(balanced_accuracy_score(truth[scanner == name], prediction[scanner == name]))
        for name in sorted(set(scanner))
    }
    slide_scores: dict[str, float] = {}
    for name in sorted(set(slide)):
        mask = slide == name
        if len(set(truth[mask])) > 1:
            slide_scores[name] = float(
                balanced_accuracy_score(truth[mask], prediction[mask])
            )
    return {
        "available": True,
        "linear": linear,
        "nonlinear": nonlinear,
        "worst_scanner_balanced_accuracy": min(scanner_scores.values()),
        "scanner_balanced_accuracy": scanner_scores,
        "slide_balanced_accuracy": slide_scores,
        "slide_averaged_balanced_accuracy": float(np.mean(list(slide_scores.values()))) if slide_scores else None,
    }


def normalize_rows(features: np.ndarray) -> np.ndarray:
    norm = np.linalg.norm(features, axis=1, keepdims=True)
    return features / np.maximum(norm, 1e-12)


def retrieval_metrics(features: np.ndarray, frame: pd.DataFrame, test: np.ndarray) -> dict[str, Any]:
    normalized = normalize_rows(features[test])
    sub = frame.iloc[test].reset_index(drop=True)
    scanners = sorted(sub["scanner_id"].astype(str).unique())
    ordered = []
    ranks_all: list[int] = []
    same_cosines: list[float] = []
    different_cosines: list[float] = []
    ranks_by_slide: dict[str, list[int]] = {}
    ranks_by_slide_pair: dict[str, dict[str, list[int]]] = {}
    for source_scanner in scanners:
        for target_scanner in scanners:
            if source_scanner == target_scanner:
                continue
            source = np.flatnonzero(sub["scanner_id"].astype(str).to_numpy() == source_scanner)
            target = np.flatnonzero(sub["scanner_id"].astype(str).to_numpy() == target_scanner)
            similarities = normalized[source] @ normalized[target].T
            target_regions = sub.iloc[target]["region_id"].astype(str).to_numpy()
            pair_ranks = []
            for local_index, source_index in enumerate(source):
                region = str(sub.iloc[source_index]["region_id"])
                order = np.argsort(-similarities[local_index])
                matches = np.flatnonzero(target_regions[order] == region)
                if len(matches) != 1:
                    raise ExperimentError("Held-out ordered scanner pair lacks one exact region match.")
                rank = int(matches[0] + 1)
                pair_ranks.append(rank)
                ranks_all.append(rank)
                slide = str(sub.iloc[source_index]["slide_id"])
                pair_name = f"{source_scanner}->{target_scanner}"
                ranks_by_slide.setdefault(slide, []).append(rank)
                ranks_by_slide_pair.setdefault(slide, {}).setdefault(pair_name, []).append(rank)
                true_target = order[matches[0]]
                same_cosines.append(float(similarities[local_index, true_target]))
                if similarities.shape[1] > 1:
                    different_cosines.append(
                        float(np.mean(np.delete(similarities[local_index], true_target)))
                    )
            ordered.append(
                {
                    "source_scanner": source_scanner,
                    "target_scanner": target_scanner,
                    "top1": float(np.mean(np.asarray(pair_ranks) <= 1)),
                    "top5": float(np.mean(np.asarray(pair_ranks) <= 5)),
                    "mean_reciprocal_rank": float(np.mean(1.0 / np.asarray(pair_ranks))),
                }
            )
    same = float(np.mean(same_cosines))
    different = float(np.mean(different_cosines))
    slide_overall = {
        slide: float(np.mean(np.asarray(ranks) <= 1))
        for slide, ranks in ranks_by_slide.items()
    }
    slide_worst = {
        slide: min(
            float(np.mean(np.asarray(ranks) <= 1)) for ranks in pairs.values()
        )
        for slide, pairs in ranks_by_slide_pair.items()
    }
    return {
        "overall_top1": float(np.mean(np.asarray(ranks_all) <= 1)),
        "overall_top5": float(np.mean(np.asarray(ranks_all) <= 5)),
        "mean_reciprocal_rank": float(np.mean(1.0 / np.asarray(ranks_all))),
        "ordered_scanner_pairs": ordered,
        "worst_ordered_scanner_pair_top1": min(item["top1"] for item in ordered),
        "same_region_cosine_similarity": same,
        "different_region_cosine_similarity": different,
        "similarity_margin": same - different,
        "slide_overall_top1": slide_overall,
        "slide_worst_ordered_pair_top1": slide_worst,
    }


def spectral_diagnostics(
    biological: np.ndarray,
    frame: pd.DataFrame,
    category_column: str | None,
    train: np.ndarray,
    test: np.ndarray,
) -> dict[str, Any]:
    centered = biological[test] - biological[test].mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    eigen = singular**2 / max(1, len(test) - 1)
    positive = eigen[eigen > np.finfo(float).eps * max(centered.shape) * eigen.max()]
    probability = positive / positive.sum() if positive.size and positive.sum() else np.array([])
    effective = float(np.exp(-(probability * np.log(probability)).sum())) if probability.size else 0.0
    participation = float(positive.sum() ** 2 / np.square(positive).sum()) if positive.size else 0.0
    pca = PCA().fit(biological[train])
    curves = []
    for components in PCA_COMPONENTS:
        if components > min(len(train) - 1, biological.shape[1]):
            curves.append({"component_count": components, "feasible": False})
            continue
        train_projection = pca.transform(biological[train])[:, :components]
        test_projection = pca.transform(biological[test])[:, :components]
        task = None
        if category_column:
            labels = frame[category_column].astype(str).to_numpy()
            task = probe(
                np.vstack([train_projection, test_projection]),
                np.concatenate([labels[train], labels[test]]),
                np.arange(len(train)),
                np.arange(len(train), len(train) + len(test)),
                nonlinear=False,
                seeds=(PROBE_SEEDS[0],),
            )["balanced_accuracy_median"]
        curves.append(
            {
                "component_count": components,
                "feasible": True,
                "held_out_category_balanced_accuracy": task,
            }
        )
    variance = np.var(centered, axis=0)
    return {
        "pca_fit_split": "train",
        "test_used_to_select_components": False,
        "component_curve": curves,
        "numerical_rank": int(np.linalg.matrix_rank(centered)),
        "effective_rank": effective,
        "participation_ratio": participation,
        "condition_number": float(singular[0] / max(singular[-1], 1e-12)),
        "variance_spectrum": eigen.tolist(),
        "near_zero_variance_fraction": float(np.mean(variance < 1e-8)),
    }


def evaluate_representation(
    biological: np.ndarray,
    acquisition: np.ndarray,
    dataset: LoadedDataset,
    fold: int,
) -> dict[str, Any]:
    frame = dataset.manifests[fold].copy()
    if dataset.category_column:
        frame[dataset.category_column] = dataset.archive_frame[dataset.category_column].to_numpy()
    train, _, test = split_indices(frame)
    acquisition_scanner = scanner_probe_battery(acquisition, frame, train, test)
    acquisition_category = category_probe_battery(
        acquisition, frame, dataset.category_column, train, test
    )
    centered_prototypes = np.vstack(
        [acquisition[frame["scanner_id"].astype(str).to_numpy() == scanner].mean(axis=0) for scanner in dataset.scanner_names]
    )
    prototype_distances = np.linalg.norm(
        centered_prototypes[:, None, :] - centered_prototypes[None, :, :], axis=2
    )
    return {
        "biological_scanner_probe": scanner_probe_battery(biological, frame, train, test),
        "acquisition_scanner_probe": {
            **acquisition_scanner,
            "scanner_pair_confusion": acquisition_scanner["linear_confusion_matrix"],
            "prototype_separation_minimum": float(
                prototype_distances[~np.eye(len(centered_prototypes), dtype=bool)].min()
            ),
        },
        "paired_region_preservation": retrieval_metrics(biological, frame, test),
        "biological_category_accessibility": category_probe_battery(
            biological, frame, dataset.category_column, train, test
        ),
        "acquisition_category_leakage": acquisition_category,
        "spectral_accessibility": spectral_diagnostics(
            biological, frame, dataset.category_column, train, test
        ),
    }


def simple_baseline(
    features: np.ndarray,
    frame: pd.DataFrame,
    category_column: str | None,
) -> dict[str, Any]:
    train, _, test = split_indices(frame)
    scaler = StandardScaler().fit(features[train])
    standardized = scaler.transform(features)
    scanner_names = sorted(frame["scanner_id"].astype(str).unique())
    centroids = np.vstack(
        [standardized[train][frame.iloc[train]["scanner_id"].astype(str).to_numpy() == scanner].mean(axis=0) for scanner in scanner_names]
    )
    directions, _ = np.linalg.qr((centroids - centroids.mean(axis=0)).T)
    rank = min(len(scanner_names) - 1, directions.shape[1])
    projected = standardized - standardized @ directions[:, :rank] @ directions[:, :rank].T
    pca = PCA(n_components=min(8, len(train) - 1, features.shape[1])).fit(standardized[train])
    pca_removed = standardized - pca.inverse_transform(pca.transform(standardized))
    canonical_scanner = scanner_names[0]
    row_lookup = {
        (str(row.region_id), str(row.scanner_id)): int(index)
        for index, row in frame.iterrows()
    }
    train_regions = sorted(frame.iloc[train]["region_id"].astype(str).unique())
    linear_pair_transformed = standardized.copy()
    for scanner in scanner_names[1:]:
        paired_regions = [
            region
            for region in train_regions
            if (region, scanner) in row_lookup
            and (region, canonical_scanner) in row_lookup
        ]
        source_train = np.asarray([row_lookup[(region, scanner)] for region in paired_regions])
        target_train = np.asarray(
            [row_lookup[(region, canonical_scanner)] for region in paired_regions]
        )
        source_values = standardized[source_train]
        target_values = standardized[target_train]
        source_centered = source_values - source_values.mean(axis=0)
        target_centered = target_values - target_values.mean(axis=0)
        slope = np.sum(source_centered * target_centered, axis=0) / np.maximum(
            np.sum(source_centered**2, axis=0), 1e-12
        )
        intercept = target_values.mean(axis=0) - slope * source_values.mean(axis=0)
        rows = np.flatnonzero(frame["scanner_id"].astype(str).to_numpy() == scanner)
        linear_pair_transformed[rows] = standardized[rows] * slope + intercept
    random_control = np.random.default_rng(8701).normal(
        size=(len(features), min(64, features.shape[1]))
    )
    variants = {
        "original_frozen_features": standardized,
        "centroid_qr_scanner_subspace_projection": projected,
        "pca_first_8_components_removed": pca_removed,
        "linear_scanner_pair_transform_to_canonical_scanner": linear_pair_transformed,
        "scanner_balanced_random_control": random_control,
    }
    output = {}
    for name, representation in variants.items():
        output[name] = {
            "biological_scanner_probe": scanner_probe_battery(representation, frame, train, test),
            "paired_region_preservation": retrieval_metrics(representation, frame, test),
            "category_accessibility": category_probe_battery(
                representation, frame, category_column, train, test
            ),
        }
    output["historical_evidence"] = {
        "comparative": False,
        "reason": "historical model estimands and architecture families differ from this forward-valid test",
    }
    return output


def metric_value(run: Mapping[str, Any], metric: str) -> float | None:
    evaluation = run["layer1"]
    paths: dict[str, Sequence[str]] = {
        "category_balanced_accuracy": ("biological_category_accessibility", "linear", "balanced_accuracy_median"),
        "biological_scanner_balanced_accuracy": ("biological_scanner_probe", "linear", "balanced_accuracy_median"),
        "worst_pair_region_retrieval": ("paired_region_preservation", "worst_ordered_scanner_pair_top1"),
        "overall_region_retrieval": ("paired_region_preservation", "overall_top1"),
        "acquisition_category_leakage": ("acquisition_category_leakage", "linear", "balanced_accuracy_median"),
        "acquisition_scanner_accuracy": ("acquisition_scanner_probe", "linear", "balanced_accuracy_median"),
    }
    value: Any = evaluation
    for key in paths[metric]:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return float(value) if value is not None else None


PRIMARY_METRICS = (
    "category_balanced_accuracy",
    "biological_scanner_balanced_accuracy",
    "worst_pair_region_retrieval",
    "overall_region_retrieval",
    "acquisition_category_leakage",
    "acquisition_scanner_accuracy",
)


def average_seeds_within_fold(runs: Sequence[Mapping[str, Any]], dataset: str) -> list[dict[str, Any]]:
    rows = []
    for fold in FOLDS:
        families = {
            family: [run for run in runs if run["dataset"] == dataset and run["fold"] == fold and run["family"] == family and not run["broken_pair_control"]]
            for family in FAMILIES
        }
        if any(len(items) != len(MODEL_SEEDS) for items in families.values()):
            raise ExperimentError(f"Incomplete seed grid for {dataset}, fold {fold}.")
        row: dict[str, Any] = {"dataset": dataset, "fold": fold, "seed_averaged_before_inference": True}
        for metric in PRIMARY_METRICS:
            family_means = {}
            for family, items in families.items():
                values = [metric_value(run, metric) for run in items]
                usable = [value for value in values if value is not None]
                family_means[family] = float(np.mean(usable)) if usable else None
            row[metric] = {
                "real_b32_reference": family_means["real_b32_reference"],
                "real_b64_parameter_matched": family_means["real_b64_parameter_matched"],
                "difference": None
                if None in family_means.values()
                else float(family_means["real_b64_parameter_matched"] - family_means["real_b32_reference"]),
            }
        rows.append(row)
    return rows


def clustered_metric_values(run: Mapping[str, Any], metric: str) -> Mapping[str, float]:
    layer1 = run["layer1"]
    paths = {
        "category_balanced_accuracy": ("biological_category_accessibility", "slide_balanced_accuracy"),
        "biological_scanner_balanced_accuracy": ("biological_scanner_probe", "slide_balanced_accuracy"),
        "worst_pair_region_retrieval": ("paired_region_preservation", "slide_worst_ordered_pair_top1"),
        "overall_region_retrieval": ("paired_region_preservation", "slide_overall_top1"),
        "acquisition_category_leakage": ("acquisition_category_leakage", "slide_balanced_accuracy"),
        "acquisition_scanner_accuracy": ("acquisition_scanner_probe", "slide_balanced_accuracy"),
    }
    value: Any = layer1
    for key in paths[metric]:
        if not isinstance(value, Mapping) or key not in value:
            return {}
        value = value[key]
    return {str(key): float(item) for key, item in value.items()}


def fold_specimen_differences(
    runs: Sequence[Mapping[str, Any]], dataset: str, metric: str
) -> dict[int, list[float]]:
    output: dict[int, list[float]] = {}
    for fold in FOLDS:
        by_family: dict[str, list[Mapping[str, float]]] = {}
        for family in FAMILIES:
            selected = [
                run
                for run in runs
                if run["dataset"] == dataset
                and run["fold"] == fold
                and run["family"] == family
                and not run["broken_pair_control"]
            ]
            by_family[family] = [clustered_metric_values(run, metric) for run in selected]
        units = set.intersection(
            *[
                set(mapping)
                for family_maps in by_family.values()
                for mapping in family_maps
            ]
        ) if all(by_family.values()) else set()
        differences = []
        for unit in sorted(units):
            means = {
                family: float(np.mean([mapping[unit] for mapping in mappings]))
                for family, mappings in by_family.items()
            }
            differences.append(
                means["real_b64_parameter_matched"] - means["real_b32_reference"]
            )
        output[fold] = differences
    return output


def deterministic_fold_then_specimen_bootstrap(
    values: Sequence[float] | Mapping[int, Sequence[float]], seed: int
) -> dict[str, Any]:
    if isinstance(values, Mapping):
        fold_values = {
            int(fold): np.asarray(items, dtype=float)
            for fold, items in values.items()
            if len(items)
        }
    else:
        fold_values = {
            fold: np.asarray([value], dtype=float)
            for fold, value in enumerate(values)
        }
    if not fold_values:
        raise ExperimentError("Clustered bootstrap requires at least one fold.")
    folds = sorted(fold_values)
    rng = np.random.default_rng(seed)
    draws = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for index in range(BOOTSTRAP_REPLICATES):
        selected_folds = rng.choice(folds, size=len(folds), replace=True)
        fold_draws = []
        for fold in selected_folds:
            specimens = fold_values[int(fold)]
            selected_specimens = rng.integers(0, len(specimens), size=len(specimens))
            fold_draws.append(float(specimens[selected_specimens].mean()))
        draws[index] = float(np.mean(fold_draws))
    return {
        "clustering": "model seeds averaged within fold; folds resampled first, then slides/specimens within fold",
        "seed": seed,
        "replicates": BOOTSTRAP_REPLICATES,
        "lower_2_5_percent": float(np.quantile(draws, 0.025)),
        "upper_97_5_percent": float(np.quantile(draws, 0.975)),
    }


def paired_contrast(runs: Sequence[Mapping[str, Any]], dataset: str) -> dict[str, Any]:
    folds = average_seeds_within_fold(runs, dataset)
    metrics = {}
    for offset, metric in enumerate(PRIMARY_METRICS):
        differences = [row[metric]["difference"] for row in folds]
        usable = [float(value) for value in differences if value is not None]
        specimen_differences = fold_specimen_differences(runs, dataset, metric)
        metrics[metric] = {
            "fold_differences": differences,
            "mean_difference": float(np.mean(usable)) if usable else None,
            "positive_fold_count": sum(value > 0 for value in usable),
            "fold_specimen_differences": specimen_differences,
            "bootstrap_interval": deterministic_fold_then_specimen_bootstrap(
                specimen_differences
                if any(specimen_differences.values())
                else usable,
                BOOTSTRAP_SEED + offset,
            )
            if usable
            else None,
        }
    repeated_seed_leakage_folds = []
    for fold in FOLDS:
        leakage_counts = {}
        for family in FAMILIES:
            selected = [
                run
                for run in runs
                if run["dataset"] == dataset
                and run["fold"] == fold
                and run["family"] == family
                and not run["broken_pair_control"]
            ]
            leakage_counts[family] = sum(
                metric_value(run, "biological_scanner_balanced_accuracy")
                > run["layer1"]["biological_scanner_probe"][
                    "paired_identity_aware_permutation_null"
                ]["range"][1]
                + MATERIAL_MARGIN
                for run in selected
            )
        if (
            leakage_counts["real_b64_parameter_matched"] >= 3
            and leakage_counts["real_b32_reference"] < 3
        ):
            repeated_seed_leakage_folds.append(fold)
    return {
        "fold_results": folds,
        "metrics": metrics,
        "repeated_seed_leakage_created_folds": repeated_seed_leakage_folds,
    }


def broken_pair_assessment(runs: Sequence[Mapping[str, Any]], dataset: str) -> dict[str, Any]:
    comparisons = []
    for family in FAMILIES:
        true = next(
            run
            for run in runs
            if run["dataset"] == dataset
            and run["fold"] == BROKEN_PAIR_FOLDS[0]
            and run["seed"] == BROKEN_PAIR_SEEDS[0]
            and run["family"] == family
            and not run["broken_pair_control"]
        )
        broken = next(
            run
            for run in runs
            if run["dataset"] == dataset
            and run["fold"] == BROKEN_PAIR_FOLDS[0]
            and run["seed"] == BROKEN_PAIR_SEEDS[0]
            and run["family"] == family
            and run["broken_pair_control"]
        )
        true_retrieval = metric_value(true, "overall_region_retrieval")
        broken_retrieval = metric_value(broken, "overall_region_retrieval")
        true_margin = true["layer1"]["paired_region_preservation"]["similarity_margin"]
        broken_margin = broken["layer1"]["paired_region_preservation"]["similarity_margin"]
        comparisons.append(
            {
                "family": family,
                "fold": BROKEN_PAIR_FOLDS[0],
                "seed": BROKEN_PAIR_SEEDS[0],
                "region_retrieval_difference": float(true_retrieval - broken_retrieval),
                "similarity_margin_difference": float(true_margin - broken_margin),
                "true_pairs_outperform": bool(
                    true_retrieval > broken_retrieval or true_margin > broken_margin
                ),
            }
        )
    return {
        "comparisons": comparisons,
        "paired_supervision_demonstrated": any(item["true_pairs_outperform"] for item in comparisons),
    }


def classify_dataset(
    readiness: Mapping[str, Any],
    contrast: Mapping[str, Any],
    broken: Mapping[str, Any],
) -> dict[str, Any]:
    if not readiness.get("feature_space_ready"):
        return {"conclusion": "dataset_not_ready"}
    if not broken["paired_supervision_demonstrated"]:
        return {"conclusion": "paired_supervision_not_demonstrated"}
    metrics = contrast["metrics"]
    category = metrics["category_balanced_accuracy"]
    scanner = metrics["biological_scanner_balanced_accuracy"]
    worst = metrics["worst_pair_region_retrieval"]
    overall = metrics["overall_region_retrieval"]
    acquisition_category = metrics["acquisition_category_leakage"]
    if not readiness.get("category_labels_available"):
        return {
            "conclusion": "feature_only_evidence",
            "biological_accessibility_improvement_claimed": False,
        }
    category_gain = bool(
        category["mean_difference"] >= MATERIAL_MARGIN
        and category["positive_fold_count"] >= 4
    )
    benefit = bool(
        category_gain
        and scanner["mean_difference"] <= MATERIAL_MARGIN
        and worst["mean_difference"] >= -MATERIAL_MARGIN
        and overall["mean_difference"] >= -MATERIAL_MARGIN
        and acquisition_category["mean_difference"] <= MATERIAL_MARGIN
    )
    scanner_tradeoff = bool(
        scanner["mean_difference"] > MATERIAL_MARGIN
        or bool(contrast.get("repeated_seed_leakage_created_folds"))
        or worst["mean_difference"] < -MATERIAL_MARGIN
        or overall["mean_difference"] < -MATERIAL_MARGIN
    )
    if category_gain and scanner_tradeoff:
        conclusion = "b64_allocation_with_scanner_tradeoff"
    elif benefit:
        conclusion = "b64_allocation_supported"
    elif scanner_tradeoff or 0 < category["positive_fold_count"] < 5:
        conclusion = "b64_allocation_mixed"
    else:
        conclusion = "b64_allocation_unsupported"
    return {
        "conclusion": conclusion,
        "biological_accessibility_improvement_claimed": benefit,
        "category_gain_detected": category_gain,
        "real_scanner_tradeoff_detected": scanner_tradeoff,
        "material_effect_margin": MATERIAL_MARGIN,
    }


def top_level_status(dataset_conclusions: Mapping[str, Mapping[str, Any]]) -> str:
    conclusions = [item["conclusion"] for item in dataset_conclusions.values()]
    if not conclusions or all(item == "dataset_not_ready" for item in conclusions):
        return "real_paired_scanner_validation_not_ready"
    if any(item == "paired_supervision_not_demonstrated" for item in conclusions):
        return "complete_paired_supervision_not_demonstrated"
    supported = {"b64_allocation_supported", "b64_allocation_with_scanner_tradeoff"}
    if len(conclusions) == 2 and all(item == "b64_allocation_supported" for item in conclusions):
        return "complete_cross_dataset_b64_allocation_supported"
    if len(conclusions) == 2 and all(item in supported for item in conclusions) and any(
        item == "b64_allocation_with_scanner_tradeoff" for item in conclusions
    ):
        return "complete_cross_dataset_b64_allocation_with_scanner_tradeoff"
    if len(conclusions) == 1 and conclusions[0] == "b64_allocation_supported":
        return "complete_single_dataset_b64_allocation_supported"
    if all(item in {"b64_allocation_unsupported", "feature_only_evidence"} for item in conclusions):
        return "complete_real_paired_scanner_allocation_unsupported"
    return "complete_mixed_real_paired_scanner_allocation_effects"


def summary_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for run in result.get("runs", []):
        rows.append(
            {
                "row_type": "run",
                "dataset": run["dataset"],
                "fold": run["fold"],
                "seed": run["seed"],
                "family": run["family"],
                "broken_pair_control": run["broken_pair_control"],
                "biological_scanner_balanced_accuracy": metric_value(run, "biological_scanner_balanced_accuracy"),
                "category_balanced_accuracy": metric_value(run, "category_balanced_accuracy"),
                "overall_region_retrieval": metric_value(run, "overall_region_retrieval"),
                "worst_pair_region_retrieval": metric_value(run, "worst_pair_region_retrieval"),
            }
        )
    for dataset, conclusion in result.get("dataset_conclusions", {}).items():
        rows.append({"row_type": "dataset_conclusion", "dataset": dataset, **conclusion})
    rows.append({"row_type": "top_level", "status": result["status"]})
    return rows


def immutable_inputs(readiness: Mapping[str, Any], synthetic: Mapping[str, Any]) -> dict[str, str]:
    inputs = {synthetic["path"]: synthetic["file_sha256"], synthetic["manifest_path"]: synthetic["manifest_file_sha256"]}
    for item in synthetic["inherited_artifacts"].values():
        inputs[item["path"]] = item["verified_file_sha256"]
    for dataset in readiness["datasets"].values():
        if dataset.get("feature_sha256"):
            inputs[dataset["feature_path"]] = dataset["feature_sha256"]
        for manifest in dataset.get("manifest_files", []):
            inputs[manifest["path"]] = manifest["sha256"]
        for kind in ("branch_embedding_archives", "decoder_or_composition_weights", "swap_or_pair_assignment_metadata"):
            for artifact in dataset.get("legacy_layer2_discovery", {}).get(kind, []):
                inputs[artifact["path"]] = artifact["sha256"]
    return inputs


def verify_inputs_unchanged(inputs: Mapping[str, str]) -> None:
    for raw_path, expected in inputs.items():
        path = Path(raw_path)
        if not path.is_file() or sha256_file(path) != expected:
            raise ExperimentError(f"Frozen/input artifact changed during execution: {path}")


def run_experiment(
    synthetic_path: Path,
    repository_root: Path,
    output_root: Path,
    device: torch.device,
) -> dict[str, Any]:
    if output_root.exists():
        raise ExperimentError(f"Output directory already exists: {output_root}")
    output_root.mkdir(parents=True, exist_ok=False)
    synthetic = verify_synthetic_factorial(synthetic_path)
    readiness = audit_readiness(repository_root)
    readiness.update(
        {
            "git_commit": git_commit(repository_root),
            "frozen_synthetic": synthetic,
        }
    )
    readiness.pop("readiness_sha256", None)
    readiness["readiness_sha256"] = canonical_hash(readiness)
    readiness_path = output_root / "real_paired_scanner_bottleneck_allocation_readiness.json"
    atomic_json(readiness_path, readiness)
    frozen_inputs = immutable_inputs(readiness, synthetic)
    if not readiness["ready_datasets"]:
        result = {
            "schema_version": SCHEMA_VERSION,
            "claim_scope": CLAIM_SCOPE,
            "git_commit": git_commit(repository_root),
            "frozen_synthetic": synthetic,
            "readiness": readiness,
            "frozen_input_hashes": frozen_inputs,
            "factorizer_fit_count": 0,
            "runs": [],
            "status": "real_paired_scanner_validation_not_ready",
            "failure_reasons": [],
        }
        verify_inputs_unchanged(frozen_inputs)
        return result
    runs: list[dict[str, Any]] = []
    baselines: dict[str, Any] = {}
    parameter_audits: dict[str, Any] = {}
    fit_count = 0
    for dataset_name in readiness["ready_datasets"]:
        dataset = load_dataset(repository_root, dataset_name)
        parameter_match = readiness["datasets"][dataset_name]["parameter_match"]
        parameter_audits[dataset_name] = copy.deepcopy(parameter_match)
        for family in FAMILIES:
            biological_dim, hidden_dim = family_dimensions(parameter_match, family)
            preflight = build_factorizer(
                dataset.features.shape[1],
                len(dataset.scanner_names),
                biological_dim,
                hidden_dim,
                torch.device("cpu"),
            )
            actual = sum(parameter.numel() for parameter in preflight.parameters())
            expected = int(parameter_match[family]["formula_parameter_count"])
            parameter_audits[dataset_name][family]["actual_pytorch_parameter_count"] = actual
            parameter_audits[dataset_name][family]["actual_matches_formula"] = actual == expected
            if actual != expected:
                raise ExperimentError(
                    f"Parameter preflight failed for {dataset_name}/{family}: {actual} != {expected}"
                )
            del preflight
        frame0 = dataset.manifests[0].copy()
        if dataset.category_column:
            frame0[dataset.category_column] = dataset.archive_frame[dataset.category_column].to_numpy()
        baselines[dataset_name] = {
            str(fold): simple_baseline(
                dataset.features,
                dataset.manifests[fold].assign(
                    **(
                        {dataset.category_column: dataset.archive_frame[dataset.category_column].to_numpy()}
                        if dataset.category_column
                        else {}
                    )
                ),
                dataset.category_column,
            )
            for fold in FOLDS
        }
        schedule = [
            (fold, seed, family, False)
            for fold in FOLDS
            for seed in MODEL_SEEDS
            for family in FAMILIES
        ] + [
            (fold, seed, family, True)
            for fold in BROKEN_PAIR_FOLDS
            for seed in BROKEN_PAIR_SEEDS
            for family in FAMILIES
        ]
        for fold, seed, family, broken in schedule:
            started = time.time()
            model, scaler, training = train_factorizer(
                dataset,
                fold,
                family,
                seed,
                parameter_match,
                device,
                broken_pairs=broken,
            )
            biological, acquisition = project_factorizer(model, scaler, dataset, fold, device)
            layer1 = evaluate_representation(biological, acquisition, dataset, fold)
            runs.append(
                {
                    "dataset": dataset_name,
                    "fold": fold,
                    "seed": seed,
                    "family": family,
                    "broken_pair_control": broken,
                    "biological_dimension": family_dimensions(parameter_match, family)[0],
                    "hidden_width": family_dimensions(parameter_match, family)[1],
                    "acquisition_dimension": ACQUISITION_DIM,
                    "training": training,
                    "layer1": layer1,
                    "layer2": {
                        "available": False,
                        "reason": readiness["datasets"][dataset_name]["layer2_unavailable_reasons"],
                    },
                    "pixel_space_evaluation_performed": False,
                    "runtime_seconds": time.time() - started,
                }
            )
            fit_count += 1
            print(
                f"completed fit {fit_count}: dataset={dataset_name} fold={fold} seed={seed} family={family} broken={broken}",
                flush=True,
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()
    contrasts = {
        name: paired_contrast(runs, name) for name in readiness["ready_datasets"]
    }
    broken_assessments = {
        name: broken_pair_assessment(runs, name) for name in readiness["ready_datasets"]
    }
    conclusions = {
        name: classify_dataset(
            readiness["datasets"][name], contrasts[name], broken_assessments[name]
        )
        for name in readiness["ready_datasets"]
    }
    result = {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": CLAIM_SCOPE,
        "git_commit": git_commit(repository_root),
        "device": str(device),
        "frozen_synthetic": synthetic,
        "readiness": readiness,
        "frozen_input_hashes": frozen_inputs,
        "common_feature_backbone": readiness["common_feature_backbone"],
        "architecture_families": list(FAMILIES),
        "model_seeds": list(MODEL_SEEDS),
        "folds": list(FOLDS),
        "parameter_audits": parameter_audits,
        "factorizer_fit_count": fit_count,
        "primary_true_pair_fit_count": len(readiness["ready_datasets"]) * len(FOLDS) * len(MODEL_SEEDS) * len(FAMILIES),
        "broken_pair_fit_count": len(readiness["ready_datasets"]) * len(BROKEN_PAIR_FOLDS) * len(BROKEN_PAIR_SEEDS) * len(FAMILIES),
        "training_uses_synthetic_generator": False,
        "factorizer_training_uses_category_or_biological_labels": False,
        "pixel_or_wsi_model_constructed": False,
        "baselines": baselines,
        "runs": runs,
        "broken_pair_controls": broken_assessments,
        "fold_aware_paired_contrasts": contrasts,
        "dataset_conclusions": conclusions,
        "status": top_level_status(conclusions),
        "failure_reasons": [],
    }
    verify_inputs_unchanged(frozen_inputs)
    return result


def write_outputs(output_root: Path, result: dict[str, Any]) -> None:
    result["result_sha256"] = canonical_hash(result)
    result_path = output_root / "real_paired_scanner_bottleneck_allocation_validation_result.json"
    summary_path = output_root / "real_paired_scanner_bottleneck_allocation_validation_summary.csv"
    manifest_path = output_root / "real_paired_scanner_bottleneck_allocation_validation_manifest.json"
    atomic_json(result_path, result)
    atomic_csv(summary_path, summary_rows(result))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": CLAIM_SCOPE,
        "git_commit": result["git_commit"],
        "status": result["status"],
        "canonical_internal_result_hash": result["result_sha256"],
        "frozen_synthetic": result["frozen_synthetic"],
        "frozen_input_hashes": result["frozen_input_hashes"],
        "parameter_audits": result.get("parameter_audits", {}),
        "factorizer_fit_count": result["factorizer_fit_count"],
        "artifacts": {
            "result": result_path.name,
            "summary": summary_path.name,
            "manifest": manifest_path.name,
            "readiness": "real_paired_scanner_bottleneck_allocation_readiness.json",
        },
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    atomic_json(manifest_path, manifest)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--synthetic-factorial-result", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-root", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ExperimentError("CUDA was requested but is unavailable.")
    result = run_experiment(
        args.synthetic_factorial_result.resolve(),
        args.repository_root.resolve(),
        args.output_root.resolve(),
        device,
    )
    write_outputs(args.output_root.resolve(), result)
    print(json.dumps({
        "status": result["status"],
        "factorizer_fit_count": result["factorizer_fit_count"],
        "result_sha256": result["result_sha256"],
        "output_root": str(args.output_root.resolve()),
    }, indent=2))


if __name__ == "__main__":
    main()
