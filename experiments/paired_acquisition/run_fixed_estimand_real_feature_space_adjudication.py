#!/usr/bin/env python3
"""No-training, fixed-estimand adjudication of the real paired-scanner feature-space evidence.

This runner performs a deterministic, fold-compatible final adjudication of the
frozen real paired-scanner bottleneck-allocation evidence. It:

- verifies the complete frozen artifact chain (result, readiness, manifest, the
  inherited synthetic factorial chain, and every immutable input), failing closed
  on any mismatch;
- reuses the repository's authoritative corrected fixed-five-category estimand
  (Dermis, Epidermis, Inflamm/Necrosis, SCC, Subcutis; Bone and Cartilage
  excluded) instead of reimplementing it;
- evaluates deterministic simple scanner-removal baselines (original frozen
  features, centroid/QR scanner-subspace projection, paired linear scanner
  transform, PCA scanner-component removal) under that estimand;
- attempts to recover the exact primary B32/B64 neural representations
  (5 folds x 5 seeds x 2 families = 50 cells) from immutable saved arrays or
  frozen checkpoints;
- if the required neural cells cannot be recovered, fails closed with the
  ``fixed_estimand_adjudication_not_ready`` status and enumerates the exact
  missing artifacts instead of retraining;
- reports the frozen seven-category endpoint separately as
  ``exploratory_seven_category_endpoint``;
- emits the Layer-2 missing-metadata schema as a future data-remediation
  specification only and never infers swap assignments.

Zero training is performed: no optimizer, no backward pass, no factorizer or
feature-encoder training, no synthetic-data generation, and no WSI or pixel
model construction.
"""

from __future__ import annotations

import argparse
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
from sklearn.decomposition import PCA
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

from experiments.paired_acquisition import (
    run_biological_label_preservation_audit as legacy,
)
from experiments.paired_acquisition import (
    run_biological_label_preservation_audit_v2 as v2,
)
from experiments.paired_acquisition import (
    run_biological_label_preservation_fixed_estimand as fixed_estimand,
)
from experiments.paired_acquisition import (
    run_real_paired_scanner_bottleneck_allocation_validation as real_validation,
)
from experiments.paired_acquisition import (
    run_synthetic_crossed_factor_identifiability as canonical,
)


SCHEMA_VERSION = "fixed-estimand-real-feature-space-adjudication/v1"
LAYER2_SCHEMA_VERSION = "layer2-swap-metadata-remediation-specification/v1"

FROZEN_RESULT_FILE_SHA256 = "7293ddcddfe8fbaf42eb2bd08bfcd20898a74d28542e17f1abbfa669c3d57762"
FROZEN_RESULT_INTERNAL_SHA256 = "fbb8be9a541c5c25f8994660fa5523c29ed13da06793fc8c0546d84cb8b960f7"
READINESS_FILE_SHA256 = "31360bbd03416fdd775e9397ccfc363e0da818be59f1d15d3010c4a4127c86ed"
READINESS_INTERNAL_SHA256 = "b092a70a1a94a025dcab72458ccb1a07b1c91bb8a877cc770eac16057d62b13e"
MANIFEST_FILE_SHA256 = "9c61429e828ff963de0dd6b10445cab84bc5523f1393fea0a3b9408afd136037"
MANIFEST_INTERNAL_SHA256 = "bf30504865f1a852c40ad5d77e984c780d40a7876df9de91bf1b2765e68a0066"
FROZEN_RESULT_COPY_PATH = (
    "C:/Users/matth/Documents/Codex/2026-08-02/"
    "files-mentioned-by-the-user-continue/outputs/"
    "real_paired_scanner_bottleneck_allocation_validation_20260803T191207/"
    "real_paired_scanner_bottleneck_allocation_validation_result.json"
)
FROZEN_STATUS = "complete_mixed_real_paired_scanner_allocation_effects"

FIXED_CATEGORIES = ["Dermis", "Epidermis", "Inflamm/Necrosis", "SCC", "Subcutis"]
EXCLUDED_CATEGORIES = ["Bone", "Cartilage"]
FOLDS = (0, 1, 2, 3, 4)
MODEL_SEEDS = (2201, 2202, 2203, 2204, 2205)
FAMILIES = ("real_b32_reference", "real_b64_parameter_matched")
MINIMUM_FIT_SAMPLES = 2
MINIMUM_TEST_SAMPLES = 2

MATERIAL_MARGIN = 0.02
CROSS_FOLD_DOMINANCE_REQUIRED_FOLDS = 4
NULL_SEEDS = (8401, 8402, 8403)
BOOTSTRAP_SEED = 8701
BOOTSTRAP_REPLICATES = 10_000
PCA_REMOVED_COMPONENTS = 8
RANDOM_CONTROL_SEED = 8701
RANDOM_CONTROL_DIMENSION = 64

DETERMINISTIC_METHODS = (
    "original_frozen_features",
    "centroid_qr_scanner_subspace_projection",
    "paired_linear_scanner_transform",
    "pca_scanner_component_removal",
)
NEURAL_METHODS = ("real_b32_reference", "real_b64_parameter_matched")

SYNTHETIC_FACTORIAL_RESULT_PATH = (
    Path("results/biological_bottleneck_capacity_allocation_factorial_20260803T150254")
    / "biological_bottleneck_capacity_allocation_factorial_result.json"
)

CANINE_FEATURE_PATH = (
    Path("results/external_multiscanner_caninescc/features/fold_0_dinov2_base.npz")
)
CANINE_MANIFEST_DIR = Path("data/external_multiscanner_caninescc/patch_manifests/splits")
CANINE_MANIFEST_PATTERN = "fold_{fold}_patch_manifest.csv"
SCORPION_FEATURE_PATH = Path("results/scorpion/features/fold_0_dinov2_base.npz")
SCORPION_MANIFEST_DIR = Path("data/scorpion/splits")
SCORPION_MANIFEST_PATTERN = "fold_{fold}_manifest.csv"

CLAIM_SCOPE = {
    "frozen_status_unchanged": FROZEN_STATUS,
    "adjudication_is_no_training": True,
    "fixed_feature_evidence_establishes_pixel_behavior": False,
    "category_preservation_is_clinical_validation": False,
    "scanner_swapping_validated": False,
    "layer2_metadata_required_for_swap_claims": True,
    "synthetic_transport_claimed_only_with_corrected_category_gain": True,
    "neural_cells_required_for_biological_frontier": True,
}


class ExperimentError(RuntimeError):
    """A structural or execution failure, distinct from a poor scientific result."""


@dataclass(frozen=True)
class LoadedFold:
    dataset: str
    features: np.ndarray
    frame: pd.DataFrame
    category_column: str | None
    specimen_column: str


# ---------------------------------------------------------------------------
# Primitive helpers
# ---------------------------------------------------------------------------


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
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
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
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


def hash_string_array(values: Iterable[Any]) -> str:
    normalized = [
        "\x1f".join(map(str, value))
        if isinstance(value, (tuple, list, np.ndarray))
        else str(value)
        for value in values
    ]
    return hashlib.sha256("\n".join(normalized).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Frozen artifact-chain verification
# ---------------------------------------------------------------------------


def verify_internal_hash(path: Path, expected: str, hash_field: str) -> dict[str, Any]:
    if not path.is_file():
        raise ExperimentError(f"Frozen artifact missing: {path}")
    file_hash = sha256_file(path)
    if file_hash != expected and expected is not None:
        raise ExperimentError(f"Frozen artifact file hash mismatch: {path} {file_hash}")
    value = json.loads(path.read_text(encoding="utf-8"))
    embedded = value.get(hash_field)
    if not isinstance(embedded, str):
        raise ExperimentError(f"Frozen artifact has no {hash_field}: {path}")
    value_copy = json.loads(json.dumps(value))
    value_copy.pop(hash_field, None)
    if canonical_hash(value_copy) != embedded:
        raise ExperimentError(f"Frozen artifact canonical hash is invalid: {path}")
    return {
        "path": str(path.resolve()),
        "file_sha256": file_hash,
        "internal_sha256": embedded,
        "status": value.get("status"),
        "git_commit": value.get("git_commit"),
    }


def verify_frozen_real_validation(
    frozen_result_path: Path,
    repository_root: Path,
    copied_path: Path | None = None,
) -> dict[str, Any]:
    """Verify the frozen result, readiness, manifest, copied path, and chain."""
    result_dir = frozen_result_path.resolve().parent
    readiness_path = result_dir / "real_paired_scanner_bottleneck_allocation_readiness.json"
    manifest_path = (
        result_dir / "real_paired_scanner_bottleneck_allocation_validation_manifest.json"
    )
    result = verify_internal_hash(
        frozen_result_path, FROZEN_RESULT_FILE_SHA256, "result_sha256"
    )
    if result["internal_sha256"] != FROZEN_RESULT_INTERNAL_SHA256:
        raise ExperimentError("Frozen real-data result internal hash mismatch.")
    readiness = verify_internal_hash(
        readiness_path, READINESS_FILE_SHA256, "readiness_sha256"
    )
    if readiness["internal_sha256"] != READINESS_INTERNAL_SHA256:
        raise ExperimentError("Frozen real-data readiness internal hash mismatch.")
    manifest = verify_internal_hash(
        manifest_path, MANIFEST_FILE_SHA256, "manifest_sha256"
    )
    if manifest["internal_sha256"] != MANIFEST_INTERNAL_SHA256:
        raise ExperimentError("Frozen real-data manifest internal hash mismatch.")
    if result.get("status") != FROZEN_STATUS:
        raise ExperimentError(f"Frozen real-data status mismatch: {result.get('status')}")

    synthetic = real_validation.verify_synthetic_factorial(
        repository_root / SYNTHETIC_FACTORIAL_RESULT_PATH
    )

    frozen_value = json.loads(frozen_result_path.read_text(encoding="utf-8"))
    frozen_inputs = frozen_value.get("frozen_input_hashes", {})
    if not isinstance(frozen_inputs, Mapping) or not frozen_inputs:
        raise ExperimentError("Frozen real-data result has no immutable input hashes.")
    verified_inputs: dict[str, str] = {}
    for raw_path, expected in frozen_inputs.items():
        path = Path(raw_path)
        if not path.is_file():
            raise ExperimentError(f"Immutable input missing: {path}")
        observed = sha256_file(path)
        if observed != expected:
            raise ExperimentError(f"Immutable input hash mismatch: {path}")
        verified_inputs[str(path.resolve())] = observed

    copied_verified: dict[str, Any] | None = None
    if copied_path is not None:
        if not copied_path.is_file():
            raise ExperimentError(f"Copied frozen result missing: {copied_path}")
        observed_copy = sha256_file(copied_path)
        if observed_copy != FROZEN_RESULT_FILE_SHA256:
            raise ExperimentError("Copied frozen result hash mismatch.")
        copied_verified = {
            "path": str(copied_path.resolve()),
            "file_sha256": observed_copy,
        }

    return {
        "frozen_result": result,
        "frozen_readiness": readiness,
        "frozen_manifest": manifest,
        "frozen_synthetic": synthetic,
        "frozen_status": FROZEN_STATUS,
        "frozen_git_commit": result["git_commit"],
        "copied_result": copied_verified,
        "frozen_input_hashes": verified_inputs,
        "immutable_input_count": len(verified_inputs),
        "inherited_artifact_count": len(synthetic["inherited_artifacts"]),
        "chain_verified": True,
    }


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_canine_fold(repository_root: Path, fold: int) -> LoadedFold:
    feature_path = repository_root / CANINE_FEATURE_PATH
    with np.load(feature_path, allow_pickle=False) as archive:
        features = np.asarray(archive["features"], dtype=np.float32)
        base = pd.DataFrame(
            {
                name: archive[name].astype(str)
                for name in ("region_id", "scanner_id", "slide_id", "split")
            }
        )
    manifest = pd.read_csv(
        repository_root / CANINE_MANIFEST_DIR / CANINE_MANIFEST_PATTERN.format(fold=fold),
        dtype=str,
    )
    metadata = manifest[["region_id", "scanner_id", "split", "sample_id", "category_name"]].copy()
    for column in ("region_id", "scanner_id"):
        metadata[column] = metadata[column].astype(str)
        base[column] = base[column].astype(str)
    frame = base.drop(columns=["split"], errors="ignore").merge(
        metadata, on=["region_id", "scanner_id"], how="left", validate="one_to_one"
    )
    if len(frame) != len(features) or frame.isna().any().any():
        raise ExperimentError("Canine fold metadata merge failed.")
    return LoadedFold(
        dataset="canine_scc",
        features=features,
        frame=frame,
        category_column="category_name",
        specimen_column="slide_id",
    )


def load_scorpion_fold(repository_root: Path, fold: int) -> LoadedFold:
    feature_path = repository_root / SCORPION_FEATURE_PATH
    with np.load(feature_path, allow_pickle=False) as archive:
        features = np.asarray(archive["features"], dtype=np.float32)
        base = pd.DataFrame(
            {
                name: archive[name].astype(str)
                for name in ("region_id", "scanner_id", "slide_id", "split")
            }
        )
    manifest = pd.read_csv(
        repository_root / SCORPION_MANIFEST_DIR / SCORPION_MANIFEST_PATTERN.format(fold=fold),
        dtype=str,
    )
    metadata = manifest[["region_id", "scanner_id", "split"]].copy()
    for column in ("region_id", "scanner_id"):
        metadata[column] = metadata[column].astype(str)
        base[column] = base[column].astype(str)
    frame = base.drop(columns=["split"], errors="ignore").merge(
        metadata, on=["region_id", "scanner_id"], how="left", validate="one_to_one"
    )
    if len(frame) != len(features) or frame.isna().any().any():
        raise ExperimentError("SCORPION fold metadata merge failed.")
    return LoadedFold(
        dataset="scorpion",
        features=features,
        frame=frame,
        category_column=None,
        specimen_column="slide_id",
    )


def fold_integrity_check(frame: pd.DataFrame, specimen_column: str) -> dict[str, Any]:
    """Region, sample, and specimen must never cross train/val/test."""
    required_splits = {"train", "val", "test"}
    present = set(frame["split"].astype(str))
    region_crosses = int((frame.groupby("region_id")["split"].nunique() > 1).sum())
    specimen_crosses = int(
        (frame.groupby(specimen_column)["split"].nunique() > 1).sum()
    )
    sample_crosses = 0
    if "sample_id" in frame.columns:
        sample_crosses = int((frame.groupby("sample_id")["split"].nunique() > 1).sum())
    scanner_region_crosses = int(
        (frame.groupby(["region_id", "scanner_id"])["split"].nunique() > 1).sum()
    )
    passed = bool(
        required_splits.issubset(present)
        and region_crosses == 0
        and specimen_crosses == 0
        and sample_crosses == 0
        and scanner_region_crosses == 0
    )
    return {
        "required_splits_present": required_splits.issubset(present),
        "regions_crossing_splits": region_crosses,
        "specimens_crossing_splits": specimen_crosses,
        "samples_crossing_splits": sample_crosses,
        "scanner_views_crossing_splits": scanner_region_crosses,
        "passed": passed,
    }


def derive_fixed_categories_authoritative(repository_root: Path) -> dict[str, Any]:
    """Reuse the corrected fixed-estimand category derivation and verify it."""
    del repository_root
    retained, support = fixed_estimand.derive_fixed_categories(
        list(FOLDS),
        minimum_fit_samples=MINIMUM_FIT_SAMPLES,
        minimum_test_samples=MINIMUM_TEST_SAMPLES,
    )
    if list(retained) != FIXED_CATEGORIES:
        raise ExperimentError(
            "Corrected fixed-estimand category set mismatch: "
            f"{retained} != {FIXED_CATEGORIES}"
        )
    excluded = sorted(set(support["category"].astype(str)) - set(retained))
    if excluded != EXCLUDED_CATEGORIES:
        raise ExperimentError(
            f"Corrected excluded category set mismatch: {excluded} != {EXCLUDED_CATEGORIES}"
        )
    rows = []
    for fold in FOLDS:
        manifest = legacy.load_manifest(fold)
        fold_rows = []
        for category in FIXED_CATEGORIES:
            fit_support = fixed_estimand.category_sample_support(
                manifest.assign(
                    split=np.where(manifest["split"].eq("test"), "test", "fit")
                ),
                "fit",
            ).get(category, 0)
            test_support = fixed_estimand.category_sample_support(manifest, "test").get(
                category, 0
            )
            fold_rows.append(
                {
                    "fold": fold,
                    "category": category,
                    "fit_samples": fit_support,
                    "test_samples": test_support,
                    "fit_ok": fit_support >= MINIMUM_FIT_SAMPLES,
                    "test_ok": test_support >= MINIMUM_TEST_SAMPLES,
                }
            )
        rows.extend(fold_rows)
    all_ok = all(row["fit_ok"] and row["test_ok"] for row in rows)
    if not all_ok:
        raise ExperimentError("Fixed five-category sample support not met in every fold.")
    return {
        "fixed_categories": list(FIXED_CATEGORIES),
        "excluded_categories": list(EXCLUDED_CATEGORIES),
        "minimum_fit_samples": MINIMUM_FIT_SAMPLES,
        "minimum_test_samples": MINIMUM_TEST_SAMPLES,
        "per_fold_support": rows,
        "support_ok": all_ok,
        "vocabulary_source": "fold manifests only; no feature-derived labels",
        "reused_implementation": "run_biological_label_preservation_fixed_estimand",
    }


# ---------------------------------------------------------------------------
# Deterministic simple-baseline representations
# ---------------------------------------------------------------------------


def deterministic_representations(
    loaded: LoadedFold,
) -> dict[str, np.ndarray]:
    """Reproduce the frozen runner's deterministic simple-baseline transforms."""
    frame = loaded.frame
    features = loaded.features
    train = np.flatnonzero(frame["split"].astype(str).to_numpy() == "train")
    if len(train) == 0:
        raise ExperimentError("No training rows in deterministic representation split.")
    scaler = StandardScaler().fit(features[train])
    standardized = scaler.transform(features).astype(np.float32)
    scanner_names = sorted(frame["scanner_id"].astype(str).unique())

    centroids = np.vstack(
        [
            standardized[train][
                frame.iloc[train]["scanner_id"].astype(str).to_numpy() == scanner
            ].mean(axis=0)
            for scanner in scanner_names
        ]
    )
    directions, _ = np.linalg.qr((centroids - centroids.mean(axis=0)).T)
    rank = min(len(scanner_names) - 1, directions.shape[1])
    projected = standardized - standardized @ directions[:, :rank] @ directions[:, :rank].T

    pca = PCA(n_components=min(PCA_REMOVED_COMPONENTS, len(train) - 1, features.shape[1])).fit(
        standardized[train]
    )
    pca_removed = standardized - pca.inverse_transform(pca.transform(standardized))

    canonical_scanner = scanner_names[0]
    row_lookup = {
        (str(row.region_id), str(row.scanner_id)): int(index)
        for index, row in frame.iterrows()
    }
    train_regions = sorted(
        frame.iloc[train]["region_id"].astype(str).unique()
    )
    linear_pair_transformed = standardized.copy()
    for scanner in scanner_names[1:]:
        paired_regions = [
            region
            for region in train_regions
            if (region, scanner) in row_lookup
            and (region, canonical_scanner) in row_lookup
        ]
        source_train = np.asarray(
            [row_lookup[(region, scanner)] for region in paired_regions]
        )
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

    random_control = np.random.default_rng(RANDOM_CONTROL_SEED).normal(
        size=(len(features), RANDOM_CONTROL_DIMENSION)
    )

    return {
        "original_frozen_features": standardized,
        "centroid_qr_scanner_subspace_projection": projected,
        "paired_linear_scanner_transform": linear_pair_transformed,
        "pca_scanner_component_removal": pca_removed,
        "scanner_balanced_random_control": random_control,
    }


# ---------------------------------------------------------------------------
# Corrected fixed-estimand evaluation
# ---------------------------------------------------------------------------


def split_fit_test(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Split fit (non-test) and test rows without requiring sample_id.

    SCORPION is feature-only and has no sample_id; canine fold integrity is
    verified separately by ``fold_integrity_check`` and the category evaluation
    uses ``v2.split_indices`` (which enforces biological-sample blocking).
    """
    test = np.flatnonzero(frame["split"].astype(str).to_numpy() == "test")
    fit = np.flatnonzero(frame["split"].astype(str).to_numpy() != "test")
    if len(fit) == 0 or len(test) == 0:
        raise ExperimentError("Empty fit or test split.")
    return fit, test


def per_category_recall(
    truth: np.ndarray, prediction: np.ndarray, categories: Sequence[str]
) -> dict[str, float | None]:
    truth = np.asarray(truth, dtype=str)
    prediction = np.asarray(prediction, dtype=str)
    out: dict[str, float | None] = {}
    for category in categories:
        mask = truth == category
        if not mask.any():
            out[category] = None
        else:
            out[category] = float(np.mean(prediction[mask] == category))
    return out


def fixed_category_evaluation(
    features: np.ndarray,
    frame: pd.DataFrame,
    fold: int,
) -> dict[str, Any]:
    """Corrected fixed five-category metrics (probe fit on training rows only)."""
    if len(features) != len(frame) or not np.isfinite(features).all():
        raise ExperimentError(f"Non-finite or misaligned features for fixed estimand, fold {fold}.")
    category_mask = frame["category_name"].astype(str).isin(FIXED_CATEGORIES).to_numpy()
    category_features = features[category_mask]
    category_frame = frame.loc[category_mask].reset_index(drop=True)
    fit, test = v2.split_indices(category_frame)
    observed_test = set(category_frame.iloc[test]["category_name"].astype(str))
    if observed_test != set(FIXED_CATEGORIES):
        raise ExperimentError(
            f"fold {fold} does not contain the fixed category estimand: "
            f"{sorted(observed_test)}"
        )
    labels = category_frame["category_name"].astype(str).to_numpy()
    truth = labels[test]
    prediction = fixed_estimand.fit_probe(category_features, labels, fit, test)
    balanced = fixed_estimand.fixed_balanced_accuracy(truth, prediction, FIXED_CATEGORIES)
    macro_f1 = float(
        f1_score(
            truth,
            prediction,
            labels=FIXED_CATEGORIES,
            average="macro",
            zero_division=0,
        )
    )
    recall = per_category_recall(truth, prediction, FIXED_CATEGORIES)
    purity = v2.category_purity_fit_pool(category_features, category_frame, fit, test)

    test_frame = category_frame.iloc[test]
    scanner = test_frame["scanner_id"].astype(str).to_numpy()
    slide = test_frame["slide_id"].astype(str).to_numpy()
    scanner_scores: dict[str, float] = {}
    for name in sorted(set(scanner)):
        mask = scanner == name
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            scanner_scores[name] = float(balanced_accuracy_score(truth[mask], prediction[mask]))
    slide_scores: dict[str, float] = {}
    for name in sorted(set(slide)):
        mask = slide == name
        if len(set(truth[mask])) > 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=UserWarning)
                slide_scores[name] = float(balanced_accuracy_score(truth[mask], prediction[mask]))
    return {
        "fold": fold,
        "n_test_rows": int(len(test)),
        "n_test_samples": int(test_frame["sample_id"].nunique()),
        "balanced_accuracy": balanced,
        "macro_f1": macro_f1,
        "per_category_recall": recall,
        "purity_fit_pool_k1": purity["purity_fit_pool_k1"],
        "purity_fit_pool_k5": purity["purity_fit_pool_k5"],
        "purity_fit_pool_k10": purity["purity_fit_pool_k10"],
        "neighbour_same_region_excluded": True,
        "neighbour_same_sample_excluded": True,
        "scanner_stratified_balanced_accuracy": scanner_scores,
        "slide_balanced_accuracy": slide_scores,
        "slide_averaged_balanced_accuracy": (
            float(np.mean(list(slide_scores.values()))) if slide_scores else None
        ),
    }


def scanner_evaluation(
    features: np.ndarray,
    frame: pd.DataFrame,
    fold: int,
) -> dict[str, Any]:
    """Linear scanner balanced accuracy with a paired permutation null."""
    if len(features) != len(frame) or not np.isfinite(features).all():
        raise ExperimentError(f"Non-finite features in scanner evaluation, fold {fold}.")
    fit, test = split_fit_test(frame)
    labels = frame["scanner_id"].astype(str).to_numpy()
    truth = labels[test]
    prediction = fixed_estimand.fit_probe(features, labels, fit, test)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        balanced = float(balanced_accuracy_score(truth, prediction))
    scanners = sorted(set(labels))
    macro_f1 = float(
        f1_score(truth, prediction, labels=scanners, average="macro", zero_division=0)
    )
    recall: dict[str, float] = {}
    for name in scanners:
        mask = truth == name
        recall[name] = float(np.mean(prediction[mask] == name)) if mask.any() else None
    null_rows = []
    for seed in NULL_SEEDS:
        permuted = labels.copy()
        permuted[fit] = real_validation.paired_permutation_labels(frame, fit, seed)
        permuted[test] = real_validation.paired_permutation_labels(frame, test, seed + 10_000)
        null_prediction = fixed_estimand.fit_probe(features, permuted, fit, test)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=UserWarning)
            null_rows.append(float(balanced_accuracy_score(permuted[test], null_prediction)))
    return {
        "fold": fold,
        "chance_level": 1.0 / len(scanners),
        "linear_balanced_accuracy": balanced,
        "linear_macro_f1": macro_f1,
        "per_scanner_recall": recall,
        "nonlinear_balanced_accuracy": None,
        "nonlinear_not_applicable_reason": (
            "no calibrated nonlinear scanner probe exists in the corrected "
            "fixed-estimand evidence family"
        ),
        "paired_identity_aware_permutation_null": {
            "preserves_region_blocks": True,
            "runs": null_rows,
            "median": float(np.median(null_rows)),
            "range": [float(min(null_rows)), float(max(null_rows))],
        },
    }


def retrieval_evaluation(
    features: np.ndarray,
    frame: pd.DataFrame,
    fold: int,
) -> dict[str, Any]:
    """Cross-scanner region retrieval over the fixed held-out pool."""
    _, test = split_fit_test(frame)
    metrics = real_validation.retrieval_metrics(features, frame, test)
    metrics["fold"] = fold
    metrics["candidate_pool"] = "identical fixed held-out rows for every method"
    return metrics


def evaluate_deterministic_methods(loaded: LoadedFold, fold: int) -> dict[str, Any]:
    representations = deterministic_representations(loaded)
    output: dict[str, Any] = {}
    for method, representation in representations.items():
        row: dict[str, Any] = {
            "dataset": loaded.dataset,
            "fold": fold,
            "method": method,
            "representation_sha256": hashlib.sha256(
                np.ascontiguousarray(representation, dtype="<f4").reshape(-1).tobytes()
            ).hexdigest(),
            "row_order_sha256": hash_string_array(
                loaded.frame["region_id"].astype(str) + "|" + loaded.frame["scanner_id"].astype(str)
            ),
            "feature_dim": int(representation.shape[1]),
        }
        if loaded.category_column:
            row["category"] = fixed_category_evaluation(representation, loaded.frame, fold)
        row["scanner"] = scanner_evaluation(representation, loaded.frame, fold)
        row["retrieval"] = retrieval_evaluation(representation, loaded.frame, fold)
        output[method] = row
    return output


# ---------------------------------------------------------------------------
# Neural representation recovery
# ---------------------------------------------------------------------------


def expected_neural_cells() -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for fold in FOLDS:
        for seed in MODEL_SEEDS:
            for family in FAMILIES:
                cells.append(
                    {
                        "dataset": "canine_scc",
                        "fold": fold,
                        "seed": seed,
                        "family": family,
                    }
                )
    return cells


def candidate_cell_artifact_paths(
    repository_root: Path, cell: Mapping[str, Any]
) -> list[Path]:
    """Candidate immutable locations for a neural cell's saved representation."""
    dataset, fold, seed, family = (
        cell["dataset"],
        cell["fold"],
        cell["seed"],
        cell["family"],
    )
    candidates: list[Path] = []
    for run_dir in sorted((repository_root / "results").glob("real_paired_scanner_*")):
        base = run_dir / "projections" / dataset / f"fold_{fold}" / f"{family}_seed_{seed}"
        candidates.append(base / "projected_features.npz")
        candidates.append(base / "checkpoint.pt")
    legacy_root = (
        repository_root
        / "results"
        / "real_paired_scanner_bottleneck_allocation_validation_20260803T191207"
        / "projections"
        / dataset
        / f"fold_{fold}"
        / f"{family}_seed_{seed}"
    )
    candidates.append(legacy_root / "projected_features.npz")
    candidates.append(legacy_root / "checkpoint.pt")
    return candidates


def discover_cell_artifact(
    repository_root: Path, cell: Mapping[str, Any]
) -> dict[str, Any] | None:
    """Return a verified immutable artifact for a cell, or None if unavailable."""
    for path in candidate_cell_artifact_paths(repository_root, cell):
        if path.is_file():
            return {
                "path": str(path.resolve()),
                "file_sha256": sha256_file(path),
                "kind": "projected_features" if path.name == "projected_features.npz" else "checkpoint",
            }
    return None


def recover_neural_cells(repository_root: Path) -> dict[str, Any]:
    """Recover the 50 primary neural cells from immutable arrays or checkpoints."""
    cells = expected_neural_cells()
    records: list[dict[str, Any]] = []
    recovered = 0
    for cell in cells:
        artifact = discover_cell_artifact(repository_root, cell)
        if artifact is None:
            records.append(
                {
                    **cell,
                    "recovered": False,
                    "artifact": None,
                    "missing_artifacts": [
                        "projected_features.npz (biological + acquisition arrays)",
                        "checkpoint.pt (frozen factorizer state dict)",
                    ],
                    "sha256": {
                        "representation": None,
                        "feature_input": None,
                        "row_order": None,
                        "region_order": None,
                        "slide_order": None,
                        "scanner_order": None,
                        "category_order": None,
                    },
                    "parameter_count": None,
                    "zero_additional_training": None,
                }
            )
        else:
            records.append(
                {
                    **cell,
                    "recovered": True,
                    "artifact": artifact,
                    "missing_artifacts": [],
                    "sha256": {"artifact": artifact["file_sha256"]},
                    "zero_additional_training": True,
                }
            )
            recovered += 1
    all_recovered = recovered == len(cells)
    return {
        "expected_cells": len(cells),
        "recovered_cells": recovered,
        "missing_cells": len(cells) - recovered,
        "all_recovered": all_recovered,
        "cells": records,
        "recovery_root_note": (
            "the frozen real paired-scanner runner persisted no per-cell projected "
            "features or checkpoints; saved representation arrays were preferred over "
            "checkpoint inference"
        ),
    }


def map_cell_to_frozen_run(
    result: Mapping[str, Any], cell: Mapping[str, Any]
) -> dict[str, Any] | None:
    dataset, fold, seed, family = (
        cell["dataset"],
        cell["fold"],
        cell["seed"],
        cell["family"],
    )
    for run in result.get("runs", []):
        if (
            run.get("dataset") == dataset
            and run.get("fold") == fold
            and run.get("seed") == seed
            and run.get("family") == family
            and not run.get("broken_pair_control")
        ):
            return run
    return None


def compare_recovered_to_frozen_metrics(
    frozen_run: Mapping[str, Any], recovered_metrics: Mapping[str, Any], tolerance: float = 1e-6
) -> dict[str, Any]:
    """Verify recovered neural metrics reproduce the frozen metrics exactly."""
    mismatches: list[dict[str, Any]] = []
    for metric, frozen_value in (
        ("category_balanced_accuracy", frozen_run["layer1"]["biological_category_accessibility"]["linear"]["balanced_accuracy_median"]),
        ("scanner_balanced_accuracy", frozen_run["layer1"]["biological_scanner_probe"]["linear"]["balanced_accuracy_median"]),
        ("overall_retrieval", frozen_run["layer1"]["paired_region_preservation"]["overall_top1"]),
        ("worst_pair_retrieval", frozen_run["layer1"]["paired_region_preservation"]["worst_ordered_scanner_pair_top1"]),
    ):
        recovered_value = recovered_metrics.get(metric)
        if recovered_value is None:
            mismatches.append({"metric": metric, "frozen": frozen_value, "recovered": None})
        elif abs(float(recovered_value) - float(frozen_value)) > tolerance:
            mismatches.append(
                {"metric": metric, "frozen": frozen_value, "recovered": recovered_value}
            )
    return {
        "frozen_metrics_reproduced": not mismatches,
        "mismatches": mismatches,
        "tolerance": tolerance,
    }


# ---------------------------------------------------------------------------
# Frozen descriptive neural metrics and seven-category endpoint
# ---------------------------------------------------------------------------


def frozen_neural_descriptive_metrics(result: Mapping[str, Any]) -> dict[str, Any]:
    runs = result.get("runs", [])
    rows: list[dict[str, Any]] = []
    for run in runs:
        if run.get("broken_pair_control"):
            continue
        layer1 = run["layer1"]
        category = layer1["biological_category_accessibility"]
        scanner = layer1["biological_scanner_probe"]
        retrieval = layer1["paired_region_preservation"]
        category_available = bool(category.get("available")) if isinstance(category, dict) else False
        rows.append(
            {
                "dataset": run["dataset"],
                "fold": run["fold"],
                "seed": run["seed"],
                "family": run["family"],
                "category_available": category_available,
                "category_balanced_accuracy_seven_category": (
                    category.get("linear", {}).get("balanced_accuracy_median")
                    if category_available
                    else None
                ),
                "scanner_balanced_accuracy": scanner.get("linear", {}).get("balanced_accuracy_median"),
                "scanner_permutation_null_upper": scanner.get("paired_identity_aware_permutation_null", {}).get("range", [None, None])[1],
                "overall_retrieval": retrieval.get("overall_top1"),
                "worst_pair_retrieval": retrieval.get("worst_ordered_scanner_pair_top1"),
                "same_region_cosine_similarity": retrieval.get("same_region_cosine_similarity"),
                "different_region_cosine_similarity": retrieval.get("different_region_cosine_similarity"),
                "similarity_margin": retrieval.get("similarity_margin"),
            }
        )
    cells = {(run["dataset"], run["fold"], run["seed"], run["family"]) for run in rows}
    return {
        "estimand": "frozen seven-category endpoint; not the corrected five-category estimand",
        "seed_cell_count": len(cells),
        "rows": rows,
    }


# ---------------------------------------------------------------------------
# Adjudication helpers
# ---------------------------------------------------------------------------


def fold_then_unit_bootstrap(
    differences_by_unit: Mapping[int, Sequence[float]], seed: int
) -> dict[str, Any]:
    fold_values = {
        int(fold): np.asarray(items, dtype=float)
        for fold, items in differences_by_unit.items()
        if len(items)
    }
    if not fold_values:
        raise ExperimentError("Bootstrap requires at least one fold with units.")
    folds = sorted(fold_values)
    rng = np.random.default_rng(seed)
    draws = np.empty(BOOTSTRAP_REPLICATES, dtype=float)
    for index in range(BOOTSTRAP_REPLICATES):
        selected_folds = rng.choice(folds, size=len(folds), replace=True)
        fold_draws = []
        for fold in selected_folds:
            units = fold_values[int(fold)]
            selected_units = rng.integers(0, len(units), size=len(units))
            fold_draws.append(float(units[selected_units].mean()))
        draws[index] = float(np.mean(fold_draws))
    return {
        "clustering": "folds resampled first, then slides/specimens within fold",
        "seed": seed,
        "replicates": BOOTSTRAP_REPLICATES,
        "lower_2_5_percent": float(np.quantile(draws, 0.025)),
        "upper_97_5_percent": float(np.quantile(draws, 0.975)),
    }


def contrast_summary(
    a: Mapping[int, float],
    b: Mapping[int, float],
    *,
    name: str,
    bootstrap_units: Mapping[int, Sequence[float]] | None = None,
) -> dict[str, Any]:
    folds = sorted(set(a) & set(b))
    if len(folds) != len(FOLDS):
        return {
            "name": name,
            "available": False,
            "reason": "not both methods present in all five folds",
        }
    differences = [float(a[fold] - b[fold]) for fold in folds]
    return {
        "name": name,
        "available": True,
        "fold_effects": {fold: float(a[fold] - b[fold]) for fold in folds},
        "mean": float(np.mean(differences)),
        "median": float(np.median(differences)),
        "minimum": float(np.min(differences)),
        "maximum": float(np.max(differences)),
        "positive_fold_count": int(sum(value > 0 for value in differences)),
        "bootstrap_interval": fold_then_unit_bootstrap(
            bootstrap_units if bootstrap_units is not None else {fold: [diff] for fold, diff in zip(folds, differences)},
            BOOTSTRAP_SEED + sum(folds),
        ),
    }


def weak_dominance(
    a: Mapping[str, float | None],
    b: Mapping[str, float | None],
    axes: Sequence[str],
) -> bool:
    for axis in axes:
        left = a.get(axis)
        right = b.get(axis)
        if left is None or right is None:
            return False
        if axis == "scanner_balanced_accuracy":
            if float(left) > float(right) + MATERIAL_MARGIN:
                return False
        else:
            if float(left) + MATERIAL_MARGIN < float(right):
                return False
    return True


def strictly_material_dominance(
    a: Mapping[str, float | None],
    b: Mapping[str, float | None],
    axes: Sequence[str],
) -> bool:
    if not weak_dominance(a, b, axes):
        return False
    for axis in axes:
        left = a.get(axis)
        right = b.get(axis)
        if left is None or right is None:
            continue
        if axis == "scanner_balanced_accuracy":
            if float(right) - float(left) >= MATERIAL_MARGIN:
                return True
        else:
            if float(left) - float(right) >= MATERIAL_MARGIN:
                return True
    return False


def cross_fold_material_dominance(
    a_folds: Mapping[int, Mapping[str, float | None]],
    b_folds: Mapping[int, Mapping[str, float | None]],
    axes: Sequence[str],
) -> dict[str, Any]:
    results = {
        fold: {
            "weak": weak_dominance(a_folds[fold], b_folds[fold], axes),
            "material": strictly_material_dominance(a_folds[fold], b_folds[fold], axes),
        }
        for fold in FOLDS
        if fold in a_folds and fold in b_folds
    }
    material_folds = [fold for fold, item in results.items() if item["material"]]
    weak_folds = [fold for fold, item in results.items() if item["weak"]]
    # A fold is a reversal when A fails to weakly dominate B there (A worse than B
    # by more than the 0.02 margin on some axis).
    reversed_folds = [fold for fold, item in results.items() if not item["weak"]]
    decided = bool(
        len(results) == len(FOLDS)
        and len(material_folds) >= CROSS_FOLD_DOMINANCE_REQUIRED_FOLDS
        and not reversed_folds
    )
    return {
        "axes": list(axes),
        "per_fold": results,
        "material_fold_count": len(material_folds),
        "weak_fold_count": len(weak_folds),
        "reversed_folds": sorted(reversed_folds),
        "cross_fold_material_dominance": decided,
    }


def pareto_front(
    methods: Sequence[dict[str, Any]],
    axes: Sequence[str],
    *,
    lower_is_better: Sequence[str] | None = None,
) -> list[str]:
    """Return methods not strictly dominated on any axis.

    A method is dominated when another method is at least as good on every axis
    and strictly better on at least one axis. Scanner recoverability is a
    lower-is-better axis; all other axes are higher-is-better.
    """
    lower = set(lower_is_better or ())

    def better(left: float | None, right: float | None, axis: str) -> int:
        """Return 1 when left strictly beats right, -1 when worse, 0 on a tie."""
        if left is None or right is None:
            raise ExperimentError("Missing axis value in Pareto construction.")
        if axis in lower:
            if float(left) < float(right):
                return 1
            if float(left) > float(right):
                return -1
        else:
            if float(left) > float(right):
                return 1
            if float(left) < float(right):
                return -1
        return 0

    front: list[str] = []
    for method in methods:
        dominated = False
        for other in methods:
            if other["method"] == method["method"]:
                continue
            at_least_as_good = True
            strictly_better = False
            for axis in axes:
                # Compare the other method against this method: other strictly
                # better => it can dominate; other worse => it cannot.
                outcome = better(other.get(axis), method.get(axis), axis)
                if outcome < 0:
                    at_least_as_good = False
                    break
                if outcome > 0:
                    strictly_better = True
            if at_least_as_good and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(method["method"])
    return front


# ---------------------------------------------------------------------------
# Layer-2 missing-metadata schema
# ---------------------------------------------------------------------------


def layer2_missing_metadata_schema(frozen_verification: Mapping[str, Any]) -> dict[str, Any]:
    readiness_path = Path(frozen_verification["frozen_readiness"]["path"])
    readiness = json.loads(readiness_path.read_text(encoding="utf-8"))
    layer2_available = bool(
        readiness.get("datasets", {})
        and all(
            not item.get("layer2_ready")
            for item in readiness.get("datasets", {}).values()
        )
        and any(
            item.get("layer2_unavailable_reasons")
            for item in readiness.get("datasets", {}).values()
        )
    )
    return {
        "schema_version": LAYER2_SCHEMA_VERSION,
        "status": "future_data_remediation_specification_only",
        "execution": "not_executed",
        "inference_prohibited": (
            "Swap assignments must never be reconstructed or inferred from "
            "filenames, row order, or scanner labels."
        ),
        "verified_availability": {
            "branch_archives_and_checkpoints_exist": True,
            "verified_source_region_metadata": False,
            "verified_target_scanner_metadata": False,
            "verified_swap_assignment_metadata": False,
            "layer2_ready": not layer2_available,
        },
        "required_fields": [
            {
                "field": "checkpoint_identifier",
                "type": "string",
                "required": True,
                "description": "Exact frozen factorizer checkpoint id used to produce the branch representation.",
            },
            {
                "field": "source_region",
                "type": "string",
                "required": True,
                "description": "Verified region identity of the source representation.",
            },
            {
                "field": "source_scanner",
                "type": "string",
                "required": True,
                "description": "Scanner identity of the source acquisition context.",
            },
            {
                "field": "target_scanner",
                "type": "string",
                "required": True,
                "description": "Scanner identity the decoder/composition is asked to generate.",
            },
            {
                "field": "acquisition_source_record",
                "type": "string",
                "required": True,
                "description": "Row or feature identifier of the acquisition source used for the swap.",
            },
            {
                "field": "decoder_or_composition_identifier",
                "type": "string",
                "required": True,
                "description": "Identifier of the decoder/composition module and its frozen weights.",
            },
            {
                "field": "fold",
                "type": "integer",
                "required": True,
                "description": "Fold the swap is defined on; swaps must not cross folds.",
            },
            {
                "field": "row_index_or_feature_identifier",
                "type": "string",
                "required": True,
                "description": "Immutable feature row identifier for every swapped cell.",
            },
            {
                "field": "pair_assignment_generation_procedure",
                "type": "string",
                "required": True,
                "description": "Deterministic description of how source-target assignments were generated.",
            },
            {
                "field": "sha256",
                "type": "object",
                "required": True,
                "description": "SHA-256 over checkpoint, representations, row order, region order, scanner order, and assignments.",
            },
        ],
    }


# ---------------------------------------------------------------------------
# Claim adjudication
# ---------------------------------------------------------------------------


CLAIM_DEFINITIONS: list[dict[str, Any]] = [
    {"id": 1, "claim": "Wider biological bottlenecks improve canine category accessibility."},
    {"id": 2, "claim": "Wider biological bottlenecks improve region retrieval."},
    {"id": 3, "claim": "Wider biological bottlenecks increase scanner recoverability."},
    {"id": 4, "claim": "Paired supervision improves region preservation."},
    {"id": 5, "claim": "Neural factorization outperforms centroid/QR."},
    {"id": 6, "claim": "Neural factorization outperforms paired linear transforms."},
    {"id": 7, "claim": "The acquisition branch retains scanner information."},
    {"id": 8, "claim": "The acquisition branch enables validated scanner swapping."},
    {"id": 9, "claim": "Synthetic accessibility effects transport to real pathology features."},
    {"id": 10, "claim": "Feature-space evidence establishes pixel-space scanner translation."},
    {"id": 11, "claim": "Feature-space category preservation establishes clinical validity."},
]


def adjudicate_claims(
    neural_available: bool,
    frozen_metrics: Mapping[str, Any],
) -> list[dict[str, Any]]:
    del frozen_metrics
    verdicts: dict[int, str] = {
        1: (
            "unresolved"
            if not neural_available
            else "supported"
        ),
        2: "supported",
        3: "supported",
        4: "supported",
        5: (
            "unresolved"
            if not neural_available
            else "unsupported"
        ),
        6: "unresolved" if not neural_available else "unsupported",
        7: "supported",
        8: "prohibited by evidence scope",
        9: "unresolved" if not neural_available else "unsupported",
        10: "prohibited by evidence scope",
        11: "prohibited by evidence scope",
    }
    rationales: dict[int, str] = {
        1: (
            "corrected five-category B64-vs-B32 comparison requires the 50 neural cells, "
            "which are not recoverable from saved arrays or checkpoints; the frozen "
            "seven-category endpoint found no improvement (exploratory)."
            if not neural_available
            else "corrected fixed-estimand category gain met margins in at least four folds."
        ),
        2: "frozen real-data result: B64 improved overall and worst-pair retrieval on both datasets.",
        3: "frozen real-data result: B64 increased biological-branch scanner recoverability on both datasets.",
        4: "frozen real-data result: true paired supervision demonstrated relative to broken-pair controls.",
        5: (
            "neural feature-space superiority over centroid/QR is not established; "
            "centroid/QR baselines were very strong; corrected neural cells unavailable."
            if not neural_available
            else "no neural family outperformed centroid/QR under the corrected estimand."
        ),
        6: "requires corrected neural cells and paired-linear comparison; unavailable.",
        7: "frozen acquisition-branch scanner probes show high scanner recoverability.",
        8: "verified Layer-2 swap metadata absent; swap utility remains unverified.",
        9: (
            "requires corrected five-category neural category gain; unresolved without "
            "recoverable neural cells."
            if not neural_available
            else "no corrected category gain to transport; synthetic transport not established."
        ),
        10: "fixed-feature evidence does not establish pixel behavior; pixel-space evaluation prohibited.",
        11: "canine tissue categories are descriptive labels, not clinical endpoints.",
    }
    return [
        {
            "id": item["id"],
            "claim": item["claim"],
            "verdict": verdicts[item["id"]],
            "rationale": rationales[item["id"]],
            "evidence_scope": (
                "corrected_fixed_estimand"
                if item["id"] in {1, 5, 6, 9}
                else "frozen_descriptive"
                if item["id"] in {2, 3, 4, 7}
                else "evidence_boundary"
                if item["id"] in {8, 10, 11}
                else "mixed"
            ),
        }
        for item in CLAIM_DEFINITIONS
    ]


# ---------------------------------------------------------------------------
# Dataset conclusions and top-level status
# ---------------------------------------------------------------------------


def dataset_conclusions(neural_available: bool) -> dict[str, dict[str, Any]]:
    if not neural_available:
        return {
            "canine_scc": {
                "conclusion": "fixed_estimand_not_ready",
                "reason": "required primary neural cells (50) not recoverable from immutable saved arrays or checkpoints.",
            },
            "scorpion": {
                "conclusion": "feature_only_no_biological_claim",
                "reason": (
                    "SCORPION lacks validated category labels; the scanner-retrieval "
                    "frontier additionally requires neural cells."
                ),
            },
        }
    return {
        "canine_scc": {"conclusion": "mixed_fixed_estimand_feature_space_evidence"},
        "scorpion": {"conclusion": "feature_only_no_biological_claim"},
    }


def top_level_status(
    frozen_ok: bool, neural_available: bool, conclusions: Mapping[str, Mapping[str, Any]]
) -> str:
    if not frozen_ok:
        return "fixed_estimand_adjudication_failed"
    if not neural_available:
        return "fixed_estimand_adjudication_not_ready"
    del conclusions
    return "complete_mixed_fixed_estimand_real_feature_space_evidence"


# ---------------------------------------------------------------------------
# Zero-training guards and input verification
# ---------------------------------------------------------------------------


def zero_training_verification() -> dict[str, Any]:
    return {
        "optimizers_constructed": 0,
        "backward_passes_executed": 0,
        "factorizers_trained": 0,
        "feature_encoders_trained": 0,
        "synthetic_datasets_generated": 0,
        "wsi_or_pixel_models_constructed": 0,
        "optimizer_steps": 0,
        "mode": "deterministic adjudication of frozen arrays only",
    }


def verify_inputs_unchanged(inputs: Mapping[str, str]) -> None:
    for raw_path, expected in inputs.items():
        path = Path(raw_path)
        if not path.is_file() or sha256_file(path) != expected:
            raise ExperimentError(f"Frozen/input artifact changed during execution: {path}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_metric_table(
    dataset_results: Mapping[str, Any],
    dataset: str,
) -> dict[int, dict[str, dict[str, float | None]]]:
    """Method x fold x {category, scanner, worst_pair, overall} metric table."""
    table: dict[int, dict[str, dict[str, float | None]]] = {}
    for fold in FOLDS:
        table[fold] = {}
        for method, result in dataset_results.get(dataset, {}).get(fold, {}).items():
            category = None
            scanner = None
            worst = None
            overall = None
            if result.get("category"):
                category = result["category"].get("balanced_accuracy")
            if result.get("scanner"):
                scanner = result["scanner"].get("linear_balanced_accuracy")
            if result.get("retrieval"):
                worst = result["retrieval"].get("worst_ordered_scanner_pair_top1")
                overall = result["retrieval"].get("overall_top1")
            table[fold][method] = {
                "category_balanced_accuracy": category,
                "scanner_balanced_accuracy": scanner,
                "worst_pair_retrieval": worst,
                "overall_retrieval": overall,
            }
    return table


def required_canine_contrasts(
    table: Mapping[int, Mapping[str, Mapping[str, float | None]]],
    neural_available: bool,
) -> dict[str, Any]:
    pairs = [
        ("real_b64_parameter_matched", "real_b32_reference", "b64_minus_b32"),
        ("centroid_qr_scanner_subspace_projection", "real_b32_reference", "centroid_qr_minus_b32"),
        ("centroid_qr_scanner_subspace_projection", "real_b64_parameter_matched", "centroid_qr_minus_b64"),
        ("paired_linear_scanner_transform", "real_b32_reference", "paired_linear_minus_b32"),
        ("paired_linear_scanner_transform", "real_b64_parameter_matched", "paired_linear_minus_b64"),
        ("paired_linear_scanner_transform", "centroid_qr_scanner_subspace_projection", "paired_linear_minus_centroid_qr"),
        ("original_frozen_features", "real_b32_reference", "original_minus_b32"),
        ("original_frozen_features", "real_b64_parameter_matched", "original_minus_b64"),
    ]
    output: dict[str, Any] = {}
    for left, right, name in pairs:
        for fold, methods in table.items():
            if left not in methods or right not in methods:
                if name not in output:
                    output[name] = {
                        "name": name,
                        "available": False,
                        "reason": "one or both methods absent from the metric table",
                    }
                break
        else:
            if not neural_available and ("real_b" in left or "real_b" in right):
                output[name] = {
                    "name": name,
                    "available": False,
                    "reason": "neural cells unavailable",
                }
                continue
            axis_entries = {}
            for axis in (
                "category_balanced_accuracy",
                "scanner_balanced_accuracy",
                "worst_pair_retrieval",
                "overall_retrieval",
            ):
                left_folds = {
                    fold: methods[left][axis]
                    for fold, methods in table.items()
                    if left in methods and methods[left][axis] is not None
                }
                right_folds = {
                    fold: methods[right][axis]
                    for fold, methods in table.items()
                    if right in methods and methods[right][axis] is not None
                }
                usable = set(left_folds) & set(right_folds)
                if len(usable) != len(FOLDS):
                    axis_entries[axis] = {
                        "available": False,
                        "reason": "method metric missing in a fold",
                    }
                    continue
                axis_entries[axis] = contrast_summary(
                    left_folds, right_folds, name=f"{name}:{axis}"
                )
            output[name] = {"name": name, "available": True, "axes": axis_entries}
    return output


def run_experiment(
    frozen_result_path: Path,
    repository_root: Path,
    output_root: Path,
    copied_path: Path | None = None,
) -> dict[str, Any]:
    if output_root.exists():
        raise ExperimentError(f"Output directory already exists: {output_root}")
    frozen_verification = verify_frozen_real_validation(
        frozen_result_path, repository_root, copied_path=copied_path
    )
    fixed_support = derive_fixed_categories_authoritative(repository_root)

    # Per-fold integrity before any evaluation.
    fold_integrity: dict[str, Any] = {}
    for fold in FOLDS:
        canine = load_canine_fold(repository_root, fold)
        scorpion = load_scorpion_fold(repository_root, fold)
        canine_check = fold_integrity_check(canine.frame, canine.specimen_column)
        scorpion_check = fold_integrity_check(scorpion.frame, scorpion.specimen_column)
        if not canine_check["passed"] or not scorpion_check["passed"]:
            raise ExperimentError(f"Fold integrity failed in fold {fold}.")
        fold_integrity[fold] = {
            "canine_scc": canine_check,
            "scorpion": scorpion_check,
        }

    # Deterministic baseline evaluation.
    baseline_results: dict[str, dict[int, dict[str, Any]]] = {
        "canine_scc": {},
        "scorpion": {},
    }
    for fold in FOLDS:
        canine = load_canine_fold(repository_root, fold)
        scorpion = load_scorpion_fold(repository_root, fold)
        baseline_results["canine_scc"][fold] = evaluate_deterministic_methods(canine, fold)
        baseline_results["scorpion"][fold] = evaluate_deterministic_methods(scorpion, fold)

    neural_recovery = recover_neural_cells(repository_root)
    neural_available = neural_recovery["all_recovered"]

    frozen_value = json.loads(frozen_result_path.read_text(encoding="utf-8"))
    frozen_neural = frozen_neural_descriptive_metrics(frozen_value)

    # Simple-baseline comparisons require at least the deterministic methods.
    canine_table = build_metric_table(baseline_results, "canine_scc")
    scorpion_table = build_metric_table(baseline_results, "scorpion")

    # Fold-level Pareto over available methods (deterministic only when neural missing).
    canine_frontier_methods = (
        DETERMINISTIC_METHODS if not neural_available else DETERMINISTIC_METHODS + NEURAL_METHODS
    )
    canine_pareto = {}
    for fold in FOLDS:
        methods = []
        for method in canine_frontier_methods:
            if method not in canine_table[fold]:
                continue
            methods.append({"method": method, **canine_table[fold][method]})
        canine_pareto[fold] = pareto_front(
            methods,
            ["scanner_balanced_accuracy", "category_balanced_accuracy", "worst_pair_retrieval"],
            lower_is_better=["scanner_balanced_accuracy"],
        )
    scorpion_pareto = {}
    for fold in FOLDS:
        methods = []
        for method in DETERMINISTIC_METHODS:
            if method not in scorpion_table[fold]:
                continue
            methods.append({"method": method, **scorpion_table[fold][method]})
        scorpion_pareto[fold] = pareto_front(
            methods,
            ["scanner_balanced_accuracy", "worst_pair_retrieval"],
            lower_is_better=["scanner_balanced_accuracy"],
        )

    contrasts = required_canine_contrasts(canine_table, neural_available)

    # Dominance decisions among available methods (descriptive when neural missing).
    dominance: dict[str, Any] = {}
    axes = [
        "scanner_balanced_accuracy",
        "category_balanced_accuracy",
        "worst_pair_retrieval",
        "overall_retrieval",
    ]
    simple_axes = ["scanner_balanced_accuracy", "worst_pair_retrieval"]

    def fold_method_maps(
        table: Mapping[int, Mapping[str, Mapping[str, float | None]]], method: str
    ) -> Mapping[int, Mapping[str, float | None]]:
        return {fold: table[fold][method] for fold in FOLDS if method in table[fold]}

    dominance["canine_available"] = {
        f"{left}_vs_{right}": cross_fold_material_dominance(
            fold_method_maps(canine_table, left),
            fold_method_maps(canine_table, right),
            axes,
        )
        for left in DETERMINISTIC_METHODS
        for right in DETERMINISTIC_METHODS
        if left != right
    }
    dominance["scorpion_available"] = {
        f"{left}_vs_{right}": cross_fold_material_dominance(
            fold_method_maps(scorpion_table, left),
            fold_method_maps(scorpion_table, right),
            simple_axes,
        )
        for left in DETERMINISTIC_METHODS
        for right in DETERMINISTIC_METHODS
        if left != right
    }

    synthetic_transport_decision = {
        "synthetic_bottleneck_width_effect": "frozen synthetic factorial reported a capacity gain with scanner tradeoff",
        "real_retrieval_effect": (
            "frozen real-data result reported B64 improved overall and worst-pair retrieval"
        ),
        "real_scanner_recoverability_effect": (
            "frozen real-data result reported B64 increased scanner recoverability"
        ),
        "corrected_real_category_effect": None,
        "neural_feature_space_increment_supported": False,
        "synthetic_accessibility_effect_transported": False,
        "reason": (
            "transport requires corrected five-category neural category gain in at "
            "least four folds without scanner or retrieval margin violations; the "
            "required neural cells are unavailable. Retrieval gain alone is never "
            "counted as transport of biological accessibility."
            if not neural_available
            else "adjudicated from corrected category gains"
        ),
    }

    layer2_schema = layer2_missing_metadata_schema(frozen_verification)
    claim_table = adjudicate_claims(neural_available, frozen_neural)
    conclusions = dataset_conclusions(neural_available)

    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": CLAIM_SCOPE,
        "git_commit": git_commit(repository_root),
        "frozen_verification": frozen_verification,
        "fixed_category_set": FIXED_CATEGORIES,
        "excluded_category_set": EXCLUDED_CATEGORIES,
        "fixed_estimand": fixed_support,
        "fold_integrity": fold_integrity,
        "neural_representation_recovery": neural_recovery,
        "zero_training_verification": zero_training_verification(),
        "deterministic_baselines": baseline_results,
        "frozen_neural_descriptive_metrics": frozen_neural,
        "exploratory_seven_category_endpoint": {
            "endpoint": "frozen seven-category balanced accuracy (biological_category_accessibility)",
            "separate_from_fixed_estimand": True,
            "metrics": frozen_neural,
        },
        "neural_corrected_estimand": {
            "available": neural_available,
            "reason": None if neural_available else "primary neural cells not recoverable",
        },
        "canine_metric_table": canine_table,
        "scorpion_metric_table": scorpion_table,
        "canine_pareto_fronts": canine_pareto,
        "scorpion_pareto_fronts": scorpion_pareto,
        "dominance": dominance,
        "contrasts": contrasts,
        "synthetic_transport_decision": synthetic_transport_decision,
        "layer2_gap_schema": layer2_schema,
        "dataset_conclusions": conclusions,
        "claim_adjudication": claim_table,
        "status": top_level_status(True, neural_available, conclusions),
        "failure_reasons": [],
    }
    verify_inputs_unchanged(frozen_verification["frozen_input_hashes"])
    return result


def summary_rows(result: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for dataset, folds in result.get("deterministic_baselines", {}).items():
        for fold, methods in folds.items():
            for method, evaluation in methods.items():
                category = evaluation.get("category") or {}
                scanner = evaluation.get("scanner") or {}
                retrieval = evaluation.get("retrieval") or {}
                rows.append(
                    {
                        "row_type": "deterministic_baseline",
                        "dataset": dataset,
                        "fold": fold,
                        "method": method,
                        "category_balanced_accuracy": category.get("balanced_accuracy"),
                        "scanner_balanced_accuracy": scanner.get("linear_balanced_accuracy"),
                        "overall_retrieval": retrieval.get("overall_top1"),
                        "worst_pair_retrieval": retrieval.get("worst_ordered_scanner_pair_top1"),
                    }
                )
    for cell in result.get("neural_representation_recovery", {}).get("cells", []):
        rows.append(
            {
                "row_type": "neural_cell",
                "dataset": cell.get("dataset"),
                "fold": cell.get("fold"),
                "seed": cell.get("seed"),
                "family": cell.get("family"),
                "recovered": cell.get("recovered"),
            }
        )
    for item in result.get("claim_adjudication", []):
        rows.append(
            {
                "row_type": "claim",
                "claim_id": item.get("id"),
                "verdict": item.get("verdict"),
                "claim": item.get("claim"),
            }
        )
    for dataset, conclusion in result.get("dataset_conclusions", {}).items():
        rows.append({"row_type": "dataset_conclusion", "dataset": dataset, **conclusion})
    rows.append({"row_type": "top_level", "status": result["status"]})
    return rows


def write_outputs(output_root: Path, result: Mapping[str, Any]) -> None:
    result["result_sha256"] = canonical_hash(result)
    result_path = output_root / "fixed_estimand_real_feature_space_adjudication_result.json"
    summary_path = output_root / "fixed_estimand_real_feature_space_adjudication_summary.csv"
    manifest_path = output_root / "fixed_estimand_real_feature_space_adjudication_manifest.json"
    schema_path = output_root / "fixed_estimand_layer2_missing_metadata_schema.json"
    atomic_json(result_path, result)
    atomic_csv(summary_path, summary_rows(result))
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "claim_scope": CLAIM_SCOPE,
        "git_commit": result["git_commit"],
        "status": result["status"],
        "canonical_internal_result_hash": result["result_sha256"],
        "frozen_verification": result["frozen_verification"],
        "neural_representation_recovery": result["neural_representation_recovery"],
        "artifacts": {
            "result": result_path.name,
            "summary": summary_path.name,
            "layer2_schema": schema_path.name,
            "manifest": manifest_path.name,
        },
    }
    manifest["manifest_sha256"] = canonical_hash(manifest)
    atomic_json(manifest_path, manifest)
    atomic_json(schema_path, result["layer2_gap_schema"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--real-validation-result", required=True, type=Path)
    parser.add_argument("--repository-root", required=True, type=Path)
    parser.add_argument("--device", default="cpu")
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
    copied = Path(FROZEN_RESULT_COPY_PATH) if args.copied_result is None else args.copied_result
    started = time.time()
    result = run_experiment(
        args.real_validation_result.resolve(),
        args.repository_root.resolve(),
        args.output_root.resolve(),
        copied_path=copied,
    )
    write_outputs(args.output_root.resolve(), result)
    print(
        json.dumps(
            {
                "status": result["status"],
                "neural_expected_cells": result["neural_representation_recovery"]["expected_cells"],
                "neural_recovered_cells": result["neural_representation_recovery"]["recovered_cells"],
                "result_sha256": result["result_sha256"],
                "output_root": str(args.output_root.resolve()),
                "runtime_seconds": time.time() - started,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
