#!/usr/bin/env python3
"""Old-style residual branch-separation audit.

Tests whether the old-style scanner-centroid/QR residual component behaves like
a clean scanner branch, or whether it also carries biological category signal.

The audit compares:
  - paired-acquisition biological/acquisition branches
  - oldstyle keep/removed branches from scanner centroid directions

Bounded interpretation only: this is a representation-probe audit for the
Canine SCC DINOv2 feature stack.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.canine import run_pair_integrity_falsification_caninescc as canine_pair  # noqa: E402
from experiments.external_multiscanner import run_canine_pathoalign_crossfold as canine_cross  # noqa: E402
from experiments.scorpion import run_pathoalign_projection as projection  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------


BRANCH = "experiment/oldstyle-residual-branch-separation-audit"
SCANNERS = ("cs2", "gt450", "nz20", "nz210", "p1000")
FOLDS = (0, 1, 2, 3, 4)
PAIR_INTEGRITY_SEEDS = (911, 912, 913, 914, 915)
K_VALUES = (1, 2, 3, 4)
NEIGHBORHOOD_K = (1, 5, 10)
OLDSTYLE_K4_SCANNER_REFERENCE = 0.2000
OLDSTYLE_K4_SCANNER_TOLERANCE = 0.05

FEATURE_PATH = Path("results/external_multiscanner_caninescc/features/fold_0_dinov2_base.npz")
MANIFESTS_DIR = Path("data/external_multiscanner_caninescc/patch_manifests/splits")
PAIR_INTEGRITY_DIR = Path("results/paired_acquisition_factorization_pair_integrity_caninescc")

EXPECTED_REPRESENTATIONS = (
    "original_frozen_features",
    "true_pair_biological",
    "true_pair_acquisition",
    "shuffled_sample_biological",
    "shuffled_sample_acquisition",
    "oldstyle_keep_k1",
    "oldstyle_removed_k1",
    "oldstyle_keep_k2",
    "oldstyle_removed_k2",
    "oldstyle_keep_k3",
    "oldstyle_removed_k3",
    "oldstyle_keep_k4",
    "oldstyle_removed_k4",
)

METRIC_COLUMNS = (
    "scanner_balanced_accuracy",
    "scanner_macro_f1",
    "category_balanced_accuracy",
    "category_macro_f1",
    "category_weighted_f1",
    "same_category_purity_k1",
    "same_category_purity_k5",
    "same_category_purity_k10",
    "same_sample_top1_retrieval",
    "scanner_category_ratio",
    "category_scanner_ratio",
)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".csv", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def finite_or_raise(name: str, values: np.ndarray) -> None:
    if not np.isfinite(values).all():
        raise projection.ExperimentError(f"{name} produced nonfinite values.")


def safe_ratio(numerator: float, denominator: float) -> float:
    return float(numerator / max(1e-8, denominator))


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def manifest_path(fold: int) -> Path:
    return MANIFESTS_DIR / f"fold_{fold}_patch_manifest.csv"


def load_manifest(fold: int) -> pd.DataFrame:
    path = manifest_path(fold)
    if not path.is_file():
        raise projection.ExperimentError(f"Missing manifest: {path}")
    frame = pd.read_csv(path, dtype=str)
    required = {"slide_id", "sample_id", "region_id", "scanner_id", "category_name", "split"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise projection.ExperimentError(f"Manifest {path} missing columns: {missing}")
    frame["scanner_id"] = frame["scanner_id"].astype(str).str.lower()
    return frame


def load_frozen_features() -> tuple[np.ndarray, pd.DataFrame]:
    features, frame, _metadata = projection.load_archive(FEATURE_PATH)
    frame["scanner_id"] = frame["scanner_id"].astype(str).str.lower()
    if features.shape != (4025, 768):
        raise projection.ExperimentError(
            f"Expected canine DINOv2 features with shape (4025, 768); observed {features.shape}"
        )
    return features, frame


def align_pair_integrity_features(
    features: np.ndarray,
    pair_frame: pd.DataFrame,
    manifest: pd.DataFrame,
) -> np.ndarray:
    pair_frame = pair_frame.copy()
    pair_frame["scanner_id"] = pair_frame["scanner_id"].astype(str).str.lower()
    pair_keys = [
        (str(row.slide_id), str(row.region_id), str(row.scanner_id))
        for _, row in pair_frame.iterrows()
    ]
    manifest_keys = [
        (str(row.slide_id), str(row.region_id), str(row.scanner_id))
        for _, row in manifest.iterrows()
    ]
    if len(pair_keys) != len(manifest_keys):
        raise projection.ExperimentError(
            f"Pair features have {len(pair_keys)} rows but manifest has {len(manifest_keys)} rows."
        )
    lookup = {key: index for index, key in enumerate(pair_keys)}
    missing = [key for key in manifest_keys if key not in lookup]
    if missing:
        raise projection.ExperimentError(
            f"{len(missing)} manifest keys were absent from pair-integrity features."
        )
    order = np.asarray([lookup[key] for key in manifest_keys], dtype=np.int64)
    return features[order]


def load_pair_integrity_features(
    fold: int, seed: int, condition: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    projected = (
        PAIR_INTEGRITY_DIR
        / f"fold_{fold}"
        / "runs"
        / f"{condition}_seed_{seed}"
        / "projected_features.npz"
    )
    if not projected.is_file():
        raise projection.ExperimentError(f"Missing pair-integrity features: {projected}")
    biological, acquisition, frame, _metadata = canine_pair.load_projected(projected)
    return biological, acquisition, frame


# ---------------------------------------------------------------------------
# Old-style centroid/QR split
# ---------------------------------------------------------------------------


def oldstyle_scanner_centroid_directions(
    features: np.ndarray,
    manifest: pd.DataFrame,
    fit: np.ndarray,
) -> np.ndarray:
    """Compute scanner centroid offset directions from fit rows.

    Direction for scanner s:
        mean(X_fit[scanner == s]) - mean(X_fit)
    """
    matrix = np.asarray(features, dtype=np.float64)
    scanner_labels = manifest["scanner_id"].astype(str).to_numpy()[fit]
    scanners = sorted(set(scanner_labels))
    fit_matrix = matrix[fit]
    grand_mean = fit_matrix.mean(axis=0)
    directions = []
    for scanner in scanners:
        mask = scanner_labels == scanner
        if int(mask.sum()) > 0:
            directions.append(fit_matrix[mask].mean(axis=0) - grand_mean)
    if not directions:
        return np.zeros((0, matrix.shape[1]), dtype=np.float64)
    return np.stack(directions, axis=0).astype(np.float64)


def oldstyle_q_basis(directions: np.ndarray, k: int) -> tuple[np.ndarray, int]:
    """QR-orthonormalize the first k scanner centroid directions."""
    if k <= 0 or directions.shape[0] == 0:
        return np.zeros((0, directions.shape[1]), dtype=np.float64), 0
    effective_k = min(int(k), directions.shape[0])
    q_matrix, _ = np.linalg.qr(directions[:effective_k].T)
    basis = q_matrix.T.astype(np.float64)
    return basis, int(basis.shape[0])


def oldstyle_split_keep_removed(
    features: np.ndarray,
    directions: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Split features into oldstyle keep and removed components.

    With Q_k as the first k QR-orthonormalized scanner-centroid directions:
        oldstyle_removed_k = X @ Q_k.T @ Q_k
        oldstyle_keep_k = X - oldstyle_removed_k
    """
    matrix = np.asarray(features, dtype=np.float64)
    basis, effective_k = oldstyle_q_basis(directions, k)
    if effective_k <= 0:
        keep = matrix.astype(np.float32).copy()
        removed = np.zeros_like(keep, dtype=np.float32)
        return keep, removed, 0
    removed64 = (matrix @ basis.T) @ basis
    keep64 = matrix - removed64
    finite_or_raise(f"oldstyle_keep_k{k}", keep64)
    finite_or_raise(f"oldstyle_removed_k{k}", removed64)
    return keep64.astype(np.float32), removed64.astype(np.float32), effective_k


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def scanner_probe_metrics(
    features: np.ndarray,
    manifest: pd.DataFrame,
    fit: np.ndarray,
    test: np.ndarray,
) -> dict[str, float]:
    labels = manifest["scanner_id"].to_numpy()
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
            random_state=0,
            solver="lbfgs",
        ),
    )
    model.fit(features[fit], labels[fit])
    predictions = model.predict(features[test])
    return {
        "scanner_balanced_accuracy": float(balanced_accuracy_score(labels[test], predictions)),
        "scanner_macro_f1": float(f1_score(labels[test], predictions, average="macro")),
    }


def category_probe_metrics(
    features: np.ndarray,
    manifest: pd.DataFrame,
    fit: np.ndarray,
    test: np.ndarray,
) -> dict[str, float]:
    labels = manifest["category_name"].to_numpy()
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
            random_state=0,
            solver="lbfgs",
        ),
    )
    model.fit(features[fit], labels[fit])
    predictions = model.predict(features[test])
    return {
        "category_balanced_accuracy": float(balanced_accuracy_score(labels[test], predictions)),
        "category_macro_f1": float(f1_score(labels[test], predictions, average="macro")),
        "category_weighted_f1": float(f1_score(labels[test], predictions, average="weighted")),
    }


def same_category_purity(
    features: np.ndarray,
    manifest: pd.DataFrame,
    test: np.ndarray,
    k: int,
) -> float:
    matrix = np.asarray(features[test], dtype=np.float64)
    labels = manifest.iloc[test]["category_name"].astype(str).to_numpy()
    n_rows = len(matrix)
    if n_rows <= k + 1:
        return float("nan")
    neighbors = NearestNeighbors(n_neighbors=min(k + 1, n_rows), metric="cosine", n_jobs=1)
    neighbors.fit(matrix)
    indices = neighbors.kneighbors(
        matrix,
        n_neighbors=min(k + 1, n_rows),
        return_distance=False,
    )[:, 1:]
    return float(np.mean(labels[indices] == labels[:, None]))


def same_sample_top1_retrieval(
    features: np.ndarray,
    manifest: pd.DataFrame,
    test: np.ndarray,
) -> float:
    if "sample_id" not in manifest.columns:
        return float("nan")
    matrix = np.asarray(features[test], dtype=np.float64)
    sample_ids = manifest.iloc[test]["sample_id"].astype(str).to_numpy()
    n_rows = len(matrix)
    if n_rows <= 1:
        return float("nan")
    neighbors = NearestNeighbors(n_neighbors=2, metric="cosine", n_jobs=1)
    neighbors.fit(matrix)
    indices = neighbors.kneighbors(matrix, n_neighbors=2, return_distance=False)[:, 1]
    return float(np.mean(sample_ids[indices] == sample_ids))


def neighborhood_metrics(
    features: np.ndarray,
    manifest: pd.DataFrame,
    test: np.ndarray,
) -> dict[str, float]:
    metrics = {
        f"same_category_purity_k{k}": same_category_purity(features, manifest, test, k)
        for k in NEIGHBORHOOD_K
    }
    metrics["same_sample_top1_retrieval"] = same_sample_top1_retrieval(features, manifest, test)
    return metrics


def evaluate_representation(
    features: np.ndarray,
    manifest: pd.DataFrame,
    fit: np.ndarray,
    test: np.ndarray,
) -> dict[str, float]:
    metrics = {
        **scanner_probe_metrics(features, manifest, fit, test),
        **category_probe_metrics(features, manifest, fit, test),
        **neighborhood_metrics(features, manifest, test),
    }
    metrics["scanner_category_ratio"] = safe_ratio(
        metrics["scanner_balanced_accuracy"],
        metrics["category_balanced_accuracy"],
    )
    metrics["category_scanner_ratio"] = safe_ratio(
        metrics["category_balanced_accuracy"],
        metrics["scanner_balanced_accuracy"],
    )
    return metrics


# ---------------------------------------------------------------------------
# Experiment
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    out_dir: Path
    folds: tuple[int, ...] = FOLDS
    pair_integrity_seeds: tuple[int, ...] = PAIR_INTEGRITY_SEEDS
    k_values: tuple[int, ...] = K_VALUES


def append_row(
    rows: list[dict[str, object]],
    *,
    representation: str,
    representation_family: str,
    branch: str,
    fold: int,
    neural_seed: int | float,
    k_value: int | float,
    scanner_centroid_rank: int | float,
    metrics: dict[str, float],
) -> None:
    rows.append({
        "representation": representation,
        "representation_family": representation_family,
        "branch": branch,
        "fold": int(fold),
        "neural_seed": neural_seed,
        "k_value": k_value,
        "scanner_centroid_rank": scanner_centroid_rank,
        **metrics,
    })


def run_experiment(config: ExperimentConfig) -> pd.DataFrame:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    canine_cross.patch_scanner_namespace()
    frozen_features, frozen_frame = load_frozen_features()

    rows: list[dict[str, object]] = []

    for fold in config.folds:
        manifest = load_manifest(fold)
        aligned_frozen, _aligned_frame = canine_cross.align_fold(
            frozen_features,
            frozen_frame,
            manifest_path(fold),
        )
        fit_indices, test_indices = canine_cross.validate_fold(manifest, fold)

        print(
            f"\nFold {fold}: fit={len(fit_indices)} test={len(test_indices)} "
            f"samples={manifest['sample_id'].nunique()} categories={manifest['category_name'].nunique()}"
        )

        oldstyle_scaler = StandardScaler()
        oldstyle_features = oldstyle_scaler.fit_transform(aligned_frozen).astype(np.float32)
        directions = oldstyle_scanner_centroid_directions(oldstyle_features, manifest, fit_indices)
        centroid_rank = int(np.linalg.matrix_rank(directions)) if len(directions) else 0
        print(f"  Old-style scanner-centroid rank: {centroid_rank}")

        original_metrics = evaluate_representation(
            oldstyle_features,
            manifest,
            fit_indices,
            test_indices,
        )
        append_row(
            rows,
            representation="original_frozen_features",
            representation_family="frozen",
            branch="original",
            fold=fold,
            neural_seed=np.nan,
            k_value=np.nan,
            scanner_centroid_rank=centroid_rank,
            metrics=original_metrics,
        )

        for k in config.k_values:
            keep, removed, effective_k = oldstyle_split_keep_removed(oldstyle_features, directions, k)
            for branch_name, branch_features in (("keep", keep), ("removed", removed)):
                rep_name = f"oldstyle_{branch_name}_k{k}"
                metrics = evaluate_representation(
                    branch_features,
                    manifest,
                    fit_indices,
                    test_indices,
                )
                append_row(
                    rows,
                    representation=rep_name,
                    representation_family="oldstyle_linear_decomposition",
                    branch=branch_name,
                    fold=fold,
                    neural_seed=np.nan,
                    k_value=int(effective_k),
                    scanner_centroid_rank=centroid_rank,
                    metrics=metrics,
                )

            keep_metrics = rows[-2]
            removed_metrics = rows[-1]
            print(
                f"  oldstyle k={k}: keep scanner={float(keep_metrics['scanner_balanced_accuracy']):.4f} "
                f"cat={float(keep_metrics['category_balanced_accuracy']):.4f}; "
                f"removed scanner={float(removed_metrics['scanner_balanced_accuracy']):.4f} "
                f"cat={float(removed_metrics['category_balanced_accuracy']):.4f}"
            )

        for neural_seed in config.pair_integrity_seeds:
            for condition, bio_name, acq_name in (
                ("true_pairs", "true_pair_biological", "true_pair_acquisition"),
                ("shuffled_sample_pairs", "shuffled_sample_biological", "shuffled_sample_acquisition"),
            ):
                biological, acquisition, pair_frame = load_pair_integrity_features(
                    fold,
                    neural_seed,
                    condition,
                )
                aligned_bio = align_pair_integrity_features(biological, pair_frame, manifest)
                aligned_acq = align_pair_integrity_features(acquisition, pair_frame, manifest)
                for rep_name, branch_features in (
                    (bio_name, aligned_bio),
                    (acq_name, aligned_acq),
                ):
                    metrics = evaluate_representation(
                        branch_features,
                        manifest,
                        fit_indices,
                        test_indices,
                    )
                    append_row(
                        rows,
                        representation=rep_name,
                        representation_family="neural_factorization",
                        branch="biological" if "biological" in rep_name else "acquisition",
                        fold=fold,
                        neural_seed=int(neural_seed),
                        k_value=np.nan,
                        scanner_centroid_rank=np.nan,
                        metrics=metrics,
                    )

    raw = pd.DataFrame(rows)
    raw = raw.sort_values(["representation_family", "representation", "fold", "neural_seed"])
    return raw.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Summary and contrasts
# ---------------------------------------------------------------------------


def build_summary(raw: pd.DataFrame) -> pd.DataFrame:
    group_cols = ["representation", "representation_family", "branch"]
    grouped = raw.groupby(group_cols, dropna=False)
    metric_summary = grouped[list(METRIC_COLUMNS)].agg(["mean", "std", "min", "max"])
    metric_summary.columns = ["_".join(col).strip("_") for col in metric_summary.columns]
    counts = grouped.agg(
        n_runs=("fold", "size"),
        n_folds=("fold", "nunique"),
        n_neural_seeds=("neural_seed", lambda values: values.dropna().nunique()),
        k_value=("k_value", lambda values: first_non_null(values)),
        scanner_centroid_rank=("scanner_centroid_rank", lambda values: first_non_null(values)),
    )
    summary = counts.join(metric_summary).reset_index()
    order = {name: index for index, name in enumerate(EXPECTED_REPRESENTATIONS)}
    summary["_order"] = summary["representation"].map(lambda name: order.get(str(name), 999))
    return summary.sort_values(["_order", "representation"]).drop(columns=["_order"])


def first_non_null(values: pd.Series) -> float:
    clean = values.dropna()
    if clean.empty:
        return float("nan")
    return float(clean.iloc[0])


def metric_value(row: pd.Series, name: str) -> float:
    return float(row[name])


def build_branch_contrasts(raw: pd.DataFrame) -> pd.DataFrame:
    contrast_rows: list[dict[str, object]] = []

    for fold in sorted(raw["fold"].unique()):
        fold_rows = raw[raw["fold"] == fold]

        for neural_seed in sorted(fold_rows["neural_seed"].dropna().astype(int).unique()):
            seed_rows = fold_rows[fold_rows["neural_seed"] == neural_seed]
            for prefix, bio_name, acq_name in (
                ("true_pair", "true_pair_biological", "true_pair_acquisition"),
                ("shuffled_sample", "shuffled_sample_biological", "shuffled_sample_acquisition"),
            ):
                bio_rows = seed_rows[seed_rows["representation"] == bio_name]
                acq_rows = seed_rows[seed_rows["representation"] == acq_name]
                if bio_rows.empty or acq_rows.empty:
                    continue
                bio = bio_rows.iloc[0]
                acq = acq_rows.iloc[0]
                contrast_rows.append({
                    "contrast_type": f"{prefix}_branch",
                    "fold": int(fold),
                    "neural_seed": int(neural_seed),
                    "k_value": np.nan,
                    "paired_category_contrast": metric_value(bio, "category_balanced_accuracy")
                    - metric_value(acq, "category_balanced_accuracy"),
                    "paired_scanner_contrast": metric_value(acq, "scanner_balanced_accuracy")
                    - metric_value(bio, "scanner_balanced_accuracy"),
                    "paired_acquisition_category_leakage": metric_value(acq, "category_balanced_accuracy"),
                    "paired_bio_scanner_leakage": metric_value(bio, "scanner_balanced_accuracy"),
                    "biological_scanner_balanced_accuracy": metric_value(bio, "scanner_balanced_accuracy"),
                    "acquisition_scanner_balanced_accuracy": metric_value(acq, "scanner_balanced_accuracy"),
                    "biological_category_balanced_accuracy": metric_value(bio, "category_balanced_accuracy"),
                    "acquisition_category_balanced_accuracy": metric_value(acq, "category_balanced_accuracy"),
                })

        for k in K_VALUES:
            keep_rows = fold_rows[fold_rows["representation"] == f"oldstyle_keep_k{k}"]
            removed_rows = fold_rows[fold_rows["representation"] == f"oldstyle_removed_k{k}"]
            if keep_rows.empty or removed_rows.empty:
                continue
            keep = keep_rows.iloc[0]
            removed = removed_rows.iloc[0]
            contrast_rows.append({
                "contrast_type": "oldstyle_decomposition",
                "fold": int(fold),
                "neural_seed": np.nan,
                "k_value": int(k),
                "oldstyle_category_contrast": metric_value(keep, "category_balanced_accuracy")
                - metric_value(removed, "category_balanced_accuracy"),
                "oldstyle_scanner_contrast": metric_value(removed, "scanner_balanced_accuracy")
                - metric_value(keep, "scanner_balanced_accuracy"),
                "oldstyle_removed_category_leakage": metric_value(removed, "category_balanced_accuracy"),
                "oldstyle_keep_scanner_leakage": metric_value(keep, "scanner_balanced_accuracy"),
                "keep_scanner_balanced_accuracy": metric_value(keep, "scanner_balanced_accuracy"),
                "removed_scanner_balanced_accuracy": metric_value(removed, "scanner_balanced_accuracy"),
                "keep_category_balanced_accuracy": metric_value(keep, "category_balanced_accuracy"),
                "removed_category_balanced_accuracy": metric_value(removed, "category_balanced_accuracy"),
            })

        keep4_rows = fold_rows[fold_rows["representation"] == "oldstyle_keep_k4"]
        removed4_rows = fold_rows[fold_rows["representation"] == "oldstyle_removed_k4"]
        if keep4_rows.empty or removed4_rows.empty:
            continue
        keep4 = keep4_rows.iloc[0]
        removed4 = removed4_rows.iloc[0]
        true_rows = fold_rows[fold_rows["representation"].isin([
            "true_pair_biological",
            "true_pair_acquisition",
        ])]
        for neural_seed in sorted(true_rows["neural_seed"].dropna().astype(int).unique()):
            bio_rows = true_rows[
                (true_rows["neural_seed"] == neural_seed)
                & (true_rows["representation"] == "true_pair_biological")
            ]
            acq_rows = true_rows[
                (true_rows["neural_seed"] == neural_seed)
                & (true_rows["representation"] == "true_pair_acquisition")
            ]
            if bio_rows.empty or acq_rows.empty:
                continue
            bio = bio_rows.iloc[0]
            acq = acq_rows.iloc[0]
            contrast_rows.append({
                "contrast_type": "paired_vs_oldstyle_k4",
                "fold": int(fold),
                "neural_seed": int(neural_seed),
                "k_value": 4,
                "oldstyle_keep_scanner_minus_true_pair_bio": metric_value(keep4, "scanner_balanced_accuracy")
                - metric_value(bio, "scanner_balanced_accuracy"),
                "oldstyle_keep_category_minus_true_pair_bio": metric_value(keep4, "category_balanced_accuracy")
                - metric_value(bio, "category_balanced_accuracy"),
                "oldstyle_removed_scanner_minus_true_pair_acquisition": metric_value(
                    removed4,
                    "scanner_balanced_accuracy",
                ) - metric_value(acq, "scanner_balanced_accuracy"),
                "oldstyle_removed_category_minus_true_pair_acquisition": metric_value(
                    removed4,
                    "category_balanced_accuracy",
                ) - metric_value(acq, "category_balanced_accuracy"),
            })

    return pd.DataFrame(contrast_rows)


def build_neighborhood_purity(raw: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "representation",
        "representation_family",
        "branch",
        "fold",
        "neural_seed",
        "k_value",
        "same_category_purity_k1",
        "same_category_purity_k5",
        "same_category_purity_k10",
        "same_sample_top1_retrieval",
    ]
    return raw[columns].copy()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_outputs(raw: pd.DataFrame, summary: pd.DataFrame, config: ExperimentConfig) -> list[str]:
    issues: list[str] = []

    expected_total = len(config.folds) * (1 + 2 * len(config.k_values))
    expected_total += len(config.folds) * len(config.pair_integrity_seeds) * 4
    if len(raw) != expected_total:
        issues.append(f"Expected {expected_total} raw rows; observed {len(raw)}.")

    duplicate_count = int(raw.duplicated(["representation", "fold", "neural_seed"]).sum())
    if duplicate_count:
        issues.append(f"Duplicate representation/fold/seed rows: {duplicate_count}.")

    observed = set(raw["representation"].unique())
    expected = set(EXPECTED_REPRESENTATIONS)
    missing = sorted(expected - observed)
    extra = sorted(observed - expected)
    if missing:
        issues.append(f"Missing representations: {missing}.")
    if extra:
        issues.append(f"Unexpected representations: {extra}.")

    observed_k = sorted(
        int(value)
        for value in raw.loc[
            raw["representation"].str.startswith("oldstyle_"),
            "k_value",
        ].dropna().unique()
    )
    if observed_k != list(config.k_values):
        issues.append(f"Expected oldstyle k values {list(config.k_values)}; observed {observed_k}.")

    for column in METRIC_COLUMNS:
        if column not in raw.columns:
            issues.append(f"Missing metric column: {column}.")
            continue
        values = pd.to_numeric(raw[column], errors="coerce")
        if values.isna().any():
            issues.append(f"Missing values in metric column: {column}.")
            continue
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            issues.append(f"Nonfinite values in metric column: {column}.")

    keep4 = summary[summary["representation"] == "oldstyle_keep_k4"]
    if keep4.empty:
        issues.append("Missing oldstyle_keep_k4 summary row.")
    else:
        scanner_mean = float(keep4.iloc[0]["scanner_balanced_accuracy_mean"])
        delta = abs(scanner_mean - OLDSTYLE_K4_SCANNER_REFERENCE)
        if delta > OLDSTYLE_K4_SCANNER_TOLERANCE:
            issues.append(
                "oldstyle_keep_k4 scanner accuracy does not align with the 0.2000 "
                f"linear_projection_k4 reference; observed {scanner_mean:.4f}."
            )

    return issues


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def summary_row(summary: pd.DataFrame, representation: str) -> pd.Series | None:
    subset = summary[summary["representation"] == representation]
    if subset.empty:
        return None
    return subset.iloc[0]


def mean_metric(summary: pd.DataFrame, representation: str, metric: str) -> float:
    row = summary_row(summary, representation)
    if row is None:
        return float("nan")
    return float(row[f"{metric}_mean"])


def format_metric_line(summary: pd.DataFrame, representation: str) -> str:
    row = summary_row(summary, representation)
    if row is None:
        return f"- {representation}: missing"
    return (
        f"- {representation}: n={int(row['n_runs'])}, "
        f"scanner_acc={float(row['scanner_balanced_accuracy_mean']):.4f}, "
        f"scanner_f1={float(row['scanner_macro_f1_mean']):.4f}, "
        f"category_acc={float(row['category_balanced_accuracy_mean']):.4f}, "
        f"category_macro_f1={float(row['category_macro_f1_mean']):.4f}, "
        f"category_weighted_f1={float(row['category_weighted_f1_mean']):.4f}, "
        f"purity_k1={float(row['same_category_purity_k1_mean']):.4f}, "
        f"purity_k5={float(row['same_category_purity_k5_mean']):.4f}, "
        f"purity_k10={float(row['same_category_purity_k10_mean']):.4f}, "
        f"same_sample_top1={float(row['same_sample_top1_retrieval_mean']):.4f}"
    )


def build_report(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    config: ExperimentConfig,
    issues: list[str],
    runtime_seconds: float,
) -> str:
    tpb_scanner = mean_metric(summary, "true_pair_biological", "scanner_balanced_accuracy")
    tpb_category = mean_metric(summary, "true_pair_biological", "category_balanced_accuracy")
    tpa_scanner = mean_metric(summary, "true_pair_acquisition", "scanner_balanced_accuracy")
    tpa_category = mean_metric(summary, "true_pair_acquisition", "category_balanced_accuracy")
    keep4_scanner = mean_metric(summary, "oldstyle_keep_k4", "scanner_balanced_accuracy")
    keep4_category = mean_metric(summary, "oldstyle_keep_k4", "category_balanced_accuracy")
    removed4_scanner = mean_metric(summary, "oldstyle_removed_k4", "scanner_balanced_accuracy")
    removed4_category = mean_metric(summary, "oldstyle_removed_k4", "category_balanced_accuracy")

    paired_category_contrast = tpb_category - tpa_category
    paired_scanner_contrast = tpa_scanner - tpb_scanner
    oldstyle_category_contrast = keep4_category - removed4_category
    oldstyle_scanner_contrast = removed4_scanner - keep4_scanner

    lines = [
        "# Old-Style Residual Branch-Separation Audit",
        "",
        "## Branch",
        "",
        BRANCH,
        "",
        "## Question",
        "",
        "Does the old-style scanner-centroid/QR residual component act like a clean",
        "scanner branch, or does it leak biological category structure?",
        "",
        "## Dataset and Protocol",
        "",
        "- Dataset: canine cutaneous SCC DINOv2 frozen features.",
        "- Label column: category_name.",
        "- Scanner column: scanner_id.",
        "- Sample column: sample_id.",
        "- Region column: region_id.",
        f"- Folds: {', '.join(str(fold) for fold in config.folds)}.",
        f"- Pair-integrity seeds: {', '.join(str(seed) for seed in config.pair_integrity_seeds)}.",
        "- Old-style k values: 1, 2, 3, 4 only.",
        "",
        "## Formulas",
        "",
        "Old-style scanner-centroid directions are fit per fold on old-style standardized",
        "features. The StandardScaler is fit on all aligned frozen features to reproduce",
        "the old linear_projection_k4 convention used by the consistency audit.",
        "",
        "For fit rows X_fit and scanner labels s:",
        "- grand_mean = mean(X_fit)",
        "- direction_s = mean(X_fit where scanner_id == s) - grand_mean",
        "- Q_k = first k QR-orthonormalized direction rows",
        "- oldstyle_removed_k = X @ Q_k.T @ Q_k",
        "- oldstyle_keep_k = X - oldstyle_removed_k",
        "",
        "There are five scanners, so the scanner-centroid rank is at most four.",
        "",
        "## Row Counts",
        "",
        f"- Raw metric rows: {len(raw)}.",
        f"- Summary rows: {len(summary)}.",
        f"- Branch contrast rows: {len(contrasts)}.",
        f"- Neighborhood rows: {len(raw)}.",
        "",
        "## Key Metrics",
        "",
    ]

    for representation in (
        "original_frozen_features",
        "true_pair_biological",
        "true_pair_acquisition",
        "shuffled_sample_biological",
        "shuffled_sample_acquisition",
        "oldstyle_keep_k1",
        "oldstyle_removed_k1",
        "oldstyle_keep_k2",
        "oldstyle_removed_k2",
        "oldstyle_keep_k3",
        "oldstyle_removed_k3",
        "oldstyle_keep_k4",
        "oldstyle_removed_k4",
    ):
        lines.append(format_metric_line(summary, representation))

    lines.extend([
        "",
        "## Paired vs Oldstyle Branch Contrasts",
        "",
        f"- paired_category_contrast = {paired_category_contrast:.4f} "
        "(true_pair_biological category_acc - true_pair_acquisition category_acc).",
        f"- paired_scanner_contrast = {paired_scanner_contrast:.4f} "
        "(true_pair_acquisition scanner_acc - true_pair_biological scanner_acc).",
        f"- oldstyle_category_contrast_k4 = {oldstyle_category_contrast:.4f} "
        "(oldstyle_keep_k4 category_acc - oldstyle_removed_k4 category_acc).",
        f"- oldstyle_scanner_contrast_k4 = {oldstyle_scanner_contrast:.4f} "
        "(oldstyle_removed_k4 scanner_acc - oldstyle_keep_k4 scanner_acc).",
        "",
        "## Leakage Findings",
        "",
        f"- paired_acquisition_category_leakage = {tpa_category:.4f}.",
        f"- oldstyle_removed_category_leakage_k4 = {removed4_category:.4f}.",
        f"- paired_bio_scanner_leakage = {tpb_scanner:.4f}.",
        f"- oldstyle_keep_scanner_leakage_k4 = {keep4_scanner:.4f}.",
        "",
        "## Key Questions",
        "",
        "1. Does oldstyle_keep_k4 suppress scanner more strongly than true_pair_biological?",
        f"   {'Yes' if keep4_scanner < tpb_scanner else 'No'}: "
        f"oldstyle_keep_k4 scanner_acc={keep4_scanner:.4f}, "
        f"true_pair_biological scanner_acc={tpb_scanner:.4f}.",
        "2. Does oldstyle_keep_k4 preserve category signal better than true_pair_biological?",
        f"   {'Yes' if keep4_category > tpb_category else 'No'}: "
        f"oldstyle_keep_k4 category_acc={keep4_category:.4f}, "
        f"true_pair_biological category_acc={tpb_category:.4f}.",
        "3. Does oldstyle_removed_k4 carry scanner signal?",
        f"   Scanner_acc={removed4_scanner:.4f}.",
        "4. Does oldstyle_removed_k4 leak category signal?",
        f"   Category_acc={removed4_category:.4f}.",
        "5. Is oldstyle_removed_k4 cleaner than true_pair_acquisition at keeping category signal out?",
        f"   {'Yes' if removed4_category < tpa_category else 'No'}: "
        f"oldstyle_removed_k4 category_acc={removed4_category:.4f}, "
        f"true_pair_acquisition category_acc={tpa_category:.4f}.",
        "6. Does old-style linear residual decomposition fully explain paired-acquisition branch separation?",
        "   This audit should be read through the leakage and contrast values above.",
        "7. Or do the two methods occupy different separation-frontier points?",
        "   This is suggested when scanner/category tradeoffs differ across keep/removed and",
        "   biological/acquisition branches.",
        "",
        "## Bounded Interpretation",
        "",
        "The oldstyle_keep_k4 result should be treated as the stronger raw scanner-removal",
        "linear baseline when compared with true_pair_biological. A paired-acquisition",
        "claim should therefore focus on structured separation, not on beating this",
        "old-style baseline on raw scanner suppression.",
        "",
        "The decisive check is whether oldstyle_removed_k4 carries scanner signal while",
        "keeping category signal out as well as, or better than, true_pair_acquisition.",
        "If it does, that supports a stronger linear residual baseline. If it does not,",
        "that suggests the paired and old-style decompositions occupy different",
        "separation frontier points.",
        "",
        "## Previous Interpretation",
        "",
        "The linear baseline consistency correction remains in force: the logistic-SVD",
        "split from the earlier residual audit was weaker for scanner removal than the",
        "old-style centroid/QR projection. Any prior statement comparing paired",
        "acquisition to the strongest simple linear scanner removal should use the",
        "old-style baseline, not the logistic-SVD split.",
        "",
        "## Validation Checks",
        "",
        f"- Expected representations present: {set(EXPECTED_REPRESENTATIONS).issubset(set(raw['representation'].unique()))}.",
        "- k values present: "
        f"{[int(value) for value in sorted(raw.loc[raw['representation'].str.startswith('oldstyle_'), 'k_value'].dropna().astype(int).unique())]}.",
        f"- oldstyle_keep_k4 scanner reference target: {OLDSTYLE_K4_SCANNER_REFERENCE:.4f}.",
        f"- oldstyle_keep_k4 scanner observed: {keep4_scanner:.4f}.",
        f"- Validation issue count: {len(issues)}.",
    ])

    if issues:
        for issue in issues:
            lines.append(f"  - {issue}")
    else:
        lines.append("  - No validation issues found.")

    lines.extend([
        "",
        "## Files Created",
        "",
        "- oldstyle_residual_raw_metrics.csv",
        "- oldstyle_residual_summary.csv",
        "- oldstyle_residual_branch_contrasts.csv",
        "- oldstyle_residual_neighborhood_purity.csv",
        "- oldstyle_residual_branch_separation_report.md",
        "- experiment_design.json",
        "- run_log.txt",
        "",
        f"Runtime seconds: {runtime_seconds:.1f}",
        "",
        "## Readiness",
        "",
        "Ready to commit after external diff hygiene checks pass; no staging or commit performed.",
        "",
    ])

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/paired_acquisition_factorization_oldstyle_residual_branch_separation_audit"),
    )
    parser.add_argument("--folds", nargs="+", type=int, default=list(FOLDS))
    parser.add_argument("--neural-seeds", nargs="+", type=int, default=list(PAIR_INTEGRITY_SEEDS))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "run_log.txt"
    start = time.perf_counter()

    config = ExperimentConfig(
        out_dir=args.out_dir,
        folds=tuple(args.folds),
        pair_integrity_seeds=tuple(args.neural_seeds),
        k_values=K_VALUES,
    )

    design = {
        "branch": BRANCH,
        "stage": "paired_acquisition_factorization_oldstyle_residual_branch_separation_audit",
        "dataset": "canine_cutaneous_scc_dinov2",
        "feature_path": str(FEATURE_PATH),
        "folds": list(config.folds),
        "pair_integrity_seeds": list(config.pair_integrity_seeds),
        "k_values": list(config.k_values),
        "neighborhood_k_values": list(NEIGHBORHOOD_K),
        "columns": {
            "label": "category_name",
            "scanner": "scanner_id",
            "sample": "sample_id",
            "region": "region_id",
        },
        "oldstyle_split_formula": {
            "centroid_direction": "mean(X_fit[scanner_id == s]) - mean(X_fit)",
            "basis": "Q_k = first k QR-orthonormalized scanner-centroid directions",
            "removed": "oldstyle_removed_k = X @ Q_k.T @ Q_k",
            "keep": "oldstyle_keep_k = X - oldstyle_removed_k",
            "standardization": "StandardScaler fit on all aligned frozen features, matching old linear_projection_k4 convention",
        },
        "representations": list(EXPECTED_REPRESENTATIONS),
        "scanners": list(SCANNERS),
        "command": " ".join(sys.argv),
    }
    atomic_text(args.out_dir / "experiment_design.json", json.dumps(design, indent=2, sort_keys=True) + "\n")

    with log_path.open("a", encoding="utf-8") as log_file:
        with redirect_stdout(Tee(sys.stdout, log_file)), redirect_stderr(Tee(sys.stderr, log_file)):
            print("\n" + "=" * 80)
            print(time.strftime("%Y-%m-%d %H:%M:%S"))
            print("COMMAND " + " ".join(sys.argv))
            try:
                raw = run_experiment(config)
                summary = build_summary(raw)
                contrasts = build_branch_contrasts(raw)
                neighborhood = build_neighborhood_purity(raw)
                issues = validate_outputs(raw, summary, config)

                atomic_csv(args.out_dir / "oldstyle_residual_raw_metrics.csv", raw)
                atomic_csv(args.out_dir / "oldstyle_residual_summary.csv", summary)
                atomic_csv(args.out_dir / "oldstyle_residual_branch_contrasts.csv", contrasts)
                atomic_csv(args.out_dir / "oldstyle_residual_neighborhood_purity.csv", neighborhood)

                runtime = time.perf_counter() - start
                report = build_report(raw, summary, contrasts, config, issues, runtime)
                atomic_text(args.out_dir / "oldstyle_residual_branch_separation_report.md", report)

                print("\n" + "=" * 80)
                print("OLD-STYLE RESIDUAL BRANCH-SEPARATION AUDIT COMPLETE")
                print(f"Rows: {len(raw)}")
                print(f"Runtime: {runtime:.1f}s")
                print(f"Validation issues: {len(issues)}")
                for issue in issues:
                    print(f"  - {issue}")
                print(f"Report: {(args.out_dir / 'oldstyle_residual_branch_separation_report.md').resolve()}")

                for rep in (
                    "true_pair_biological",
                    "true_pair_acquisition",
                    "oldstyle_keep_k4",
                    "oldstyle_removed_k4",
                ):
                    row = summary_row(summary, rep)
                    if row is not None:
                        print(
                            f"{rep}: scanner={float(row['scanner_balanced_accuracy_mean']):.4f} "
                            f"category={float(row['category_balanced_accuracy_mean']):.4f}"
                        )

            except Exception:
                traceback.print_exc()
                raise


if __name__ == "__main__":
    main()
