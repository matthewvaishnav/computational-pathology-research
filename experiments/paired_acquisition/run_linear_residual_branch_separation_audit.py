#!/usr/bin/env python3
"""Linear residual branch-separation audit for paired-acquisition factorization.

Directly compares paired-acquisition branch separation against a linear
scanner-subspace split: keep (scanner subspace removed) vs removed (scanner
subspace component).

Key question: Does paired-acquisition produce cleaner biological/acquisition
separation than a linear scanner-subspace decomposition?
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
import sys
import tempfile
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Callable

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

from experiments.external_multiscanner import run_canine_pathoalign_crossfold as canine_cross  # noqa: E402
from experiments.scorpion import run_pathoalign_projection as projection  # noqa: E402
from experiments.canine import run_pair_integrity_falsification_caninescc as canine_pair  # noqa: E402


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SCANNERS = ("cs2", "gt450", "nz20", "nz210", "p1000")
FOLDS = (0, 1, 2, 3, 4)
PAIR_INTEGRITY_SEEDS = (911, 912, 913, 914, 915)
K_VALUES = (1, 2, 4, 8, 16, 32)
NEIGHBORHOOD_K = (1, 5, 10)

FEATURE_PATH = Path("results/external_multiscanner_caninescc/features/fold_0_dinov2_base.npz")
MANIFESTS_DIR = Path("data/external_multiscanner_caninescc/patch_manifests/splits")
PAIR_INTEGRITY_DIR = Path("results/paired_acquisition_factorization_pair_integrity_caninescc")

# Primary linear representations
LINEAR_REP_METADATA = {}

for k in K_VALUES:
    LINEAR_REP_METADATA[f"linear_keep_k{k}"] = {
        "family": "linear_decomposition",
        "branch": "keep",
        "k": k,
        "label": f"Linear scanner subspace removed, k={k}",
    }
    LINEAR_REP_METADATA[f"linear_removed_k{k}"] = {
        "family": "linear_decomposition",
        "branch": "removed",
        "k": k,
        "label": f"Linear scanner subspace residual, k={k}",
    }

NEURAL_REP_METADATA = {
    "original_frozen_features": {"family": "frozen", "label": "Original frozen DINOv2 features"},
    "true_pair_biological": {"family": "neural_factorization", "branch": "biological", "label": "True-pair biological branch"},
    "true_pair_acquisition": {"family": "neural_factorization", "branch": "acquisition", "label": "True-pair acquisition branch"},
    "shuffled_sample_biological": {"family": "neural_factorization", "branch": "biological", "label": "Shuffled-sample biological branch"},
    "shuffled_sample_acquisition": {"family": "neural_factorization", "branch": "acquisition", "label": "Shuffled-sample acquisition branch"},
}


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


def numeric_rank(singular_values: np.ndarray, shape: tuple[int, int]) -> int:
    if singular_values.size == 0:
        return 0
    tolerance = max(shape) * np.finfo(np.float64).eps * float(singular_values[0])
    return int(np.sum(singular_values > tolerance))


# ---------------------------------------------------------------------------
# Scanner subspace computation
# ---------------------------------------------------------------------------


def scanner_subspace_directions(
    features: np.ndarray, frame: pd.DataFrame, fit: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Return orthonormal basis for the scanner-discriminative subspace."""
    labels = frame["scanner_id"].astype(str).to_numpy()
    model = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=5000, random_state=0, solver="lbfgs",
    )
    model.fit(features[fit], labels[fit])
    coefficients = np.asarray(model.coef_, dtype=np.float64)
    coefficients = coefficients - coefficients.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(coefficients, full_matrices=False)
    rank = numeric_rank(singular_values, coefficients.shape)
    return vt[:rank].astype(np.float64), rank


def split_keep_removed(
    features: np.ndarray,
    directions: np.ndarray,
    k: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Split standardized features into keep (subspace removed) and removed (subspace projection).

    Formula:
        keep   = features - features @ basis.T @ basis    (subspace removed)
        removed = features @ basis.T @ basis               (subspace component)

    Returns (keep, removed, effective_k).
    """
    effective_k = min(int(k), len(directions))
    if effective_k <= 0:
        return (
            np.asarray(features, dtype=np.float32).copy(),
            np.zeros_like(features, dtype=np.float32),
            0,
        )
    matrix = np.asarray(features, dtype=np.float64)
    basis = np.asarray(directions[:effective_k], dtype=np.float64)
    projection = (matrix @ basis.T) @ basis
    keep = (matrix - projection).astype(np.float32)
    removed = projection.astype(np.float32)
    if not np.isfinite(keep).all():
        raise projection.ExperimentError(f"linear_keep_k{k} produced nonfinite values.")
    if not np.isfinite(removed).all():
        raise projection.ExperimentError(f"linear_removed_k{k} produced nonfinite values.")
    return keep, removed, effective_k


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _manifest_path(fold: int) -> Path:
    return MANIFESTS_DIR / f"fold_{fold}_patch_manifest.csv"


def load_manifest(fold: int) -> pd.DataFrame:
    path = _manifest_path(fold)
    if not path.is_file():
        raise projection.ExperimentError(f"Missing manifest: {path}")
    df = pd.read_csv(path, dtype=str)
    required = {"slide_id", "region_id", "scanner_id", "category_name", "split"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise projection.ExperimentError(f"Manifest {path} missing columns: {missing}")
    df["scanner_id"] = df["scanner_id"].astype(str).str.lower()
    return df


def load_frozen_features() -> tuple[np.ndarray, pd.DataFrame]:
    features, frame, metadata = projection.load_archive(FEATURE_PATH)
    frame["scanner_id"] = frame["scanner_id"].astype(str).str.lower()
    if len(features) != 4025 or features.shape[1] != 768:
        raise projection.ExperimentError(
            f"Expected canine DINOv2 (4025, 768); observed {features.shape}"
        )
    return features, frame


def _align_pair_integrity_features(
    features: np.ndarray,
    pair_frame: pd.DataFrame,
    manifest: pd.DataFrame,
) -> np.ndarray:
    """Align pair-integrity projected features to manifest row order."""
    pair_frame = pair_frame.copy()
    pair_frame["scanner_id"] = pair_frame["scanner_id"].astype(str).str.lower()
    pair_keys = [
        (str(r.slide_id), str(r.region_id), str(r.scanner_id))
        for _, r in pair_frame.iterrows()
    ]
    manifest_keys = [
        (str(r.slide_id), str(r.region_id), str(r.scanner_id))
        for _, r in manifest.iterrows()
    ]
    if len(pair_keys) != len(manifest_keys):
        raise projection.ExperimentError(
            f"Pair-integrity features ({len(pair_keys)} rows) do not match "
            f"manifest ({len(manifest_keys)} rows)"
        )
    lookup = {key: idx for idx, key in enumerate(pair_keys)}
    missing = sum(1 for k in manifest_keys if k not in lookup)
    if missing:
        raise projection.ExperimentError(
            f"{missing} manifest keys not found in pair-integrity features"
        )
    order = np.asarray([lookup[k] for k in manifest_keys], dtype=np.int64)
    return features[order]


def load_pair_integrity_features(
    fold: int, seed: int, condition: str,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    projected = (
        PAIR_INTEGRITY_DIR / f"fold_{fold}" / "runs" / f"{condition}_seed_{seed}" / "projected_features.npz"
    )
    if not projected.is_file():
        raise projection.ExperimentError(f"Missing pair-integrity features: {projected}")
    biological, acquisition, frame, _metadata = canine_pair.load_projected(projected)
    return biological, acquisition, frame


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------


def scanner_probe_metrics(
    features: np.ndarray, frame: pd.DataFrame, fit: np.ndarray, test: np.ndarray,
) -> dict[str, float]:
    y = frame["scanner_id"].to_numpy()
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=0),
    )
    model.fit(features[fit], y[fit])
    pred = model.predict(features[test])
    return {
        "scanner_balanced_accuracy": float(balanced_accuracy_score(y[test], pred)),
        "scanner_macro_f1": float(f1_score(y[test], pred, average="macro")),
    }


def category_probe_metrics(
    features: np.ndarray, manifest: pd.DataFrame, fit: np.ndarray, test: np.ndarray,
) -> dict[str, float]:
    y = manifest.iloc[test]["category_name"].to_numpy()
    X_train = features[fit]
    y_train = manifest.iloc[fit]["category_name"].to_numpy()
    X_test = features[test]
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=0),
    )
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return {
        "category_balanced_accuracy": float(balanced_accuracy_score(y, pred)),
        "category_macro_f1": float(f1_score(y, pred, average="macro")),
        "category_weighted_f1": float(f1_score(y, pred, average="weighted")),
    }


def neighborhood_purity(
    features: np.ndarray, manifest: pd.DataFrame, test: np.ndarray, k: int,
) -> dict[str, float]:
    """Fraction of k-nearest neighbors that share the same category label.

    Computed on the test set only (no fit leakage).
    """
    X = features[test]
    y = manifest.iloc[test]["category_name"].to_numpy()
    n = len(X)
    if n <= k + 1:
        return {f"neighborhood_purity_k{k}": float("nan")}

    nn = NearestNeighbors(n_neighbors=min(k + 1, n), metric="cosine", n_jobs=1)
    nn.fit(X)
    # +1 because the first neighbor is the point itself
    indices = nn.kneighbors(X, n_neighbors=min(k + 1, n), return_distance=False)[:, 1:]
    same_category = y[indices] == y[:, None]
    purity = float(np.mean(same_category))
    return {f"neighborhood_purity_k{k}": purity}


def all_neighborhood_purity(
    features: np.ndarray, manifest: pd.DataFrame, test: np.ndarray,
) -> dict[str, float]:
    result = {}
    for k in NEIGHBORHOOD_K:
        result.update(neighborhood_purity(features, manifest, test, k))
    return result


# ---------------------------------------------------------------------------
# Branch separation metrics
# ---------------------------------------------------------------------------


def branch_contrast_metrics(
    scanner_metrics: dict[str, float],
    category_metrics: dict[str, float],
) -> dict[str, float]:
    """Compute scanner/category ratio and category/scanner ratio."""
    s_acc = scanner_metrics["scanner_balanced_accuracy"]
    c_acc = category_metrics["category_balanced_accuracy"]
    return {
        "scanner_category_ratio": float(s_acc / max(1e-8, c_acc)),
        "category_scanner_ratio": float(c_acc / max(1e-8, s_acc)),
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    out_dir: Path
    folds: tuple[int, ...] = FOLDS
    pair_integrity_seeds: tuple[int, ...] = PAIR_INTEGRITY_SEEDS
    k_values: tuple[int, ...] = K_VALUES


def run_experiment(config: ExperimentConfig) -> pd.DataFrame:
    config.out_dir.mkdir(parents=True, exist_ok=True)
    frozen_features, frozen_frame = load_frozen_features()
    canine_cross.patch_scanner_namespace()

    rows: list[dict[str, object]] = []

    for fold in config.folds:
        manifest = load_manifest(fold)
        aligned_frozen, _ = canine_cross.align_fold(frozen_features, frozen_frame, _manifest_path(fold))
        fit_indices, test_indices = canine_cross.validate_fold(manifest, fold)

        print(
            f"\nFold {fold}: fit={len(fit_indices)} test={len(test_indices)} "
            f"categories={sorted(manifest['category_name'].unique())}"
        )

        # Standardize frozen features
        standardized, _mean, _std = projection.standardize(aligned_frozen, fit_indices)

        # Compute scanner subspace directions (fitted once per fold)
        directions, fitted_rank = scanner_subspace_directions(standardized, manifest, fit_indices)
        print(f"  Scanner subspace rank: {fitted_rank}")

        # ----- Simple representations (single evaluation per fold) -----
        # original_frozen_features
        metrics = {
            **scanner_probe_metrics(standardized, manifest, fit_indices, test_indices),
            **category_probe_metrics(standardized, manifest, fit_indices, test_indices),
            **all_neighborhood_purity(standardized, manifest, test_indices),
        }
        metrics["scanner_category_ratio"] = float(
            metrics["scanner_balanced_accuracy"] / max(1e-8, metrics["category_balanced_accuracy"])
        )
        metrics["category_scanner_ratio"] = float(
            metrics["category_balanced_accuracy"] / max(1e-8, metrics["scanner_balanced_accuracy"])
        )
        rows.append({
            "representation": "original_frozen_features",
            "representation_family": "frozen",
            "branch": "original",
            "fold": int(fold),
            "neural_seed": "",
            "k_value": "",
            **metrics,
        })

        # Linear keep/removed for each k
        for k in config.k_values:
            keep, removed, effective_k = split_keep_removed(standardized, directions, k)

            for branch_name, features in [("keep", keep), ("removed", removed)]:
                rep_name = f"linear_{branch_name}_k{k}"
                metrics = {
                    **scanner_probe_metrics(features, manifest, fit_indices, test_indices),
                    **category_probe_metrics(features, manifest, fit_indices, test_indices),
                    **all_neighborhood_purity(features, manifest, test_indices),
                }
                metrics["scanner_category_ratio"] = float(
                    metrics["scanner_balanced_accuracy"] / max(1e-8, metrics["category_balanced_accuracy"])
                )
                metrics["category_scanner_ratio"] = float(
                    metrics["category_balanced_accuracy"] / max(1e-8, metrics["scanner_balanced_accuracy"])
                )
                rows.append({
                    "representation": rep_name,
                    "representation_family": "linear_decomposition",
                    "branch": branch_name,
                    "fold": int(fold),
                    "neural_seed": "",
                    "k_value": int(effective_k),
                    **metrics,
                })

        # ----- Neural factorization representations (per neural seed) -----
        for neural_seed in config.pair_integrity_seeds:
            for condition, bio_name, acq_name in [
                ("true_pairs", "true_pair_biological", "true_pair_acquisition"),
                ("shuffled_sample_pairs", "shuffled_sample_biological", "shuffled_sample_acquisition"),
            ]:
                bio, acq, pair_frame = load_pair_integrity_features(fold, neural_seed, condition)
                aligned_bio = _align_pair_integrity_features(bio, pair_frame, manifest)
                aligned_acq = _align_pair_integrity_features(acq, pair_frame, manifest)

                for rep_name, features in [(bio_name, aligned_bio), (acq_name, aligned_acq)]:
                    metrics = {
                        **scanner_probe_metrics(features, manifest, fit_indices, test_indices),
                        **category_probe_metrics(features, manifest, fit_indices, test_indices),
                        **all_neighborhood_purity(features, manifest, test_indices),
                    }
                    metrics["scanner_category_ratio"] = float(
                        metrics["scanner_balanced_accuracy"] / max(1e-8, metrics["category_balanced_accuracy"])
                    )
                    metrics["category_scanner_ratio"] = float(
                        metrics["category_balanced_accuracy"] / max(1e-8, metrics["scanner_balanced_accuracy"])
                    )
                    rows.append({
                        "representation": rep_name,
                        "representation_family": "neural_factorization",
                        "branch": "biological" if "biological" in rep_name else "acquisition",
                        "fold": int(fold),
                        "neural_seed": int(neural_seed),
                        "k_value": "",
                        **metrics,
                    })

    raw = pd.DataFrame(rows).sort_values([
        "representation_family", "representation", "fold", "neural_seed",
    ]).reset_index(drop=True)
    return raw


# ---------------------------------------------------------------------------
# Branch contrast computation
# ---------------------------------------------------------------------------


def build_branch_contrasts(raw: pd.DataFrame) -> pd.DataFrame:
    """Compute paired and linear branch separation contrasts."""
    metric_cols = [
        "scanner_balanced_accuracy", "scanner_macro_f1",
        "category_balanced_accuracy", "category_macro_f1", "category_weighted_f1",
        "scanner_category_ratio", "category_scanner_ratio",
    ]
    for k_metric in NEIGHBORHOOD_K:
        col = f"neighborhood_purity_k{k_metric}"
        if col in raw.columns:
            metric_cols.append(col)

    contrasts = []

    # Paired-acquisition branch contrast (per fold, per neural seed)
    for fold in sorted(raw["fold"].unique()):
        neural_seeds_for_fold = raw.loc[
            raw["neural_seed"].notna(), "neural_seed"
        ].astype(float).unique()
        for neural_seed in sorted(neural_seeds_for_fold):
            neural_seed_int = int(neural_seed)
            for condition_prefix, bio_name, acq_name in [
                ("true_pair", "true_pair_biological", "true_pair_acquisition"),
                ("shuffled_sample", "shuffled_sample_biological", "shuffled_sample_acquisition"),
            ]:
                bio_row = raw[
                    (raw["fold"] == fold) & (raw["neural_seed"] == float(neural_seed_int))
                    & (raw["representation"] == bio_name)
                ]
                acq_row = raw[
                    (raw["fold"] == fold) & (raw["neural_seed"] == float(neural_seed_int))
                    & (raw["representation"] == acq_name)
                ]
                if bio_row.empty or acq_row.empty:
                    continue
                bio = bio_row.iloc[0]
                acq = acq_row.iloc[0]
                row_dict = {
                    "contrast_type": f"{condition_prefix}_branch",
                    "fold": int(fold),
                    "neural_seed": neural_seed_int,
                    "k_value": "",
                }
                for col in metric_cols:
                    if col in bio.index and col in acq.index:
                        bio_val = float(bio[col])
                        acq_val = float(acq[col])
                        row_dict[f"biological_{col}"] = bio_val
                        row_dict[f"acquisition_{col}"] = acq_val
                        row_dict[f"bio_minus_acq_{col}"] = float(bio_val - acq_val)
                contrasts.append(row_dict)

    # Linear keep vs removed contrast (per fold)
    for fold in sorted(raw["fold"].unique()):
        for k in K_VALUES:
            keep_row = raw[
                (raw["fold"] == fold) & (raw["representation"] == f"linear_keep_k{k}")
            ]
            removed_row = raw[
                (raw["fold"] == fold) & (raw["representation"] == f"linear_removed_k{k}")
            ]
            if keep_row.empty or removed_row.empty:
                continue
            keep = keep_row.iloc[0]
            removed = removed_row.iloc[0]
            row_dict = {
                "contrast_type": "linear_decomposition",
                "fold": int(fold),
                "neural_seed": "",
                "k_value": int(k),
            }
            for col in metric_cols:
                if col in keep.index and col in removed.index:
                    keep_val = float(keep[col])
                    rem_val = float(removed[col])
                    row_dict[f"keep_{col}"] = keep_val
                    row_dict[f"removed_{col}"] = rem_val
                    row_dict[f"keep_minus_removed_{col}"] = float(keep_val - rem_val)
            contrasts.append(row_dict)

    return pd.DataFrame(contrasts)


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------


def build_summary(raw: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "scanner_balanced_accuracy", "scanner_macro_f1",
        "category_balanced_accuracy", "category_macro_f1", "category_weighted_f1",
        "scanner_category_ratio", "category_scanner_ratio",
    ]
    for k in NEIGHBORHOOD_K:
        col = f"neighborhood_purity_k{k}"
        if col in raw.columns:
            metric_cols.append(col)

    group_cols = ["representation", "representation_family", "branch"]
    grouped = raw.groupby(group_cols, dropna=False)
    metric_summary = grouped[metric_cols].agg(["mean", "std", "min", "max"])
    metric_summary.columns = ["_".join(col).strip("_") for col in metric_summary.columns]

    counts = grouped.agg(
        n_runs=("fold", "size"),
        n_folds=("fold", "nunique"),
        n_neural_seeds=("neural_seed", lambda x: x.dropna().nunique()),
    )
    summary = counts.join(metric_summary).reset_index()
    return summary.sort_values(["representation_family", "branch", "representation"])


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_outputs(raw: pd.DataFrame, config: ExperimentConfig) -> list[str]:
    issues: list[str] = []

    required_metrics = [
        "scanner_balanced_accuracy", "scanner_macro_f1",
        "category_balanced_accuracy", "category_macro_f1", "category_weighted_f1",
    ]
    for metric in required_metrics:
        if metric not in raw.columns:
            issues.append(f"Missing metric column: {metric}")
            continue
        values = pd.to_numeric(raw[metric], errors="coerce")
        if values.isna().any():
            issues.append(f"Missing values in {metric}")
        if not np.isfinite(values.to_numpy(float)).all():
            issues.append(f"Nonfinite values in {metric}")

    # Check for duplicate rows
    key_cols = ["representation", "fold", "neural_seed"]
    if raw.duplicated(key_cols).any():
        n_dup = raw.duplicated(key_cols).sum()
        issues.append(f"Duplicate rows: {n_dup}")

    # Expected row counts
    n_simple = 1 + 2 * len(config.k_values)  # frozen + linear keep*k + linear removed*k
    n_neural = 4  # 2 conditions x 2 branches
    expected_simple = n_simple * len(config.folds)
    expected_neural = n_neural * len(config.folds) * len(config.pair_integrity_seeds)
    expected_total = expected_simple + expected_neural
    if len(raw) != expected_total:
        issues.append(
            f"Expected {expected_total} rows (simple={expected_simple}, neural={expected_neural}); "
            f"observed {len(raw)}"
        )

    # Check all expected representations present
    expected_reps = {"original_frozen_features"}
    for k in config.k_values:
        expected_reps.add(f"linear_keep_k{k}")
        expected_reps.add(f"linear_removed_k{k}")
    for cond in ["true_pair", "shuffled_sample"]:
        for branch in ["biological", "acquisition"]:
            expected_reps.add(f"{cond}_{branch}")
    observed_reps = set(raw["representation"].unique())
    missing_reps = expected_reps - observed_reps
    if missing_reps:
        issues.append(f"Missing representations: {sorted(missing_reps)}")

    return issues


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def build_report(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    config: ExperimentConfig,
    issues: list[str],
    runtime_seconds: float,
) -> str:
    lines = [
        "# Linear Residual Branch-Separation Audit",
        "",
        "## Branch",
        "",
        "experiment/linear-residual-branch-separation-audit",
        "",
        "## Question",
        "",
        "Does paired-acquisition produce cleaner biological/acquisition separation than a",
        "linear scanner-subspace decomposition (keep vs removed components)?",
        "",
        "## Dataset",
        "",
        "- Canine cutaneous SCC, DINOv2-Base frozen features",
        "- 4025 patches, 44 samples, 5 scanners, 805 regions",
        "- 7 categories: Bone, Cartilage, Dermis, Epidermis, Inflamm/Necrosis, SCC, Subcutis",
        f"- {len(config.folds)} folds, {len(config.pair_integrity_seeds)} neural seeds",
        "",
        "## Formulas",
        "",
        "Linear decomposition (per fold, fit-set standardized features):",
        "- Logistic regression scanner classifier on fit set.",
        "- SVD of centered coefficient matrix to obtain orthonormal scanner subspace basis.",
        "- `linear_keep_kN = features - features @ basis[:N].T @ basis[:N]`  (subspace removed)",
        "- `linear_removed_kN = features @ basis[:N].T @ basis[:N]`  (subspace projection)",
        "- For k=0, keep = original standardized features, removed = zero matrix.",
        "",
        "Paired-acquisition factorization (pre-computed, frozen hyperparameters):",
        "- true_pair_biological: biological branch from true-pair factorization.",
        "- true_pair_acquisition: acquisition branch from true-pair factorization.",
        "- Shuffled-sample variants use deranged sample-pair training.",
        "",
        "## Representations",
        "",
        f"- {len(config.k_values) * 2} linear decomposition representations (keep + removed for each k)",
        f"- 1 frozen representation (original_frozen_features)",
        "- 4 neural factorization representations (2 pair conditions x 2 branches)",
        "",
        "k values: " + ", ".join(str(k) for k in config.k_values),
        "",
        "Fitted scanner subspace rank: " + _get_scanner_rank(raw, config),
        "",
        "## Key Metrics (mean across all folds/seeds)",
        "",
    ]

    # Primary representations summary table
    primary_reps = [
        "original_frozen_features",
        "true_pair_biological", "true_pair_acquisition",
        "shuffled_sample_biological", "shuffled_sample_acquisition",
    ]
    for k in config.k_values:
        primary_reps.append(f"linear_keep_k{k}")
        primary_reps.append(f"linear_removed_k{k}")

    display_cols = [
        "representation", "n_runs",
        "scanner_balanced_accuracy_mean", "category_balanced_accuracy_mean",
        "scanner_category_ratio_mean", "category_scanner_ratio_mean",
    ]
    for k_nn in NEIGHBORHOOD_K:
        col = f"neighborhood_purity_k{k_nn}_mean"
        if col in summary.columns:
            display_cols.append(col)

    sub = summary[summary["representation"].isin(primary_reps)].copy()
    available = [c for c in display_cols if c in sub.columns]
    sub_display = sub[available].copy()
    for col in sub_display.columns:
        if "mean" in col and col != "representation":
            sub_display[col] = sub_display[col].map(
                lambda v: f"{float(v):.6f}" if pd.notna(v) else "NA"
            )
    # Sort to keep frozen first, then linear, then neural
    rep_order = {r: i for i, r in enumerate(primary_reps)}
    sub_display["_order"] = sub_display["representation"].map(lambda r: rep_order.get(r, 999))
    sub_display = sub_display.sort_values("_order").drop(columns=["_order"])
    lines.append(sub_display.to_string(index=False))
    lines.append("")

    # Key findings
    lines.extend([
        "",
        "## Branch Separation Analysis",
        "",
    ])

    # Paired: biological should have lower scanner, higher category than acquisition
    for condition, bio_name, acq_name in [
        ("true_pair", "true_pair_biological", "true_pair_acquisition"),
        ("shuffled_sample", "shuffled_sample_biological", "shuffled_sample_acquisition"),
    ]:
        bio_sub = summary[summary["representation"] == bio_name]
        acq_sub = summary[summary["representation"] == acq_name]
        if bio_sub.empty or acq_sub.empty:
            continue
        bio = bio_sub.iloc[0]
        acq = acq_sub.iloc[0]

        lines.append(f"### {condition} paired-acquisition")
        lines.append("")
        lines.append(
            f"- Biological scanner acc: {float(bio['scanner_balanced_accuracy_mean']):.4f}  |  "
            f"Acquisition scanner acc: {float(acq['scanner_balanced_accuracy_mean']):.4f}"
        )
        lines.append(
            f"- Biological category acc: {float(bio['category_balanced_accuracy_mean']):.4f}  |  "
            f"Acquisition category acc: {float(acq['category_balanced_accuracy_mean']):.4f}"
        )
        # Leakage: acquisition branch should have low category signal
        acq_cat_leak = float(acq["category_balanced_accuracy_mean"])
        lines.append(f"- Acquisition branch category leakage: {acq_cat_leak:.4f}")
        lines.append("")

        # Check if separation is clean
        bio_scanner_lower = float(bio["scanner_balanced_accuracy_mean"]) < float(acq["scanner_balanced_accuracy_mean"])
        bio_cat_higher = float(bio["category_balanced_accuracy_mean"]) > float(acq["category_balanced_accuracy_mean"])
        lines.append(f"- Biological has lower scanner than acquisition: {bio_scanner_lower}")
        lines.append(f"- Biological has higher category than acquisition: {bio_cat_higher}")
        lines.append("")

    # Linear: keep should have lower scanner, higher category; removed should have high scanner, low category
    for k in config.k_values:
        keep_sub = summary[summary["representation"] == f"linear_keep_k{k}"]
        rem_sub = summary[summary["representation"] == f"linear_removed_k{k}"]
        if keep_sub.empty or rem_sub.empty:
            continue
        keep = keep_sub.iloc[0]
        rem = rem_sub.iloc[0]

        lines.append(f"### linear decomposition k={k}")
        lines.append("")
        lines.append(
            f"- Keep scanner acc: {float(keep['scanner_balanced_accuracy_mean']):.4f}  |  "
            f"Removed scanner acc: {float(rem['scanner_balanced_accuracy_mean']):.4f}"
        )
        lines.append(
            f"- Keep category acc: {float(keep['category_balanced_accuracy_mean']):.4f}  |  "
            f"Removed category acc: {float(rem['category_balanced_accuracy_mean']):.4f}"
        )
        # Leakage: removed should have low category signal
        rem_cat_leak = float(rem["category_balanced_accuracy_mean"])
        lines.append(f"- Removed residual category leakage: {rem_cat_leak:.4f}")
        lines.append("")

    # Direct comparison
    lines.extend([
        "",
        "## Direct Comparison: Paired vs Linear Separation Quality",
        "",
        "A cleaner separation means:",
        "- The 'keep' or 'biological' branch has low scanner and high category signal.",
        "- The 'removed' or 'acquisition' branch has high scanner and low category signal.",
        "- Lower leakage = better separation.",
        "",
    ])

    # Paired leakage
    tpb_sub = summary[summary["representation"] == "true_pair_biological"]
    tpa_sub = summary[summary["representation"] == "true_pair_acquisition"]
    if not tpb_sub.empty and not tpa_sub.empty:
        tpb = tpb_sub.iloc[0]
        tpa = tpa_sub.iloc[0]
        paired_acq_cat_leak = float(tpa["category_balanced_accuracy_mean"])
        lines.append(f"true_pair acquisition branch category leakage: {paired_acq_cat_leak:.4f}")

        # Compare to linear k=4 (best scanner separation)
        lin4_rem = summary[summary["representation"] == "linear_removed_k4"]
        if not lin4_rem.empty:
            lin4 = lin4_rem.iloc[0]
            linear_rem_cat_leak = float(lin4["category_balanced_accuracy_mean"])
            lines.append(f"linear_removed_k4 category leakage: {linear_rem_cat_leak:.4f}")

            if paired_acq_cat_leak < linear_rem_cat_leak:
                lines.append(
                    f"- Paired acquisition leaks less category signal than linear removed "
                    f"({paired_acq_cat_leak:.4f} < {linear_rem_cat_leak:.4f})."
                )
            else:
                lines.append(
                    f"- Linear removed leaks less or equal category signal than paired acquisition "
                    f"({linear_rem_cat_leak:.4f} <= {paired_acq_cat_leak:.4f})."
                )

    lines.extend([
        "",
        "### Scanner/Category Signal Separation Summary",
        "",
    ])

    # best linear k for separation
    best_sep_k = None
    best_sep_score = -1
    for k in config.k_values:
        keep_sub = summary[summary["representation"] == f"linear_keep_k{k}"]
        rem_sub = summary[summary["representation"] == f"linear_removed_k{k}"]
        if keep_sub.empty or rem_sub.empty:
            continue
        keep = keep_sub.iloc[0]
        rem = rem_sub.iloc[0]
        # Separation quality: keep should have high category, low scanner;
        # removed should have high scanner, low category
        sep_score = (
            float(keep["category_balanced_accuracy_mean"])
            - float(keep["scanner_balanced_accuracy_mean"])
            + float(rem["scanner_balanced_accuracy_mean"])
            - float(rem["category_balanced_accuracy_mean"])
        )
        if sep_score > best_sep_score:
            best_sep_score = sep_score
            best_sep_k = k

    if best_sep_k is not None:
        lines.append(f"- Best linear separation k: {best_sep_k} (separation score: {best_sep_score:.4f})")

    lines.extend([
        "",
        "## Key Questions",
        "",
    ])

    # Answer key questions
    frozen_sub = summary[summary["representation"] == "original_frozen_features"]
    tpb_sub = summary[summary["representation"] == "true_pair_biological"]
    tpa_sub = summary[summary["representation"] == "true_pair_acquisition"]

    if not frozen_sub.empty and not tpb_sub.empty:
        frozen = frozen_sub.iloc[0]
        tpb = tpb_sub.iloc[0]

        frozen_scanner = float(frozen["scanner_balanced_accuracy_mean"])
        tpb_scanner = float(tpb["scanner_balanced_accuracy_mean"])
        lines.append(
            f"1. Does true_pair_biological have lower scanner signal than frozen? "
            f"{'Yes' if tpb_scanner < frozen_scanner else 'No'} "
            f"(true_pair_bio={tpb_scanner:.4f}, frozen={frozen_scanner:.4f})"
        )

    if not tpa_sub.empty:
        tpa = tpa_sub.iloc[0]
        tpa_scanner = float(tpa["scanner_balanced_accuracy_mean"])
        lines.append(f"2. Does true_pair_acquisition carry scanner signal? Scanner acc={tpa_scanner:.4f}")

    if not tpb_sub.empty and not tpa_sub.empty:
        tpb_cat = float(tpb["category_balanced_accuracy_mean"])
        tpa_cat = float(tpa["category_balanced_accuracy_mean"])
        lines.append(
            f"3. Does true_pair_acquisition have lower category signal than biological? "
            f"{'Yes' if tpa_cat < tpb_cat else 'No'} "
            f"(bio_cat={tpb_cat:.4f}, acq_cat={tpa_cat:.4f})"
        )

    lin4_keep = summary[summary["representation"] == "linear_keep_k4"]
    lin4_rem = summary[summary["representation"] == "linear_removed_k4"]
    if not lin4_keep.empty and not lin4_rem.empty:
        lk4 = lin4_keep.iloc[0]
        lr4 = lin4_rem.iloc[0]
        lines.append(
            f"4. Does linear_removed_k4 carry scanner signal? Scanner acc={float(lr4['scanner_balanced_accuracy_mean']):.4f}"
        )
        lines.append(
            f"5. Does linear_removed_k4 also leak category signal? Category acc={float(lr4['category_balanced_accuracy_mean']):.4f}"
        )

    # Cleaner separation comparison
    if not tpa_sub.empty and not lin4_rem.empty:
        paired_leak = float(tpa["category_balanced_accuracy_mean"])
        linear_leak = float(lr4["category_balanced_accuracy_mean"])
        if paired_leak < linear_leak:
            lines.append(
                f"6. Paired-acquisition produces cleaner branch separation in this audit: "
                f"acquisition category leakage ({paired_leak:.4f}) < linear removed leakage ({linear_leak:.4f})."
            )
        elif paired_leak > linear_leak:
            lines.append(
                f"6. Linear residual decomposition produces cleaner or comparable branch separation: "
                f"linear removed leakage ({linear_leak:.4f}) <= paired acquisition leakage ({paired_leak:.4f})."
            )
        else:
            lines.append(
                f"6. Paired and linear have equal branch separation leakage ({paired_leak:.4f})."
            )

    lines.append(
        "7. Is the linear baseline sufficient to explain the paired-acquisition result? See interpretation below."
    )

    lines.extend([
        "",
        "## Bounded Interpretation",
        "",
        "This is a branch-separation audit. It does not claim clinical validation, diagnostic",
        "performance, patient-care utility, deployment readiness, or that scanner bias is solved.",
        "",
        "If paired-acquisition's acquisition branch carries less category signal than the linear",
        "removed residual, this supports the interpretation that neural factorization produces",
        "cleaner structured separation than a simple linear scanner-subspace decomposition in",
        "this audit.",
        "",
        "If the linear removed residual leaks comparable or less category signal, this suggests",
        "that a linear scanner-subspace split is sufficient to explain the branch-separation",
        "behavior — an important honesty check.",
        "",
        "The key metric is acquisition/removed branch category leakage: which branch carries",
        "less biological category signal while still capturing scanner information.",
        "",
    ])

    lines.extend([
        "",
        "## Validation",
        "",
        f"- Total raw rows: {len(raw)}",
        f"- Validation issues: {len(issues)}",
    ])
    if issues:
        for issue in issues:
            lines.append(f"  - {issue}")
    else:
        lines.append("  - No validation issues found.")

    lines.extend([
        "",
        "## Output Files",
        "",
        f"- linear_residual_raw_metrics.csv ({len(raw)} rows)",
        f"- linear_residual_summary.csv",
        f"- linear_residual_branch_contrasts.csv ({len(contrasts)} rows)",
        f"- linear_residual_branch_separation_report.md",
        f"- experiment_design.json",
        f"- run_log.txt",
        "",
        f"Runtime: {runtime_seconds:.1f}s",
        "",
        "## Readiness",
        "",
        "Pending validation.",
        "",
    ])

    return "\n".join(lines)


def _get_scanner_rank(raw: pd.DataFrame, config: ExperimentConfig) -> str:
    """Extract the fitted scanner rank from the max effective k across all k values."""
    sub = raw[raw["representation"].str.startswith("linear_keep_k")]
    if sub.empty:
        return "unknown"
    k_vals = sub["k_value"].dropna().unique()
    if len(k_vals) > 0:
        return str(int(max(k_vals)))
    return "unknown"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/paired_acquisition_factorization_linear_residual_branch_separation_audit"),
    )
    parser.add_argument("--folds", nargs="+", type=int, default=list(FOLDS))
    parser.add_argument("--neural-seeds", nargs="+", type=int, default=list(PAIR_INTEGRITY_SEEDS))
    parser.add_argument("--k-values", nargs="+", type=int, default=list(K_VALUES))
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
        k_values=tuple(args.k_values),
    )

    design = {
        "stage": "paired_acquisition_factorization_linear_residual_branch_separation_audit",
        "dataset": "canine_cutaneous_scc_dinov2",
        "folds": list(config.folds),
        "pair_integrity_seeds": list(config.pair_integrity_seeds),
        "k_values": list(config.k_values),
        "neighborhood_k_values": list(NEIGHBORHOOD_K),
        "linear_split_formula": {
            "keep": "features - features @ basis[:N].T @ basis[:N]",
            "removed": "features @ basis[:N].T @ basis[:N]",
            "basis": "SVD of centered logistic regression coefficient matrix",
        },
        "neural_representations": list(NEURAL_REP_METADATA.keys()),
        "linear_representations": [
            f"linear_keep_k{k}" for k in config.k_values
        ] + [
            f"linear_removed_k{k}" for k in config.k_values
        ],
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

                issues = validate_outputs(raw, config)
                summary = build_summary(raw)
                contrasts = build_branch_contrasts(raw)

                # Write outputs
                atomic_csv(args.out_dir / "linear_residual_raw_metrics.csv", raw)
                atomic_csv(args.out_dir / "linear_residual_summary.csv", summary)
                atomic_csv(args.out_dir / "linear_residual_branch_contrasts.csv", contrasts)

                runtime = time.perf_counter() - start
                report = build_report(raw, summary, contrasts, config, issues, runtime)
                atomic_text(args.out_dir / "linear_residual_branch_separation_report.md", report)

                print("\n" + "=" * 80)
                print("LINEAR RESIDUAL BRANCH-SEPARATION AUDIT COMPLETE")
                print(f"Rows: {len(raw)}")
                print(f"Runtime: {runtime:.1f}s")
                print(f"Validation issues: {len(issues)}")
                for issue in issues:
                    print(f"  - {issue}")
                print(f"Report: {(args.out_dir / 'linear_residual_branch_separation_report.md').resolve()}")

                # Print key findings
                tpb = summary[summary["representation"] == "true_pair_biological"]
                tpa = summary[summary["representation"] == "true_pair_acquisition"]
                lin4k = summary[summary["representation"] == "linear_keep_k4"]
                lin4r = summary[summary["representation"] == "linear_removed_k4"]
                frozen = summary[summary["representation"] == "original_frozen_features"]

                if not tpb.empty and not tpa.empty:
                    print(f"\ntrue_pair_bio: scanner={float(tpb.iloc[0]['scanner_balanced_accuracy_mean']):.4f} cat={float(tpb.iloc[0]['category_balanced_accuracy_mean']):.4f}")
                    print(f"true_pair_acq: scanner={float(tpa.iloc[0]['scanner_balanced_accuracy_mean']):.4f} cat={float(tpa.iloc[0]['category_balanced_accuracy_mean']):.4f}")
                if not lin4k.empty and not lin4r.empty:
                    print(f"linear_keep_k4: scanner={float(lin4k.iloc[0]['scanner_balanced_accuracy_mean']):.4f} cat={float(lin4k.iloc[0]['category_balanced_accuracy_mean']):.4f}")
                    print(f"linear_removed_k4: scanner={float(lin4r.iloc[0]['scanner_balanced_accuracy_mean']):.4f} cat={float(lin4r.iloc[0]['category_balanced_accuracy_mean']):.4f}")

            except Exception:
                traceback.print_exc()
                raise


if __name__ == "__main__":
    main()
