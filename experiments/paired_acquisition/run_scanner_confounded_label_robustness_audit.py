#!/usr/bin/env python3
"""Scanner-confounded label robustness audit for paired-acquisition factorization.

Tests whether true_pair_biological resists scanner-category shortcut learning
better than original frozen features, acquisition branch, shuffled controls,
PCA removal, and linear scanner projection.

The experiment creates artificial training splits where scanner and category are
deliberately confounded, then measures which representations maintain category
classification accuracy on a balanced held-out evaluation set.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, confusion_matrix
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
CONFOUNDING_STRENGTHS = ("mild", "moderate", "severe")
CONFOUNDING_SEEDS = (2001, 2002, 2003, 2004, 2005)
K_VALUES = (0, 1, 2, 4, 8, 16, 32)

FEATURE_PATH = Path("results/external_multiscanner_caninescc/features/fold_0_dinov2_base.npz")
MANIFESTS_DIR = Path("data/external_multiscanner_caninescc/patch_manifests/splits")
PAIR_INTEGRITY_DIR = Path("results/paired_acquisition_factorization_pair_integrity_caninescc")

# Confounding: fraction of training patches retained from the *assigned* scanners
# for each category.  The remainder comes from non-assigned scanners.
CONFOUNDING_RETAIN_FRACTIONS = {
    "mild": 0.60,
    "moderate": 0.80,
    "severe": 0.95,
}

# Rare-class minimum: categories with fewer than this many total fit-regions
# per scanner get special handling.
RARE_CLASS_MIN_REGIONS = 5

REPRESENTATION_METADATA = {
    "original_frozen_features": {
        "family": "frozen",
        "dim": 768,
        "label": "Original frozen DINOv2 features",
    },
    "true_pair_biological": {
        "family": "neural_factorization",
        "dim": 256,
        "label": "True-pair biological branch",
    },
    "true_pair_acquisition": {
        "family": "neural_factorization",
        "dim": 64,
        "label": "True-pair acquisition branch",
    },
    "shuffled_sample_biological": {
        "family": "neural_factorization",
        "dim": 256,
        "label": "Shuffled-sample biological branch",
    },
    "shuffled_sample_acquisition": {
        "family": "neural_factorization",
        "dim": 64,
        "label": "Shuffled-sample acquisition branch",
    },
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


def mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    """Estimate mutual information between two categorical variables (in nats)."""
    n = len(x)
    if n <= 1:
        return 0.0
    x_vals, x_inv = np.unique(x, return_inverse=True)
    y_vals, y_inv = np.unique(y, return_inverse=True)
    joint = np.zeros((len(x_vals), len(y_vals)), dtype=np.float64)
    np.add.at(joint, (x_inv, y_inv), 1)
    joint /= n
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    mi = 0.0
    for i in range(len(x_vals)):
        for j in range(len(y_vals)):
            if joint[i, j] > 0:
                mi += joint[i, j] * math.log(joint[i, j] / (px[i] * py[j]))
    return float(max(0.0, mi))


def numeric_rank(singular_values: np.ndarray, shape: tuple[int, int]) -> int:
    if singular_values.size == 0:
        return 0
    tolerance = max(shape) * np.finfo(np.float64).eps * float(singular_values[0])
    return int(np.sum(singular_values > tolerance))


# ---------------------------------------------------------------------------
# Data loading helpers
# ---------------------------------------------------------------------------


def load_manifest(fold: int) -> pd.DataFrame:
    path = MANIFESTS_DIR / f"fold_{fold}_patch_manifest.csv"
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


def load_pair_integrity_features(
    fold: int, seed: int, condition: str
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Return (biological, acquisition, frame) for a pair-integrity run."""
    projected = (
        PAIR_INTEGRITY_DIR / f"fold_{fold}" / "runs" / f"{condition}_seed_{seed}" / "projected_features.npz"
    )
    if not projected.is_file():
        raise projection.ExperimentError(f"Missing pair-integrity features: {projected}")
    biological, acquisition, frame, metadata = canine_pair.load_projected(projected)
    return biological, acquisition, frame


def _align_pair_integrity_features(
    features: np.ndarray,
    pair_frame: pd.DataFrame,
    manifest: pd.DataFrame,
) -> np.ndarray:
    """Align pair-integrity projected features to manifest row order.

    Pair-integrity frames lack the ``path`` column required by
    ``canine_cross.align_fold``, so we align via (slide_id, region_id, scanner_id).
    """
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


def representation_features(
    representation: str,
    *,
    fold: int,
    seed: int | None,
    frozen_features: np.ndarray,
    frozen_frame: pd.DataFrame,
    manifest: pd.DataFrame,
    fit: np.ndarray,
) -> np.ndarray:
    """Return the feature matrix for a given representation, fold, and seed."""
    if representation == "original_frozen_features":
        aligned_features, _ = canine_cross.align_fold(frozen_features, frozen_frame, _manifest_path(fold))
        return aligned_features

    if representation in ("true_pair_biological", "true_pair_acquisition"):
        condition = "true_pairs"
    elif representation in ("shuffled_sample_biological", "shuffled_sample_acquisition"):
        condition = "shuffled_sample_pairs"
    else:
        raise projection.ExperimentError(f"Unknown representation: {representation}")

    bio, acq, frame = load_pair_integrity_features(fold, int(seed), condition)
    manifest_for_align = load_manifest(fold)

    if "biological" in representation:
        return _align_pair_integrity_features(bio, frame, manifest_for_align)
    return _align_pair_integrity_features(acq, frame, manifest_for_align)


def _manifest_path(fold: int) -> Path:
    return MANIFESTS_DIR / f"fold_{fold}_patch_manifest.csv"


# ---------------------------------------------------------------------------
# PCA / linear scanner projection (computed on-the-fly)
# ---------------------------------------------------------------------------


def compute_pca_removal(
    features: np.ndarray, fit: np.ndarray, k: int
) -> np.ndarray:
    """Remove top-k PCA components from standardized features."""
    standardized, mean, std = projection.standardize(features, fit)
    fit_features = np.asarray(standardized[fit], dtype=np.float64)
    center = fit_features.mean(axis=0, keepdims=True)
    centered = fit_features - center
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    rank = numeric_rank(singular_values, centered.shape)
    effective_k = min(int(k), rank)
    if effective_k <= 0:
        return standardized.copy()
    matrix = np.asarray(standardized, dtype=np.float64)
    residual = matrix - (matrix @ vt[:effective_k].T) @ vt[:effective_k]
    residual = residual.astype(np.float32)
    if not np.isfinite(residual).all():
        raise projection.ExperimentError(f"PCA k={k} produced nonfinite values.")
    return residual


def compute_linear_projection(
    features: np.ndarray, frame: pd.DataFrame, fit: np.ndarray, k: int
) -> np.ndarray:
    """Remove top-k linear scanner-discriminative directions from standardized features."""
    standardized, mean, std = projection.standardize(features, fit)
    labels = frame["scanner_id"].astype(str).to_numpy()
    model = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=5000, random_state=0, solver="lbfgs",
    )
    model.fit(standardized[fit], labels[fit])
    coefficients = np.asarray(model.coef_, dtype=np.float64)
    coefficients = coefficients - coefficients.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(coefficients, full_matrices=False)
    rank = numeric_rank(singular_values, coefficients.shape)
    effective_k = min(int(k), rank)
    if effective_k <= 0:
        return standardized.copy()
    basis = vt[:effective_k].astype(np.float64)
    matrix = np.asarray(standardized, dtype=np.float64)
    residual = matrix - (matrix @ basis.T) @ basis
    residual = residual.astype(np.float32)
    if not np.isfinite(residual).all():
        raise projection.ExperimentError(f"Linear projection k={k} produced nonfinite values.")
    return residual


# ---------------------------------------------------------------------------
# Confounded split construction
# ---------------------------------------------------------------------------


def build_confounded_training_set(
    manifest: pd.DataFrame,
    fit_indices: np.ndarray,
    *,
    strength: str,
    confounding_seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    """Build a confounded training subset from the fit indices.

    For each category, assigns 2 scanners as the "confounded" scanners and retains
    `retain_fraction` of those patches while keeping only the complementary fraction
    from non-assigned scanners.

    Returns (selected_indices, diagnostics_df).
    """
    retain = CONFOUNDING_RETAIN_FRACTIONS[strength]
    rng = np.random.default_rng(confounding_seed)
    fit_df = manifest.iloc[fit_indices].copy()
    fit_df["_orig_idx"] = fit_indices

    categories = sorted(fit_df["category_name"].unique())
    scanners = sorted(fit_df["scanner_id"].unique())

    selected = []
    diagnostics = []

    for category in categories:
        cat_df = fit_df[fit_df["category_name"] == category]
        cat_scanners = sorted(cat_df["scanner_id"].unique())

        # Assign 2 scanners as "confounded" for this category
        n_assigned = min(2, len(cat_scanners))
        assigned_scanners = list(rng.choice(cat_scanners, size=n_assigned, replace=False))
        other_scanners = [s for s in cat_scanners if s not in assigned_scanners]

        assigned_df = cat_df[cat_df["scanner_id"].isin(assigned_scanners)]
        other_df = cat_df[cat_df["scanner_id"].isin(other_scanners)]

        # Retain most from assigned, fewer from others
        n_assigned_keep = max(1, int(len(assigned_df) * retain)) if len(assigned_df) > 0 else 0
        n_other_keep = max(
            1 if len(other_df) > 0 else 0,
            int(len(other_df) * (1.0 - retain)),
        ) if len(other_df) > 0 else 0

        assigned_keep = rng.choice(assigned_df.index, size=n_assigned_keep, replace=False).tolist() if n_assigned_keep > 0 else []
        other_keep = rng.choice(other_df.index, size=n_other_keep, replace=False).tolist() if n_other_keep > 0 else []

        kept_indices = assigned_keep + other_keep
        selected.extend(kept_indices)

        # Per-category diagnostic
        for s in scanners:
            scanner_cat = cat_df[cat_df["scanner_id"] == s]
            n_total = len(scanner_cat)
            n_kept = len([i for i in kept_indices if i in scanner_cat.index])
            diagnostics.append({
                "category": category,
                "scanner": s,
                "is_assigned": s in assigned_scanners,
                "n_total_fit": int(n_total),
                "n_kept": int(n_kept),
                "retain_fraction_actual": float(n_kept / max(1, n_total)),
            })

    selected_indices = fit_df.loc[selected, "_orig_idx"].to_numpy(dtype=np.int64)
    diag_df = pd.DataFrame(diagnostics)
    return selected_indices, diag_df


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------


def evaluate_category_classifier(
    features: np.ndarray,
    manifest: pd.DataFrame,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
) -> dict[str, object]:
    """Train a category classifier and evaluate on the test set."""
    y_train = manifest.iloc[train_indices]["category_name"].to_numpy()
    y_test = manifest.iloc[test_indices]["category_name"].to_numpy()

    X_train = features[train_indices]
    X_test = features[test_indices]

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=0),
    )
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    balanced_acc = float(balanced_accuracy_score(y_test, y_pred))
    macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
    weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))

    categories = sorted(np.unique(y_test))
    per_class = {}
    for cat in categories:
        mask = y_test == cat
        if mask.sum() > 0:
            per_class[f"recall_{cat}"] = float(np.mean(y_pred[mask] == cat))
        else:
            per_class[f"recall_{cat}"] = float("nan")

    # Scanner prediction from classifier errors
    test_scanners = manifest.iloc[test_indices]["scanner_id"].to_numpy()
    error_mask = y_pred != y_test
    scanner_error = {}
    for s in sorted(np.unique(test_scanners)):
        s_mask = test_scanners == s
        if s_mask.sum() > 0:
            scanner_error[f"error_rate_{s}"] = float(error_mask[s_mask].mean())
        else:
            scanner_error[f"error_rate_{s}"] = float("nan")

    # Error concentration by scanner
    if error_mask.sum() > 0:
        scanner_error_dist = {}
        for s in sorted(np.unique(test_scanners)):
            errors_on_s = error_mask[s_mask].sum()
            scanner_error_dist[s] = int(errors_on_s)
        total_errors = int(error_mask.sum())
        max_share = max(scanner_error_dist.values()) / max(1, total_errors)
    else:
        total_errors = 0
        max_share = 0.0

    return {
        "balanced_accuracy": balanced_acc,
        "macro_f1": macro_f1,
        "weighted_f1": weighted_f1,
        "n_test": int(len(test_indices)),
        "n_test_errors": total_errors,
        "max_scanner_error_share": float(max_share),
        **per_class,
        **scanner_error,
    }


# ---------------------------------------------------------------------------
# Main experiment
# ---------------------------------------------------------------------------


@dataclass
class ExperimentConfig:
    out_dir: Path
    folds: tuple[int, ...] = FOLDS
    pair_integrity_seeds: tuple[int, ...] = PAIR_INTEGRITY_SEEDS
    confounding_strengths: tuple[str, ...] = CONFOUNDING_STRENGTHS
    confounding_seeds: tuple[int, ...] = CONFOUNDING_SEEDS
    k_values: tuple[int, ...] = K_VALUES


def run_experiment(config: ExperimentConfig) -> dict[str, object]:
    config.out_dir.mkdir(parents=True, exist_ok=True)

    # Load frozen features once
    frozen_features, frozen_frame = load_frozen_features()
    canine_cross.patch_scanner_namespace()

    raw_rows: list[dict[str, object]] = []
    split_diag_rows: list[dict[str, object]] = []

    for fold in config.folds:
        manifest = load_manifest(fold)
        aligned_frozen, _ = canine_cross.align_fold(frozen_features, frozen_frame, _manifest_path(fold))
        fit_indices, test_indices = canine_cross.validate_fold(manifest, fold)

        print(
            f"\nFold {fold}: fit={len(fit_indices)} test={len(test_indices)} "
            f"categories={sorted(manifest['category_name'].unique())}"
        )

        # Pre-compute PCA and linear projection features for all k values
        pca_features: dict[int, np.ndarray] = {}
        linear_features: dict[int, np.ndarray] = {}
        for k in config.k_values:
            pca_features[k] = compute_pca_removal(aligned_frozen, fit_indices, k)
            linear_features[k] = compute_linear_projection(aligned_frozen, manifest, fit_indices, k)

        # Frozen features and simple baselines (single source, no neural seed)
        simple_representations: dict[str, np.ndarray] = {
            "original_frozen_features": aligned_frozen,
        }
        for k in config.k_values:
            simple_representations[f"pca_removal_k{k}"] = pca_features[k]
            simple_representations[f"linear_projection_k{k}"] = linear_features[k]

        for rep_name, features in simple_representations.items():
            for cseed in config.confounding_seeds:
                for strength in config.confounding_strengths:
                    confounded_train, split_diag = build_confounded_training_set(
                        manifest, fit_indices, strength=strength, confounding_seed=cseed
                    )
                    # Record split diagnostics once per fold/seed/strength
                    if rep_name == "original_frozen_features":
                        for row in split_diag.to_dict("records"):
                            row.update({"fold": int(fold), "confounding_seed": int(cseed), "strength": strength})
                            split_diag_rows.append(row)

                    metrics = evaluate_category_classifier(
                        features, manifest, confounded_train, test_indices
                    )
                    raw_rows.append({
                        "dataset": "caninescc",
                        "representation": rep_name,
                        "representation_family": _rep_family(rep_name),
                        "fold": int(fold),
                        "neural_seed": "",
                        "confounding_seed": int(cseed),
                        "confounding_strength": strength,
                        **metrics,
                    })

        # Neural factorization representations (per neural seed)
        for neural_seed in config.pair_integrity_seeds:
            for condition, bio_name, acq_name in [
                ("true_pairs", "true_pair_biological", "true_pair_acquisition"),
                ("shuffled_sample_pairs", "shuffled_sample_biological", "shuffled_sample_acquisition"),
            ]:
                bio, acq, pair_frame = load_pair_integrity_features(fold, neural_seed, condition)
                aligned_bio = _align_pair_integrity_features(bio, pair_frame, manifest)
                aligned_acq = _align_pair_integrity_features(acq, pair_frame, manifest)

                for rep_name, features in [(bio_name, aligned_bio), (acq_name, aligned_acq)]:
                    for cseed in config.confounding_seeds:
                        for strength in config.confounding_strengths:
                            confounded_train, _ = build_confounded_training_set(
                                manifest, fit_indices, strength=strength, confounding_seed=cseed
                            )
                            metrics = evaluate_category_classifier(
                                features, manifest, confounded_train, test_indices
                            )
                            raw_rows.append({
                                "dataset": "caninescc",
                                "representation": rep_name,
                                "representation_family": _rep_family(rep_name),
                                "fold": int(fold),
                                "neural_seed": int(neural_seed),
                                "confounding_seed": int(cseed),
                                "confounding_strength": strength,
                                **metrics,
                            })

    raw = pd.DataFrame(raw_rows).sort_values([
        "representation_family", "representation", "fold", "neural_seed",
        "confounding_strength", "confounding_seed",
    ]).reset_index(drop=True)

    split_diag = pd.DataFrame(split_diag_rows)

    return {"raw": raw, "split_diagnostics": split_diag}


def _rep_family(name: str) -> str:
    if name in REPRESENTATION_METADATA:
        return REPRESENTATION_METADATA[name]["family"]
    if name.startswith("pca_"):
        return "pca_removal"
    if name.startswith("linear_"):
        return "linear_projection"
    return "unknown"


# ---------------------------------------------------------------------------
# Summarization
# ---------------------------------------------------------------------------


def build_summary(raw: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [
        "balanced_accuracy", "macro_f1", "weighted_f1",
        "n_test_errors", "max_scanner_error_share",
    ]
    # Add per-class recall columns
    recall_cols = [c for c in raw.columns if c.startswith("recall_")]
    error_rate_cols = [c for c in raw.columns if c.startswith("error_rate_")]
    all_metric_cols = [c for c in metric_cols + recall_cols + error_rate_cols if c in raw.columns]

    group_cols = ["representation", "representation_family", "confounding_strength"]
    grouped = raw.groupby(group_cols, dropna=False)
    metric_summary = grouped[all_metric_cols].agg(["mean", "std", "min", "max"])
    metric_summary.columns = ["_".join(col).strip("_") for col in metric_summary.columns]

    counts = grouped.agg(
        n_runs=("fold", "size"),
        n_folds=("fold", "nunique"),
        n_confounding_seeds=("confounding_seed", "nunique"),
        n_neural_seeds=("neural_seed", lambda x: x[x.astype(str).str.strip() != ""].nunique()),
    )
    summary = counts.join(metric_summary).reset_index()
    return summary.sort_values(["confounding_strength", "representation_family", "representation"])


def build_per_class_recall(raw: pd.DataFrame) -> pd.DataFrame:
    recall_cols = ["representation", "representation_family", "confounding_strength"]
    recall_metrics = [c for c in raw.columns if c.startswith("recall_")]
    if not recall_metrics:
        return pd.DataFrame()
    melted = raw.melt(
        id_vars=recall_cols,
        value_vars=recall_metrics,
        var_name="class",
        value_name="recall",
    )
    melted["class"] = melted["class"].str.replace("recall_", "", regex=False)
    return melted.groupby(
        ["representation", "representation_family", "confounding_strength", "class"],
        dropna=False,
    )["recall"].agg(["mean", "std", "count"]).reset_index()


def build_per_scanner_errors(raw: pd.DataFrame) -> pd.DataFrame:
    error_cols = ["representation", "representation_family", "confounding_strength"]
    error_metrics = [c for c in raw.columns if c.startswith("error_rate_")]
    if not error_metrics:
        return pd.DataFrame()
    melted = raw.melt(
        id_vars=error_cols,
        value_vars=error_metrics,
        var_name="scanner",
        value_name="error_rate",
    )
    melted["scanner"] = melted["scanner"].str.replace("error_rate_", "", regex=False)
    return melted.groupby(
        ["representation", "representation_family", "confounding_strength", "scanner"],
        dropna=False,
    )["error_rate"].agg(["mean", "std", "count"]).reset_index()


# ---------------------------------------------------------------------------
# Train/test MI computation
# ---------------------------------------------------------------------------


def compute_split_mi(manifest: pd.DataFrame, train_indices: np.ndarray, test_indices: np.ndarray) -> dict[str, float]:
    """Compute scanner/category mutual information for train and test splits."""
    y_train = manifest.iloc[train_indices]["category_name"].to_numpy()
    s_train = manifest.iloc[train_indices]["scanner_id"].to_numpy()
    y_test = manifest.iloc[test_indices]["category_name"].to_numpy()
    s_test = manifest.iloc[test_indices]["scanner_id"].to_numpy()
    return {
        "train_scanner_category_mi": mutual_information(s_train, y_train),
        "test_scanner_category_mi": mutual_information(s_test, y_test),
    }


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_outputs(raw: pd.DataFrame, split_diag: pd.DataFrame, config: ExperimentConfig) -> list[str]:
    issues: list[str] = []

    required_metrics = ["balanced_accuracy", "macro_f1", "weighted_f1"]
    for metric in required_metrics:
        values = pd.to_numeric(raw[metric], errors="coerce")
        if values.isna().any():
            issues.append(f"Missing values in {metric}")

        finite_mask = np.isfinite(values.to_numpy(float))
        if not finite_mask.all():
            bad_count = (~finite_mask).sum()
            issues.append(f"Nonfinite values in {metric}: {bad_count} rows")

    # Check for duplicate rows
    dup_cols = ["representation", "fold", "neural_seed", "confounding_seed", "confounding_strength"]
    available_dup_cols = [c for c in dup_cols if c in raw.columns]
    if raw.duplicated(available_dup_cols).any():
        n_dup = raw.duplicated(available_dup_cols).sum()
        issues.append(f"Duplicate rows found: {n_dup}")

    # Check expected row counts
    n_simple_reps = 1 + 2 * len(config.k_values)  # frozen + pca k*s + linear k*s
    n_neural_reps = 4  # true_pair_bio, true_pair_acq, shuffled_sample_bio, shuffled_sample_acq
    n_neural_sources = len(config.folds) * len(config.pair_integrity_seeds)
    expected_simple = n_simple_reps * len(config.folds) * len(config.confounding_seeds) * len(config.confounding_strengths)
    expected_neural = n_neural_reps * n_neural_sources * len(config.confounding_seeds) * len(config.confounding_strengths)
    expected_total = expected_simple + expected_neural
    if len(raw) != expected_total:
        issues.append(
            f"Expected {expected_total} rows (simple={expected_simple}, neural={expected_neural}); "
            f"observed {len(raw)}"
        )

    # Check confounding strengths covered
    observed_strengths = set(raw["confounding_strength"].unique())
    expected_strengths = set(config.confounding_strengths)
    if observed_strengths != expected_strengths:
        issues.append(f"Confounding strength mismatch: {observed_strengths} vs {expected_strengths}")

    return issues


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def build_report(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    per_class: pd.DataFrame,
    per_scanner: pd.DataFrame,
    split_diag: pd.DataFrame,
    config: ExperimentConfig,
    validation_issues: list[str],
    runtime_seconds: float,
) -> str:
    lines = [
        "# Scanner-Confounded Label Robustness Audit",
        "",
        "## Branch",
        "",
        "experiment/scanner-confounded-label-robustness-audit",
        "",
        "## Question",
        "",
        "Does true_pair_biological resist scanner-category shortcut learning better than "
        "original frozen features, acquisition branch, shuffled controls, PCA removal, and "
        "linear scanner projection when training on deliberately confounded splits?",
        "",
        "## Dataset",
        "",
        "- Canine cutaneous SCC, DINOv2-Base frozen features",
        "- 4025 patches, 44 samples, 5 scanners, 805 regions",
        "- 7 categories: Bone, Cartilage, Dermis, Epidermis, Inflamm/Necrosis, SCC, Subcutis",
        f"- {len(config.folds)} folds, {len(config.pair_integrity_seeds)} neural seeds",
        "",
        "## Confounding Design",
        "",
        "For each fold, confounding seed, and strength:",
        "1. Each category is randomly assigned to 2 of the 5 scanners as its 'confounded' scanners.",
        "2. Training patches from the assigned scanners are retained at the confounding fraction;",
        "   patches from non-assigned scanners are kept at the complementary fraction.",
        "3. The held-out test set is left unmodified (sample-disjoint from training).",
        "",
        "Confounding strengths:",
        f"- mild: {CONFOUNDING_RETAIN_FRACTIONS['mild']:.0%} of training patches from confounded scanners",
        f"- moderate: {CONFOUNDING_RETAIN_FRACTIONS['moderate']:.0%}",
        f"- severe: {CONFOUNDING_RETAIN_FRACTIONS['severe']:.0%}",
        "",
        f"Confounding seeds: {', '.join(map(str, config.confounding_seeds))}",
        "",
        "## Representations",
        "",
    ]

    for rep_name, meta in REPRESENTATION_METADATA.items():
        lines.append(f"- `{rep_name}` ({meta['label']}, dim={meta['dim']})")
    for k in config.k_values:
        lines.append(f"- `pca_removal_k{k}`: PCA top-{k} component removal")
        lines.append(f"- `linear_projection_k{k}`: logistic scanner subspace top-{k} direction removal")

    lines.extend([
        "",
        "## Key Metrics (mean across all folds/seeds)",
        "",
    ])

    # Table by confounding strength
    for strength in config.confounding_strengths:
        lines.append(f"### Confounding: {strength}")
        lines.append("")
        sub = summary[summary["confounding_strength"] == strength].copy()
        if sub.empty:
            lines.append("_No rows._")
            continue
        display_cols = [
            "representation", "n_runs",
            "balanced_accuracy_mean", "macro_f1_mean", "weighted_f1_mean",
            "n_test_errors_mean", "max_scanner_error_share_mean",
        ]
        available = [c for c in display_cols if c in sub.columns]
        sub_display = sub[available].copy()
        for col in sub_display.columns:
            if "mean" in col and col != "representation":
                sub_display[col] = sub_display[col].map(
                    lambda v: f"{float(v):.6f}" if pd.notna(v) else "NA"
                )
        lines.append(sub_display.to_string(index=False))
        lines.append("")

    # Best representation per strength
    lines.append("## Best Representation per Confounding Strength")
    lines.append("")

    for strength in config.confounding_strengths:
        sub = summary[summary["confounding_strength"] == strength]
        if sub.empty:
            continue
        best_idx = sub["balanced_accuracy_mean"].astype(float).idxmax()
        best = sub.loc[best_idx]
        lines.append(
            f"- {strength}: `{best['representation']}` "
            f"(balanced_acc={float(best['balanced_accuracy_mean']):.4f}, "
            f"macro_f1={float(best['macro_f1_mean']):.4f})"
        )

    # Generalization drop
    lines.extend([
        "",
        "## Confounding Impact: Generalization Drop",
        "",
        "Difference between original_frozen_features balanced accuracy and each representation's "
        "balanced accuracy, by confounding strength. Positive = representation lost more accuracy.",
        "",
    ])

    frozen = summary[summary["representation"] == "original_frozen_features"].set_index("confounding_strength")
    for rep in sorted(summary["representation"].unique()):
        if rep == "original_frozen_features":
            continue
        rep_sub = summary[summary["representation"] == rep].set_index("confounding_strength")
        for strength in config.confounding_strengths:
            if strength in frozen.index and strength in rep_sub.index:
                frozen_acc = float(frozen.loc[strength, "balanced_accuracy_mean"])
                rep_acc = float(rep_sub.loc[strength, "balanced_accuracy_mean"])
                delta = rep_acc - frozen_acc
                lines.append(f"- {rep} vs frozen @ {strength}: {delta:+.4f}")

    lines.extend([
        "",
        "## Rare-Class Notes",
        "",
        "- Cartilage has only 2 regions per scanner (10 total patches). Classifier recall for",
        "  Cartilage may be unstable under severe confounding.",
        "- Training subsets may drop below minimum samples for rare classes at severe strength.",
        "",
    ])

    # Scanner error concentration
    lines.extend([
        "",
        "## Scanner Error Concentration",
        "",
        "Max scanner error share = proportion of total test errors occurring on a single scanner.",
        "Lower values suggest errors are less scanner-concentrated.",
        "",
    ])
    for strength in config.confounding_strengths:
        sub = summary[summary["confounding_strength"] == strength]
        col = "max_scanner_error_share_mean"
        if col in sub.columns:
            valid = sub[col].astype(float)
            valid_mask = valid.notna()
            if valid_mask.any():
                best_idx = valid[valid_mask].idxmin()
                best = sub.loc[best_idx]
                lines.append(
                    f"- {strength}: best (lowest concentration) = `{best['representation']}` "
                    f"({float(best[col]):.4f})"
                )
            else:
                lines.append(f"- {strength}: no valid error concentration data.")

    # Split diagnostics
    lines.extend([
        "",
        "## Split Diagnostics",
        "",
        f"- Split diagnostic rows: {len(split_diag)}",
    ])
    if not split_diag.empty:
        lines.append("- Train scanner/category distribution by confounding strength:")
        for strength in config.confounding_strengths:
            diag_sub = split_diag[split_diag["strength"] == strength]
            if diag_sub.empty:
                continue
            n_cats = diag_sub["category"].nunique()
            lines.append(f"  - {strength}: {n_cats} categories, {len(diag_sub)} scanner×category entries")

    # Validation
    lines.extend([
        "",
        "## Validation",
        "",
        f"- Total raw rows: {len(raw)}",
        f"- Validation issues: {len(validation_issues)}",
    ])
    if validation_issues:
        for issue in validation_issues:
            lines.append(f"  - {issue}")
    else:
        lines.append("  - No validation issues found.")

    # Key questions
    lines.extend([
        "",
        "## Key Questions",
        "",
    ])

    # Answer key questions based on data
    for strength in config.confounding_strengths:
        lines.append(f"### Confounding: {strength}")
        sub = summary[summary["confounding_strength"] == strength].set_index("representation")
        if sub.empty:
            lines.append("_No data._")
            continue

        tp_bio = sub.loc["true_pair_biological"] if "true_pair_biological" in sub.index else None
        frozen_acc = sub.loc["original_frozen_features"] if "original_frozen_features" in sub.index else None
        tp_acq = sub.loc["true_pair_acquisition"] if "true_pair_acquisition" in sub.index else None
        sh_bio = sub.loc["shuffled_sample_biological"] if "shuffled_sample_biological" in sub.index else None
        pca32 = sub.loc["pca_removal_k32"] if "pca_removal_k32" in sub.index else None
        lin4 = sub.loc["linear_projection_k4"] if "linear_projection_k4" in sub.index else None

        def _acc(row) -> float:
            return float(row["balanced_accuracy_mean"]) if row is not None else float("nan")

        ta = _acc(tp_bio)
        fa = _acc(frozen_acc)
        aa = _acc(tp_acq)
        sa = _acc(sh_bio)
        pa = _acc(pca32)
        la = _acc(lin4)

        lines.append(f"1. true_pair_biological ({ta:.4f}) vs frozen ({fa:.4f}): {'beats' if ta > fa else 'trails'} frozen")
        lines.append(f"2. true_pair_biological ({ta:.4f}) vs acquisition ({aa:.4f}): {'beats' if ta > aa else 'trails'} acquisition")
        lines.append(f"3. true_pair_biological ({ta:.4f}) vs shuffled_biological ({sa:.4f}): {'beats' if ta > sa else 'trails'} shuffled")
        lines.append(f"4. true_pair_biological ({ta:.4f}) vs pca_k32 ({pa:.4f}): {'beats' if ta > pa else 'trails'} PCA")
        lines.append(f"5. true_pair_biological ({ta:.4f}) vs linear_k4 ({la:.4f}): {'beats' if ta > la else 'trails'} linear")
        lines.append("")

    # Bounded interpretation
    lines.extend([
        "",
        "## Bounded Interpretation",
        "",
        "This is a scanner-category confounding stress test. It does not claim clinical validation,",
        "diagnostic performance, patient-care utility, deployment readiness, or that scanner bias",
        "is solved.",
        "",
        "If true_pair_biological maintains higher balanced accuracy than frozen features under",
        "severe confounding, this supports the interpretation that structured scanner/biology",
        "separation helps resist scanner-category shortcut learning in this audit.",
        "",
        "If linear_projection_k4 or pca_removal_k32 outperform true_pair_biological under",
        "confounding, this suggests that simple post-hoc scanner removal may be sufficient",
        "for confounded training robustness — an important honesty check.",
        "",
    ])

    lines.extend([
        "",
        "## Output Files",
        "",
        f"- `{(config.out_dir / 'scanner_confounded_raw_metrics.csv').as_posix()}`",
        f"- `{(config.out_dir / 'scanner_confounded_summary.csv').as_posix()}`",
        f"- `{(config.out_dir / 'scanner_confounded_per_class_recall.csv').as_posix()}`",
        f"- `{(config.out_dir / 'scanner_confounded_per_scanner_errors.csv').as_posix()}`",
        f"- `{(config.out_dir / 'scanner_confounded_split_diagnostics.csv').as_posix()}`",
        f"- `{(config.out_dir / 'scanner_confounded_robustness_report.md').as_posix()}`",
        f"- `{(config.out_dir / 'experiment_design.json').as_posix()}`",
        f"- `{(config.out_dir / 'run_log.txt').as_posix()}`",
        "",
        f"Runtime: {runtime_seconds:.1f}s",
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
        default=Path("results/paired_acquisition_factorization_scanner_confounded_label_robustness_audit"),
    )
    parser.add_argument("--folds", nargs="+", type=int, default=list(FOLDS))
    parser.add_argument("--neural-seeds", nargs="+", type=int, default=list(PAIR_INTEGRITY_SEEDS))
    parser.add_argument("--confounding-seeds", nargs="+", type=int, default=list(CONFOUNDING_SEEDS))
    parser.add_argument("--confounding-strengths", nargs="+", choices=CONFOUNDING_STRENGTHS, default=list(CONFOUNDING_STRENGTHS))
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
        confounding_strengths=tuple(args.confounding_strengths),
        confounding_seeds=tuple(args.confounding_seeds),
        k_values=tuple(args.k_values),
    )

    design = {
        "stage": "paired_acquisition_factorization_scanner_confounded_label_robustness_audit",
        "dataset": "canine_cutaneous_scc_dinov2",
        "folds": list(config.folds),
        "pair_integrity_seeds": list(config.pair_integrity_seeds),
        "confounding_seeds": list(config.confounding_seeds),
        "confounding_strengths": list(config.confounding_strengths),
        "confounding_retain_fractions": CONFOUNDING_RETAIN_FRACTIONS,
        "k_values": list(config.k_values),
        "representations": {
            "neural": list(REPRESENTATION_METADATA.keys()),
            "simple": ["original_frozen_features"]
            + [f"pca_removal_k{k}" for k in config.k_values]
            + [f"linear_projection_k{k}" for k in config.k_values],
        },
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
                results = run_experiment(config)
                raw = results["raw"]
                split_diag = results["split_diagnostics"]

                validation_issues = validate_outputs(raw, split_diag, config)

                # Compute train/test MI per fold
                mi_rows = []
                for fold in config.folds:
                    manifest = load_manifest(fold)
                    fit_indices, test_indices = canine_cross.validate_fold(manifest, fold)
                    mi = compute_split_mi(manifest, fit_indices, test_indices)
                    mi["fold"] = int(fold)
                    mi_rows.append(mi)

                summary = build_summary(raw)
                per_class = build_per_class_recall(raw)
                per_scanner = build_per_scanner_errors(raw)

                # Write outputs
                atomic_csv(args.out_dir / "scanner_confounded_raw_metrics.csv", raw)
                atomic_csv(args.out_dir / "scanner_confounded_summary.csv", summary)
                if not per_class.empty:
                    atomic_csv(args.out_dir / "scanner_confounded_per_class_recall.csv", per_class)
                if not per_scanner.empty:
                    atomic_csv(args.out_dir / "scanner_confounded_per_scanner_errors.csv", per_scanner)
                if not split_diag.empty:
                    atomic_csv(args.out_dir / "scanner_confounded_split_diagnostics.csv", split_diag)

                runtime = time.perf_counter() - start
                report = build_report(
                    raw, summary, per_class, per_scanner, split_diag,
                    config, validation_issues, runtime,
                )
                atomic_text(args.out_dir / "scanner_confounded_robustness_report.md", report)

                print("\n" + "=" * 80)
                print("SCANNER-CONFOUNDED LABEL ROBUSTNESS AUDIT COMPLETE")
                print(f"Rows: {len(raw)}")
                print(f"Runtime: {runtime:.1f}s")
                print(f"Validation issues: {len(validation_issues)}")
                for issue in validation_issues:
                    print(f"  - {issue}")
                print(f"Report: {(args.out_dir / 'scanner_confounded_robustness_report.md').resolve()}")

                # Best representation per strength
                for strength in config.confounding_strengths:
                    sub = summary[summary["confounding_strength"] == strength]
                    if sub.empty:
                        continue
                    best_idx = sub["balanced_accuracy_mean"].astype(float).idxmax()
                    best = sub.loc[best_idx]
                    print(
                        f"  {strength}: best={best['representation']} "
                        f"acc={float(best['balanced_accuracy_mean']):.4f}"
                    )

            except Exception:
                traceback.print_exc()
                raise


if __name__ == "__main__":
    main()
