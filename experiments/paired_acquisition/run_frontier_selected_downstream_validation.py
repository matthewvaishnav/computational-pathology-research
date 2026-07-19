#!/usr/bin/env python3
"""Frontier-selected downstream validation for canine SCC paired acquisition.

This targeted follow-up asks whether the acquisition bottleneck variants that
reduced acquisition-branch category leakage still hold up under the downstream
stress tests that mattered in the audit ladder.
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
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.canine import run_pair_integrity_falsification_caninescc as canine_pair  # noqa: E402
from experiments.external_multiscanner import run_canine_pathoalign_crossfold as canine_cross  # noqa: E402
from experiments.scorpion import run_pathoalign_projection as projection  # noqa: E402


BRANCH = "experiment/frontier-selected-downstream-validation"
FEATURE_PATH = Path("results/external_multiscanner_caninescc/features/fold_0_dinov2_base.npz")
MANIFESTS_DIR = Path("data/external_multiscanner_caninescc/patch_manifests/splits")
PAIR_INTEGRITY_DIR = Path("results/paired_acquisition_factorization_pair_integrity_caninescc")
FRONTIER_DIR = Path(
    "results/paired_acquisition_factorization_acquisition_bottleneck_separation_frontier"
)
OUT_DIR = Path("results/paired_acquisition_factorization_frontier_selected_downstream_validation")

SCANNERS = ("cs2", "gt450", "nz20", "nz210", "p1000")
FOLDS = (0, 1, 2, 3, 4)
NEURAL_SEEDS = (911, 912, 913, 914, 915)
SAMPLE_SPLIT_SEEDS = (911, 912, 913, 914, 915)
CONFOUNDING_SEEDS = (2001, 2002, 2003, 2004, 2005)
CONFOUNDING_STRENGTHS = ("mild", "moderate", "severe")
TRAIN_FRAC = 0.70

BRANCH_AUDIT_REFERENCES = {
    "true_pair_acquisition": {
        "scanner_balanced_accuracy": 0.8651,
        "category_balanced_accuracy": 0.3456,
    },
    "acq_dim8_default_acquisition": {
        "scanner_balanced_accuracy": 0.8643,
        "category_balanced_accuracy": 0.1598,
    },
    "acq_dim16_stronger_xcov_acquisition": {
        "scanner_balanced_accuracy": 0.8638,
        "category_balanced_accuracy": 0.1689,
    },
    "oldstyle_keep_k4": {
        "scanner_balanced_accuracy": 0.2000,
        "category_balanced_accuracy": 0.4004,
    },
}

CONFOUNDING_RETAIN_FRACTIONS = {
    "mild": 0.60,
    "moderate": 0.80,
    "severe": 0.95,
}

METRIC_COLUMNS = (
    "category_balanced_accuracy",
    "category_macro_f1",
    "category_weighted_f1",
    "n_test_errors",
    "max_scanner_error_share",
)


@dataclass(frozen=True)
class RepresentationSpec:
    representation: str
    variant: str
    branch: str
    source_family: str
    acquisition_dim: int | None
    cross_covariance_weight: float | None


REPRESENTATIONS = (
    RepresentationSpec(
        "true_pair_biological",
        "true_pair",
        "biological",
        "reference_true_pair",
        64,
        0.05,
    ),
    RepresentationSpec(
        "true_pair_acquisition",
        "true_pair",
        "acquisition",
        "reference_true_pair",
        64,
        0.05,
    ),
    RepresentationSpec(
        "acq_dim8_default_biological",
        "acq_dim8_default",
        "biological",
        "frontier_variant",
        8,
        0.05,
    ),
    RepresentationSpec(
        "acq_dim8_default_acquisition",
        "acq_dim8_default",
        "acquisition",
        "frontier_variant",
        8,
        0.05,
    ),
    RepresentationSpec(
        "acq_dim16_stronger_xcov_biological",
        "acq_dim16_stronger_xcov",
        "biological",
        "frontier_variant",
        16,
        0.20,
    ),
    RepresentationSpec(
        "acq_dim16_stronger_xcov_acquisition",
        "acq_dim16_stronger_xcov",
        "acquisition",
        "frontier_variant",
        16,
        0.20,
    ),
)


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


def manifest_path(fold: int) -> Path:
    return MANIFESTS_DIR / f"fold_{fold}_patch_manifest.csv"


def load_manifest(fold: int) -> pd.DataFrame:
    path = manifest_path(fold)
    if not path.is_file():
        raise projection.ExperimentError(f"Missing manifest: {path}")
    manifest = pd.read_csv(path, dtype=str)
    required = {"slide_id", "sample_id", "region_id", "scanner_id", "category_name", "split", "path"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise projection.ExperimentError(f"Manifest {path} missing columns: {missing}")
    manifest["scanner_id"] = manifest["scanner_id"].astype(str).str.lower()
    return manifest.reset_index(drop=True)


def load_frozen_features() -> tuple[np.ndarray, pd.DataFrame]:
    features, frame, metadata = projection.load_archive(FEATURE_PATH)
    frame["scanner_id"] = frame["scanner_id"].astype(str).str.lower()
    return features, frame


def align_by_keys(features: np.ndarray, frame: pd.DataFrame, manifest: pd.DataFrame) -> np.ndarray:
    frame = frame.copy()
    frame["scanner_id"] = frame["scanner_id"].astype(str).str.lower()
    feature_keys = [
        (str(row.slide_id), str(row.region_id), str(row.scanner_id))
        for _, row in frame.iterrows()
    ]
    manifest_keys = [
        (str(row.slide_id), str(row.region_id), str(row.scanner_id))
        for _, row in manifest.iterrows()
    ]
    if len(feature_keys) != len(features):
        raise projection.ExperimentError("Feature frame length does not match feature matrix.")
    lookup = {key: index for index, key in enumerate(feature_keys)}
    missing = [key for key in manifest_keys if key not in lookup]
    if missing:
        raise projection.ExperimentError(f"{len(missing)} manifest keys are missing from projected features.")
    order = np.asarray([lookup[key] for key in manifest_keys], dtype=np.int64)
    return np.asarray(features[order], dtype=np.float32)


def projected_path(spec: RepresentationSpec, fold: int, seed: int) -> Path:
    if spec.source_family == "reference_true_pair":
        return PAIR_INTEGRITY_DIR / f"fold_{fold}" / "runs" / f"true_pairs_seed_{seed}" / "projected_features.npz"
    return (
        FRONTIER_DIR
        / "trained_runs"
        / "full"
        / f"fold_{fold}"
        / "runs"
        / f"{spec.variant}_seed_{seed}"
        / "projected_features.npz"
    )


def load_representation(spec: RepresentationSpec, fold: int, seed: int, manifest: pd.DataFrame) -> np.ndarray:
    path = projected_path(spec, fold, seed)
    if not path.is_file():
        raise projection.ExperimentError(f"Missing projected features: {path}")
    biological, acquisition, frame, metadata = canine_pair.load_projected(path)
    features = biological if spec.branch == "biological" else acquisition
    return align_by_keys(features, frame, manifest)


def sample_disjoint_split(sample_ids: np.ndarray, seed: int) -> tuple[list[str], list[str]]:
    rng = np.random.default_rng(seed)
    unique_samples = np.asarray(sorted(set(sample_ids.astype(str))), dtype=object)
    shuffled = rng.permutation(unique_samples)
    n_train = max(1, int(len(shuffled) * TRAIN_FRAC))
    train_samples = sorted(str(x) for x in shuffled[:n_train])
    test_samples = sorted(str(x) for x in shuffled[n_train:])
    if not test_samples:
        test_samples = train_samples[-1:]
        train_samples = train_samples[:-1]
    return train_samples, test_samples


def split_scanner_heldout(manifest: pd.DataFrame, held_out_scanner: str) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    scanners = manifest["scanner_id"].astype(str).to_numpy()
    train = np.flatnonzero(scanners != held_out_scanner)
    test = np.flatnonzero(scanners == held_out_scanner)
    diagnostics = {
        "held_out_scanner": held_out_scanner,
        "n_train_patches": int(len(train)),
        "n_test_patches": int(len(test)),
        "n_train_samples": int(manifest.iloc[train]["sample_id"].nunique()),
        "n_test_samples": int(manifest.iloc[test]["sample_id"].nunique()),
        "sample_overlap": int(
            len(set(manifest.iloc[train]["sample_id"].astype(str)) & set(manifest.iloc[test]["sample_id"].astype(str)))
        ),
        "train_has_heldout_scanner": bool((scanners[train] == held_out_scanner).any()),
    }
    return train, test, diagnostics


def split_sample_disjoint(
    manifest: pd.DataFrame, held_out_scanner: str, sample_split_seed: int
) -> tuple[np.ndarray, np.ndarray, dict[str, object]]:
    train_samples, test_samples = sample_disjoint_split(
        manifest["sample_id"].astype(str).to_numpy(), sample_split_seed
    )
    scanners = manifest["scanner_id"].astype(str).to_numpy()
    samples = manifest["sample_id"].astype(str).to_numpy()
    train_mask = (scanners != held_out_scanner) & np.isin(samples, train_samples)
    test_mask = (scanners == held_out_scanner) & np.isin(samples, test_samples)
    train = np.flatnonzero(train_mask)
    test = np.flatnonzero(test_mask)
    train_sample_set = set(samples[train])
    test_sample_set = set(samples[test])
    diagnostics = {
        "held_out_scanner": held_out_scanner,
        "sample_split_seed": int(sample_split_seed),
        "n_train_patches": int(len(train)),
        "n_test_patches": int(len(test)),
        "n_train_samples": int(len(train_sample_set)),
        "n_test_samples": int(len(test_sample_set)),
        "sample_overlap": int(len(train_sample_set & test_sample_set)),
        "train_has_heldout_scanner": bool((scanners[train] == held_out_scanner).any()) if len(train) else False,
    }
    return train, test, diagnostics


def mutual_information(x: np.ndarray, y: np.ndarray) -> float:
    n = len(x)
    if n <= 1:
        return 0.0
    x_values, x_inv = np.unique(x, return_inverse=True)
    y_values, y_inv = np.unique(y, return_inverse=True)
    joint = np.zeros((len(x_values), len(y_values)), dtype=np.float64)
    np.add.at(joint, (x_inv, y_inv), 1)
    joint /= n
    px = joint.sum(axis=1)
    py = joint.sum(axis=0)
    mi = 0.0
    for i in range(len(x_values)):
        for j in range(len(y_values)):
            if joint[i, j] > 0:
                mi += joint[i, j] * math.log(joint[i, j] / (px[i] * py[j]))
    return float(max(0.0, mi))


def build_confounded_training_set(
    manifest: pd.DataFrame,
    fit_indices: np.ndarray,
    strength: str,
    confounding_seed: int,
) -> tuple[np.ndarray, pd.DataFrame]:
    retain = CONFOUNDING_RETAIN_FRACTIONS[strength]
    rng = np.random.default_rng(confounding_seed)
    fit_df = manifest.iloc[fit_indices].copy()
    fit_df["_orig_idx"] = fit_indices

    selected_index_labels: list[object] = []
    diagnostics: list[dict[str, object]] = []
    categories = sorted(fit_df["category_name"].unique())
    scanners = sorted(fit_df["scanner_id"].unique())

    for category in categories:
        cat_df = fit_df[fit_df["category_name"] == category]
        cat_scanners = sorted(cat_df["scanner_id"].unique())
        n_assigned = min(2, len(cat_scanners))
        assigned = list(rng.choice(cat_scanners, size=n_assigned, replace=False))
        other = [scanner for scanner in cat_scanners if scanner not in assigned]
        assigned_df = cat_df[cat_df["scanner_id"].isin(assigned)]
        other_df = cat_df[cat_df["scanner_id"].isin(other)]

        n_assigned_keep = max(1, int(len(assigned_df) * retain)) if len(assigned_df) else 0
        n_other_keep = max(1, int(len(other_df) * (1.0 - retain))) if len(other_df) else 0
        assigned_keep = (
            rng.choice(assigned_df.index.to_numpy(), size=n_assigned_keep, replace=False).tolist()
            if n_assigned_keep
            else []
        )
        other_keep = (
            rng.choice(other_df.index.to_numpy(), size=n_other_keep, replace=False).tolist()
            if n_other_keep
            else []
        )
        kept = assigned_keep + other_keep
        selected_index_labels.extend(kept)

        for scanner in scanners:
            scanner_cat = cat_df[cat_df["scanner_id"] == scanner]
            kept_on_scanner = len([idx for idx in kept if idx in set(scanner_cat.index)])
            diagnostics.append(
                {
                    "category": category,
                    "scanner": scanner,
                    "is_assigned": bool(scanner in assigned),
                    "n_total_fit": int(len(scanner_cat)),
                    "n_kept": int(kept_on_scanner),
                    "retain_fraction_actual": float(kept_on_scanner / max(1, len(scanner_cat))),
                }
            )

    selected = fit_df.loc[selected_index_labels, "_orig_idx"].to_numpy(dtype=np.int64)
    diag_df = pd.DataFrame(diagnostics)
    return selected, diag_df


def split_confounded(
    manifest: pd.DataFrame,
    fold: int,
    strength: str,
    confounding_seed: int,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, object]]:
    fit, test = canine_cross.validate_fold(manifest, fold)
    train, category_diag = build_confounded_training_set(manifest, fit, strength, confounding_seed)
    y_train = manifest.iloc[train]["category_name"].astype(str).to_numpy()
    s_train = manifest.iloc[train]["scanner_id"].astype(str).to_numpy()
    y_test = manifest.iloc[test]["category_name"].astype(str).to_numpy()
    s_test = manifest.iloc[test]["scanner_id"].astype(str).to_numpy()
    diagnostics = {
        "confounding_strength": strength,
        "confounding_seed": int(confounding_seed),
        "n_train_patches": int(len(train)),
        "n_test_patches": int(len(test)),
        "n_train_samples": int(manifest.iloc[train]["sample_id"].nunique()),
        "n_test_samples": int(manifest.iloc[test]["sample_id"].nunique()),
        "sample_overlap": int(
            len(set(manifest.iloc[train]["sample_id"].astype(str)) & set(manifest.iloc[test]["sample_id"].astype(str)))
        ),
        "train_has_heldout_scanner": False,
        "train_scanner_category_mi": mutual_information(s_train, y_train),
        "test_scanner_category_mi": mutual_information(s_test, y_test),
    }
    return train, test, category_diag, diagnostics


def evaluate_category_probe(
    features: np.ndarray,
    manifest: pd.DataFrame,
    train_indices: np.ndarray,
    test_indices: np.ndarray,
    seed: int,
) -> dict[str, object]:
    if len(train_indices) == 0 or len(test_indices) == 0:
        return {
            "category_balanced_accuracy": float("nan"),
            "category_macro_f1": float("nan"),
            "category_weighted_f1": float("nan"),
            "n_test_errors": 0,
            "max_scanner_error_share": 0.0,
            "error": "empty_split",
        }
    y_train = manifest.iloc[train_indices]["category_name"].astype(str).to_numpy()
    y_test = manifest.iloc[test_indices]["category_name"].astype(str).to_numpy()
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=seed),
    )
    model.fit(features[train_indices], y_train)
    y_pred = model.predict(features[test_indices])

    error_mask = y_pred != y_test
    test_scanners = manifest.iloc[test_indices]["scanner_id"].astype(str).to_numpy()
    scanner_error_rates: dict[str, float] = {}
    scanner_error_counts: dict[str, int] = {}
    for scanner in sorted(set(test_scanners)):
        mask = test_scanners == scanner
        scanner_error_rates[f"error_rate_{scanner}"] = float(error_mask[mask].mean()) if mask.any() else float("nan")
        scanner_error_counts[scanner] = int(error_mask[mask].sum())
    total_errors = int(error_mask.sum())
    max_error_share = max(scanner_error_counts.values()) / max(1, total_errors) if scanner_error_counts else 0.0

    per_class: dict[str, float] = {}
    for category in sorted(set(y_test)):
        mask = y_test == category
        per_class[f"recall_{category}"] = float(np.mean(y_pred[mask] == category))

    return {
        "category_balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "category_macro_f1": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "category_weighted_f1": float(f1_score(y_test, y_pred, average="weighted", zero_division=0)),
        "n_test_errors": total_errors,
        "max_scanner_error_share": float(max_error_share),
        "n_classes_train": int(len(set(y_train))),
        "n_classes_test": int(len(set(y_test))),
        "error": "",
        **per_class,
        **scanner_error_rates,
    }


def row_base(spec: RepresentationSpec, fold: int, neural_seed: int) -> dict[str, object]:
    return {
        "representation": spec.representation,
        "variant": spec.variant,
        "branch": spec.branch,
        "source_family": spec.source_family,
        "fold": int(fold),
        "neural_seed": int(neural_seed),
        "acquisition_dim": "" if spec.acquisition_dim is None else int(spec.acquisition_dim),
        "cross_covariance_weight": "" if spec.cross_covariance_weight is None else float(spec.cross_covariance_weight),
    }


def run_scanner_heldout(
    spec: RepresentationSpec,
    features: np.ndarray,
    manifest: pd.DataFrame,
    fold: int,
    neural_seed: int,
    scanners: tuple[str, ...],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for held_out in scanners:
        train, test, diag = split_scanner_heldout(manifest, held_out)
        metrics = evaluate_category_probe(features, manifest, train, test, neural_seed)
        common = row_base(spec, fold, neural_seed)
        row = {
            "audit": "scanner_heldout_label_transfer",
            **common,
            "held_out_scanner": held_out,
            "sample_split_seed": "",
            "confounding_seed": "",
            "confounding_strength": "",
            **diag,
            **metrics,
        }
        rows.append(row)
        diagnostics.append({"audit": "scanner_heldout_label_transfer", **common, **diag})
    return rows, diagnostics


def run_sample_disjoint(
    spec: RepresentationSpec,
    features: np.ndarray,
    manifest: pd.DataFrame,
    fold: int,
    neural_seed: int,
    scanners: tuple[str, ...],
    sample_split_seeds: tuple[int, ...],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for held_out in scanners:
        for split_seed in sample_split_seeds:
            train, test, diag = split_sample_disjoint(manifest, held_out, split_seed)
            metrics = evaluate_category_probe(features, manifest, train, test, split_seed)
            common = row_base(spec, fold, neural_seed)
            row = {
                "audit": "sample_disjoint_scanner_heldout_transfer",
                **common,
                "held_out_scanner": held_out,
                "sample_split_seed": int(split_seed),
                "confounding_seed": "",
                "confounding_strength": "",
                **diag,
                **metrics,
            }
            rows.append(row)
            diagnostics.append({"audit": "sample_disjoint_scanner_heldout_transfer", **common, **diag})
    return rows, diagnostics


def run_confounding(
    spec: RepresentationSpec,
    features: np.ndarray,
    manifest: pd.DataFrame,
    fold: int,
    neural_seed: int,
    strengths: tuple[str, ...],
    confounding_seeds: tuple[int, ...],
    split_cache: dict[tuple[int, str, int], tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, object]]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    rows: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    for strength in strengths:
        for cseed in confounding_seeds:
            key = (fold, strength, cseed)
            if key not in split_cache:
                split_cache[key] = split_confounded(manifest, fold, strength, cseed)
            train, test, category_diag, diag = split_cache[key]
            metrics = evaluate_category_probe(features, manifest, train, test, cseed)
            common = row_base(spec, fold, neural_seed)
            row = {
                "audit": "scanner_confounded_label_robustness",
                **common,
                "held_out_scanner": "",
                "sample_split_seed": "",
                "confounding_seed": int(cseed),
                "confounding_strength": strength,
                **diag,
                **metrics,
            }
            rows.append(row)
            diagnostics.append({"audit": "scanner_confounded_label_robustness", **common, **diag})
    return rows, diagnostics


def build_summary(raw: pd.DataFrame) -> pd.DataFrame:
    metric_cols = [c for c in METRIC_COLUMNS if c in raw.columns]
    group_cols = ["audit", "representation", "variant", "branch", "source_family"]
    grouped = raw.groupby(group_cols, dropna=False)
    summary = grouped.agg(
        n_rows=("fold", "size"),
        n_folds=("fold", "nunique"),
        n_neural_seeds=("neural_seed", "nunique"),
        n_held_out_scanners=("held_out_scanner", lambda x: len({str(v) for v in x if str(v)})),
        n_sample_split_seeds=("sample_split_seed", lambda x: len({str(v) for v in x if str(v)})),
        n_confounding_seeds=("confounding_seed", lambda x: len({str(v) for v in x if str(v)})),
        n_confounding_strengths=("confounding_strength", lambda x: len({str(v) for v in x if str(v)})),
    )
    for metric in metric_cols:
        metric_summary = grouped[metric].agg(["mean", "std", "min", "max"])
        metric_summary.columns = [f"{metric}_{col}" for col in metric_summary.columns]
        summary = summary.join(metric_summary)
    return summary.reset_index().sort_values(group_cols)


def build_contrasts(summary: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    metrics = (
        "category_balanced_accuracy_mean",
        "category_macro_f1_mean",
        "category_weighted_f1_mean",
        "max_scanner_error_share_mean",
    )
    reference = summary[summary["variant"] == "true_pair"]
    for _, row in summary[summary["source_family"] == "frontier_variant"].iterrows():
        ref = reference[(reference["audit"] == row["audit"]) & (reference["branch"] == row["branch"])]
        if ref.empty:
            continue
        ref_row = ref.iloc[0]
        contrast = {
            "audit": row["audit"],
            "representation": row["representation"],
            "variant": row["variant"],
            "branch": row["branch"],
            "reference_representation": ref_row["representation"],
        }
        for metric in metrics:
            if metric in row.index and metric in ref_row.index:
                value = pd.to_numeric(pd.Series([row[metric]]), errors="coerce").iloc[0]
                ref_value = pd.to_numeric(pd.Series([ref_row[metric]]), errors="coerce").iloc[0]
                contrast[metric] = float(value)
                contrast[f"reference_{metric}"] = float(ref_value)
                contrast[f"delta_vs_true_pair_{metric}"] = float(value - ref_value)
        rows.append(contrast)
    return pd.DataFrame(rows)


def build_per_class(raw: pd.DataFrame) -> pd.DataFrame:
    recall_cols = [c for c in raw.columns if c.startswith("recall_")]
    if not recall_cols:
        return pd.DataFrame()
    melted = raw.melt(
        id_vars=["audit", "representation", "variant", "branch", "source_family"],
        value_vars=recall_cols,
        var_name="category",
        value_name="recall",
    )
    melted["category"] = melted["category"].str.replace("recall_", "", regex=False)
    return (
        melted.dropna(subset=["recall"])
        .groupby(["audit", "representation", "variant", "branch", "source_family", "category"], dropna=False)[
            "recall"
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )


def build_per_scanner_errors(raw: pd.DataFrame) -> pd.DataFrame:
    error_cols = [c for c in raw.columns if c.startswith("error_rate_")]
    if not error_cols:
        return pd.DataFrame()
    melted = raw.melt(
        id_vars=["audit", "representation", "variant", "branch", "source_family"],
        value_vars=error_cols,
        var_name="scanner",
        value_name="error_rate",
    )
    melted["scanner"] = melted["scanner"].str.replace("error_rate_", "", regex=False)
    return (
        melted.dropna(subset=["error_rate"])
        .groupby(["audit", "representation", "variant", "branch", "source_family", "scanner"], dropna=False)[
            "error_rate"
        ]
        .agg(["mean", "std", "count"])
        .reset_index()
    )


def validate_outputs(raw: pd.DataFrame, diagnostics: pd.DataFrame, summary: pd.DataFrame) -> list[str]:
    issues: list[str] = []
    key_cols = [
        "audit",
        "representation",
        "fold",
        "neural_seed",
        "held_out_scanner",
        "sample_split_seed",
        "confounding_seed",
        "confounding_strength",
    ]
    if raw.duplicated(key_cols).any():
        issues.append(f"Duplicate raw metric rows: {int(raw.duplicated(key_cols).sum())}")
    metric_cols = [c for c in METRIC_COLUMNS if c in raw.columns]
    for col in metric_cols:
        values = pd.to_numeric(raw[col], errors="coerce")
        if values.isna().any():
            issues.append(f"Missing metric values in {col}: {int(values.isna().sum())}")
        finite = np.isfinite(values.to_numpy(dtype=float))
        if not finite.all():
            issues.append(f"Nonfinite metric values in {col}: {int((~finite).sum())}")
    if "sample_overlap" in diagnostics.columns:
        sample_disjoint = diagnostics[diagnostics["audit"] == "sample_disjoint_scanner_heldout_transfer"]
        overlaps = pd.to_numeric(sample_disjoint["sample_overlap"], errors="coerce").fillna(0)
        if (overlaps > 0).any():
            issues.append("Sample-disjoint split has sample overlap.")
    if "train_has_heldout_scanner" in diagnostics.columns:
        sample_disjoint = diagnostics[diagnostics["audit"] == "sample_disjoint_scanner_heldout_transfer"]
        heldout_leak = sample_disjoint["train_has_heldout_scanner"].astype(str).str.lower().isin(["true", "1"])
        if heldout_leak.any():
            issues.append("Sample-disjoint split has held-out scanner in training.")
    expected_audits = {
        "scanner_heldout_label_transfer",
        "sample_disjoint_scanner_heldout_transfer",
        "scanner_confounded_label_robustness",
    }
    observed_audits = set(raw["audit"].unique())
    if observed_audits != expected_audits:
        issues.append(f"Audit coverage mismatch: observed {sorted(observed_audits)}")
    expected_reps = {spec.representation for spec in REPRESENTATIONS}
    observed_reps = set(raw["representation"].unique())
    if observed_reps != expected_reps:
        issues.append(f"Representation coverage mismatch: observed {sorted(observed_reps)}")
    if summary.empty:
        issues.append("Summary is empty.")
    return issues


def fmt(value: object) -> str:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return "NA"
    if not math.isfinite(val):
        return "NA"
    return f"{val:.4f}"


def metric_lookup(summary: pd.DataFrame, audit: str, representation: str, metric: str) -> float:
    sub = summary[(summary["audit"] == audit) & (summary["representation"] == representation)]
    if sub.empty or metric not in sub.columns:
        return float("nan")
    return float(sub.iloc[0][metric])


def build_report(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    validation_issues: list[str],
    runtime_seconds: float,
    smoke: bool,
) -> str:
    lines: list[str] = [
        "# Frontier-Selected Downstream Validation",
        "",
        "## Branch",
        "",
        BRANCH,
        "",
        "## Question",
        "",
        "Did reducing acquisition-branch category leakage preserve or improve downstream transfer robustness, or did it only make the branch audit cleaner?",
        "",
        "## Frontier Variants",
        "",
        "- acq_dim8_default",
        "- acq_dim16_stronger_xcov",
        "",
        "## Reference Boundary",
        "",
        "- true_pair_acquisition branch audit: scanner 0.8651, category leakage 0.3456.",
        "- acq_dim8_default acquisition branch audit: scanner 0.8643, category leakage 0.1598.",
        "- acq_dim16_stronger_xcov acquisition branch audit: scanner 0.8638, category leakage 0.1689.",
        "- oldstyle_keep_k4 remains best raw scanner removal: scanner 0.2000, category 0.4004.",
        "",
        "## Protocol",
        "",
        "- Scanner-heldout label transfer: train category probe on four scanners and test on the held-out scanner.",
        "- Sample-disjoint scanner-heldout transfer: train on four scanners and train-sample subset; test on held-out scanner and disjoint samples.",
        "- Scanner-confounded label robustness: train on scanner/category-confounded fit subsets and test on the unmodified held-out fold.",
        "- Probe model: standardized LogisticRegression with balanced class weights.",
        "",
        "## Row Counts",
        "",
        f"- Raw metric rows: {len(raw)}.",
        f"- Summary rows: {len(summary)}.",
        f"- Contrast rows: {len(contrasts)}.",
        f"- Smoke mode: {bool(smoke)}.",
        f"- Runtime seconds: {runtime_seconds:.1f}.",
        "",
        "## Key Metrics",
        "",
    ]

    audits = [
        "scanner_heldout_label_transfer",
        "sample_disjoint_scanner_heldout_transfer",
        "scanner_confounded_label_robustness",
    ]
    reps = [
        "true_pair_biological",
        "acq_dim8_default_biological",
        "acq_dim16_stronger_xcov_biological",
        "true_pair_acquisition",
        "acq_dim8_default_acquisition",
        "acq_dim16_stronger_xcov_acquisition",
    ]
    for audit in audits:
        lines.append(f"### {audit}")
        lines.append("")
        for rep in reps:
            acc = metric_lookup(summary, audit, rep, "category_balanced_accuracy_mean")
            f1 = metric_lookup(summary, audit, rep, "category_macro_f1_mean")
            worst = metric_lookup(summary, audit, rep, "category_balanced_accuracy_min")
            lines.append(f"- {rep}: balanced_acc={fmt(acc)}, macro_f1={fmt(f1)}, min_acc={fmt(worst)}.")
        lines.append("")

    lines.extend(
        [
            "## Frontier Contrasts",
            "",
            "For biological branches, positive category-accuracy deltas indicate stronger downstream category transfer.",
            "For acquisition branches, negative category-accuracy deltas indicate lower downstream category leakage.",
            "",
        ]
    )
    for _, row in contrasts.iterrows():
        delta = row.get("delta_vs_true_pair_category_balanced_accuracy_mean", float("nan"))
        lines.append(
            f"- {row['audit']} {row['representation']} vs {row['reference_representation']}: "
            f"delta_balanced_acc={fmt(delta)}."
        )
    lines.extend(
        [
            "",
            "## Validation",
            "",
            f"- Validation issues: {len(validation_issues)}.",
        ]
    )
    if validation_issues:
        for issue in validation_issues:
            lines.append(f"- {issue}")
    else:
        lines.append("- No validation issues found.")
    lines.extend(
        [
            "",
            "## Bounded Interpretation",
            "",
            "This is a downstream stress check for the frontier-selected variants, not a broad generalization claim.",
            "The central question is whether the cleaner acquisition branch remains compatible with downstream category transfer in the biological branch.",
            "The oldstyle centroid/QR result remains the strongest raw scanner-removal boundary.",
            "",
            "## Files Created",
            "",
            "- frontier_downstream_raw_metrics.csv",
            "- frontier_downstream_summary.csv",
            "- frontier_downstream_contrasts.csv",
            "- frontier_downstream_split_diagnostics.csv",
            "- frontier_downstream_per_class_recall.csv",
            "- frontier_downstream_per_scanner_errors.csv",
            "- frontier_selected_downstream_validation_report.md",
            "- experiment_design.json",
            "- run_log.txt",
        ]
    )
    return "\n".join(lines) + "\n"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Frontier-selected downstream validation")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "run_log.txt"
    t0 = time.time()

    folds = (0,) if args.smoke else FOLDS
    neural_seeds = (NEURAL_SEEDS[0],) if args.smoke else NEURAL_SEEDS
    scanners = (SCANNERS[0],) if args.smoke else SCANNERS
    sample_split_seeds = (SAMPLE_SPLIT_SEEDS[0],) if args.smoke else SAMPLE_SPLIT_SEEDS
    confounding_seeds = (CONFOUNDING_SEEDS[0],) if args.smoke else CONFOUNDING_SEEDS
    confounding_strengths = ("moderate",) if args.smoke else CONFOUNDING_STRENGTHS

    with log_path.open("w", encoding="utf-8") as log_file:
        stdout = Tee(sys.stdout, log_file)
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = stdout
        sys.stderr = stdout
        try:
            print("Frontier-selected downstream validation started")
            print(f"Branch: {BRANCH}")
            print(f"Smoke: {args.smoke}")
            print(f"Folds: {folds}")
            print(f"Neural seeds: {neural_seeds}")
            print(f"Scanners: {scanners}")

            canine_cross.patch_scanner_namespace()
            raw_rows: list[dict[str, object]] = []
            diag_rows: list[dict[str, object]] = []
            split_cache: dict[tuple[int, str, int], tuple[np.ndarray, np.ndarray, pd.DataFrame, dict[str, object]]] = {}

            for fold in folds:
                manifest = load_manifest(fold)
                print(
                    f"Fold {fold}: rows={len(manifest)} samples={manifest['sample_id'].nunique()} "
                    f"categories={manifest['category_name'].nunique()}"
                )
                for neural_seed in neural_seeds:
                    for spec in REPRESENTATIONS:
                        print(f"Evaluate {spec.representation} fold={fold} seed={neural_seed}")
                        features = load_representation(spec, fold, neural_seed, manifest)
                        scanner_rows, scanner_diag = run_scanner_heldout(
                            spec, features, manifest, fold, neural_seed, scanners
                        )
                        # Match the original sample-disjoint audit: the neural
                        # seed also defines the sample split seed.
                        sample_rows, sample_diag = run_sample_disjoint(
                            spec,
                            features,
                            manifest,
                            fold,
                            neural_seed,
                            scanners,
                            (neural_seed,),
                        )
                        conf_rows, conf_diag = run_confounding(
                            spec,
                            features,
                            manifest,
                            fold,
                            neural_seed,
                            confounding_strengths,
                            confounding_seeds,
                            split_cache,
                        )
                        raw_rows.extend(scanner_rows)
                        raw_rows.extend(sample_rows)
                        raw_rows.extend(conf_rows)
                        diag_rows.extend(scanner_diag)
                        diag_rows.extend(sample_diag)
                        diag_rows.extend(conf_diag)

            raw = pd.DataFrame(raw_rows)
            diagnostics = pd.DataFrame(diag_rows)
            summary = build_summary(raw)
            contrasts = build_contrasts(summary)
            per_class = build_per_class(raw)
            per_scanner = build_per_scanner_errors(raw)
            validation_issues = validate_outputs(raw, diagnostics, summary)

            atomic_csv(out_dir / "frontier_downstream_raw_metrics.csv", raw)
            atomic_csv(out_dir / "frontier_downstream_summary.csv", summary)
            atomic_csv(out_dir / "frontier_downstream_contrasts.csv", contrasts)
            atomic_csv(out_dir / "frontier_downstream_split_diagnostics.csv", diagnostics)
            atomic_csv(out_dir / "frontier_downstream_per_class_recall.csv", per_class)
            atomic_csv(out_dir / "frontier_downstream_per_scanner_errors.csv", per_scanner)

            runtime = time.time() - t0
            report = build_report(raw, summary, contrasts, validation_issues, runtime, args.smoke)
            atomic_text(out_dir / "frontier_selected_downstream_validation_report.md", report)
            design = {
                "stage": "frontier_selected_downstream_validation",
                "branch": BRANCH,
                "dataset": "canine_cutaneous_scc_dinov2",
                "columns": {
                    "label": "category_name",
                    "scanner": "scanner_id",
                    "sample": "sample_id",
                    "region": "region_id",
                },
                "smoke": bool(args.smoke),
                "folds": list(folds),
                "neural_seeds": list(neural_seeds),
                "held_out_scanners": list(scanners),
                "sample_split_seeds": list(sample_split_seeds),
                "confounding_seeds": list(confounding_seeds),
                "confounding_strengths": list(confounding_strengths),
                "representations": [spec.__dict__ for spec in REPRESENTATIONS],
                "branch_audit_references": BRANCH_AUDIT_REFERENCES,
                "input_frontier_dir": str(FRONTIER_DIR),
                "input_true_pair_dir": str(PAIR_INTEGRITY_DIR),
                "probe_model": "StandardScaler + LogisticRegression(C=1.0, class_weight=balanced, max_iter=5000)",
                "outputs": [
                    "frontier_downstream_raw_metrics.csv",
                    "frontier_downstream_summary.csv",
                    "frontier_downstream_contrasts.csv",
                    "frontier_downstream_split_diagnostics.csv",
                    "frontier_downstream_per_class_recall.csv",
                    "frontier_downstream_per_scanner_errors.csv",
                    "frontier_selected_downstream_validation_report.md",
                    "experiment_design.json",
                    "run_log.txt",
                ],
            }
            atomic_text(out_dir / "experiment_design.json", json.dumps(design, indent=2, sort_keys=True) + "\n")

            print("")
            print("=" * 80)
            print("FRONTIER-SELECTED DOWNSTREAM VALIDATION COMPLETE")
            print(f"Raw rows: {len(raw)}")
            print(f"Summary rows: {len(summary)}")
            print(f"Contrast rows: {len(contrasts)}")
            print(f"Validation issues: {len(validation_issues)}")
            print(f"Runtime: {runtime:.1f}s")
            print(f"Report: {out_dir / 'frontier_selected_downstream_validation_report.md'}")
            return 0 if not validation_issues else 1
        except Exception:
            traceback.print_exc()
            return 1
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


if __name__ == "__main__":
    raise SystemExit(main())
