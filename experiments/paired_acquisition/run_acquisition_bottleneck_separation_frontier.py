#!/usr/bin/env python3
"""Acquisition bottleneck separation-frontier sweep for canine SCC DINOv2.

This experiment tests whether changing acquisition branch capacity and a small
cross-covariance regularization sweep can reduce acquisition-branch category
leakage while retaining scanner capture and preserving the biological branch.
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
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
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
from src.models.scorpion_pathoalign import ProjectionConfig  # noqa: E402


BRANCH = "experiment/acquisition-bottleneck-separation-frontier"
FEATURE_PATH = Path("results/external_multiscanner_caninescc/features/fold_0_dinov2_base.npz")
MANIFESTS_DIR = Path("data/external_multiscanner_caninescc/patch_manifests/splits")
REFERENCE_PAIR_DIR = Path("results/paired_acquisition_factorization_pair_integrity_caninescc")

FOLDS = (0, 1, 2, 3, 4)
FULL_SEEDS = (911, 912, 913, 914, 915)
SMOKE_FOLD = 0
SMOKE_SEED = 0
NEIGHBORHOOD_K = (1, 5, 10)

BASELINE_REFERENCES = {
    "true_pair_biological": {
        "scanner_balanced_accuracy": 0.3614,
        "category_balanced_accuracy": 0.3860,
    },
    "true_pair_acquisition": {
        "scanner_balanced_accuracy": 0.8651,
        "category_balanced_accuracy": 0.3456,
    },
    "oldstyle_keep_k4": {
        "scanner_balanced_accuracy": 0.2000,
        "category_balanced_accuracy": 0.4004,
    },
    "oldstyle_removed_k4": {
        "scanner_balanced_accuracy": 0.5384,
        "category_balanced_accuracy": 0.2421,
    },
}

METRIC_COLUMNS = (
    "scanner_balanced_accuracy",
    "scanner_macro_f1",
    "category_balanced_accuracy",
    "category_macro_f1",
    "category_weighted_f1",
    "same_category_purity_k1",
    "same_category_purity_k5",
    "same_category_purity_k10",
)


@dataclass(frozen=True)
class FrontierVariant:
    name: str
    acquisition_dim: int
    biological_dim: int = 256
    hidden_dim: int = 512
    cross_covariance_weight: float = 0.05
    scanner_adversary_weight: float = 0.5
    scanner_acquisition_weight: float = 0.5
    scanner_dependence_weight: float = 20.0
    gradient_reversal_strength: float = 1.0
    reconstruction_weight: float = 1.0
    variance_weight: float = 1.0
    covariance_weight: float = 0.01
    temperature: float = 0.1
    variant_family: str = "acquisition_bottleneck"


VARIANTS = (
    FrontierVariant("acq_dim8_default", acquisition_dim=8),
    FrontierVariant("acq_dim16_default", acquisition_dim=16),
    FrontierVariant("acq_dim32_default", acquisition_dim=32),
    FrontierVariant("acq_dim64_current", acquisition_dim=64),
    FrontierVariant(
        "acq_dim16_stronger_xcov",
        acquisition_dim=16,
        cross_covariance_weight=0.20,
        variant_family="acquisition_bottleneck_stronger_separation",
    ),
    FrontierVariant(
        "acq_dim32_stronger_xcov",
        acquisition_dim=32,
        cross_covariance_weight=0.20,
        variant_family="acquisition_bottleneck_stronger_separation",
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


def load_existing_csv(path: Path) -> pd.DataFrame:
    if not path.is_file():
        return pd.DataFrame()
    return pd.read_csv(path)


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
    return manifest


def config_for_variant(input_dim: int, variant: FrontierVariant) -> ProjectionConfig:
    return ProjectionConfig(
        input_dim=input_dim,
        biological_dim=variant.biological_dim,
        acquisition_dim=variant.acquisition_dim,
        hidden_dim=variant.hidden_dim,
        temperature=variant.temperature,
        reconstruction_weight=variant.reconstruction_weight,
        variance_weight=variant.variance_weight,
        covariance_weight=variant.covariance_weight,
        scanner_adversary_weight=variant.scanner_adversary_weight,
        scanner_acquisition_weight=variant.scanner_acquisition_weight,
        scanner_dependence_weight=variant.scanner_dependence_weight,
        cross_covariance_weight=variant.cross_covariance_weight,
        gradient_reversal_strength=variant.gradient_reversal_strength,
    )


def variant_dict(variant: FrontierVariant) -> dict[str, object]:
    return {
        "variant": variant.name,
        "variant_family": variant.variant_family,
        "acquisition_dim": int(variant.acquisition_dim),
        "biological_dim": int(variant.biological_dim),
        "hidden_dim": int(variant.hidden_dim),
        "cross_covariance_weight": float(variant.cross_covariance_weight),
        "scanner_adversary_weight": float(variant.scanner_adversary_weight),
        "scanner_acquisition_weight": float(variant.scanner_acquisition_weight),
        "scanner_dependence_weight": float(variant.scanner_dependence_weight),
        "gradient_reversal_strength": float(variant.gradient_reversal_strength),
        "reconstruction_weight": float(variant.reconstruction_weight),
        "variance_weight": float(variant.variance_weight),
        "covariance_weight": float(variant.covariance_weight),
        "temperature": float(variant.temperature),
    }


def row_keys(frame: pd.DataFrame) -> list[tuple[str, str, str]]:
    return [
        (str(row.slide_id), str(row.region_id), str(row.scanner_id).lower())
        for _, row in frame.iterrows()
    ]


def align_projected(
    features: np.ndarray,
    projected_frame: pd.DataFrame,
    manifest: pd.DataFrame,
) -> np.ndarray:
    projected_frame = projected_frame.copy()
    projected_frame["scanner_id"] = projected_frame["scanner_id"].astype(str).str.lower()
    manifest = manifest.copy()
    manifest["scanner_id"] = manifest["scanner_id"].astype(str).str.lower()
    lookup = {key: index for index, key in enumerate(row_keys(projected_frame))}
    keys = row_keys(manifest)
    missing = [key for key in keys if key not in lookup]
    if missing:
        raise projection.ExperimentError(f"{len(missing)} manifest rows missing from projected features.")
    order = np.asarray([lookup[key] for key in keys], dtype=np.int64)
    return features[order]


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


def same_category_purity(features: np.ndarray, manifest: pd.DataFrame, test: np.ndarray, k: int) -> float:
    matrix = np.asarray(features[test], dtype=np.float64)
    labels = manifest.iloc[test]["category_name"].astype(str).to_numpy()
    n_rows = len(matrix)
    if n_rows <= k + 1:
        return float("nan")
    model = NearestNeighbors(n_neighbors=min(k + 1, n_rows), metric="cosine", n_jobs=1)
    model.fit(matrix)
    indices = model.kneighbors(matrix, return_distance=False)[:, 1:]
    return float(np.mean(labels[indices] == labels[:, None]))


def neighborhood_metrics(features: np.ndarray, manifest: pd.DataFrame, test: np.ndarray) -> dict[str, float]:
    return {
        f"same_category_purity_k{k}": same_category_purity(features, manifest, test, k)
        for k in NEIGHBORHOOD_K
    }


def evaluate_branch(
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
    return metrics


def projected_path_for_run(out_dir: Path, phase: str, variant: FrontierVariant, fold: int, seed: int) -> Path:
    return out_dir / "trained_runs" / phase / f"fold_{fold}" / "runs" / f"{variant.name}_seed_{seed}" / "projected_features.npz"


def reference_projected_path(fold: int, seed: int) -> Path:
    return REFERENCE_PAIR_DIR / f"fold_{fold}" / "runs" / f"true_pairs_seed_{seed}" / "projected_features.npz"


def train_or_reuse_projected(
    *,
    out_dir: Path,
    phase: str,
    variant: FrontierVariant,
    fold: int,
    seed: int,
    features: np.ndarray,
    frame: pd.DataFrame,
    fit: np.ndarray,
    groups: list[np.ndarray],
    device: torch.device,
    epochs: int,
    region_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    reuse_current_baseline: bool,
) -> tuple[Path, str]:
    if (
        reuse_current_baseline
        and phase == "full"
        and variant.name == "acq_dim64_current"
        and seed in FULL_SEEDS
    ):
        path = reference_projected_path(fold, seed)
        if not path.is_file():
            raise projection.ExperimentError(f"Missing baseline reference projection: {path}")
        return path, "existing_true_pair_reference"

    path = projected_path_for_run(out_dir, phase, variant, fold, seed)
    if path.is_file():
        return path, "existing_frontier_run"

    run_dir = path.parent
    result = projection.train_one(
        method="pathoalign",
        seed=seed,
        features=features,
        frame=frame,
        train_indices=fit,
        development_indices=np.arange(len(frame), dtype=np.int64),
        groups=groups,
        config=config_for_variant(features.shape[1], variant),
        device=device,
        epochs=epochs,
        region_batch_size=region_batch_size,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        run_dir=run_dir,
    )
    metadata_update = {
        "evaluation_stage": "acquisition_bottleneck_separation_frontier",
        "branch": BRANCH,
        "phase": phase,
        "variant": variant.name,
        "fold": int(fold),
        "seed": int(seed),
        "fit_splits": ["train", "val"],
        "evaluation_split": "test",
        "contains_test_rows": True,
        "frontier_config": variant_dict(variant),
        "training_result": result,
    }
    canine_pair.mark_projection_metadata(path, metadata_update)
    return path, "trained"


def evaluate_projected_file(
    *,
    path: Path,
    manifest: pd.DataFrame,
    fit: np.ndarray,
    test: np.ndarray,
    phase: str,
    variant: FrontierVariant,
    fold: int,
    seed: int,
    source: str,
) -> list[dict[str, object]]:
    biological, acquisition, projected_frame, metadata = canine_pair.load_projected(path)
    biological = align_projected(biological, projected_frame, manifest)
    acquisition = align_projected(acquisition, projected_frame, manifest)
    rows = []
    for branch, branch_features in (("biological", biological), ("acquisition", acquisition)):
        metrics = evaluate_branch(branch_features, manifest, fit, test)
        rows.append({
            "phase": phase,
            "variant": variant.name,
            "branch": branch,
            "fold": int(fold),
            "seed": int(seed),
            "source": source,
            "projected_path": str(path),
            "metadata_method": metadata.get("method", ""),
            **variant_dict(variant),
            **metrics,
        })
    return rows


def write_phase_metrics(out_dir: Path, phase: str, rows: list[dict[str, object]]) -> None:
    filename = "frontier_smoke_raw_metrics.csv" if phase == "smoke" else "frontier_full_raw_metrics.csv"
    atomic_csv(out_dir / filename, pd.DataFrame(rows))


def metric_mean(rows: pd.DataFrame, variant: str, branch: str, metric: str) -> float:
    subset = rows[(rows["variant"] == variant) & (rows["branch"] == branch)]
    if subset.empty:
        return float("nan")
    return float(pd.to_numeric(subset[metric], errors="coerce").mean())


def clipped(value: float, lower: float = 0.0, upper: float = 1.5) -> float:
    if not math.isfinite(value):
        return 0.0
    return float(min(upper, max(lower, value)))


def frontier_score_for_variant(rows: pd.DataFrame, variant: str) -> dict[str, float]:
    acq_scanner = metric_mean(rows, variant, "acquisition", "scanner_balanced_accuracy")
    acq_category = metric_mean(rows, variant, "acquisition", "category_balanced_accuracy")
    bio_scanner = metric_mean(rows, variant, "biological", "scanner_balanced_accuracy")
    bio_category = metric_mean(rows, variant, "biological", "category_balanced_accuracy")

    true_acq = BASELINE_REFERENCES["true_pair_acquisition"]
    true_bio = BASELINE_REFERENCES["true_pair_biological"]
    old_keep = BASELINE_REFERENCES["oldstyle_keep_k4"]
    old_removed = BASELINE_REFERENCES["oldstyle_removed_k4"]

    scanner_span = true_acq["scanner_balanced_accuracy"] - old_removed["scanner_balanced_accuracy"]
    leakage_span = true_acq["category_balanced_accuracy"] - old_removed["category_balanced_accuracy"]
    bio_scanner_span = true_bio["scanner_balanced_accuracy"] - old_keep["scanner_balanced_accuracy"]

    acq_scanner_component = clipped(
        (acq_scanner - old_removed["scanner_balanced_accuracy"]) / max(1e-8, scanner_span)
    )
    acq_leakage_component = clipped(
        (true_acq["category_balanced_accuracy"] - acq_category) / max(1e-8, leakage_span)
    )
    bio_scanner_component = clipped(
        (true_bio["scanner_balanced_accuracy"] - bio_scanner) / max(1e-8, bio_scanner_span)
    )
    bio_category_component = clipped(1.0 - abs(bio_category - true_bio["category_balanced_accuracy"]) / 0.10)

    frontier_score = (
        0.35 * acq_scanner_component
        + 0.35 * acq_leakage_component
        + 0.15 * bio_scanner_component
        + 0.15 * bio_category_component
    )
    return {
        "frontier_score": float(frontier_score),
        "acquisition_scanner_capture": acq_scanner,
        "acquisition_category_leakage": acq_category,
        "biological_scanner_leakage": bio_scanner,
        "biological_category_preservation": bio_category,
        "acq_scanner_component": float(acq_scanner_component),
        "acq_leakage_component": float(acq_leakage_component),
        "bio_scanner_component": float(bio_scanner_component),
        "bio_category_component": float(bio_category_component),
        "criterion_a": bool(
            acq_category < true_acq["category_balanced_accuracy"]
            and acq_scanner > old_removed["scanner_balanced_accuracy"]
        ),
        "criterion_b": bool(
            bio_scanner < true_bio["scanner_balanced_accuracy"]
            and bio_category >= true_bio["category_balanced_accuracy"] - 0.02
        ),
        "criterion_c": bool(
            bio_scanner < true_bio["scanner_balanced_accuracy"]
            and acq_scanner > old_removed["scanner_balanced_accuracy"]
        ),
    }


def build_selection_log(
    out_dir: Path,
    smoke_rows: pd.DataFrame,
    failed_smoke: list[dict[str, object]],
    full_variant_count: int,
) -> tuple[pd.DataFrame, list[str]]:
    rows = []
    for variant in VARIANTS:
        status = "completed" if variant.name in set(smoke_rows["variant"]) else "failed"
        row = {
            "variant": variant.name,
            "phase": "smoke",
            "status": status,
            **variant_dict(variant),
        }
        if status == "completed":
            row.update(frontier_score_for_variant(smoke_rows, variant.name))
        else:
            failure = next((item for item in failed_smoke if item["variant"] == variant.name), {})
            row.update({"failure_reason": failure.get("failure_reason", "unknown")})
        rows.append(row)

    selection = pd.DataFrame(rows)
    completed = selection[selection["status"] == "completed"].copy()
    completed = completed.sort_values(
        ["frontier_score", "criterion_a", "criterion_b", "criterion_c"],
        ascending=[False, False, False, False],
    )
    selected = [str(name) for name in completed.head(full_variant_count)["variant"].tolist()]
    selection["selected_for_full"] = selection["variant"].isin(selected)
    atomic_csv(out_dir / "frontier_variant_selection_log.csv", selection)
    return selection, selected


def build_summary(smoke: pd.DataFrame, full: pd.DataFrame) -> pd.DataFrame:
    frames = []
    for phase, frame in (("smoke", smoke), ("full", full)):
        if frame.empty:
            continue
        grouped = frame.groupby(["phase", "variant", "branch"], dropna=False)
        metrics = grouped[list(METRIC_COLUMNS)].agg(["mean", "std", "min", "max"])
        metrics.columns = ["_".join(col).strip("_") for col in metrics.columns]
        counts = grouped.agg(
            n_rows=("fold", "size"),
            n_folds=("fold", "nunique"),
            n_seeds=("seed", "nunique"),
            acquisition_dim=("acquisition_dim", "first"),
            cross_covariance_weight=("cross_covariance_weight", "first"),
        )
        frames.append(counts.join(metrics).reset_index())
    if not frames:
        return pd.DataFrame()
    summary = pd.concat(frames, ignore_index=True)
    return summary.sort_values(["phase", "variant", "branch"]).reset_index(drop=True)


def build_branch_contrasts(smoke: pd.DataFrame, full: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for phase, frame in (("smoke", smoke), ("full", full)):
        if frame.empty:
            continue
        keys = frame[["variant", "fold", "seed"]].drop_duplicates()
        for key in keys.itertuples(index=False):
            subset = frame[
                (frame["variant"] == key.variant)
                & (frame["fold"] == key.fold)
                & (frame["seed"] == key.seed)
            ]
            bio = subset[subset["branch"] == "biological"]
            acq = subset[subset["branch"] == "acquisition"]
            if bio.empty or acq.empty:
                continue
            bio_row = bio.iloc[0]
            acq_row = acq.iloc[0]
            rows.append({
                "phase": phase,
                "variant": key.variant,
                "fold": int(key.fold),
                "seed": int(key.seed),
                "acquisition_dim": int(bio_row["acquisition_dim"]),
                "cross_covariance_weight": float(bio_row["cross_covariance_weight"]),
                "paired_category_contrast": float(bio_row["category_balanced_accuracy"])
                - float(acq_row["category_balanced_accuracy"]),
                "paired_scanner_contrast": float(acq_row["scanner_balanced_accuracy"])
                - float(bio_row["scanner_balanced_accuracy"]),
                "biological_scanner_leakage": float(bio_row["scanner_balanced_accuracy"]),
                "biological_category_preservation": float(bio_row["category_balanced_accuracy"]),
                "acquisition_scanner_capture": float(acq_row["scanner_balanced_accuracy"]),
                "acquisition_category_leakage": float(acq_row["category_balanced_accuracy"]),
                "acq_category_delta_vs_true_pair_reference": float(acq_row["category_balanced_accuracy"])
                - BASELINE_REFERENCES["true_pair_acquisition"]["category_balanced_accuracy"],
                "acq_scanner_delta_vs_true_pair_reference": float(acq_row["scanner_balanced_accuracy"])
                - BASELINE_REFERENCES["true_pair_acquisition"]["scanner_balanced_accuracy"],
                "bio_scanner_delta_vs_true_pair_reference": float(bio_row["scanner_balanced_accuracy"])
                - BASELINE_REFERENCES["true_pair_biological"]["scanner_balanced_accuracy"],
                "bio_category_delta_vs_true_pair_reference": float(bio_row["category_balanced_accuracy"])
                - BASELINE_REFERENCES["true_pair_biological"]["category_balanced_accuracy"],
            })
    return pd.DataFrame(rows)


def validate_outputs(
    *,
    smoke: pd.DataFrame,
    full: pd.DataFrame,
    selection: pd.DataFrame,
    selected_variants: list[str],
) -> list[str]:
    issues = []
    if smoke.empty:
        issues.append("Smoke metrics are empty.")
    if not set(variant.name for variant in VARIANTS).issubset(set(selection["variant"])):
        issues.append("Selection log does not include all smoke variants.")
    completed = set(selection.loc[selection["status"] == "completed", "variant"])
    missing_completed = completed - set(smoke["variant"])
    if missing_completed:
        issues.append(f"Smoke variants marked completed but missing raw rows: {sorted(missing_completed)}.")
    if selected_variants and not set(selected_variants).issubset(set(full["variant"])):
        issues.append("At least one selected full variant is missing full raw metrics.")

    for label, frame in (("smoke", smoke), ("full", full)):
        if frame.empty:
            continue
        duplicates = int(frame.duplicated(["variant", "branch", "fold", "seed"]).sum())
        if duplicates:
            issues.append(f"{label} duplicate variant/branch/fold/seed rows: {duplicates}.")
        for metric in METRIC_COLUMNS:
            values = pd.to_numeric(frame[metric], errors="coerce")
            if values.isna().any():
                issues.append(f"{label} has missing values in {metric}.")
                continue
            if not np.isfinite(values.to_numpy(dtype=float)).all():
                issues.append(f"{label} has nonfinite values in {metric}.")

    if not BASELINE_REFERENCES:
        issues.append("Baseline references are missing.")
    return issues


def fmt(value: float) -> str:
    if not math.isfinite(float(value)):
        return "NA"
    return f"{float(value):.4f}"


def summary_metric(summary: pd.DataFrame, phase: str, variant: str, branch: str, metric: str) -> float:
    subset = summary[
        (summary["phase"] == phase)
        & (summary["variant"] == variant)
        & (summary["branch"] == branch)
    ]
    if subset.empty:
        return float("nan")
    return float(subset.iloc[0][f"{metric}_mean"])


def build_report(
    *,
    smoke: pd.DataFrame,
    full: pd.DataFrame,
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    selection: pd.DataFrame,
    selected_variants: list[str],
    issues: list[str],
    runtime_seconds: float,
    args: argparse.Namespace,
) -> str:
    lines = [
        "# Acquisition Bottleneck Separation-Frontier Sweep",
        "",
        "## Branch",
        "",
        BRANCH,
        "",
        "## Controllable Parameters Found",
        "",
        "- acquisition_dim: controllable through ProjectionConfig.",
        "- biological_dim: controllable through ProjectionConfig; held fixed at 256.",
        "- hidden_dim: controllable through ProjectionConfig; held fixed at 512.",
        "- reconstruction_weight, variance_weight, covariance_weight: controllable; held at current values.",
        "- scanner_adversary_weight, scanner_acquisition_weight, scanner_dependence_weight: controllable; held at current values.",
        "- cross_covariance_weight: controllable; swept at current 0.05 and stronger 0.20 for selected dimensions.",
        "- gradient_reversal_strength: controllable; held at current 1.0.",
        "- fold and seed controls: available and used.",
        "",
        "## Fixed References",
        "",
        "- true_pair_biological: scanner_acc 0.3614, category_acc 0.3860.",
        "- true_pair_acquisition: scanner_acc 0.8651, category_acc 0.3456.",
        "- oldstyle_keep_k4: scanner_acc 0.2000, category_acc 0.4004.",
        "- oldstyle_removed_k4: scanner_acc 0.5384, category_acc 0.2421.",
        "",
        "## Smoke Variants",
        "",
    ]
    for variant in VARIANTS:
        lines.append(
            f"- {variant.name}: acq_dim={variant.acquisition_dim}, "
            f"cross_covariance_weight={variant.cross_covariance_weight:.2f}."
        )

    lines.extend([
        "",
        "## Selected Full Variants",
        "",
    ])
    if selected_variants:
        for variant in selected_variants:
            score = selection.loc[selection["variant"] == variant, "frontier_score"]
            score_text = fmt(float(score.iloc[0])) if not score.empty and pd.notna(score.iloc[0]) else "NA"
            lines.append(f"- {variant}: selected from smoke frontier_score={score_text}.")
    else:
        lines.append("- None selected.")

    lines.extend([
        "",
        "## Row Counts",
        "",
        f"- Smoke raw rows: {len(smoke)}.",
        f"- Full raw rows: {len(full)}.",
        f"- Variant summary rows: {len(summary)}.",
        f"- Branch contrast rows: {len(contrasts)}.",
        f"- Selection log rows: {len(selection)}.",
        "",
        "## Key Metrics",
        "",
    ])

    for phase in ("smoke", "full"):
        phase_rows = summary[summary["phase"] == phase]
        if phase_rows.empty:
            continue
        lines.append(f"### {phase}")
        for variant in sorted(phase_rows["variant"].unique()):
            bio_scanner = summary_metric(summary, phase, variant, "biological", "scanner_balanced_accuracy")
            bio_category = summary_metric(summary, phase, variant, "biological", "category_balanced_accuracy")
            acq_scanner = summary_metric(summary, phase, variant, "acquisition", "scanner_balanced_accuracy")
            acq_category = summary_metric(summary, phase, variant, "acquisition", "category_balanced_accuracy")
            lines.append(
                f"- {variant}: bio scanner={fmt(bio_scanner)}, bio category={fmt(bio_category)}, "
                f"acq scanner={fmt(acq_scanner)}, acq category={fmt(acq_category)}."
            )
        lines.append("")

    lines.extend([
        "## Frontier Comparison",
        "",
        "Lower acquisition category accuracy means less category leakage in the",
        "scanner/acquisition branch. Higher acquisition scanner accuracy means stronger",
        "scanner capture. Lower biological scanner accuracy means less scanner leakage",
        "in the biological branch.",
        "",
    ])
    if selected_variants:
        for variant in selected_variants:
            bio_scanner = summary_metric(summary, "full", variant, "biological", "scanner_balanced_accuracy")
            bio_category = summary_metric(summary, "full", variant, "biological", "category_balanced_accuracy")
            acq_scanner = summary_metric(summary, "full", variant, "acquisition", "scanner_balanced_accuracy")
            acq_category = summary_metric(summary, "full", variant, "acquisition", "category_balanced_accuracy")
            lines.append(
                f"- {variant}: acq_category_delta_vs_true_pair={fmt(acq_category - 0.3456)}, "
                f"acq_scanner_delta_vs_oldstyle_removed={fmt(acq_scanner - 0.5384)}, "
                f"bio_scanner_delta_vs_true_pair={fmt(bio_scanner - 0.3614)}, "
                f"bio_category_delta_vs_true_pair={fmt(bio_category - 0.3860)}."
            )
    else:
        lines.append("- No full variants available for frontier comparison.")

    lines.extend([
        "",
        "## Validation Checks",
        "",
        f"- Duplicate checks passed: {not any('duplicate' in issue.lower() for issue in issues)}.",
        f"- Nonfinite metric checks passed: {not any('nonfinite' in issue.lower() for issue in issues)}.",
        f"- Smoke variants documented: {len(selection)} / {len(VARIANTS)}.",
        f"- Selected full variants documented: {', '.join(selected_variants) if selected_variants else 'none'}.",
        f"- Baseline references included: {bool(BASELINE_REFERENCES)}.",
        f"- Validation issue count: {len(issues)}.",
    ])
    if issues:
        for issue in issues:
            lines.append(f"  - {issue}")
    else:
        lines.append("  - No validation issues found.")

    lines.extend([
        "",
        "## Bounded Interpretation",
        "",
        "This sweep tests whether acquisition bottleneck capacity moves the separation",
        "frontier in this audit. It is not a use-context or downstream-care claim.",
        "The oldstyle centroid/QR reference remains the strongest raw scanner-removal",
        "baseline. The paired-acquisition target here is structured separation: keeping",
        "an explicit scanner-bearing acquisition branch while reducing category leakage.",
        "",
        "## Key Questions",
        "",
        "1. Does reducing acquisition capacity reduce acquisition category leakage?",
        "   See acquisition category metrics and deltas above.",
        "2. Does scanner capture survive the bottleneck?",
        "   See acquisition scanner metrics above.",
        "3. Does biological branch scanner leakage improve?",
        "   See biological scanner metrics above.",
        "4. Does category preservation degrade?",
        "   See biological category metrics above.",
        "5. Does any variant move the separation frontier?",
        "   A move is supported when leakage falls while scanner capture remains above oldstyle_removed_k4.",
        "6. Does this weaken or strengthen the paired-acquisition mechanism story?",
        "   It strengthens the mechanism story only if structured separation improves without hiding the oldstyle raw-removal boundary.",
        "7. Does oldstyle centroid/QR remain the best raw scanner-removal baseline?",
        "   Yes unless a full variant moves biological scanner below 0.2000 with comparable category preservation.",
        "",
        "## Files Created",
        "",
        "- frontier_smoke_raw_metrics.csv",
        "- frontier_full_raw_metrics.csv",
        "- frontier_variant_summary.csv",
        "- frontier_branch_contrasts.csv",
        "- frontier_variant_selection_log.csv",
        "- acquisition_bottleneck_separation_frontier_report.md",
        "- experiment_design.json",
        "- run_log.txt",
        "",
        f"Runtime seconds: {runtime_seconds:.1f}",
        f"Epochs: {args.epochs}",
        f"Device: {args.device}",
        "",
        "## Readiness",
        "",
        "Ready to commit after external diff hygiene checks pass; no staging or commit performed.",
        "",
    ])
    return "\n".join(lines)


def prepare_fold(
    *,
    fold: int,
    base_features: np.ndarray,
    base_frame: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    manifest = load_manifest(fold)
    features, frame = canine_cross.align_fold(base_features, base_frame, manifest_path(fold))
    fit, test = canine_cross.validate_fold(manifest, fold)
    transformed, _mean, _std = projection.standardize(features, fit)
    return manifest, transformed, frame, fit, test, np.arange(len(frame), dtype=np.int64)


def run_single_variant(
    *,
    out_dir: Path,
    phase: str,
    variant: FrontierVariant,
    fold: int,
    seed: int,
    base_features: np.ndarray,
    base_frame: pd.DataFrame,
    device: torch.device,
    args: argparse.Namespace,
) -> list[dict[str, object]]:
    manifest, transformed, frame, fit, test, _all_indices = prepare_fold(
        fold=fold,
        base_features=base_features,
        base_frame=base_frame,
    )
    groups, assignments, audit = canine_pair.build_pair_groups(
        frame,
        fit,
        condition="true_pairs",
        fold=fold,
        seed=seed,
    )
    assignment_path = out_dir / "pair_assignments" / phase / f"fold_{fold}_{variant.name}_seed_{seed}.csv"
    atomic_csv(assignment_path, assignments)
    projected, source = train_or_reuse_projected(
        out_dir=out_dir,
        phase=phase,
        variant=variant,
        fold=fold,
        seed=seed,
        features=transformed,
        frame=frame,
        fit=fit,
        groups=groups,
        device=device,
        epochs=args.epochs,
        region_batch_size=args.region_batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        reuse_current_baseline=args.reuse_current_baseline,
    )
    rows = evaluate_projected_file(
        path=projected,
        manifest=manifest,
        fit=fit,
        test=test,
        phase=phase,
        variant=variant,
        fold=fold,
        seed=seed,
        source=source,
    )
    for row in rows:
        row.update({
            "n_fit_rows": int(len(fit)),
            "n_test_rows": int(len(test)),
            "n_fit_samples": int(manifest.iloc[fit]["sample_id"].nunique()),
            "n_test_samples": int(manifest.iloc[test]["sample_id"].nunique()),
            "n_fit_regions": int(manifest.iloc[fit]["region_id"].nunique()),
            "n_test_regions": int(manifest.iloc[test]["region_id"].nunique()),
            "pair_non_anchor_region_mismatch_fraction": float(audit["non_anchor_region_mismatch_fraction"]),
            "pair_non_anchor_same_sample_fraction": float(audit["non_anchor_same_sample_fraction"]),
        })
    return rows


def append_rows(existing: pd.DataFrame, new_rows: list[dict[str, object]]) -> pd.DataFrame:
    if not new_rows:
        return existing
    combined = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True) if not existing.empty else pd.DataFrame(new_rows)
    return combined.sort_values(["phase", "variant", "fold", "seed", "branch"]).reset_index(drop=True)


def has_completed_run(frame: pd.DataFrame, variant: str, fold: int, seed: int) -> bool:
    if frame.empty:
        return False
    subset = frame[
        (frame["variant"] == variant)
        & (frame["fold"] == fold)
        & (frame["seed"] == seed)
    ]
    return set(subset["branch"]) == {"biological", "acquisition"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/paired_acquisition_factorization_acquisition_bottleneck_separation_frontier"),
    )
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--region-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--full-variant-count", type=int, default=2)
    parser.add_argument("--reuse-current-baseline", action="store_true", default=True)
    parser.add_argument("--no-reuse-current-baseline", dest="reuse_current_baseline", action="store_false")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "run_log.txt"
    start = time.perf_counter()

    with log_path.open("a", encoding="utf-8") as log_file:
        with redirect_stdout(Tee(sys.stdout, log_file)), redirect_stderr(Tee(sys.stderr, log_file)):
            print("\n" + "=" * 80)
            print(time.strftime("%Y-%m-%d %H:%M:%S"))
            print("COMMAND " + " ".join(sys.argv))
            try:
                canine_cross.patch_scanner_namespace()
                base_features, base_frame, source_metadata = projection.load_archive(FEATURE_PATH)
                base_frame["scanner_id"] = base_frame["scanner_id"].astype(str).str.lower()
                if base_features.shape != (4025, 768):
                    raise projection.ExperimentError(f"Unexpected canine feature shape: {base_features.shape}")

                device = torch.device(args.device)
                if device.type == "cuda" and not torch.cuda.is_available():
                    raise projection.ExperimentError("CUDA requested but unavailable.")

                design = {
                    "branch": BRANCH,
                    "stage": "acquisition_bottleneck_separation_frontier",
                    "dataset": "canine_cutaneous_scc_dinov2",
                    "feature_path": str(FEATURE_PATH),
                    "columns": {
                        "label": "category_name",
                        "scanner": "scanner_id",
                        "sample": "sample_id",
                        "region": "region_id",
                    },
                    "controllable_parameters": [
                        "acquisition_dim",
                        "biological_dim",
                        "hidden_dim",
                        "reconstruction_weight",
                        "variance_weight",
                        "covariance_weight",
                        "scanner_adversary_weight",
                        "scanner_acquisition_weight",
                        "scanner_dependence_weight",
                        "cross_covariance_weight",
                        "gradient_reversal_strength",
                        "fold",
                        "seed",
                    ],
                    "baseline_references": BASELINE_REFERENCES,
                    "smoke": {"fold": SMOKE_FOLD, "seed": SMOKE_SEED, "variants": [variant_dict(v) for v in VARIANTS]},
                    "full": {"folds": list(FOLDS), "seeds": list(FULL_SEEDS), "variant_count": args.full_variant_count},
                    "epochs": args.epochs,
                    "region_batch_size": args.region_batch_size,
                    "learning_rate": args.learning_rate,
                    "weight_decay": args.weight_decay,
                    "device": str(device),
                    "reuse_current_baseline": bool(args.reuse_current_baseline),
                    "source_metadata": source_metadata,
                    "command": " ".join(sys.argv),
                }
                atomic_text(args.out_dir / "experiment_design.json", json.dumps(design, indent=2, sort_keys=True) + "\n")

                smoke_path = args.out_dir / "frontier_smoke_raw_metrics.csv"
                full_path = args.out_dir / "frontier_full_raw_metrics.csv"
                smoke_rows = load_existing_csv(smoke_path)
                full_rows = load_existing_csv(full_path)
                failed_smoke: list[dict[str, object]] = []

                print("\nPHASE A SMOKE")
                for variant in VARIANTS:
                    if has_completed_run(smoke_rows, variant.name, SMOKE_FOLD, SMOKE_SEED):
                        print(f"Skipping completed smoke variant={variant.name}")
                        continue
                    print(
                        f"Smoke train/eval variant={variant.name} acq_dim={variant.acquisition_dim} "
                        f"xcov={variant.cross_covariance_weight}"
                    )
                    try:
                        new_rows = run_single_variant(
                            out_dir=args.out_dir,
                            phase="smoke",
                            variant=variant,
                            fold=SMOKE_FOLD,
                            seed=SMOKE_SEED,
                            base_features=base_features,
                            base_frame=base_frame,
                            device=device,
                            args=args,
                        )
                        smoke_rows = append_rows(smoke_rows, new_rows)
                        write_phase_metrics(args.out_dir, "smoke", smoke_rows.to_dict("records"))
                    except Exception as exc:
                        failed_smoke.append({
                            "variant": variant.name,
                            "phase": "smoke",
                            "status": "failed",
                            "failure_reason": repr(exc),
                        })
                        print(f"Smoke variant failed: {variant.name}: {exc!r}")

                if smoke_rows.empty:
                    raise projection.ExperimentError("All smoke variants failed; cannot select full variants.")

                selection, selected_variants = build_selection_log(
                    args.out_dir,
                    smoke_rows,
                    failed_smoke,
                    args.full_variant_count,
                )
                print("Selected full variants: " + ", ".join(selected_variants))

                variant_lookup = {variant.name: variant for variant in VARIANTS}
                print("\nPHASE B FULL")
                for variant_name in selected_variants:
                    variant = variant_lookup[variant_name]
                    for fold in FOLDS:
                        for seed in FULL_SEEDS:
                            if has_completed_run(full_rows, variant.name, fold, seed):
                                print(f"Skipping completed full variant={variant.name} fold={fold} seed={seed}")
                                continue
                            print(
                                f"Full train/eval variant={variant.name} fold={fold} seed={seed} "
                                f"acq_dim={variant.acquisition_dim} xcov={variant.cross_covariance_weight}"
                            )
                            new_rows = run_single_variant(
                                out_dir=args.out_dir,
                                phase="full",
                                variant=variant,
                                fold=fold,
                                seed=seed,
                                base_features=base_features,
                                base_frame=base_frame,
                                device=device,
                                args=args,
                            )
                            full_rows = append_rows(full_rows, new_rows)
                            write_phase_metrics(args.out_dir, "full", full_rows.to_dict("records"))

                summary = build_summary(smoke_rows, full_rows)
                contrasts = build_branch_contrasts(smoke_rows, full_rows)
                issues = validate_outputs(
                    smoke=smoke_rows,
                    full=full_rows,
                    selection=selection,
                    selected_variants=selected_variants,
                )
                atomic_csv(args.out_dir / "frontier_variant_summary.csv", summary)
                atomic_csv(args.out_dir / "frontier_branch_contrasts.csv", contrasts)
                runtime = time.perf_counter() - start
                report = build_report(
                    smoke=smoke_rows,
                    full=full_rows,
                    summary=summary,
                    contrasts=contrasts,
                    selection=selection,
                    selected_variants=selected_variants,
                    issues=issues,
                    runtime_seconds=runtime,
                    args=args,
                )
                atomic_text(args.out_dir / "acquisition_bottleneck_separation_frontier_report.md", report)

                print("\n" + "=" * 80)
                print("ACQUISITION BOTTLENECK SEPARATION-FRONTIER SWEEP COMPLETE")
                print(f"Smoke rows: {len(smoke_rows)}")
                print(f"Full rows: {len(full_rows)}")
                print(f"Selected: {', '.join(selected_variants)}")
                print(f"Validation issues: {len(issues)}")
                for issue in issues:
                    print(f"  - {issue}")
                print(f"Runtime: {runtime:.1f}s")
                print(f"Report: {(args.out_dir / 'acquisition_bottleneck_separation_frontier_report.md').resolve()}")

            except Exception:
                traceback.print_exc()
                raise


if __name__ == "__main__":
    main()
