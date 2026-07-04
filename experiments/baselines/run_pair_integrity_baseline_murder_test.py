#!/usr/bin/env python3
"""Run scanner-removal baseline murder tests for paired-acquisition factorization.

The test asks whether simple post-hoc operations on frozen embeddings can match
the scanner suppression of Paired-Acquisition Neural Factorization without
damaging same-tissue preservation metrics.
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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.canine import run_pair_integrity_falsification_caninescc as canine_pair  # noqa: E402
from experiments.external_multiscanner import run_canine_pathoalign_crossfold as canine_cross  # noqa: E402
from experiments.scorpion import run_pair_integrity_falsification as scorpion_pair  # noqa: E402
from experiments.scorpion import run_pathoalign_crossfold as scorpion_cross  # noqa: E402
from experiments.scorpion import run_pathoalign_projection as projection  # noqa: E402


K_VALUES = (0, 1, 2, 4, 8, 16, 32)
REQUIRED_METRICS = (
    "scanner_probe_accuracy",
    "mean_paired_cosine",
    "worst_paired_cosine",
    "mean_top1_retrieval",
    "worst_top1_retrieval",
    "effective_rank",
    "runtime_seconds",
)
LOWER_IS_BETTER = {"scanner_probe_accuracy", "runtime_seconds"}
NEURAL_REFERENCE = "paired_acquisition_neural_factorization_reference"
PAIR_REFERENCE = "paired_consistency_reference"
SIMPLE_FAMILIES = {
    "original_frozen_features",
    "linear_scanner_subspace_projection",
    "pca_component_removal",
}


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


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    label: str
    feature_path: Path
    manifests_dir: Path
    manifest_patterns: tuple[str, ...]
    reference_dir: Path
    reference_seeds: tuple[int, ...]
    expected_rows: int
    expected_blocks: int
    block_label: str
    align_fold: Callable[[np.ndarray, pd.DataFrame, Path], tuple[np.ndarray, pd.DataFrame]]
    validate_fold: Callable[[pd.DataFrame, int], tuple[np.ndarray, np.ndarray]]
    scanner_probe: Callable[[np.ndarray, pd.DataFrame, np.ndarray, np.ndarray], tuple[float, pd.DataFrame]]
    paired_metrics: Callable[[np.ndarray, pd.DataFrame, np.ndarray], tuple[dict[str, float], pd.DataFrame]]
    effective_rank: Callable[[np.ndarray], float]
    patch_namespace: Callable[[], None] | None = None


DATASETS = {
    "scorpion": DatasetConfig(
        name="scorpion",
        label="SCORPION DINOv2",
        feature_path=Path("results/scorpion/features/fold_0_dinov2_base.npz"),
        manifests_dir=Path("data/scorpion/splits"),
        manifest_patterns=("fold_{fold}_manifest.csv",),
        reference_dir=Path("results/scorpion/pathoalign_dinov2_crossfold"),
        reference_seeds=(601, 602, 603, 604, 605),
        expected_rows=2400,
        expected_blocks=48,
        block_label="slide",
        align_fold=scorpion_cross.align_fold,
        validate_fold=scorpion_cross.validate_fold,
        scanner_probe=scorpion_pair.scanner_probe,
        paired_metrics=scorpion_pair.paired_metrics,
        effective_rank=scorpion_pair.effective_rank,
    ),
    "caninescc": DatasetConfig(
        name="caninescc",
        label="External canine SCC DINOv2",
        feature_path=Path("results/external_multiscanner_caninescc/features/fold_0_dinov2_base.npz"),
        manifests_dir=Path("data/external_multiscanner_caninescc/patch_manifests/splits"),
        manifest_patterns=("fold_{fold}_patch_manifest.csv", "fold_{fold}_manifest.csv"),
        reference_dir=Path("results/external_multiscanner_caninescc/pathoalign_dinov2_crossfold"),
        reference_seeds=(911, 912, 913, 914, 915),
        expected_rows=4025,
        expected_blocks=44,
        block_label="sample",
        align_fold=canine_cross.align_fold,
        validate_fold=canine_cross.validate_fold,
        scanner_probe=canine_pair.scanner_probe,
        paired_metrics=canine_pair.paired_metrics,
        effective_rank=canine_pair.effective_rank,
        patch_namespace=canine_cross.patch_scanner_namespace,
    ),
}


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


def resolve_manifest(config: DatasetConfig, fold: int) -> Path:
    attempted = []
    for pattern in config.manifest_patterns:
        path = config.manifests_dir / pattern.format(fold=fold)
        attempted.append(str(path))
        if path.is_file():
            return path
    raise projection.ExperimentError(
        f"Missing {config.name} manifest for fold={fold}; attempted {attempted}"
    )


def split_indices(frame: pd.DataFrame, block_column: str) -> tuple[np.ndarray, np.ndarray]:
    split = frame["split"].astype(str).to_numpy()
    test = np.flatnonzero(split == "test")
    fit = np.flatnonzero(split != "test")
    if len(test) == 0 or len(fit) == 0:
        raise projection.ExperimentError("Empty fit or test split.")
    fit_blocks = set(frame.iloc[fit][block_column].astype(str))
    test_blocks = set(frame.iloc[test][block_column].astype(str))
    overlap = sorted(fit_blocks & test_blocks)
    if overlap:
        raise projection.ExperimentError(f"Fit/test leakage through {block_column}: {overlap[:10]}")
    return fit, test


def load_projected(path: Path) -> tuple[np.ndarray, pd.DataFrame, dict[str, object]]:
    with np.load(path, allow_pickle=False) as archive:
        required = {"features", "slide_id", "region_id", "scanner_id", "split"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise projection.ExperimentError(f"{path} is missing arrays: {missing}")
        features = np.asarray(archive["features"], dtype=np.float32)
        frame = pd.DataFrame(
            {name: archive[name].astype(str) for name in ("slide_id", "region_id", "scanner_id", "split")}
        )
        metadata = {}
        if "metadata_json" in archive.files:
            metadata = json.loads(str(archive["metadata_json"].item()))
    if features.ndim != 2 or len(features) != len(frame):
        raise projection.ExperimentError(f"Projected feature/frame mismatch in {path}")
    if not np.isfinite(features).all():
        raise projection.ExperimentError(f"Projected features contain nonfinite values: {path}")
    return features, frame, metadata


def evaluate_feature_matrix(
    *,
    config: DatasetConfig,
    baseline: str,
    baseline_family: str,
    reference_type: str,
    fold: int,
    seed: int,
    k: int | None,
    effective_k: int | None,
    fitted_rank: int | None,
    features: np.ndarray,
    frame: pd.DataFrame,
    fit: np.ndarray,
    test: np.ndarray,
    source_path: Path | None,
    runtime_start: float,
    extra_runtime_seconds: float = 0.0,
) -> dict[str, object]:
    scanner_accuracy, _ = config.scanner_probe(features, frame, fit, test)
    paired, _ = config.paired_metrics(features, frame, test)
    test_features = np.asarray(features[test], dtype=np.float32)
    row = {
        "dataset": config.name,
        "dataset_label": config.label,
        "baseline": baseline,
        "baseline_family": baseline_family,
        "reference_type": reference_type,
        "fold": int(fold),
        "seed": int(seed),
        "k": "" if k is None else int(k),
        "effective_k": "" if effective_k is None else int(effective_k),
        "fitted_rank": "" if fitted_rank is None else int(fitted_rank),
        "scanner_probe_accuracy": float(scanner_accuracy),
        "mean_paired_cosine": float(paired["mean_paired_cosine"]),
        "worst_paired_cosine": float(paired["worst_paired_cosine"]),
        "mean_top1_retrieval": float(paired["mean_top1_retrieval"]),
        "worst_top1_retrieval": float(paired["worst_top1_retrieval"]),
        "effective_rank": float(config.effective_rank(test_features)),
        "runtime_seconds": float(extra_runtime_seconds + time.perf_counter() - runtime_start),
        "n_test_rows": int(len(test)),
        "n_test_blocks": int(frame.iloc[test]["slide_id"].nunique()),
        "source_path": "" if source_path is None else str(source_path.resolve()),
    }
    return row


def scanner_projection_directions(
    features: np.ndarray,
    frame: pd.DataFrame,
    fit: np.ndarray,
) -> tuple[np.ndarray, int, float]:
    start = time.perf_counter()
    labels = frame["scanner_id"].astype(str).to_numpy()
    model = LogisticRegression(
        C=1.0,
        class_weight="balanced",
        max_iter=5000,
        random_state=0,
        solver="lbfgs",
    )
    model.fit(features[fit], labels[fit])
    coefficients = np.asarray(model.coef_, dtype=np.float64)
    coefficients = coefficients - coefficients.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(coefficients, full_matrices=False)
    rank = numeric_rank(singular_values, coefficients.shape)
    return vt[:rank].astype(np.float32), rank, time.perf_counter() - start


def pca_directions(features: np.ndarray, fit: np.ndarray) -> tuple[np.ndarray, np.ndarray, int, float]:
    start = time.perf_counter()
    fit_features = np.asarray(features[fit], dtype=np.float64)
    center = fit_features.mean(axis=0, keepdims=True)
    centered = fit_features - center
    _, singular_values, vt = np.linalg.svd(centered, full_matrices=False)
    rank = numeric_rank(singular_values, centered.shape)
    return center.astype(np.float32), vt[:rank].astype(np.float32), rank, time.perf_counter() - start


def numeric_rank(singular_values: np.ndarray, shape: tuple[int, int]) -> int:
    if singular_values.size == 0:
        return 0
    tolerance = max(shape) * np.finfo(np.float64).eps * float(singular_values[0])
    return int(np.sum(singular_values > tolerance))


def remove_directions(
    features: np.ndarray,
    directions: np.ndarray,
    k: int,
    *,
    center: np.ndarray | None = None,
) -> tuple[np.ndarray, int]:
    effective_k = min(int(k), len(directions))
    if effective_k <= 0:
        return np.asarray(features, dtype=np.float32).copy(), 0
    matrix = np.asarray(features, dtype=np.float64)
    if center is None:
        centered = matrix
        offset = 0.0
    else:
        offset = np.asarray(center, dtype=np.float64)
        centered = matrix - offset
    basis = np.asarray(directions[:effective_k], dtype=np.float64)
    residual = centered - (centered @ basis.T) @ basis
    if center is not None:
        residual = residual + offset
    residual = residual.astype(np.float32)
    if not np.isfinite(residual).all():
        raise projection.ExperimentError("Component removal produced nonfinite values.")
    return residual, effective_k


def evaluate_reference_variant(
    config: DatasetConfig,
    *,
    folds: tuple[int, ...],
    variant: str,
    baseline: str,
    reference_type: str,
) -> tuple[list[dict[str, object]], str | None]:
    expected_paths = [
        config.reference_dir / f"fold_{fold}" / "runs" / f"{variant}_seed_{seed}" / "projected_features.npz"
        for fold in folds
        for seed in config.reference_seeds
    ]
    missing = [str(path) for path in expected_paths if not path.is_file()]
    if missing:
        return [], f"{baseline} skipped for {config.name}; missing projected feature files: {missing[:3]}"

    rows = []
    for fold in folds:
        for seed in config.reference_seeds:
            projected = (
                config.reference_dir
                / f"fold_{fold}"
                / "runs"
                / f"{variant}_seed_{seed}"
                / "projected_features.npz"
            )
            start = time.perf_counter()
            features, frame, metadata = load_projected(projected)
            if metadata.get("fold") is not None and int(metadata["fold"]) != int(fold):
                raise projection.ExperimentError(f"Reference metadata fold mismatch in {projected}")
            fit, test = split_indices(frame, "slide_id")
            rows.append(
                evaluate_feature_matrix(
                    config=config,
                    baseline=baseline,
                    baseline_family=baseline,
                    reference_type=reference_type,
                    fold=fold,
                    seed=seed,
                    k=None,
                    effective_k=None,
                    fitted_rank=None,
                    features=features,
                    frame=frame,
                    fit=fit,
                    test=test,
                    source_path=projected,
                    runtime_start=start,
                )
            )
    return rows, None


def run_dataset(config: DatasetConfig, folds: tuple[int, ...]) -> tuple[list[dict[str, object]], list[str], list[str]]:
    if config.patch_namespace is not None:
        config.patch_namespace()

    base_features, base_frame, source_metadata = projection.load_archive(config.feature_path)
    if len(base_features) != config.expected_rows:
        raise projection.ExperimentError(
            f"{config.name} expected {config.expected_rows} frozen rows; observed {len(base_features)}"
        )
    if "dinov2" not in str(source_metadata.get("model", "")).lower():
        raise projection.ExperimentError(
            f"{config.name} source metadata does not identify DINOv2: {source_metadata.get('model')!r}"
        )

    print(f"\nDATASET {config.label}")
    print(f"Frozen features: {config.feature_path.resolve()}")
    rows: list[dict[str, object]] = []
    skipped: list[str] = []
    manifest_notes: list[str] = []

    for fold in folds:
        manifest_path = resolve_manifest(config, fold)
        manifest_notes.append(f"{config.name} fold {fold}: {manifest_path.resolve()}")
        features, frame = config.align_fold(base_features, base_frame, manifest_path)
        fit, test = config.validate_fold(frame, fold)
        print(
            f"Evaluating {config.name} fold={fold}: fit_rows={len(fit)} test_rows={len(test)} "
            f"test_{config.block_label}s={frame.iloc[test]['slide_id'].nunique()}"
        )

        start = time.perf_counter()
        rows.append(
            evaluate_feature_matrix(
                config=config,
                baseline="original_frozen_features",
                baseline_family="original_frozen_features",
                reference_type="simple_baseline",
                fold=fold,
                seed=0,
                k=None,
                effective_k=None,
                fitted_rank=None,
                features=features,
                frame=frame,
                fit=fit,
                test=test,
                source_path=config.feature_path,
                runtime_start=start,
            )
        )

        standardized, _, _ = projection.standardize(features, fit)
        scanner_directions, scanner_rank, scanner_fit_seconds = scanner_projection_directions(
            standardized, frame, fit
        )
        pca_center, pca_basis, pca_rank, pca_fit_seconds = pca_directions(standardized, fit)

        for k in K_VALUES:
            start = time.perf_counter()
            transformed, effective_k = remove_directions(standardized, scanner_directions, k)
            rows.append(
                evaluate_feature_matrix(
                    config=config,
                    baseline=f"linear_scanner_subspace_projection_k{k}",
                    baseline_family="linear_scanner_subspace_projection",
                    reference_type="simple_baseline",
                    fold=fold,
                    seed=0,
                    k=k,
                    effective_k=effective_k,
                    fitted_rank=scanner_rank,
                    features=transformed,
                    frame=frame,
                    fit=fit,
                    test=test,
                    source_path=config.feature_path,
                    runtime_start=start,
                    extra_runtime_seconds=scanner_fit_seconds,
                )
            )

            start = time.perf_counter()
            transformed, effective_k = remove_directions(standardized, pca_basis, k, center=pca_center)
            rows.append(
                evaluate_feature_matrix(
                    config=config,
                    baseline=f"pca_component_removal_k{k}",
                    baseline_family="pca_component_removal",
                    reference_type="simple_baseline",
                    fold=fold,
                    seed=0,
                    k=k,
                    effective_k=effective_k,
                    fitted_rank=pca_rank,
                    features=transformed,
                    frame=frame,
                    fit=fit,
                    test=test,
                    source_path=config.feature_path,
                    runtime_start=start,
                    extra_runtime_seconds=pca_fit_seconds,
                )
            )

    reference_rows, reason = evaluate_reference_variant(
        config,
        folds=folds,
        variant="paired_reference",
        baseline=PAIR_REFERENCE,
        reference_type="paired_reference",
    )
    rows.extend(reference_rows)
    if reason:
        skipped.append(reason)

    reference_rows, reason = evaluate_reference_variant(
        config,
        folds=folds,
        variant="pathoalign_dep20",
        baseline=NEURAL_REFERENCE,
        reference_type="neural_reference",
    )
    rows.extend(reference_rows)
    if reason:
        skipped.append(reason)

    skipped.append(
        f"random_pair_training skipped for {config.name}; no safe existing baseline mode was available without "
        "new pair-training semantics beyond the completed shuffled-pair falsification controls."
    )
    skipped.append(
        f"optional ablations skipped for {config.name}; adversarial_only/no_acquisition_branch/"
        "no_covariance_penalty were not available as clean locked runners."
    )
    return rows, skipped, manifest_notes


def summarize(raw: pd.DataFrame) -> pd.DataFrame:
    group_columns = ["dataset", "dataset_label", "baseline", "baseline_family", "reference_type"]
    grouped = raw.groupby(group_columns, dropna=False)
    metric_summary = grouped[list(REQUIRED_METRICS)].agg(["mean", "std", "min", "max"])
    metric_summary.columns = ["_".join(column).strip("_") for column in metric_summary.columns]
    counts = grouped.agg(
        n_runs=("fold", "size"),
        n_folds=("fold", "nunique"),
        n_seeds=("seed", "nunique"),
        k=("k", "first"),
        max_effective_k=("effective_k", "max"),
        max_fitted_rank=("fitted_rank", "max"),
    )
    summary = counts.join(metric_summary).reset_index()
    return summary.sort_values(["dataset", "reference_type", "baseline"]).reset_index(drop=True)


def build_contrasts(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, group in summary.groupby("dataset", sort=True):
        neural = group.loc[group["baseline"] == NEURAL_REFERENCE]
        if neural.empty:
            continue
        neural_row = neural.iloc[0]
        for _, row in group.iterrows():
            if row["baseline"] == NEURAL_REFERENCE:
                continue
            for metric in REQUIRED_METRICS:
                baseline_value = float(row[f"{metric}_mean"])
                neural_value = float(neural_row[f"{metric}_mean"])
                difference = baseline_value - neural_value
                favorable = difference < 0 if metric in LOWER_IS_BETTER else difference > 0
                rows.append(
                    {
                        "dataset": dataset,
                        "baseline": row["baseline"],
                        "baseline_family": row["baseline_family"],
                        "metric": metric,
                        "difference_definition": f"{row['baseline']}_minus_{NEURAL_REFERENCE}",
                        "baseline_mean": baseline_value,
                        "neural_factorization_mean": neural_value,
                        "mean_difference": float(difference),
                        "favorable_vs_neural_factorization": bool(favorable),
                    }
                )
    return pd.DataFrame(rows)


def validate_outputs(
    raw: pd.DataFrame,
    *,
    datasets: tuple[str, ...],
    folds: tuple[int, ...],
) -> None:
    missing = [metric for metric in REQUIRED_METRICS if metric not in raw.columns]
    if missing:
        raise projection.ExperimentError(f"Missing required metric columns: {missing}")
    for metric in REQUIRED_METRICS:
        values = pd.to_numeric(raw[metric], errors="coerce")
        if values.isna().any():
            bad = raw.loc[values.isna(), ["dataset", "baseline", "fold", "seed"]].head().to_dict("records")
            raise projection.ExperimentError(f"Missing values for {metric}: {bad}")
        if not np.isfinite(values.to_numpy(float)).all():
            raise projection.ExperimentError(f"Nonfinite values found in {metric}")
    if raw.duplicated(["dataset", "fold", "seed", "baseline"]).any():
        duplicates = raw.loc[
            raw.duplicated(["dataset", "fold", "seed", "baseline"], keep=False),
            ["dataset", "fold", "seed", "baseline"],
        ].head().to_dict("records")
        raise projection.ExperimentError(f"Duplicate dataset/fold/seed/baseline rows: {duplicates}")

    for dataset in datasets:
        config = DATASETS[dataset]
        dataset_rows = raw.loc[raw["dataset"] == dataset]
        if set(dataset_rows["fold"].astype(int).unique()) != set(folds):
            raise projection.ExperimentError(f"{dataset} did not cover expected folds {folds}")
        simple = dataset_rows.loc[dataset_rows["reference_type"] == "simple_baseline"]
        expected_simple = len(folds) * (1 + 2 * len(K_VALUES))
        if len(simple) != expected_simple:
            raise projection.ExperimentError(
                f"{dataset} expected {expected_simple} simple-baseline rows; observed {len(simple)}"
            )
        for baseline in (PAIR_REFERENCE, NEURAL_REFERENCE):
            rows = dataset_rows.loc[dataset_rows["baseline"] == baseline]
            if len(rows) == 0:
                continue
            expected_reference = len(folds) * len(config.reference_seeds)
            if len(rows) != expected_reference:
                raise projection.ExperimentError(
                    f"{dataset} {baseline} expected {expected_reference} rows; observed {len(rows)}"
                )
            if set(rows["seed"].astype(int).unique()) != set(config.reference_seeds):
                raise projection.ExperimentError(f"{dataset} {baseline} seed coverage mismatch.")


def classify_dataset(summary: pd.DataFrame, dataset: str) -> dict[str, object]:
    group = summary.loc[summary["dataset"] == dataset].copy()
    neural = group.loc[group["baseline"] == NEURAL_REFERENCE]
    original = group.loc[group["baseline"] == "original_frozen_features"]
    if neural.empty:
        return {
            "dataset": dataset,
            "classification": "neural factorization reference unavailable; no conclusion.",
            "best_simple_baseline": "unavailable",
        }
    neural_row = neural.iloc[0]
    original_scanner = (
        float(original.iloc[0]["scanner_probe_accuracy_mean"])
        if not original.empty
        else math.inf
    )
    simple = group.loc[group["baseline_family"].isin(SIMPLE_FAMILIES)].copy()
    simple["scanner_matches_neural"] = (
        simple["scanner_probe_accuracy_mean"].astype(float)
        <= float(neural_row["scanner_probe_accuracy_mean"]) + 0.03
    )
    simple["tissue_preserved_vs_neural"] = (
        (simple["mean_paired_cosine_mean"].astype(float) >= float(neural_row["mean_paired_cosine_mean"]) - 0.005)
        & (simple["worst_paired_cosine_mean"].astype(float) >= float(neural_row["worst_paired_cosine_mean"]) - 0.005)
        & (simple["mean_top1_retrieval_mean"].astype(float) >= float(neural_row["mean_top1_retrieval_mean"]) - 0.005)
        & (simple["worst_top1_retrieval_mean"].astype(float) >= float(neural_row["worst_top1_retrieval_mean"]) - 0.005)
    )
    simple["suppresses_vs_original"] = (
        simple["scanner_probe_accuracy_mean"].astype(float) <= original_scanner - 0.05
    )
    simple["outperforms_neural"] = (
        (simple["scanner_probe_accuracy_mean"].astype(float) < float(neural_row["scanner_probe_accuracy_mean"]) - 0.01)
        & (simple["mean_paired_cosine_mean"].astype(float) >= float(neural_row["mean_paired_cosine_mean"]))
        & (simple["mean_top1_retrieval_mean"].astype(float) >= float(neural_row["mean_top1_retrieval_mean"]) - 0.001)
    )
    simple["tissue_damage_score"] = np.maximum(
        0.0,
        float(neural_row["mean_paired_cosine_mean"]) - simple["mean_paired_cosine_mean"].astype(float),
    ) + np.maximum(
        0.0,
        float(neural_row["mean_top1_retrieval_mean"]) - simple["mean_top1_retrieval_mean"].astype(float),
    )
    simple["scanner_gap_to_neural"] = (
        simple["scanner_probe_accuracy_mean"].astype(float) - float(neural_row["scanner_probe_accuracy_mean"])
    ).abs()
    best = simple.sort_values(
        ["tissue_damage_score", "scanner_gap_to_neural", "scanner_probe_accuracy_mean"],
        ascending=[True, True, True],
    ).iloc[0]

    linear_matches = bool(
        (
            (simple["baseline_family"] == "linear_scanner_subspace_projection")
            & simple["scanner_matches_neural"]
            & simple["tissue_preserved_vs_neural"]
        ).any()
    )
    pca_matches = bool(
        (
            (simple["baseline_family"] == "pca_component_removal")
            & simple["scanner_matches_neural"]
            & simple["tissue_preserved_vs_neural"]
        ).any()
    )
    any_outperforms = bool(simple["outperforms_neural"].any())
    any_suppresses_but_damages = bool(
        ((simple["scanner_matches_neural"] | simple["suppresses_vs_original"]) & ~simple["tissue_preserved_vs_neural"]).any()
    )

    if any_outperforms:
        classification = "method requires rework; paper should be reframed as benchmark/evaluation framework."
    elif linear_matches:
        classification = "core value may be paired-acquisition scanner-subspace correction rather than neural factorization."
    elif pca_matches:
        classification = "scanner signal may be dominant-variance aligned; neural factorization claim weakened."
    elif any_suppresses_but_damages:
        classification = "scanner suppression alone is insufficient; tissue-preserving factorization remains valuable."
    else:
        classification = "neural factorization adds value beyond simple scanner projection/PCA."

    return {
        "dataset": dataset,
        "classification": classification,
        "best_simple_baseline": str(best["baseline"]),
        "best_simple_scanner_probe_accuracy": float(best["scanner_probe_accuracy_mean"]),
        "best_simple_mean_paired_cosine": float(best["mean_paired_cosine_mean"]),
        "best_simple_mean_top1_retrieval": float(best["mean_top1_retrieval_mean"]),
        "linear_matches_neural": linear_matches,
        "pca_matches_neural": pca_matches,
        "simple_outperforms_neural": any_outperforms,
    }


def markdown_table(frame: pd.DataFrame, columns: list[str], rename: dict[str, str] | None = None) -> str:
    if frame.empty:
        return "_No rows._"
    table = frame.loc[:, columns].copy()
    if rename:
        table = table.rename(columns=rename)
    for column in table.columns:
        if pd.api.types.is_numeric_dtype(table[column]):
            table[column] = table[column].map(lambda value: f"{float(value):.6f}")
    return table.to_markdown(index=False)


def build_report(
    *,
    args: argparse.Namespace,
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    contrasts: pd.DataFrame,
    classifications: list[dict[str, object]],
    skipped: list[str],
    manifest_notes: list[str],
) -> str:
    lines = [
        "# Pair-Integrity Baseline Murder Test",
        "",
        "## Question",
        "",
        "Can linear scanner-subspace projection or PCA component removal achieve the same scanner "
        "suppression as Paired-Acquisition Neural Factorization without damaging tissue preservation?",
        "",
        "## Datasets",
        "",
    ]
    for dataset in args.datasets:
        config = DATASETS[dataset]
        lines.append(f"- {config.label}: folds {', '.join(map(str, args.folds))}.")
    lines.extend(
        [
            "",
            "## Baselines run",
            "",
            "- `original_frozen_features`: frozen DINOv2 embeddings evaluated directly.",
            "- `paired_consistency_reference`: existing locked paired-consistency projected features.",
            "- `linear_scanner_subspace_projection_k*`: top-k logistic scanner-discriminative directions removed after fold-fit standardization.",
            "- `pca_component_removal_k*`: top-k PCA directions removed after fold-fit standardization.",
            "- `paired_acquisition_neural_factorization_reference`: existing locked Paired-Acquisition Neural Factorization dep20 projected features used as the method reference.",
            "",
            "## Baselines skipped",
            "",
        ]
    )
    lines.extend([f"- {reason}" for reason in skipped] or ["- None."])
    lines.extend(["", "## Result summary", ""])

    display_columns = [
        "dataset",
        "baseline",
        "n_runs",
        "scanner_probe_accuracy_mean",
        "mean_paired_cosine_mean",
        "worst_paired_cosine_mean",
        "mean_top1_retrieval_mean",
        "worst_top1_retrieval_mean",
        "effective_rank_mean",
        "runtime_seconds_mean",
    ]
    lines.append(
        markdown_table(
            summary.sort_values(["dataset", "reference_type", "baseline"]),
            display_columns,
            {
                "scanner_probe_accuracy_mean": "scanner_probe",
                "mean_paired_cosine_mean": "mean_cosine",
                "worst_paired_cosine_mean": "worst_cosine",
                "mean_top1_retrieval_mean": "mean_top1",
                "worst_top1_retrieval_mean": "worst_top1",
                "effective_rank_mean": "effective_rank",
                "runtime_seconds_mean": "runtime_s",
            },
        )
    )
    lines.extend(["", "## Best simple baseline and decision", ""])
    for item in classifications:
        lines.append(f"### {DATASETS[item['dataset']].label}")
        lines.append("")
        lines.append(f"- Best simple baseline: `{item['best_simple_baseline']}`.")
        lines.append(f"- Classification: {item['classification']}")
        lines.append(
            "- Best simple metrics: "
            f"scanner_probe={item['best_simple_scanner_probe_accuracy']:.6f}, "
            f"mean_paired_cosine={item['best_simple_mean_paired_cosine']:.6f}, "
            f"mean_top1_retrieval={item['best_simple_mean_top1_retrieval']:.6f}."
        )
        lines.append("")
    lines.extend(
        [
            "## Failure cases",
            "",
            "- k values above the learned scanner-subspace rank collapse to the maximum available scanner rank; this is reported as `effective_k`.",
            "- If a simple baseline lowers scanner probe but lowers paired cosine or retrieval relative to the neural reference, it is counted as scanner suppression with tissue damage.",
            "",
            "## Validation",
            "",
            f"- Raw metric rows: {len(raw)}.",
            "- Required metric columns are present with no missing or nonfinite values.",
            "- Duplicate dataset/fold/seed/baseline rows were rejected during validation.",
            "- Expected folds and reference seeds were validated where applicable.",
            "",
            "## Reproduction",
            "",
            "```powershell",
            "python experiments/baselines/run_pair_integrity_baseline_murder_test.py "
            f"--out-dir {args.out_dir.as_posix()} "
            f"--datasets {' '.join(args.datasets)} "
            f"--folds {' '.join(map(str, args.folds))}",
            "```",
            "",
            "## Output files",
            "",
            f"- `{(args.out_dir / 'raw_baseline_metrics.csv').as_posix()}`",
            f"- `{(args.out_dir / 'baseline_summary.csv').as_posix()}`",
            f"- `{(args.out_dir / 'baseline_contrasts.csv').as_posix()}`",
            f"- `{(args.out_dir / 'baseline_murder_test_report.md').as_posix()}`",
            f"- `{(args.out_dir / 'run_log.txt').as_posix()}`",
            "",
            "## Manifests used",
            "",
        ]
    )
    lines.extend([f"- {note}" for note in manifest_notes])
    lines.extend(
        [
            "",
            "## Claim boundary",
            "",
            "This is a peer-review-hardening baseline test. It does not claim clinical validation, "
            "diagnostic performance, disease biology discovery, complete scanner invariance, or deployment readiness.",
            "",
        ]
    )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/paired_acquisition_factorization_baseline_murder_test"),
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=sorted(DATASETS),
        default=["scorpion"],
    )
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    return parser.parse_args()


def run(args: argparse.Namespace) -> None:
    args.out_dir.mkdir(parents=True, exist_ok=True)
    folds = tuple(int(fold) for fold in args.folds)
    datasets = tuple(args.datasets)
    all_rows: list[dict[str, object]] = []
    skipped: list[str] = []
    manifest_notes: list[str] = []
    start = time.perf_counter()

    design = {
        "stage": "paired_acquisition_factorization_baseline_murder_test",
        "datasets": list(datasets),
        "folds": list(folds),
        "k_values": list(K_VALUES),
        "required_metrics": list(REQUIRED_METRICS),
        "simple_baselines": sorted(SIMPLE_FAMILIES),
        "reference_baselines": [PAIR_REFERENCE, NEURAL_REFERENCE],
        "command": " ".join(sys.argv),
    }
    atomic_text(args.out_dir / "experiment_design.json", json.dumps(design, indent=2, sort_keys=True) + "\n")

    for dataset in datasets:
        rows, dataset_skipped, dataset_manifest_notes = run_dataset(DATASETS[dataset], folds)
        all_rows.extend(rows)
        skipped.extend(dataset_skipped)
        manifest_notes.extend(dataset_manifest_notes)

    raw = pd.DataFrame(all_rows).sort_values(["dataset", "reference_type", "baseline", "fold", "seed"])
    validate_outputs(raw, datasets=datasets, folds=folds)
    summary = summarize(raw)
    contrasts = build_contrasts(summary)
    classifications = [classify_dataset(summary, dataset) for dataset in datasets]

    atomic_csv(args.out_dir / "raw_baseline_metrics.csv", raw)
    atomic_csv(args.out_dir / "baseline_summary.csv", summary)
    atomic_csv(args.out_dir / "baseline_contrasts.csv", contrasts)
    report = build_report(
        args=args,
        raw=raw,
        summary=summary,
        contrasts=contrasts,
        classifications=classifications,
        skipped=skipped,
        manifest_notes=manifest_notes,
    )
    atomic_text(args.out_dir / "baseline_murder_test_report.md", report)

    print("\nBASELINE MURDER TEST COMPLETE")
    print(f"Datasets: {', '.join(datasets)}")
    print(f"Rows: {len(raw)}")
    print(f"Runtime seconds: {time.perf_counter() - start:.2f}")
    for item in classifications:
        print(f"{item['dataset']}: {item['classification']}")
    print(f"Report: {(args.out_dir / 'baseline_murder_test_report.md').resolve()}")


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "run_log.txt"
    with log_path.open("a", encoding="utf-8") as log_file:
        with redirect_stdout(Tee(sys.stdout, log_file)), redirect_stderr(Tee(sys.stderr, log_file)):
            print("\n" + "=" * 80)
            print(time.strftime("%Y-%m-%d %H:%M:%S"))
            print("COMMAND " + " ".join(sys.argv))
            try:
                run(args)
            except Exception:
                traceback.print_exc()
                raise


if __name__ == "__main__":
    main()
