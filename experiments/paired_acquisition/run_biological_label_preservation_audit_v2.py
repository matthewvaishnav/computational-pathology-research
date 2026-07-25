#!/usr/bin/env python3
"""Leakage-safe biological-label preservation audit (v2).

This runner supersedes ``run_biological_label_preservation_audit.py`` for new
claim evidence. It corrects four scientific problems in the historical audit:

1. every representation is standardized using fit rows only before probing;
2. the linear scanner-subspace baseline fits its scaler on fit rows only;
3. category nearest-neighbour purity searches only the fit pool and excludes
   candidates from the same region or biological sample;
4. category support is validated fail-closed for every fold.

Uncertainty is reported with a biological-sample cluster bootstrap. No metric
from the historical audit is automatically promoted into this v2 release.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from experiments.paired_acquisition import run_biological_label_preservation_audit as legacy


class AuditError(RuntimeError):
    """Raised when a scientific precondition is not satisfied."""


def split_indices(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    fit = np.flatnonzero(frame["split"].to_numpy() != "test")
    test = np.flatnonzero(frame["split"].to_numpy() == "test")
    if len(fit) == 0 or len(test) == 0:
        raise AuditError("empty fit or test split")

    fit_samples = set(frame.iloc[fit]["sample_id"].astype(str))
    test_samples = set(frame.iloc[test]["sample_id"].astype(str))
    overlap = sorted(fit_samples & test_samples)
    if overlap:
        raise AuditError(f"biological-sample leakage: {overlap[:10]}")
    return fit, test


def validate_category_support(
    frame: pd.DataFrame,
    fit: np.ndarray,
    test: np.ndarray,
) -> dict[str, dict[str, int]]:
    summary: dict[str, dict[str, int]] = {}
    fit_labels = set(frame.iloc[fit]["category_name"].astype(str))
    test_labels = set(frame.iloc[test]["category_name"].astype(str))
    missing = sorted(test_labels - fit_labels)
    if missing:
        raise AuditError(f"test categories absent from fit split: {missing}")

    for label in sorted(fit_labels | test_labels):
        fit_rows = frame.iloc[fit]
        test_rows = frame.iloc[test]
        summary[label] = {
            "fit_rows": int((fit_rows["category_name"].astype(str) == label).sum()),
            "test_rows": int((test_rows["category_name"].astype(str) == label).sum()),
            "fit_samples": int(
                fit_rows.loc[
                    fit_rows["category_name"].astype(str) == label,
                    "sample_id",
                ].nunique()
            ),
            "test_samples": int(
                test_rows.loc[
                    test_rows["category_name"].astype(str) == label,
                    "sample_id",
                ].nunique()
            ),
        }
        if summary[label]["fit_rows"] == 0 or summary[label]["test_rows"] == 0:
            raise AuditError(f"category {label!r} has empty fit or test support")
    return summary


def fit_probe(
    features: np.ndarray,
    labels: np.ndarray,
    fit: np.ndarray,
    test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
            random_state=0,
        ),
    )
    model.fit(features[fit], labels[fit])
    return labels[test], model.predict(features[test])


def cluster_bootstrap_ci(
    truth: np.ndarray,
    prediction: np.ndarray,
    sample_ids: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    seed: int,
    draws: int,
) -> tuple[float, float]:
    unique_samples = np.asarray(sorted(set(sample_ids.astype(str))))
    if len(unique_samples) < 2:
        raise AuditError("cluster bootstrap requires at least two test samples")

    row_groups = {
        sample: np.flatnonzero(sample_ids.astype(str) == sample)
        for sample in unique_samples
    }
    rng = np.random.default_rng(seed)
    values = np.empty(draws, dtype=float)
    for draw in range(draws):
        selected = rng.choice(unique_samples, size=len(unique_samples), replace=True)
        rows = np.concatenate([row_groups[str(sample)] for sample in selected])
        values[draw] = metric(truth[rows], prediction[rows])
    return float(np.quantile(values, 0.025)), float(np.quantile(values, 0.975))


def normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise AuditError("zero-norm feature encountered")
    return features / norms


def category_purity_fit_pool(
    features: np.ndarray,
    frame: pd.DataFrame,
    fit: np.ndarray,
    test: np.ndarray,
    ks: tuple[int, ...] = (1, 5, 10),
) -> dict[str, float]:
    normalized = normalize(features)
    fit_features = normalized[fit]
    test_features = normalized[test]
    similarity = test_features @ fit_features.T

    fit_regions = frame.iloc[fit]["region_id"].astype(str).to_numpy()
    test_regions = frame.iloc[test]["region_id"].astype(str).to_numpy()
    fit_samples = frame.iloc[fit]["sample_id"].astype(str).to_numpy()
    test_samples = frame.iloc[test]["sample_id"].astype(str).to_numpy()
    fit_labels = frame.iloc[fit]["category_name"].astype(str).to_numpy()
    test_labels = frame.iloc[test]["category_name"].astype(str).to_numpy()

    for row in range(len(test)):
        forbidden = (fit_regions == test_regions[row]) | (fit_samples == test_samples[row])
        similarity[row, forbidden] = -np.inf

    valid_counts = np.isfinite(similarity).sum(axis=1)
    if np.any(valid_counts < max(ks)):
        raise AuditError("insufficient leakage-safe neighbours for requested k")

    result: dict[str, float] = {}
    for k in ks:
        neighbours = np.argpartition(-similarity, kth=k - 1, axis=1)[:, :k]
        result[f"purity_fit_pool_k{k}"] = float(
            np.mean(
                [
                    np.mean(fit_labels[neighbours[row]] == test_labels[row])
                    for row in range(len(test))
                ]
            )
        )
    return result


def linear_removal_fit_only(
    fold: int,
    k: int,
    manifest: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    features, frame = legacy.load_base_features()
    columns = manifest[
        ["region_id", "scanner_id", "split", "sample_id", "category_name"]
    ].copy()
    columns["region_id"] = columns["region_id"].astype(str)
    columns["scanner_id"] = columns["scanner_id"].astype(str)
    frame["region_id"] = frame["region_id"].astype(str)
    frame["scanner_id"] = frame["scanner_id"].astype(str)
    frame = frame.drop(columns=["split"], errors="ignore")
    frame = frame.merge(
        columns,
        on=["region_id", "scanner_id"],
        how="left",
        validate="one_to_one",
    )
    fit, _ = split_indices(frame)
    scaler = StandardScaler().fit(features[fit])
    transformed = scaler.transform(features)
    directions = legacy._fit_scanner_directions(transformed, frame, fit)
    cleaned, effective_k = legacy._remove_directions(transformed, directions, k)
    frame = frame.copy()
    frame["effective_k"] = effective_k
    return cleaned.astype(np.float32), frame


def evaluate(
    features: np.ndarray,
    frame: pd.DataFrame,
    *,
    representation: str,
    fold: int,
    seed: int,
    details: dict[str, object],
    bootstrap_draws: int,
) -> tuple[dict[str, object], dict[str, dict[str, int]]]:
    if len(features) != len(frame) or not np.isfinite(features).all():
        raise AuditError("invalid or misaligned representation")
    if frame[["sample_id", "region_id", "category_name", "scanner_id", "split"]].isna().any().any():
        raise AuditError("missing audit metadata")

    fit, test = split_indices(frame)
    support = validate_category_support(frame, fit, test)
    test_samples = frame.iloc[test]["sample_id"].astype(str).to_numpy()

    scanner_truth, scanner_pred = fit_probe(
        features,
        frame["scanner_id"].astype(str).to_numpy(),
        fit,
        test,
    )
    category_truth, category_pred = fit_probe(
        features,
        frame["category_name"].astype(str).to_numpy(),
        fit,
        test,
    )

    scanner_acc = float(balanced_accuracy_score(scanner_truth, scanner_pred))
    category_acc = float(balanced_accuracy_score(category_truth, category_pred))
    category_f1 = float(f1_score(category_truth, category_pred, average="macro"))

    scanner_ci = cluster_bootstrap_ci(
        scanner_truth,
        scanner_pred,
        test_samples,
        balanced_accuracy_score,
        seed=10000 + fold * 100 + seed,
        draws=bootstrap_draws,
    )
    category_ci = cluster_bootstrap_ci(
        category_truth,
        category_pred,
        test_samples,
        balanced_accuracy_score,
        seed=20000 + fold * 100 + seed,
        draws=bootstrap_draws,
    )

    centered = features[test] - features[test].mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False, full_matrices=False)
    energy = singular**2
    probabilities = energy / energy.sum() if float(energy.sum()) > 0 else np.asarray([])
    effective_rank = (
        float(math.exp(-np.sum(probabilities[probabilities > 0] * np.log(probabilities[probabilities > 0]))))
        if len(probabilities)
        else 0.0
    )

    row: dict[str, object] = {
        "representation": representation,
        "fold": fold,
        "seed": seed,
        "n_fit_rows": int(len(fit)),
        "n_test_rows": int(len(test)),
        "n_test_samples": int(len(set(test_samples))),
        "feature_dim": int(features.shape[1]),
        "scanner_probe_balanced_accuracy": scanner_acc,
        "scanner_probe_cluster_ci_025": scanner_ci[0],
        "scanner_probe_cluster_ci_975": scanner_ci[1],
        "category_probe_balanced_accuracy": category_acc,
        "category_probe_cluster_ci_025": category_ci[0],
        "category_probe_cluster_ci_975": category_ci[1],
        "category_probe_macro_f1": category_f1,
        "effective_rank_test": effective_rank,
        **category_purity_fit_pool(features, frame, fit, test),
        **details,
    }
    return row, support


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/paired_acquisition_factorization_biological_label_preservation_audit_v2"),
    )
    parser.add_argument("--bootstrap-draws", type=int, default=2000)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.bootstrap_draws < 100:
        raise SystemExit("bootstrap-draws must be at least 100")

    folds = legacy.FOLDS[:1] if args.smoke else legacy.FOLDS
    seeds = legacy.SEEDS[:1] if args.smoke else legacy.SEEDS
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, object]] = []
    support_records: list[dict[str, object]] = []
    started = time.time()

    for fold in folds:
        manifest = legacy.load_manifest(fold)
        loaders: list[tuple[str, int, Callable[[], tuple[np.ndarray, pd.DataFrame]], dict[str, object]]] = []

        loaders.append(
            (
                "original_frozen_features",
                0,
                lambda fold=fold, manifest=manifest: legacy.load_representation_frozen(fold, 0, manifest),
                {"family": "original_frozen"},
            )
        )
        for condition, label in [
            ("true_pairs", "true_pair"),
            ("shuffled_sample_pairs", "shuffled_sample"),
        ]:
            for branch in ("biological", "acquisition"):
                for seed in seeds:
                    loaders.append(
                        (
                            f"{label}_{branch}",
                            seed,
                            lambda fold=fold, seed=seed, condition=condition, branch=branch, manifest=manifest: legacy.load_representation_pair_integrity(
                                fold,
                                seed,
                                condition,
                                branch,
                                manifest,
                            ),
                            {"family": "pair_integrity", "condition": condition, "branch": branch},
                        )
                    )
        for k in legacy.PCA_K_VALUES:
            loaders.append(
                (
                    f"pca_removal_k{k}",
                    0,
                    lambda fold=fold, k=k, manifest=manifest: legacy.load_representation_pca_removal(fold, k, manifest),
                    {"family": "pca_removal", "k": k},
                )
            )
        for k in legacy.LINEAR_K_VALUES:
            loaders.append(
                (
                    f"linear_projection_k{k}",
                    0,
                    lambda fold=fold, k=k, manifest=manifest: linear_removal_fit_only(fold, k, manifest),
                    {"family": "linear_scanner_subspace", "k": k},
                )
            )

        for representation, seed, loader, details in loaders:
            features, frame = loader()
            row, support = evaluate(
                features,
                frame,
                representation=representation,
                fold=fold,
                seed=seed,
                details=details,
                bootstrap_draws=args.bootstrap_draws,
            )
            rows.append(row)
            for category, counts in support.items():
                support_records.append(
                    {
                        "fold": fold,
                        "representation": representation,
                        "seed": seed,
                        "category": category,
                        **counts,
                    }
                )

    raw = pd.DataFrame(rows).sort_values(["representation", "fold", "seed"])
    if raw.empty or raw.select_dtypes(include=[np.number]).isna().any().any():
        raise AuditError("refusing to publish incomplete v2 metrics")
    raw.to_csv(args.out_dir / "raw_metrics.csv", index=False)
    pd.DataFrame(support_records).to_csv(
        args.out_dir / "category_support_by_fold.csv",
        index=False,
    )

    numeric = [
        column
        for column in raw.select_dtypes(include=[np.number]).columns
        if column not in {"fold", "seed"}
    ]
    raw.groupby("representation", as_index=False)[numeric].mean().to_csv(
        args.out_dir / "descriptive_summary.csv",
        index=False,
    )

    design = {
        "status": "completed",
        "protocol_version": 2,
        "supersedes": "experiments/paired_acquisition/run_biological_label_preservation_audit.py",
        "historical_results_promoted": False,
        "fit_only_probe_standardization": True,
        "fit_only_linear_baseline_standardization": True,
        "neighbour_reference_pool": "fit_only",
        "same_region_candidates_excluded": True,
        "same_sample_candidates_excluded": True,
        "category_support_fail_closed": True,
        "uncertainty_unit": "biological_sample_cluster_bootstrap",
        "bootstrap_draws": args.bootstrap_draws,
        "folds": folds,
        "seeds": seeds,
        "runtime_seconds": time.time() - started,
    }
    (args.out_dir / "experiment_design.json").write_text(
        json.dumps(design, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"BIOLOGICAL LABEL PRESERVATION AUDIT V2 PASSED: {args.out_dir}")


if __name__ == "__main__":
    try:
        main()
    except (AuditError, OSError, ValueError) as exc:
        print(f"BIOLOGICAL LABEL PRESERVATION AUDIT V2 FAILED: {exc}")
        raise SystemExit(1) from exc
