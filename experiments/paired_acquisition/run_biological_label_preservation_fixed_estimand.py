#!/usr/bin/env python3
"""Fixed-estimand canine biological-label preservation audit.

This is the current category-level audit protocol. It evaluates one fixed set of
tissue categories across all five biological-sample-blocked folds. Categories
without at least the requested number of fit and test biological samples in every
fold are excluded before any representation is evaluated.

Scientific corrections relative to the historical audit:

- one fixed category estimand across folds;
- fit-only standardization for every probe;
- category neighbours drawn from the fit pool only;
- same-region and same-sample neighbours excluded;
- strongest historical oldstyle centroid/QR keep and removed branches included;
- no patch-bootstrap confidence intervals or slide-independent p-values;
- seeds are averaged within fold before the five fold summaries are reported.

The output is descriptive five-fold evidence. It is not clinical evidence and
does not establish information-theoretic independence.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.paired_acquisition import run_biological_label_preservation_audit as legacy
from experiments.paired_acquisition import run_biological_label_preservation_audit_v2 as v2


class AuditError(RuntimeError):
    """Raised when the fixed scientific estimand cannot be evaluated."""


def category_sample_support(frame: pd.DataFrame, split: str) -> dict[str, int]:
    subset = frame.loc[frame["split"] == split, ["category_name", "sample_id"]].drop_duplicates()
    return {
        str(category): int(group["sample_id"].nunique())
        for category, group in subset.groupby("category_name")
    }


def derive_fixed_categories(
    folds: list[int],
    *,
    minimum_fit_samples: int,
    minimum_test_samples: int,
) -> tuple[list[str], pd.DataFrame]:
    rows: list[dict[str, object]] = []
    all_categories: set[str] = set()
    manifests: dict[int, pd.DataFrame] = {}
    for fold in folds:
        manifest = legacy.load_manifest(fold)
        manifests[fold] = manifest
        all_categories.update(manifest["category_name"].astype(str).unique())

    retained: list[str] = []
    for category in sorted(all_categories):
        eligible = True
        for fold in folds:
            manifest = manifests[fold]
            fit_support = category_sample_support(
                manifest.assign(split=np.where(manifest["split"].eq("test"), "test", "fit")),
                "fit",
            ).get(category, 0)
            test_support = category_sample_support(manifest, "test").get(category, 0)
            rows.append(
                {
                    "fold": fold,
                    "category": category,
                    "fit_samples": fit_support,
                    "test_samples": test_support,
                    "minimum_fit_samples": minimum_fit_samples,
                    "minimum_test_samples": minimum_test_samples,
                }
            )
            eligible &= fit_support >= minimum_fit_samples and test_support >= minimum_test_samples
        if eligible:
            retained.append(category)

    support = pd.DataFrame(rows)
    if len(retained) < 2:
        raise AuditError("fewer than two categories satisfy the fixed sample-support estimand")
    support["retained_in_fixed_estimand"] = support["category"].isin(retained)
    return retained, support


def fixed_balanced_accuracy(
    truth: np.ndarray,
    prediction: np.ndarray,
    categories: list[str],
) -> float:
    recalls = []
    truth = truth.astype(str)
    prediction = prediction.astype(str)
    for category in categories:
        mask = truth == category
        if not np.any(mask):
            raise AuditError(f"fixed category absent from test truth: {category}")
        recalls.append(float(np.mean(prediction[mask] == category)))
    return float(np.mean(recalls))


def fit_probe(
    features: np.ndarray,
    labels: np.ndarray,
    fit: np.ndarray,
    test: np.ndarray,
) -> np.ndarray:
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
    return model.predict(features[test])


def effective_rank(features: np.ndarray) -> float:
    centered = features - features.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False, full_matrices=False)
    energy = singular**2
    if float(energy.sum()) <= 0:
        return 0.0
    probabilities = energy / energy.sum()
    probabilities = probabilities[probabilities > 0]
    return float(math.exp(-np.sum(probabilities * np.log(probabilities))))


def merge_base_manifest(manifest: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
    features, frame = legacy.load_base_features()
    metadata = manifest[["region_id", "scanner_id", "split", "sample_id", "category_name"]].copy()
    metadata["region_id"] = metadata["region_id"].astype(str)
    metadata["scanner_id"] = metadata["scanner_id"].astype(str)
    frame["region_id"] = frame["region_id"].astype(str)
    frame["scanner_id"] = frame["scanner_id"].astype(str)
    frame = frame.drop(columns=["split"], errors="ignore").merge(
        metadata,
        on=["region_id", "scanner_id"],
        how="left",
        validate="one_to_one",
    )
    if len(frame) != len(features) or frame.isna().any().any():
        raise AuditError("base-feature metadata merge failed")
    return features, frame


def oldstyle_representation(
    fold: int,
    k: int,
    branch: str,
    manifest: pd.DataFrame,
) -> tuple[np.ndarray, pd.DataFrame]:
    del fold
    features, frame = merge_base_manifest(manifest)
    fit, _ = v2.split_indices(frame)
    scanner_labels = frame["scanner_id"].astype(str).to_numpy()
    fit_matrix = np.asarray(features[fit], dtype=np.float64)
    grand_mean = fit_matrix.mean(axis=0)
    directions = []
    for scanner in sorted(set(scanner_labels[fit])):
        directions.append(fit_matrix[scanner_labels[fit] == scanner].mean(axis=0) - grand_mean)
    direction_matrix = np.stack(directions)
    effective_k = min(k, direction_matrix.shape[0])
    basis, _ = np.linalg.qr(direction_matrix[:effective_k].T)
    removed = (np.asarray(features, dtype=np.float64) @ basis) @ basis.T
    keep = np.asarray(features, dtype=np.float64) - removed
    selected = keep if branch == "keep" else removed
    if not np.isfinite(selected).all():
        raise AuditError("oldstyle projection produced non-finite features")
    return selected.astype(np.float32), frame


def representation_plan(
    fold: int,
    manifest: pd.DataFrame,
    seeds: list[int],
) -> list[tuple[str, int, Callable[[], tuple[np.ndarray, pd.DataFrame]], dict[str, object]]]:
    plan = v2.loader_plan(fold, manifest, seeds)
    for k in range(1, 5):
        for branch in ("keep", "removed"):
            plan.append(
                (
                    f"oldstyle_{branch}_k{k}",
                    0,
                    lambda k=k, branch=branch: oldstyle_representation(
                        fold,
                        k,
                        branch,
                        manifest,
                    ),
                    {
                        "family": "oldstyle_centroid_qr",
                        "k": k,
                        "branch": branch,
                    },
                )
            )
    return plan


def evaluate_representation(
    features: np.ndarray,
    frame: pd.DataFrame,
    *,
    fixed_categories: list[str],
    representation: str,
    fold: int,
    seed: int,
    details: dict[str, object],
) -> dict[str, object]:
    if len(features) != len(frame) or not np.isfinite(features).all():
        raise AuditError("invalid or misaligned representation")
    required = ["sample_id", "region_id", "scanner_id", "category_name", "split"]
    if frame[required].isna().any().any():
        raise AuditError("missing representation metadata")

    all_fit, all_test = v2.split_indices(frame)
    scanner_truth = frame.iloc[all_test]["scanner_id"].astype(str).to_numpy()
    scanner_prediction = fit_probe(
        features,
        frame["scanner_id"].astype(str).to_numpy(),
        all_fit,
        all_test,
    )
    scanner_categories = sorted(set(frame["scanner_id"].astype(str)))
    scanner_accuracy = fixed_balanced_accuracy(
        scanner_truth,
        scanner_prediction,
        scanner_categories,
    )

    category_mask = frame["category_name"].astype(str).isin(fixed_categories).to_numpy()
    category_features = features[category_mask]
    category_frame = frame.loc[category_mask].reset_index(drop=True)
    category_fit, category_test = v2.split_indices(category_frame)
    observed_test_categories = set(category_frame.iloc[category_test]["category_name"].astype(str))
    if observed_test_categories != set(fixed_categories):
        raise AuditError(
            f"fold {fold} does not contain the fixed category estimand: "
            f"observed={sorted(observed_test_categories)}"
        )

    category_truth = category_frame.iloc[category_test]["category_name"].astype(str).to_numpy()
    category_prediction = fit_probe(
        category_features,
        category_frame["category_name"].astype(str).to_numpy(),
        category_fit,
        category_test,
    )
    category_accuracy = fixed_balanced_accuracy(
        category_truth,
        category_prediction,
        fixed_categories,
    )
    category_f1 = float(
        f1_score(
            category_truth,
            category_prediction,
            labels=fixed_categories,
            average="macro",
            zero_division=0,
        )
    )

    purity = v2.category_purity_fit_pool(
        category_features,
        category_frame,
        category_fit,
        category_test,
    )
    return {
        "representation": representation,
        "fold": fold,
        "seed": seed,
        "fixed_categories": json.dumps(fixed_categories),
        "n_fixed_categories": len(fixed_categories),
        "n_test_samples": int(category_frame.iloc[category_test]["sample_id"].nunique()),
        "scanner_probe_balanced_accuracy": scanner_accuracy,
        "category_probe_balanced_accuracy": category_accuracy,
        "category_probe_macro_f1": category_f1,
        "effective_rank_all_test": effective_rank(features[all_test]),
        **purity,
        **details,
    }


def summarize(raw: pd.DataFrame, metric_columns: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    seed_averaged = raw.groupby(
        ["representation", "fold"],
        as_index=False,
    )[metric_columns].mean()
    rows = []
    for representation, group in seed_averaged.groupby("representation"):
        row: dict[str, object] = {
            "representation": representation,
            "n_folds": int(group["fold"].nunique()),
        }
        for metric in metric_columns:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_fold_min"] = float(group[metric].min())
            row[f"{metric}_fold_max"] = float(group[metric].max())
        rows.append(row)
    return seed_averaged, pd.DataFrame(rows).sort_values("representation")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path(
            "results/paired_acquisition_factorization_"
            "biological_label_preservation_fixed_estimand"
        ),
    )
    parser.add_argument("--minimum-fit-samples", type=int, default=2)
    parser.add_argument("--minimum-test-samples", type=int, default=2)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if args.minimum_fit_samples < 1 or args.minimum_test_samples < 1:
        raise SystemExit("minimum sample counts must be positive")

    folds = legacy.FOLDS[:1] if args.smoke else legacy.FOLDS
    seeds = legacy.SEEDS[:1] if args.smoke else legacy.SEEDS
    fixed_categories, support = derive_fixed_categories(
        folds,
        minimum_fit_samples=args.minimum_fit_samples,
        minimum_test_samples=args.minimum_test_samples,
    )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    started = time.time()
    for fold in folds:
        manifest = legacy.load_manifest(fold)
        for representation, seed, loader, details in representation_plan(
            fold,
            manifest,
            seeds,
        ):
            features, frame = loader()
            rows.append(
                evaluate_representation(
                    features,
                    frame,
                    fixed_categories=fixed_categories,
                    representation=representation,
                    fold=fold,
                    seed=seed,
                    details=details,
                )
            )

    raw = pd.DataFrame(rows).sort_values(["representation", "fold", "seed"])
    metrics = [
        "scanner_probe_balanced_accuracy",
        "category_probe_balanced_accuracy",
        "category_probe_macro_f1",
        "purity_fit_pool_k1",
        "purity_fit_pool_k5",
        "purity_fit_pool_k10",
        "effective_rank_all_test",
    ]
    if raw.empty or raw[metrics].isna().any().any():
        raise AuditError("refusing to publish incomplete fixed-estimand metrics")

    seed_averaged, summary = summarize(raw, metrics)
    raw.to_csv(args.out_dir / "raw_metrics.csv", index=False)
    seed_averaged.to_csv(args.out_dir / "fold_seed_averaged_metrics.csv", index=False)
    summary.to_csv(args.out_dir / "five_fold_descriptive_summary.csv", index=False)
    support.to_csv(args.out_dir / "fixed_category_support.csv", index=False)

    design = {
        "status": "completed",
        "protocol": "biological_label_preservation_fixed_estimand_v1",
        "fixed_categories": fixed_categories,
        "excluded_categories": sorted(set(support["category"].astype(str)) - set(fixed_categories)),
        "minimum_fit_samples_per_category_per_fold": args.minimum_fit_samples,
        "minimum_test_samples_per_category_per_fold": args.minimum_test_samples,
        "fit_only_probe_standardization": True,
        "category_neighbour_pool": "fit_only",
        "same_region_neighbours_excluded": True,
        "same_sample_neighbours_excluded": True,
        "oldstyle_centroid_qr_included": True,
        "uncertainty": "five fold mean and range after seed averaging; no p-values",
        "folds": folds,
        "seeds": seeds,
        "historical_category_metrics_promoted": False,
        "runtime_seconds": time.time() - started,
        "claim_boundary": (
            "Tissue-category representation audit only; not diagnosis, clinical "
            "utility, complete factorization, or patient-level evidence."
        ),
    }
    (args.out_dir / "experiment_design.json").write_text(
        json.dumps(design, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"FIXED-ESTIMAND BIOLOGICAL LABEL AUDIT PASSED: {args.out_dir}")
    print(f"Fixed categories: {fixed_categories}")


if __name__ == "__main__":
    try:
        main()
    except (AuditError, v2.AuditError, OSError, ValueError) as exc:
        print(f"FIXED-ESTIMAND BIOLOGICAL LABEL AUDIT FAILED: {exc}")
        raise SystemExit(1) from exc
