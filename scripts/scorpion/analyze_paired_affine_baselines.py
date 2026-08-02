#!/usr/bin/env python3
"""Analyze the prospective SCORPION paired affine comparison.

The five reference-scanner conditions are averaged within each original slide
before inference. They are sensitivity conditions and never contribute five
independent observations. Neural factorization rows are imported from the
completed capacity-matched analysis and remain seed-averaged within fold/slide.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

SCANNERS = ("AT2", "B300", "DP200", "GT450", "P1000")
FOLDS = tuple(range(5))
AFFINE_VARIANTS = (
    "identity_standardized",
    "centroid_translation",
    "orthogonal_procrustes",
    "affine_least_squares",
    "ridge_affine",
)
METRICS = (
    "scanner_probe_accuracy",
    "pair_cosine_average",
    "pair_cosine_worst",
    "retrieval_top1_average",
    "retrieval_top1_worst",
)
LOWER_IS_FAVORABLE = {"scanner_probe_accuracy"}
SCHEMA_VERSION = "scorpion-paired-affine-analysis/v1"


class AnalysisError(ValueError):
    pass


def two_stage_cluster_bootstrap(
    frame: pd.DataFrame,
    metric: str,
    *,
    seed: int,
    draws: int,
) -> np.ndarray:
    """Resample five folds, then slides within each sampled fold."""
    folds = sorted(frame["fold"].unique())
    if folds != list(FOLDS):
        raise AnalysisError("Fold-aware bootstrap requires folds 0 through 4.")
    groups = [
        frame.loc[frame["fold"] == fold, metric].to_numpy(float) for fold in folds
    ]
    if any(not len(group) or not np.isfinite(group).all() for group in groups):
        raise AnalysisError(f"Invalid bootstrap input for {metric}.")
    rng = np.random.default_rng(seed)
    sampled_folds = rng.integers(0, len(folds), size=(draws, len(folds)))
    totals = np.zeros(draws, dtype=np.float64)
    counts = np.zeros(draws, dtype=np.int64)
    for slot in range(len(folds)):
        selections = sampled_folds[:, slot]
        for fold_index, group in enumerate(groups):
            mask = selections == fold_index
            selected_count = int(mask.sum())
            if not selected_count:
                continue
            indices = rng.integers(0, len(group), size=(selected_count, len(group)))
            totals[mask] += group[indices].sum(axis=1)
            counts[mask] += len(group)
    if np.any(counts == 0):
        raise AnalysisError("Bootstrap produced an empty draw.")
    return totals / counts


def load_factorization_rows(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"fold", "variant", "slide_id", *METRICS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise AnalysisError(f"Factorization slide metrics are missing: {missing}")
    frame = frame.loc[
        frame["variant"] == "pathoalign_dep20", ["fold", "slide_id", *METRICS]
    ]
    if len(frame) != 48 or frame["slide_id"].nunique() != 48:
        raise AnalysisError("Expected 48 seed-averaged PathoAlign slide rows.")
    if sorted(frame["fold"].unique()) != list(FOLDS):
        raise AnalysisError("PathoAlign rows do not cover all five folds.")
    if not np.isfinite(frame[list(METRICS)].to_numpy(float)).all():
        raise AnalysisError("PathoAlign metrics contain non-finite values.")
    frame.insert(1, "variant", "pathoalign_dep20")
    frame.insert(3, "reference_scanner", "not_applicable")
    return frame


def load_affine_rows(experiment_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    from scripts.scorpion import analyze_pathoalign_crossfold as common

    run_rows: list[dict[str, Any]] = []
    slide_rows: list[pd.DataFrame] = []
    for fold in FOLDS:
        for reference_scanner in SCANNERS:
            for variant in AFFINE_VARIANTS:
                path = (
                    experiment_dir
                    / f"fold_{fold}"
                    / f"reference_{reference_scanner}"
                    / variant
                    / "projected_features.npz"
                )
                if not path.is_file():
                    raise AnalysisError(f"Missing projected baseline: {path}")
                features, acquisition, frame, metadata = common.load_projected(path)
                if acquisition is not None:
                    raise AnalysisError(
                        "Affine baseline unexpectedly contains an acquisition branch."
                    )
                expected_metadata = {
                    "fold": fold,
                    "reference_scanner": reference_scanner,
                    "variant": variant,
                    "transform_estimation_uses_test_rows": False,
                }
                for key, expected in expected_metadata.items():
                    if metadata.get(key) != expected:
                        raise AnalysisError(f"Metadata mismatch for {path}: {key}")
                fit, test = common.split_indices(frame)
                scanner_overall, scanner_slides = common.scanner_probe(
                    features, frame, fit, test
                )
                pair_overall, pair_slides = common.paired_slide_metrics(
                    features, frame, test
                )
                slides = scanner_slides.merge(
                    pair_slides,
                    on="slide_id",
                    how="inner",
                    validate="one_to_one",
                )
                if (
                    slides["slide_id"].nunique()
                    != frame.iloc[test]["slide_id"].nunique()
                ):
                    raise AnalysisError(f"Incomplete per-slide metrics for {path}")
                slides.insert(0, "fold", fold)
                slides.insert(1, "variant", variant)
                slides.insert(3, "reference_scanner", reference_scanner)
                slide_rows.append(slides)
                run_rows.append(
                    {
                        "fold": fold,
                        "variant": variant,
                        "reference_scanner": reference_scanner,
                        "scanner_probe_accuracy": scanner_overall["balanced_accuracy"],
                        **pair_overall,
                    }
                )

    runs = pd.DataFrame(run_rows).sort_values(["variant", "reference_scanner", "fold"])
    slides = pd.concat(slide_rows, ignore_index=True).sort_values(
        ["variant", "reference_scanner", "fold", "slide_id"]
    )
    expected_runs = len(FOLDS) * len(SCANNERS) * len(AFFINE_VARIANTS)
    if len(runs) != expected_runs:
        raise AnalysisError(
            f"Expected {expected_runs} affine run rows, observed {len(runs)}."
        )
    expected_slide_rows = len(SCANNERS) * len(AFFINE_VARIANTS) * 48
    if len(slides) != expected_slide_rows:
        raise AnalysisError(
            f"Expected {expected_slide_rows} affine slide rows, observed {len(slides)}."
        )
    return runs, slides


def average_references(slides: pd.DataFrame) -> pd.DataFrame:
    counts = slides.groupby(["variant", "slide_id"])["reference_scanner"].nunique()
    if not (counts == len(SCANNERS)).all():
        raise AnalysisError("Every affine variant/slide requires all five references.")
    folds = slides.groupby(["variant", "slide_id"])["fold"].nunique()
    if not (folds == 1).all():
        raise AnalysisError("A slide appeared in multiple folds.")
    averaged = (
        slides.groupby(["fold", "variant", "slide_id"], as_index=False)[list(METRICS)]
        .mean()
        .sort_values(["variant", "fold", "slide_id"])
    )
    averaged.insert(3, "reference_scanner", "average_of_five")
    if len(averaged) != len(AFFINE_VARIANTS) * 48:
        raise AnalysisError("Reference averaging returned an unexpected row count.")
    return averaged


def build_contrasts(methods: pd.DataFrame) -> pd.DataFrame:
    identity = methods.loc[methods["variant"] == "identity_standardized"].set_index(
        "slide_id"
    )
    if len(identity) != 48:
        raise AnalysisError("Identity baseline must contain exactly 48 slides.")
    comparison_pairs = [
        (variant, "identity_standardized")
        for variant in (*AFFINE_VARIANTS[1:], "pathoalign_dep20")
    ]
    comparison_pairs.append(("pathoalign_dep20", "ridge_affine"))
    rows: list[pd.DataFrame] = []
    for candidate_name, comparator_name in comparison_pairs:
        candidate = methods.loc[methods["variant"] == candidate_name].set_index(
            "slide_id"
        )
        comparator = methods.loc[methods["variant"] == comparator_name].set_index(
            "slide_id"
        )
        if set(candidate.index) != set(comparator.index) or len(candidate) != 48:
            raise AnalysisError(
                f"Unmatched slides for {candidate_name} minus {comparator_name}."
            )
        values = (
            candidate.loc[comparator.index, list(METRICS)] - comparator[list(METRICS)]
        )
        contrast = values.reset_index()
        contrast.insert(
            0, "fold", candidate.loc[contrast["slide_id"], "fold"].to_numpy(int)
        )
        contrast.insert(1, "comparison_id", f"{candidate_name}_minus_{comparator_name}")
        rows.append(contrast)
    result = pd.concat(rows, ignore_index=True)
    if result.groupby("comparison_id")["slide_id"].nunique().ne(48).any():
        raise AnalysisError("Every registered comparison must contain 48 slides.")
    return result


def summarize_contrasts(
    contrasts: pd.DataFrame,
    *,
    draws: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    metric_index = 0
    for comparison_id, comparison in contrasts.groupby("comparison_id", sort=True):
        for metric in METRICS:
            bootstrap = two_stage_cluster_bootstrap(
                comparison,
                metric,
                seed=20260729 + metric_index,
                draws=draws,
            )
            values = comparison[metric].to_numpy(float)
            fold_means = comparison.groupby("fold")[metric].mean()
            lower = float(np.quantile(bootstrap, 0.025))
            upper = float(np.quantile(bootstrap, 0.975))
            mean = float(values.mean())
            lower_is_favorable = metric in LOWER_IS_FAVORABLE
            if upper < 0:
                interpretation = (
                    "interval_supported_favorable_change"
                    if lower_is_favorable
                    else "interval_supported_regression"
                )
            elif lower > 0:
                interpretation = (
                    "interval_supported_regression"
                    if lower_is_favorable
                    else "interval_supported_favorable_change"
                )
            else:
                interpretation = "direction_uncertain"
            if metric.startswith("retrieval_") and lower >= -0.02:
                interpretation = "preserved_within_registered_0.02_margin"
            rows.append(
                {
                    "comparison_id": comparison_id,
                    "metric": metric,
                    "direction": (
                        "lower_is_favorable"
                        if lower_is_favorable
                        else "higher_is_favorable"
                    ),
                    "n_folds": 5,
                    "n_slides": 48,
                    "mean_difference": mean,
                    "median_difference": float(np.median(values)),
                    "fold_mean_min": float(fold_means.min()),
                    "fold_mean_max": float(fold_means.max()),
                    "cluster_bootstrap_ci_025": lower,
                    "cluster_bootstrap_ci_975": upper,
                    "bootstrap_draws": draws,
                    "interpretation_class": interpretation,
                    "p_value_reported": False,
                }
            )
            metric_index += 1
    return pd.DataFrame(rows)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.bootstrap_draws < 1000:
        raise AnalysisError("bootstrap-draws must be at least 1000.")
    runs, raw_slides = load_affine_rows(args.experiment_dir.resolve())
    affine = average_references(raw_slides)
    factorization = load_factorization_rows(args.factorization_slide_metrics.resolve())
    methods = pd.concat([affine, factorization], ignore_index=True).sort_values(
        ["variant", "fold", "slide_id"]
    )
    contrasts = build_contrasts(methods)
    summary = summarize_contrasts(contrasts, draws=args.bootstrap_draws)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    runs.to_csv(args.out_dir / "affine_run_metrics.csv", index=False)
    raw_slides.to_csv(
        args.out_dir / "affine_reference_specific_slide_metrics.csv", index=False
    )
    methods.to_csv(
        args.out_dir / "reference_averaged_method_slide_metrics.csv", index=False
    )
    contrasts.to_csv(args.out_dir / "slide_level_contrasts.csv", index=False)
    summary.to_csv(args.out_dir / "fold_aware_contrasts.csv", index=False)
    design = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "reference_handling": "average five reference-scanner outcomes within original slide",
        "factorization_handling": "average five optimization seeds within fold/slide before import",
        "bootstrap": "resample five folds, then slides within each sampled fold",
        "bootstrap_draws": args.bootstrap_draws,
        "registered_primary_direct_comparison": "pathoalign_dep20_minus_ridge_affine",
        "p_values_reported": False,
        "claim_boundaries": [
            "Scanner suppression is evaluated as linear recoverability only.",
            "Same-region retrieval and cosine agreement do not prove biological preservation.",
            "Affine methods harmonize into a reference domain and do not expose an acquisition branch.",
            "PathoAlign is not claimed to be the best raw scanner-removal method.",
        ],
    }
    (args.out_dir / "analysis_design.json").write_text(
        json.dumps(design, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    result = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "affine_runs": len(runs),
        "affine_reference_specific_slide_rows": len(raw_slides),
        "method_slide_rows": len(methods),
        "registered_comparisons": int(summary["comparison_id"].nunique()),
        "summary_rows": len(summary),
    }
    (args.out_dir / "analysis_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment-dir", type=Path, required=True)
    parser.add_argument("--factorization-slide-metrics", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--bootstrap-draws", type=int, default=100000)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (
        AnalysisError,
        OSError,
        RuntimeError,
        ValueError,
        np.linalg.LinAlgError,
    ) as exc:
        print(f"SCORPION PAIRED AFFINE ANALYSIS FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
