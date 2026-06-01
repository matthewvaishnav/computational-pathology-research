#!/usr/bin/env python3
"""Tune observable dominance-aware switch detectors.

This script searches simple clean-calibrated detector rules for the dominance-aware
switch experiment. It uses saved predictions and only FedAvg validation diagnostics.

The base detector in analyze_dominance_detector_switch.py used an OR rule: switch
when any clean-calibrated diagnostic is out of range. That recovered corrupted
regimes, but produced clean-regime false triggers. This tuner evaluates stricter
rules such as "at least 2 diagnostics must fire" and different clean quantiles.

Outputs:
- detector_grid_summary.csv: one row per detector configuration
- best_detector_summary.csv: per-noise/per-metric summary for the best config
- best_detector_run_diagnostics.csv: seed-level diagnostics for the best config
- best_detector_thresholds.json: thresholds for the best config
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Make sibling experiment modules importable when the script is run from repo root.
sys.path.append(str(Path(__file__).resolve().parent))

from analyze_dominance_detector_switch import (  # noqa: E402
    DEFAULT_METRICS,
    build_thresholds,
    ci95,
    iter_prediction_files,
    summarize_run,
)

DIAGNOSTIC_SPECS = (
    ("global_qwk_low", "clean_global_qwk", "lt"),
    ("worst_site_qwk_low", "clean_worst_site_qwk", "lt"),
    ("site_qwk_spread_high", "clean_site_qwk_spread", "gt"),
    ("mean_abs_error_high", "clean_mean_abs_error", "gt"),
    ("severe_error_rate_high", "clean_severe_error_rate", "gt"),
)


def load_runs(patterns: list[str], clean_strategy: str, corrupted_strategy: str, n_classes: int) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for path in iter_prediction_files(patterns):
        row = summarize_run(path, clean_strategy, corrupted_strategy, n_classes)
        if row is not None:
            rows.append(row)

    runs = pd.DataFrame(rows).drop_duplicates(["noise", "seed"])
    if runs.empty:
        raise ValueError("No usable prediction files found.")
    if runs[runs["noise"] == 0].empty:
        raise ValueError("No clean noise=0 runs found for threshold calibration.")
    return runs


def failed_diagnostics(row: pd.Series, thresholds: dict[str, float], use_entropy: bool) -> list[str]:
    failures: list[str] = []
    for threshold_name, column, direction in DIAGNOSTIC_SPECS:
        threshold = thresholds[threshold_name]
        value = row[column]
        if direction == "lt" and value < threshold:
            failures.append(threshold_name)
        elif direction == "gt" and value > threshold:
            failures.append(threshold_name)

    if use_entropy:
        entropy = float(row.get("clean_prediction_entropy", float("nan")))
        threshold = thresholds.get("prediction_entropy_high", float("nan"))
        if math.isfinite(entropy) and math.isfinite(threshold) and entropy > threshold:
            failures.append("prediction_entropy_high")

    return failures


def apply_tuned_detector(
    runs: pd.DataFrame,
    thresholds: dict[str, float],
    min_trigger_count: int,
    use_entropy: bool,
) -> pd.DataFrame:
    output = runs.copy()
    failure_lists = [failed_diagnostics(row, thresholds, use_entropy) for _, row in output.iterrows()]
    output["detector_failure_count"] = [len(items) for items in failure_lists]
    output["detector_failed_diagnostics"] = [";".join(items) for items in failure_lists]
    output["detector_triggered"] = output["detector_failure_count"] >= min_trigger_count
    output["chosen_strategy"] = np.where(
        output["detector_triggered"],
        output["corrupted_strategy"],
        output["clean_strategy"],
    )

    for metric in DEFAULT_METRICS:
        output[f"detector_{metric}"] = np.where(
            output["detector_triggered"],
            output[f"corrupted_{metric}"],
            output[f"clean_{metric}"],
        )
        output[f"oracle_{metric}"] = np.where(
            output["noise"] == 0,
            output[f"clean_{metric}"],
            output[f"corrupted_{metric}"],
        )
        output[f"delta_detector_vs_clean_{metric}"] = output[f"detector_{metric}"] - output[f"clean_{metric}"]
        output[f"detector_regret_vs_oracle_{metric}"] = output[f"detector_{metric}"] - output[f"oracle_{metric}"]

    return output


def summarize_detector(detected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for noise, group in detected.groupby("noise"):
        for metric in DEFAULT_METRICS:
            deltas = group[f"delta_detector_vs_clean_{metric}"].to_numpy(dtype=float)
            regrets = group[f"detector_regret_vs_oracle_{metric}"].to_numpy(dtype=float)
            mean, sd, low, high = ci95(deltas)
            regret_mean, regret_sd, regret_low, regret_high = ci95(regrets)

            rows.append(
                {
                    "noise": int(noise),
                    "metric": metric,
                    "n_seeds": int(group["seed"].nunique()),
                    "detector_trigger_rate": float(group["detector_triggered"].mean()),
                    "clean_mean": float(group[f"clean_{metric}"].mean()),
                    "corrupted_strategy_mean": float(group[f"corrupted_{metric}"].mean()),
                    "detector_mean": float(group[f"detector_{metric}"].mean()),
                    "oracle_mean": float(group[f"oracle_{metric}"].mean()),
                    "mean_delta_detector_vs_clean": mean,
                    "sd_delta_detector_vs_clean": sd,
                    "ci95_low": low,
                    "ci95_high": high,
                    "positive_seed_count": int((group[f"delta_detector_vs_clean_{metric}"] > 0).sum()),
                    "negative_seed_count": int((group[f"delta_detector_vs_clean_{metric}"] < 0).sum()),
                    "mean_regret_detector_vs_oracle": regret_mean,
                    "regret_ci95_low": regret_low,
                    "regret_ci95_high": regret_high,
                }
            )
    return pd.DataFrame(rows).sort_values(["noise", "metric"])


def config_score(summary: pd.DataFrame) -> dict[str, float]:
    global_qwk = summary[summary["metric"] == "global_qwk"].set_index("noise")
    worst_site = summary[summary["metric"] == "worst_site_qwk"].set_index("noise")
    macro_f1 = summary[summary["metric"] == "macro_f1"].set_index("noise")

    clean_trigger = float(global_qwk.loc[0, "detector_trigger_rate"])
    clean_global_delta = float(global_qwk.loc[0, "mean_delta_detector_vs_clean"])
    clean_worst_delta = float(worst_site.loc[0, "mean_delta_detector_vs_clean"])

    noisy_global_mean = float(global_qwk.loc[[25, 35, 45], "mean_delta_detector_vs_clean"].mean())
    noisy_global_positive = int((global_qwk.loc[[25, 35, 45], "mean_delta_detector_vs_clean"] > 0).sum())
    significant_global_count = int((global_qwk.loc[[25, 35, 45], "ci95_low"] > 0).sum())
    noisy_trigger_mean = float(global_qwk.loc[[25, 35, 45], "detector_trigger_rate"].mean())
    corrupted_regret = float(global_qwk.loc[[25, 35, 45], "mean_regret_detector_vs_oracle"].mean())
    noisy_macro_mean = float(macro_f1.loc[[25, 35, 45], "mean_delta_detector_vs_clean"].mean())
    noisy_worst_mean = float(worst_site.loc[[25, 35, 45], "mean_delta_detector_vs_clean"].mean())

    # Prefer strong noisy-regime gains, low clean false trigger, low oracle regret,
    # and at least some significant corrupted-regime intervals.
    score = (
        noisy_global_mean
        + 0.25 * noisy_macro_mean
        + 0.25 * noisy_worst_mean
        + 0.0025 * significant_global_count
        - 0.0040 * clean_trigger
        + clean_global_delta
        + 0.25 * clean_worst_delta
        + 0.25 * corrupted_regret
    )

    return {
        "score": score,
        "clean_trigger_rate": clean_trigger,
        "noisy_trigger_rate_mean": noisy_trigger_mean,
        "clean_global_qwk_delta": clean_global_delta,
        "clean_worst_site_qwk_delta": clean_worst_delta,
        "noisy_global_qwk_delta_mean": noisy_global_mean,
        "noisy_macro_f1_delta_mean": noisy_macro_mean,
        "noisy_worst_site_qwk_delta_mean": noisy_worst_mean,
        "noisy_global_positive_regime_count": noisy_global_positive,
        "noisy_global_significant_regime_count": significant_global_count,
        "noisy_global_regret_vs_oracle_mean": corrupted_regret,
    }


def tune(
    runs: pd.DataFrame,
    low_quantiles: list[float],
    high_quantiles: list[float],
    min_trigger_counts: list[int],
    use_entropy_values: list[bool],
) -> tuple[pd.DataFrame, dict[str, object], pd.DataFrame, pd.DataFrame, dict[str, float]]:
    clean_runs = runs[runs["noise"] == 0]
    grid_rows: list[dict[str, object]] = []
    best: dict[str, object] | None = None
    best_summary: pd.DataFrame | None = None
    best_detected: pd.DataFrame | None = None
    best_thresholds: dict[str, float] | None = None

    for low_q in low_quantiles:
        for high_q in high_quantiles:
            thresholds = build_thresholds(clean_runs, low_q=low_q, high_q=high_q)
            for min_count in min_trigger_counts:
                for use_entropy in use_entropy_values:
                    detected = apply_tuned_detector(
                        runs,
                        thresholds=thresholds,
                        min_trigger_count=min_count,
                        use_entropy=use_entropy,
                    )
                    summary = summarize_detector(detected)
                    score_parts = config_score(summary)
                    row = {
                        "low_quantile": low_q,
                        "high_quantile": high_q,
                        "min_trigger_count": min_count,
                        "use_entropy": use_entropy,
                        **score_parts,
                    }
                    grid_rows.append(row)
                    if best is None or float(row["score"]) > float(best["score"]):
                        best = row
                        best_summary = summary
                        best_detected = detected
                        best_thresholds = thresholds

    assert best is not None
    assert best_summary is not None
    assert best_detected is not None
    assert best_thresholds is not None
    grid = pd.DataFrame(grid_rows).sort_values("score", ascending=False)
    return grid, best, best_summary, best_detected, best_thresholds


def parse_float_list(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def parse_int_list(text: str) -> list[int]:
    return [int(x.strip()) for x in text.split(",") if x.strip()]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patterns", nargs="+", required=True, help="Glob patterns for predictions.csv files.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--clean-strategy", default="fedavg")
    parser.add_argument("--corrupted-strategy", default="cross_site_blend_50")
    parser.add_argument("--n-classes", type=int, default=6)
    parser.add_argument("--low-quantiles", default="0.05,0.10,0.15,0.20")
    parser.add_argument("--high-quantiles", default="0.80,0.85,0.90,0.95")
    parser.add_argument("--min-trigger-counts", default="1,2,3")
    parser.add_argument("--include-entropy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(
        args.patterns,
        clean_strategy=args.clean_strategy,
        corrupted_strategy=args.corrupted_strategy,
        n_classes=args.n_classes,
    )

    use_entropy_values = [False, True] if args.include_entropy else [False]
    grid, best, best_summary, best_detected, best_thresholds = tune(
        runs,
        low_quantiles=parse_float_list(args.low_quantiles),
        high_quantiles=parse_float_list(args.high_quantiles),
        min_trigger_counts=parse_int_list(args.min_trigger_counts),
        use_entropy_values=use_entropy_values,
    )

    grid.to_csv(out_dir / "detector_grid_summary.csv", index=False)
    best_summary.to_csv(out_dir / "best_detector_summary.csv", index=False)
    best_detected.to_csv(out_dir / "best_detector_run_diagnostics.csv", index=False)
    (out_dir / "best_detector_config.json").write_text(json.dumps(best, indent=2), encoding="utf-8")
    (out_dir / "best_detector_thresholds.json").write_text(json.dumps(best_thresholds, indent=2), encoding="utf-8")

    print("Top detector configs:")
    print(grid.head(10).to_string(index=False))
    print("\nBest detector summary:")
    print(best_summary[best_summary["metric"].isin({"global_qwk", "worst_site_qwk", "macro_f1"})].to_string(index=False))
    print(f"\nWrote detector tuning outputs to {out_dir}")


if __name__ == "__main__":
    main()
