#!/usr/bin/env python3
"""
Sweep nearby dominance-detector calibration settings from run-level diagnostics.

This script reuses a detector diagnostics CSV containing clean FedAvg diagnostic
columns and clean/corrupted strategy metric columns. It recalibrates thresholds
from clean stress=0 runs for each low/high quantile pair, applies a
min-trigger-count rule, and summarizes whether the detector-transfer result is
stable around the chosen configuration.

Example:
    python scripts/experiments/sweep_detector_calibration_sensitivity.py \
        --diagnostics results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed/best_detector_run_diagnostics.csv \
        --out-dir results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed_calibration_sensitivity
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable, Dict, Iterable, List

import pandas as pd

DEFAULT_METRICS = (
    "global_qwk",
    "macro_f1",
    "worst_site_qwk",
    "mean_site_qwk",
    "global_accuracy",
)

DIAGNOSTIC_DEFINITIONS: Dict[str, tuple[str, str, Callable[[float, float], bool], str]] = {
    "global_qwk_low": ("clean_global_qwk", "low", lambda value, threshold: value < threshold, "lower"),
    "worst_site_qwk_low": ("clean_worst_site_qwk", "low", lambda value, threshold: value < threshold, "lower"),
    "site_qwk_spread_high": ("clean_site_qwk_spread", "high", lambda value, threshold: value > threshold, "upper"),
    "mean_abs_error_high": ("clean_mean_abs_error", "high", lambda value, threshold: value > threshold, "upper"),
    "severe_error_rate_high": ("clean_severe_error_rate", "high", lambda value, threshold: value > threshold, "upper"),
    "prediction_entropy_high": ("clean_prediction_entropy", "high", lambda value, threshold: value > threshold, "upper"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", required=True, help="Path to run-level detector diagnostics CSV")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--stress-column", default=None, help="Optional stress column override. Defaults to threshold_shift, then noise.")
    parser.add_argument("--low-quantiles", nargs="+", type=float, default=[0.05, 0.10, 0.15])
    parser.add_argument("--high-quantiles", nargs="+", type=float, default=[0.75, 0.80, 0.85, 0.90])
    parser.add_argument("--min-trigger-counts", nargs="+", type=int, default=[2, 3, 4])
    parser.add_argument("--include-entropy", action="store_true", help="Include prediction entropy as a high-threshold diagnostic")
    parser.add_argument("--target-stresses", nargs="+", type=int, default=[35, 45], help="Stress levels used for headline robustness")
    parser.add_argument("--max-clean-trigger-rate", type=float, default=0.20, help="Clean-regime trigger-rate cutoff for robust configurations")
    return parser.parse_args()


def to_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def pick_stress_column(frame: pd.DataFrame, requested: str | None) -> str:
    if requested:
        if requested not in frame.columns:
            raise ValueError(f"Requested stress column not found: {requested}")
        return requested
    if "threshold_shift" in frame.columns:
        return "threshold_shift"
    if "noise" in frame.columns:
        return "noise"
    raise ValueError("Could not infer stress column. Expected threshold_shift or noise.")


def mean_or_nan(values: Iterable[float]) -> float:
    values = [v for v in values if math.isfinite(v)]
    return sum(values) / len(values) if values else float("nan")


def ci95(values: Iterable[float]) -> tuple[float, float, float, float]:
    values = [v for v in values if math.isfinite(v)]
    n = len(values)
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mean = sum(values) / n
    if n == 1:
        return mean, 0.0, mean, mean
    variance = sum((v - mean) ** 2 for v in values) / (n - 1)
    sd = math.sqrt(variance)
    se = sd / math.sqrt(n)
    tcrit = 2.145 if n == 15 else 2.776 if n == 5 else 1.96
    return mean, sd, mean - tcrit * se, mean + tcrit * se


def percentile(values: pd.Series, q: float) -> float:
    clean = values.dropna().astype(float)
    if clean.empty:
        return float("nan")
    return float(clean.quantile(q))


def available_diagnostics(frame: pd.DataFrame, include_entropy: bool) -> list[str]:
    diagnostics: list[str] = []
    for name, (column, _, _, _) in DIAGNOSTIC_DEFINITIONS.items():
        if name == "prediction_entropy_high" and not include_entropy:
            continue
        if column in frame.columns:
            diagnostics.append(name)
    return diagnostics


def build_thresholds(clean_runs: pd.DataFrame, diagnostics: list[str], low_q: float, high_q: float) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for name in diagnostics:
        column, quantile_kind, _, _ = DIAGNOSTIC_DEFINITIONS[name]
        thresholds[name] = percentile(clean_runs[column], low_q if quantile_kind == "low" else high_q)
    return thresholds


def failed_diagnostics_for_row(row: pd.Series, thresholds: dict[str, float], diagnostics: list[str]) -> list[str]:
    failed: list[str] = []
    for name in diagnostics:
        column, _, predicate, _ = DIAGNOSTIC_DEFINITIONS[name]
        value = to_float(row.get(column))
        threshold = to_float(thresholds.get(name))
        if math.isfinite(value) and math.isfinite(threshold) and predicate(value, threshold):
            failed.append(name)
    return failed


def apply_detector(frame: pd.DataFrame, diagnostics: list[str], thresholds: dict[str, float], min_trigger_count: int) -> pd.DataFrame:
    output = frame.copy()
    failed_lists: list[list[str]] = []
    trigger_flags: list[bool] = []

    for _, row in output.iterrows():
        failed = failed_diagnostics_for_row(row, thresholds, diagnostics)
        failed_lists.append(failed)
        trigger_flags.append(len(failed) >= min_trigger_count)

    output["sweep_failed_diagnostics"] = [";".join(items) for items in failed_lists]
    output["sweep_failure_count"] = [len(items) for items in failed_lists]
    output["sweep_triggered"] = trigger_flags

    for metric in DEFAULT_METRICS:
        output[f"sweep_{metric}"] = output.apply(
            lambda row, metric=metric: row[f"corrupted_{metric}"] if row["sweep_triggered"] else row[f"clean_{metric}"],
            axis=1,
        )
        output[f"delta_sweep_vs_clean_{metric}"] = output[f"sweep_{metric}"].astype(float) - output[f"clean_{metric}"].astype(float)
    return output


def summarize_by_stress(detected: pd.DataFrame, stress_col: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for stress, group in detected.groupby(stress_col, dropna=False):
        row: dict[str, object] = {
            stress_col: stress,
            "runs": int(len(group)),
            "triggered_runs": int(group["sweep_triggered"].sum()),
            "trigger_rate": float(group["sweep_triggered"].mean()),
            "mean_failure_count": float(group["sweep_failure_count"].mean()),
        }
        for metric in DEFAULT_METRICS:
            values = group[f"delta_sweep_vs_clean_{metric}"].astype(float).tolist()
            mean, sd, low, high = ci95(values)
            row[f"mean_delta_{metric}"] = mean
            row[f"ci95_low_{metric}"] = low
            row[f"ci95_high_{metric}"] = high
            row[f"positive_seed_count_{metric}"] = int((group[f"delta_sweep_vs_clean_{metric}"] > 0).sum())
            row[f"negative_seed_count_{metric}"] = int((group[f"delta_sweep_vs_clean_{metric}"] < 0).sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(stress_col).reset_index(drop=True)


def headline_from_summary(
    summary: pd.DataFrame,
    stress_col: str,
    target_stresses: list[int],
    clean_stress: int = 0,
) -> dict[str, object]:
    selected = summary[summary[stress_col].astype(int).isin(target_stresses)]
    clean = summary[summary[stress_col].astype(int) == clean_stress]

    row: dict[str, object] = {
        "target_stresses": ",".join(str(item) for item in target_stresses),
        "mean_trigger_rate_target": mean_or_nan(selected["trigger_rate"].astype(float).tolist()),
        "clean_trigger_rate": float(clean["trigger_rate"].iloc[0]) if not clean.empty else float("nan"),
        "clean_global_qwk_delta": float(clean["mean_delta_global_qwk"].iloc[0]) if not clean.empty else float("nan"),
    }
    for metric in DEFAULT_METRICS:
        row[f"mean_delta_{metric}_target"] = mean_or_nan(selected[f"mean_delta_{metric}"].astype(float).tolist())
        row[f"positive_regimes_{metric}_target"] = int((selected[f"mean_delta_{metric}"] > 0).sum())
        row[f"significant_regimes_{metric}_target"] = int((selected[f"ci95_low_{metric}"] > 0).sum())
    return row


def main() -> None:
    args = parse_args()
    diagnostics_path = Path(args.diagnostics)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(diagnostics_path)
    stress_col = pick_stress_column(frame, args.stress_column)
    clean_runs = frame[frame[stress_col].astype(int) == 0]
    if clean_runs.empty:
        raise ValueError(f"No clean {stress_col}=0 rows found for threshold calibration")

    diagnostics = available_diagnostics(frame, include_entropy=args.include_entropy)
    if not diagnostics:
        raise ValueError("No detector diagnostics available")

    all_summary_rows: list[pd.DataFrame] = []
    headline_rows: list[dict[str, object]] = []
    config_rows: list[dict[str, object]] = []

    for low_q in args.low_quantiles:
        for high_q in args.high_quantiles:
            for min_count in args.min_trigger_counts:
                thresholds = build_thresholds(clean_runs, diagnostics, low_q=low_q, high_q=high_q)
                detected = apply_detector(frame, diagnostics, thresholds, min_trigger_count=min_count)
                summary = summarize_by_stress(detected, stress_col)

                config_id = f"low_{low_q:g}__high_{high_q:g}__min_{min_count}"
                summary.insert(0, "config_id", config_id)
                summary.insert(1, "low_quantile", low_q)
                summary.insert(2, "high_quantile", high_q)
                summary.insert(3, "min_trigger_count", min_count)
                all_summary_rows.append(summary)

                headline = headline_from_summary(summary, stress_col, target_stresses=args.target_stresses)
                headline.update(
                    {
                        "config_id": config_id,
                        "low_quantile": low_q,
                        "high_quantile": high_q,
                        "min_trigger_count": min_count,
                    }
                )
                headline_rows.append(headline)
                config_rows.append(
                    {
                        "config_id": config_id,
                        "low_quantile": low_q,
                        "high_quantile": high_q,
                        "min_trigger_count": min_count,
                        "diagnostics": ";".join(diagnostics),
                        **{f"threshold_{key}": value for key, value in thresholds.items()},
                    }
                )

    summary_all = pd.concat(all_summary_rows, ignore_index=True)
    headline_all = pd.DataFrame(headline_rows)
    configs = pd.DataFrame(config_rows)

    headline_all["passes_clean_trigger_cutoff"] = headline_all["clean_trigger_rate"] <= args.max_clean_trigger_rate
    headline_all["positive_global_qwk_all_targets"] = headline_all["positive_regimes_global_qwk_target"] == len(args.target_stresses)
    headline_all["significant_global_qwk_all_targets"] = headline_all["significant_regimes_global_qwk_target"] == len(args.target_stresses)
    headline_all["positive_macro_f1_all_targets"] = headline_all["positive_regimes_macro_f1_target"] == len(args.target_stresses)
    headline_all["positive_worst_site_qwk_all_targets"] = headline_all["positive_regimes_worst_site_qwk_target"] == len(args.target_stresses)
    headline_all["robust_positive_config"] = (
        headline_all["passes_clean_trigger_cutoff"]
        & headline_all["positive_global_qwk_all_targets"]
        & headline_all["positive_macro_f1_all_targets"]
        & headline_all["positive_worst_site_qwk_all_targets"]
    )

    summary_path = out_dir / "calibration_sensitivity_summary_by_stress.csv"
    headline_path = out_dir / "calibration_sensitivity_headline.csv"
    configs_path = out_dir / "calibration_sensitivity_thresholds.csv"
    config_path = out_dir / "calibration_sensitivity_config.json"

    summary_all.to_csv(summary_path, index=False)
    headline_all.sort_values(
        ["robust_positive_config", "significant_global_qwk_all_targets", "mean_delta_global_qwk_target"],
        ascending=[False, False, False],
    ).to_csv(headline_path, index=False)
    configs.to_csv(configs_path, index=False)
    config_path.write_text(
        json.dumps(
            {
                "diagnostics_path": str(diagnostics_path),
                "stress_column": stress_col,
                "diagnostics": diagnostics,
                "low_quantiles": args.low_quantiles,
                "high_quantiles": args.high_quantiles,
                "min_trigger_counts": args.min_trigger_counts,
                "include_entropy": args.include_entropy,
                "target_stresses": args.target_stresses,
                "max_clean_trigger_rate": args.max_clean_trigger_rate,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    robust = headline_all[headline_all["robust_positive_config"]]
    print(f"Diagnostics: {', '.join(diagnostics)}")
    print(f"Evaluated configurations: {len(headline_all)}")
    print(f"Robust positive configurations: {len(robust)}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {headline_path}")
    print(f"Wrote {configs_path}")
    print(f"Wrote {config_path}")
    print("\nTop configurations:")
    columns = [
        "config_id",
        "clean_trigger_rate",
        "mean_trigger_rate_target",
        "mean_delta_global_qwk_target",
        "mean_delta_macro_f1_target",
        "mean_delta_worst_site_qwk_target",
        "significant_regimes_global_qwk_target",
        "robust_positive_config",
    ]
    print(
        headline_all.sort_values(
            ["robust_positive_config", "significant_global_qwk_all_targets", "mean_delta_global_qwk_target"],
            ascending=[False, False, False],
        )[columns]
        .head(12)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
