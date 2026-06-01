#!/usr/bin/env python3
"""
Run leave-one-diagnostic-family-out ablations for a dominance detector.

This script reuses a detector run-level diagnostics CSV and threshold JSON. It does
not need the original prediction files. It recomputes detector-trigger decisions
from the clean FedAvg diagnostic columns, then evaluates:

    - full detector
    - minus each diagnostic family
    - each diagnostic family alone

Example:
    python scripts/experiments/ablate_detector_diagnostics.py \
        --diagnostics results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed/best_detector_run_diagnostics.csv \
        --thresholds results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed/best_detector_thresholds.json \
        --out-dir results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed_leave_one_out
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

DIAGNOSTIC_DEFINITIONS: Dict[str, tuple[str, str, Callable[[float, float], bool]]] = {
    "global_qwk_low": ("clean_global_qwk", "global_qwk_low", lambda value, threshold: value < threshold),
    "worst_site_qwk_low": ("clean_worst_site_qwk", "worst_site_qwk_low", lambda value, threshold: value < threshold),
    "site_qwk_spread_high": ("clean_site_qwk_spread", "site_qwk_spread_high", lambda value, threshold: value > threshold),
    "mean_abs_error_high": ("clean_mean_abs_error", "mean_abs_error_high", lambda value, threshold: value > threshold),
    "severe_error_rate_high": ("clean_severe_error_rate", "severe_error_rate_high", lambda value, threshold: value > threshold),
    "prediction_entropy_high": ("clean_prediction_entropy", "prediction_entropy_high", lambda value, threshold: value > threshold),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", required=True, help="Path to run-level detector diagnostics CSV")
    parser.add_argument("--thresholds", required=True, help="Path to detector thresholds JSON")
    parser.add_argument("--out-dir", required=True, help="Output directory")
    parser.add_argument("--stress-column", default=None, help="Optional stress column override. Defaults to threshold_shift, then noise.")
    parser.add_argument("--min-trigger-count", type=int, default=3, help="Number of failed diagnostics needed to trigger")
    parser.add_argument("--include-entropy", action="store_true", help="Include prediction_entropy_high in the full detector")
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


def load_inputs(diagnostics_path: Path, thresholds_path: Path) -> tuple[pd.DataFrame, dict[str, float]]:
    frame = pd.read_csv(diagnostics_path)
    thresholds = json.loads(thresholds_path.read_text(encoding="utf-8"))
    thresholds = {key: to_float(value) for key, value in thresholds.items()}

    required = {"clean_strategy", "corrupted_strategy"}
    for metric in DEFAULT_METRICS:
        required.add(f"clean_{metric}")
        required.add(f"corrupted_{metric}")
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Diagnostics CSV is missing required columns: {sorted(missing)}")
    return frame, thresholds


def available_diagnostics(frame: pd.DataFrame, thresholds: dict[str, float], include_entropy: bool) -> list[str]:
    diagnostics: list[str] = []
    for name, (column, threshold_key, _) in DIAGNOSTIC_DEFINITIONS.items():
        if name == "prediction_entropy_high" and not include_entropy:
            continue
        if column in frame.columns and threshold_key in thresholds and math.isfinite(thresholds[threshold_key]):
            diagnostics.append(name)
    return diagnostics


def failed_diagnostics_for_row(row: pd.Series, thresholds: dict[str, float], diagnostics: list[str]) -> list[str]:
    failed: list[str] = []
    for name in diagnostics:
        column, threshold_key, predicate = DIAGNOSTIC_DEFINITIONS[name]
        value = to_float(row.get(column))
        threshold = to_float(thresholds.get(threshold_key))
        if math.isfinite(value) and math.isfinite(threshold) and predicate(value, threshold):
            failed.append(name)
    return failed


def apply_variant(frame: pd.DataFrame, thresholds: dict[str, float], diagnostics: list[str], min_trigger_count: int, variant_name: str) -> pd.DataFrame:
    output = frame.copy()
    failed_lists: list[list[str]] = []
    trigger_flags: list[bool] = []

    for _, row in output.iterrows():
        failed = failed_diagnostics_for_row(row, thresholds, diagnostics)
        failed_lists.append(failed)
        trigger_flags.append(len(failed) >= min_trigger_count)

    output["variant"] = variant_name
    output["variant_diagnostics"] = ";".join(diagnostics)
    output["variant_failed_diagnostics"] = [";".join(items) for items in failed_lists]
    output["variant_failure_count"] = [len(items) for items in failed_lists]
    output["variant_triggered"] = trigger_flags
    output["variant_chosen_strategy"] = output.apply(
        lambda row: row["corrupted_strategy"] if row["variant_triggered"] else row["clean_strategy"], axis=1
    )

    for metric in DEFAULT_METRICS:
        output[f"variant_{metric}"] = output.apply(
            lambda row, metric=metric: row[f"corrupted_{metric}"] if row["variant_triggered"] else row[f"clean_{metric}"],
            axis=1,
        )
        output[f"delta_variant_vs_clean_{metric}"] = output[f"variant_{metric}"].astype(float) - output[f"clean_{metric}"].astype(float)
    return output


def build_variants(base_diagnostics: list[str], min_trigger_count: int) -> dict[str, tuple[list[str], int]]:
    variants: dict[str, tuple[list[str], int]] = {"full": (base_diagnostics, min_trigger_count)}

    for diagnostic in base_diagnostics:
        remaining = [item for item in base_diagnostics if item != diagnostic]
        variants[f"minus_{diagnostic}"] = (remaining, min_trigger_count)

    # Single-diagnostic variants use min_trigger_count=1 because there is only one diagnostic family present.
    for diagnostic in base_diagnostics:
        variants[f"only_{diagnostic}"] = ([diagnostic], 1)

    return variants


def summarize_variant_runs(variant_runs: pd.DataFrame, stress_col: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (variant, stress), group in variant_runs.groupby(["variant", stress_col], dropna=False):
        row: dict[str, object] = {
            "variant": variant,
            stress_col: stress,
            "runs": int(len(group)),
            "triggered_runs": int(group["variant_triggered"].sum()),
            "trigger_rate": float(group["variant_triggered"].mean()),
            "mean_failure_count": float(group["variant_failure_count"].mean()),
            "mean_failure_count_when_triggered": mean_or_nan(group.loc[group["variant_triggered"], "variant_failure_count"].astype(float).tolist()),
        }
        for metric in DEFAULT_METRICS:
            values = group[f"delta_variant_vs_clean_{metric}"].astype(float).tolist()
            mean, sd, low, high = ci95(values)
            row[f"mean_delta_{metric}"] = mean
            row[f"sd_delta_{metric}"] = sd
            row[f"ci95_low_{metric}"] = low
            row[f"ci95_high_{metric}"] = high
            row[f"positive_seed_count_{metric}"] = int((group[f"delta_variant_vs_clean_{metric}"] > 0).sum())
            row[f"negative_seed_count_{metric}"] = int((group[f"delta_variant_vs_clean_{metric}"] < 0).sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values(["variant", stress_col]).reset_index(drop=True)


def build_headline_comparison(summary: pd.DataFrame, stress_col: str) -> pd.DataFrame:
    headline_stresses = [35, 45]
    rows: list[dict[str, object]] = []
    for variant, group in summary.groupby("variant"):
        selected = group[group[stress_col].astype(int).isin(headline_stresses)].copy()
        if selected.empty:
            continue
        rows.append(
            {
                "variant": variant,
                "mean_trigger_rate_35_45": float(selected["trigger_rate"].mean()),
                "mean_global_qwk_delta_35_45": float(selected["mean_delta_global_qwk"].mean()),
                "mean_macro_f1_delta_35_45": float(selected["mean_delta_macro_f1"].mean()),
                "mean_worst_site_qwk_delta_35_45": float(selected["mean_delta_worst_site_qwk"].mean()),
                "significant_global_qwk_regimes_35_45": int((selected["ci95_low_global_qwk"] > 0).sum()),
                "positive_global_qwk_regimes_35_45": int((selected["mean_delta_global_qwk"] > 0).sum()),
            }
        )
    return pd.DataFrame(rows).sort_values(
        ["mean_global_qwk_delta_35_45", "mean_macro_f1_delta_35_45"], ascending=False
    ).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    diagnostics_path = Path(args.diagnostics)
    thresholds_path = Path(args.thresholds)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame, thresholds = load_inputs(diagnostics_path, thresholds_path)
    stress_col = pick_stress_column(frame, args.stress_column)
    base_diagnostics = available_diagnostics(frame, thresholds, include_entropy=args.include_entropy)
    if not base_diagnostics:
        raise ValueError("No detector diagnostics were available for ablation")

    variants = build_variants(base_diagnostics, min_trigger_count=args.min_trigger_count)
    variant_frames: List[pd.DataFrame] = []
    for variant_name, (diagnostics, trigger_count) in variants.items():
        variant_frames.append(apply_variant(frame, thresholds, diagnostics, trigger_count, variant_name))

    variant_runs = pd.concat(variant_frames, ignore_index=True)
    summary = summarize_variant_runs(variant_runs, stress_col)
    headline = build_headline_comparison(summary, stress_col)

    variant_runs_path = out_dir / "diagnostic_ablation_per_run.csv"
    summary_path = out_dir / "diagnostic_ablation_summary_by_stress.csv"
    headline_path = out_dir / "diagnostic_ablation_headline_35_45.csv"
    config_path = out_dir / "diagnostic_ablation_config.json"

    variant_runs.to_csv(variant_runs_path, index=False)
    summary.to_csv(summary_path, index=False)
    headline.to_csv(headline_path, index=False)
    config_path.write_text(
        json.dumps(
            {
                "diagnostics_path": str(diagnostics_path),
                "thresholds_path": str(thresholds_path),
                "stress_column": stress_col,
                "base_diagnostics": base_diagnostics,
                "min_trigger_count": args.min_trigger_count,
                "include_entropy": args.include_entropy,
                "thresholds": thresholds,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"Base diagnostics: {', '.join(base_diagnostics)}")
    print(f"Wrote {variant_runs_path}")
    print(f"Wrote {summary_path}")
    print(f"Wrote {headline_path}")
    print(f"Wrote {config_path}")
    print("\nHeadline 35/45 comparison:")
    print(headline.to_string(index=False))


if __name__ == "__main__":
    main()
