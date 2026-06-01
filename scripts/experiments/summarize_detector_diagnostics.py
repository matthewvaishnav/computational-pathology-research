#!/usr/bin/env python3
"""
Summarize dominance-detector diagnostics from a run-level diagnostics CSV.

The detector-transfer scripts write one row per stress run with columns such as:

    detector_triggered
    detector_failure_count
    detector_failed_diagnostics
    delta_detector_vs_clean_global_qwk
    delta_detector_vs_clean_macro_f1
    delta_detector_vs_clean_worst_site_qwk

This helper answers two practical questions:

1. Which diagnostics trigger most often by stress level?
2. When the detector triggers, are the resulting metric deltas positive or harmful?

Example:
    python scripts/experiments/summarize_detector_diagnostics.py \
        --diagnostics results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed/best_detector_run_diagnostics.csv \
        --out-dir results/threshold_shift_detector_conservative_fixed_labelnoise_rule_15seed_diagnostic_summary
"""

from __future__ import annotations

import argparse
import math
import re
from collections import Counter
from pathlib import Path
from typing import Iterable

import pandas as pd

KEY_METRICS = (
    "global_qwk",
    "macro_f1",
    "worst_site_qwk",
    "mean_site_qwk",
    "global_accuracy",
)

DIAGNOSTIC_ALIASES = {
    "FedAvg global QWK below clean-calibrated lower bound": "global_qwk_low",
    "FedAvg worst-site QWK below clean-calibrated lower bound": "worst_site_qwk_low",
    "FedAvg site-QWK spread above clean-calibrated upper bound": "site_qwk_spread_high",
    "FedAvg mean absolute ordinal error above clean-calibrated upper bound": "mean_abs_error_high",
    "FedAvg severe ordinal error rate above clean-calibrated upper bound": "severe_error_rate_high",
    "FedAvg prediction entropy above clean-calibrated upper bound": "prediction_entropy_high",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnostics", required=True, help="Path to best_detector_run_diagnostics.csv or detector_run_diagnostics.csv")
    parser.add_argument("--out-dir", required=True, help="Output directory for diagnostic summaries")
    parser.add_argument("--stress-column", default=None, help="Optional stress column override. Defaults to threshold_shift, then noise.")
    return parser.parse_args()


def to_bool(value: object) -> bool:
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def to_float(value: object) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def clean_diagnostic_name(name: str) -> str:
    name = name.strip()
    if not name:
        return ""
    if name in DIAGNOSTIC_ALIASES:
        return DIAGNOSTIC_ALIASES[name]
    normalized = re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()
    return normalized or name


def split_diagnostics(value: object) -> list[str]:
    text = str(value).strip()
    if not text or text.lower() == "nan":
        return []
    parts = re.split(r"\s*;\s*|\s*\|\s*|\s*,\s*", text)
    return [clean_diagnostic_name(part) for part in parts if clean_diagnostic_name(part)]


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


def load_diagnostics(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    if "detector_triggered" not in frame.columns:
        raise ValueError("Diagnostics CSV is missing detector_triggered column")
    if "detector_failed_diagnostics" not in frame.columns:
        if "detector_reasons" in frame.columns:
            frame["detector_failed_diagnostics"] = frame["detector_reasons"]
        else:
            raise ValueError("Diagnostics CSV is missing detector_failed_diagnostics or detector_reasons column")

    frame["detector_triggered_bool"] = frame["detector_triggered"].map(to_bool)
    frame["parsed_failed_diagnostics"] = frame["detector_failed_diagnostics"].map(split_diagnostics)
    if "detector_failure_count" in frame.columns:
        frame["detector_failure_count_numeric"] = frame["detector_failure_count"].map(to_float)
    else:
        frame["detector_failure_count_numeric"] = frame["parsed_failed_diagnostics"].map(len)
    return frame


def build_trigger_summary(frame: pd.DataFrame, stress_col: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for stress, group in frame.groupby(stress_col, dropna=False):
        triggered = group[group["detector_triggered_bool"]]
        row: dict[str, object] = {
            stress_col: stress,
            "runs": int(len(group)),
            "triggered_runs": int(len(triggered)),
            "trigger_rate": float(len(triggered) / len(group)) if len(group) else float("nan"),
            "mean_failure_count": mean_or_nan(group["detector_failure_count_numeric"].tolist()),
            "mean_failure_count_when_triggered": mean_or_nan(triggered["detector_failure_count_numeric"].tolist()),
        }
        for metric in KEY_METRICS:
            col = f"delta_detector_vs_clean_{metric}"
            if col in group.columns:
                values = group[col].map(to_float)
                trig_values = triggered[col].map(to_float) if not triggered.empty else pd.Series(dtype=float)
                row[f"mean_delta_{metric}"] = mean_or_nan(values.tolist())
                row[f"mean_delta_{metric}_when_triggered"] = mean_or_nan(trig_values.tolist())
                row[f"positive_triggered_{metric}"] = int((trig_values > 0).sum()) if not trig_values.empty else 0
                row[f"negative_triggered_{metric}"] = int((trig_values < 0).sum()) if not trig_values.empty else 0
        rows.append(row)
    return pd.DataFrame(rows).sort_values(stress_col).reset_index(drop=True)


def build_diagnostic_frequency(frame: pd.DataFrame, stress_col: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for stress, group in frame.groupby(stress_col, dropna=False):
        counter: Counter[str] = Counter()
        trigger_count = int(group["detector_triggered_bool"].sum())
        for diagnostics in group["parsed_failed_diagnostics"]:
            counter.update(diagnostics)
        for diagnostic, count in sorted(counter.items(), key=lambda item: (-item[1], item[0])):
            rows.append(
                {
                    stress_col: stress,
                    "diagnostic": diagnostic,
                    "count": int(count),
                    "triggered_runs": trigger_count,
                    "share_of_triggered_runs": float(count / trigger_count) if trigger_count else 0.0,
                    "total_runs": int(len(group)),
                    "share_of_total_runs": float(count / len(group)) if len(group) else 0.0,
                }
            )
    return pd.DataFrame(rows).sort_values([stress_col, "count", "diagnostic"], ascending=[True, False, True]).reset_index(drop=True)


def build_diagnostic_outcomes(frame: pd.DataFrame, stress_col: str) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    exploded = frame.explode("parsed_failed_diagnostics").rename(columns={"parsed_failed_diagnostics": "diagnostic"})
    exploded = exploded[exploded["diagnostic"].notna() & (exploded["diagnostic"].astype(str) != "")]

    if exploded.empty:
        return pd.DataFrame(columns=[stress_col, "diagnostic", "count"])

    for (stress, diagnostic), group in exploded.groupby([stress_col, "diagnostic"], dropna=False):
        row: dict[str, object] = {stress_col: stress, "diagnostic": diagnostic, "count": int(len(group))}
        for metric in KEY_METRICS:
            col = f"delta_detector_vs_clean_{metric}"
            if col in group.columns:
                values = group[col].map(to_float)
                row[f"mean_delta_{metric}"] = mean_or_nan(values.tolist())
                row[f"positive_count_{metric}"] = int((values > 0).sum())
                row[f"negative_count_{metric}"] = int((values < 0).sum())
        rows.append(row)
    return pd.DataFrame(rows).sort_values([stress_col, "count", "diagnostic"], ascending=[True, False, True]).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    diagnostics_path = Path(args.diagnostics)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = load_diagnostics(diagnostics_path)
    stress_col = pick_stress_column(frame, args.stress_column)

    trigger_summary = build_trigger_summary(frame, stress_col)
    diagnostic_frequency = build_diagnostic_frequency(frame, stress_col)
    diagnostic_outcomes = build_diagnostic_outcomes(frame, stress_col)

    trigger_summary_path = out_dir / "trigger_summary_by_stress.csv"
    frequency_path = out_dir / "diagnostic_frequency_by_stress.csv"
    outcomes_path = out_dir / "diagnostic_outcomes_by_stress.csv"

    trigger_summary.to_csv(trigger_summary_path, index=False)
    diagnostic_frequency.to_csv(frequency_path, index=False)
    diagnostic_outcomes.to_csv(outcomes_path, index=False)

    print(f"Wrote {trigger_summary_path}")
    print(f"Wrote {frequency_path}")
    print(f"Wrote {outcomes_path}")
    print("\nTrigger summary:")
    print(trigger_summary.to_string(index=False))
    if not diagnostic_frequency.empty:
        print("\nTop diagnostic frequencies:")
        print(diagnostic_frequency.groupby("diagnostic")["count"].sum().sort_values(ascending=False).to_string())


if __name__ == "__main__":
    main()
