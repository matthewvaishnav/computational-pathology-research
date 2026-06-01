#!/usr/bin/env python3
"""Evaluate dominance-detector transfer on ordinal threshold-shift runs.

The generic dominance detector expects result paths containing
`noise_<level>_seed_<seed>`. Threshold-shift experiments use paths such as:

    threshold_shift_panda_all_aggressive_25_seed_42/predictions.csv

This wrapper parses `_<shift>_seed_<seed>` from threshold-shift paths and then
reuses the same metric computation, detector, and summary logic. The output
schema intentionally keeps the column name `noise` for compatibility with the
existing detector summary tools; in this script it means threshold-shift percent.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterable

import pandas as pd

sys.path.append(str(Path(__file__).resolve().parent))

from analyze_dominance_detector_switch import (  # noqa: E402
    SPLIT_COL,
    STRATEGY_COL,
    apply_detector,
    build_summary,
    build_thresholds,
    compute_strategy_metrics,
    iter_prediction_files,
)


def parse_shift_seed(path: Path) -> tuple[int, int] | None:
    """Extract threshold-shift level and seed from threshold-shift result paths."""
    text = str(path)
    match = re.search(r"threshold_shift_panda_all_[^_]+_(\d+)_seed_(\d+)", text)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def summarize_threshold_shift_run(
    path: Path,
    clean_strategy: str,
    corrupted_strategy: str,
    n_classes: int,
) -> dict[str, object] | None:
    parsed = parse_shift_seed(path)
    if parsed is None:
        return None
    shift, seed = parsed

    df = pd.read_csv(path)
    if SPLIT_COL in df.columns:
        df = df[df[SPLIT_COL].astype(str).str.lower() == "val"].copy()

    strategies = set(df[STRATEGY_COL].astype(str))
    if clean_strategy not in strategies or corrupted_strategy not in strategies:
        return None

    clean_df = df[df[STRATEGY_COL] == clean_strategy]
    corrupted_df = df[df[STRATEGY_COL] == corrupted_strategy]
    clean_metrics = compute_strategy_metrics(clean_df, n_classes=n_classes)
    corrupted_metrics = compute_strategy_metrics(corrupted_df, n_classes=n_classes)

    row: dict[str, object] = {
        "source_file": str(path),
        "noise": shift,
        "threshold_shift": shift,
        "seed": seed,
        "clean_strategy": clean_strategy,
        "corrupted_strategy": corrupted_strategy,
    }
    for key, value in clean_metrics.items():
        row[f"clean_{key}"] = value
    for key, value in corrupted_metrics.items():
        row[f"corrupted_{key}"] = value
        if key in clean_metrics:
            row[f"delta_corrupted_vs_clean_{key}"] = value - clean_metrics[key]
    return row


def load_runs(patterns: Iterable[str], clean_strategy: str, corrupted_strategy: str, n_classes: int) -> pd.DataFrame:
    rows = []
    for path in iter_prediction_files(patterns):
        row = summarize_threshold_shift_run(path, clean_strategy, corrupted_strategy, n_classes)
        if row is not None:
            rows.append(row)

    runs = pd.DataFrame(rows).drop_duplicates(["noise", "seed"])
    if runs.empty:
        raise ValueError("No usable threshold-shift prediction files found.")
    if runs[runs["noise"] == 0].empty:
        raise ValueError("No clean threshold_shift=0 runs found for detector calibration.")
    return runs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patterns", nargs="+", required=True, help="Glob patterns for threshold-shift predictions.csv files.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--clean-strategy", default="fedavg")
    parser.add_argument("--corrupted-strategy", default="cross_site_blend_50")
    parser.add_argument("--low-quantile", type=float, default=0.10)
    parser.add_argument("--high-quantile", type=float, default=0.90)
    parser.add_argument("--n-classes", type=int, default=6)
    parser.add_argument("--use-entropy", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(
        patterns=args.patterns,
        clean_strategy=args.clean_strategy,
        corrupted_strategy=args.corrupted_strategy,
        n_classes=args.n_classes,
    )

    clean_runs = runs[runs["noise"] == 0]
    thresholds = build_thresholds(clean_runs, low_q=args.low_quantile, high_q=args.high_quantile)
    detected = apply_detector(runs, thresholds, use_entropy=args.use_entropy)
    summary = build_summary(detected)

    run_path = out_dir / "detector_run_diagnostics.csv"
    summary_path = out_dir / "detector_switch_summary.csv"
    threshold_path = out_dir / "detector_thresholds.json"

    detected.to_csv(run_path, index=False)
    summary.to_csv(summary_path, index=False)
    threshold_path.write_text(json.dumps(thresholds, indent=2), encoding="utf-8")

    key_metrics = {"global_qwk", "worst_site_qwk", "macro_f1"}
    print(summary[summary["metric"].isin(key_metrics)].to_string(index=False))
    print(f"Wrote threshold-shift detector run diagnostics to {run_path}")
    print(f"Wrote threshold-shift detector summary to {summary_path}")
    print(f"Wrote threshold-shift detector thresholds to {threshold_path}")


if __name__ == "__main__":
    main()
