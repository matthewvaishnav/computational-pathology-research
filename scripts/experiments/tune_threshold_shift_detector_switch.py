#!/usr/bin/env python3
"""Tune dominance-detector transfer on ordinal threshold-shift runs.

This is the threshold-shift companion to tune_dominance_detector_switch.py.
It parses threshold-shift result paths, then reuses the same clean-calibrated
rule search:

- calibrate FedAvg diagnostics on threshold_shift=0 runs
- search low/high quantiles and minimum trigger counts
- choose a simple detector that balances clean-regime false switching and
  corrupted-regime gains

The output schema keeps the column name `noise` for compatibility with existing
summary tooling; here it means threshold-shift percent.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent))

from analyze_threshold_shift_detector_switch import load_runs  # noqa: E402
from tune_dominance_detector_switch import (  # noqa: E402
    parse_float_list,
    parse_int_list,
    tune,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patterns", nargs="+", required=True, help="Glob patterns for threshold-shift predictions.csv files.")
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
        patterns=args.patterns,
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

    print("Top threshold-shift detector configs:")
    print(grid.head(10).to_string(index=False))
    print("\nBest threshold-shift detector summary:")
    print(best_summary[best_summary["metric"].isin({"global_qwk", "worst_site_qwk", "macro_f1"})].to_string(index=False))
    print(f"\nWrote tuned threshold-shift detector outputs to {out_dir}")


if __name__ == "__main__":
    main()
