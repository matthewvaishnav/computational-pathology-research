#!/usr/bin/env python3
"""Analyze a simple dominance-aware/oracle aggregation switch.

This script compares a clean-regime strategy against a corrupted-regime strategy
using existing per-seed summary.csv files. It is intended for controlled stress
experiments where a known noise level is encoded in the result directory name.

Default behavior:
- choose fedavg when noise == 0
- choose cross_site_blend_50 when noise > 0

This gives an oracle upper bound for a future detector that would decide when
FedAvg's sample-size weighting is unsafe.
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Iterable

import pandas as pd

DEFAULT_METRICS = (
    "global_qwk",
    "worst_site_qwk",
    "mean_site_qwk",
    "global_accuracy",
    "macro_f1",
)


def _parse_noise_seed(path: Path) -> tuple[int, int] | None:
    """Extract noise and seed from paths containing noise_<n>_seed_<s>."""
    match = re.search(r"noise_(\d+)_seed_(\d+)", str(path))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def _t_critical_95(n: int) -> float:
    """Small table for two-sided 95% t critical values.

    Falls back to 1.96 for large n. The common cases here are n=5 and n=15.
    """
    table = {
        2: 12.706,
        3: 4.303,
        4: 3.182,
        5: 2.776,
        6: 2.571,
        7: 2.447,
        8: 2.365,
        9: 2.306,
        10: 2.262,
        11: 2.228,
        12: 2.201,
        13: 2.179,
        14: 2.160,
        15: 2.145,
        16: 2.131,
        17: 2.120,
        18: 2.110,
        19: 2.101,
        20: 2.093,
    }
    return table.get(n, 1.96)


def _iter_summary_files(patterns: Iterable[str]) -> Iterable[Path]:
    for pattern in patterns:
        yield from Path(".").glob(pattern)


def build_switch_tables(
    patterns: list[str],
    clean_strategy: str,
    corrupted_strategy: str,
    metrics: tuple[str, ...],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []

    for path in _iter_summary_files(patterns):
        parsed = _parse_noise_seed(path)
        if parsed is None:
            continue

        noise, seed = parsed
        df = pd.read_csv(path)
        strategies = set(df["strategy"].astype(str))

        if clean_strategy not in strategies or corrupted_strategy not in strategies:
            continue

        clean_row = df[df["strategy"] == clean_strategy].iloc[0]
        corrupted_row = df[df["strategy"] == corrupted_strategy].iloc[0]

        chosen_row = clean_row if noise == 0 else corrupted_row
        chosen_strategy = clean_strategy if noise == 0 else corrupted_strategy

        for metric in metrics:
            if metric not in df.columns:
                raise KeyError(f"Metric '{metric}' missing from {path}")

            clean_value = float(clean_row[metric])
            corrupted_value = float(corrupted_row[metric])
            chosen_value = float(chosen_row[metric])

            rows.append(
                {
                    "noise": noise,
                    "seed": seed,
                    "metric": metric,
                    "clean_strategy": clean_strategy,
                    "corrupted_strategy": corrupted_strategy,
                    "chosen_strategy": chosen_strategy,
                    "clean_value": clean_value,
                    "corrupted_value": corrupted_value,
                    "switch_value": chosen_value,
                    "delta_corrupted_vs_clean": corrupted_value - clean_value,
                    "delta_switch_vs_clean": chosen_value - clean_value,
                }
            )

    long = pd.DataFrame(rows)
    if long.empty:
        raise ValueError("No matching summary.csv files found for the requested patterns.")

    long = long.drop_duplicates(["noise", "seed", "metric"])

    summary_rows: list[dict[str, object]] = []
    for (noise, metric), group in long.groupby(["noise", "metric"]):
        deltas = group["delta_switch_vs_clean"].tolist()
        n = len(deltas)
        mean_delta = sum(deltas) / n
        sd_delta = (
            math.sqrt(sum((x - mean_delta) ** 2 for x in deltas) / (n - 1))
            if n > 1
            else 0.0
        )
        se_delta = sd_delta / math.sqrt(n) if n else 0.0
        tcrit = _t_critical_95(n)
        ci95_low = mean_delta - tcrit * se_delta
        ci95_high = mean_delta + tcrit * se_delta

        summary_rows.append(
            {
                "noise": noise,
                "metric": metric,
                "n_seeds": n,
                "clean_strategy": clean_strategy,
                "corrupted_strategy": corrupted_strategy,
                "clean_mean": group["clean_value"].mean(),
                "corrupted_mean": group["corrupted_value"].mean(),
                "switch_mean": group["switch_value"].mean(),
                "mean_delta_switch_vs_clean": mean_delta,
                "sd_delta": sd_delta,
                "se_delta": se_delta,
                "ci95_low": ci95_low,
                "ci95_high": ci95_high,
                "positive_seed_count": int((group["delta_switch_vs_clean"] > 0).sum()),
                "negative_seed_count": int((group["delta_switch_vs_clean"] < 0).sum()),
                "tie_seed_count": int((group["delta_switch_vs_clean"] == 0).sum()),
                "chosen_strategy": clean_strategy if noise == 0 else corrupted_strategy,
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(["noise", "metric"])
    return long, summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--patterns",
        nargs="+",
        required=True,
        help="Glob patterns for per-seed summary.csv files.",
    )
    parser.add_argument("--output-dir", required=True, help="Directory for output CSV files.")
    parser.add_argument("--clean-strategy", default="fedavg")
    parser.add_argument("--corrupted-strategy", default="cross_site_blend_50")
    parser.add_argument(
        "--metrics",
        nargs="+",
        default=list(DEFAULT_METRICS),
        help="Metrics to compare.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    long, summary = build_switch_tables(
        patterns=args.patterns,
        clean_strategy=args.clean_strategy,
        corrupted_strategy=args.corrupted_strategy,
        metrics=tuple(args.metrics),
    )

    long_path = out_dir / "oracle_switch_seed_deltas_long.csv"
    summary_path = out_dir / "oracle_switch_summary.csv"

    long.to_csv(long_path, index=False)
    summary.to_csv(summary_path, index=False)

    key_metrics = {"global_qwk", "worst_site_qwk", "macro_f1"}
    print(summary[summary["metric"].isin(key_metrics)].to_string(index=False))
    print(f"Wrote seed-level switch table to {long_path}")
    print(f"Wrote switch summary to {summary_path}")


if __name__ == "__main__":
    main()
