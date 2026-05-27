"""
Aggregate FAIR-WEIGHTS-H experiment summary CSV files.

This utility reads per-seed summary.csv files produced by the synthetic,
cross-site, and stress-scenario FAIR-WEIGHTS-H experiments and writes compact
aggregate CSV/JSON reports with mean, standard deviation, and deltas against a
baseline strategy such as FedAvg.

Research-only. This is not clinical validation.

Examples:
    python scripts/experiments/aggregate_fair_weights_h_results.py \
        --pattern "results/fair_weights_h_stress_seed_*/summary.csv" \
        --output-dir results/fair_weights_h_stress_aggregate \
        --baseline fedavg

    python scripts/experiments/aggregate_fair_weights_h_results.py \
        --pattern "results/fair_weights_h_cross_site_seed_*/summary.csv" \
        --output-dir results/fair_weights_h_cross_site_aggregate \
        --baseline fedavg
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev
from typing import Dict, Iterable, List, Mapping

METRIC_COLUMNS = (
    "global_auc",
    "global_accuracy",
    "global_loss",
    "worst_site_auc",
    "worst_site_accuracy",
    "mean_site_auc",
    "weight_entropy",
    "n_eff",
)

PRIMARY_METRICS = (
    "global_auc",
    "worst_site_auc",
    "mean_site_auc",
    "global_accuracy",
)


def parse_seed_from_path(path: Path) -> str:
    """Infer seed label from a path like fair_weights_h_stress_seed_42/summary.csv."""
    text = str(path)
    match = re.search(r"seed[_-](\d+)", text)
    if match:
        return match.group(1)
    return path.parent.name


def safe_float(value: str) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except ValueError:
        return float("nan")


def safe_mean(values: Iterable[float]) -> float:
    valid = [v for v in values if not math.isnan(v)]
    return float(mean(valid)) if valid else float("nan")


def safe_stdev(values: Iterable[float]) -> float:
    valid = [v for v in values if not math.isnan(v)]
    if len(valid) < 2:
        return 0.0 if len(valid) == 1 else float("nan")
    return float(stdev(valid))


def read_summary(path: Path) -> List[Dict[str, object]]:
    seed = parse_seed_from_path(path)
    rows: List[Dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            parsed: Dict[str, object] = {
                "seed": seed,
                "source_path": str(path),
                "strategy": row["strategy"],
            }
            for col in METRIC_COLUMNS:
                parsed[col] = safe_float(row.get(col, ""))
            rows.append(parsed)
    return rows


def collect_rows(patterns: List[str]) -> List[Dict[str, object]]:
    paths: List[Path] = []
    for pattern in patterns:
        matches = sorted(Path(p) for p in glob.glob(pattern))
        paths.extend(matches)

    unique_paths = sorted(set(paths))
    if not unique_paths:
        raise FileNotFoundError(f"No summary files matched patterns: {patterns}")

    rows: List[Dict[str, object]] = []
    for path in unique_paths:
        rows.extend(read_summary(path))
    return rows


def index_by_seed_strategy(rows: List[Dict[str, object]]) -> Dict[str, Dict[str, Dict[str, object]]]:
    indexed: Dict[str, Dict[str, Dict[str, object]]] = defaultdict(dict)
    for row in rows:
        seed = str(row["seed"])
        strategy = str(row["strategy"])
        indexed[seed][strategy] = row
    return indexed


def aggregate(rows: List[Dict[str, object]], baseline: str) -> List[Dict[str, object]]:
    by_strategy: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_strategy[str(row["strategy"])].append(row)

    by_seed = index_by_seed_strategy(rows)
    aggregate_rows: List[Dict[str, object]] = []

    for strategy in sorted(by_strategy):
        strategy_rows = by_strategy[strategy]
        out: Dict[str, object] = {
            "strategy": strategy,
            "n_seeds": len(strategy_rows),
        }
        for metric in METRIC_COLUMNS:
            values = [float(row[metric]) for row in strategy_rows]
            out[f"{metric}_mean"] = safe_mean(values)
            out[f"{metric}_std"] = safe_stdev(values)

        deltas_by_metric: Dict[str, List[float]] = {metric: [] for metric in PRIMARY_METRICS}
        wins_by_metric: Dict[str, int] = {metric: 0 for metric in PRIMARY_METRICS}
        comparable_seeds = 0

        for seed, strategies_for_seed in by_seed.items():
            if strategy not in strategies_for_seed or baseline not in strategies_for_seed:
                continue
            comparable_seeds += 1
            current = strategies_for_seed[strategy]
            base = strategies_for_seed[baseline]
            for metric in PRIMARY_METRICS:
                delta = float(current[metric]) - float(base[metric])
                deltas_by_metric[metric].append(delta)
                if delta > 0:
                    wins_by_metric[metric] += 1

        out["n_comparable_seeds"] = comparable_seeds
        for metric in PRIMARY_METRICS:
            deltas = deltas_by_metric[metric]
            out[f"delta_vs_{baseline}_{metric}_mean"] = safe_mean(deltas)
            out[f"delta_vs_{baseline}_{metric}_std"] = safe_stdev(deltas)
            out[f"wins_vs_{baseline}_{metric}"] = wins_by_metric[metric]

        aggregate_rows.append(out)

    return aggregate_rows


def write_aggregate_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    if not rows:
        raise ValueError("No aggregate rows to write")
    fieldnames = list(rows[0].keys())
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_long_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    fieldnames = ["seed", "strategy", *METRIC_COLUMNS, "source_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def best_strategy_by_metric(aggregate_rows: List[Dict[str, object]], metric: str, lower_is_better: bool = False) -> Mapping[str, object]:
    key = f"{metric}_mean"
    valid = [row for row in aggregate_rows if not math.isnan(float(row[key]))]
    if not valid:
        return {}
    return min(valid, key=lambda r: float(r[key])) if lower_is_better else max(valid, key=lambda r: float(r[key]))


def make_report(aggregate_rows: List[Dict[str, object]], baseline: str) -> Dict[str, object]:
    best_global_auc = best_strategy_by_metric(aggregate_rows, "global_auc")
    best_worst_auc = best_strategy_by_metric(aggregate_rows, "worst_site_auc")
    best_accuracy = best_strategy_by_metric(aggregate_rows, "global_accuracy")

    report = {
        "baseline": baseline,
        "best_global_auc_strategy": best_global_auc,
        "best_worst_site_auc_strategy": best_worst_auc,
        "best_global_accuracy_strategy": best_accuracy,
        "interpretation_template": (
            "Use this report to identify whether a candidate weighting strategy improves "
            "global AUC and worst-site AUC relative to the baseline. Synthetic stress-test "
            "results are hypothesis evidence only, not clinical validation."
        ),
    }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate FAIR-WEIGHTS-H experiment summaries")
    parser.add_argument(
        "--pattern",
        action="append",
        required=True,
        help="Glob pattern for summary.csv files. Can be passed multiple times.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=str, default="fedavg")
    args = parser.parse_args()

    rows = collect_rows(args.pattern)
    aggregate_rows = aggregate(rows, baseline=args.baseline)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_csv = args.output_dir / "aggregate_summary.csv"
    long_csv = args.output_dir / "per_seed_long.csv"
    report_json = args.output_dir / "aggregate_report.json"

    write_aggregate_csv(aggregate_rows, aggregate_csv)
    write_long_csv(rows, long_csv)
    report = make_report(aggregate_rows, baseline=args.baseline)
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Read {len(rows)} strategy rows")
    print(f"Wrote aggregate summary to {aggregate_csv}")
    print(f"Wrote per-seed long table to {long_csv}")
    print(f"Wrote JSON report to {report_json}")

    best_global = report["best_global_auc_strategy"]
    best_worst = report["best_worst_site_auc_strategy"]
    if best_global:
        print(
            "Best global AUC: "
            f"{best_global['strategy']} "
            f"mean={float(best_global['global_auc_mean']):.4f}"
        )
    if best_worst:
        print(
            "Best worst-site AUC: "
            f"{best_worst['strategy']} "
            f"mean={float(best_worst['worst_site_auc_mean']):.4f}"
        )


if __name__ == "__main__":
    main()
