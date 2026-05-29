#!/usr/bin/env python3
"""
Aggregate PANDA centralized vs local-only vs federated benchmark summaries.

Reads summary.csv files produced by:

    scripts/experiments/run_panda_centralized_vs_federated.py

and writes aggregate mean/std/delta tables across seeds.

Research-only. This is simulated-federation benchmark evidence, not clinical
validation and not diagnostic software.

Example:
    python scripts/experiments/aggregate_panda_centralized_vs_federated.py \
        --pattern "results/panda_centralized_vs_federated_1000_seed_*/summary.csv" \
        --output-dir results/panda_centralized_vs_federated_1000_aggregate \
        --baseline centralized_all
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

METRICS = (
    "global_qwk",
    "global_accuracy",
    "macro_f1",
    "global_loss",
    "worst_site_qwk",
    "worst_site_accuracy",
    "mean_site_qwk",
)

PRIMARY_METRICS = (
    "global_qwk",
    "global_accuracy",
    "macro_f1",
    "worst_site_qwk",
    "mean_site_qwk",
)


def parse_seed_from_path(path: Path) -> str:
    match = re.search(r"seed[_-](\d+)", str(path))
    if match:
        return match.group(1)
    return path.parent.name


def safe_float(value: object) -> float:
    if value is None or value == "":
        return float("nan")
    try:
        return float(value)
    except (TypeError, ValueError):
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
        if not reader.fieldnames or "regime" not in reader.fieldnames:
            raise ValueError(f"Summary file missing regime column: {path}")
        for row in reader:
            parsed: Dict[str, object] = {
                "seed": seed,
                "source_path": str(path),
                "regime": row["regime"],
                "train_description": row.get("train_description", ""),
            }
            for metric in METRICS:
                parsed[metric] = safe_float(row.get(metric, ""))
            rows.append(parsed)
    return rows


def collect_rows(patterns: List[str]) -> List[Dict[str, object]]:
    paths: List[Path] = []
    for pattern in patterns:
        paths.extend(Path(p) for p in glob.glob(pattern))
    paths = sorted(set(paths))
    if not paths:
        raise FileNotFoundError(f"No summary files matched patterns: {patterns}")
    rows: List[Dict[str, object]] = []
    for path in paths:
        rows.extend(read_summary(path))
    return rows


def index_by_seed_regime(rows: List[Dict[str, object]]) -> Dict[str, Dict[str, Dict[str, object]]]:
    indexed: Dict[str, Dict[str, Dict[str, object]]] = defaultdict(dict)
    for row in rows:
        indexed[str(row["seed"])][str(row["regime"])] = row
    return indexed


def aggregate(rows: List[Dict[str, object]], baseline: str) -> List[Dict[str, object]]:
    by_regime: Dict[str, List[Dict[str, object]]] = defaultdict(list)
    for row in rows:
        by_regime[str(row["regime"])].append(row)

    by_seed = index_by_seed_regime(rows)
    aggregate_rows: List[Dict[str, object]] = []

    for regime in sorted(by_regime):
        regime_rows = by_regime[regime]
        out: Dict[str, object] = {
            "regime": regime,
            "n_seeds": len(regime_rows),
        }
        for metric in METRICS:
            values = [float(row[metric]) for row in regime_rows]
            out[f"{metric}_mean"] = safe_mean(values)
            out[f"{metric}_std"] = safe_stdev(values)

        comparable = 0
        for metric in PRIMARY_METRICS:
            deltas: List[float] = []
            wins = 0
            for _, regimes_for_seed in by_seed.items():
                if regime not in regimes_for_seed or baseline not in regimes_for_seed:
                    continue
                current = regimes_for_seed[regime]
                base = regimes_for_seed[baseline]
                delta = float(current[metric]) - float(base[metric])
                if math.isnan(delta):
                    continue
                deltas.append(delta)
                if delta > 0:
                    wins += 1
            out[f"delta_vs_{baseline}_{metric}_mean"] = safe_mean(deltas)
            out[f"delta_vs_{baseline}_{metric}_std"] = safe_stdev(deltas)
            out[f"wins_vs_{baseline}_{metric}"] = wins
            comparable = max(comparable, len(deltas))
        out["n_comparable_seeds"] = comparable
        aggregate_rows.append(out)
    return aggregate_rows


def aggregate_local_family(rows: List[Dict[str, object]]) -> Dict[str, object]:
    local_rows = [row for row in rows if str(row["regime"]).startswith("local_site_")]
    out: Dict[str, object] = {"family": "local_site_mean", "n_rows": len(local_rows)}
    for metric in METRICS:
        values = [float(row[metric]) for row in local_rows]
        out[f"{metric}_mean"] = safe_mean(values)
        out[f"{metric}_std"] = safe_stdev(values)
    return out


def best_by_metric(aggregate_rows: List[Dict[str, object]], metric: str, lower_is_better: bool = False) -> Mapping[str, object]:
    key = f"{metric}_mean"
    valid = [row for row in aggregate_rows if key in row and not math.isnan(float(row[key]))]
    if not valid:
        return {}
    return min(valid, key=lambda r: float(r[key])) if lower_is_better else max(valid, key=lambda r: float(r[key]))


def write_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    if not rows:
        raise ValueError("No rows to write")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_long_csv(rows: List[Dict[str, object]], output_path: Path) -> None:
    fieldnames = ["seed", "regime", *METRICS, "train_description", "source_path"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fieldnames})


def main() -> None:
    parser = argparse.ArgumentParser(description="Aggregate PANDA centralized vs federated benchmark summaries")
    parser.add_argument("--pattern", action="append", required=True, help="Glob pattern for summary.csv files")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=str, default="centralized_all")
    args = parser.parse_args()

    rows = collect_rows(args.pattern)
    aggregate_rows = aggregate(rows, baseline=args.baseline)
    local_family = aggregate_local_family(rows)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    aggregate_csv = args.output_dir / "aggregate_summary.csv"
    long_csv = args.output_dir / "per_seed_long.csv"
    report_json = args.output_dir / "aggregate_report.json"

    write_csv(aggregate_rows, aggregate_csv)
    write_long_csv(rows, long_csv)

    report = {
        "baseline": args.baseline,
        "best_by_metric": {metric: best_by_metric(aggregate_rows, metric) for metric in PRIMARY_METRICS},
        "local_family": local_family,
        "interpretation_template": (
            "Compare centralized_all, fedavg, and local_site_* regimes. This is a "
            "PANDA-derived simulated-federation benchmark, not real multi-center clinical validation."
        ),
    }
    report_json.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Read {len(rows)} regime rows")
    print(f"Wrote aggregate summary to {aggregate_csv}")
    print(f"Wrote per-seed long table to {long_csv}")
    print(f"Wrote JSON report to {report_json}")

    for metric in PRIMARY_METRICS:
        best = report["best_by_metric"].get(metric, {})
        if best:
            print(f"Best {metric}: {best['regime']} mean={float(best[f'{metric}_mean']):.4f}")

    print(
        "Local-only family mean global_qwk="
        f"{float(local_family['global_qwk_mean']):.4f}, "
        "global_accuracy="
        f"{float(local_family['global_accuracy_mean']):.4f}, "
        "macro_f1="
        f"{float(local_family['macro_f1_mean']):.4f}"
    )


if __name__ == "__main__":
    main()
