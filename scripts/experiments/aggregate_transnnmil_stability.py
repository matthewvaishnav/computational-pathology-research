#!/usr/bin/env python3
"""
Aggregate PANDA TransnnMIL stabilization grid results.

This script summarizes repeated-seed optimizer-stability runs produced by:

    scripts/training/train_panda_transnnmil_baseline.py

Example:
    python scripts/experiments/aggregate_transnnmil_stability.py \
        --results-dir results \
        --pattern "transnnmil_stability_warmup_cosine_clip_lr_*_seed_*" \
        --out-dir results/transnnmil_stability_summary

Outputs:
    - per_run_metrics.csv
    - by_lr_summary.csv
    - transnnmil_stability_qwk_by_lr.png
    - transnnmil_stability_best_epoch_by_lr.png
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import pandas as pd

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:  # pragma: no cover - plotting is optional at runtime
    plt = None


LR_SEED_RE = re.compile(r"lr_(?P<lr>[^_]+)_seed_(?P<seed>\d+)")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aggregate TransnnMIL stability grid metrics")
    parser.add_argument("--results-dir", default="results", help="Root directory containing run output folders")
    parser.add_argument(
        "--pattern",
        default="transnnmil_stability_warmup_cosine_clip_lr_*_seed_*",
        help="Glob pattern, relative to --results-dir, for run folders",
    )
    parser.add_argument("--out-dir", default="results/transnnmil_stability_summary", help="Directory for summary outputs")
    parser.add_argument("--sort-by", default="best_val_qwk_mean", help="Summary column to sort descending for console output")
    return parser.parse_args()


def as_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(result) or math.isinf(result):
        return None
    return result


def as_int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def parse_lr_seed(run_dir: Path, metrics: Dict[str, Any]) -> tuple[float | None, int | None]:
    config = metrics.get("config", {}) if isinstance(metrics.get("config"), dict) else {}
    lr = as_float(config.get("lr"))
    seed = as_int(config.get("seed"))

    match = LR_SEED_RE.search(run_dir.name)
    if match:
        lr = lr if lr is not None else as_float(match.group("lr"))
        seed = seed if seed is not None else as_int(match.group("seed"))
    return lr, seed


def best_epoch_metrics(history: Iterable[Dict[str, Any]], best_epoch: int | None) -> Dict[str, float | None]:
    if best_epoch is None:
        return {"best_val_loss": None, "best_val_accuracy": None, "best_val_macro_f1": None}

    for row in history:
        if as_int(row.get("epoch")) != best_epoch:
            continue
        val = row.get("val", {}) if isinstance(row.get("val"), dict) else {}
        return {
            "best_val_loss": as_float(val.get("loss")),
            "best_val_accuracy": as_float(val.get("accuracy")),
            "best_val_macro_f1": as_float(val.get("macro_f1")),
        }
    return {"best_val_loss": None, "best_val_accuracy": None, "best_val_macro_f1": None}


def read_run(run_dir: Path) -> Dict[str, Any] | None:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        print(f"Skipping {run_dir}: metrics.json not found")
        return None

    with metrics_path.open("r", encoding="utf-8") as handle:
        metrics = json.load(handle)

    config = metrics.get("config", {}) if isinstance(metrics.get("config"), dict) else {}
    dataset = metrics.get("dataset", {}) if isinstance(metrics.get("dataset"), dict) else {}
    final = metrics.get("final_val_metrics", {}) if isinstance(metrics.get("final_val_metrics"), dict) else {}
    history = metrics.get("history", []) if isinstance(metrics.get("history"), list) else []

    lr, seed = parse_lr_seed(run_dir, metrics)
    best_epoch = as_int(metrics.get("best_epoch"))
    best_metrics = best_epoch_metrics(history, best_epoch)

    first_epoch = history[0] if history else {}
    last_epoch = history[-1] if history else {}
    first_train = first_epoch.get("train", {}) if isinstance(first_epoch.get("train"), dict) else {}
    last_train = last_epoch.get("train", {}) if isinstance(last_epoch.get("train"), dict) else {}

    return {
        "run_dir": str(run_dir),
        "run_name": run_dir.name,
        "lr": lr,
        "seed": seed,
        "best_val_qwk": as_float(metrics.get("best_val_qwk")),
        "best_epoch": best_epoch,
        "epochs_ran": as_int(metrics.get("epochs_ran")),
        "final_val_qwk": as_float(final.get("qwk")),
        "final_val_accuracy": as_float(final.get("accuracy")),
        "final_val_macro_f1": as_float(final.get("macro_f1")),
        "scheduler": config.get("scheduler"),
        "warmup_epochs": as_int(config.get("warmup_epochs")),
        "min_lr": as_float(config.get("min_lr")),
        "grad_clip_norm": as_float(config.get("grad_clip_norm")),
        "early_stopping_patience": as_int(config.get("early_stopping_patience")),
        "early_stopping_min_delta": as_float(config.get("early_stopping_min_delta")),
        "batch_size": as_int(config.get("batch_size")),
        "weight_decay": as_float(config.get("weight_decay")),
        "dropout": as_float(config.get("dropout")),
        "hidden_dim": as_int(config.get("hidden_dim")),
        "num_layers": as_int(config.get("num_layers")),
        "num_heads": as_int(config.get("num_heads")),
        "feature_dim": as_int(config.get("feature_dim")),
        "max_patches": as_int(config.get("max_patches")),
        "manifest_rows_used": as_int(dataset.get("manifest_rows_used")),
        "train_rows": as_int(dataset.get("train_rows")),
        "val_rows": as_int(dataset.get("val_rows")),
        "first_epoch_grad_norm_before_clip": as_float(first_train.get("mean_grad_norm_before_clip")),
        "last_epoch_grad_norm_before_clip": as_float(last_train.get("mean_grad_norm_before_clip")),
        **best_metrics,
    }


def summarize_by_lr(per_run: pd.DataFrame) -> pd.DataFrame:
    grouped = per_run.groupby("lr", dropna=False)
    summary = grouped.agg(
        runs=("best_val_qwk", "count"),
        seeds=("seed", lambda values: ",".join(str(int(v)) for v in sorted(values.dropna().unique()))),
        best_val_qwk_mean=("best_val_qwk", "mean"),
        best_val_qwk_std=("best_val_qwk", "std"),
        best_val_qwk_min=("best_val_qwk", "min"),
        best_val_qwk_max=("best_val_qwk", "max"),
        best_epoch_mean=("best_epoch", "mean"),
        best_epoch_std=("best_epoch", "std"),
        epochs_ran_mean=("epochs_ran", "mean"),
        final_val_qwk_mean=("final_val_qwk", "mean"),
        final_val_accuracy_mean=("final_val_accuracy", "mean"),
        final_val_macro_f1_mean=("final_val_macro_f1", "mean"),
    ).reset_index()

    summary["best_val_qwk_std"] = summary["best_val_qwk_std"].fillna(0.0)
    summary["best_epoch_std"] = summary["best_epoch_std"].fillna(0.0)
    return summary.sort_values("lr").reset_index(drop=True)


def plot_metric(per_run: pd.DataFrame, out_path: Path, metric: str, ylabel: str, title: str) -> None:
    if plt is None:
        print(f"matplotlib not installed; skipping plot: {out_path}")
        return

    plot_df = per_run.dropna(subset=["lr", metric]).copy()
    if plot_df.empty:
        print(f"No data available for plot: {metric}")
        return

    lrs = sorted(plot_df["lr"].unique())
    x_positions = list(range(len(lrs)))
    by_lr = [plot_df.loc[plot_df["lr"] == lr, metric].tolist() for lr in lrs]
    means = [sum(values) / len(values) if values else float("nan") for values in by_lr]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.boxplot(by_lr, positions=x_positions, widths=0.55, showmeans=True)
    ax.plot(x_positions, means, marker="o", label="mean")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{lr:g}" for lr in lrs], rotation=30, ha="right")
    ax.set_xlabel("Learning rate")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_dirs = sorted(path for path in results_dir.glob(args.pattern) if path.is_dir())
    if not run_dirs:
        raise FileNotFoundError(f"No run directories matched: {results_dir / args.pattern}")

    rows: List[Dict[str, Any]] = []
    for run_dir in run_dirs:
        row = read_run(run_dir)
        if row is not None:
            rows.append(row)

    if not rows:
        raise RuntimeError("No readable metrics.json files found")

    per_run = pd.DataFrame(rows).sort_values(["lr", "seed", "run_name"], na_position="last").reset_index(drop=True)
    summary = summarize_by_lr(per_run)

    per_run_path = out_dir / "per_run_metrics.csv"
    summary_path = out_dir / "by_lr_summary.csv"
    per_run.to_csv(per_run_path, index=False)
    summary.to_csv(summary_path, index=False)

    plot_metric(
        per_run,
        out_dir / "transnnmil_stability_qwk_by_lr.png",
        metric="best_val_qwk",
        ylabel="Best validation QWK",
        title="TransnnMIL stabilization: best validation QWK by learning rate",
    )
    plot_metric(
        per_run,
        out_dir / "transnnmil_stability_best_epoch_by_lr.png",
        metric="best_epoch",
        ylabel="Best epoch",
        title="TransnnMIL stabilization: best epoch by learning rate",
    )

    sort_col = args.sort_by if args.sort_by in summary.columns else "best_val_qwk_mean"
    printable = summary.sort_values(sort_col, ascending=False).copy()
    numeric_cols = printable.select_dtypes(include="number").columns
    printable[numeric_cols] = printable[numeric_cols].round(6)

    print(f"\nWrote: {per_run_path}")
    print(f"Wrote: {summary_path}")
    print("\nBy-LR summary sorted by", sort_col)
    print(printable.to_string(index=False))

    best = printable.iloc[0]
    print(
        "\nBest mean best-val QWK: "
        f"lr={best['lr']:g}, mean={best['best_val_qwk_mean']:.4f}, "
        f"std={best['best_val_qwk_std']:.4f}, runs={int(best['runs'])}"
    )


if __name__ == "__main__":
    main()
