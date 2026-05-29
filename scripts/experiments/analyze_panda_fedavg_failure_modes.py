#!/usr/bin/env python3
"""
Analyze PANDA-derived FAIR-WEIGHTS-H prediction logs to identify where FedAvg
fails and where contribution-aware strategies help.

Inputs are prediction CSV files produced by:
    scripts/experiments/run_fair_weights_h_panda_feature_stress.py --save-predictions

The analyzer writes:
    per_site_metrics.csv
    per_grade_metrics.csv
    site_grade_recall.csv
    confusion_matrices.json
    fedavg_vs_strategy_deltas.csv
    failure_mode_summary.json

This is simulated-federation failure forensics on PANDA-derived feature caches.
It is not clinical validation and is not diagnostic software.
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score, precision_recall_fscore_support


METRIC_COLUMNS = ["qwk", "accuracy", "macro_f1", "support"]


def infer_noise_seed_from_path(path: Path) -> Tuple[float | None, int | None]:
    """Best-effort parsing for directories like noise_25_seed_42."""
    text = str(path).replace("\\", "/")
    noise = None
    seed = None
    parts = text.split("/")
    for part in parts:
        tokens = part.split("_")
        if "noise" in tokens:
            try:
                i = tokens.index("noise")
                noise = float(tokens[i + 1]) / 100.0 if float(tokens[i + 1]) > 1 else float(tokens[i + 1])
            except Exception:  # noqa: BLE001
                pass
        if "seed" in tokens:
            try:
                i = tokens.index("seed")
                seed = int(tokens[i + 1])
            except Exception:  # noqa: BLE001
                pass
    return noise, seed


def safe_qwk(y_true: Iterable[int], y_pred: Iterable[int]) -> float:
    y_true = list(y_true)
    y_pred = list(y_pred)
    if len(set(y_true)) <= 1 and len(set(y_pred)) <= 1:
        return 1.0 if y_true == y_pred else 0.0
    value = cohen_kappa_score(y_true, y_pred, weights="quadratic")
    return float(0.0 if np.isnan(value) else value)


def compute_group_metrics(frame: pd.DataFrame, group_cols: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    for keys, group in frame.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        y_true = group["isup_grade_true"].astype(int).to_numpy()
        y_pred = group["isup_grade_pred"].astype(int).to_numpy()
        row = {col: key for col, key in zip(group_cols, keys)}
        row.update(
            {
                "qwk": safe_qwk(y_true, y_pred),
                "accuracy": float(accuracy_score(y_true, y_pred)),
                "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
                "support": int(len(group)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def compute_per_grade_metrics(frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    group_cols = ["noise_level", "seed", "strategy"]
    for keys, group in frame.groupby(group_cols, dropna=False):
        y_true = group["isup_grade_true"].astype(int).to_numpy()
        y_pred = group["isup_grade_pred"].astype(int).to_numpy()
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true,
            y_pred,
            labels=list(range(6)),
            zero_division=0,
        )
        base = {col: key for col, key in zip(group_cols, keys)}
        for grade in range(6):
            rows.append(
                {
                    **base,
                    "grade": grade,
                    "precision": float(precision[grade]),
                    "recall": float(recall[grade]),
                    "f1": float(f1[grade]),
                    "support": int(support[grade]),
                }
            )
        high_true = np.isin(y_true, [4, 5]).astype(int)
        high_pred = np.isin(y_pred, [4, 5]).astype(int)
        high_precision, high_recall, high_f1, high_support = precision_recall_fscore_support(
            high_true,
            high_pred,
            labels=[1],
            zero_division=0,
        )
        rows.append(
            {
                **base,
                "grade": "high_grade_4_5",
                "precision": float(high_precision[0]),
                "recall": float(high_recall[0]),
                "f1": float(high_f1[0]),
                "support": int(high_support[0]),
            }
        )
    return pd.DataFrame(rows)


def compute_site_grade_recall(frame: pd.DataFrame) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    group_cols = ["noise_level", "seed", "strategy", "site_id"]
    for keys, group in frame.groupby(group_cols, dropna=False):
        y_true = group["isup_grade_true"].astype(int).to_numpy()
        y_pred = group["isup_grade_pred"].astype(int).to_numpy()
        base = {col: key for col, key in zip(group_cols, keys)}
        for grade in range(6):
            mask = y_true == grade
            rows.append(
                {
                    **base,
                    "grade": grade,
                    "recall": float((y_pred[mask] == grade).mean()) if mask.any() else np.nan,
                    "support": int(mask.sum()),
                }
            )
        high_mask = np.isin(y_true, [4, 5])
        rows.append(
            {
                **base,
                "grade": "high_grade_4_5",
                "recall": float(np.isin(y_pred[high_mask], [4, 5]).mean()) if high_mask.any() else np.nan,
                "support": int(high_mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def compute_confusion_matrices(frame: pd.DataFrame) -> Dict[str, object]:
    matrices: Dict[str, object] = {}
    for keys, group in frame.groupby(["noise_level", "seed", "strategy"], dropna=False):
        noise, seed, strategy = keys
        y_true = group["isup_grade_true"].astype(int).to_numpy()
        y_pred = group["isup_grade_pred"].astype(int).to_numpy()
        key = f"noise={noise}|seed={seed}|strategy={strategy}"
        matrices[key] = confusion_matrix(y_true, y_pred, labels=list(range(6))).astype(int).tolist()
    return matrices


def aggregate_deltas(metric_frame: pd.DataFrame, group_cols: List[str], metrics: List[str]) -> pd.DataFrame:
    rows: List[Dict[str, object]] = []
    id_cols = [col for col in group_cols if col != "strategy"]
    for keys, group in metric_frame.groupby(id_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = {col: key for col, key in zip(id_cols, keys)}
        fedavg = group[group["strategy"] == "fedavg"]
        if fedavg.empty:
            continue
        fedavg_row = fedavg.iloc[0]
        for _, row in group.iterrows():
            if row["strategy"] == "fedavg":
                continue
            out = {**base, "strategy": row["strategy"]}
            for metric in metrics:
                out[f"delta_{metric}_vs_fedavg"] = float(row[metric] - fedavg_row[metric])
                out[f"fedavg_{metric}"] = float(fedavg_row[metric])
                out[f"strategy_{metric}"] = float(row[metric])
            rows.append(out)
    return pd.DataFrame(rows)


def make_failure_summary(
    global_metrics: pd.DataFrame,
    per_site_metrics: pd.DataFrame,
    per_grade_metrics: pd.DataFrame,
    site_grade_recall: pd.DataFrame,
) -> Dict[str, object]:
    summary: Dict[str, object] = {
        "clinical_status": "PANDA-derived simulated federation; not clinical validation; not diagnostic software",
        "best_by_noise": {},
        "fedavg_failure_signals": [],
        "high_grade_recall_findings": [],
        "worst_site_findings": [],
    }

    agg = global_metrics.groupby(["noise_level", "strategy"], dropna=False)[["qwk", "accuracy", "macro_f1"]].mean().reset_index()
    for noise, group in agg.groupby("noise_level", dropna=False):
        summary["best_by_noise"][str(noise)] = {
            "best_global_qwk": str(group.sort_values("qwk", ascending=False).iloc[0]["strategy"]),
            "best_accuracy": str(group.sort_values("accuracy", ascending=False).iloc[0]["strategy"]),
            "best_macro_f1": str(group.sort_values("macro_f1", ascending=False).iloc[0]["strategy"]),
        }

    site_agg = per_site_metrics.groupby(["noise_level", "seed", "strategy"], dropna=False)["qwk"].min().reset_index()
    site_agg = site_agg.rename(columns={"qwk": "worst_site_qwk"})
    worst_delta = aggregate_deltas(site_agg, ["noise_level", "seed", "strategy"], ["worst_site_qwk"])
    if not worst_delta.empty:
        best = worst_delta.groupby(["noise_level", "strategy"], dropna=False)["delta_worst_site_qwk_vs_fedavg"].mean().reset_index()
        for _, row in best.sort_values("delta_worst_site_qwk_vs_fedavg", ascending=False).head(10).iterrows():
            summary["worst_site_findings"].append(
                {
                    "noise_level": float(row["noise_level"]),
                    "strategy": str(row["strategy"]),
                    "mean_delta_worst_site_qwk_vs_fedavg": float(row["delta_worst_site_qwk_vs_fedavg"]),
                }
            )

    high = per_grade_metrics[per_grade_metrics["grade"].astype(str) == "high_grade_4_5"].copy()
    if not high.empty:
        high_delta = aggregate_deltas(high, ["noise_level", "seed", "strategy", "grade"], ["recall", "f1"])
        high_best = high_delta.groupby(["noise_level", "strategy"], dropna=False)[
            ["delta_recall_vs_fedavg", "delta_f1_vs_fedavg"]
        ].mean().reset_index()
        for _, row in high_best.sort_values("delta_recall_vs_fedavg", ascending=False).head(10).iterrows():
            summary["high_grade_recall_findings"].append(
                {
                    "noise_level": float(row["noise_level"]),
                    "strategy": str(row["strategy"]),
                    "mean_delta_high_grade_recall_vs_fedavg": float(row["delta_recall_vs_fedavg"]),
                    "mean_delta_high_grade_f1_vs_fedavg": float(row["delta_f1_vs_fedavg"]),
                }
            )

    global_delta = aggregate_deltas(global_metrics, ["noise_level", "seed", "strategy"], ["qwk", "accuracy", "macro_f1"])
    if not global_delta.empty:
        global_best = global_delta.groupby(["noise_level", "strategy"], dropna=False)[
            ["delta_qwk_vs_fedavg", "delta_accuracy_vs_fedavg", "delta_macro_f1_vs_fedavg"]
        ].mean().reset_index()
        for _, row in global_best.sort_values("delta_qwk_vs_fedavg", ascending=False).head(12).iterrows():
            summary["fedavg_failure_signals"].append(
                {
                    "noise_level": float(row["noise_level"]),
                    "strategy": str(row["strategy"]),
                    "mean_delta_global_qwk_vs_fedavg": float(row["delta_qwk_vs_fedavg"]),
                    "mean_delta_accuracy_vs_fedavg": float(row["delta_accuracy_vs_fedavg"]),
                    "mean_delta_macro_f1_vs_fedavg": float(row["delta_macro_f1_vs_fedavg"]),
                }
            )

    return summary


def load_predictions(pattern: str) -> pd.DataFrame:
    paths = sorted(Path(p) for p in glob.glob(pattern))
    if not paths:
        raise FileNotFoundError(f"No prediction files matched pattern: {pattern}")
    frames: List[pd.DataFrame] = []
    for path in paths:
        frame = pd.read_csv(path)
        noise, seed = infer_noise_seed_from_path(path)
        if "noise_level" not in frame.columns or frame["noise_level"].isna().all():
            frame["noise_level"] = noise
        if "seed" not in frame.columns or frame["seed"].isna().all():
            frame["seed"] = seed
        frame["source_file"] = str(path)
        frames.append(frame)
    combined = pd.concat(frames, axis=0, ignore_index=True)
    combined["noise_level"] = combined["noise_level"].astype(float)
    combined["seed"] = combined["seed"].astype(int)
    combined["site_id"] = combined["site_id"].astype(int)
    combined["isup_grade_true"] = combined["isup_grade_true"].astype(int)
    combined["isup_grade_pred"] = combined["isup_grade_pred"].astype(int)
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze PANDA FedAvg failure modes from prediction logs")
    parser.add_argument("--pattern", required=True, help="Glob pattern for predictions.csv files")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    predictions = load_predictions(args.pattern)

    global_metrics = compute_group_metrics(predictions, ["noise_level", "seed", "strategy"])
    per_site_metrics = compute_group_metrics(predictions, ["noise_level", "seed", "strategy", "site_id"])
    per_grade_metrics = compute_per_grade_metrics(predictions)
    site_grade_recall = compute_site_grade_recall(predictions)
    confusion_matrices = compute_confusion_matrices(predictions)

    global_deltas = aggregate_deltas(global_metrics, ["noise_level", "seed", "strategy"], ["qwk", "accuracy", "macro_f1"])
    worst_site = per_site_metrics.groupby(["noise_level", "seed", "strategy"], dropna=False)["qwk"].min().reset_index()
    worst_site = worst_site.rename(columns={"qwk": "worst_site_qwk"})
    worst_site_deltas = aggregate_deltas(worst_site, ["noise_level", "seed", "strategy"], ["worst_site_qwk"])

    summary = make_failure_summary(global_metrics, per_site_metrics, per_grade_metrics, site_grade_recall)
    summary["input_prediction_files"] = int(predictions["source_file"].nunique())
    summary["prediction_rows"] = int(len(predictions))
    summary["noise_levels"] = sorted(float(v) for v in predictions["noise_level"].unique())
    summary["seeds"] = sorted(int(v) for v in predictions["seed"].unique())
    summary["strategies"] = sorted(str(v) for v in predictions["strategy"].unique())

    global_metrics.to_csv(args.output_dir / "global_metrics.csv", index=False)
    per_site_metrics.to_csv(args.output_dir / "per_site_metrics.csv", index=False)
    per_grade_metrics.to_csv(args.output_dir / "per_grade_metrics.csv", index=False)
    site_grade_recall.to_csv(args.output_dir / "site_grade_recall.csv", index=False)
    global_deltas.to_csv(args.output_dir / "fedavg_vs_strategy_deltas.csv", index=False)
    worst_site_deltas.to_csv(args.output_dir / "fedavg_vs_strategy_worst_site_deltas.csv", index=False)
    (args.output_dir / "confusion_matrices.json").write_text(json.dumps(confusion_matrices, indent=2), encoding="utf-8")
    (args.output_dir / "failure_mode_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Read {predictions['source_file'].nunique()} prediction files")
    print(f"Read {len(predictions)} prediction rows")
    print(f"Wrote global metrics to {args.output_dir / 'global_metrics.csv'}")
    print(f"Wrote per-site metrics to {args.output_dir / 'per_site_metrics.csv'}")
    print(f"Wrote per-grade metrics to {args.output_dir / 'per_grade_metrics.csv'}")
    print(f"Wrote site-grade recall to {args.output_dir / 'site_grade_recall.csv'}")
    print(f"Wrote JSON summary to {args.output_dir / 'failure_mode_summary.json'}")

    print("\nTop FedAvg failure signals by mean global QWK delta:")
    if not global_deltas.empty:
        top = global_deltas.groupby(["noise_level", "strategy"], dropna=False)["delta_qwk_vs_fedavg"].mean().reset_index()
        for _, row in top.sort_values("delta_qwk_vs_fedavg", ascending=False).head(8).iterrows():
            print(f"noise={row['noise_level']:.2f} strategy={row['strategy']} delta_qwk={row['delta_qwk_vs_fedavg']:.4f}")

    print("\nTop worst-site robustness deltas:")
    if not worst_site_deltas.empty:
        top = worst_site_deltas.groupby(["noise_level", "strategy"], dropna=False)[
            "delta_worst_site_qwk_vs_fedavg"
        ].mean().reset_index()
        for _, row in top.sort_values("delta_worst_site_qwk_vs_fedavg", ascending=False).head(8).iterrows():
            print(
                f"noise={row['noise_level']:.2f} strategy={row['strategy']} "
                f"delta_worst_site_qwk={row['delta_worst_site_qwk_vs_fedavg']:.4f}"
            )


if __name__ == "__main__":
    main()
