#!/usr/bin/env python3
"""Evaluate an observable dominance-aware switch from saved predictions.

This script replaces the oracle rule (noise == 0 -> FedAvg, noise > 0 ->
cross-site blending) with a detector that uses only FedAvg validation behavior.
It is designed for saved prediction files produced by
run_fair_weights_h_panda_feature_stress.py with --save-predictions.

The default detector is calibrated from clean runs only:
- use clean FedAvg runs to estimate normal ranges for validation diagnostics
- switch to the corrupted-regime strategy when FedAvg looks outside that clean range

Default trigger:
- FedAvg worst-site QWK below the clean 10th percentile, OR
- FedAvg global QWK below the clean 10th percentile, OR
- FedAvg site-QWK spread above the clean 90th percentile, OR
- FedAvg mean absolute ordinal error above the clean 90th percentile, OR
- FedAvg severe ordinal error rate above the clean 90th percentile

This is not a clinical detector. It is an experimental stress-test detector that
answers: can observable validation signals recover the oracle switch behavior?
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

LABEL_COL = "isup_grade_true"
PRED_COL = "isup_grade_pred"
SITE_COL = "site_id"
STRATEGY_COL = "strategy"
SPLIT_COL = "split"

DEFAULT_METRICS = (
    "global_qwk",
    "worst_site_qwk",
    "mean_site_qwk",
    "global_accuracy",
    "macro_f1",
)


def parse_noise_seed(path: Path) -> tuple[int, int] | None:
    match = re.search(r"noise_(\d+)_seed_(\d+)", str(path))
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def iter_prediction_files(patterns: Iterable[str]) -> Iterable[Path]:
    seen: set[Path] = set()
    for pattern in patterns:
        for path in Path(".").glob(pattern):
            if path not in seen:
                seen.add(path)
                yield path


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int | None = None) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    if y_true.size == 0:
        return float("nan")

    if n_classes is None:
        max_label = int(max(y_true.max(initial=0), y_pred.max(initial=0)))
        n_classes = max_label + 1

    conf = np.zeros((n_classes, n_classes), dtype=float)
    for truth, pred in zip(y_true, y_pred):
        if 0 <= truth < n_classes and 0 <= pred < n_classes:
            conf[truth, pred] += 1.0

    hist_true = conf.sum(axis=1)
    hist_pred = conf.sum(axis=0)
    expected = np.outer(hist_true, hist_pred) / max(conf.sum(), 1.0)

    weights = np.zeros((n_classes, n_classes), dtype=float)
    denom = max((n_classes - 1) ** 2, 1)
    for i in range(n_classes):
        for j in range(n_classes):
            weights[i, j] = ((i - j) ** 2) / denom

    observed = float((weights * conf).sum())
    expected_weighted = float((weights * expected).sum())
    if expected_weighted == 0.0:
        return 1.0 if observed == 0.0 else 0.0
    return 1.0 - observed / expected_weighted


def macro_f1(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 6) -> float:
    scores: list[float] = []
    for cls in range(n_classes):
        tp = int(((y_true == cls) & (y_pred == cls)).sum())
        fp = int(((y_true != cls) & (y_pred == cls)).sum())
        fn = int(((y_true == cls) & (y_pred != cls)).sum())
        denom = 2 * tp + fp + fn
        scores.append((2 * tp / denom) if denom else 0.0)
    return float(np.mean(scores))


def prediction_entropy(df: pd.DataFrame) -> float:
    prob_cols = sorted([c for c in df.columns if re.fullmatch(r"prob_\d+", c)])
    if not prob_cols:
        return float("nan")
    probs = df[prob_cols].to_numpy(dtype=float)
    probs = np.clip(probs, 1e-12, 1.0)
    ent = -(probs * np.log(probs)).sum(axis=1)
    return float(np.mean(ent))


def compute_strategy_metrics(df: pd.DataFrame, n_classes: int = 6) -> dict[str, float]:
    y_true = df[LABEL_COL].to_numpy(dtype=int)
    y_pred = df[PRED_COL].to_numpy(dtype=int)
    err = y_pred - y_true

    site_qwks = []
    for _, site_df in df.groupby(SITE_COL):
        site_qwks.append(
            quadratic_weighted_kappa(
                site_df[LABEL_COL].to_numpy(dtype=int),
                site_df[PRED_COL].to_numpy(dtype=int),
                n_classes=n_classes,
            )
        )
    site_qwks_arr = np.asarray(site_qwks, dtype=float)
    valid_site_qwks = site_qwks_arr[np.isfinite(site_qwks_arr)]

    return {
        "global_qwk": quadratic_weighted_kappa(y_true, y_pred, n_classes=n_classes),
        "worst_site_qwk": float(np.min(valid_site_qwks)) if valid_site_qwks.size else float("nan"),
        "mean_site_qwk": float(np.mean(valid_site_qwks)) if valid_site_qwks.size else float("nan"),
        "site_qwk_spread": float(np.max(valid_site_qwks) - np.min(valid_site_qwks)) if valid_site_qwks.size else float("nan"),
        "global_accuracy": float(np.mean(y_true == y_pred)),
        "macro_f1": macro_f1(y_true, y_pred, n_classes=n_classes),
        "mean_abs_error": float(np.mean(np.abs(err))),
        "severe_error_rate": float(np.mean(np.abs(err) >= 3)),
        "mean_overgrade": float(np.mean(np.where(err > 0, err, 0))),
        "mean_undergrade": float(np.mean(np.where(err < 0, -err, 0))),
        "prediction_entropy": prediction_entropy(df),
    }


def summarize_run(path: Path, clean_strategy: str, corrupted_strategy: str, n_classes: int) -> dict[str, object] | None:
    parsed = parse_noise_seed(path)
    if parsed is None:
        return None
    noise, seed = parsed

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
        "noise": noise,
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


def percentile(series: pd.Series, q: float) -> float:
    values = series.dropna().astype(float).to_numpy()
    if values.size == 0:
        return float("nan")
    return float(np.quantile(values, q))


def build_thresholds(clean_runs: pd.DataFrame, low_q: float, high_q: float) -> dict[str, float]:
    return {
        "global_qwk_low": percentile(clean_runs["clean_global_qwk"], low_q),
        "worst_site_qwk_low": percentile(clean_runs["clean_worst_site_qwk"], low_q),
        "site_qwk_spread_high": percentile(clean_runs["clean_site_qwk_spread"], high_q),
        "mean_abs_error_high": percentile(clean_runs["clean_mean_abs_error"], high_q),
        "severe_error_rate_high": percentile(clean_runs["clean_severe_error_rate"], high_q),
        "prediction_entropy_high": percentile(clean_runs["clean_prediction_entropy"], high_q),
    }


def detector_trigger(row: pd.Series, thresholds: dict[str, float], use_entropy: bool) -> tuple[bool, list[str]]:
    reasons: list[str] = []

    checks = [
        ("global_qwk_low", row["clean_global_qwk"] < thresholds["global_qwk_low"], "FedAvg global QWK below clean-calibrated lower bound"),
        ("worst_site_qwk_low", row["clean_worst_site_qwk"] < thresholds["worst_site_qwk_low"], "FedAvg worst-site QWK below clean-calibrated lower bound"),
        ("site_qwk_spread_high", row["clean_site_qwk_spread"] > thresholds["site_qwk_spread_high"], "FedAvg site-QWK spread above clean-calibrated upper bound"),
        ("mean_abs_error_high", row["clean_mean_abs_error"] > thresholds["mean_abs_error_high"], "FedAvg mean absolute ordinal error above clean-calibrated upper bound"),
        ("severe_error_rate_high", row["clean_severe_error_rate"] > thresholds["severe_error_rate_high"], "FedAvg severe ordinal error rate above clean-calibrated upper bound"),
    ]

    if use_entropy and math.isfinite(float(row.get("clean_prediction_entropy", float("nan")))):
        checks.append(
            (
                "prediction_entropy_high",
                row["clean_prediction_entropy"] > thresholds["prediction_entropy_high"],
                "FedAvg prediction entropy above clean-calibrated upper bound",
            )
        )

    for _, passed, reason in checks:
        if bool(passed):
            reasons.append(reason)

    return bool(reasons), reasons


def apply_detector(runs: pd.DataFrame, thresholds: dict[str, float], use_entropy: bool) -> pd.DataFrame:
    output = runs.copy()
    chosen = []
    trigger_flags = []
    reason_strings = []

    for _, row in output.iterrows():
        trigger, reasons = detector_trigger(row, thresholds, use_entropy=use_entropy)
        trigger_flags.append(trigger)
        reason_strings.append("; ".join(reasons))
        chosen.append(row["corrupted_strategy"] if trigger else row["clean_strategy"])

    output["detector_triggered"] = trigger_flags
    output["detector_reasons"] = reason_strings
    output["chosen_strategy"] = chosen

    for metric in DEFAULT_METRICS:
        output[f"detector_{metric}"] = np.where(
            output["detector_triggered"],
            output[f"corrupted_{metric}"],
            output[f"clean_{metric}"],
        )
        output[f"delta_detector_vs_clean_{metric}"] = output[f"detector_{metric}"] - output[f"clean_{metric}"]
        output[f"oracle_{metric}"] = np.where(
            output["noise"] == 0,
            output[f"clean_{metric}"],
            output[f"corrupted_{metric}"],
        )
        output[f"delta_oracle_vs_clean_{metric}"] = output[f"oracle_{metric}"] - output[f"clean_{metric}"]
        output[f"detector_regret_vs_oracle_{metric}"] = output[f"detector_{metric}"] - output[f"oracle_{metric}"]

    return output


def ci95(values: np.ndarray) -> tuple[float, float, float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    n = values.size
    if n == 0:
        return float("nan"), float("nan"), float("nan"), float("nan")
    mean = float(values.mean())
    sd = float(values.std(ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else 0.0
    tcrit = 2.145 if n == 15 else 2.776 if n == 5 else 1.96
    return mean, sd, mean - tcrit * se, mean + tcrit * se


def build_summary(detected: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []

    for noise, group in detected.groupby("noise"):
        for metric in DEFAULT_METRICS:
            deltas = group[f"delta_detector_vs_clean_{metric}"].to_numpy(dtype=float)
            regrets = group[f"detector_regret_vs_oracle_{metric}"].to_numpy(dtype=float)
            mean, sd, low, high = ci95(deltas)
            regret_mean, regret_sd, regret_low, regret_high = ci95(regrets)

            rows.append(
                {
                    "noise": int(noise),
                    "metric": metric,
                    "n_seeds": int(group["seed"].nunique()),
                    "detector_trigger_rate": float(group["detector_triggered"].mean()),
                    "clean_mean": float(group[f"clean_{metric}"].mean()),
                    "corrupted_strategy_mean": float(group[f"corrupted_{metric}"].mean()),
                    "detector_mean": float(group[f"detector_{metric}"].mean()),
                    "oracle_mean": float(group[f"oracle_{metric}"].mean()),
                    "mean_delta_detector_vs_clean": mean,
                    "sd_delta_detector_vs_clean": sd,
                    "ci95_low": low,
                    "ci95_high": high,
                    "positive_seed_count": int((group[f"delta_detector_vs_clean_{metric}"] > 0).sum()),
                    "negative_seed_count": int((group[f"delta_detector_vs_clean_{metric}"] < 0).sum()),
                    "mean_regret_detector_vs_oracle": regret_mean,
                    "regret_ci95_low": regret_low,
                    "regret_ci95_high": regret_high,
                }
            )
    return pd.DataFrame(rows).sort_values(["noise", "metric"])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--patterns", nargs="+", required=True, help="Glob patterns for predictions.csv files.")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--clean-strategy", default="fedavg")
    parser.add_argument("--corrupted-strategy", default="cross_site_blend_50")
    parser.add_argument("--low-quantile", type=float, default=0.10)
    parser.add_argument("--high-quantile", type=float, default=0.90)
    parser.add_argument("--n-classes", type=int, default=6)
    parser.add_argument("--use-entropy", action="store_true", help="Include prediction entropy in the trigger rule.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    run_rows = []
    for path in iter_prediction_files(args.patterns):
        row = summarize_run(path, args.clean_strategy, args.corrupted_strategy, args.n_classes)
        if row is not None:
            run_rows.append(row)

    runs = pd.DataFrame(run_rows).drop_duplicates(["noise", "seed"])
    if runs.empty:
        raise ValueError("No usable prediction files found.")

    clean_runs = runs[runs["noise"] == 0]
    if clean_runs.empty:
        raise ValueError("No clean noise=0 runs found for detector calibration.")

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
    print(f"Wrote detector run diagnostics to {run_path}")
    print(f"Wrote detector switch summary to {summary_path}")
    print(f"Wrote detector thresholds to {threshold_path}")


if __name__ == "__main__":
    main()
