#!/usr/bin/env python3
"""
Optimize ordinal prediction thresholds for PANDA ISUP-grade validation outputs.

This utility takes a validation prediction CSV with columns:

    image_id,isup_grade,prob_0,prob_1,prob_2,prob_3,prob_4,prob_5

It converts class probabilities into a continuous expected-grade score and then
searches for five ordered thresholds that maximize quadratic weighted kappa.

Example:
    python scripts/evaluation/optimize_panda_thresholds.py \
        --predictions results/panda_attention_mil_baseline/val_predictions.csv \
        --out-dir results/panda_threshold_optimization/attention_mil

Note:
    Older TransnnMIL outputs that only contain a single `probability` column are
    not sufficient for threshold optimization. Re-run the trainer after updating
    it to save per-class probabilities.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score


PROB_COLUMNS = [f"prob_{idx}" for idx in range(6)]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize PANDA ISUP ordinal thresholds for QWK")
    parser.add_argument("--predictions", required=True, help="Validation prediction CSV with prob_0..prob_5 columns")
    parser.add_argument("--out-dir", required=True, help="Output directory for optimized predictions and metrics")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-iters", type=int, default=20000)
    parser.add_argument("--refine-rounds", type=int, default=4)
    return parser.parse_args()


def validate_predictions(frame: pd.DataFrame) -> None:
    required = {"image_id", "isup_grade"} | set(PROB_COLUMNS)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(
            "Prediction CSV is missing required columns: "
            f"{sorted(missing)}. Need per-class probabilities prob_0..prob_5."
        )


def expected_grade(frame: pd.DataFrame) -> np.ndarray:
    probs = frame[PROB_COLUMNS].to_numpy(dtype=np.float64)
    row_sums = probs.sum(axis=1, keepdims=True)
    probs = probs / np.clip(row_sums, 1e-12, None)
    grades = np.arange(6, dtype=np.float64)
    return probs @ grades


def apply_thresholds(scores: np.ndarray, thresholds: Iterable[float]) -> np.ndarray:
    thresholds_arr = np.asarray(list(thresholds), dtype=np.float64)
    return np.digitize(scores, thresholds_arr).astype(np.int64)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, object]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "qwk": float(cohen_kappa_score(y_true, y_pred, weights="quadratic")),
        "confusion_matrix_labels_0_to_5": confusion_matrix(y_true, y_pred, labels=list(range(6))).tolist(),
    }


def qwk_for_thresholds(y_true: np.ndarray, scores: np.ndarray, thresholds: Iterable[float]) -> float:
    preds = apply_thresholds(scores, thresholds)
    return float(cohen_kappa_score(y_true, preds, weights="quadratic"))


def initial_thresholds_from_quantiles(scores: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    # Match the empirical cumulative label distribution as a robust starting point.
    counts = np.bincount(y_true.astype(np.int64), minlength=6)
    cumulative = np.cumsum(counts)[:-1] / max(len(y_true), 1)
    thresholds = np.quantile(scores, cumulative)
    return np.maximum.accumulate(thresholds + np.linspace(0.0, 1e-6, len(thresholds)))


def random_search(
    y_true: np.ndarray,
    scores: np.ndarray,
    seed: int,
    random_iters: int,
    refine_rounds: int,
) -> Tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)

    candidates: List[np.ndarray] = []
    candidates.append(np.array([0.5, 1.5, 2.5, 3.5, 4.5], dtype=np.float64))
    candidates.append(initial_thresholds_from_quantiles(scores, y_true))

    best_thresholds = candidates[0]
    best_qwk = -1.0

    for candidate in candidates:
        score = qwk_for_thresholds(y_true, scores, candidate)
        if score > best_qwk:
            best_qwk = score
            best_thresholds = candidate.copy()

    low = float(np.min(scores))
    high = float(np.max(scores))

    for _ in range(random_iters):
        candidate = np.sort(rng.uniform(low, high, size=5))
        score = qwk_for_thresholds(y_true, scores, candidate)
        if score > best_qwk:
            best_qwk = score
            best_thresholds = candidate.copy()

    step = max((high - low) / 8.0, 1e-3)
    for _round_idx in range(refine_rounds):
        improved = True
        while improved:
            improved = False
            for idx in range(5):
                for direction in (-1.0, 1.0):
                    candidate = best_thresholds.copy()
                    candidate[idx] += direction * step
                    candidate = np.sort(candidate)
                    score = qwk_for_thresholds(y_true, scores, candidate)
                    if score > best_qwk:
                        best_qwk = score
                        best_thresholds = candidate.copy()
                        improved = True
        step *= 0.5

    return best_thresholds, best_qwk


def main() -> None:
    args = parse_args()
    predictions_path = Path(args.predictions)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    frame = pd.read_csv(predictions_path)
    validate_predictions(frame)

    y_true = frame["isup_grade"].to_numpy(dtype=np.int64)
    argmax_pred = frame[PROB_COLUMNS].to_numpy(dtype=np.float64).argmax(axis=1).astype(np.int64)
    scores = expected_grade(frame)

    baseline = metrics(y_true, argmax_pred)
    thresholds, best_qwk = random_search(
        y_true=y_true,
        scores=scores,
        seed=args.seed,
        random_iters=args.random_iters,
        refine_rounds=args.refine_rounds,
    )
    threshold_pred = apply_thresholds(scores, thresholds)
    optimized = metrics(y_true, threshold_pred)

    output = frame.copy()
    output["expected_grade_score"] = scores
    output["threshold_pred_isup_grade"] = threshold_pred
    output.to_csv(out_dir / "threshold_predictions.csv", index=False)

    report = {
        "source_predictions": str(predictions_path),
        "thresholds": [float(x) for x in thresholds],
        "baseline_argmax_metrics": baseline,
        "optimized_threshold_metrics": optimized,
        "qwk_improvement": float(optimized["qwk"] - baseline["qwk"]),
        "search": {
            "seed": args.seed,
            "random_iters": args.random_iters,
            "refine_rounds": args.refine_rounds,
        },
    }

    with open(out_dir / "threshold_metrics.json", "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)

    print("PANDA threshold optimization complete")
    print(f"Source: {predictions_path}")
    print(f"Baseline argmax QWK: {baseline['qwk']:.4f}")
    print(f"Optimized threshold QWK: {optimized['qwk']:.4f}")
    print(f"Improvement: {report['qwk_improvement']:.4f}")
    print(f"Thresholds: {[round(float(x), 4) for x in thresholds]}")
    print(f"Metrics: {out_dir / 'threshold_metrics.json'}")
    print(f"Predictions: {out_dir / 'threshold_predictions.csv'}")


if __name__ == "__main__":
    main()
