#!/usr/bin/env python3
"""Analyze frozen canine SCC embeddings with paired scanner metrics.

This mirrors the SCORPION frozen-feature audit but uses the canine scanner IDs
and biological samples as the leakage-blocking unit.
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd


SCANNERS = ("cs2", "gt450", "nz20", "nz210", "p1000")


class AnalysisError(ValueError):
    pass


def load_archive(path: Path):
    with np.load(path, allow_pickle=False) as z:
        needed = {"features", "slide_id", "region_id", "scanner_id", "path", "split"}
        missing = sorted(needed - set(z.files))
        if missing:
            raise AnalysisError(f"Missing arrays: {missing}")
        features = np.asarray(z["features"], dtype=np.float32)
        frame = pd.DataFrame({name: z[name].astype(str) for name in needed - {"features"}})
        for optional in ("fold", "sample_id", "category_name", "source_filename"):
            if optional in z.files:
                frame[optional] = z[optional].astype(str)
        metadata = json.loads(str(z["metadata_json"].item())) if "metadata_json" in z.files else {}
    frame["scanner_id"] = frame["scanner_id"].astype(str).str.lower()
    if features.ndim != 2 or len(features) != len(frame):
        raise AnalysisError("Feature matrix and metadata are not aligned.")
    if not np.isfinite(features).all():
        raise AnalysisError("Features contain NaN or infinite values.")
    if frame.duplicated(["slide_id", "region_id", "scanner_id"]).any():
        raise AnalysisError("Duplicate sample/region/scanner rows found.")
    if set(frame["scanner_id"].unique()) != set(SCANNERS):
        raise AnalysisError(f"Unexpected scanner set: {sorted(frame['scanner_id'].unique())}")
    return features, frame, metadata


def normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise AnalysisError("Zero-norm feature vector found.")
    return features / norms


def split_indices(frame: pd.DataFrame, split: str) -> np.ndarray:
    indices = np.flatnonzero(frame["split"].to_numpy() == split)
    if len(indices) == 0:
        raise AnalysisError(f"No rows found for split={split!r}")
    return indices


def region_map(features: np.ndarray, frame: pd.DataFrame, indices: np.ndarray, scanner: str):
    result: dict[str, np.ndarray] = {}
    for index in indices:
        row = frame.iloc[index]
        if row["scanner_id"] == scanner:
            region = str(row["region_id"])
            if region in result:
                raise AnalysisError(f"Duplicate {scanner}/{region}")
            result[region] = features[index]
    return result


def retrieval(map_a: dict[str, np.ndarray], map_b: dict[str, np.ndarray]):
    regions = sorted(set(map_a) & set(map_b))
    if not regions:
        raise AnalysisError("No paired regions for retrieval")
    a = np.stack([map_a[region] for region in regions])
    b = np.stack([map_b[region] for region in regions])
    similarities = a @ b.T

    def one_direction(matrix: np.ndarray):
        order = np.argsort(-matrix, axis=1)
        truth = np.arange(len(regions))
        top1 = np.mean(order[:, 0] == truth)
        top5 = np.mean([truth[i] in order[i, : min(5, len(regions))] for i in truth])
        ranks = np.array([np.flatnonzero(order[i] == i)[0] + 1 for i in truth])
        return float(top1), float(top5), float(np.mean(1.0 / ranks))

    forward = one_direction(similarities)
    reverse = one_direction(similarities.T)
    return tuple((forward[i] + reverse[i]) / 2.0 for i in range(3))


def paired_metrics(normalized: np.ndarray, frame: pd.DataFrame, indices: np.ndarray):
    maps = {scanner: region_map(normalized, frame, indices, scanner) for scanner in SCANNERS}
    if {scanner for scanner, values in maps.items() if values} != set(SCANNERS):
        raise AnalysisError("Evaluation split does not contain all five scanners.")

    pair_rows: list[dict[str, object]] = []
    region_rows: list[dict[str, object]] = []
    for scanner_a, scanner_b in itertools.combinations(SCANNERS, 2):
        regions = sorted(set(maps[scanner_a]) & set(maps[scanner_b]))
        cosine = np.array([maps[scanner_a][region] @ maps[scanner_b][region] for region in regions])
        distance = np.array([
            np.linalg.norm(maps[scanner_a][region] - maps[scanner_b][region])
            for region in regions
        ])
        top1, top5, mrr = retrieval(maps[scanner_a], maps[scanner_b])
        pair_rows.append({
            "scanner_a": scanner_a,
            "scanner_b": scanner_b,
            "n_regions": len(regions),
            "cosine_mean": float(cosine.mean()),
            "cosine_std": float(cosine.std(ddof=1)) if len(cosine) > 1 else 0.0,
            "cosine_median": float(np.median(cosine)),
            "cosine_min": float(cosine.min()),
            "euclidean_mean": float(distance.mean()),
            "retrieval_top1": top1,
            "retrieval_top5": top5,
            "retrieval_mrr": mrr,
        })
        region_rows.extend({
            "region_id": region,
            "scanner_a": scanner_a,
            "scanner_b": scanner_b,
            "cosine_similarity": float(cosine_value),
            "euclidean_distance": float(distance_value),
        } for region, cosine_value, distance_value in zip(regions, cosine, distance))
    return pd.DataFrame(pair_rows), pd.DataFrame(region_rows)


def reference_deviations(normalized: np.ndarray, frame: pd.DataFrame, indices: np.ndarray, reference_scanner: str):
    reference = region_map(normalized, frame, indices, reference_scanner)
    rows: list[dict[str, object]] = []
    for scanner in SCANNERS:
        current = region_map(normalized, frame, indices, scanner)
        for region in sorted(set(reference) & set(current)):
            delta = current[region] - reference[region]
            rows.append({
                "region_id": region,
                "scanner_id": scanner,
                f"{reference_scanner}_cosine_similarity": float(current[region] @ reference[region]),
                f"{reference_scanner}_delta_l2": float(np.linalg.norm(delta)),
            })
    return pd.DataFrame(rows)


def effective_rank(features: np.ndarray) -> float:
    centered = features - features.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    energy = singular_values ** 2
    if float(energy.sum()) <= 0:
        return 0.0
    probabilities = energy / energy.sum()
    probabilities = probabilities[probabilities > 0]
    return float(math.exp(-np.sum(probabilities * np.log(probabilities))))


def scanner_probe(features: np.ndarray, frame: pd.DataFrame, train_indices: np.ndarray, eval_indices: np.ndarray):
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise AnalysisError("Install scikit-learn for scanner probing.") from exc

    train_samples = set(frame.iloc[train_indices]["slide_id"])
    eval_samples = set(frame.iloc[eval_indices]["slide_id"])
    overlap = train_samples & eval_samples
    if overlap:
        raise AnalysisError(f"Sample leakage in scanner probe: {sorted(overlap)[:10]}")

    labels = frame["scanner_id"].to_numpy()
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=0),
    )
    model.fit(features[train_indices], labels[train_indices])
    prediction = model.predict(features[eval_indices])
    return {
        "accuracy": float(accuracy_score(labels[eval_indices], prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(labels[eval_indices], prediction)),
        "chance_accuracy": 0.2,
        "confusion_matrix": confusion_matrix(labels[eval_indices], prediction, labels=list(SCANNERS)).tolist(),
        "labels": list(SCANNERS),
        "n_train_samples": len(train_samples),
        "n_eval_samples": len(eval_samples),
    }


def analyze(feature_path: Path, out_dir: Path, train_split: str, eval_split: str, reference_scanner: str):
    features, frame, extraction_metadata = load_archive(feature_path)
    normalized = normalize(features)
    train_indices = split_indices(frame, train_split)
    eval_indices = split_indices(frame, eval_split)

    pair_summary, region_pairs = paired_metrics(normalized, frame, eval_indices)
    deviations = reference_deviations(normalized, frame, eval_indices, reference_scanner)
    probe = scanner_probe(features, frame, train_indices, eval_indices)
    eval_features = features[eval_indices]
    variances = eval_features.var(axis=0)

    summary = {
        "feature_archive": str(feature_path.resolve()),
        "probe_train_split": train_split,
        "eval_split": eval_split,
        "reference_scanner": reference_scanner,
        "n_eval_rows": len(eval_indices),
        "n_eval_samples": int(frame.iloc[eval_indices]["slide_id"].nunique()),
        "n_eval_regions": int(frame.iloc[eval_indices]["region_id"].nunique()),
        "pair_cosine_average": float(pair_summary["cosine_mean"].mean()),
        "pair_cosine_worst": float(pair_summary["cosine_mean"].min()),
        "retrieval_top1_average": float(pair_summary["retrieval_top1"].mean()),
        "retrieval_top1_worst": float(pair_summary["retrieval_top1"].min()),
        "retrieval_mrr_average": float(pair_summary["retrieval_mrr"].mean()),
        "scanner_probe": probe,
        "feature_variance_mean": float(variances.mean()),
        "feature_variance_nonzero_fraction": float(np.mean(variances > 1e-12)),
        "effective_rank": effective_rank(eval_features),
        "extraction_metadata": extraction_metadata,
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    pair_summary.to_csv(out_dir / "scanner_pair_summary.csv", index=False)
    region_pairs.to_csv(out_dir / "paired_region_metrics.csv", index=False)
    deviations.to_csv(out_dir / f"{reference_scanner}_paired_deviations.csv", index=False)
    (out_dir / "frozen_feature_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "frozen_feature_report.md").write_text(
        "\n".join([
            "# Canine SCC frozen-feature paired audit",
            "",
            f"- Evaluation split: `{eval_split}`",
            f"- Samples: {summary['n_eval_samples']}",
            f"- Regions: {summary['n_eval_regions']}",
            f"- Mean pair cosine: {summary['pair_cosine_average']:.6f}",
            f"- Worst pair cosine: {summary['pair_cosine_worst']:.6f}",
            f"- Mean cross-scanner top-1 retrieval: {summary['retrieval_top1_average']:.6f}",
            f"- Worst cross-scanner top-1 retrieval: {summary['retrieval_top1_worst']:.6f}",
            f"- Scanner-probe accuracy: {probe['accuracy']:.6f} (chance 0.2)",
            f"- Effective rank: {summary['effective_rank']:.3f}",
            "",
            "## Scanner-pair results",
            "",
            pair_summary.to_markdown(index=False),
        ]) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--probe-train-split", default="train")
    parser.add_argument("--eval-split", default="val")
    parser.add_argument("--reference-scanner", choices=SCANNERS, default="cs2")
    args = parser.parse_args()
    try:
        summary = analyze(
            args.features,
            args.out_dir,
            args.probe_train_split,
            args.eval_split,
            args.reference_scanner,
        )
    except (AnalysisError, OSError, RuntimeError, np.linalg.LinAlgError) as exc:
        print(f"CANINE SCC FROZEN-FEATURE ANALYSIS FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print("CANINE SCC FROZEN-FEATURE ANALYSIS PASSED")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
