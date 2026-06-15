#!/usr/bin/env python3
"""Analyze frozen SCORPION embeddings with paired scanner metrics.

Use validation during development. Reserve the test split until the method and
hyperparameters are frozen.
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


SCANNERS = ("AT2", "GT450", "DP200", "P1000", "B300")


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
        metadata = json.loads(str(z["metadata_json"].item())) if "metadata_json" in z.files else {}
    if features.ndim != 2 or len(features) != len(frame):
        raise AnalysisError("Feature matrix and metadata are not aligned.")
    if not np.isfinite(features).all():
        raise AnalysisError("Features contain NaN or infinite values.")
    if frame.duplicated(["slide_id", "region_id", "scanner_id"]).any():
        raise AnalysisError("Duplicate slide/region/scanner rows found.")
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


def region_map(features, frame, indices, scanner):
    result = {}
    for index in indices:
        row = frame.iloc[index]
        if row["scanner_id"] == scanner:
            region = str(row["region_id"])
            if region in result:
                raise AnalysisError(f"Duplicate {scanner}/{region}")
            result[region] = features[index]
    return result


def retrieval(map_a, map_b):
    regions = sorted(set(map_a) & set(map_b))
    a = np.stack([map_a[r] for r in regions])
    b = np.stack([map_b[r] for r in regions])
    similarities = a @ b.T

    def one_direction(matrix):
        order = np.argsort(-matrix, axis=1)
        truth = np.arange(len(regions))
        top1 = np.mean(order[:, 0] == truth)
        top5 = np.mean([truth[i] in order[i, : min(5, len(regions))] for i in truth])
        ranks = np.array([np.flatnonzero(order[i] == i)[0] + 1 for i in truth])
        return float(top1), float(top5), float(np.mean(1.0 / ranks))

    ab = one_direction(similarities)
    ba = one_direction(similarities.T)
    return tuple((ab[i] + ba[i]) / 2.0 for i in range(3))


def paired_metrics(normalized, frame, indices):
    maps = {scanner: region_map(normalized, frame, indices, scanner) for scanner in SCANNERS}
    if {scanner for scanner, values in maps.items() if values} != set(SCANNERS):
        raise AnalysisError("Evaluation split does not contain all five scanners.")

    pair_rows = []
    region_rows = []
    for scanner_a, scanner_b in itertools.combinations(SCANNERS, 2):
        regions = sorted(set(maps[scanner_a]) & set(maps[scanner_b]))
        cosine = np.array([maps[scanner_a][r] @ maps[scanner_b][r] for r in regions])
        distance = np.array([
            np.linalg.norm(maps[scanner_a][r] - maps[scanner_b][r]) for r in regions
        ])
        top1, top5, mrr = retrieval(maps[scanner_a], maps[scanner_b])
        pair_rows.append({
            "scanner_a": scanner_a,
            "scanner_b": scanner_b,
            "n_regions": len(regions),
            "cosine_mean": float(cosine.mean()),
            "cosine_std": float(cosine.std(ddof=1)),
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
            "cosine_similarity": float(c),
            "euclidean_distance": float(d),
        } for region, c, d in zip(regions, cosine, distance))
    return pd.DataFrame(pair_rows), pd.DataFrame(region_rows)


def at2_deviations(normalized, frame, indices):
    reference = region_map(normalized, frame, indices, "AT2")
    rows = []
    for scanner in SCANNERS:
        current = region_map(normalized, frame, indices, scanner)
        for region in sorted(set(reference) & set(current)):
            delta = current[region] - reference[region]
            rows.append({
                "region_id": region,
                "scanner_id": scanner,
                "at2_cosine_similarity": float(current[region] @ reference[region]),
                "at2_delta_l2": float(np.linalg.norm(delta)),
            })
    return pd.DataFrame(rows)


def effective_rank(features):
    centered = features - features.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    energy = singular_values ** 2
    if float(energy.sum()) <= 0:
        return 0.0
    probabilities = energy / energy.sum()
    probabilities = probabilities[probabilities > 0]
    return float(math.exp(-np.sum(probabilities * np.log(probabilities))))


def scanner_probe(features, frame, train_indices, eval_indices):
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError as exc:
        raise AnalysisError("Install scikit-learn for scanner probing.") from exc

    train_slides = set(frame.iloc[train_indices]["slide_id"])
    eval_slides = set(frame.iloc[eval_indices]["slide_id"])
    overlap = train_slides & eval_slides
    if overlap:
        raise AnalysisError(f"Slide leakage in scanner probe: {sorted(overlap)[:10]}")

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
        "confusion_matrix": confusion_matrix(
            labels[eval_indices], prediction, labels=list(SCANNERS)
        ).tolist(),
        "labels": list(SCANNERS),
        "n_train_slides": len(train_slides),
        "n_eval_slides": len(eval_slides),
    }


def analyze(feature_path, out_dir, train_split, eval_split):
    features, frame, extraction_metadata = load_archive(feature_path)
    normalized = normalize(features)
    train_indices = split_indices(frame, train_split)
    eval_indices = split_indices(frame, eval_split)

    pair_summary, region_pairs = paired_metrics(normalized, frame, eval_indices)
    deviations = at2_deviations(normalized, frame, eval_indices)
    probe = scanner_probe(features, frame, train_indices, eval_indices)
    eval_features = features[eval_indices]
    variances = eval_features.var(axis=0)

    summary = {
        "feature_archive": str(feature_path.resolve()),
        "probe_train_split": train_split,
        "eval_split": eval_split,
        "n_eval_rows": len(eval_indices),
        "n_eval_slides": int(frame.iloc[eval_indices]["slide_id"].nunique()),
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
    deviations.to_csv(out_dir / "at2_paired_deviations.csv", index=False)
    (out_dir / "frozen_feature_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (out_dir / "frozen_feature_report.md").write_text(
        "\n".join([
            "# SCORPION frozen-feature paired audit",
            "",
            f"- Evaluation split: `{eval_split}`",
            f"- Slides: {summary['n_eval_slides']}",
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


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--probe-train-split", default="train")
    parser.add_argument("--eval-split", default="val")
    args = parser.parse_args()
    try:
        summary = analyze(
            args.features, args.out_dir, args.probe_train_split, args.eval_split
        )
    except (AnalysisError, OSError, RuntimeError, np.linalg.LinAlgError) as exc:
        print(f"SCORPION FROZEN-FEATURE ANALYSIS FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print("SCORPION FROZEN-FEATURE ANALYSIS PASSED")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
