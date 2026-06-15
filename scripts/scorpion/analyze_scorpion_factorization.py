#!/usr/bin/env python3
"""Audit PathoAlign biological/acquisition factorization on SCORPION.

The projected archive must contain both ``features`` (biological representation)
and ``acquisition_features``. Scanner probes are trained on the train slides and
evaluated on the disjoint validation slides. The script also reports tissue-region
retrieval in each branch and normalized biological/acquisition cross-covariance.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


SCANNERS = ("AT2", "GT450", "DP200", "P1000", "B300")


class FactorAuditError(ValueError):
    pass


def load_archive(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        required = {
            "features",
            "acquisition_features",
            "slide_id",
            "region_id",
            "scanner_id",
            "split",
        }
        missing = sorted(required - set(archive.files))
        if missing:
            raise FactorAuditError(f"Projected archive is missing arrays: {missing}")
        biological = np.asarray(archive["features"], dtype=np.float32)
        acquisition = np.asarray(archive["acquisition_features"], dtype=np.float32)
        frame = pd.DataFrame(
            {
                name: archive[name].astype(str)
                for name in ("slide_id", "region_id", "scanner_id", "split")
            }
        )
        metadata = (
            json.loads(str(archive["metadata_json"].item()))
            if "metadata_json" in archive.files
            else {}
        )
    if len(biological) != len(acquisition) or len(frame) != len(biological):
        raise FactorAuditError("Biological, acquisition, and metadata rows differ.")
    if not np.isfinite(biological).all() or not np.isfinite(acquisition).all():
        raise FactorAuditError("Representations contain NaN or infinite values.")
    return biological, acquisition, frame, metadata


def split_indices(frame: pd.DataFrame, split: str) -> np.ndarray:
    indices = np.flatnonzero(frame["split"].to_numpy() == split)
    if len(indices) == 0:
        raise FactorAuditError(f"No rows found for split={split!r}")
    return indices


def validate_disjoint(frame: pd.DataFrame, train_indices, eval_indices) -> None:
    train_slides = set(frame.iloc[train_indices]["slide_id"])
    eval_slides = set(frame.iloc[eval_indices]["slide_id"])
    overlap = sorted(train_slides & eval_slides)
    if overlap:
        raise FactorAuditError(f"Slide leakage detected: {overlap[:20]}")


def scanner_probe(features, labels, train_indices, eval_indices):
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=5000,
            random_state=0,
        ),
    )
    model.fit(features[train_indices], labels[train_indices])
    prediction = model.predict(features[eval_indices])
    return {
        "accuracy": float(accuracy_score(labels[eval_indices], prediction)),
        "balanced_accuracy": float(
            balanced_accuracy_score(labels[eval_indices], prediction)
        ),
        "chance_accuracy": 1.0 / len(SCANNERS),
    }


def normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise FactorAuditError("Zero-norm feature row found.")
    return features / norms


def region_map(features, frame, indices, scanner):
    mapping = {}
    for local_index, global_index in enumerate(indices):
        row = frame.iloc[global_index]
        if row["scanner_id"] == scanner:
            mapping[str(row["region_id"])] = features[local_index]
    return mapping


def directional_top1(similarity: np.ndarray) -> float:
    order = np.argmax(similarity, axis=1)
    return float(np.mean(order == np.arange(len(similarity))))


def tissue_retrieval(features, frame, eval_indices):
    normalized = normalize(features[eval_indices])
    mappings = {
        scanner: region_map(normalized, frame, eval_indices, scanner)
        for scanner in SCANNERS
    }
    pair_top1 = []
    pair_cosine = []
    for scanner_a, scanner_b in itertools.combinations(SCANNERS, 2):
        regions = sorted(set(mappings[scanner_a]) & set(mappings[scanner_b]))
        matrix_a = np.stack([mappings[scanner_a][region] for region in regions])
        matrix_b = np.stack([mappings[scanner_b][region] for region in regions])
        similarity = matrix_a @ matrix_b.T
        pair_top1.append(
            0.5
            * (
                directional_top1(similarity)
                + directional_top1(similarity.T)
            )
        )
        pair_cosine.append(float(np.mean(np.diag(similarity))))
    return {
        "top1_average": float(np.mean(pair_top1)),
        "top1_worst": float(np.min(pair_top1)),
        "paired_cosine_average": float(np.mean(pair_cosine)),
        "paired_cosine_worst": float(np.min(pair_cosine)),
    }


def effective_rank(features: np.ndarray) -> float:
    centered = features - features.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    energy = singular_values**2
    if float(energy.sum()) <= 0:
        return 0.0
    probabilities = energy / energy.sum()
    probabilities = probabilities[probabilities > 0]
    return float(math.exp(-np.sum(probabilities * np.log(probabilities))))


def normalized_cross_covariance(biological, acquisition, indices):
    biological = StandardScaler().fit_transform(biological[indices])
    acquisition = StandardScaler().fit_transform(acquisition[indices])
    cross = biological.T @ acquisition / max(1, len(indices) - 1)
    return {
        "mean_absolute": float(np.mean(np.abs(cross))),
        "root_mean_square": float(np.sqrt(np.mean(cross**2))),
        "maximum_absolute": float(np.max(np.abs(cross))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="val")
    args = parser.parse_args()

    try:
        biological, acquisition, frame, metadata = load_archive(args.features)
        train_indices = split_indices(frame, args.train_split)
        eval_indices = split_indices(frame, args.eval_split)
        validate_disjoint(frame, train_indices, eval_indices)
        labels = frame["scanner_id"].to_numpy()

        summary = {
            "feature_archive": str(args.features.resolve()),
            "train_split": args.train_split,
            "eval_split": args.eval_split,
            "n_train_slides": int(frame.iloc[train_indices]["slide_id"].nunique()),
            "n_eval_slides": int(frame.iloc[eval_indices]["slide_id"].nunique()),
            "biological_scanner_probe": scanner_probe(
                biological, labels, train_indices, eval_indices
            ),
            "acquisition_scanner_probe": scanner_probe(
                acquisition, labels, train_indices, eval_indices
            ),
            "biological_tissue_retrieval": tissue_retrieval(
                biological, frame, eval_indices
            ),
            "acquisition_tissue_retrieval": tissue_retrieval(
                acquisition, frame, eval_indices
            ),
            "biological_effective_rank": effective_rank(biological[eval_indices]),
            "acquisition_effective_rank": effective_rank(acquisition[eval_indices]),
            "normalized_cross_covariance": normalized_cross_covariance(
                biological, acquisition, eval_indices
            ),
            "metadata": metadata,
        }
    except (FactorAuditError, OSError, RuntimeError, np.linalg.LinAlgError) as exc:
        print(f"SCORPION FACTOR AUDIT FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    args.out_dir.mkdir(parents=True, exist_ok=True)
    output = args.out_dir / "factorization_summary.json"
    output.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print("SCORPION FACTOR AUDIT PASSED")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Wrote {output.resolve()}")


if __name__ == "__main__":
    main()
