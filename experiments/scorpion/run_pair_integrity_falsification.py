#!/usr/bin/env python3
"""Run SCORPION pair-integrity falsification for paired-acquisition factorization.

This experiment keeps the frozen DINOv2 SCORPION objective fixed and changes only
the positive-pair construction used during training:

* true_pairs: all five scanner views come from the same tissue region.
* shuffled_region_pairs: non-anchor scanner views are deranged within the same
  slide, breaking exact region identity while preserving slide context.
* shuffled_sample_pairs: non-anchor scanner views are deranged across different
  slides, breaking same-sample/slide identity.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
import tempfile
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

# PyTorch/cuBLAS needs this before CUDA is initialized when deterministic
# algorithms are requested by the imported training helpers.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.scorpion.run_pathoalign_crossfold import (
    align_fold,
    atomic_npz,
    validate_fold,
)
from experiments.scorpion.run_pathoalign_projection import (
    ExperimentError,
    load_archive,
    standardize,
    train_one,
    write_results,
)
from src.models.scorpion_pathoalign import ProjectionConfig


SCANNERS = ("AT2", "GT450", "DP200", "P1000", "B300")
SCANNER_TO_INDEX = {name: index for index, name in enumerate(SCANNERS)}
CONDITIONS = ("true_pairs", "shuffled_region_pairs", "shuffled_sample_pairs")
METRICS = (
    "scanner_probe_accuracy",
    "mean_paired_cosine",
    "worst_paired_cosine",
    "mean_top1_retrieval",
    "worst_top1_retrieval",
    "effective_rank",
    "biological_acquisition_cross_covariance",
)
LOWER_IS_BETTER = {
    "scanner_probe_accuracy",
    "biological_acquisition_cross_covariance",
}


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text: str) -> int:
        for stream in self.streams:
            stream.write(text)
            stream.flush()
        return len(text)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def string_array(values: Iterable[object]) -> np.ndarray:
    strings = [str(value) for value in values]
    width = max(1, max(map(len, strings)))
    return np.asarray(strings, dtype=f"<U{width}")


def config_for(input_dim: int) -> ProjectionConfig:
    return ProjectionConfig(
        input_dim=input_dim,
        biological_dim=256,
        acquisition_dim=64,
        hidden_dim=512,
        temperature=0.1,
        reconstruction_weight=1.0,
        variance_weight=1.0,
        covariance_weight=0.01,
        scanner_adversary_weight=0.5,
        scanner_acquisition_weight=0.5,
        scanner_dependence_weight=20.0,
        cross_covariance_weight=0.05,
        gradient_reversal_strength=1.0,
    )


def row_keys(frame: pd.DataFrame) -> list[tuple[str, str, str]]:
    return [
        (str(row.slide_id), str(row.region_id), str(row.scanner_id))
        for row in frame.itertuples(index=False)
    ]


def deterministic_seed(condition: str, fold: int, seed: int) -> int:
    payload = f"{condition}|fold={fold}|seed={seed}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def derange_indices(
    candidates: list[int],
    *,
    rng: np.random.Generator,
    forbidden: dict[int, set[int]],
    max_attempts: int = 10000,
) -> list[int]:
    if len(candidates) <= 1:
        raise ExperimentError("Cannot derange fewer than two candidate regions.")
    base = np.asarray(candidates, dtype=np.int64)
    for _ in range(max_attempts):
        proposed = rng.permutation(base)
        if all(int(value) not in forbidden[int(anchor)] for anchor, value in zip(base, proposed)):
            return [int(value) for value in proposed]
    raise ExperimentError("Failed to construct deterministic derangement.")


def complete_region_table(frame: pd.DataFrame, fit_indices: np.ndarray) -> pd.DataFrame:
    subset = frame.iloc[fit_indices].copy()
    subset["_row_index"] = subset.index.to_numpy(dtype=np.int64)
    rows = []
    for region_index, (region_id, group) in enumerate(subset.groupby("region_id", sort=True)):
        if len(group) != len(SCANNERS) or set(group["scanner_id"]) != set(SCANNERS):
            raise ExperimentError("Every fitting region must contain all five scanners.")
        slide_values = set(group["slide_id"].astype(str))
        sample_series = (
            group["sample_number"]
            if "sample_number" in group.columns
            else pd.Series([""] * len(group), index=group.index)
        )
        sample_values = set(sample_series.astype(str))
        if len(slide_values) != 1 or len(sample_values) > 1:
            raise ExperimentError(f"Region metadata is not unique for {region_id}.")
        row = {
            "region_index": int(region_index),
            "region_id": str(region_id),
            "slide_id": next(iter(slide_values)),
            "sample_number": next(iter(sample_values)) if sample_values else "",
        }
        for scanner in SCANNERS:
            scanner_rows = group.loc[group["scanner_id"] == scanner, "_row_index"]
            row[scanner] = int(scanner_rows.iloc[0])
        rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        raise ExperimentError("No fitting regions were available.")
    return table


def build_pair_groups(
    frame: pd.DataFrame,
    fit_indices: np.ndarray,
    *,
    condition: str,
    fold: int,
    seed: int,
) -> tuple[list[np.ndarray], pd.DataFrame, dict[str, object]]:
    if condition not in CONDITIONS:
        raise ExperimentError(f"Unknown condition: {condition}")
    table = complete_region_table(frame, fit_indices)
    rng = np.random.default_rng(deterministic_seed(condition, fold, seed))
    assignments = pd.DataFrame(
        {
            "fold": fold,
            "seed": seed,
            "condition": condition,
            "anchor_region_index": table["region_index"],
            "anchor_region_id": table["region_id"],
            "anchor_slide_id": table["slide_id"],
            "anchor_sample_number": table["sample_number"],
        }
    )

    for scanner in SCANNERS:
        assignments[f"{scanner}_region_index"] = table["region_index"]
        assignments[f"{scanner}_region_id"] = table["region_id"]
        assignments[f"{scanner}_slide_id"] = table["slide_id"]
        assignments[f"{scanner}_row_index"] = table[scanner]

    if condition == "shuffled_region_pairs":
        for scanner in SCANNERS[1:]:
            chosen = np.empty(len(table), dtype=np.int64)
            for _, slide_regions in table.groupby("slide_id", sort=True):
                candidates = [int(value) for value in slide_regions["region_index"]]
                forbidden = {candidate: {candidate} for candidate in candidates}
                deranged = derange_indices(candidates, rng=rng, forbidden=forbidden)
                for anchor, assigned in zip(candidates, deranged):
                    chosen[anchor] = assigned
            source = table.set_index("region_index").loc[chosen]
            assignments[f"{scanner}_region_index"] = chosen
            assignments[f"{scanner}_region_id"] = source["region_id"].to_numpy()
            assignments[f"{scanner}_slide_id"] = source["slide_id"].to_numpy()
            assignments[f"{scanner}_row_index"] = source[scanner].to_numpy(dtype=np.int64)

    if condition == "shuffled_sample_pairs":
        slides = sorted(table["slide_id"].astype(str).unique())
        forbidden_slides = {index: {index} for index in range(len(slides))}
        slide_order = derange_indices(
            list(range(len(slides))), rng=rng, forbidden=forbidden_slides
        )
        slide_map = {slides[index]: slides[slide_order[index]] for index in range(len(slides))}
        for scanner in SCANNERS[1:]:
            chosen = np.empty(len(table), dtype=np.int64)
            for source_slide, target_slide in slide_map.items():
                anchors = table.loc[table["slide_id"] == source_slide, "region_index"].to_numpy(dtype=np.int64)
                targets = table.loc[table["slide_id"] == target_slide, "region_index"].to_numpy(dtype=np.int64)
                if len(anchors) != len(targets):
                    raise ExperimentError("Sample-shuffled slide groups must have equal region counts.")
                shuffled_targets = rng.permutation(targets)
                for anchor, assigned in zip(anchors, shuffled_targets):
                    chosen[int(anchor)] = int(assigned)
            source = table.set_index("region_index").loc[chosen]
            assignments[f"{scanner}_region_index"] = chosen
            assignments[f"{scanner}_region_id"] = source["region_id"].to_numpy()
            assignments[f"{scanner}_slide_id"] = source["slide_id"].to_numpy()
            assignments[f"{scanner}_row_index"] = source[scanner].to_numpy(dtype=np.int64)

    groups = []
    for row in assignments.itertuples(index=False):
        groups.append(
            np.asarray(
                [getattr(row, f"{scanner}_row_index") for scanner in SCANNERS],
                dtype=np.int64,
            )
        )

    mismatch_columns = [f"{scanner}_region_id" for scanner in SCANNERS[1:]]
    mismatch_matrix = np.column_stack(
        [
            assignments[column].astype(str).to_numpy() != assignments["anchor_region_id"].astype(str).to_numpy()
            for column in mismatch_columns
        ]
    )
    same_slide_matrix = np.column_stack(
        [
            assignments[f"{scanner}_slide_id"].astype(str).to_numpy()
            == assignments["anchor_slide_id"].astype(str).to_numpy()
            for scanner in SCANNERS[1:]
        ]
    )
    audit = {
        "condition": condition,
        "fold": fold,
        "seed": seed,
        "n_fit_regions": int(len(table)),
        "n_pair_groups": int(len(groups)),
        "anchor_scanner": SCANNERS[0],
        "deterministic_shuffle_seed": int(deterministic_seed(condition, fold, seed)),
        "non_anchor_region_mismatch_fraction": float(mismatch_matrix.mean()),
        "non_anchor_same_slide_fraction": float(same_slide_matrix.mean()),
        "unique_training_rows_used": int(len(np.unique(np.concatenate(groups)))),
        "expected_training_rows_used": int(len(table) * len(SCANNERS)),
    }
    if audit["unique_training_rows_used"] != audit["expected_training_rows_used"]:
        raise ExperimentError(f"Pair construction reused or dropped rows: {audit}")
    if condition == "true_pairs" and audit["non_anchor_region_mismatch_fraction"] != 0.0:
        raise ExperimentError("True pairs unexpectedly mismatched regions.")
    if condition == "shuffled_region_pairs":
        if audit["non_anchor_region_mismatch_fraction"] != 1.0:
            raise ExperimentError("Region-shuffled pairs did not fully break region identity.")
        if audit["non_anchor_same_slide_fraction"] != 1.0:
            raise ExperimentError("Region-shuffled pairs did not preserve slide context.")
    if condition == "shuffled_sample_pairs":
        if audit["non_anchor_region_mismatch_fraction"] != 1.0:
            raise ExperimentError("Sample-shuffled pairs did not fully break region identity.")
        if audit["non_anchor_same_slide_fraction"] != 0.0:
            raise ExperimentError("Sample-shuffled pairs did not fully break slide identity.")
    return groups, assignments, audit


def mark_projection_metadata(path: Path, metadata_update: dict[str, object]) -> None:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    metadata.update(metadata_update)
    text = json.dumps(metadata, sort_keys=True)
    arrays["metadata_json"] = np.asarray(text, dtype=f"<U{len(text)}")
    atomic_npz(path, arrays)


def load_projected(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        required = {"features", "acquisition_features", "slide_id", "region_id", "scanner_id", "split"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ExperimentError(f"{path} is missing arrays: {missing}")
        biological = np.asarray(archive["features"], dtype=np.float32)
        acquisition = np.asarray(archive["acquisition_features"], dtype=np.float32)
        frame = pd.DataFrame(
            {
                name: archive[name].astype(str)
                for name in ("slide_id", "region_id", "scanner_id", "split")
            }
        )
        metadata = json.loads(str(archive["metadata_json"].item()))
    if len(biological) != len(frame) or len(acquisition) != len(frame):
        raise ExperimentError("Projected feature arrays and metadata are misaligned.")
    return biological, acquisition, frame, metadata


def split_indices(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    test = np.flatnonzero(frame["split"].to_numpy() == "test")
    fit = np.flatnonzero(frame["split"].to_numpy() != "test")
    if len(test) == 0 or len(fit) == 0:
        raise ExperimentError("Empty fit or test split in projected archive.")
    if set(frame.iloc[test]["slide_id"]) & set(frame.iloc[fit]["slide_id"]):
        raise ExperimentError("Slide leakage between fit and test splits.")
    return fit, test


def scanner_probe(features: np.ndarray, frame: pd.DataFrame, fit: np.ndarray, test: np.ndarray):
    labels = frame["scanner_id"].to_numpy()
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=0),
    )
    model.fit(features[fit], labels[fit])
    predictions = model.predict(features[test])
    test_frame = frame.iloc[test].reset_index(drop=True)
    per_slide = (
        pd.DataFrame(
            {
                "slide_id": test_frame["slide_id"],
                "scanner_probe_accuracy": predictions == labels[test],
            }
        )
        .groupby("slide_id", as_index=False)["scanner_probe_accuracy"]
        .mean()
    )
    return float(balanced_accuracy_score(labels[test], predictions)), per_slide


def normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise ExperimentError("Zero-norm feature row found.")
    return features / norms


def paired_metrics(features: np.ndarray, frame: pd.DataFrame, test: np.ndarray):
    normalized = normalize(features[test])
    test_frame = frame.iloc[test].reset_index(drop=True)
    maps = {scanner: {} for scanner in SCANNERS}
    region_to_slide = {}
    for index, row in test_frame.iterrows():
        maps[str(row["scanner_id"])][str(row["region_id"])] = normalized[index]
        region_to_slide[str(row["region_id"])] = str(row["slide_id"])

    cosine_rows = []
    retrieval_rows = []
    pair_rows = []
    for index_a, scanner_a in enumerate(SCANNERS):
        for scanner_b in SCANNERS[index_a + 1 :]:
            pair = f"{scanner_a}__{scanner_b}"
            regions = sorted(set(maps[scanner_a]) & set(maps[scanner_b]))
            if not regions:
                raise ExperimentError(f"No paired test regions for {pair}.")
            a = np.stack([maps[scanner_a][region] for region in regions])
            b = np.stack([maps[scanner_b][region] for region in regions])
            similarity = a @ b.T
            diagonal = np.diag(similarity)
            truth = np.arange(len(regions))
            prediction_ab = np.argmax(similarity, axis=1)
            prediction_ba = np.argmax(similarity.T, axis=1)
            pair_rows.append(
                {
                    "pair": pair,
                    "cosine": float(diagonal.mean()),
                    "retrieval": float(0.5 * (np.mean(prediction_ab == truth) + np.mean(prediction_ba == truth))),
                }
            )
            for region_index, region in enumerate(regions):
                slide = region_to_slide[region]
                cosine_rows.append({"slide_id": slide, "pair": pair, "cosine": float(diagonal[region_index])})
                retrieval_rows.append(
                    {"slide_id": slide, "pair": pair, "correct": float(prediction_ab[region_index] == region_index)}
                )
                retrieval_rows.append(
                    {"slide_id": slide, "pair": pair, "correct": float(prediction_ba[region_index] == region_index)}
                )

    cosine = pd.DataFrame(cosine_rows)
    retrieval = pd.DataFrame(retrieval_rows)
    cosine_by_pair = cosine.groupby(["slide_id", "pair"])["cosine"].mean()
    retrieval_by_pair = retrieval.groupby(["slide_id", "pair"])["correct"].mean()
    slide_rows = []
    for slide_id in sorted(cosine["slide_id"].unique()):
        slide_rows.append(
            {
                "slide_id": slide_id,
                "mean_paired_cosine": float(cosine.loc[cosine["slide_id"] == slide_id, "cosine"].mean()),
                "worst_paired_cosine": float(cosine_by_pair.loc[slide_id].min()),
                "mean_top1_retrieval": float(retrieval.loc[retrieval["slide_id"] == slide_id, "correct"].mean()),
                "worst_top1_retrieval": float(retrieval_by_pair.loc[slide_id].min()),
            }
        )
    pair_frame = pd.DataFrame(pair_rows)
    overall = {
        "mean_paired_cosine": float(pair_frame["cosine"].mean()),
        "worst_paired_cosine": float(pair_frame["cosine"].min()),
        "mean_top1_retrieval": float(pair_frame["retrieval"].mean()),
        "worst_top1_retrieval": float(pair_frame["retrieval"].min()),
    }
    return overall, pd.DataFrame(slide_rows)


def effective_rank(features: np.ndarray) -> float:
    centered = features - features.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    energy = singular_values**2
    if float(energy.sum()) <= 0:
        return 0.0
    probabilities = energy / energy.sum()
    probabilities = probabilities[probabilities > 0]
    return float(math.exp(-np.sum(probabilities * np.log(probabilities))))


def cross_covariance_rms(biological: np.ndarray, acquisition: np.ndarray, test: np.ndarray) -> float:
    b = StandardScaler().fit_transform(biological[test])
    a = StandardScaler().fit_transform(acquisition[test])
    cross = b.T @ a / max(1, len(test) - 1)
    return float(np.sqrt(np.mean(cross**2)))


def load_existing_rows(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    return pd.read_csv(path).to_dict("records")


def atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".csv", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def train_runs(args: argparse.Namespace) -> None:
    base_features, base_frame, source_metadata = load_archive(args.base_features)
    if len(base_features) != 2400 or base_features.shape[1] != 768:
        raise ExperimentError(
            f"Expected DINOv2-Base SCORPION features with shape (2400, 768); observed {base_features.shape}"
        )
    if "dinov2" not in str(source_metadata.get("model", "")).lower():
        raise ExperimentError(f"Feature metadata does not identify DINOv2: {source_metadata.get('model')!r}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ExperimentError("CUDA requested but unavailable.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    training_path = args.out_dir / "training_results.csv"
    rows = load_existing_rows(training_path)
    completed = {
        (int(row["fold"]), int(row["seed"]), str(row["condition"]))
        for row in rows
    }
    design = {
        "stage": "scorpion_pair_integrity_falsification",
        "dataset": "SCORPION",
        "backbone": "DINOv2-Base",
        "base_features": str(args.base_features.resolve()),
        "source_metadata": source_metadata,
        "manifests_dir": str(args.manifests_dir.resolve()),
        "folds": list(args.folds),
        "seeds": list(args.seeds),
        "conditions": list(args.conditions),
        "scanner_adversary_only": "unavailable; no clean existing condition was found in the SCORPION runners",
        "epochs": args.epochs,
        "region_batch_size": args.region_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "device": str(device),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "config": asdict(config_for(base_features.shape[1])),
    }
    (args.out_dir / "experiment_design.json").write_text(
        json.dumps(design, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    pair_audits = []
    assignment_dir = args.out_dir / "pair_assignments"
    for fold in args.folds:
        manifest_path = args.manifests_dir / f"fold_{fold}_manifest.csv"
        features, frame = align_fold(base_features, base_frame, manifest_path)
        fit_indices, test_indices = validate_fold(frame, fold)
        transformed, mean, std = standardize(features, fit_indices)
        fold_dir = args.out_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(fold_dir / "fit_standardization.npz", mean=mean, std=std)

        for seed in args.seeds:
            for condition in args.conditions:
                key = (fold, seed, condition)
                if key in completed:
                    print(f"Skipping completed fold={fold} seed={seed} condition={condition}")
                    continue
                groups, assignments, audit = build_pair_groups(
                    frame, fit_indices, condition=condition, fold=fold, seed=seed
                )
                pair_audits.append(audit)
                assignment_path = assignment_dir / f"fold_{fold}_{condition}_seed_{seed}.csv"
                atomic_csv(assignment_path, assignments)
                print(
                    f"Training fold={fold} seed={seed} condition={condition} "
                    f"mismatch={audit['non_anchor_region_mismatch_fraction']:.3f} "
                    f"same_slide={audit['non_anchor_same_slide_fraction']:.3f}"
                )
                run_dir = fold_dir / "runs" / f"{condition}_seed_{seed}"
                result = train_one(
                    method="pathoalign",
                    seed=seed,
                    features=transformed,
                    frame=frame,
                    train_indices=fit_indices,
                    development_indices=np.arange(len(frame), dtype=np.int64),
                    groups=groups,
                    config=config_for(features.shape[1]),
                    device=device,
                    epochs=args.epochs,
                    region_batch_size=args.region_batch_size,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    run_dir=run_dir,
                )
                mark_projection_metadata(
                    run_dir / "projected_features.npz",
                    {
                        "contains_test_rows": True,
                        "evaluation_stage": "pair_integrity_falsification",
                        "fold": int(fold),
                        "seed": int(seed),
                        "condition": condition,
                        "fit_splits": ["train", "val"],
                        "evaluation_split": "test",
                        "pair_construction_audit": audit,
                        "hyperparameters_frozen": True,
                    },
                )
                rows.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "condition": condition,
                        **result,
                        **audit,
                        "n_fit_slides": int(frame.iloc[fit_indices]["slide_id"].nunique()),
                        "n_test_slides": int(frame.iloc[test_indices]["slide_id"].nunique()),
                    }
                )
                write_results(training_path, rows)
                completed.add(key)

    if pair_audits:
        existing = []
        audit_path = args.out_dir / "pair_construction_audit.csv"
        if audit_path.is_file():
            existing = pd.read_csv(audit_path).to_dict("records")
        atomic_csv(audit_path, pd.DataFrame(existing + pair_audits))


def evaluate_runs(args: argparse.Namespace) -> dict[str, object]:
    training_path = args.out_dir / "training_results.csv"
    if not training_path.is_file():
        raise ExperimentError(f"Missing training results: {training_path}")
    training = pd.read_csv(training_path)
    expected = len(args.folds) * len(args.seeds) * len(args.conditions)
    if len(training) != expected:
        raise ExperimentError(f"Expected {expected} completed fits, observed {len(training)}")
    if training.duplicated(["fold", "seed", "condition"]).any():
        raise ExperimentError("Duplicate fold/seed/condition training rows found.")

    run_rows = []
    slide_rows = []
    for row in training.itertuples(index=False):
        fold = int(row.fold)
        seed = int(row.seed)
        condition = str(row.condition)
        projected = (
            args.out_dir
            / f"fold_{fold}"
            / "runs"
            / f"{condition}_seed_{seed}"
            / "projected_features.npz"
        )
        biological, acquisition, frame, metadata = load_projected(projected)
        if metadata.get("condition") != condition:
            raise ExperimentError(f"Projected metadata condition mismatch in {projected}")
        fit, test = split_indices(frame)
        probe, slide_probe = scanner_probe(biological, frame, fit, test)
        paired, slide_paired = paired_metrics(biological, frame, test)
        test_features = biological[test]
        variances = test_features.var(axis=0)
        cross_covariance = cross_covariance_rms(biological, acquisition, test)
        run_rows.append(
            {
                "fold": fold,
                "seed": seed,
                "condition": condition,
                **paired,
                "scanner_probe_accuracy": probe,
                "effective_rank": effective_rank(test_features),
                "biological_acquisition_cross_covariance": cross_covariance,
                "feature_variance_nonzero_fraction": float(np.mean(variances > 1e-12)),
                "n_test_slides": int(frame.iloc[test]["slide_id"].nunique()),
            }
        )
        merged = slide_paired.merge(slide_probe, on="slide_id", validate="one_to_one")
        merged.insert(0, "seed", seed)
        merged.insert(0, "condition", condition)
        merged.insert(0, "fold", fold)
        slide_rows.extend(merged.to_dict("records"))

    runs = pd.DataFrame(run_rows).sort_values(["condition", "fold", "seed"])
    slides = pd.DataFrame(slide_rows).sort_values(["condition", "slide_id", "seed"])
    atomic_csv(args.out_dir / "raw_run_metrics.csv", runs)
    atomic_csv(args.out_dir / "raw_slide_metrics.csv", slides)

    numeric = [metric for metric in METRICS if metric in runs.columns]
    summary = (
        runs.groupby("condition", as_index=False)[numeric]
        .agg(["mean", "std", "min", "max"])
    )
    summary.columns = [
        "_".join(column).strip("_") if isinstance(column, tuple) else str(column)
        for column in summary.columns
    ]
    summary.insert(
        1,
        "n_runs",
        runs.groupby("condition")["seed"].size().reindex(summary["condition"]).to_numpy(),
    )
    atomic_csv(args.out_dir / "condition_summary.csv", summary)

    slide_means = slides.groupby(["condition", "slide_id"], as_index=False)[
        [
            "scanner_probe_accuracy",
            "mean_paired_cosine",
            "worst_paired_cosine",
            "mean_top1_retrieval",
            "worst_top1_retrieval",
        ]
    ].mean()
    atomic_csv(args.out_dir / "slide_seed_averaged_metrics.csv", slide_means)
    contrasts = []
    true = slide_means[slide_means["condition"] == "true_pairs"].set_index("slide_id")
    if len(true) != 48 and set(args.folds) == set(range(5)):
        raise ExperimentError(f"Full experiment should cover 48 true-pair slide blocks; observed {len(true)}")
    for condition in args.conditions:
        if condition == "true_pairs":
            continue
        other = slide_means[slide_means["condition"] == condition].set_index("slide_id")
        if set(other.index) != set(true.index):
            raise ExperimentError(f"Slide blocks are not matched for {condition}.")
        for metric in [
            "scanner_probe_accuracy",
            "mean_paired_cosine",
            "worst_paired_cosine",
            "mean_top1_retrieval",
            "worst_top1_retrieval",
        ]:
            differences = other.loc[true.index, metric].to_numpy(float) - true[metric].to_numpy(float)
            favorable = differences < 0 if metric in LOWER_IS_BETTER else differences > 0
            contrasts.append(
                {
                    "condition": condition,
                    "metric": metric,
                    "difference_definition": f"{condition}_minus_true_pairs",
                    "n_slide_blocks": int(len(differences)),
                    "mean_difference": float(differences.mean()),
                    "median_difference": float(np.median(differences)),
                    "fraction_blocks_above_true": float(np.mean(differences > 0)),
                    "fraction_blocks_favorable_vs_true": float(np.mean(favorable)),
                }
            )
    contrasts_frame = pd.DataFrame(contrasts)
    atomic_csv(args.out_dir / "slide_blocked_contrasts.csv", contrasts_frame)
    # Also write the requested alternative filename for tools expecting fold-level naming.
    atomic_csv(args.out_dir / "fold_blocked_contrasts.csv", contrasts_frame)

    return {
        "runs": runs,
        "summary": summary,
        "contrasts": contrasts_frame,
        "completed_runs": int(len(runs)),
    }


def interpretation(evaluation: dict[str, object], args: argparse.Namespace, runtime_seconds: float) -> None:
    runs: pd.DataFrame = evaluation["runs"]
    summary: pd.DataFrame = evaluation["summary"]
    contrasts: pd.DataFrame = evaluation["contrasts"]
    means = runs.groupby("condition")[list(METRICS)].mean(numeric_only=True)
    required_conditions = set(args.conditions)
    smoke_passed = required_conditions.issubset(set(means.index)) and len(runs) == len(args.folds) * len(args.seeds) * len(args.conditions)

    def fmt(value: float) -> str:
        if pd.isna(value):
            return "NA"
        return f"{float(value):.6f}"

    table_lines = [
        "| Condition | Scanner probe | Mean paired cosine | Worst paired cosine | Mean top-1 retrieval | Worst top-1 retrieval | Effective rank | Bio/acq cross-cov RMS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in args.conditions:
        if condition not in means.index:
            continue
        row = means.loc[condition]
        table_lines.append(
            "| "
            + " | ".join(
                [
                    condition,
                    fmt(row.get("scanner_probe_accuracy")),
                    fmt(row.get("mean_paired_cosine")),
                    fmt(row.get("worst_paired_cosine")),
                    fmt(row.get("mean_top1_retrieval")),
                    fmt(row.get("worst_top1_retrieval")),
                    fmt(row.get("effective_rank")),
                    fmt(row.get("biological_acquisition_cross_covariance")),
                ]
            )
            + " |"
        )

    true_beats = []
    if "true_pairs" in means.index:
        true_row = means.loc["true_pairs"]
        for condition in [c for c in args.conditions if c != "true_pairs" and c in means.index]:
            other = means.loc[condition]
            scanner_similar = abs(other["scanner_probe_accuracy"] - true_row["scanner_probe_accuracy"]) <= 0.03
            tissue_damage = (
                other["mean_top1_retrieval"] < true_row["mean_top1_retrieval"] - 0.005
                or other["mean_paired_cosine"] < true_row["mean_paired_cosine"] - 0.005
            )
            true_better_tissue = (
                true_row["mean_top1_retrieval"] >= other["mean_top1_retrieval"]
                and true_row["mean_paired_cosine"] >= other["mean_paired_cosine"]
            )
            true_beats.append(
                f"- `{condition}`: scanner_similar_to_true={scanner_similar}; "
                f"tissue_damage_vs_true={tissue_damage}; true_better_tissue_metrics={true_better_tissue}."
            )

    verdict = "incomplete"
    if smoke_passed and len(args.folds) == 5 and len(args.seeds) == 5:
        if true_beats:
            verdict = "peer-review-hardening"
    elif smoke_passed:
        verdict = "smoke-passed"

    markdown = f"""# SCORPION Pair-Integrity Falsification Summary

## Run Status

- Dataset: SCORPION
- Backbone: DINOv2-Base
- Seeds: {', '.join(map(str, args.seeds))}
- Folds: {', '.join(map(str, args.folds))}
- Conditions: {', '.join(args.conditions)}
- Runtime seconds: {runtime_seconds:.1f}
- Completed runs: {len(runs)} / {len(args.folds) * len(args.seeds) * len(args.conditions)}
- Smoke/full pass status for this command: {smoke_passed}
- Scanner-adversary-only condition: unavailable; no clean existing SCORPION implementation was found, so it was not added.

## Main Result Table

{chr(10).join(table_lines)}

## Pair-Integrity Falsification Logic

Expected result: true pairs should reduce scanner probe while preserving or improving tissue identity metrics.

Falsification logic: if shuffled pairs perform similarly to true pairs, the paired-acquisition claim is weakened. If shuffled pairs reduce scanner signal but damage tissue identity metrics, the claim is strengthened. If true pairs fail, that failure should be reported honestly.

## True-Pair Comparison

{chr(10).join(true_beats) if true_beats else '- True-pair comparison unavailable.'}

## Claim Boundary

This experiment does not prove clinical robustness, diagnosis, disease biology, deployment readiness, complete scanner invariance, or perfect disentanglement. It only tests whether the factorization effect depends on true pair integrity.

## Metric Availability

- scanner_probe_accuracy: available.
- mean_paired_cosine: available.
- worst_paired_cosine: available.
- mean_top1_retrieval: available.
- worst_top1_retrieval: available.
- effective_rank: available.
- biological_acquisition_cross_covariance: available as normalized biological/acquisition cross-covariance RMS on held-out test slides.

## Readiness

Current classification: {verdict}.

## Artifacts

- raw_run_metrics.csv
- condition_summary.csv
- slide_blocked_contrasts.csv
- fold_blocked_contrasts.csv
- pair_integrity_falsification_summary.md
- run_log.txt
- pair_construction_audit.csv
- experiment_design.json

## Exact Retry Command

```powershell
python experiments/scorpion/run_pair_integrity_falsification.py --base-features results/scorpion/features/fold_0_dinov2_base.npz --manifests-dir data/scorpion/splits --out-dir {args.out_dir.as_posix()} --seeds {' '.join(map(str, args.seeds))} --folds {' '.join(map(str, args.folds))} --conditions {' '.join(args.conditions)} --epochs {args.epochs} --region-batch-size {args.region_batch_size} --learning-rate {args.learning_rate} --weight-decay {args.weight_decay} --device {args.device}
```
"""
    (args.out_dir / "pair_integrity_falsification_summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print("\nCONDITION SUMMARY")
    print(summary.to_string(index=False))
    print("\nSLIDE-BLOCKED CONTRASTS")
    print(contrasts.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-features", type=Path, required=True)
    parser.add_argument("--manifests-dir", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/paired_acquisition_factorization_pair_integrity_scorpion"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[701, 702, 703, 704, 705])
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    parser.add_argument("--conditions", nargs="+", choices=CONDITIONS, default=list(CONDITIONS))
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--region-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "run_log.txt"
    start = time.time()
    command = "python " + " ".join(sys.argv)
    with log_path.open("a", encoding="utf-8") as log_handle:
        print(f"\n=== pair-integrity run start {time.strftime('%Y-%m-%d %H:%M:%S')} ===", file=log_handle)
        print(f"command: {command}", file=log_handle)
        log_handle.flush()
        with redirect_stdout(Tee(sys.stdout, log_handle)), redirect_stderr(Tee(sys.stderr, log_handle)):
            try:
                print(f"Command: {command}")
                train_runs(args)
                evaluation = evaluate_runs(args)
                interpretation(evaluation, args, time.time() - start)
                print(f"Run completed in {time.time() - start:.1f} seconds")
            except Exception as exc:
                failure_path = args.out_dir / "failure_report.md"
                tb = traceback.format_exc()
                likely = "See traceback; likely causes are missing SCORPION inputs, CUDA/runtime failure, or invalid pair construction."
                failure_path.write_text(
                    "\n".join(
                        [
                            "# Pair-Integrity Falsification Failure",
                            "",
                            f"Command: `{command}`",
                            "",
                            "## Error",
                            "",
                            f"```text\n{exc}\n```",
                            "",
                            "## Traceback",
                            "",
                            f"```text\n{tb}\n```",
                            "",
                            "## Likely Cause",
                            "",
                            likely,
                            "",
                            "## Next Retry Command",
                            "",
                            f"```powershell\n{command}\n```",
                            "",
                        ]
                    ),
                    encoding="utf-8",
                )
                print(f"PAIR-INTEGRITY FALSIFICATION FAILED: {exc}", file=sys.stderr)
                print(tb, file=sys.stderr)
                raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
