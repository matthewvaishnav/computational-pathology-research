#!/usr/bin/env python3
"""Run canine SCC pair-integrity falsification for paired-acquisition factorization."""

from __future__ import annotations

import argparse
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

from experiments.external_multiscanner.run_canine_pathoalign_crossfold import (  # noqa: E402
    align_fold,
    patch_scanner_namespace,
    validate_fold,
)
from experiments.scorpion import run_pathoalign_projection as projection  # noqa: E402
from src.models.scorpion_pathoalign import ProjectionConfig  # noqa: E402


SCANNERS = ("cs2", "gt450", "nz20", "nz210", "p1000")
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
LOWER_IS_BETTER = {"scanner_probe_accuracy", "biological_acquisition_cross_covariance"}


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


def deterministic_seed(condition: str, fold: int, seed: int) -> int:
    payload = f"caninescc|{condition}|fold={fold}|seed={seed}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


def derange_indices(
    candidates: list[int],
    *,
    rng: np.random.Generator,
    forbidden: dict[int, set[int]],
    max_attempts: int = 10000,
) -> list[int]:
    if len(candidates) <= 1:
        raise projection.ExperimentError("Cannot derange fewer than two candidates.")
    base = np.asarray(candidates, dtype=np.int64)
    for _ in range(max_attempts):
        proposed = rng.permutation(base)
        if all(int(value) not in forbidden[int(anchor)] for anchor, value in zip(base, proposed)):
            return [int(value) for value in proposed]
    raise projection.ExperimentError("Failed to construct deterministic derangement.")


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


def complete_region_table(frame: pd.DataFrame, fit_indices: np.ndarray) -> pd.DataFrame:
    subset = frame.iloc[fit_indices].copy()
    subset["scanner_id"] = subset["scanner_id"].astype(str).str.lower()
    subset["_row_index"] = subset.index.to_numpy(dtype=np.int64)
    rows = []
    for region_index, (region_id, group) in enumerate(subset.groupby("region_id", sort=True)):
        if len(group) != len(SCANNERS) or set(group["scanner_id"]) != set(SCANNERS):
            raise projection.ExperimentError("Every fitting region must contain all five canine scanners.")
        sample_values = set(group["slide_id"].astype(str))
        if "sample_id" in group.columns:
            sample_values |= set(group["sample_id"].astype(str))
        sample_values = {value for value in sample_values if value}
        if len(sample_values) != 1:
            raise projection.ExperimentError(f"Region metadata is not sample-unique for {region_id}.")
        row = {
            "region_index": int(region_index),
            "region_id": str(region_id),
            "sample_id": next(iter(sample_values)),
        }
        for scanner in SCANNERS:
            scanner_rows = group.loc[group["scanner_id"] == scanner, "_row_index"]
            row[scanner] = int(scanner_rows.iloc[0])
        rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        raise projection.ExperimentError("No fitting regions were available.")
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
        raise projection.ExperimentError(f"Unknown condition: {condition}")
    table = complete_region_table(frame, fit_indices)
    rng = np.random.default_rng(deterministic_seed(condition, fold, seed))
    assignments = pd.DataFrame(
        {
            "fold": fold,
            "seed": seed,
            "condition": condition,
            "anchor_region_index": table["region_index"],
            "anchor_region_id": table["region_id"],
            "anchor_sample_id": table["sample_id"],
        }
    )

    for scanner in SCANNERS:
        assignments[f"{scanner}_region_index"] = table["region_index"]
        assignments[f"{scanner}_region_id"] = table["region_id"]
        assignments[f"{scanner}_sample_id"] = table["sample_id"]
        assignments[f"{scanner}_row_index"] = table[scanner]

    if condition == "shuffled_region_pairs":
        candidates = [int(value) for value in table["region_index"]]
        forbidden = {candidate: {candidate} for candidate in candidates}
        for scanner in SCANNERS[1:]:
            chosen = np.asarray(derange_indices(candidates, rng=rng, forbidden=forbidden), dtype=np.int64)
            source = table.set_index("region_index").loc[chosen]
            assignments[f"{scanner}_region_index"] = chosen
            assignments[f"{scanner}_region_id"] = source["region_id"].to_numpy()
            assignments[f"{scanner}_sample_id"] = source["sample_id"].to_numpy()
            assignments[f"{scanner}_row_index"] = source[scanner].to_numpy(dtype=np.int64)

    if condition == "shuffled_sample_pairs":
        samples = sorted(table["sample_id"].astype(str).unique())
        forbidden_samples = {index: {index} for index in range(len(samples))}
        sample_order = derange_indices(
            list(range(len(samples))), rng=rng, forbidden=forbidden_samples
        )
        sample_map = {samples[index]: samples[sample_order[index]] for index in range(len(samples))}
        for scanner in SCANNERS[1:]:
            chosen = np.empty(len(table), dtype=np.int64)
            for source_sample, target_sample in sample_map.items():
                anchors = table.loc[
                    table["sample_id"] == source_sample, "region_index"
                ].to_numpy(dtype=np.int64)
                targets = table.loc[
                    table["sample_id"] == target_sample, "region_index"
                ].to_numpy(dtype=np.int64)
                if len(targets) == 0:
                    raise projection.ExperimentError("Sample shuffle selected an empty target sample.")
                shuffled_targets = rng.choice(targets, size=len(anchors), replace=len(targets) < len(anchors))
                for anchor, assigned in zip(anchors, shuffled_targets):
                    chosen[int(anchor)] = int(assigned)
            source = table.set_index("region_index").loc[chosen]
            assignments[f"{scanner}_region_index"] = chosen
            assignments[f"{scanner}_region_id"] = source["region_id"].to_numpy()
            assignments[f"{scanner}_sample_id"] = source["sample_id"].to_numpy()
            assignments[f"{scanner}_row_index"] = source[scanner].to_numpy(dtype=np.int64)

    groups = [
        np.asarray([getattr(row, f"{scanner}_row_index") for scanner in SCANNERS], dtype=np.int64)
        for row in assignments.itertuples(index=False)
    ]
    region_mismatch = np.column_stack(
        [
            assignments[f"{scanner}_region_id"].astype(str).to_numpy()
            != assignments["anchor_region_id"].astype(str).to_numpy()
            for scanner in SCANNERS[1:]
        ]
    )
    same_sample = np.column_stack(
        [
            assignments[f"{scanner}_sample_id"].astype(str).to_numpy()
            == assignments["anchor_sample_id"].astype(str).to_numpy()
            for scanner in SCANNERS[1:]
        ]
    )
    audit = {
        "condition": condition,
        "fold": int(fold),
        "seed": int(seed),
        "n_fit_regions": int(len(table)),
        "n_pair_groups": int(len(groups)),
        "anchor_scanner": SCANNERS[0],
        "deterministic_shuffle_seed": int(deterministic_seed(condition, fold, seed)),
        "non_anchor_region_mismatch_fraction": float(region_mismatch.mean()),
        "non_anchor_same_sample_fraction": float(same_sample.mean()),
        "unique_training_rows_used": int(len(np.unique(np.concatenate(groups)))),
        "expected_training_rows_used": int(len(table) * len(SCANNERS)),
    }
    if condition in {"true_pairs", "shuffled_region_pairs"}:
        if audit["unique_training_rows_used"] != audit["expected_training_rows_used"]:
            raise projection.ExperimentError(f"Pair construction reused or dropped rows: {audit}")
    if condition == "true_pairs" and audit["non_anchor_region_mismatch_fraction"] != 0.0:
        raise projection.ExperimentError("True pairs unexpectedly mismatched regions.")
    if condition == "shuffled_region_pairs":
        if audit["non_anchor_region_mismatch_fraction"] != 1.0:
            raise projection.ExperimentError("Region-shuffled pairs did not fully break region identity.")
    if condition == "shuffled_sample_pairs":
        if audit["non_anchor_region_mismatch_fraction"] != 1.0:
            raise projection.ExperimentError("Sample-shuffled pairs did not fully break region identity.")
        if audit["non_anchor_same_sample_fraction"] != 0.0:
            raise projection.ExperimentError("Sample-shuffled pairs did not fully break sample identity.")
    return groups, assignments, audit


def mark_projection_metadata(path: Path, metadata_update: dict[str, object]) -> None:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    metadata.update(metadata_update)
    text = json.dumps(metadata, sort_keys=True)
    arrays["metadata_json"] = np.asarray(text, dtype=f"<U{len(text)}")
    projection.atomic_npz(path, arrays)


def load_projected(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        required = {"features", "acquisition_features", "slide_id", "region_id", "scanner_id", "split"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise projection.ExperimentError(f"{path} is missing arrays: {missing}")
        biological = np.asarray(archive["features"], dtype=np.float32)
        acquisition = np.asarray(archive["acquisition_features"], dtype=np.float32)
        frame = pd.DataFrame(
            {name: archive[name].astype(str) for name in ("slide_id", "region_id", "scanner_id", "split")}
        )
        frame["scanner_id"] = frame["scanner_id"].astype(str).str.lower()
        metadata = json.loads(str(archive["metadata_json"].item()))
    if len(biological) != len(frame) or len(acquisition) != len(frame):
        raise projection.ExperimentError("Projected feature arrays and metadata are misaligned.")
    if set(frame["scanner_id"].unique()) != set(SCANNERS):
        raise projection.ExperimentError(f"Unexpected scanner set in {path}.")
    return biological, acquisition, frame, metadata


def split_indices(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    test = np.flatnonzero(frame["split"].to_numpy() == "test")
    fit = np.flatnonzero(frame["split"].to_numpy() != "test")
    if len(test) == 0 or len(fit) == 0:
        raise projection.ExperimentError("Empty fit or test split in projected archive.")
    if set(frame.iloc[test]["slide_id"]) & set(frame.iloc[fit]["slide_id"]):
        raise projection.ExperimentError("Biological sample leakage between fit and test splits.")
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
    per_sample = (
        pd.DataFrame({"sample_id": test_frame["slide_id"], "scanner_probe_accuracy": predictions == labels[test]})
        .groupby("sample_id", as_index=False)["scanner_probe_accuracy"]
        .mean()
    )
    return float(balanced_accuracy_score(labels[test], predictions)), per_sample


def normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    if np.any(norms <= 0):
        raise projection.ExperimentError("Zero-norm feature row found.")
    return features / norms


def paired_metrics(features: np.ndarray, frame: pd.DataFrame, test: np.ndarray):
    normalized = normalize(features[test])
    test_frame = frame.iloc[test].reset_index(drop=True)
    maps = {scanner: {} for scanner in SCANNERS}
    region_to_sample = {}
    for index, row in test_frame.iterrows():
        maps[str(row["scanner_id"]).lower()][str(row["region_id"])] = normalized[index]
        region_to_sample[str(row["region_id"])] = str(row["slide_id"])

    cosine_rows = []
    retrieval_rows = []
    pair_rows = []
    for i, scanner_a in enumerate(SCANNERS):
        for scanner_b in SCANNERS[i + 1 :]:
            pair = f"{scanner_a}__{scanner_b}"
            regions = sorted(set(maps[scanner_a]) & set(maps[scanner_b]))
            if not regions:
                raise projection.ExperimentError(f"No paired test regions for {pair}.")
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
                sample = region_to_sample[region]
                cosine_rows.append({"sample_id": sample, "pair": pair, "cosine": float(diagonal[region_index])})
                retrieval_rows.append({"sample_id": sample, "pair": pair, "correct": float(prediction_ab[region_index] == region_index)})
                retrieval_rows.append({"sample_id": sample, "pair": pair, "correct": float(prediction_ba[region_index] == region_index)})

    cosine = pd.DataFrame(cosine_rows)
    retrieval = pd.DataFrame(retrieval_rows)
    cosine_by_pair = cosine.groupby(["sample_id", "pair"])["cosine"].mean()
    retrieval_by_pair = retrieval.groupby(["sample_id", "pair"])["correct"].mean()
    sample_rows = []
    for sample_id in sorted(cosine["sample_id"].unique()):
        sample_rows.append(
            {
                "sample_id": sample_id,
                "mean_paired_cosine": float(cosine.loc[cosine["sample_id"] == sample_id, "cosine"].mean()),
                "worst_paired_cosine": float(cosine_by_pair.loc[sample_id].min()),
                "mean_top1_retrieval": float(retrieval.loc[retrieval["sample_id"] == sample_id, "correct"].mean()),
                "worst_top1_retrieval": float(retrieval_by_pair.loc[sample_id].min()),
            }
        )
    pair_frame = pd.DataFrame(pair_rows)
    return {
        "mean_paired_cosine": float(pair_frame["cosine"].mean()),
        "worst_paired_cosine": float(pair_frame["cosine"].min()),
        "mean_top1_retrieval": float(pair_frame["retrieval"].mean()),
        "worst_top1_retrieval": float(pair_frame["retrieval"].min()),
    }, pd.DataFrame(sample_rows)


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


def write_results(path: Path, rows: list[dict[str, object]]) -> None:
    if rows:
        pd.DataFrame(rows).sort_values(["fold", "seed", "condition"]).to_csv(path, index=False)


def train_runs(args: argparse.Namespace) -> None:
    patch_scanner_namespace()
    base_features, base_frame, source_metadata = projection.load_archive(args.base_features)
    base_frame["scanner_id"] = base_frame["scanner_id"].astype(str).str.lower()
    if len(base_features) != 4025 or base_features.shape[1] != 768:
        raise projection.ExperimentError(
            f"Expected canine DINOv2 features with shape (4025, 768); observed {base_features.shape}"
        )
    if "dinov2" not in str(source_metadata.get("model", "")).lower():
        raise projection.ExperimentError(f"Feature metadata does not identify DINOv2: {source_metadata.get('model')!r}")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise projection.ExperimentError("CUDA requested but unavailable.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    training_path = args.out_dir / "training_results.csv"
    rows = load_existing_rows(training_path)
    completed = {(int(row["fold"]), int(row["seed"]), str(row["condition"])) for row in rows}
    design = {
        "stage": "canine_scc_pair_integrity_falsification",
        "dataset": "external_multiscanner_caninescc",
        "backbone": "DINOv2-Base",
        "base_features": str(args.base_features.resolve()),
        "source_metadata": source_metadata,
        "manifests_dir": str(args.manifests_dir.resolve()),
        "folds": list(args.folds),
        "seeds": list(args.seeds),
        "conditions": list(args.conditions),
        "scanner_namespace": list(SCANNERS),
        "scanner_adversary_only": "unavailable; no clean existing canine condition was found",
        "epochs": args.epochs,
        "region_batch_size": args.region_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "device": str(device),
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "config": asdict(config_for(base_features.shape[1])),
    }
    (args.out_dir / "experiment_design.json").write_text(json.dumps(design, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    pair_audits = []
    assignment_dir = args.out_dir / "pair_assignments"
    for fold in args.folds:
        manifest_path = args.manifests_dir / f"fold_{fold}_patch_manifest.csv"
        features, frame = align_fold(base_features, base_frame, manifest_path)
        fit_indices, test_indices = validate_fold(frame, fold)
        transformed, mean, std = projection.standardize(features, fit_indices)
        fold_dir = args.out_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(fold_dir / "fit_standardization.npz", mean=mean, std=std)

        for seed in args.seeds:
            for condition in args.conditions:
                key = (fold, seed, condition)
                if key in completed:
                    print(f"Skipping completed fold={fold} seed={seed} condition={condition}")
                    continue
                groups, assignments, audit = build_pair_groups(frame, fit_indices, condition=condition, fold=fold, seed=seed)
                pair_audits.append(audit)
                atomic_csv(assignment_dir / f"fold_{fold}_{condition}_seed_{seed}.csv", assignments)
                print(
                    f"Training fold={fold} seed={seed} condition={condition} "
                    f"mismatch={audit['non_anchor_region_mismatch_fraction']:.3f} "
                    f"same_sample={audit['non_anchor_same_sample_fraction']:.3f}"
                )
                run_dir = fold_dir / "runs" / f"{condition}_seed_{seed}"
                result = projection.train_one(
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
                audit_for_metadata = dict(audit)
                mark_projection_metadata(
                    run_dir / "projected_features.npz",
                    {
                        "contains_test_rows": True,
                        "evaluation_stage": "canine_pair_integrity_falsification",
                        "fold": int(fold),
                        "seed": int(seed),
                        "condition": condition,
                        "fit_splits": ["train", "val"],
                        "evaluation_split": "test",
                        "pair_construction_audit": audit_for_metadata,
                        "hyperparameters_frozen": True,
                        "scanner_namespace": list(SCANNERS),
                    },
                )
                rows.append(
                    {
                        "fold": fold,
                        "seed": seed,
                        "condition": condition,
                        **result,
                        **audit,
                        "n_fit_samples": int(frame.iloc[fit_indices]["slide_id"].nunique()),
                        "n_test_samples": int(frame.iloc[test_indices]["slide_id"].nunique()),
                        "n_fit_regions": int(frame.iloc[fit_indices]["region_id"].nunique()),
                        "n_test_regions": int(frame.iloc[test_indices]["region_id"].nunique()),
                    }
                )
                write_results(training_path, rows)
                completed.add(key)

    if pair_audits:
        audit_path = args.out_dir / "pair_construction_audit.csv"
        existing = pd.read_csv(audit_path).to_dict("records") if audit_path.is_file() else []
        atomic_csv(audit_path, pd.DataFrame(existing + pair_audits))


def evaluate_runs(args: argparse.Namespace) -> dict[str, object]:
    training_path = args.out_dir / "training_results.csv"
    if not training_path.is_file():
        raise projection.ExperimentError(f"Missing training results: {training_path}")
    training = pd.read_csv(training_path)
    expected = len(args.folds) * len(args.seeds) * len(args.conditions)
    if len(training) != expected:
        raise projection.ExperimentError(f"Expected {expected} completed fits, observed {len(training)}")
    if training.duplicated(["fold", "seed", "condition"]).any():
        raise projection.ExperimentError("Duplicate fold/seed/condition training rows found.")

    run_rows = []
    sample_rows = []
    for row in training.itertuples(index=False):
        fold = int(row.fold)
        seed = int(row.seed)
        condition = str(row.condition)
        projected = args.out_dir / f"fold_{fold}" / "runs" / f"{condition}_seed_{seed}" / "projected_features.npz"
        biological, acquisition, frame, metadata = load_projected(projected)
        if metadata.get("condition") != condition:
            raise projection.ExperimentError(f"Projected metadata condition mismatch in {projected}")
        fit, test = split_indices(frame)
        probe, sample_probe = scanner_probe(biological, frame, fit, test)
        paired, sample_paired = paired_metrics(biological, frame, test)
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
                "n_test_samples": int(frame.iloc[test]["slide_id"].nunique()),
                "n_test_regions": int(frame.iloc[test]["region_id"].nunique()),
            }
        )
        merged = sample_paired.merge(sample_probe, on="sample_id", validate="one_to_one")
        merged.insert(0, "seed", seed)
        merged.insert(0, "condition", condition)
        merged.insert(0, "fold", fold)
        sample_rows.extend(merged.to_dict("records"))

    runs = pd.DataFrame(run_rows).sort_values(["condition", "fold", "seed"])
    samples = pd.DataFrame(sample_rows).sort_values(["condition", "sample_id", "seed"])
    atomic_csv(args.out_dir / "raw_run_metrics.csv", runs)
    atomic_csv(args.out_dir / "raw_sample_metrics.csv", samples)

    numeric = [metric for metric in METRICS if metric in runs.columns]
    summary = runs.groupby("condition", as_index=False)[numeric].agg(["mean", "std", "min", "max"])
    summary.columns = ["_".join(column).strip("_") if isinstance(column, tuple) else str(column) for column in summary.columns]
    summary.insert(1, "n_runs", runs.groupby("condition")["seed"].size().reindex(summary["condition"]).to_numpy())
    atomic_csv(args.out_dir / "condition_summary.csv", summary)

    sample_means = samples.groupby(["condition", "sample_id"], as_index=False)[
        [
            "scanner_probe_accuracy",
            "mean_paired_cosine",
            "worst_paired_cosine",
            "mean_top1_retrieval",
            "worst_top1_retrieval",
        ]
    ].mean()
    atomic_csv(args.out_dir / "sample_seed_averaged_metrics.csv", sample_means)
    true = sample_means[sample_means["condition"] == "true_pairs"].set_index("sample_id")
    if len(true) != 44 and set(args.folds) == set(range(5)):
        raise projection.ExperimentError(f"Full experiment should cover 44 true-pair sample blocks; observed {len(true)}")
    contrasts = []
    for condition in args.conditions:
        if condition == "true_pairs":
            continue
        other = sample_means[sample_means["condition"] == condition].set_index("sample_id")
        if set(other.index) != set(true.index):
            raise projection.ExperimentError(f"Sample blocks are not matched for {condition}.")
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
                    "n_sample_blocks": int(len(differences)),
                    "mean_difference": float(differences.mean()),
                    "median_difference": float(np.median(differences)),
                    "fraction_blocks_above_true": float(np.mean(differences > 0)),
                    "fraction_blocks_favorable_vs_true": float(np.mean(favorable)),
                }
            )
    contrasts_frame = pd.DataFrame(contrasts)
    atomic_csv(args.out_dir / "sample_blocked_contrasts.csv", contrasts_frame)
    return {"runs": runs, "summary": summary, "contrasts": contrasts_frame, "completed_runs": int(len(runs))}


def interpretation(evaluation: dict[str, object], args: argparse.Namespace, runtime_seconds: float) -> None:
    runs: pd.DataFrame = evaluation["runs"]
    summary: pd.DataFrame = evaluation["summary"]
    contrasts: pd.DataFrame = evaluation["contrasts"]
    means = runs.groupby("condition")[list(METRICS)].mean(numeric_only=True)
    command_conditions = [condition for condition in args.conditions if condition in means.index]
    complete = len(runs) == len(args.folds) * len(args.seeds) * len(args.conditions)

    def fmt(value: float) -> str:
        return "NA" if pd.isna(value) else f"{float(value):.6f}"

    table_lines = [
        "| Condition | Scanner probe | Mean paired cosine | Worst paired cosine | Mean top-1 retrieval | Worst top-1 retrieval | Effective rank | Bio/acq cross-cov RMS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for condition in command_conditions:
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
        for condition in [c for c in command_conditions if c != "true_pairs"]:
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

    readiness = "incomplete"
    if complete and len(args.folds) == 5 and len(args.seeds) == 5:
        readiness = "peer-review-hardening"
    elif complete:
        readiness = "smoke-passed"

    markdown = f"""# Canine SCC Pair-Integrity Falsification Summary

## Run Status

- Dataset: external multi-scanner canine cutaneous SCC
- Backbone: DINOv2-Base
- Seeds: {', '.join(map(str, args.seeds))}
- Folds: {', '.join(map(str, args.folds))}
- Conditions: {', '.join(args.conditions)}
- Runtime seconds: {runtime_seconds:.1f}
- Completed runs: {len(runs)} / {len(args.folds) * len(args.seeds) * len(args.conditions)}
- Smoke/full pass status for this command: {complete}
- Scanner-adversary-only condition: unavailable; no clean existing canine implementation was found, so it was not added.

## Main Result Table

{chr(10).join(table_lines)}

## Pair-Integrity Falsification Logic

Expected result: true pairs should preserve tissue identity metrics better than shuffled-pair controls.

Falsification logic: if shuffled pairs suppress scanner signal but damage paired-tissue consistency/retrieval, that supports the interpretation that true same-tissue pairing matters. If shuffled pairs perform similarly to true pairs on tissue preservation, the paired-acquisition claim is weakened and must be reported honestly. If true pairs fail, report failure honestly.

## True-Pair Comparison

{chr(10).join(true_beats) if true_beats else '- True-pair comparison unavailable.'}

## Claim Boundary

This does not prove clinical robustness, diagnosis, disease biology discovery, human clinical generalization from canine SCC, deployment readiness, complete scanner invariance, or perfect disentanglement. It is an external pair-structure falsification control.

## Metric Availability

- scanner_probe_accuracy: available.
- mean_paired_cosine: available.
- worst_paired_cosine: available.
- mean_top1_retrieval: available.
- worst_top1_retrieval: available.
- effective_rank: available.
- biological_acquisition_cross_covariance: available as normalized biological/acquisition cross-covariance RMS on held-out test samples.

## Readiness

Current classification: {readiness}.

## Artifacts

- raw_run_metrics.csv
- condition_summary.csv
- sample_blocked_contrasts.csv
- pair_integrity_falsification_summary.md
- run_log.txt
- pair_construction_audit.csv
- experiment_design.json

## Exact Retry Command

```powershell
python experiments/canine/run_pair_integrity_falsification_caninescc.py --base-features {args.base_features.as_posix()} --manifests-dir {args.manifests_dir.as_posix()} --out-dir {args.out_dir.as_posix()} --seeds {' '.join(map(str, args.seeds))} --folds {' '.join(map(str, args.folds))} --conditions {' '.join(args.conditions)} --epochs {args.epochs} --region-batch-size {args.region_batch_size} --learning-rate {args.learning_rate} --weight-decay {args.weight_decay} --device {args.device}
```
"""
    (args.out_dir / "pair_integrity_falsification_summary.md").write_text(markdown, encoding="utf-8")
    print(markdown)
    print("\nCONDITION SUMMARY")
    print(summary.to_string(index=False))
    print("\nSAMPLE-BLOCKED CONTRASTS")
    print(contrasts.to_string(index=False))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", "--base-features", dest="base_features", type=Path, required=True)
    parser.add_argument("--manifest", "--manifests-dir", dest="manifests_dir", type=Path, required=True)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/paired_acquisition_factorization_pair_integrity_caninescc"),
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[911, 912, 913, 914, 915])
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
        print(f"\n=== canine pair-integrity run start {time.strftime('%Y-%m-%d %H:%M:%S')} ===", file=log_handle)
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
                tb = traceback.format_exc()
                failure_path = args.out_dir / "failure_report.md"
                failure_path.write_text(
                    "\n".join(
                        [
                            "# Canine SCC Pair-Integrity Falsification Failure",
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
                            "See traceback; likely causes are missing canine SCC inputs, CUDA/runtime failure, or invalid pair construction.",
                            "",
                            "## Next Retry Command",
                            "",
                            f"```powershell\n{command}\n```",
                            "",
                        ]
                    ),
                    encoding="utf-8",
                )
                print(f"CANINE SCC PAIR-INTEGRITY FALSIFICATION FAILED: {exc}", file=sys.stderr)
                print(tb, file=sys.stderr)
                raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
