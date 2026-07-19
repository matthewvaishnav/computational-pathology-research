#!/usr/bin/env python3
"""Pair-structure boundary test for Paired-Acquisition Neural Factorization.

Scientific question:
  How exact does the paired-acquisition structure need to be for the method
  to preserve tissue identity while reducing scanner/acquisition signal?

The experiment compares a ladder of pairing conditions, from strictest
biological correspondence to loosest, and measures how tissue-identity
preservation and scanner suppression degrade.

Pairing ladder (SCORPION):
  1. true_same_region_pairs      — same tissue region, 5 scanner views
  2. same_slide_different_region — same slide, different tissue region
  3. shuffled_sample_pairs       — different slide (existing condition)
  4. scanner_balanced_random     — random regions across all slides,
                                   preserving scanner assignment structure
  5. fully_random_pairs          — no anchor, all views random

Pairing ladder (canine SCC — adds):
  6. same_category_different_sample — same tissue category, different slide

Conditions 1–3 reuse existing trained models.
Conditions 4–6 require new training (lighter: fewer seeds per smoke check).
"""

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
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

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
    align_fold as scorpion_align_fold,
    atomic_npz,
    validate_fold as scorpion_validate_fold,
)
from experiments.external_multiscanner.run_canine_pathoalign_crossfold import (
    align_fold as canine_align_fold,
    patch_scanner_namespace as canine_patch_namespace,
    validate_fold as canine_validate_fold,
)
from experiments.scorpion.run_pathoalign_projection import (
    ExperimentError,
    load_archive,
    standardize,
    train_one,
    write_results,
)
from src.models.scorpion_pathoalign import ProjectionConfig

# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------

SCORPION_SCANNERS = ("AT2", "GT450", "DP200", "P1000", "B300")
CANINE_SCANNERS = ("cs2", "gt450", "nz20", "nz210", "p1000")

DATASETS = {
    "SCORPION_DINOv2": {
        "base_features": "results/scorpion/features/fold_0_dinov2_base.npz",
        "manifests_dir": "data/scorpion/splits",
        "manifest_pattern": "fold_{fold}_manifest.csv",
        "scanners": SCORPION_SCANNERS,
        "folds": [0, 1, 2, 3, 4],
        "seeds": [701, 702, 703, 704, 705],
        "existing_result_dir": "results/paired_acquisition_factorization_pair_integrity_scorpion",
        "input_dim": 768,
        "feature_dim": 768,
        "n_images": 2400,
        "align_fn": scorpion_align_fold,
        "validate_fn": scorpion_validate_fold,
        "patch_fn": None,
        "has_category": False,
        "block_label": "slide_id",
    },
    "SCORPION_Phikon": {
        "base_features": "results/scorpion/features/fold_0_phikon.npz",
        "manifests_dir": "data/scorpion/splits",
        "manifest_pattern": "fold_{fold}_manifest.csv",
        "scanners": SCORPION_SCANNERS,
        "folds": [0, 1, 2, 3, 4],
        "seeds": [701, 702, 703, 704, 705],
        "existing_result_dir": "results/paired_acquisition_factorization_pair_integrity_scorpion_phikon",
        "input_dim": 768,
        "feature_dim": 768,
        "n_images": 2400,
        "align_fn": scorpion_align_fold,
        "validate_fn": scorpion_validate_fold,
        "patch_fn": None,
        "has_category": False,
        "block_label": "slide_id",
    },
    "SCORPION_ResNet50": {
        "base_features": "results/scorpion/features/fold_0_resnet50_imagenet.npz",
        "manifests_dir": "data/scorpion/splits",
        "manifest_pattern": "fold_{fold}_manifest.csv",
        "scanners": SCORPION_SCANNERS,
        "folds": [0, 1, 2, 3, 4],
        "seeds": [701, 702, 703, 704, 705],
        "existing_result_dir": "results/paired_acquisition_factorization_pair_integrity_scorpion_resnet50",
        "input_dim": 2048,
        "feature_dim": 2048,
        "n_images": 2400,
        "align_fn": scorpion_align_fold,
        "validate_fn": scorpion_validate_fold,
        "patch_fn": None,
        "has_category": False,
        "block_label": "slide_id",
    },
    "canineSCC_DINOv2": {
        "base_features": "results/external_multiscanner_caninescc/features/fold_0_dinov2_base.npz",
        "manifests_dir": "data/external_multiscanner_caninescc/patch_manifests/splits",
        "manifest_pattern": "fold_{fold}_patch_manifest.csv",
        "scanners": CANINE_SCANNERS,
        "folds": [0, 1, 2, 3, 4],
        "seeds": [911, 912, 913, 914, 915],
        "existing_result_dir": "results/paired_acquisition_factorization_pair_integrity_caninescc",
        "input_dim": 768,
        "feature_dim": 768,
        "n_images": 4025,
        "align_fn": canine_align_fold,
        "validate_fn": canine_validate_fold,
        "patch_fn": canine_patch_namespace,
        "has_category": True,
        "block_label": "sample_id",
    },
}

# New conditions to train (existing ones loaded from disk)
NEW_CONDITIONS = [
    "scanner_balanced_random_pairs",
    "fully_random_pairs",
]

# Additional conditions for datasets with category metadata
CATEGORY_CONDITIONS = [
    "same_category_different_sample_pairs",
]

# The complete ladder for SCORPION
SCORPION_LADDER = [
    "true_same_region_pairs",
    "same_slide_different_region_pairs",
    "shuffled_sample_pairs",
    "scanner_balanced_random_pairs",
    "fully_random_pairs",
]

# Display names
CONDITION_DISPLAY = {
    "true_pairs": "true_same_region_pairs",
    "shuffled_region_pairs": "same_slide_different_region_pairs",
    "shuffled_sample_pairs": "shuffled_sample_pairs",
    "scanner_balanced_random_pairs": "scanner_balanced_random_pairs",
    "fully_random_pairs": "fully_random_pairs",
    "same_category_different_sample_pairs": "same_category_different_sample_pairs",
}

# Simplifying assumption: ladder level 0 = strictest biology, higher = looser
LADDER_ORDER = {
    "true_same_region_pairs": 0,
    "same_slide_different_region_pairs": 1,
    "shuffled_sample_pairs": 2,
    "scanner_balanced_random_pairs": 3,
    "fully_random_pairs": 4,
    "same_category_different_sample_pairs": 2,  # between slide and balanced-random
}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Deterministic shuffle helper
# ---------------------------------------------------------------------------

def deterministic_seed(condition: str, fold: int, seed: int) -> int:
    payload = f"{condition}|fold={fold}|seed={seed}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**32)


# ---------------------------------------------------------------------------
# Pair-group construction  (reused + new conditions)
# ---------------------------------------------------------------------------

def _complete_region_table(frame: pd.DataFrame, fit_indices: np.ndarray,
                           scanners: Tuple[str, ...]) -> pd.DataFrame:
    subset = frame.iloc[fit_indices].copy()
    subset["_row_index"] = subset.index.to_numpy(dtype=np.int64)
    rows = []
    for region_id, group in subset.groupby("region_id", sort=True):
        if len(group) != len(scanners) or set(group["scanner_id"]) != set(scanners):
            raise ExperimentError("Every fitting region must contain all scanners.")
        slide_values = set(group["slide_id"].astype(str))
        if len(slide_values) != 1:
            raise ExperimentError(f"Region metadata not unique for {region_id}.")
        row = {
            "region_index": int(len(rows)),
            "region_id": str(region_id),
            "slide_id": next(iter(slide_values)),
        }
        for scanner in scanners:
            scanner_rows = group.loc[group["scanner_id"] == scanner, "_row_index"]
            row[scanner] = int(scanner_rows.iloc[0])
        rows.append(row)
    table = pd.DataFrame(rows)
    if table.empty:
        raise ExperimentError("No fitting regions available.")
    return table


def _derange_indices(candidates: list[int], *, rng: np.random.Generator,
                     forbidden: dict[int, set[int]],
                     max_attempts: int = 10000) -> list[int]:
    if len(candidates) <= 1:
        raise ExperimentError("Cannot derange fewer than two candidate regions.")
    base = np.asarray(candidates, dtype=np.int64)
    for _ in range(max_attempts):
        proposed = rng.permutation(base)
        if all(int(value) not in forbidden[int(anchor)]
               for anchor, value in zip(base, proposed)):
            return [int(value) for value in proposed]
    raise ExperimentError("Failed to construct deterministic derangement.")


def build_pair_groups(frame: pd.DataFrame, fit_indices: np.ndarray, *,
                      condition: str, fold: int, seed: int,
                      scanners: Tuple[str, ...],
                      category_map: Optional[dict] = None,
                      ) -> Tuple[list[np.ndarray], pd.DataFrame, dict[str, object]]:
    """Build pair groups for any supported condition."""
    table = _complete_region_table(frame, fit_indices, scanners)
    rng = np.random.default_rng(deterministic_seed(condition, fold, seed))

    assignments = pd.DataFrame({
        "fold": fold, "seed": seed, "condition": condition,
        "anchor_region_index": table["region_index"],
        "anchor_region_id": table["region_id"],
        "anchor_slide_id": table["slide_id"],
    })
    for scanner in scanners:
        assignments[f"{scanner}_region_index"] = table["region_index"]
        assignments[f"{scanner}_region_id"] = table["region_id"]
        assignments[f"{scanner}_slide_id"] = table["slide_id"]
        assignments[f"{scanner}_row_index"] = table[scanner]

    n_regions = len(table)

    # --- Existing conditions ---
    if condition in ("true_pairs", "true_same_region_pairs"):
        pass  # no derangement needed — all views from same region

    elif condition in ("shuffled_region_pairs", "same_slide_different_region_pairs"):
        for scanner in scanners[1:]:
            chosen = np.empty(n_regions, dtype=np.int64)
            for _, slide_regions in table.groupby("slide_id", sort=True):
                candidates = [int(v) for v in slide_regions["region_index"]]
                forbidden = {c: {c} for c in candidates}
                deranged = _derange_indices(candidates, rng=rng, forbidden=forbidden)
                for anchor, assigned in zip(candidates, deranged):
                    chosen[anchor] = assigned
            source = table.set_index("region_index").loc[chosen]
            _update_assignment(assignments, scanner, chosen, source, scanners)

    elif condition == "shuffled_sample_pairs":
        slides = sorted(table["slide_id"].astype(str).unique())
        forbidden_slides = {i: {i} for i in range(len(slides))}
        slide_order = _derange_indices(
            list(range(len(slides))), rng=rng, forbidden=forbidden_slides)
        slide_map = {slides[i]: slides[slide_order[i]] for i in range(len(slides))}
        for scanner in scanners[1:]:
            chosen = np.empty(n_regions, dtype=np.int64)
            for source_slide, target_slide in slide_map.items():
                anchors = table.loc[table["slide_id"] == source_slide,
                                    "region_index"].to_numpy(dtype=np.int64)
                targets = table.loc[table["slide_id"] == target_slide,
                                    "region_index"].to_numpy(dtype=np.int64)
                if len(anchors) != len(targets):
                    raise ExperimentError(
                        "Sample-shuffled slide groups must have equal region counts.")
                shuffled = rng.permutation(targets)
                for anchor, assigned in zip(anchors, shuffled):
                    chosen[int(anchor)] = int(assigned)
            source = table.set_index("region_index").loc[chosen]
            _update_assignment(assignments, scanner, chosen, source, scanners)

    # --- New conditions ---
    elif condition == "scanner_balanced_random_pairs":
        # For each anchor region, each non-anchor scanner gets a random region
        # from ANYWHERE in the dataset (excluding the anchor region itself).
        all_indices = np.arange(n_regions, dtype=np.int64)
        for scanner in scanners[1:]:
            chosen = np.empty(n_regions, dtype=np.int64)
            for anchor_idx in range(n_regions):
                # Exclude anchor region
                candidates = all_indices[all_indices != anchor_idx]
                chosen[anchor_idx] = int(rng.choice(candidates))
            source = table.set_index("region_index").loc[chosen]
            _update_assignment(assignments, scanner, chosen, source, scanners)

    elif condition == "fully_random_pairs":
        # Each of the 5 positions gets a random region from anywhere.
        # No anchor structure. All 5 regions are selected randomly.
        all_indices = np.arange(n_regions, dtype=np.int64)
        for scanner in scanners:  # ALL scanners including "anchor"
            chosen = np.empty(n_regions, dtype=np.int64)
            for idx in range(n_regions):
                chosen[idx] = int(rng.choice(all_indices))
            source = table.set_index("region_index").loc[chosen]
            _update_assignment(assignments, scanner, chosen, source, scanners)

    elif condition == "same_category_different_sample_pairs":
        if category_map is None:
            raise ExperimentError(
                "category_map required for same_category_different_sample_pairs")
        # Each anchor region gets non-anchor views from different slides
        # but same tissue category.
        # Group regions by category
        cat_to_regions: Dict[str, list[int]] = {}
        for _, row in table.iterrows():
            rid = str(row["region_id"])
            cat = category_map.get(rid, "unknown")
            cat_to_regions.setdefault(cat, []).append(int(row["region_index"]))

        slide_of = dict(zip(table["region_index"], table["slide_id"].astype(str)))
        for scanner in scanners[1:]:
            chosen = np.empty(n_regions, dtype=np.int64)
            for anchor_idx in range(n_regions):
                anchor_rid = str(table.iloc[anchor_idx]["region_id"])
                anchor_cat = category_map.get(anchor_rid, "unknown")
                anchor_slide = slide_of[anchor_idx]
                # Candidates: same category, different slide
                candidates = [ri for ri in cat_to_regions.get(anchor_cat, [anchor_idx])
                              if ri != anchor_idx and slide_of.get(ri) != anchor_slide]
                if not candidates:
                    # Fall back to same category, different region
                    candidates = [ri for ri in cat_to_regions.get(anchor_cat, [anchor_idx])
                                  if ri != anchor_idx]
                if not candidates:
                    candidates = [anchor_idx]
                chosen[anchor_idx] = int(rng.choice(np.asarray(candidates, dtype=np.int64)))
            source = table.set_index("region_index").loc[chosen]
            _update_assignment(assignments, scanner, chosen, source, scanners)

    else:
        raise ExperimentError(f"Unknown condition: {condition}")

    # Build groups
    groups = []
    for row in assignments.itertuples(index=False):
        groups.append(np.asarray(
            [getattr(row, f"{scanner}_row_index") for scanner in scanners],
            dtype=np.int64))

    # Audit
    mismatch_cols = [f"{scanner}_region_id" for scanner in scanners[1:]]
    mismatch_matrix = np.column_stack([
        assignments[col].astype(str).to_numpy() !=
        assignments["anchor_region_id"].astype(str).to_numpy()
        for col in mismatch_cols
    ])
    same_slide_matrix = np.column_stack([
        assignments[f"{scanner}_slide_id"].astype(str).to_numpy() ==
        assignments["anchor_slide_id"].astype(str).to_numpy()
        for scanner in scanners[1:]
    ])
    audit = {
        "condition": condition, "fold": fold, "seed": seed,
        "n_fit_regions": int(len(table)),
        "n_pair_groups": int(len(groups)),
        "anchor_scanner": scanners[0],
        "deterministic_shuffle_seed": int(deterministic_seed(condition, fold, seed)),
        "non_anchor_region_mismatch_fraction": float(mismatch_matrix.mean()),
        "non_anchor_same_slide_fraction": float(same_slide_matrix.mean()),
        "unique_training_rows_used": int(len(np.unique(np.concatenate(groups)))),
        "expected_training_rows_used": int(len(table) * len(scanners)),
    }
    return groups, assignments, audit


def _update_assignment(assignments: pd.DataFrame, scanner: str,
                       chosen: np.ndarray, source: pd.DataFrame,
                       scanners: Tuple[str, ...]) -> None:
    assignments[f"{scanner}_region_index"] = chosen
    assignments[f"{scanner}_region_id"] = source["region_id"].to_numpy()
    assignments[f"{scanner}_slide_id"] = source["slide_id"].to_numpy()
    assignments[f"{scanner}_row_index"] = source[scanner].to_numpy(dtype=np.int64)


# ---------------------------------------------------------------------------
# Atomic helpers
# ---------------------------------------------------------------------------

def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".csv", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _mark_projection_metadata(path: Path, metadata_update: dict) -> None:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    metadata.update(metadata_update)
    text = json.dumps(metadata, sort_keys=True)
    arrays["metadata_json"] = np.asarray(text, dtype=f"<U{len(text)}")
    atomic_npz(path, arrays)


# ---------------------------------------------------------------------------
# Evaluation (same as falsification script)
# ---------------------------------------------------------------------------

def _load_projected(path: Path):
    with np.load(path, allow_pickle=False) as archive:
        required = {"features", "acquisition_features", "slide_id",
                     "region_id", "scanner_id", "split"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise ExperimentError(f"{path} missing arrays: {missing}")
        biological = np.asarray(archive["features"], dtype=np.float32)
        acquisition = np.asarray(archive["acquisition_features"], dtype=np.float32)
        frame = pd.DataFrame({
            name: archive[name].astype(str)
            for name in ("slide_id", "region_id", "scanner_id", "split")
        })
        metadata = json.loads(str(archive["metadata_json"].item()))
    return biological, acquisition, frame, metadata


def _split_indices(frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    test = np.flatnonzero(frame["split"].to_numpy() == "test")
    fit = np.flatnonzero(frame["split"].to_numpy() != "test")
    if len(test) == 0 or len(fit) == 0:
        raise ExperimentError("Empty fit or test split.")
    if set(frame.iloc[test]["slide_id"]) & set(frame.iloc[fit]["slide_id"]):
        raise ExperimentError("Slide leakage between fit and test.")
    return fit, test


def _scanner_probe(features: np.ndarray, frame: pd.DataFrame,
                   fit: np.ndarray, test: np.ndarray) -> float:
    labels = frame["scanner_id"].to_numpy()
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000,
                           random_state=0))
    model.fit(features[fit], labels[fit])
    predictions = model.predict(features[test])
    return float(balanced_accuracy_score(labels[test], predictions))


def _normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms <= 0] = 1.0
    return features / norms


def _paired_metrics(features: np.ndarray, frame: pd.DataFrame,
                    test: np.ndarray, scanners: Tuple[str, ...]) -> Dict[str, float]:
    normalized = _normalize(features[test])
    test_frame = frame.iloc[test].reset_index(drop=True)
    maps = {scanner: {} for scanner in scanners}
    for index, row in test_frame.iterrows():
        maps[str(row["scanner_id"])][str(row["region_id"])] = normalized[index]
    pair_rows = []
    for index_a, scanner_a in enumerate(scanners):
        for scanner_b in scanners[index_a + 1:]:
            regions = sorted(set(maps[scanner_a]) & set(maps[scanner_b]))
            if not regions:
                continue
            a = np.stack([maps[scanner_a][region] for region in regions])
            b = np.stack([maps[scanner_b][region] for region in regions])
            similarity = a @ b.T
            diagonal = np.diag(similarity)
            truth = np.arange(len(regions))
            p_ab = np.argmax(similarity, axis=1)
            p_ba = np.argmax(similarity.T, axis=1)
            pair_rows.append({
                "cosine": float(diagonal.mean()),
                "retrieval": float(0.5 * (np.mean(p_ab == truth) +
                                         np.mean(p_ba == truth))),
            })
    if not pair_rows:
        return {"mean_paired_cosine": float("nan"),
                "worst_paired_cosine": float("nan"),
                "mean_top1_retrieval": float("nan"),
                "worst_top1_retrieval": float("nan")}
    pf = pd.DataFrame(pair_rows)
    return {
        "mean_paired_cosine": float(pf["cosine"].mean()),
        "worst_paired_cosine": float(pf["cosine"].min()),
        "mean_top1_retrieval": float(pf["retrieval"].mean()),
        "worst_top1_retrieval": float(pf["retrieval"].min()),
    }


def _effective_rank(features: np.ndarray) -> float:
    centered = features - features.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    energy = sv ** 2
    total = float(energy.sum())
    if total <= 0:
        return 0.0
    probs = energy / total
    probs = probs[probs > 0]
    return float(math.exp(-np.sum(probs * np.log(probs))))


def _cross_covariance_rms(bio: np.ndarray, acq: np.ndarray,
                          test: np.ndarray) -> float:
    b = StandardScaler().fit_transform(bio[test])
    a = StandardScaler().fit_transform(acq[test])
    cross = b.T @ a / max(1, len(test) - 1)
    return float(np.sqrt(np.mean(cross ** 2)))


# ---------------------------------------------------------------------------
# Evaluation runner
# ---------------------------------------------------------------------------

def evaluate_one_run(projected_path: Path, scanners: Tuple[str, ...],
                     condition: str) -> Dict[str, object]:
    """Compute all metrics on one projected_features.npz."""
    biological, acquisition, frame, metadata = _load_projected(projected_path)
    fit, test = _split_indices(frame)

    bio_probe = _scanner_probe(biological, frame, fit, test)
    bio_paired = _paired_metrics(biological, frame, test, scanners)
    acq_probe = _scanner_probe(acquisition, frame, fit, test)
    acq_paired = _paired_metrics(acquisition, frame, test, scanners)
    bio_erank = _effective_rank(biological[test])
    acq_erank = _effective_rank(acquisition[test])
    cross_cov = _cross_covariance_rms(biological, acquisition, test)

    return {
        "bio_scanner_probe_accuracy": bio_probe,
        "bio_mean_paired_cosine": bio_paired["mean_paired_cosine"],
        "bio_worst_paired_cosine": bio_paired["worst_paired_cosine"],
        "bio_mean_top1_retrieval": bio_paired["mean_top1_retrieval"],
        "bio_worst_top1_retrieval": bio_paired["worst_top1_retrieval"],
        "bio_effective_rank": bio_erank,
        "acq_scanner_probe_accuracy": acq_probe,
        "acq_mean_paired_cosine": acq_paired["mean_paired_cosine"],
        "acq_worst_paired_cosine": acq_paired["worst_paired_cosine"],
        "acq_mean_top1_retrieval": acq_paired["mean_top1_retrieval"],
        "acq_worst_top1_retrieval": acq_paired["worst_top1_retrieval"],
        "acq_effective_rank": acq_erank,
        "biological_acquisition_cross_covariance": cross_cov,
    }


# ---------------------------------------------------------------------------
# Training runner for new conditions
# ---------------------------------------------------------------------------

def train_new_conditions(ds_name: str, ds_cfg: dict, conditions: List[str],
                         out_dir: Path, device: torch.device,
                         smoke: bool = False) -> List[Dict]:
    """Train new conditions not already in the existing result dir."""
    scanners = ds_cfg["scanners"]
    base_features_path = Path(ds_cfg["base_features"])
    manifests_dir = Path(ds_cfg["manifests_dir"])
    manifest_pattern = ds_cfg["manifest_pattern"]
    folds = ds_cfg["folds"][:1] if smoke else ds_cfg["folds"]
    seeds = ds_cfg["seeds"][:1] if smoke else ds_cfg["seeds"]
    align_fn = ds_cfg["align_fn"]
    validate_fn = ds_cfg["validate_fn"]
    block_label = ds_cfg.get("block_label", "slide_id")

    # Apply dataset-specific namespace patches (e.g. canine scanner mapping)
    patch_fn = ds_cfg.get("patch_fn")
    if patch_fn is not None:
        patch_fn()

    base_features, base_frame, source_metadata = load_archive(base_features_path)
    input_dim = base_features.shape[1]

    # Build category map for datasets with category metadata
    # Note: category_name is in the manifest CSV but align_fold strips it.
    # We read the manifest directly.
    category_map = None
    if ds_cfg.get("has_category"):
        try:
            manifest0 = manifests_dir / manifest_pattern.format(fold=folds[0])
            manifest_df = pd.read_csv(manifest0, dtype=str)
            if "category_name" in manifest_df.columns:
                category_map = {}
                for _, row in manifest_df.iterrows():
                    category_map[str(row["region_id"])] = str(row["category_name"])
                print(f"  [{ds_name}] Built category map with "
                      f"{len(set(category_map.values()))} categories, "
                      f"{len(category_map)} regions")
            else:
                print(f"  [{ds_name}] category_name not in manifest columns: "
                      f"{sorted(manifest_df.columns)}")
        except Exception:
            print(f"  [{ds_name}] Failed to build category map: {traceback.format_exc()}")

    train_dir = out_dir / ds_name
    train_dir.mkdir(parents=True, exist_ok=True)
    training_path = train_dir / "training_results.csv"

    rows = []
    completed = set()
    if training_path.is_file():
        existing = pd.read_csv(training_path)
        rows = existing.to_dict("records")
        completed = {(int(r["fold"]), int(r["seed"]), str(r["condition"]))
                     for r in rows}

    for fold in folds:
        manifest_path = manifests_dir / manifest_pattern.format(fold=fold)
        features, frame = align_fn(base_features, base_frame, manifest_path)
        fit_indices, test_indices = validate_fn(frame, fold)
        transformed, mean, std = standardize(features, fit_indices)

        fold_dir = train_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(fold_dir / "fit_standardization.npz",
                            mean=mean, std=std)

        for seed in seeds:
            for condition in conditions:
                key = (fold, seed, condition)
                if key in completed:
                    continue

                groups, assignments, audit = build_pair_groups(
                    frame, fit_indices,
                    condition=condition, fold=fold, seed=seed,
                    scanners=scanners, category_map=category_map)

                run_dir = fold_dir / "runs" / f"{condition}_seed_{seed}"
                print(f"  [{ds_name}] Training fold={fold} seed={seed} "
                      f"condition={condition} "
                      f"mismatch={audit['non_anchor_region_mismatch_fraction']:.3f} "
                      f"same_slide={audit['non_anchor_same_slide_fraction']:.3f}")

                result = train_one(
                    method="pathoalign", seed=seed,
                    features=transformed, frame=frame,
                    train_indices=fit_indices,
                    development_indices=np.arange(len(frame), dtype=np.int64),
                    groups=groups, config=config_for(input_dim),
                    device=device, epochs=75, region_batch_size=32,
                    learning_rate=0.0003, weight_decay=0.0001,
                    run_dir=run_dir)

                _mark_projection_metadata(
                    run_dir / "projected_features.npz",
                    {"contains_test_rows": True,
                     "evaluation_stage": "pair_structure_boundary_test",
                     "fold": int(fold), "seed": int(seed),
                     "condition": condition,
                     "fit_splits": ["train", "val"],
                     "evaluation_split": "test",
                     "pair_construction_audit": audit,
                     "hyperparameters_frozen": True})

                rows.append({
                    "fold": fold, "seed": seed, "condition": condition,
                    **result, **audit,
                    "n_fit_slides": int(frame.iloc[fit_indices]["slide_id"].nunique()),
                    "n_test_slides": int(frame.iloc[test_indices]["slide_id"].nunique()),
                })
                write_results(training_path, rows)
                completed.add(key)

    return rows


# ---------------------------------------------------------------------------
# Load existing results from pair-integrity falsification
# ---------------------------------------------------------------------------

def load_existing_results(result_dir: Path, scanners: Tuple[str, ...],
                          ds_name: str) -> List[Dict]:
    """Load and remap existing condition names."""
    training_path = result_dir / "training_results.csv"
    if not training_path.is_file():
        return []
    training = pd.read_csv(training_path)
    rows = []
    for _, row in training.iterrows():
        fold = int(row["fold"])
        seed = int(row["seed"])
        condition = str(row["condition"])
        projected_path = (result_dir / f"fold_{fold}" / "runs" /
                          f"{condition}_seed_{seed}" / "projected_features.npz")
        if not projected_path.is_file():
            continue
        metrics = evaluate_one_run(projected_path, scanners, condition)
        display_name = CONDITION_DISPLAY.get(condition, condition)
        rows.append({
            "dataset_backbone": ds_name,
            "fold": fold, "seed": seed,
            "condition": display_name,
            "original_condition": condition,
            "source": "existing",
            "ladder_level": LADDER_ORDER.get(display_name, 99),
            **metrics,
        })
    return rows


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report(all_rows: List[Dict], design: dict,
                 out_dir: Path, runtime: float, smoke: bool) -> str:
    """Build the boundary test report with per-dataset interpretation."""
    df = pd.DataFrame(all_rows)
    lines = []

    # --- Determine smoke/full status per dataset ---
    # Full means: new conditions (not just reused existing) have 5+ folds and 5+ seeds
    ds_status = {}
    for ds_name in sorted(df["dataset_backbone"].unique()):
        sub = df[df["dataset_backbone"] == ds_name]
        new_sub = sub[sub["source"] == "new"] if "source" in sub.columns else sub
        if len(new_sub) == 0:
            new_sub = sub  # all existing — use overall
        n_folds_new = new_sub["fold"].nunique()
        n_seeds_new = new_sub["seed"].nunique()
        n_folds_all = sub["fold"].nunique()
        n_seeds_all = sub["seed"].nunique()
        is_full = (n_folds_new >= 5 and n_seeds_new >= 5)
        ds_status[ds_name] = {
            "n_folds_all": n_folds_all,
            "n_seeds_all": n_seeds_all,
            "n_folds_new": n_folds_new,
            "n_seeds_new": n_seeds_new,
            "is_full": is_full,
            "label": "full (5-fold x 5-seed)" if is_full else
                     f"smoke ({n_folds_new}-fold x {n_seeds_new}-seed new, "
                     f"{n_folds_all}-fold x {n_seeds_all}-seed overall)",
        }

    lines.append("# Pair-Structure Boundary Test Report")
    lines.append("")
    lines.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Runtime:** {runtime:.1f} s")
    lines.append("")

    lines.append("## Scientific question")
    lines.append("")
    lines.append("How exact does the paired-acquisition structure need to be "
                 "for the method to preserve tissue identity while reducing "
                 "scanner/acquisition signal?")
    lines.append("")

    lines.append("## Evidence tiers")
    lines.append("")
    for ds_name in sorted(ds_status):
        st = ds_status[ds_name]
        lines.append(f"- **{ds_name}**: {st['label']}")
    lines.append("")
    if any(not st["is_full"] for st in ds_status.values()):
        lines.append("**Caution:** smoke-only results use 1 fold and 1 seed. "
                     "Variance estimates are unreliable. Full 5-fold x 5-seed "
                     "validation is needed before strong cross-dataset claims.")
        lines.append("")

    lines.append("## Pairing ladder")
    lines.append("")
    lines.append("| Level | Condition | Description |")
    lines.append("|---|---|")
    for cond, level in sorted(LADDER_ORDER.items(), key=lambda x: x[1]):
        desc = {
            "true_same_region_pairs":
                "Same tissue region, different scanners",
            "same_slide_different_region_pairs":
                "Same slide, different tissue region",
            "shuffled_sample_pairs":
                "Different slides (existing falsification condition)",
            "scanner_balanced_random_pairs":
                "Random regions preserving scanner assignment",
            "fully_random_pairs":
                "All views randomly assigned, no structure (lower bound)",
            "same_category_different_sample_pairs":
                "Same tissue category, different sample (canine SCC only)",
        }.get(cond, "")
        status = "tested" if cond in df["condition"].unique() else "not tested"
        lines.append(f"| {level} | `{cond}` | {desc} ({status}) |")
    lines.append("")

    lines.append("## Datasets and conditions")
    lines.append("")
    for ds_name in sorted(df["dataset_backbone"].unique()):
        sub = df[df["dataset_backbone"] == ds_name]
        st = ds_status[ds_name]
        lines.append(f"### {ds_name} ({st['label']})")
        lines.append("")
        lines.append(f"- {len(sub)} runs across {sub['condition'].nunique()} "
                     f"conditions, {sub['fold'].nunique()} folds, "
                     f"{sub['seed'].nunique()} seeds")
        cond_list = []
        for cond in sorted(sub["condition"].unique()):
            csub = sub[sub["condition"] == cond]
            src = csub["source"].iloc[0]
            level = csub["ladder_level"].iloc[0]
            cond_list.append(f"  - L{int(level)} `{cond}` ({src}, {len(csub)} runs)")
        lines.extend(sorted(cond_list))
        lines.append("")

    # --- Biological branch table ---
    lines.append("## Biological branch: tissue identity preservation")
    lines.append("")
    lines.append("Scanner probe should be low (scanner suppressed); "
                 "paired cosine and top-1 retrieval should be high "
                 "(tissue identity preserved).")
    lines.append("")
    lines.append("| Dataset | Level | Condition | Scanner probe | Paired cosine | "
                 "Top-1 retrieval | Effective rank |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    bio_metric_cols = ["bio_scanner_probe_accuracy", "bio_mean_paired_cosine",
                       "bio_mean_top1_retrieval", "bio_effective_rank"]
    for ds_name in sorted(df["dataset_backbone"].unique()):
        sub = df[df["dataset_backbone"] == ds_name]
        # Sort by ladder level
        cond_order = sorted(sub["condition"].unique(),
                           key=lambda c: sub[sub["condition"] == c]["ladder_level"].iloc[0])
        for cond in cond_order:
            grp = sub[sub["condition"] == cond]
            means = {col: grp[col].mean() for col in bio_metric_cols if col in grp.columns}
            level = int(grp["ladder_level"].iloc[0])
            lines.append(
                f"| {ds_name} | {level} | `{cond}` | "
                f"{means.get('bio_scanner_probe_accuracy', float('nan')):.4f} | "
                f"{means.get('bio_mean_paired_cosine', float('nan')):.4f} | "
                f"{means.get('bio_mean_top1_retrieval', float('nan')):.4f} | "
                f"{means.get('bio_effective_rank', float('nan')):.1f} |")
    lines.append("")

    # --- Acquisition branch table ---
    lines.append("## Acquisition branch: scanner capture")
    lines.append("")
    lines.append("Scanner probe should be high (scanner captured); "
                 "paired cosine and top-1 retrieval should be low "
                 "(tissue identity removed from acquisition branch). "
                 "Cross-covariance should be low (branches decoupled).")
    lines.append("")
    lines.append("| Dataset | Level | Condition | Scanner probe | Paired cosine | "
                 "Top-1 retrieval | Cross-cov RMS |")
    lines.append("|---|---:|---:|---:|---:|---:|")

    acq_metric_cols = ["acq_scanner_probe_accuracy", "acq_mean_paired_cosine",
                       "acq_mean_top1_retrieval",
                       "biological_acquisition_cross_covariance"]
    for ds_name in sorted(df["dataset_backbone"].unique()):
        sub = df[df["dataset_backbone"] == ds_name]
        cond_order = sorted(sub["condition"].unique(),
                           key=lambda c: sub[sub["condition"] == c]["ladder_level"].iloc[0])
        for cond in cond_order:
            grp = sub[sub["condition"] == cond]
            means = {col: grp[col].mean() for col in acq_metric_cols if col in grp.columns}
            level = int(grp["ladder_level"].iloc[0])
            lines.append(
                f"| {ds_name} | {level} | `{cond}` | "
                f"{means.get('acq_scanner_probe_accuracy', float('nan')):.4f} | "
                f"{means.get('acq_mean_paired_cosine', float('nan')):.4f} | "
                f"{means.get('acq_mean_top1_retrieval', float('nan')):.4f} | "
                f"{means.get('biological_acquisition_cross_covariance', float('nan')):.6f} |")
    lines.append("")

    # --- Per-dataset interpretation ---
    lines.append("## Interpretation")
    lines.append("")

    for ds_name in sorted(df["dataset_backbone"].unique()):
        sub = df[df["dataset_backbone"] == ds_name]
        st = ds_status[ds_name]
        lines.append(f"### {ds_name} ({st['label']})")
        lines.append("")

        # Find level-0 (true pairs) and compare with higher levels
        l0 = sub[sub["ladder_level"] == 0]
        higher = sub[sub["ladder_level"] > 0]

        if l0.empty:
            lines.append("No true-pair baseline available for this dataset.")
            lines.append("")
            continue

        l0_cos = l0["bio_mean_paired_cosine"].mean()
        l0_ret = l0["bio_mean_top1_retrieval"].mean()
        l0_probe = l0["bio_scanner_probe_accuracy"].mean()
        l0_acq_cos = l0["acq_mean_paired_cosine"].mean()

        lines.append(f"**True same-region pairs (L0):** "
                     f"paired cosine = {l0_cos:.4f}, "
                     f"top-1 retrieval = {l0_ret:.4f}, "
                     f"scanner probe = {l0_probe:.4f}")
        lines.append("")

        # Compare each higher level to L0
        lines.append("| Level | Condition | Delta cosine vs L0 | "
                     "Delta retrieval vs L0 | Tissue damage? |")
        lines.append("|---|---:|---:|:---:|")
        cond_order = sorted(higher["condition"].unique(),
                           key=lambda c: higher[higher["condition"] == c]["ladder_level"].iloc[0])
        for cond in cond_order:
            grp = higher[higher["condition"] == cond]
            level = int(grp["ladder_level"].iloc[0])
            d_cos = grp["bio_mean_paired_cosine"].mean() - l0_cos
            d_ret = grp["bio_mean_top1_retrieval"].mean() - l0_ret
            # Tissue damage: cosine drops more than 0.01 AND retrieval drops
            damaged = abs(d_cos) > 0.01 and d_ret < -0.001
            lines.append(
                f"| {level} | `{cond}` | {d_cos:+.4f} | {d_ret:+.4f} | "
                f"{'Yes' if damaged else 'Marginal'} |")
        lines.append("")

        # Summary verdict
        if not higher.empty:
            best_higher_cos = higher.groupby("condition")["bio_mean_paired_cosine"].mean().max()
            best_higher_ret = higher.groupby("condition")["bio_mean_top1_retrieval"].mean().max()
            cos_gap = l0_cos - best_higher_cos
            ret_gap = l0_ret - best_higher_ret

            if cos_gap > 0.05 and ret_gap > 0.002:
                lines.append(f"**Verdict:** True same-region pairs clearly "
                             f"outperform all looser conditions. Paired cosine "
                             f"gap = {cos_gap:.4f}, retrieval gap = {ret_gap:.4f}. "
                             f"This supports the paired-acquisition mechanism: "
                             f"exact biological correspondence in pairing matters "
                             f"for tissue identity preservation.")
            elif cos_gap > 0.02:
                lines.append(f"**Verdict:** True pairs are best with a "
                             f"meaningful margin (cosine gap = {cos_gap:.4f}, "
                             f"retrieval gap = {ret_gap:.4f}). Same-slide "
                             f"pairing preserves substantial tissue identity "
                             f"but exact-region pairing is measurably better.")
            else:
                lines.append(f"**Verdict:** Tissue identity preservation does "
                             f"not strongly depend on pairing strictness "
                             f"(cosine gap = {cos_gap:.4f}). The paired-"
                             f"acquisition mechanism may be weaker than claimed "
                             f"for this dataset, or scanner suppression alone "
                             f"may account for the effect.")
            lines.append("")

        # Scanner suppression check
        if not higher.empty:
            higher_probe = higher["bio_scanner_probe_accuracy"].mean()
            probe_delta = higher_probe - l0_probe
            lines.append(f"**Scanner suppression:** Biological branch scanner "
                         f"probe is {l0_probe:.4f} (L0) vs "
                         f"{higher_probe:.4f} (higher levels, "
                         f"delta = {probe_delta:+.4f}).")
            if abs(probe_delta) < 0.05:
                lines.append("Scanner suppression is maintained across all "
                             "pairing conditions — the scanner adversary "
                             "works regardless of pair quality.")
            else:
                lines.append("Scanner probe varies with pairing strictness.")
            lines.append("")

        # Acquisition disentanglement check
        if not higher.empty:
            higher_acq_cos = higher["acq_mean_paired_cosine"].mean()
            acq_delta = higher_acq_cos - l0_acq_cos
            lines.append(f"**Acquisition disentanglement:** Acquisition branch "
                         f"paired cosine is {l0_acq_cos:.4f} (L0) vs "
                         f"{higher_acq_cos:.4f} (higher levels, "
                         f"delta = {acq_delta:+.4f}).")
            if acq_delta > 0.05:
                lines.append("Looser pairing causes the acquisition branch to "
                             "encode more tissue-level information, reducing "
                             "disentanglement quality.")
            else:
                lines.append("Disentanglement quality is similar across "
                             "pairing conditions.")
            lines.append("")

    lines.append("## Claim boundaries")
    lines.append("")
    lines.append("- Existing conditions (true_pairs, shuffled_region_pairs, "
                 "shuffled_sample_pairs) reuse trained models from the "
                 "pair-integrity falsification experiment. No retraining.")
    if any((r.get("source") == "new") for r in all_rows if isinstance(r, dict)):
        lines.append("- New conditions (scanner_balanced_random_pairs, "
                     "fully_random_pairs, same_category_different_sample_pairs) "
                     "were trained on the same base features with modified "
                     "pair constructions.")
    lines.append("- All metrics computed on held-out test slides only.")
    lines.append("- The fully_random_pairs condition is intentionally "
                 "degraded; it serves as a lower bound, not a recommended "
                 "training configuration.")
    for ds_name in sorted(ds_status):
        if not ds_status[ds_name]["is_full"]:
            lines.append(f"- **{ds_name} is smoke-only.** Full 5-fold x "
                         f"5-seed validation is pending. Per-condition "
                         f"variance estimates from smoke data are unreliable. "
                         f"Cross-dataset comparisons involving this dataset "
                         f"are provisional.")
    lines.append("")
    lines.append("This experiment does not claim clinical validation, "
                 "diagnostic performance, disease biology discovery, or "
                 "deployment readiness. It tests only whether the "
                 "paired-acquisition factorization effect depends on "
                 "pair-structure strictness.")
    lines.append("")

    lines.append("## Output files")
    lines.append("")
    lines.append("| File | Description |")
    lines.append("|---|---|")
    lines.append("| boundary_raw_metrics.csv | Per-run, per-condition metrics |")
    lines.append("| boundary_summary.csv | Aggregated by dataset and condition |")
    lines.append("| boundary_condition_contrasts.csv | Level-vs-level contrasts |")
    lines.append("| experiment_design.json | Experiment configuration |")
    lines.append("| run_log.txt | Timestamped run log |")
    lines.append("| pair_structure_boundary_report.md | This report |")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Pair-structure boundary test for paired-acquisition")
    p.add_argument("--out-dir", type=Path,
                   default=Path("results/paired_acquisition_factorization_pair_structure_boundary_test"))
    p.add_argument("--smoke", action="store_true",
                   help="Smoke test: 1 fold, 1 seed, 1 dataset only")
    p.add_argument("--datasets", nargs="*",
                   help="Dataset names (default: SCORPION_DINOv2, canineSCC_DINOv2)")
    p.add_argument("--device", default="cuda")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    log_path = out_dir / "run_log.txt"
    log_file = open(str(log_path), "w", encoding="utf-8")

    def log(msg: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {msg}"
        print(line)
        log_file.write(line + "\n")
        log_file.flush()

    log(f"Pair-structure boundary test started (smoke={args.smoke})")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        log("CUDA unavailable, falling back to CPU")
        device = torch.device("cpu")

    ds_names = args.datasets or ["SCORPION_DINOv2"]
    log(f"Datasets: {ds_names}")

    all_rows = []

    for ds_name in ds_names:
        if ds_name not in DATASETS:
            log(f"Unknown dataset: {ds_name}, skipping")
            continue
        ds_cfg = DATASETS[ds_name]
        scanners = ds_cfg["scanners"]

        # 1. Load existing results
        existing_dir = Path(ds_cfg["existing_result_dir"])
        log(f"[{ds_name}] Loading existing results from {existing_dir}")
        existing_rows = load_existing_results(existing_dir, scanners, ds_name)
        log(f"[{ds_name}]   Loaded {len(existing_rows)} existing runs")
        all_rows.extend(existing_rows)

        # 2. Determine conditions to train for this dataset
        ds_conditions = list(NEW_CONDITIONS)
        if ds_cfg.get("has_category"):
            ds_conditions.extend(CATEGORY_CONDITIONS)
            log(f"[{ds_name}] Category metadata available — adding {CATEGORY_CONDITIONS}")

        if args.smoke:
            log(f"[{ds_name}] Smoke mode — training 1 fold, 1 seed for: {ds_conditions}")
        else:
            log(f"[{ds_name}] Training new conditions: {ds_conditions}")

        try:
            train_rows = train_new_conditions(
                ds_name, ds_cfg, ds_conditions, out_dir, device,
                smoke=args.smoke)
            log(f"[{ds_name}]   Trained {len(train_rows)} new runs")
        except Exception:
            log(f"[{ds_name}]   Training FAILED: {traceback.format_exc()}")
            train_rows = []

        # 3. Evaluate new trained runs
        new_eval_rows = []
        train_dir = out_dir / ds_name
        for fold_dir in sorted(train_dir.glob("fold_*")):
            fold = int(fold_dir.name.split("_")[-1])
            for run_dir in sorted((fold_dir / "runs").glob("*_seed_*")):
                parts = run_dir.name.rsplit("_seed_", 1)
                condition = parts[0]
                seed = int(parts[1])
                projected_path = run_dir / "projected_features.npz"
                if not projected_path.is_file():
                    continue
                try:
                    metrics = evaluate_one_run(projected_path, scanners, condition)
                    display_name = CONDITION_DISPLAY.get(condition, condition)
                    new_eval_rows.append({
                        "dataset_backbone": ds_name,
                        "fold": fold, "seed": seed,
                        "condition": display_name,
                        "original_condition": condition,
                        "source": "new",
                        "ladder_level": LADDER_ORDER.get(display_name, 99),
                        **metrics,
                    })
                except Exception:
                    log(f"  Eval failed for {projected_path}: {traceback.format_exc()}")
        log(f"[{ds_name}]   Evaluated {len(new_eval_rows)} new runs")
        all_rows.extend(new_eval_rows)

    if not all_rows:
        log("No data loaded. Aborting.")
        log_file.close()
        return 1

    df = pd.DataFrame(all_rows)
    log(f"Total rows: {len(df)}")

    # Write design
    new_conditions_trained = sorted(
        df[df["source"] == "new"]["condition"].unique().tolist()
    ) if "source" in df.columns and not df.empty else []
    design = {
        "stage": "pair_structure_boundary_test",
        "smoke_test": args.smoke,
        "datasets": ds_names,
        "conditions_tested": sorted(df["condition"].unique().tolist()),
        "existing_conditions_reused": True,
        "new_conditions_trained": new_conditions_trained,
        "ladder_definition": LADDER_ORDER,
    }
    (out_dir / "experiment_design.json").write_text(
        json.dumps(design, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Write raw metrics
    _atomic_csv(out_dir / "boundary_raw_metrics.csv", df)

    # Summary
    metric_cols = [c for c in df.columns
                   if c.startswith("bio_") or c.startswith("acq_") or
                   c == "biological_acquisition_cross_covariance"]
    summary_rows = []
    for (ds, cond), grp in df.groupby(["dataset_backbone", "condition"]):
        agg = {"dataset_backbone": ds, "condition": cond,
               "n_runs": len(grp),
               "ladder_level": grp["ladder_level"].iloc[0]}
        for col in metric_cols:
            if col in grp.columns and grp[col].notna().any():
                agg[f"{col}_mean"] = float(grp[col].mean())
                agg[f"{col}_std"] = float(grp[col].std())
        summary_rows.append(agg)
    summary = pd.DataFrame(summary_rows).sort_values(
        ["dataset_backbone", "ladder_level"])
    _atomic_csv(out_dir / "boundary_summary.csv", summary)

    # Contrasts: level vs level within each dataset
    contrast_rows = []
    for ds in df["dataset_backbone"].unique():
        sub = df[df["dataset_backbone"] == ds]
        levels = sorted(sub["ladder_level"].unique())
        for i in range(len(levels) - 1):
            lo = sub[sub["ladder_level"] == levels[i]]
            hi = sub[sub["ladder_level"] == levels[i + 1]]
            lo_cond = lo["condition"].iloc[0]
            hi_cond = hi["condition"].iloc[0]
            for col in metric_cols:
                if col in lo.columns and col in hi.columns:
                    delta = hi[col].mean() - lo[col].mean()
                    contrast_rows.append({
                        "dataset_backbone": ds,
                        "from_condition": lo_cond,
                        "to_condition": hi_cond,
                        "from_level": levels[i],
                        "to_level": levels[i + 1],
                        "metric": col,
                        "mean_delta": float(delta),
                    })
    contrasts = pd.DataFrame(contrast_rows)
    _atomic_csv(out_dir / "boundary_condition_contrasts.csv", contrasts)

    # Report
    runtime = time.time() - t0
    report = build_report(all_rows, design, out_dir, runtime, args.smoke)
    (out_dir / "pair_structure_boundary_report.md").write_text(
        report, encoding="utf-8")

    log(f"Done in {runtime:.1f} s")
    log_file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
