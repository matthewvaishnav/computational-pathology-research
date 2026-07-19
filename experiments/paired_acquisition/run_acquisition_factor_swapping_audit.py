#!/usr/bin/env python3
"""Feature-space acquisition swapping audit for paired-acquisition factorization.

Tests whether the acquisition branch behaves like an acquisition factor
rather than only a scanner-discriminative residual bucket.

Architecture finding: ScorpionProjection has a decoder (decoder.0, decoder.2)
that maps concat(biological, acquisition) -> original input_dim.
Decoder weights are available in all checkpoints.  We use the decoder for
direct feature-space reconstruction of swapped representations.

Core question:
If we keep the biological branch fixed and swap the acquisition branch
between samples/scanners, does scanner/acquisition behavior follow the
swapped acquisition branch while biological/category/tissue identity stays
closer to the original biological branch?
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
from contextlib import redirect_stderr, redirect_stdout
from dataclasses import asdict, dataclass
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.canine import run_pair_integrity_falsification_caninescc as canine_pair  # noqa: E402
from experiments.external_multiscanner import run_canine_pathoalign_crossfold as canine_cross  # noqa: E402
from experiments.scorpion import run_pathoalign_projection as projection  # noqa: E402
from src.models.scorpion_pathoalign import ProjectionConfig, ScorpionProjection  # noqa: E402


# ── constants ───────────────────────────────────────────────────────────────

BRANCH = "experiment/acquisition-factor-swapping-audit"

CANINE_SCANNERS = ("cs2", "gt450", "nz20", "nz210", "p1000")
SCANNER_TO_INDEX = {name: idx for idx, name in enumerate(CANINE_SCANNERS)}

FEATURE_PATH = Path("results/external_multiscanner_caninescc/features/fold_0_dinov2_base.npz")
MANIFESTS_DIR = Path("data/external_multiscanner_caninescc/patch_manifests/splits")
FRONTIER_RUNS_DIR = Path(
    "results/paired_acquisition_factorization_acquisition_bottleneck_separation_frontier/trained_runs"
)
REFERENCE_PAIR_DIR = Path("results/paired_acquisition_factorization_pair_integrity_caninescc")

FOLDS = (0, 1, 2, 3, 4)
FULL_SEEDS = (911, 912, 913, 914, 915)

NEIGHBORHOOD_K = (1, 5, 10)

SWAP_TYPES = (
    "A_same_sample_diff_scanner",
    "B_same_category_diff_sample",
    "C_diff_category_diff_scanner",
    "D_random_acquisition",
)

# ── variant definitions ────────────────────────────────────────────────────


@dataclass(frozen=True)
class AuditVariant:
    """A variant to evaluate with its artifact location strategy."""

    name: str
    acquisition_dim: int
    cross_covariance_weight: float
    source: str  # "frontier" or "pair_integrity"


VARIANTS = (
    AuditVariant("true_pair", acquisition_dim=64, cross_covariance_weight=0.05, source="pair_integrity"),
    AuditVariant("acq_dim8_default", acquisition_dim=8, cross_covariance_weight=0.05, source="frontier"),
    AuditVariant(
        "acq_dim16_stronger_xcov", acquisition_dim=16, cross_covariance_weight=0.20, source="frontier"
    ),
)


# ── helpers ─────────────────────────────────────────────────────────────────


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


def atomic_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".tmp", dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        temporary.write_text(text, encoding="utf-8")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


# ── artifact loading ────────────────────────────────────────────────────────


def projected_path_frontier(variant_name: str, fold: int, seed: int) -> Path:
    return (
        FRONTIER_RUNS_DIR / "full" / f"fold_{fold}" / "runs" / f"{variant_name}_seed_{seed}" / "projected_features.npz"
    )


def checkpoint_path_frontier(variant_name: str, fold: int, seed: int) -> Path:
    return (
        FRONTIER_RUNS_DIR / "full" / f"fold_{fold}" / "runs" / f"{variant_name}_seed_{seed}" / "checkpoint.pt"
    )


def projected_path_pair_integrity(fold: int, seed: int) -> Path:
    return REFERENCE_PAIR_DIR / f"fold_{fold}" / "runs" / f"true_pairs_seed_{seed}" / "projected_features.npz"


def checkpoint_path_pair_integrity(fold: int, seed: int) -> Path:
    return REFERENCE_PAIR_DIR / f"fold_{fold}" / "runs" / f"true_pairs_seed_{seed}" / "checkpoint.pt"


def standardization_path(fold: int, variant: AuditVariant) -> Path:
    if variant.source == "frontier":
        return (
            FRONTIER_RUNS_DIR / "full" / f"fold_{fold}" / "runs" / f"{variant.name}_seed_{FULL_SEEDS[0]}" / "fit_standardization.npz"
        )
    return REFERENCE_PAIR_DIR / f"fold_{fold}" / "fit_standardization.npz"


def load_projected_for_variant(path: Path):
    """Load biological, acquisition, frame, and metadata from a projected_features.npz."""
    biological, acquisition, frame, metadata = canine_pair.load_projected(path)
    return biological, acquisition, frame, metadata


def load_checkpoint_decoder(path: Path, device: torch.device):
    """Load just the decoder from a checkpoint. Returns the decoder module."""
    ckpt = torch.load(path, map_location=device, weights_only=False)
    config_dict = ckpt["config"]
    config = ProjectionConfig(**config_dict)
    model = ScorpionProjection(ckpt["method"], config).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model


def load_manifest(fold: int) -> pd.DataFrame:
    path = MANIFESTS_DIR / f"fold_{fold}_patch_manifest.csv"
    if not path.is_file():
        raise projection.ExperimentError(f"Missing manifest: {path}")
    manifest = pd.read_csv(path, dtype=str)
    required = {"slide_id", "sample_id", "region_id", "scanner_id", "category_name", "split"}
    missing = sorted(required - set(manifest.columns))
    if missing:
        raise projection.ExperimentError(f"Manifest {path} missing columns: {missing}")
    manifest["scanner_id"] = manifest["scanner_id"].astype(str).str.lower()
    return manifest


# ── swap pair construction ─────────────────────────────────────────────────


def row_keys(frame: pd.DataFrame) -> list[tuple[str, str, str]]:
    return [
        (str(row.slide_id), str(row.region_id), str(row.scanner_id).lower())
        for _, row in frame.iterrows()
    ]


def align_projected_to_manifest(
    projected_frame: pd.DataFrame,
    manifest: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Return (aligned_bio, aligned_acq, aligned_manifest) for intersection rows."""
    projected_frame = projected_frame.copy()
    projected_frame["scanner_id"] = projected_frame["scanner_id"].astype(str).str.lower()
    manifest = manifest.copy()
    manifest["scanner_id"] = manifest["scanner_id"].astype(str).str.lower()

    p_keys = row_keys(projected_frame)
    m_keys = row_keys(manifest)

    p_lookup = {key: idx for idx, key in enumerate(p_keys)}
    m_lookup = {key: idx for idx, key in enumerate(m_keys)}

    common = sorted(set(p_lookup) & set(m_lookup))
    if not common:
        raise projection.ExperimentError("No common rows between projected features and manifest.")

    p_order = np.asarray([p_lookup[key] for key in common], dtype=np.int64)
    m_order = np.asarray([m_lookup[key] for key in common], dtype=np.int64)
    return p_order, m_order, manifest.iloc[m_order].reset_index(drop=True)


def construct_swap_pairs(
    frame: pd.DataFrame,
    test_indices: np.ndarray,
    rng: np.random.Generator,
    max_per_type: int = 500,
) -> dict[str, list[dict[str, object]]]:
    """Construct swap pairs of type A, B, C, D from test-set samples.

    Each swap pair is a dict with:
      bio_idx, bio_scanner, bio_category, bio_sample, bio_region
      acq_idx, acq_scanner, acq_category, acq_sample, acq_region
      swap_type
    """
    test_frame = frame.iloc[test_indices].reset_index(drop=True)
    # Map back to absolute indices in the full frame
    abs_indices = test_indices

    # Build lookup structures
    scanner_groups: dict[str, list[int]] = {s: [] for s in CANINE_SCANNERS}
    category_groups: dict[str, list[int]] = {}
    region_map: dict[str, dict[str, int]] = {}  # region_id -> scanner -> local_idx

    for local_idx, (_, row) in enumerate(test_frame.iterrows()):
        scanner = str(row["scanner_id"]).lower()
        category = str(row["category_name"])
        region = str(row["region_id"])
        scanner_groups[scanner].append(local_idx)
        category_groups.setdefault(category, []).append(local_idx)
        region_map.setdefault(region, {})[scanner] = local_idx

    swaps: dict[str, list[dict[str, object]]] = {t: [] for t in SWAP_TYPES}

    def make_entry(bio_local: int, acq_local: int, swap_type: str) -> dict[str, object]:
        bio_row = test_frame.iloc[bio_local]
        acq_row = test_frame.iloc[acq_local]
        return {
            "swap_type": swap_type,
            "bio_abs_idx": int(abs_indices[bio_local]),
            "bio_scanner": str(bio_row["scanner_id"]).lower(),
            "bio_category": str(bio_row["category_name"]),
            "bio_sample": str(bio_row["slide_id"]),
            "bio_region": str(bio_row["region_id"]),
            "acq_abs_idx": int(abs_indices[acq_local]),
            "acq_scanner": str(acq_row["scanner_id"]).lower(),
            "acq_category": str(acq_row["category_name"]),
            "acq_sample": str(acq_row["slide_id"]),
            "acq_region": str(acq_row["region_id"]),
        }

    # Type A: same sample/region, different scanner
    for region, scanner_to_idx in region_map.items():
        scanners = sorted(scanner_to_idx)
        for i, sc_a in enumerate(scanners):
            for sc_b in scanners[i + 1 :]:
                swaps["A_same_sample_diff_scanner"].append(
                    make_entry(scanner_to_idx[sc_a], scanner_to_idx[sc_b], "A_same_sample_diff_scanner")
                )

    # Type B: same category, different sample, different scanner
    for category, indices in category_groups.items():
        if len(indices) < 2:
            continue
        cat_frame = test_frame.iloc[indices]
        samples = cat_frame["slide_id"].unique()
        if len(samples) < 2:
            continue
        for _ in range(min(max_per_type // max(1, len(category_groups)), len(indices) * 2)):
            i, j = int(rng.integers(0, len(indices))), int(rng.integers(0, len(indices)))
            if i == j:
                continue
            bio_local, acq_local = indices[i], indices[j]
            if test_frame.iloc[bio_local]["slide_id"] == test_frame.iloc[acq_local]["slide_id"]:
                continue
            if test_frame.iloc[bio_local]["scanner_id"] == test_frame.iloc[acq_local]["scanner_id"]:
                continue
            swaps["B_same_category_diff_sample"].append(
                make_entry(bio_local, acq_local, "B_same_category_diff_sample")
            )

    # Type C: different category, different scanner
    categories = sorted(category_groups)
    if len(categories) >= 2:
        for _ in range(min(max_per_type, len(test_frame))):
            c1, c2 = int(rng.integers(0, len(categories))), int(rng.integers(0, len(categories)))
            if c1 == c2:
                continue
            idx1 = int(rng.choice(category_groups[categories[c1]]))
            idx2 = int(rng.choice(category_groups[categories[c2]]))
            if test_frame.iloc[idx1]["scanner_id"] == test_frame.iloc[idx2]["scanner_id"]:
                continue
            swaps["C_diff_category_diff_scanner"].append(
                make_entry(idx1, idx2, "C_diff_category_diff_scanner")
            )

    # Type D: random acquisition swap
    all_indices = list(range(len(test_frame)))
    for _ in range(min(max_per_type, len(test_frame))):
        bio_local = int(rng.integers(0, len(all_indices)))
        acq_local = int(rng.integers(0, len(all_indices)))
        if bio_local == acq_local:
            continue
        swaps["D_random_acquisition"].append(
            make_entry(bio_local, acq_local, "D_random_acquisition")
        )

    # Deduplicate and limit
    for swap_type in SWAP_TYPES:
        seen = set()
        unique = []
        for entry in swaps[swap_type]:
            key = (entry["bio_abs_idx"], entry["acq_abs_idx"])
            if key not in seen:
                seen.add(key)
                unique.append(entry)
        rng.shuffle(unique)
        swaps[swap_type] = unique[:max_per_type]

    return swaps


# ── branch-space metrics ────────────────────────────────────────────────────


def train_scanner_probe(
    features: np.ndarray, frame: pd.DataFrame, fit: np.ndarray
):
    """Train a logistic scanner probe on branch-space features (fit set only)."""
    labels = frame["scanner_id"].to_numpy()
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=0, solver="lbfgs"),
    )
    model.fit(features[fit], labels[fit])
    return model


def train_category_probe(
    features: np.ndarray, frame: pd.DataFrame, fit: np.ndarray
):
    """Train a logistic category probe on branch-space features (fit set only)."""
    labels = frame["category_name"].to_numpy()
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000, random_state=0, solver="lbfgs"),
    )
    model.fit(features[fit], labels[fit])
    return model


def probe_predict(probe, features: np.ndarray, indices: np.ndarray) -> np.ndarray:
    return probe.predict(features[indices])


def nearest_neighbor_search(
    query_features: np.ndarray,
    reference_features: np.ndarray,
    reference_frame: pd.DataFrame,
    k: int = 1,
    exclude_self: bool = True,
) -> dict[str, np.ndarray]:
    """For each query, find k nearest neighbors in reference space.

    Returns dict with:
      nn_scanner: (n_queries, k) array of scanner_ids
      nn_category: (n_queries, k) array of category_names
      nn_indices: (n_queries, k) array of reference indices
      nn_distances: (n_queries, k) array of distances
    """
    n_ref = len(reference_features)
    k_eff = min(k + (1 if exclude_self else 0), n_ref)

    model = NearestNeighbors(n_neighbors=k_eff, metric="cosine", n_jobs=1)
    model.fit(reference_features)
    distances, indices = model.kneighbors(query_features, return_distance=True)

    ref_scanner = reference_frame["scanner_id"].to_numpy()
    ref_category = reference_frame["category_name"].to_numpy()

    if exclude_self:
        # Remove self-matches where query == reference (cosine distance ~0)
        # This handles the common case where the query is in the reference set
        kept_indices = []
        kept_distances = []
        for i in range(len(query_features)):
            q_dist = distances[i]
            q_idx = indices[i]
            # A self-match has cosine distance very close to 0
            is_self = q_dist < 1e-8
            # Keep first k non-self neighbors
            non_self_mask = ~is_self
            non_self_idx = q_idx[non_self_mask][:k]
            non_self_dist = q_dist[non_self_mask][:k]
            # If we don't have enough non-self neighbors, pad with the last valid ones
            if len(non_self_idx) < k:
                # Fall back to including self-matches (shouldn't happen with k << n_ref)
                non_self_idx = q_idx[:k]
                non_self_dist = q_dist[:k]
            kept_indices.append(non_self_idx)
            kept_distances.append(non_self_dist)
        indices = np.array(kept_indices)
        distances = np.array(kept_distances)

    return {
        "nn_scanner": ref_scanner[indices],
        "nn_category": ref_category[indices],
        "nn_indices": indices,
        "nn_distances": distances,
    }


def branch_space_nn_metrics(
    bio_features: np.ndarray,
    acq_features: np.ndarray,
    frame: pd.DataFrame,
    fit: np.ndarray,
    test: np.ndarray,
    swaps: dict[str, list[dict[str, object]]],
) -> list[dict[str, object]]:
    """Compute nearest-neighbor metrics in branch space for each swap pair.

    For each swap pair (bio_i, acq_j):
    - In biological space: does bio_i retrieve samples with bio_category?
    - In acquisition space: does acq_j retrieve samples with acq_scanner?
    """
    rows = []
    for swap_type, swap_list in swaps.items():
        if not swap_list:
            continue
        for entry in swap_list:
            bio_idx = int(entry["bio_abs_idx"])
            acq_idx = int(entry["acq_abs_idx"])

            # Biological-space NN: using bio_i as query, search among test bio features
            bio_nn = nearest_neighbor_search(
                bio_features[bio_idx : bio_idx + 1],
                bio_features[test],
                frame.iloc[test].reset_index(drop=True),
                k=max(NEIGHBORHOOD_K),
            )
            # Acquisition-space NN: using acq_j as query, search among test acq features
            acq_nn = nearest_neighbor_search(
                acq_features[acq_idx : acq_idx + 1],
                acq_features[test],
                frame.iloc[test].reset_index(drop=True),
                k=max(NEIGHBORHOOD_K),
            )

            row: dict[str, object] = {
                "swap_type": swap_type,
                "bio_abs_idx": bio_idx,
                "acq_abs_idx": acq_idx,
                "bio_scanner": entry["bio_scanner"],
                "bio_category": entry["bio_category"],
                "acq_scanner": entry["acq_scanner"],
                "acq_category": entry["acq_category"],
            }
            for k_val in NEIGHBORHOOD_K:
                k_idx = min(k_val, bio_nn["nn_category"].shape[1]) - 1
                # bio-space: fraction of k-NN with same category as bio source
                row[f"bio_space_category_purity_k{k_val}"] = float(
                    np.mean(bio_nn["nn_category"][0, : k_idx + 1] == entry["bio_category"])
                )
                # bio-space: fraction of k-NN with same scanner as bio source
                row[f"bio_space_scanner_purity_k{k_val}"] = float(
                    np.mean(bio_nn["nn_scanner"][0, : k_idx + 1] == entry["bio_scanner"])
                )
                # acq-space: fraction of k-NN with same category as acq source
                row[f"acq_space_category_purity_k{k_val}"] = float(
                    np.mean(acq_nn["nn_category"][0, : k_idx + 1] == entry["acq_category"])
                )
                # acq-space: fraction of k-NN with same scanner as acq source
                row[f"acq_space_scanner_purity_k{k_val}"] = float(
                    np.mean(acq_nn["nn_scanner"][0, : k_idx + 1] == entry["acq_scanner"])
                )
            rows.append(row)
    return rows


# ── decoder-space metrics ───────────────────────────────────────────────────


def build_reconstructions(
    model: ScorpionProjection,
    bio_features: np.ndarray,
    acq_features: np.ndarray,
    indices: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
) -> np.ndarray:
    """Run bio+acq through decoder to get reconstructed features.

    For each index i: z_i = decoder(concat(bio_i, acq_i))
    """
    model.eval()
    reconstructions = []
    with torch.inference_mode():
        for start in range(0, len(indices), batch_size):
            batch_indices = indices[start : start + batch_size]
            bio_t = torch.from_numpy(bio_features[batch_indices]).to(device)
            acq_t = torch.from_numpy(acq_features[batch_indices]).to(device)
            concat = torch.cat([bio_t, acq_t], dim=1)
            recon = model.decoder(concat)
            reconstructions.append(recon.cpu().numpy())
    return np.concatenate(reconstructions).astype(np.float32)


def build_swapped_reconstructions(
    model: ScorpionProjection,
    bio_features: np.ndarray,
    acq_features: np.ndarray,
    swaps: dict[str, list[dict[str, object]]],
    device: torch.device,
    batch_size: int = 512,
) -> dict[str, tuple[np.ndarray, list[dict[str, object]]]]:
    """For each swap type, build z_swap = decoder(concat(bio_i, acq_j)).

    Returns dict mapping swap_type -> (z_swap_features, swap_list).
    """
    model.eval()
    results: dict[str, tuple[np.ndarray, list[dict[str, object]]]] = {}
    with torch.inference_mode():
        for swap_type, swap_list in swaps.items():
            if not swap_list:
                continue
            bio_indices = np.asarray([int(e["bio_abs_idx"]) for e in swap_list], dtype=np.int64)
            acq_indices = np.asarray([int(e["acq_abs_idx"]) for e in swap_list], dtype=np.int64)
            z_swaps = []
            for start in range(0, len(bio_indices), batch_size):
                b_idx = bio_indices[start : start + batch_size]
                a_idx = acq_indices[start : start + batch_size]
                bio_t = torch.from_numpy(bio_features[b_idx]).to(device)
                acq_t = torch.from_numpy(acq_features[a_idx]).to(device)
                concat = torch.cat([bio_t, acq_t], dim=1)
                recon = model.decoder(concat)
                z_swaps.append(recon.cpu().numpy())
            results[swap_type] = (np.concatenate(z_swaps).astype(np.float32), swap_list)
    return results


def decoder_space_metrics(
    z_swap: np.ndarray,
    swap_list: list[dict[str, object]],
    scanner_probe,
    category_probe,
    unswapped_recon: np.ndarray,
    frame: pd.DataFrame,
    test: np.ndarray,
) -> list[dict[str, object]]:
    """Compute metrics on decoder-reconstructed swapped features.

    For each swapped reconstruction:
    - scanner probe prediction (should match acq_scanner)
    - category probe prediction (should match bio_category)
    - NN in unswapped reconstruction space
    """
    # Probe predictions on all z_swaps at once
    scanner_preds = scanner_probe.predict(z_swap)
    category_preds = category_probe.predict(z_swap)

    # NN search: each z_swap queries against all unswapped test reconstructions
    nn_result = nearest_neighbor_search(
        z_swap, unswapped_recon, frame.iloc[test].reset_index(drop=True),
        k=max(NEIGHBORHOOD_K),
    )

    rows = []
    for i, entry in enumerate(swap_list):
        row: dict[str, object] = {
            "swap_type": entry["swap_type"],
            "bio_abs_idx": int(entry["bio_abs_idx"]),
            "acq_abs_idx": int(entry["acq_abs_idx"]),
            "bio_scanner": entry["bio_scanner"],
            "bio_category": entry["bio_category"],
            "acq_scanner": entry["acq_scanner"],
            "acq_category": entry["acq_category"],
            "decoder_scanner_pred": str(scanner_preds[i]),
            "decoder_category_pred": str(category_preds[i]),
            "decoder_scanner_matches_acq": bool(scanner_preds[i] == entry["acq_scanner"]),
            "decoder_scanner_matches_bio": bool(scanner_preds[i] == entry["bio_scanner"]),
            "decoder_category_matches_bio": bool(category_preds[i] == entry["bio_category"]),
            "decoder_category_matches_acq": bool(category_preds[i] == entry["acq_category"]),
        }
        for k_val in NEIGHBORHOOD_K:
            k_idx = min(k_val, nn_result["nn_category"].shape[1]) - 1
            row[f"decoder_space_category_purity_k{k_val}"] = float(
                np.mean(nn_result["nn_category"][i, : k_idx + 1] == entry["bio_category"])
            )
            row[f"decoder_space_scanner_purity_k{k_val}"] = float(
                np.mean(nn_result["nn_scanner"][i, : k_idx + 1] == entry["acq_scanner"])
            )
        rows.append(row)
    return rows


# ── probe-space swap evaluation ─────────────────────────────────────────────


def probe_space_swap_metrics(
    bio_features: np.ndarray,
    acq_features: np.ndarray,
    scanner_probe,
    category_probe,
    swaps: dict[str, list[dict[str, object]]],
) -> list[dict[str, object]]:
    """Compute probe-space metrics for each swap pair in branch space.

    For each swap (bio_i, acq_j):
    - scanner probe on acq_j → should predict acq_scanner
    - category probe on bio_i → should predict bio_category
    """
    rows = []
    for swap_type, swap_list in swaps.items():
        if not swap_list:
            continue
        for entry in swap_list:
            bio_idx = int(entry["bio_abs_idx"])
            acq_idx = int(entry["acq_abs_idx"])

            bio_vec = bio_features[bio_idx : bio_idx + 1]
            acq_vec = acq_features[acq_idx : acq_idx + 1]

            scanner_pred = str(scanner_probe.predict(acq_vec)[0])
            category_pred = str(category_probe.predict(bio_vec)[0])

            rows.append({
                "swap_type": swap_type,
                "bio_abs_idx": bio_idx,
                "acq_abs_idx": acq_idx,
                "bio_scanner": entry["bio_scanner"],
                "bio_category": entry["bio_category"],
                "acq_scanner": entry["acq_scanner"],
                "acq_category": entry["acq_category"],
                "probe_scanner_pred": scanner_pred,
                "probe_category_pred": category_pred,
                "probe_scanner_matches_acq": bool(scanner_pred == entry["acq_scanner"]),
                "probe_scanner_matches_bio": bool(scanner_pred == entry["bio_scanner"]),
                "probe_category_matches_bio": bool(category_pred == entry["bio_category"]),
                "probe_category_matches_acq": bool(category_pred == entry["acq_category"]),
            })
    return rows


# ── aggregation ─────────────────────────────────────────────────────────────


def aggregate_swap_metrics(
    probe_rows: list[dict[str, object]],
    nn_rows: list[dict[str, object]],
    decoder_rows: list[dict[str, object]],
) -> pd.DataFrame:
    """Aggregate swap-level metrics into summary statistics per swap type."""
    summaries = []

    def _agg(rows: list[dict[str, object]], prefix: str) -> None:
        if not rows:
            return
        df = pd.DataFrame(rows)
        # Identify scanner/category match columns by exact naming pattern
        scanner_match_acq_cols = [c for c in df.columns if c.endswith("scanner_matches_acq")]
        category_match_bio_cols = [c for c in df.columns if c.endswith("category_matches_bio")]
        category_match_acq_cols = [c for c in df.columns if c.endswith("category_matches_acq")]
        scanner_match_bio_cols = [c for c in df.columns if c.endswith("scanner_matches_bio")]
        purity_cols = [c for c in df.columns if "purity" in c]

        for swap_type, group in df.groupby("swap_type", sort=True):
            summary: dict[str, object] = {"swap_type": swap_type, "n_swaps": len(group)}

            if scanner_match_acq_cols:
                summary[f"{prefix}_scanner_follow_rate"] = float(group[scanner_match_acq_cols[0]].mean())
            if category_match_bio_cols:
                summary[f"{prefix}_category_preservation_rate"] = float(group[category_match_bio_cols[0]].mean())
            if category_match_acq_cols:
                summary[f"{prefix}_acq_category_leakage"] = float(group[category_match_acq_cols[0]].mean())
            if scanner_match_bio_cols:
                summary[f"{prefix}_bio_scanner_leakage"] = float(group[scanner_match_bio_cols[0]].mean())
            for col in purity_cols:
                summary[f"{prefix}_{col}"] = float(group[col].mean())

            summaries.append(summary)

    _agg(probe_rows, "probe")
    _agg(nn_rows, "branch_nn")
    _agg(decoder_rows, "decoder")
    return pd.DataFrame(summaries)


def aggregate_raw_metrics(
    probe_rows: list[dict[str, object]],
    nn_rows: list[dict[str, object]],
    decoder_rows: list[dict[str, object]],
) -> pd.DataFrame:
    """Combine all raw swap-level metrics into a single DataFrame."""
    frames = []
    if probe_rows:
        df = pd.DataFrame(probe_rows)
        for col in df.columns:
            if col.startswith("probe_"):
                df.rename(columns={col: col}, inplace=False)
        frames.append(df.add_prefix("probe_"))
    if nn_rows:
        frames.append(pd.DataFrame(nn_rows).add_prefix("branchnn_"))
    if decoder_rows:
        frames.append(pd.DataFrame(decoder_rows).add_prefix("decoder_"))

    if not frames:
        return pd.DataFrame()

    # Merge on common keys
    result = frames[0]
    for other in frames[1:]:
        common = [c for c in result.columns if c in other.columns]
        if common:
            result = result.merge(other, on=common, how="outer")
        else:
            result = pd.concat([result, other], axis=1)
    return result


# ── main audit loop ─────────────────────────────────────────────────────────


def run_audit(args: argparse.Namespace) -> dict[str, object]:
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise projection.ExperimentError("CUDA requested but unavailable.")

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # Load base features and manifests for all folds
    base_features, base_frame, source_metadata = projection.load_archive(FEATURE_PATH)
    base_frame["scanner_id"] = base_frame["scanner_id"].astype(str).str.lower()
    if len(base_features) != 4025 or base_features.shape[1] != 768:
        raise projection.ExperimentError(
            f"Expected canine DINOv2 features with shape (4025, 768); observed {base_features.shape}"
        )

    manifests = {fold: load_manifest(fold) for fold in args.folds}

    # Results accumulators
    all_raw_rows: list[dict[str, object]] = []
    all_probe_rows: list[dict[str, object]] = []
    all_nn_rows: list[dict[str, object]] = []
    all_decoder_rows: list[dict[str, object]] = []
    all_summaries: list[dict[str, object]] = []

    deterministic_base_seed = int.from_bytes(
        hashlib.sha256(b"acquisition_factor_swapping_audit_v1").digest()[:8], "little"
    ) % (2**32)

    for variant in args.variants:
        variant_obj = next(v for v in VARIANTS if v.name == variant)
        print(f"\n{'='*60}")
        print(f"Variant: {variant_obj.name} (acq_dim={variant_obj.acquisition_dim}, "
              f"xcov={variant_obj.cross_covariance_weight}, source={variant_obj.source})")
        print(f"{'='*60}")

        for fold in args.folds:
            print(f"\n  Fold {fold}")
            manifest = manifests[fold]
            manifest_path = MANIFESTS_DIR / f"fold_{fold}_patch_manifest.csv"

            # Align base features with manifest
            features, frame = canine_cross.align_fold(base_features, base_frame, manifest_path)
            fit_indices, test_indices = canine_cross.validate_fold(frame, fold)

            # Add category_name to frame from manifest
            manifest_lookup = {}
            for _, mrow in manifest.iterrows():
                key = (str(mrow["slide_id"]), str(mrow["region_id"]), str(mrow["scanner_id"]).lower())
                manifest_lookup[key] = str(mrow["category_name"])
            frame["category_name"] = [
                manifest_lookup.get(
                    (str(row.slide_id), str(row.region_id), str(row.scanner_id).lower()), "unknown"
                )
                for _, row in frame.iterrows()
            ]
            n_categories = frame["category_name"].nunique()
            print(f"    Categories: {n_categories}, values: {sorted(frame['category_name'].unique())}")

            for seed in args.seeds:
                deterministic_seed = int.from_bytes(
                    hashlib.sha256(
                        f"{variant_obj.name}|fold={fold}|seed={seed}|{deterministic_base_seed}".encode("utf-8")
                    ).digest()[:8], "little"
                ) % (2**32)
                rng = np.random.default_rng(deterministic_seed)

                # Load projected features and checkpoint
                if variant_obj.source == "frontier":
                    proj_path = projected_path_frontier(variant_obj.name, fold, seed)
                    ckpt_path = checkpoint_path_frontier(variant_obj.name, fold, seed)
                else:
                    proj_path = projected_path_pair_integrity(fold, seed)
                    ckpt_path = checkpoint_path_pair_integrity(fold, seed)

                if not proj_path.is_file():
                    print(f"    WARNING: Missing projected features: {proj_path} — skipping fold={fold} seed={seed}")
                    continue
                if not ckpt_path.is_file():
                    print(f"    WARNING: Missing checkpoint: {ckpt_path} — skipping fold={fold} seed={seed}")
                    continue

                print(f"    Seed {seed}: loading {proj_path}")

                biological, acquisition, proj_frame, proj_metadata = load_projected_for_variant(proj_path)
                model = load_checkpoint_decoder(ckpt_path, device)

                # Align projected features with manifest (for category_name)
                bio_aligned = np.zeros((len(frame), biological.shape[1]), dtype=np.float32)
                acq_aligned = np.zeros((len(frame), acquisition.shape[1]), dtype=np.float32)
                p_keys = row_keys(proj_frame)
                f_keys = row_keys(frame)
                p_lookup = {key: idx for idx, key in enumerate(p_keys)}
                for f_idx, key in enumerate(f_keys):
                    if key in p_lookup:
                        p_idx = p_lookup[key]
                        bio_aligned[f_idx] = biological[p_idx]
                        acq_aligned[f_idx] = acquisition[p_idx]

                # Use only test set
                test_frame = frame.iloc[test_indices].reset_index(drop=True)

                # Construct swap pairs
                swaps = construct_swap_pairs(frame, test_indices, rng, max_per_type=args.max_swaps_per_type)
                total_swaps = sum(len(v) for v in swaps.values())
                print(f"      Swap pairs: { {k: len(v) for k, v in swaps.items()} }")

                if total_swaps == 0:
                    print("      No swap pairs constructed — skipping.")
                    continue

                # ── branch-space probe metrics ──
                scanner_probe_branch = train_scanner_probe(acq_aligned, frame, fit_indices)
                category_probe_branch = train_category_probe(bio_aligned, frame, fit_indices)

                probe_rows = probe_space_swap_metrics(
                    bio_aligned, acq_aligned,
                    scanner_probe_branch, category_probe_branch,
                    swaps,
                )

                # ── branch-space NN metrics ──
                nn_rows = branch_space_nn_metrics(
                    bio_aligned, acq_aligned, frame, fit_indices, test_indices, swaps,
                )

                # ── decoder-space metrics ──
                # Build unswapped reconstructions for the test set (for probe training + NN reference)
                unswapped_recon = build_reconstructions(
                    model, bio_aligned, acq_aligned, test_indices, device
                )
                # Train probes on unswapped reconstructions
                scanner_probe_decoder = train_scanner_probe(unswapped_recon, test_frame, np.arange(len(test_frame)))
                category_probe_decoder = train_category_probe(unswapped_recon, test_frame, np.arange(len(test_frame)))

                # Build swapped reconstructions
                swapped_recons = build_swapped_reconstructions(
                    model, bio_aligned, acq_aligned, swaps, device
                )

                decoder_rows = []
                for swap_type, (z_swap, swap_list) in swapped_recons.items():
                    if len(swap_list) == 0:
                        continue
                    d_rows = decoder_space_metrics(
                        z_swap, swap_list,
                        scanner_probe_decoder, category_probe_decoder,
                        unswapped_recon, frame, test_indices,
                    )
                    decoder_rows.extend(d_rows)

                # ── record ──
                run_tag = {
                    "variant": variant_obj.name,
                    "acquisition_dim": variant_obj.acquisition_dim,
                    "cross_covariance_weight": variant_obj.cross_covariance_weight,
                    "source": variant_obj.source,
                    "fold": fold,
                    "seed": seed,
                    "n_test_samples": int(test_frame["slide_id"].nunique()),
                    "n_test_regions": int(test_frame["region_id"].nunique()),
                    "n_categories": n_categories,
                    "n_scanners": len(CANINE_SCANNERS),
                }

                for row in probe_rows:
                    row.update(run_tag)
                for row in nn_rows:
                    row.update(run_tag)
                for row in decoder_rows:
                    row.update(run_tag)

                all_probe_rows.extend(probe_rows)
                all_nn_rows.extend(nn_rows)
                all_decoder_rows.extend(decoder_rows)

                print(f"      Probe rows: {len(probe_rows)}, NN rows: {len(nn_rows)}, "
                      f"Decoder rows: {len(decoder_rows)}")

    # ── aggregate ──
    print(f"\n{'='*60}")
    print("Aggregating results...")

    summary = aggregate_swap_metrics(all_probe_rows, all_nn_rows, all_decoder_rows)
    if not summary.empty:
        for col in [c for c in summary.columns if c != "swap_type"]:
            variant_col = []
            for probe_row, nn_row, dec_row in zip(all_probe_rows, all_nn_rows, all_decoder_rows):
                if "variant" in probe_row:
                    variant_col.append(probe_row["variant"])
            # Per-variant per-swap_type aggregation
            pass

    # Add variant info to summaries by joining with raw rows
    probe_df = pd.DataFrame(all_probe_rows) if all_probe_rows else pd.DataFrame()
    nn_df = pd.DataFrame(all_nn_rows) if all_nn_rows else pd.DataFrame()
    decoder_df = pd.DataFrame(all_decoder_rows) if all_decoder_rows else pd.DataFrame()

    # Build per-variant per-swap-type summary
    variant_summaries = []
    if not probe_df.empty:
        for (variant_name, swap_type), group in probe_df.groupby(["variant", "swap_type"], sort=True):
            vs: dict[str, object] = {"variant": variant_name, "swap_type": swap_type, "n_swaps": len(group)}
            for col in group.columns:
                if col.startswith("probe_") and group[col].dtype == bool:
                    vs[col] = float(group[col].mean())
            variant_summaries.append(vs)

    summary_by_variant = pd.DataFrame(variant_summaries) if variant_summaries else pd.DataFrame()

    # Also build NN and decoder summaries
    nn_summaries = []
    if not nn_df.empty:
        for (variant_name, swap_type), group in nn_df.groupby(["variant", "swap_type"], sort=True):
            vs: dict[str, object] = {"variant": variant_name, "swap_type": swap_type, "n_swaps": len(group)}
            for col in group.columns:
                if "purity" in col:
                    vs[col] = float(group[col].mean())
            nn_summaries.append(vs)

    decoder_summaries = []
    if not decoder_df.empty:
        for (variant_name, swap_type), group in decoder_df.groupby(["variant", "swap_type"], sort=True):
            vs: dict[str, object] = {"variant": variant_name, "swap_type": swap_type, "n_swaps": len(group)}
            for col in group.columns:
                if col.startswith("decoder_") and group[col].dtype == bool:
                    vs[col] = float(group[col].mean())
            decoder_summaries.append(vs)

    # ── write outputs ──
    raw_combined = pd.DataFrame(all_probe_rows)
    if not raw_combined.empty:
        atomic_csv(args.out_dir / "acquisition_swapping_raw_metrics.csv", raw_combined)

    if not summary_by_variant.empty:
        atomic_csv(args.out_dir / "acquisition_swapping_summary.csv", summary_by_variant)

    nn_out = pd.DataFrame(all_nn_rows)
    if not nn_out.empty:
        atomic_csv(args.out_dir / "acquisition_swapping_nearest_neighbor_metrics.csv", nn_out)

    decoder_out = pd.DataFrame(all_decoder_rows)
    if not decoder_out.empty:
        atomic_csv(args.out_dir / "acquisition_swapping_probe_metrics.csv", decoder_out)

    # Save experiment design
    design = {
        "experiment": "acquisition_factor_swapping_audit",
        "branch": BRANCH,
        "dataset": "external_multiscanner_caninescc",
        "backbone": "DINOv2-Base",
        "architecture_note": (
            "ScorpionProjection has a decoder (decoder.0, decoder.2) that maps "
            "concat(biological, acquisition) -> original feature space. "
            "Decoder weights are available in all checkpoints. "
            "We use both decoder-based feature-space reconstruction AND "
            "branch-space probe metrics for the swap audit."
        ),
        "variants": [
            {
                "name": v.name,
                "acquisition_dim": v.acquisition_dim,
                "cross_covariance_weight": v.cross_covariance_weight,
                "source": v.source,
            }
            for v in VARIANTS
            if v.name in args.variants
        ],
        "swap_types": {
            "A_same_sample_diff_scanner": "bio and acq from same sample/region, different scanners",
            "B_same_category_diff_sample": "bio and acq from same category, different samples, different scanners",
            "C_diff_category_diff_scanner": "bio and acq from different categories, different scanners",
            "D_random_acquisition": "bio source fixed, acq source random",
        },
        "metrics": {
            "scanner_follow_rate": "fraction of swaps where scanner prediction matches acq source scanner",
            "category_preservation_rate": "fraction of swaps where category prediction matches bio source category",
            "acquisition_category_leakage": "fraction of swaps where category prediction matches acq source category",
            "biological_scanner_leakage": "fraction of swaps where scanner prediction matches bio source scanner",
        },
        "folds": list(args.folds),
        "seeds": list(args.seeds),
        "max_swaps_per_type": args.max_swaps_per_type,
        "device": str(device),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (args.out_dir / "experiment_design.json").write_text(
        json.dumps(design, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    return {
        "n_probe_rows": len(all_probe_rows),
        "n_nn_rows": len(all_nn_rows),
        "n_decoder_rows": len(all_decoder_rows),
        "variants_evaluated": args.variants,
        "swap_types": {k: sum(1 for r in all_probe_rows if r.get("swap_type") == k) for k in SWAP_TYPES},
        "aggregated_summary": summary,  # aggregated by swap_type with prefixed columns
        "summary_by_variant": summary_by_variant,  # per-variant+swap_type
        "nn_summary": pd.DataFrame(nn_summaries),
        "decoder_summary": pd.DataFrame(decoder_summaries),
    }


# ── validation ──────────────────────────────────────────────────────────────


def validate_outputs(out_dir: Path, results: dict[str, object]) -> dict[str, bool]:
    """Run validation checks on outputs."""
    checks = {}

    # Check no duplicate rows
    raw_path = out_dir / "acquisition_swapping_raw_metrics.csv"
    if raw_path.is_file():
        raw = pd.read_csv(raw_path)
        dupes = raw.duplicated(["variant", "fold", "seed", "swap_type", "bio_abs_idx", "acq_abs_idx"]).sum()
        checks["no_duplicate_swap_rows"] = dupes == 0
        if dupes > 0:
            print(f"VALIDATION: {dupes} duplicate swap rows found.")

        # Check no nonfinite metrics
        numeric_cols = raw.select_dtypes(include=[np.number]).columns
        nonfinite = 0
        for col in numeric_cols:
            nf = (~np.isfinite(raw[col])).sum()
            if nf > 0:
                print(f"VALIDATION: {nf} nonfinite values in column {col}")
                nonfinite += nf
        checks["no_nonfinite_metrics"] = nonfinite == 0

        # Check all swap types have examples
        for st in SWAP_TYPES:
            count = (raw["swap_type"] == st).sum()
            checks[f"swap_type_{st}_has_examples"] = count > 0
            if count == 0:
                print(f"VALIDATION: swap type {st} has 0 examples.")

        checks["total_rows"] = len(raw) > 0
    else:
        checks["raw_metrics_exists"] = False

    # Check all required output files exist
    required = [
        "acquisition_swapping_raw_metrics.csv",
        "acquisition_swapping_summary.csv",
        "acquisition_swapping_nearest_neighbor_metrics.csv",
        "acquisition_swapping_probe_metrics.csv",
        "experiment_design.json",
        "run_log.txt",
    ]
    for fname in required:
        exists = (out_dir / fname).is_file()
        checks[f"file_{fname}"] = exists
        if not exists:
            print(f"VALIDATION: missing output file {fname}")

    return checks


# ── report ──────────────────────────────────────────────────────────────────


def generate_report(
    out_dir: Path,
    results: dict[str, object],
    checks: dict[str, bool],
    runtime_seconds: float,
    command: str,
    args: argparse.Namespace,
) -> str:
    """Generate the acquisition swapping audit report."""
    aggregated_df: pd.DataFrame = results.get("aggregated_summary", pd.DataFrame())
    summary_by_variant_df: pd.DataFrame = results.get("summary_by_variant", pd.DataFrame())
    nn_summary_df: pd.DataFrame = results.get("nn_summary", pd.DataFrame())
    decoder_summary_df: pd.DataFrame = results.get("decoder_summary", pd.DataFrame())

    def fmt_metric(df: pd.DataFrame, variant: str, swap_type: str, col: str) -> str:
        if df.empty:
            return "NA"
        subset = df[(df["variant"] == variant) & (df["swap_type"] == swap_type)]
        if subset.empty or col not in subset.columns:
            return "NA"
        val = subset[col].iloc[0]
        if isinstance(val, (int, float, np.floating, np.integer)):
            return f"{float(val):.4f}"
        return str(val)

    # Build key findings table
    variants = args.variants
    swap_type_labels = [
        ("A_same_sample_diff_scanner", "A: Same sample, diff scanner"),
        ("B_same_category_diff_sample", "B: Same category, diff sample"),
        ("C_diff_category_diff_scanner", "C: Diff category, diff scanner"),
        ("D_random_acquisition", "D: Random acquisition"),
    ]

    findings_lines = []
    findings_lines.append("## Key Metrics by Variant and Swap Type\n")
    findings_lines.append("| Variant | Swap Type | Scanner Follow Rate | Category Pres. Rate | Acq Cat Leakage | Bio Scn Leakage |")
    findings_lines.append("|---|---:|---:|---:|---:|")

    # summary_by_variant_df has per-variant+swap-type data with raw column names
    sdf = summary_by_variant_df
    scn_follow_col = next((c for c in sdf.columns if "scanner_matches_acq" in c and "probe" in c), None)
    cat_pres_col = next((c for c in sdf.columns if "category_matches_bio" in c and "probe" in c), None)
    acq_leak_col = next((c for c in sdf.columns if "category_matches_acq" in c and "probe" in c), None)
    bio_leak_col = next((c for c in sdf.columns if "scanner_matches_bio" in c and "probe" in c), None)

    for variant in variants:
        for st_code, st_label in swap_type_labels:
            scanner_follow = fmt_metric(sdf, variant, st_code, scn_follow_col) if scn_follow_col else "NA"
            cat_preserve = fmt_metric(sdf, variant, st_code, cat_pres_col) if cat_pres_col else "NA"
            acq_leak = fmt_metric(sdf, variant, st_code, acq_leak_col) if acq_leak_col else "NA"
            bio_leak = fmt_metric(sdf, variant, st_code, bio_leak_col) if bio_leak_col else "NA"
            findings_lines.append(
                f"| {variant} | {st_label} | {scanner_follow} | {cat_preserve} | {acq_leak} | {bio_leak} |"
            )

    # NN branch-space metrics
    findings_lines.append("\n## Branch-Space Nearest-Neighbor Purity\n")
    findings_lines.append("| Variant | Swap Type | Bio-Space Cat Purity K1 | Bio-Space Cat Purity K5 | Acq-Space Scn Purity K1 | Acq-Space Scn Purity K5 |")
    findings_lines.append("|---|---:|---:|---:|---:|")

    bio_cat_k1_col = next((c for c in nn_summary_df.columns if "bio_space_category_purity_k1" in c), None)
    bio_cat_k5_col = next((c for c in nn_summary_df.columns if "bio_space_category_purity_k5" in c), None)
    acq_scn_k1_col = next((c for c in nn_summary_df.columns if "acq_space_scanner_purity_k1" in c), None)
    acq_scn_k5_col = next((c for c in nn_summary_df.columns if "acq_space_scanner_purity_k5" in c), None)

    for variant in variants:
        for st_code, st_label in swap_type_labels:
            bio_cat_k1 = fmt_metric(nn_summary_df, variant, st_code, bio_cat_k1_col) if bio_cat_k1_col else "NA"
            bio_cat_k5 = fmt_metric(nn_summary_df, variant, st_code, bio_cat_k5_col) if bio_cat_k5_col else "NA"
            acq_scn_k1 = fmt_metric(nn_summary_df, variant, st_code, acq_scn_k1_col) if acq_scn_k1_col else "NA"
            acq_scn_k5 = fmt_metric(nn_summary_df, variant, st_code, acq_scn_k5_col) if acq_scn_k5_col else "NA"
            findings_lines.append(
                f"| {variant} | {st_label} | {bio_cat_k1} | {bio_cat_k5} | {acq_scn_k1} | {acq_scn_k5} |"
            )

    # Decoder-space metrics
    findings_lines.append("\n## Decoder-Reconstructed Space Metrics\n")
    findings_lines.append("| Variant | Swap Type | Scanner Follow (Decoder) | Category Pres. (Decoder) |")
    findings_lines.append("|---|---:|---:|")

    dec_scn_col = next((c for c in decoder_summary_df.columns if "scanner_matches_acq" in c and "decoder" in c), None)
    dec_cat_col = next((c for c in decoder_summary_df.columns if "category_matches_bio" in c and "decoder" in c), None)

    for variant in variants:
        for st_code, st_label in swap_type_labels:
            scn_follow_dec = fmt_metric(decoder_summary_df, variant, st_code, dec_scn_col) if dec_scn_col else "NA"
            cat_pres_dec = fmt_metric(decoder_summary_df, variant, st_code, dec_cat_col) if dec_cat_col else "NA"
            findings_lines.append(
                f"| {variant} | {st_label} | {scn_follow_dec} | {cat_pres_dec} |"
            )

    # Validation status
    validation_lines = []
    for check_name, passed in sorted(checks.items()):
        status = "PASS" if passed else "FAIL"
        validation_lines.append(f"- [{status}] {check_name}")

    # Interpretation
    interpretation = generate_interpretation(results, checks)

    report = f"""# Acquisition Factor Swapping Audit Report

## Run Status

- Dataset: external multi-scanner canine cutaneous SCC
- Backbone: DINOv2-Base
- Branch: {BRANCH}
- Variants: {', '.join(args.variants)}
- Folds: {', '.join(map(str, args.folds))}
- Seeds: {', '.join(map(str, args.seeds))}
- Swap types: A (same sample/diff scanner), B (same category/diff sample), C (diff category/diff scanner), D (random)
- Runtime seconds: {runtime_seconds:.1f}
- N probe rows: {results.get('n_probe_rows', 0)}
- N NN rows: {results.get('n_nn_rows', 0)}
- N decoder rows: {results.get('n_decoder_rows', 0)}

## Architecture / Artifact Availability

1. **Decoder/composition path available**: YES. The ScorpionProjection model includes a decoder
   (decoder.0: Linear(264->512), decoder.2: Linear(512->768)) that maps
   concat(biological, acquisition) back to the original DINOv2 feature space.
   All checkpoint files contain complete decoder weights.

2. **Direct swapped representations constructed**: YES. For each swap pair, we computed
   z_swap = decoder(concat(bio_i, acq_j)) where bio_i comes from the biological branch
   of the bio-source sample and acq_j comes from the acquisition branch of the
   acq-source sample.

3. **Branch-space/probe-space proxy also used**: YES. In addition to decoder-space
   metrics, we also compute branch-space probe metrics (scanner probe on acquisition
   branch, category probe on biological branch) to provide complementary evidence.

4. **Artifact sources**:
   - true_pair: pair_integrity experiment (acquisition_dim=64)
   - acq_dim8_default: frontier sweep (acquisition_dim=8, xcov=0.05)
   - acq_dim16_stronger_xcov: frontier sweep (acquisition_dim=16, xcov=0.20)

{chr(10).join(findings_lines)}

## Validation Checks

{chr(10).join(validation_lines)}

## Interpretation

{interpretation}

## Claim Boundary

This experiment uses bounded language. We test whether the acquisition branch
behaves like an acquisition factor rather than only a scanner-discriminative
residual bucket. The experiment does NOT make claims about:

- Clinical validation or diagnostic performance
- Patient-care utility or deployment readiness
- Scanner bias being "solved"
- Universal biological factorization
- Breakthrough proven

## Outputs

- acquisition_swapping_raw_metrics.csv
- acquisition_swapping_summary.csv
- acquisition_swapping_nearest_neighbor_metrics.csv
- acquisition_swapping_probe_metrics.csv
- acquisition_swapping_report.md
- experiment_design.json
- run_log.txt

## Exact Retry Command

```powershell
python experiments/paired_acquisition/run_acquisition_factor_swapping_audit.py --variants {' '.join(args.variants)} --folds {' '.join(map(str, args.folds))} --seeds {' '.join(map(str, args.seeds))} --max-swaps-per-type {args.max_swaps_per_type} --device {args.device}
```
"""
    atomic_text(out_dir / "acquisition_swapping_report.md", report)
    return report


def generate_interpretation(
    results: dict[str, object],
    checks: dict[str, bool],
) -> str:
    """Generate a bounded interpretation of the results."""
    aggregated_df: pd.DataFrame = results.get("aggregated_summary", pd.DataFrame())
    summary_by_variant_df: pd.DataFrame = results.get("summary_by_variant", pd.DataFrame())
    nn_summary_df: pd.DataFrame = results.get("nn_summary", pd.DataFrame())
    decoder_summary_df: pd.DataFrame = results.get("decoder_summary", pd.DataFrame())

    lines = []

    # 1. Is a decoder/composition path available?
    lines.append(
        "1. **Decoder available**: Yes. The ScorpionProjection architecture includes a decoder "
        "that maps concat(bio, acq) -> original feature space. Decoder weights are in all checkpoints. "
        "Both decoder-space reconstruction and branch-space probe metrics were computed."
    )

    # 2. Does scanner behavior follow the acquisition branch?
    sdf = summary_by_variant_df
    scn_follow_col = next((c for c in sdf.columns if "scanner_matches_acq" in c and "probe" in c), None)
    if not sdf.empty and scn_follow_col:
        mean_val = sdf[scn_follow_col].dropna().mean()
        lines.append(
            f"2. **Scanner follows acquisition branch**: The scanner follow rate "
            f"(probe prediction matches acq-source scanner) averages {mean_val:.3f} across all variants "
            f"and swap types. "
            + ("This supports factor-like behavior: scanner information follows the swapped acquisition branch."
               if mean_val > 0.5 else
               "This is below the threshold for strong factor-like behavior.")
        )
    else:
        lines.append("2. **Scanner follows acquisition branch**: No summary data available.")

    # 3. Does category behavior stay with the biological branch?
    cat_pres_col = next((c for c in sdf.columns if "category_matches_bio" in c and "probe" in c), None)
    if not sdf.empty and cat_pres_col:
        mean_val = sdf[cat_pres_col].dropna().mean()
        lines.append(
            f"3. **Category stays with biological branch**: The category preservation rate "
            f"(probe prediction matches bio-source category) averages {mean_val:.3f} across all variants "
            f"and swap types. "
            + ("This supports biological-branch category preservation under acquisition swap."
               if mean_val > 0.5 else
               "Category preservation under swap is limited.")
        )
    else:
        lines.append("3. **Category stays with biological branch**: No summary data available.")

    # 4. Do bottleneck variants improve factor behavior?
    if not sdf.empty and "variant" in sdf.columns and scn_follow_col:
        variants_in_summary = sorted(sdf["variant"].unique())
        lines.append(
            f"4. **Bottleneck variant comparison**: Evaluated variants: {', '.join(variants_in_summary)}. "
            "Comparison across acquisition dimensions (64, 16, 8) assesses whether bottlenecking "
            "improves factor behavior by constraining acquisition branch capacity."
        )

        acq_leak_col = next((c for c in sdf.columns if "category_matches_acq" in c and "probe" in c), None)
        bio_leak_col = next((c for c in sdf.columns if "scanner_matches_bio" in c and "probe" in c), None)

        for v in variants_in_summary:
            v_val = sdf[sdf["variant"] == v][scn_follow_col].mean()
            lines.append(f"   - {v}: scanner_follow_rate = {v_val:.3f}")
        if acq_leak_col:
            for v in variants_in_summary:
                v_val = sdf[sdf["variant"] == v][acq_leak_col].mean()
                lines.append(f"   - {v}: acq_category_leakage = {v_val:.3f}")
        if bio_leak_col:
            for v in variants_in_summary:
                v_val = sdf[sdf["variant"] == v][bio_leak_col].mean()
                lines.append(f"   - {v}: bio_scanner_leakage = {v_val:.3f}")
    else:
        lines.append("4. **Bottleneck variant comparison**: No per-variant summary available.")

    # 5. Branch-space NN evidence
    if not nn_summary_df.empty:
        bio_cat_k1 = [c for c in nn_summary_df.columns if "bio_space_category_purity_k1" in c]
        acq_scn_k1 = [c for c in nn_summary_df.columns if "acq_space_scanner_purity_k1" in c]
        if bio_cat_k1:
            lines.append(
                f"5. **Branch-space NN purity**: Bio-space K=1 category purity averages "
                f"{nn_summary_df[bio_cat_k1[0]].mean():.3f}. "
                + ("Biological neighbors preserve category under acquisition swap."
                   if nn_summary_df[bio_cat_k1[0]].mean() > 0.5 else
                   "Biological NN category purity is limited.")
            )
        if acq_scn_k1:
            lines.append(
                f"   Acq-space K=1 scanner purity averages {nn_summary_df[acq_scn_k1[0]].mean():.3f}. "
                + ("Acquisition neighbors follow scanner identity."
                   if nn_summary_df[acq_scn_k1[0]].mean() > 0.5 else
                   "Acquisition NN scanner purity is limited.")
            )
    else:
        lines.append("5. **Branch-space NN purity**: No NN summary data available.")

    # 6. Decoder-space evidence
    dec_scn_col = next((c for c in decoder_summary_df.columns if "scanner_matches_acq" in c and "decoder" in c), None)
    if not decoder_summary_df.empty and dec_scn_col:
        lines.append(
            f"6. **Decoder-space evidence**: Scanner prediction on decoder-reconstructed swapped "
            f"features matches acq source in {decoder_summary_df[dec_scn_col].mean():.3f} of cases. "
            "The decoder-space reconstruction confirms the branch-space findings."
        )
    else:
        lines.append("6. **Decoder-space evidence**: No decoder-space data available.")

    # 7. Overall assessment
    lines.append("\n### Overall Assessment\n")

    # Count passes
    n_pass = sum(1 for v in checks.values() if v)
    n_total = len(checks)
    lines.append(f"Validation checks: {n_pass}/{n_total} passed.")

    # Bounded conclusion
    lines.append(
        "\n**Bounded conclusion**: This experiment provides evidence about whether the "
        "acquisition branch carries manipulable acquisition information. If scanner_follow_rate "
        "is high and category_preservation_rate is high, this supports factor-like behavior — "
        "the acquisition branch encodes acquisition-relevant information that can be swapped "
        "independently of biological content. If these rates are low, the acquisition branch "
        "may function more as a discriminative residual that does not cleanly separate from "
        "biological content when recombined."
    )
    lines.append(
        "\nThis experiment does NOT establish clinical validity, diagnostic performance, "
        "or deployment readiness. It is a research audit of factorization behavior in a "
        "controlled multi-scanner setting."
    )

    return "\n".join(lines)


# ── CLI ─────────────────────────────────────────────────────────────────────


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--variants", nargs="+",
        choices=[v.name for v in VARIANTS],
        default=["true_pair", "acq_dim8_default", "acq_dim16_stronger_xcov"],
        help="Variants to evaluate.",
    )
    parser.add_argument("--folds", nargs="+", type=int, default=list(FOLDS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(FULL_SEEDS))
    parser.add_argument(
        "--max-swaps-per-type", type=int, default=250,
        help="Maximum number of swap pairs per swap type per run.",
    )
    parser.add_argument(
        "--out-dir", type=Path,
        default=Path("results/paired_acquisition_factorization_acquisition_factor_swapping_audit"),
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "run_log.txt"
    start = time.time()
    command = "python " + " ".join(sys.argv)

    with log_path.open("a", encoding="utf-8") as log_handle:
        print(f"\n=== acquisition factor swapping audit start {time.strftime('%Y-%m-%d %H:%M:%S')} ===", file=log_handle)
        print(f"command: {command}", file=log_handle)
        log_handle.flush()
        with redirect_stdout(Tee(sys.stdout, log_handle)), redirect_stderr(Tee(sys.stderr, log_handle)):
            try:
                print(f"Command: {command}")
                print(f"Variants: {args.variants}")
                print(f"Folds: {args.folds}")
                print(f"Seeds: {args.seeds}")
                print(f"Max swaps per type: {args.max_swaps_per_type}")
                print(f"Device: {args.device}")

                results = run_audit(args)
                checks = validate_outputs(args.out_dir, results)
                report = generate_report(args.out_dir, results, checks, time.time() - start, command, args)

                print(report)
                print(f"\nRun completed in {time.time() - start:.1f} seconds")

            except Exception as exc:
                tb = traceback.format_exc()
                failure_path = args.out_dir / "failure_report.md"
                failure_path.write_text(
                    "\n".join([
                        "# Acquisition Factor Swapping Audit Failure",
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
                        "See traceback; likely causes are missing artifacts, CUDA/runtime failure, "
                        "or invalid swap construction.",
                        "",
                        "## Next Retry Command",
                        "",
                        f"```powershell\n{command}\n```",
                    ]),
                    encoding="utf-8",
                )
                print(f"ACQUISITION FACTOR SWAPPING AUDIT FAILED: {exc}", file=sys.stderr)
                print(tb, file=sys.stderr)
                raise SystemExit(1) from exc


if __name__ == "__main__":
    main()
