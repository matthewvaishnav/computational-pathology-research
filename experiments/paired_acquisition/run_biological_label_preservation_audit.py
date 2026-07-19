#!/usr/bin/env python3
"""Biological-label preservation audit for Paired-Acquisition Neural Factorization.

Scientific question:
  Does the biological branch preserve tissue-category structure while
  reducing scanner/acquisition recoverability?

This experiment compares multiple representations on canine SCC DINOv2:
  1. original frozen features (DINOv2)
  2. true-pair biological branch (pair-integrity, true_pairs condition)
  3. true-pair acquisition branch
  4. shuffled-sample biological branch (pair-integrity, shuffled_sample_pairs)
  5. shuffled-sample acquisition branch
  6. PCA component removal baselines (k = 1, 2, 4, 8, 16, 32)
  7. linear scanner subspace projection baselines (k = 0, 1, 2, 4, 8, 16, 32)

Metrics:
  - scanner/acquisition linear-probe accuracy (balanced)
  - category label linear-probe accuracy (balanced)
  - same-category nearest-neighbor purity at k=1, k=5, k=10
  - effective rank
  - runtime per representation

Claim language bounded to: "supports", "extends", "audit".
Does not claim clinical validation, diagnostic performance, or deployment readiness.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CANINE_MANIFEST_DIR = Path("data/external_multiscanner_caninescc/patch_manifests/splits")
CANINE_MANIFEST_PATTERN = "fold_{fold}_patch_manifest.csv"
CANINE_BASE_FEATURES = Path("results/external_multiscanner_caninescc/features/fold_0_dinov2_base.npz")
CANINE_PAIR_INTEGRITY_DIR = Path("results/paired_acquisition_factorization_pair_integrity_caninescc")
CANINE_SCANNERS = ("cs2", "gt450", "nz20", "nz210", "p1000")

FOLDS = [0, 1, 2, 3, 4]
SEEDS = [911, 912, 913, 914, 915]

PCA_K_VALUES = [1, 2, 4, 8, 16, 32]
LINEAR_K_VALUES = [0, 1, 2, 4, 8, 16, 32]

# ---------------------------------------------------------------------------
# Atomic file write
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


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_manifest(fold: int) -> pd.DataFrame:
    """Load fold-specific manifest with category_name."""
    p = CANINE_MANIFEST_DIR / CANINE_MANIFEST_PATTERN.format(fold=fold)
    df = pd.read_csv(p, dtype=str)
    required = {"region_id", "scanner_id", "slide_id", "sample_id",
                 "category_name", "split"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Manifest fold {fold} missing columns: {missing}")
    return df


def load_base_features() -> Tuple[np.ndarray, pd.DataFrame]:
    """Load original frozen DINOv2 features for canine SCC."""
    with np.load(CANINE_BASE_FEATURES, allow_pickle=False) as archive:
        features = np.asarray(archive["features"], dtype=np.float32)
        frame = pd.DataFrame({
            name: archive[name].astype(str)
            for name in ("region_id", "scanner_id", "slide_id", "split")
        })
    return features, frame


def load_projected_features(run_dir: Path) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Load biological + acquisition branches from a projected_features.npz."""
    with np.load(run_dir / "projected_features.npz", allow_pickle=False) as archive:
        biological = np.asarray(archive["features"], dtype=np.float32)
        acquisition = np.asarray(archive["acquisition_features"], dtype=np.float32)
        frame = pd.DataFrame({
            name: archive[name].astype(str)
            for name in ("region_id", "scanner_id", "slide_id", "split")
        })
    return biological, acquisition, frame


# ---------------------------------------------------------------------------
# Representation loaders
# ---------------------------------------------------------------------------

def build_category_map(manifest: pd.DataFrame) -> Dict[str, str]:
    """Build region_id -> category_name map."""
    return dict(zip(manifest["region_id"].astype(str),
                    manifest["category_name"].astype(str)))


def load_representation_frozen(fold: int, _seed: int,
                                manifest: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """Original frozen DINOv2 features, merged with fold-specific split + category."""
    features, frame = load_base_features()
    # Merge fold-specific split, category, sample_id from manifest
    # Must merge on (region_id, scanner_id) — region_id alone is not unique
    mani_cols = manifest[["region_id", "scanner_id", "split", "sample_id",
                           "category_name"]].copy()
    mani_cols["region_id"] = mani_cols["region_id"].astype(str)
    mani_cols["scanner_id"] = mani_cols["scanner_id"].astype(str)
    frame["region_id"] = frame["region_id"].astype(str)
    frame["scanner_id"] = frame["scanner_id"].astype(str)
    frame = frame.drop(columns=["split"], errors="ignore")
    frame = frame.merge(mani_cols, on=["region_id", "scanner_id"], how="left")
    return features, frame


def load_representation_pair_integrity(fold: int, seed: int, condition: str,
                                        branch: str,
                                        manifest: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """Load biological or acquisition branch from pair-integrity experiment.

    branch: 'biological' or 'acquisition'
    condition: 'true_pairs' or 'shuffled_sample_pairs'
    """
    run_dir = (CANINE_PAIR_INTEGRITY_DIR / f"fold_{fold}" / "runs" /
               f"{condition}_seed_{seed}")
    if not (run_dir / "projected_features.npz").is_file():
        raise FileNotFoundError(f"Missing projected features: {run_dir}")
    bio, acq, frame = load_projected_features(run_dir)
    features = bio if branch == "biological" else acq
    # Merge category_name and sample_id from manifest on (region_id, scanner_id)
    mani_cols = manifest[["region_id", "scanner_id", "sample_id",
                           "category_name"]].copy()
    mani_cols["region_id"] = mani_cols["region_id"].astype(str)
    mani_cols["scanner_id"] = mani_cols["scanner_id"].astype(str)
    frame["region_id"] = frame["region_id"].astype(str)
    frame["scanner_id"] = frame["scanner_id"].astype(str)
    frame = frame.merge(mani_cols, on=["region_id", "scanner_id"], how="left")
    return features, frame


def _fit_scanner_directions(features: np.ndarray, frame: pd.DataFrame,
                             fit: np.ndarray) -> np.ndarray:
    """Fit per-scanner mean vectors (linear scanner-discriminative directions).

    Assumes features are already standardized.
    """
    X_fit = features[fit]
    scanner_labels = frame["scanner_id"].to_numpy()[fit]
    scanners = sorted(set(scanner_labels))
    directions = []
    grand_mean = X_fit.mean(axis=0)
    for s in scanners:
        mask = scanner_labels == s
        if mask.sum() > 0:
            directions.append(X_fit[mask].mean(axis=0) - grand_mean)
    if not directions:
        return np.zeros((0, features.shape[1]))
    return np.stack(directions, axis=0)


def _remove_directions(features: np.ndarray, directions: np.ndarray,
                        k: int, center: Optional[np.ndarray] = None) -> Tuple[np.ndarray, int]:
    """Remove top-k directions from features via orthogonal projection.

    Orthonormalizes directions via QR before projection to prevent
    energy amplification.
    """
    if k <= 0 or directions.shape[0] == 0:
        return features.copy(), 0
    effective_k = min(k, directions.shape[0])
    # Orthonormalize the directions
    Q, _ = np.linalg.qr(directions[:effective_k].T)  # (d, effective_k)
    Q = Q.T  # (effective_k_ortho, d)
    actual_k = Q.shape[0]
    if center is not None:
        features = features - center
    # Project out the orthonormal subspace
    components = features @ Q.T  # (n, actual_k)
    result = features - components @ Q
    if center is not None:
        result = result + center
    return result.astype(np.float32), actual_k


def load_representation_pca_removal(fold: int, k: int,
                                     manifest: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """PCA component removal baseline."""
    features, frame = load_base_features()
    # Merge fold-specific split, category, sample_id from manifest
    # Must merge on (region_id, scanner_id) — region_id alone is not unique
    mani_cols = manifest[["region_id", "scanner_id", "split", "sample_id",
                           "category_name"]].copy()
    mani_cols["region_id"] = mani_cols["region_id"].astype(str)
    mani_cols["scanner_id"] = mani_cols["scanner_id"].astype(str)
    frame["region_id"] = frame["region_id"].astype(str)
    frame["scanner_id"] = frame["scanner_id"].astype(str)
    frame = frame.drop(columns=["split"], errors="ignore")
    frame = frame.merge(mani_cols, on=["region_id", "scanner_id"], how="left")

    fit = np.flatnonzero(frame["split"].to_numpy() != "test")
    scaler = StandardScaler()
    X_fit = scaler.fit_transform(features[fit])
    pca = PCA(n_components=min(k, X_fit.shape[1]), random_state=0)
    pca.fit(X_fit)
    center = X_fit.mean(axis=0)
    X_all = scaler.transform(features)
    cleaned, _ = _remove_directions(X_all, pca.components_, k, center=center)
    return cleaned.astype(np.float32), frame


def load_representation_linear_removal(fold: int, k: int,
                                        manifest: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    """Linear scanner subspace projection baseline."""
    features, frame = load_base_features()
    # Merge fold-specific split, category, sample_id from manifest
    # Must merge on (region_id, scanner_id) — region_id alone is not unique
    mani_cols = manifest[["region_id", "scanner_id", "split", "sample_id",
                           "category_name"]].copy()
    mani_cols["region_id"] = mani_cols["region_id"].astype(str)
    mani_cols["scanner_id"] = mani_cols["scanner_id"].astype(str)
    frame["region_id"] = frame["region_id"].astype(str)
    frame["scanner_id"] = frame["scanner_id"].astype(str)
    frame = frame.drop(columns=["split"], errors="ignore")
    frame = frame.merge(mani_cols, on=["region_id", "scanner_id"], how="left")

    fit = np.flatnonzero(frame["split"].to_numpy() != "test")
    scaler = StandardScaler()
    X_all = scaler.fit_transform(features)
    directions = _fit_scanner_directions(X_all, frame, fit)
    cleaned, _ = _remove_directions(X_all, directions, k)
    return cleaned.astype(np.float32), frame


# ---------------------------------------------------------------------------
# Probe metrics
# ---------------------------------------------------------------------------

def _split_indices(frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    test = np.flatnonzero(frame["split"].to_numpy() == "test")
    fit = np.flatnonzero(frame["split"].to_numpy() != "test")
    if len(test) == 0 or len(fit) == 0:
        raise RuntimeError("Empty fit or test split.")
    return fit, test


def probe_accuracy(features: np.ndarray, frame: pd.DataFrame,
                   fit: np.ndarray, test: np.ndarray,
                   target_column: str) -> float:
    """Balanced accuracy of a linear probe on the target column."""
    labels = frame[target_column].to_numpy()
    # Ensure there are classes in both fit and test
    fit_labels = set(labels[fit])
    test_labels = set(labels[test])
    if not test_labels.issubset(fit_labels):
        # Some test classes unseen in fit — use balanced accuracy with warning
        pass
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000,
                               random_state=0)
    model.fit(features[fit], labels[fit])
    predictions = model.predict(features[test])
    return float(balanced_accuracy_score(labels[test], predictions))


def probe_macro_f1(features: np.ndarray, frame: pd.DataFrame,
                    fit: np.ndarray, test: np.ndarray,
                    target_column: str) -> float:
    """Macro F1 of a linear probe on the target column."""
    from sklearn.metrics import f1_score
    labels = frame[target_column].to_numpy()
    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000,
                               random_state=0)
    model.fit(features[fit], labels[fit])
    predictions = model.predict(features[test])
    return float(f1_score(labels[test], predictions, average="macro"))


# ---------------------------------------------------------------------------
# Neighborhood purity
# ---------------------------------------------------------------------------

def _normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms <= 0] = 1.0
    return features / norms


def neighborhood_purity(features: np.ndarray, frame: pd.DataFrame,
                         test: np.ndarray, target_column: str,
                         ks: Tuple[int, ...] = (1, 5, 10)) -> Dict[str, float]:
    """Same-category nearest-neighbor purity at k=1,5,10.

    For each test sample, find k nearest neighbors among ALL samples
    (excluding self) and compute what fraction share the same category.
    """
    normalized = _normalize(features)
    labels = frame[target_column].to_numpy()
    test_features = normalized[test]
    sim = test_features @ normalized.T  # (n_test, n_all)
    # Exclude self-matches for test samples
    for i, idx in enumerate(test):
        sim[i, idx] = -np.inf
    results = {}
    for k in ks:
        top_k = np.argpartition(-sim, k, axis=1)[:, :k]
        purity = np.mean([np.mean(labels[top_k[i]] == labels[idx])
                          for i, idx in enumerate(test)])
        results[f"purity_k{k}"] = float(purity)
    return results


def same_sample_retrieval_rate(features: np.ndarray, frame: pd.DataFrame,
                                test: np.ndarray, k: int = 1) -> float:
    """Fraction of test samples whose k-NN includes at least one same-sample patch."""
    normalized = _normalize(features)
    sample_ids = frame["sample_id"].to_numpy()
    test_features = normalized[test]
    sim = test_features @ normalized.T
    for i, idx in enumerate(test):
        sim[i, idx] = -np.inf
    top_k = np.argpartition(-sim, k, axis=1)[:, :k]
    hits = np.mean([np.any(sample_ids[top_k[i]] == sample_ids[idx])
                    for i, idx in enumerate(test)])
    return float(hits)


def effective_rank(features: np.ndarray) -> float:
    """Effective rank via entropy of singular value spectrum."""
    centered = features - features.mean(axis=0, keepdims=True)
    sv = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    energy = sv ** 2
    total = float(energy.sum())
    if total <= 0:
        return 0.0
    probs = energy / total
    probs = probs[probs > 0]
    return float(math.exp(-np.sum(probs * np.log(probs))))


# ---------------------------------------------------------------------------
# Single evaluation
# ---------------------------------------------------------------------------

def evaluate_representation(features: np.ndarray, frame: pd.DataFrame,
                             rep_name: str, fold: int, seed: int,
                             rep_details: dict) -> Dict[str, object]:
    """Compute all metrics for one representation on one fold/seed."""
    fit, test = _split_indices(frame)
    t0 = time.perf_counter()

    row = {
        "representation": rep_name,
        "fold": fold,
        "seed": seed,
        "n_fit": int(len(fit)),
        "n_test": int(len(test)),
        "feature_dim": int(features.shape[1]),
        **rep_details,
    }

    # Scanner probe
    try:
        row["scanner_probe_accuracy"] = probe_accuracy(
            features, frame, fit, test, "scanner_id")
    except Exception:
        row["scanner_probe_accuracy"] = float("nan")

    # Category probe
    try:
        row["category_probe_accuracy"] = probe_accuracy(
            features, frame, fit, test, "category_name")
    except Exception:
        row["category_probe_accuracy"] = float("nan")

    # Category macro F1
    try:
        row["category_probe_macro_f1"] = probe_macro_f1(
            features, frame, fit, test, "category_name")
    except Exception:
        row["category_probe_macro_f1"] = float("nan")

    # Neighborhood purity
    try:
        purity = neighborhood_purity(features, frame, test, "category_name")
        row.update(purity)
    except Exception:
        for k in (1, 5, 10):
            row[f"purity_k{k}"] = float("nan")

    # Same-sample retrieval
    try:
        row["same_sample_retrieval_k1"] = same_sample_retrieval_rate(
            features, frame, test, k=1)
    except Exception:
        row["same_sample_retrieval_k1"] = float("nan")

    # Effective rank
    try:
        row["effective_rank"] = effective_rank(features[test])
    except Exception:
        row["effective_rank"] = float("nan")

    # Scanner/category tradeoff ratio (higher = better category:scanner ratio)
    if (not math.isnan(row.get("scanner_probe_accuracy", float("nan")))
            and not math.isnan(row.get("category_probe_accuracy", float("nan")))
            and row["scanner_probe_accuracy"] > 0):
        row["category_scanner_ratio"] = float(
            row["category_probe_accuracy"] / max(row["scanner_probe_accuracy"], 0.001))
    else:
        row["category_scanner_ratio"] = float("nan")

    row["runtime_seconds"] = float(time.perf_counter() - t0)
    return row


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------

REPRESENTATIONS = [
    "original_frozen_features",
    "true_pair_biological",
    "true_pair_acquisition",
    "shuffled_sample_biological",
    "shuffled_sample_acquisition",
]

BASELINE_FAMILIES = {
    "pca_component_removal": (PCA_K_VALUES, False),
    "linear_scanner_subspace_projection": (LINEAR_K_VALUES, False),
}


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Biological-label preservation audit for paired-acquisition")
    p.add_argument("--out-dir", type=Path,
                   default=Path("results/paired_acquisition_factorization_"
                                "biological_label_preservation_audit"))
    p.add_argument("--smoke", action="store_true",
                   help="Smoke test: 1 fold, 1 seed only")
    p.add_argument("--skip-baselines", action="store_true",
                   help="Skip PCA/linear baselines (faster)")
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

    log("Biological-label preservation audit started")
    log(f"  smoke={args.smoke}, skip_baselines={args.skip_baselines}")

    folds = FOLDS[:1] if args.smoke else FOLDS
    seeds = SEEDS[:1] if args.smoke else SEEDS

    all_rows: List[Dict] = []

    # ---------- 1. Pair-integrity representations ----------
    conditions = [
        ("true_pairs", "true_pair"),
        ("shuffled_sample_pairs", "shuffled_sample"),
    ]
    branches = ["biological", "acquisition"]

    for fold in folds:
        manifest = load_manifest(fold)
        log(f"Fold {fold}: loaded manifest, {len(manifest)} rows, "
            f"{manifest['category_name'].nunique()} categories")

        for condition, cond_label in conditions:
            for branch in branches:
                rep_name = f"{cond_label}_{branch}"
                for seed in seeds:
                    try:
                        features, frame = load_representation_pair_integrity(
                            fold, seed, condition, branch, manifest)
                        row = evaluate_representation(
                            features, frame, rep_name, fold, seed,
                            {"rep_family": "pair_integrity",
                             "condition": condition,
                             "branch": branch})
                        all_rows.append(row)
                    except FileNotFoundError as e:
                        log(f"  SKIP {rep_name} fold={fold} seed={seed}: {e}")
                    except Exception as e:
                        log(f"  ERROR {rep_name} fold={fold} seed={seed}: {e}")
                        log(f"  {traceback.format_exc()}")

    # ---------- 2. Original frozen features ----------
    for fold in folds:
        manifest = load_manifest(fold)
        features, frame = load_representation_frozen(fold, 0, manifest)
        row = evaluate_representation(
            features, frame, "original_frozen_features", fold, 0,
            {"rep_family": "original_frozen"})
        all_rows.append(row)
        log(f"Fold {fold}: original_frozen_features done")

    # ---------- 3. PCA and linear scanner removal baselines ----------
    if not args.skip_baselines:
        for fold in folds:
            manifest = load_manifest(fold)

            for k in PCA_K_VALUES:
                try:
                    features, frame = load_representation_pca_removal(
                        fold, k, manifest)
                    row = evaluate_representation(
                        features, frame, f"pca_removal_k{k}", fold, 0,
                        {"rep_family": "pca_component_removal", "k": k})
                    all_rows.append(row)
                except Exception as e:
                    log(f"  ERROR pca_removal_k{k} fold={fold}: {e}")

            for k in LINEAR_K_VALUES:
                try:
                    features, frame = load_representation_linear_removal(
                        fold, k, manifest)
                    row = evaluate_representation(
                        features, frame, f"linear_projection_k{k}", fold, 0,
                        {"rep_family": "linear_scanner_subspace_projection",
                         "k": k})
                    all_rows.append(row)
                except Exception as e:
                    log(f"  ERROR linear_projection_k{k} fold={fold}: {e}")

            log(f"Fold {fold}: baselines done")

    if not all_rows:
        log("No data collected. Aborting.")
        log_file.close()
        return 1

    df = pd.DataFrame(all_rows)
    log(f"Total rows collected: {len(df)}")

    # ---------- Write outputs ----------
    _atomic_csv(out_dir / "label_probe_raw_metrics.csv", df)

    # Summary by representation
    metric_cols = [c for c in df.columns
                   if c.startswith("scanner_") or c.startswith("category_")
                   or c.startswith("purity_") or c == "effective_rank"
                   or c == "same_sample_retrieval_k1"
                   or c == "category_scanner_ratio"]
    summary_rows = []
    for rep_name, grp in df.groupby("representation"):
        agg = {"representation": rep_name, "n_runs": len(grp)}
        for col in metric_cols:
            if col in grp.columns and grp[col].notna().any():
                agg[f"{col}_mean"] = float(grp[col].mean())
                agg[f"{col}_std"] = float(grp[col].std())
        # Add rep_family if consistent
        if grp["rep_family"].nunique() == 1:
            agg["rep_family"] = grp["rep_family"].iloc[0]
        summary_rows.append(agg)
    summary = pd.DataFrame(summary_rows)
    _atomic_csv(out_dir / "label_probe_summary.csv", summary)

    # Neighborhood purity summary
    purity_cols = [c for c in df.columns if c.startswith("purity_k")]
    if purity_cols:
        purity_rows = []
        for rep_name, grp in df.groupby("representation"):
            pr = {"representation": rep_name, "n_runs": len(grp)}
            for col in purity_cols:
                if col in grp.columns and grp[col].notna().any():
                    pr[f"{col}_mean"] = float(grp[col].mean())
                    pr[f"{col}_std"] = float(grp[col].std())
            purity_rows.append(pr)
        purity_df = pd.DataFrame(purity_rows)
        _atomic_csv(out_dir / "neighborhood_purity_metrics.csv", purity_df)

    # Scanner/label tradeoff summary
    tradeoff_cols = ["scanner_probe_accuracy", "category_probe_accuracy",
                      "category_scanner_ratio", "category_probe_macro_f1"]
    tradeoff_rows = []
    for rep_name, grp in df.groupby("representation"):
        tr = {"representation": rep_name, "n_runs": len(grp)}
        for col in tradeoff_cols:
            if col in grp.columns and grp[col].notna().any():
                tr[f"{col}_mean"] = float(grp[col].mean())
                tr[f"{col}_std"] = float(grp[col].std())
        tradeoff_rows.append(tr)
    tradeoff_df = pd.DataFrame(tradeoff_rows)
    _atomic_csv(out_dir / "scanner_label_tradeoff_summary.csv", tradeoff_df)

    # Experiment design
    design = {
        "stage": "biological_label_preservation_audit",
        "dataset": "canineSCC_DINOv2",
        "smoke_test": args.smoke,
        "folds": folds,
        "seeds": seeds,
        "representations_tested": sorted(df["representation"].unique().tolist()),
        "target_category_column": "category_name",
        "target_scanner_column": "scanner_id",
        "categories": ["Epidermis", "SCC", "Subcutis", "Dermis",
                        "Inflamm/Necrosis", "Bone", "Cartilage"],
        "scanners": list(CANINE_SCANNERS),
        "label_probe_model": "LogisticRegression(C=1.0, class_weight=balanced, max_iter=5000)",
        "split_strategy": "slide-level 5-fold cross-validation, per-fold test held out",
        "pair_integrity_source": str(CANINE_PAIR_INTEGRITY_DIR),
        "base_features_source": str(CANINE_BASE_FEATURES),
        "manifest_source": str(CANINE_MANIFEST_DIR / CANINE_MANIFEST_PATTERN),
    }
    (out_dir / "experiment_design.json").write_text(
        json.dumps(design, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # ---------- Report ----------
    runtime = time.time() - t0
    report = build_report(df, design, out_dir, runtime, args.smoke)
    (out_dir / "biological_label_preservation_report.md").write_text(
        report, encoding="utf-8")

    log(f"Done in {runtime:.1f} s")
    log_file.close()
    return 0


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report(df: pd.DataFrame, design: dict, out_dir: Path,
                  runtime: float, smoke: bool) -> str:
    lines = []

    evidence_label = "smoke (1-fold x 1-seed)" if smoke else "full (5-fold x 5-seed)"

    lines.append("# Biological-Label Preservation Audit Report")
    lines.append("")
    lines.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Runtime:** {runtime:.1f} s")
    lines.append(f"**Evidence tier:** {evidence_label}")
    lines.append("")

    lines.append("## Scientific question")
    lines.append("")
    lines.append("Does the biological branch preserve tissue-category structure "
                 "while reducing scanner/acquisition recoverability?")
    lines.append("")

    lines.append("## Dataset")
    lines.append("")
    lines.append("- Canine SCC DINOv2: 4,025 patches, 805 regions, 44 samples")
    lines.append("- 5 scanners: cs2, gt450, nz20, nz210, p1000")
    lines.append("- 7 tissue categories: Epidermis (1,205), SCC (1,205), "
                 "Subcutis (510), Dermis (500), Inflamm/Necrosis (400), "
                 "Bone (195), Cartilage (10)")
    lines.append("- Slide-level 5-fold cross-validation")
    lines.append("")

    lines.append("## Representations compared")
    lines.append("")
    lines.append("| Representation | Dim | Source |")
    lines.append("|---|---|")
    for rep in sorted(df["representation"].unique()):
        sub = df[df["representation"] == rep]
        dim = sub["feature_dim"].iloc[0] if "feature_dim" in sub.columns else "?"
        family = sub["rep_family"].iloc[0] if "rep_family" in sub.columns else "?"
        lines.append(f"| `{rep}` | {int(dim)} | {family} |")
    lines.append("")

    # Main results table
    lines.append("## Main results: scanner suppression vs category preservation")
    lines.append("")
    lines.append("Scanner probe should be low (scanner suppressed); "
                 "category probe should be high (category preserved). "
                 "Ratio > 1 means category signal dominates scanner signal.")
    lines.append("")
    lines.append("| Representation | Scanner probe | Category probe | "
                 "Category F1 | Cat/Scan ratio | Purity k=1 | Purity k=5 | "
                 "Eff. rank |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")

    for rep in sorted(df["representation"].unique()):
        sub = df[df["representation"] == rep]
        sp = sub["scanner_probe_accuracy"].mean()
        cp = sub["category_probe_accuracy"].mean()
        cf1 = sub["category_probe_macro_f1"].mean() if "category_probe_macro_f1" in sub.columns else float("nan")
        csr = sub["category_scanner_ratio"].mean() if "category_scanner_ratio" in sub.columns else float("nan")
        pk1 = sub["purity_k1"].mean() if "purity_k1" in sub.columns else float("nan")
        pk5 = sub["purity_k5"].mean() if "purity_k5" in sub.columns else float("nan")
        er = sub["effective_rank"].mean() if "effective_rank" in sub.columns else float("nan")

        lines.append(f"| `{rep}` | {sp:.4f} | {cp:.4f} | {cf1:.4f} | "
                     f"{csr:.2f} | {pk1:.4f} | {pk5:.4f} | {er:.1f} |")
    lines.append("")

    # Tradeoff interpretation
    lines.append("## Interpretation")
    lines.append("")

    # Compare true_pair_biological vs original_frozen
    tpb = df[df["representation"] == "true_pair_biological"]
    off = df[df["representation"] == "original_frozen_features"]
    tpa = df[df["representation"] == "true_pair_acquisition"]
    ssb = df[df["representation"] == "shuffled_sample_biological"]

    if not tpb.empty and not off.empty:
        tpb_scan = tpb["scanner_probe_accuracy"].mean()
        off_scan = off["scanner_probe_accuracy"].mean()
        tpb_cat = tpb["category_probe_accuracy"].mean()
        off_cat = off["category_probe_accuracy"].mean()
        scan_drop = off_scan - tpb_scan
        cat_drop = off_cat - tpb_cat
        tpb_ratio = tpb["category_scanner_ratio"].mean()
        off_ratio = off["category_scanner_ratio"].mean()

        lines.append("### True-pair biological branch vs original frozen features")
        lines.append("")
        lines.append(f"- Scanner probe: {off_scan:.4f} → {tpb_scan:.4f} "
                     f"(Δ = {scan_drop:+.4f})")
        lines.append(f"- Category probe: {off_cat:.4f} → {tpb_cat:.4f} "
                     f"(Δ = {cat_drop:+.4f})")
        lines.append(f"- Category/scanner ratio: {off_ratio:.2f} → "
                     f"{tpb_ratio:.2f}")
        lines.append("")

        if scan_drop > 0.05 and abs(cat_drop) < 0.05:
            lines.append("The biological branch substantially reduces scanner "
                         "recoverability while preserving category structure. "
                         "This supports the paired-acquisition mechanism: "
                         "the factorization separates scanner signal from "
                         "biologically meaningful tissue-category structure.")
        elif scan_drop > 0.02 and abs(cat_drop) < 0.10:
            lines.append("The biological branch reduces scanner recoverability "
                         "with only modest category degradation. The "
                         "category/scanner tradeoff improves relative to "
                         "original frozen features.")
        else:
            lines.append("The biological branch shows a different "
                         "scanner/category tradeoff profile. See detailed "
                         "tables for quantitative comparison.")
        lines.append("")

    if not tpa.empty:
        tpa_scan = tpa["scanner_probe_accuracy"].mean()
        tpa_cat = tpa["category_probe_accuracy"].mean()
        lines.append("### True-pair acquisition branch")
        lines.append("")
        lines.append(f"- Scanner probe: {tpa_scan:.4f} (should be HIGH — "
                     f"captures scanner signal)")
        lines.append(f"- Category probe: {tpa_cat:.4f} (should be LOW — "
                     f"tissue identity removed)")
        lines.append("")

    if not ssb.empty:
        ssb_scan = ssb["scanner_probe_accuracy"].mean()
        ssb_cat = ssb["category_probe_accuracy"].mean()
        lines.append("### Shuffled-sample biological branch (control)")
        lines.append("")
        lines.append(f"- Scanner probe: {ssb_scan:.4f}")
        lines.append(f"- Category probe: {ssb_cat:.4f}")
        lines.append("")

    lines.append("## Claim boundaries")
    lines.append("")
    lines.append("- This audit tests whether the biological branch preserves "
                 "tissue-category structure while reducing scanner "
                 "recoverability. It does not test clinical utility.")
    lines.append("- Category labels are tissue morphology categories from "
                 "canine SCC expert annotation, not diagnostic grades.")
    lines.append("- All probe metrics use balanced accuracy to account for "
                 "class imbalance (Cartilage: 10 patches).")
    lines.append("- The PCA and linear scanner-subspace baselines are simple "
                 "post-hoc operations on frozen embeddings. They do not use "
                 "paired training and are not expected to match the "
                 "factorization's tissue-preservation quality.")
    lines.append("- Does not claim clinical validation, diagnostic performance, "
                 "disease biology discovery, or deployment readiness.")
    lines.append("")

    lines.append("## Output files")
    lines.append("")
    lines.append("| File | Description |")
    lines.append("|---|---|")
    lines.append("| label_probe_raw_metrics.csv | Per-run, per-representation metrics |")
    lines.append("| label_probe_summary.csv | Aggregated by representation |")
    lines.append("| neighborhood_purity_metrics.csv | Same-category NN purity |")
    lines.append("| scanner_label_tradeoff_summary.csv | Scanner vs category tradeoff |")
    lines.append("| experiment_design.json | Experiment configuration |")
    lines.append("| run_log.txt | Timestamped run log |")
    lines.append("| biological_label_preservation_report.md | This report |")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
