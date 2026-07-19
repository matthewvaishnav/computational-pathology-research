#!/usr/bin/env python3
"""Sample-subset-disjoint scanner-heldout label transfer audit.

Scientific question:
  Does the biological branch preserve tissue-category structure under BOTH
  scanner shift AND sample shift?

Protocol:
  For each held-out scanner X:
    - Split sample_ids into disjoint train_sample_ids and test_sample_ids
    - Train category probe on (scanner != X, sample in train_sample_ids)
    - Test on (scanner == X, sample in test_sample_ids)
    - No sample appears in both train and test
    - No held-out scanner patches appear in training

This directly addresses the objection: "did the scanner-heldout transfer
benefit from seeing the same samples under other scanners?"
"""

from __future__ import annotations

import argparse
import json
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
from sklearn.metrics import (balanced_accuracy_score, f1_score,
                              classification_report)
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

CANINE_MANIFEST_DIR = Path(
    "data/external_multiscanner_caninescc/patch_manifests/splits")
CANINE_MANIFEST_PATTERN = "fold_{fold}_patch_manifest.csv"
CANINE_BASE_FEATURES = Path(
    "results/external_multiscanner_caninescc/features/fold_0_dinov2_base.npz")
CANINE_PAIR_INTEGRITY_DIR = Path(
    "results/paired_acquisition_factorization_pair_integrity_caninescc")
CANINE_SCANNERS = ("cs2", "gt450", "nz20", "nz210", "p1000")

FOLD = 0
SCANNERS = list(CANINE_SCANNERS)
SEEDS = [911, 912, 913, 914, 915]
TRAIN_FRAC = 0.70

PCA_K_VALUES = [1, 2, 4, 8, 16, 32]
LINEAR_K_VALUES = [0, 1, 2, 4, 8, 16, 32]


# ---------------------------------------------------------------------------
# Atomic write
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
# Data loading (self-contained)
# ---------------------------------------------------------------------------

def load_manifest(fold: int) -> pd.DataFrame:
    p = CANINE_MANIFEST_DIR / CANINE_MANIFEST_PATTERN.format(fold=fold)
    df = pd.read_csv(p, dtype=str)
    required = {"region_id", "scanner_id", "slide_id", "sample_id",
                 "category_name", "split"}
    missing = sorted(required - set(df.columns))
    if missing:
        raise RuntimeError(f"Manifest fold {fold} missing: {missing}")
    return df


def load_base_features() -> Tuple[np.ndarray, pd.DataFrame]:
    with np.load(CANINE_BASE_FEATURES, allow_pickle=False) as archive:
        features = np.asarray(archive["features"], dtype=np.float32)
        frame = pd.DataFrame({
            name: archive[name].astype(str)
            for name in ("region_id", "scanner_id", "slide_id", "split")
        })
    return features, frame


def load_projected_features(
        run_dir: Path) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    with np.load(run_dir / "projected_features.npz", allow_pickle=False) as a:
        biological = np.asarray(a["features"], dtype=np.float32)
        acquisition = np.asarray(a["acquisition_features"], dtype=np.float32)
        frame = pd.DataFrame({
            name: a[name].astype(str)
            for name in ("region_id", "scanner_id", "slide_id", "split")
        })
    return biological, acquisition, frame


def _merge_manifest(frame: pd.DataFrame,
                     manifest: pd.DataFrame) -> pd.DataFrame:
    mani_cols = manifest[["region_id", "scanner_id", "split", "sample_id",
                           "category_name"]].copy()
    mani_cols["region_id"] = mani_cols["region_id"].astype(str)
    mani_cols["scanner_id"] = mani_cols["scanner_id"].astype(str)
    frame["region_id"] = frame["region_id"].astype(str)
    frame["scanner_id"] = frame["scanner_id"].astype(str)
    frame = frame.drop(columns=["split"], errors="ignore")
    frame = frame.merge(mani_cols, on=["region_id", "scanner_id"], how="left")
    return frame


def load_representation_frozen(
        fold: int, _seed: int, manifest: pd.DataFrame
) -> Tuple[np.ndarray, pd.DataFrame]:
    features, frame = load_base_features()
    frame = _merge_manifest(frame, manifest)
    return features, frame


def load_representation_pair_integrity(
        fold: int, seed: int, condition: str, branch: str,
        manifest: pd.DataFrame
) -> Tuple[np.ndarray, pd.DataFrame]:
    run_dir = (CANINE_PAIR_INTEGRITY_DIR / f"fold_{fold}" / "runs" /
               f"{condition}_seed_{seed}")
    if not (run_dir / "projected_features.npz").is_file():
        raise FileNotFoundError(f"Missing: {run_dir}")
    bio, acq, frame = load_projected_features(run_dir)
    features = bio if branch == "biological" else acq
    frame = _merge_manifest(frame, manifest)
    return features, frame


def _remove_directions(features: np.ndarray, directions: np.ndarray,
                        k: int, center: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, int]:
    if k <= 0 or directions.shape[0] == 0:
        return features.copy(), 0
    effective_k = min(k, directions.shape[0])
    Q, _ = np.linalg.qr(directions[:effective_k].T)
    Q = Q.T
    actual_k = Q.shape[0]
    if center is not None:
        features = features - center
    components = features @ Q.T
    result = features - components @ Q
    if center is not None:
        result = result + center
    return result.astype(np.float32), actual_k


def load_representation_pca_removal(
        fold: int, k: int, manifest: pd.DataFrame
) -> Tuple[np.ndarray, pd.DataFrame]:
    features, frame = load_base_features()
    frame = _merge_manifest(frame, manifest)
    fit = np.flatnonzero(frame["split"].to_numpy() != "test")
    scaler = StandardScaler()
    X_fit = scaler.fit_transform(features[fit])
    pca = PCA(n_components=min(k, X_fit.shape[1]), random_state=0)
    pca.fit(X_fit)
    center = X_fit.mean(axis=0)
    X_all = scaler.transform(features)
    cleaned, _ = _remove_directions(X_all, pca.components_, k, center=center)
    return cleaned.astype(np.float32), frame


def _fit_scanner_directions(features: np.ndarray, frame: pd.DataFrame,
                             fit: np.ndarray) -> np.ndarray:
    X_fit = features[fit]
    scanner_labels = frame["scanner_id"].to_numpy()[fit]
    scanners_list = sorted(set(scanner_labels))
    directions = []
    grand_mean = X_fit.mean(axis=0)
    for s in scanners_list:
        mask = scanner_labels == s
        if mask.sum() > 0:
            directions.append(X_fit[mask].mean(axis=0) - grand_mean)
    if not directions:
        return np.zeros((0, features.shape[1]))
    return np.stack(directions, axis=0)


def load_representation_linear_removal(
        fold: int, k: int, manifest: pd.DataFrame
) -> Tuple[np.ndarray, pd.DataFrame]:
    features, frame = load_base_features()
    frame = _merge_manifest(frame, manifest)
    fit = np.flatnonzero(frame["split"].to_numpy() != "test")
    scaler = StandardScaler()
    X_all = scaler.fit_transform(features)
    directions = _fit_scanner_directions(X_all, frame, fit)
    cleaned, _ = _remove_directions(X_all, directions, k)
    return cleaned.astype(np.float32), frame


# ---------------------------------------------------------------------------
# Sample-disjoint split
# ---------------------------------------------------------------------------

def sample_disjoint_split(sample_ids: List[str], seed: int,
                           train_frac: float = TRAIN_FRAC
) -> Tuple[List[str], List[str]]:
    """Split sample IDs into disjoint train/test sets."""
    rng = np.random.default_rng(seed)
    unique_samples = sorted(set(sample_ids))
    n_train = max(1, int(len(unique_samples) * train_frac))
    shuffled = rng.permutation(unique_samples)
    train_samples = sorted(shuffled[:n_train].tolist())
    test_samples = sorted(shuffled[n_train:].tolist())
    return train_samples, test_samples


def build_disjoint_indices(frame: pd.DataFrame, held_out_scanner: str,
                            train_samples: List[str],
                            test_samples: List[str]
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Build train/test indices satisfying both constraints.

    Train: scanner != held_out_scanner AND sample_id in train_samples
    Test:  scanner == held_out_scanner AND sample_id in test_samples
    """
    scanners = frame["scanner_id"].to_numpy().astype(str)
    samples = frame["sample_id"].to_numpy().astype(str)

    train_mask = (scanners != held_out_scanner) & np.isin(samples, train_samples)
    test_mask = (scanners == held_out_scanner) & np.isin(samples, test_samples)

    train = np.flatnonzero(train_mask)
    test = np.flatnonzero(test_mask)

    # Diagnostic info
    train_sample_set = set(samples[train])
    test_sample_set = set(samples[test])
    overlap = train_sample_set & test_sample_set
    train_has_heldout = (scanners[train] == held_out_scanner).any()

    diag = {
        "n_train_patches": int(len(train)),
        "n_test_patches": int(len(test)),
        "n_train_samples": int(len(train_sample_set)),
        "n_test_samples": int(len(test_sample_set)),
        "sample_overlap": int(len(overlap)),
        "train_has_heldout_scanner": bool(train_has_heldout),
        "train_samples": json.dumps(sorted(train_sample_set)),
        "test_samples": json.dumps(sorted(test_sample_set)),
    }
    return train, test, diag


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_disjoint(features: np.ndarray, frame: pd.DataFrame,
                       held_out_scanner: str, seed: int,
                       rep_name: str, rep_details: dict) -> Dict[str, object]:
    """Train on 4 scanners + train samples, test on held-out + test samples."""
    # Generate sample split
    all_samples = sorted(set(frame["sample_id"].to_numpy().astype(str)))
    train_samples, test_samples = sample_disjoint_split(all_samples, seed)

    train, test, diag = build_disjoint_indices(
        frame, held_out_scanner, train_samples, test_samples)

    if len(train) == 0 or len(test) == 0:
        return {
            "representation": rep_name,
            "held_out_scanner": held_out_scanner,
            "seed": seed,
            "n_train": 0, "n_test": 0,
            "category_balanced_accuracy": float("nan"),
            "category_macro_f1": float("nan"),
            "category_weighted_f1": float("nan"),
            "error": "empty_split",
            **diag, **rep_details,
        }

    t0 = time.perf_counter()
    y_train = frame["category_name"].to_numpy()[train]
    y_test = frame["category_name"].to_numpy()[test]

    train_class_counts = {str(k): int(v) for k, v in
                          zip(*np.unique(y_train, return_counts=True))}
    test_class_counts = {str(k): int(v) for k, v in
                         zip(*np.unique(y_test, return_counts=True))}

    row = {
        "representation": rep_name,
        "held_out_scanner": held_out_scanner,
        "seed": seed,
        "n_train": int(len(train)),
        "n_test": int(len(test)),
        "feature_dim": int(features.shape[1]),
        "train_class_counts": json.dumps(train_class_counts),
        "test_class_counts": json.dumps(test_class_counts),
        **diag,
        **rep_details,
    }

    scaler = StandardScaler()
    X_train = scaler.fit_transform(features[train])
    X_test = scaler.transform(features[test])

    model = LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000,
                               random_state=seed)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    classes_missing = sorted(str(c) for c in set(y_train) - set(y_test))

    row["category_balanced_accuracy"] = float(
        balanced_accuracy_score(y_test, y_pred))
    row["category_macro_f1"] = float(
        f1_score(y_test, y_pred, average="macro", zero_division=0))
    row["category_weighted_f1"] = float(
        f1_score(y_test, y_pred, average="weighted", zero_division=0))
    row["n_classes_test"] = int(len(set(y_test)))
    row["n_classes_train"] = int(len(set(y_train)))
    row["classes_missing_from_test"] = json.dumps(classes_missing)
    row["runtime_seconds"] = float(time.perf_counter() - t0)

    report = classification_report(
        y_test, y_pred, output_dict=True, zero_division=0)
    for cls_name, cls_metrics in report.items():
        if cls_name not in ("accuracy", "macro avg", "weighted avg"):
            row[f"recall_{cls_name}"] = float(cls_metrics["recall"])
            row[f"precision_{cls_name}"] = float(cls_metrics["precision"])
            row[f"f1_{cls_name}"] = float(cls_metrics["f1-score"])
            row[f"support_{cls_name}"] = int(cls_metrics["support"])

    return row


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

REPRESENTATIONS = [
    ("original_frozen_features", "frozen", None, None),
    ("true_pair_biological", "pair_integrity", "true_pairs", "biological"),
    ("true_pair_acquisition", "pair_integrity", "true_pairs", "acquisition"),
    ("shuffled_sample_biological", "pair_integrity",
     "shuffled_sample_pairs", "biological"),
    ("shuffled_sample_acquisition", "pair_integrity",
     "shuffled_sample_pairs", "acquisition"),
]


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Sample-subset-disjoint scanner-heldout transfer audit")
    p.add_argument("--out-dir", type=Path,
                   default=Path("results/paired_acquisition_factorization_"
                                "sample_subset_disjoint_scanner_heldout_"
                                "transfer_audit"))
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--skip-baselines", action="store_true")
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

    log("Sample-subset-disjoint scanner-heldout transfer audit started")
    log(f"  smoke={args.smoke}, skip_baselines={args.skip_baselines}")

    manifest = load_manifest(FOLD)
    log(f"Loaded fold-{FOLD} manifest: {len(manifest)} rows, "
        f"{manifest['category_name'].nunique()} categories, "
        f"{manifest['sample_id'].nunique()} samples")

    scanners = SCANNERS[:1] if args.smoke else SCANNERS
    seeds = SEEDS[:1] if args.smoke else SEEDS
    log(f"Held-out scanners: {scanners}")
    log(f"Seeds: {seeds}, train_frac: {TRAIN_FRAC}")

    all_rows: List[Dict] = []

    # ---------- 1. Pair-integrity and frozen representations ----------
    for rep_name, rep_type, condition, branch in REPRESENTATIONS:
        rep_details = {"rep_family": rep_type}
        if condition:
            rep_details["condition"] = condition
        if branch:
            rep_details["branch"] = branch

        for held_out in scanners:
            if rep_type == "frozen":
                features, frame = load_representation_frozen(
                    FOLD, 0, manifest)
                for seed in seeds:
                    row = evaluate_disjoint(
                        features, frame, held_out, seed, rep_name,
                        rep_details)
                    all_rows.append(row)
                    log(f"  {rep_name} held_out={held_out} seed={seed} "
                        f"acc={row.get('category_balanced_accuracy', float('nan')):.4f} "
                        f"f1={row.get('category_macro_f1', float('nan')):.4f} "
                        f"n_test={row.get('n_test', 0)}")
            elif rep_type == "pair_integrity":
                for seed in seeds:
                    try:
                        features, frame = \
                            load_representation_pair_integrity(
                                FOLD, seed, condition, branch, manifest)
                        row = evaluate_disjoint(
                            features, frame, held_out, seed, rep_name,
                            rep_details)
                        all_rows.append(row)
                    except FileNotFoundError:
                        log(f"  SKIP {rep_name} held_out={held_out} "
                            f"seed={seed}")
                    except Exception:
                        log(f"  ERROR {rep_name} held_out={held_out} "
                            f"seed={seed}: {traceback.format_exc()}")
                sub = [r for r in all_rows
                       if r["representation"] == rep_name
                       and r["held_out_scanner"] == held_out]
                if sub:
                    accs = [r.get("category_balanced_accuracy",
                                  float("nan")) for r in sub]
                    f1s = [r.get("category_macro_f1",
                                 float("nan")) for r in sub]
                    log(f"  {rep_name} held_out={held_out} "
                        f"mean_acc={np.nanmean(accs):.4f} "
                        f"mean_f1={np.nanmean(f1s):.4f}")

    # ---------- 2. Baselines ----------
    if not args.skip_baselines:
        for held_out in scanners:
            for k in PCA_K_VALUES:
                try:
                    features, frame = load_representation_pca_removal(
                        FOLD, k, manifest)
                    for seed in seeds:
                        row = evaluate_disjoint(
                            features, frame, held_out, seed,
                            f"pca_removal_k{k}",
                            {"rep_family": "pca_component_removal", "k": k})
                        all_rows.append(row)
                except Exception:
                    log(f"  ERROR pca_removal_k{k} held_out={held_out}: "
                        f"{traceback.format_exc()}")

            for k in LINEAR_K_VALUES:
                try:
                    features, frame = load_representation_linear_removal(
                        FOLD, k, manifest)
                    for seed in seeds:
                        row = evaluate_disjoint(
                            features, frame, held_out, seed,
                            f"linear_projection_k{k}",
                            {"rep_family":
                             "linear_scanner_subspace_projection", "k": k})
                        all_rows.append(row)
                except Exception:
                    log(f"  ERROR linear_projection_k{k} held_out={held_out}: "
                        f"{traceback.format_exc()}")

            log(f"  baselines done for held_out={held_out}")

    if not all_rows:
        log("No data collected. Aborting.")
        log_file.close()
        return 1

    df = pd.DataFrame(all_rows)
    log(f"Total rows: {len(df)}")

    # ---------- Validation ----------
    df_valid = df[df["error"] != "empty_split"] if "error" in df.columns else df
    log(f"Valid (non-empty) rows: {len(df_valid)}")
    if "sample_overlap" in df.columns:
        overlaps = df[df["sample_overlap"] > 0]
        if len(overlaps) > 0:
            log(f"WARNING: {len(overlaps)} rows with sample overlap")
    if "train_has_heldout_scanner" in df.columns:
        leaks = df[df["train_has_heldout_scanner"] == True]
        if len(leaks) > 0:
            log(f"WARNING: {len(leaks)} rows with held-out scanner in train")

    # ---------- Write outputs ----------
    _atomic_csv(out_dir / "sample_disjoint_scanner_heldout_raw_metrics.csv", df)

    # Split diagnostics
    diag_cols = ["representation", "held_out_scanner", "seed",
                  "n_train_patches", "n_test_patches",
                  "n_train_samples", "n_test_samples",
                  "sample_overlap", "train_has_heldout_scanner"]
    diag_cols = [c for c in diag_cols if c in df.columns]
    diag_df = df[diag_cols].copy()
    _atomic_csv(out_dir / "sample_disjoint_scanner_heldout_split_diagnostics.csv",
                diag_df)

    metric_cols = ["category_balanced_accuracy", "category_macro_f1",
                    "category_weighted_f1"]
    summary_rows = []
    for rep_name, grp in df_valid.groupby("representation"):
        agg = {"representation": rep_name, "n_runs": len(grp),
               "n_scanners": grp["held_out_scanner"].nunique()}
        for col in metric_cols:
            if col in grp.columns and grp[col].notna().any():
                agg[f"{col}_mean"] = float(grp[col].mean())
                agg[f"{col}_std"] = float(grp[col].std())
                agg[f"{col}_worst"] = float(
                    grp.groupby("held_out_scanner")[col].mean().min())
        if grp["rep_family"].nunique() == 1:
            agg["rep_family"] = grp["rep_family"].iloc[0]
        summary_rows.append(agg)
    summary = pd.DataFrame(summary_rows)
    _atomic_csv(out_dir / "sample_disjoint_scanner_heldout_summary.csv", summary)

    per_scanner_rows = []
    for (rep_name, scanner), grp in df_valid.groupby(
            ["representation", "held_out_scanner"]):
        row = {"representation": rep_name, "held_out_scanner": scanner,
               "n_runs": len(grp)}
        for col in metric_cols:
            if col in grp.columns and grp[col].notna().any():
                row[f"{col}_mean"] = float(grp[col].mean())
                row[f"{col}_std"] = float(grp[col].std())
        if "n_test" in grp.columns:
            row["mean_n_test"] = float(grp["n_test"].mean())
        if "n_test_patches" in grp.columns:
            row["mean_n_test_patches"] = float(grp["n_test_patches"].mean())
        per_scanner_rows.append(row)
    per_scanner = pd.DataFrame(per_scanner_rows)
    _atomic_csv(out_dir / "sample_disjoint_scanner_heldout_per_scanner.csv",
                per_scanner)

    recall_cols = [c for c in df_valid.columns if c.startswith("recall_")]
    class_df = None
    if recall_cols:
        class_rows = []
        for rep_name, grp in df_valid.groupby("representation"):
            row = {"representation": rep_name, "n_runs": len(grp)}
            for col in recall_cols:
                if col in grp.columns and grp[col].notna().any():
                    row[f"{col}_mean"] = float(grp[col].mean())
                    row[f"{col}_std"] = float(grp[col].std())
            class_rows.append(row)
        class_df = pd.DataFrame(class_rows)
        _atomic_csv(
            out_dir / "sample_disjoint_scanner_heldout_per_class_recall.csv",
            class_df)

    design = {
        "stage": "sample_subset_disjoint_scanner_heldout_transfer_audit",
        "dataset": "canineSCC_DINOv2",
        "smoke_test": args.smoke,
        "fold": FOLD,
        "held_out_scanners": scanners,
        "seeds": seeds,
        "train_fraction": TRAIN_FRAC,
        "representations_tested":
            sorted(df["representation"].unique().tolist()),
        "target_category_column": "category_name",
        "target_scanner_column": "scanner_id",
        "target_sample_column": "sample_id",
        "protocol": ("sample-disjoint leave-one-scanner-out: train on 4 "
                     "scanners with train samples only, test on held-out "
                     "scanner with disjoint test samples"),
        "probe_model":
            "LogisticRegression(C=1.0, class_weight=balanced, max_iter=5000)",
        "categories": ["Epidermis", "SCC", "Subcutis", "Dermis",
                        "Inflamm/Necrosis", "Bone", "Cartilage"],
        "scanners": SCANNERS,
    }
    (out_dir / "experiment_design.json").write_text(
        json.dumps(design, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    runtime = time.time() - t0
    report = build_report(df_valid, summary, per_scanner, class_df,
                           out_dir, runtime, args.smoke)
    (out_dir / "sample_disjoint_scanner_heldout_report.md").write_text(
        report, encoding="utf-8")

    log(f"Done in {runtime:.1f} s")
    log_file.close()
    return 0


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report(df: pd.DataFrame, summary: pd.DataFrame,
                  per_scanner: pd.DataFrame,
                  class_df: Optional[pd.DataFrame],
                  out_dir: Path, runtime: float, smoke: bool) -> str:
    lines = []
    tier = "smoke (1 held-out scanner, 5 seeds)" if smoke else \
           "full (5 held-out scanners, 5 seeds)"

    lines.append("# Sample-Disjoint Scanner-Heldout Transfer Audit Report")
    lines.append("")
    lines.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Runtime:** {runtime:.1f} s")
    lines.append(f"**Evidence tier:** {tier}")
    lines.append("")

    lines.append("## Scientific question")
    lines.append("")
    lines.append("Does the biological branch preserve tissue-category "
                 "structure under BOTH scanner shift AND sample shift?")
    lines.append("")

    lines.append("## Protocol")
    lines.append("")
    lines.append("Sample-disjoint leave-one-scanner-out:")
    lines.append("- Split sample IDs into disjoint train/test sets")
    lines.append("- Train: 4 scanners, train samples only")
    lines.append("- Test: held-out scanner, test samples only")
    lines.append("- No sample appears in both train and test")
    lines.append("- No held-out scanner patches appear in training")
    lines.append("- Repeated across 5 random sample-split seeds")
    lines.append(f"- Train fraction: {TRAIN_FRAC}")
    lines.append("")

    lines.append("## Dataset")
    lines.append("")
    lines.append("- Canine SCC DINOv2, fold 0 manifest")
    lines.append("- 44 samples, all present in all 5 scanners")
    lines.append("- 7 tissue categories")
    lines.append("- Class imbalance: Cartilage (10 patches), Bone (195)")
    lines.append("")

    lines.append("## Split diagnostics")
    lines.append("")
    if "n_test_patches" in df.columns and "n_train_patches" in df.columns:
        lines.append(f"- Mean train patches: "
                     f"{df['n_train_patches'].mean():.0f}")
        lines.append(f"- Mean test patches: "
                     f"{df['n_test_patches'].mean():.0f}")
    if "n_train_samples" in df.columns and "n_test_samples" in df.columns:
        lines.append(f"- Mean train samples: "
                     f"{df['n_train_samples'].mean():.0f}")
        lines.append(f"- Mean test samples: "
                     f"{df['n_test_samples'].mean():.0f}")
    if "sample_overlap" in df.columns:
        lines.append(f"- Sample overlap violations: "
                     f"{int(df['sample_overlap'].sum())}")
    if "train_has_heldout_scanner" in df.columns:
        lines.append(f"- Held-out scanner in train violations: "
                     f"{int(df['train_has_heldout_scanner'].sum())}")
    lines.append("")

    lines.append("## Key results")
    lines.append("")
    lines.append("| Representation | Mean balanced acc | Mean macro F1 | "
                 "Worst-scanner acc | Worst-scanner F1 |")
    lines.append("|---|---:|---:|---:|---:|")
    for _, row in summary.iterrows():
        lines.append(
            f"| `{row['representation']}` | "
            f"{row.get('category_balanced_accuracy_mean', float('nan')):.4f} | "
            f"{row.get('category_macro_f1_mean', float('nan')):.4f} | "
            f"{row.get('category_balanced_accuracy_worst', float('nan')):.4f} | "
            f"{row.get('category_macro_f1_worst', float('nan')):.4f} |")
    lines.append("")

    key_reps = ["original_frozen_features", "true_pair_biological",
                "true_pair_acquisition", "shuffled_sample_biological",
                "linear_projection_k4"]
    lines.append("## Per-scanner breakdown (key representations)")
    lines.append("")
    for rep in key_reps:
        sub = per_scanner[per_scanner["representation"] == rep]
        if sub.empty:
            continue
        lines.append(f"### `{rep}`")
        lines.append("")
        lines.append("| Held-out scanner | Balanced acc | Macro F1 | "
                     "Mean n_test |")
        lines.append("|---|---:|---:|---:|")
        for _, r in sub.iterrows():
            lines.append(
                f"| {r['held_out_scanner']} | "
                f"{r.get('category_balanced_accuracy_mean', float('nan')):.4f} | "
                f"{r.get('category_macro_f1_mean', float('nan')):.4f} | "
                f"{int(r.get('mean_n_test', 0))} |")
        lines.append("")

    if class_df is not None and not class_df.empty:
        lines.append("## Per-class recall (key representations)")
        lines.append("")
        recall_cols = sorted(
            [c for c in class_df.columns
             if c.startswith("recall_") and c.endswith("_mean")])
        header = "| Representation | " + " | ".join(
            c.replace("recall_", "").replace("_mean", "")
            for c in recall_cols) + " |"
        lines.append(header)
        sep = "|---|" + "|".join(["---:" for _ in recall_cols]) + "|"
        lines.append(sep)
        for _, row in class_df.iterrows():
            if row["representation"] in key_reps:
                vals = " | ".join(
                    f"{row.get(c, float('nan')):.4f}" for c in recall_cols)
                lines.append(f"| `{row['representation']}` | {vals} |")
        lines.append("")

    lines.append("## Interpretation")
    lines.append("")

    tpb = df[df["representation"] == "true_pair_biological"]
    off = df[df["representation"] == "original_frozen_features"]
    tpa = df[df["representation"] == "true_pair_acquisition"]
    ssb = df[df["representation"] == "shuffled_sample_biological"]

    if not tpb.empty and not off.empty:
        tpb_acc = tpb["category_balanced_accuracy"].mean()
        off_acc = off["category_balanced_accuracy"].mean()
        tpb_f1 = tpb["category_macro_f1"].mean()
        off_f1 = off["category_macro_f1"].mean()
        acc_delta = tpb_acc - off_acc

        lines.append("### True-pair biological vs original frozen features")
        lines.append("")
        lines.append(f"- Sample-disjoint balanced accuracy: "
                     f"{off_acc:.4f} -> {tpb_acc:.4f} "
                     f"(delta = {acc_delta:+.4f})")
        lines.append(f"- Sample-disjoint macro F1: "
                     f"{off_f1:.4f} -> {tpb_f1:.4f} "
                     f"(delta = {tpb_f1 - off_f1:+.4f})")
        lines.append("")

        if acc_delta > 0.01:
            lines.append("The biological branch improves sample-disjoint "
                         "held-out-scanner category transfer relative to "
                         "original frozen features. This directly addresses "
                         "the sample-leakage objection: the biological "
                         "branch preserves category structure under both "
                         "scanner shift and sample shift.")
        elif abs(acc_delta) <= 0.01:
            lines.append("The biological branch preserves sample-disjoint "
                         "held-out-scanner category transfer at a level "
                         "comparable to original frozen features. This "
                         "suggests the previous scanner-heldout result was "
                         "not driven by sample leakage.")
        else:
            lines.append("The biological branch shows lower sample-disjoint "
                         "held-out-scanner category transfer than original "
                         "frozen features. This may indicate that some "
                         "cross-scanner transfer in the biological branch "
                         "depends on sample-level information present in "
                         "the factorization training.")
        lines.append("")

    if not tpa.empty:
        tpa_acc = tpa["category_balanced_accuracy"].mean()
        lines.append("### True-pair acquisition branch")
        lines.append("")
        lines.append(f"- Sample-disjoint balanced accuracy: {tpa_acc:.4f}")
        lines.append("")

    if not ssb.empty:
        ssb_acc = ssb["category_balanced_accuracy"].mean()
        lines.append("### Shuffled-sample biological branch (control)")
        lines.append("")
        lines.append(f"- Sample-disjoint balanced accuracy: {ssb_acc:.4f}")
        lines.append("")

    lines.append("## Claim boundaries")
    lines.append("")
    lines.append("- This audit tests whether the biological branch preserves "
                 "category structure under both scanner shift and sample "
                 "shift. It does not test clinical utility.")
    lines.append("- Test sets are smaller than full scanner-heldout (subset of "
                 "samples), so variance is higher.")
    lines.append("- Cartilage (10 patches) is absent from many test splits. "
                 "Cartilage-specific claims should not be drawn.")
    lines.append("- Does not claim: clinical validation, diagnostic "
                 "performance, patient-care utility, universal biological "
                 "factorization, scanner bias solved, or deployment "
                 "readiness.")
    lines.append("")

    lines.append("## Output files")
    lines.append("")
    lines.append("| File | Description |")
    lines.append("|---|---|")
    lines.append("| sample_disjoint_scanner_heldout_raw_metrics.csv | "
                 "Per-run metrics |")
    lines.append("| sample_disjoint_scanner_heldout_summary.csv | "
                 "Aggregated by representation |")
    lines.append("| sample_disjoint_scanner_heldout_per_scanner.csv | "
                 "Per-scanner breakdown |")
    lines.append("| sample_disjoint_scanner_heldout_per_class_recall.csv | "
                 "Per-class recall |")
    lines.append("| sample_disjoint_scanner_heldout_split_diagnostics.csv | "
                 "Split diagnostics |")
    lines.append("| experiment_design.json | Experiment configuration |")
    lines.append("| run_log.txt | Timestamped run log |")
    lines.append("| sample_disjoint_scanner_heldout_report.md | This report |")
    lines.append("")

    return "\n".join(lines)


if __name__ == "__main__":
    raise SystemExit(main())
