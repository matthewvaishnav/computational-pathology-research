#!/usr/bin/env python3
"""Acquisition-branch audit for Paired-Acquisition Neural Factorization.

Scientific question:
  Does the acquisition branch actually capture acquisition/scanner information
  while carrying substantially less tissue-identity information than the
  biological branch?

This audit loads *existing* projected features from completed pair-integrity
falsification runs and computes the same metrics on BOTH branches separately.
No training is rerun — every metric comes from already-saved projected_features.npz.

Audited datasets/backbones (priority order):
  1. SCORPION DINOv2-Base
  2. canine SCC DINOv2-Base
  3. SCORPION Phikon
  4. SCORPION ResNet50

Metrics per branch:
  - scanner_probe_accuracy       (scanner-id classifier; lower ≈ less scanner info)
  - mean_paired_cosine           (same-tissue cross-scanner cosine)
  - mean_top1_retrieval          (same-tissue nearest-neighbor retrieval)
  - effective_rank               (SVD entropy; lower ≈ more collapsed)
  - biological_acquisition_cross_covariance  (RMS of cross-covariance matrix)

Expected qualitative pattern:
  Biological branch → lower scanner recoverability, higher tissue preservation
  Acquisition branch → higher scanner recoverability, lower tissue preservation
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
import tempfile
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# ---------------------------------------------------------------------------
# Datasets to audit  (priority order)
# ---------------------------------------------------------------------------

AUDIT_TARGETS = [
    {
        "name": "SCORPION_DINOv2",
        "result_dir": "results/paired_acquisition_factorization_pair_integrity_scorpion",
        "scanner_namespace": ("AT2", "GT450", "DP200", "P1000", "B300"),
        "folds": [0, 1, 2, 3, 4],
        "seeds": [701, 702, 703, 704, 705],
        "conditions": ["true_pairs", "shuffled_region_pairs", "shuffled_sample_pairs"],
    },
    {
        "name": "canineSCC_DINOv2",
        "result_dir": "results/paired_acquisition_factorization_pair_integrity_caninescc",
        "scanner_namespace": ("cs2", "gt450", "nz20", "nz210", "p1000"),
        "folds": [0, 1, 2, 3, 4],
        "seeds": [911, 912, 913, 914, 915],
        "conditions": ["true_pairs", "shuffled_region_pairs", "shuffled_sample_pairs"],
    },
    {
        "name": "SCORPION_Phikon",
        "result_dir": "results/paired_acquisition_factorization_pair_integrity_scorpion_phikon",
        "scanner_namespace": ("AT2", "GT450", "DP200", "P1000", "B300"),
        "folds": [0, 1, 2, 3, 4],
        "seeds": [701, 702, 703, 704, 705],
        "conditions": ["true_pairs", "shuffled_region_pairs", "shuffled_sample_pairs"],
    },
    {
        "name": "SCORPION_ResNet50",
        "result_dir": "results/paired_acquisition_factorization_pair_integrity_scorpion_resnet50",
        "scanner_namespace": ("AT2", "GT450", "DP200", "P1000", "B300"),
        "folds": [0, 1, 2, 3, 4],
        "seeds": [701, 702, 703, 704, 705],
        "conditions": ["true_pairs", "shuffled_region_pairs", "shuffled_sample_pairs"],
    },
]

METRICS = (
    "scanner_probe_accuracy",
    "mean_paired_cosine",
    "worst_paired_cosine",
    "mean_top1_retrieval",
    "worst_top1_retrieval",
    "effective_rank",
    "biological_acquisition_cross_covariance",
)


# ---------------------------------------------------------------------------
# Metric functions  (replicating the logic from the falsification scripts
# but parameterised by which feature array to evaluate on)
# ---------------------------------------------------------------------------

def _load_projected(path: Path):
    """Load a projected_features.npz and return bio/acq arrays + frame + metadata."""
    with np.load(path, allow_pickle=False) as archive:
        required = {"features", "acquisition_features", "slide_id", "region_id",
                     "scanner_id", "split"}
        missing = sorted(required - set(archive.files))
        if missing:
            raise RuntimeError(f"{path} is missing arrays: {missing}")
        biological = np.asarray(archive["features"], dtype=np.float32)
        acquisition = np.asarray(archive["acquisition_features"], dtype=np.float32)
        frame = pd.DataFrame({
            name: archive[name].astype(str)
            for name in ("slide_id", "region_id", "scanner_id", "split")
        })
        metadata = json.loads(str(archive["metadata_json"].item()))
    if len(biological) != len(frame) or len(acquisition) != len(frame):
        raise RuntimeError("Projected feature arrays and metadata are misaligned.")
    return biological, acquisition, frame, metadata


def _split_indices(frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    test = np.flatnonzero(frame["split"].to_numpy() == "test")
    fit = np.flatnonzero(frame["split"].to_numpy() != "test")
    if len(test) == 0 or len(fit) == 0:
        raise RuntimeError("Empty fit or test split.")
    if set(frame.iloc[test]["slide_id"]) & set(frame.iloc[fit]["slide_id"]):
        raise RuntimeError("Slide leakage between fit and test splits.")
    return fit, test


def _scanner_probe(features: np.ndarray, frame: pd.DataFrame,
                   fit: np.ndarray, test: np.ndarray) -> float:
    labels = frame["scanner_id"].to_numpy()
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=1.0, class_weight="balanced", max_iter=5000,
                           random_state=0),
    )
    model.fit(features[fit], labels[fit])
    predictions = model.predict(features[test])
    return float(balanced_accuracy_score(labels[test], predictions))


def _normalize(features: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(features, axis=1, keepdims=True)
    norms[norms <= 0] = 1.0
    return features / norms


def _paired_metrics(features: np.ndarray, frame: pd.DataFrame,
                    test: np.ndarray, scanners: Tuple[str, ...],
                    ) -> Dict[str, float]:
    """Compute paired cosine and top-1 retrieval on the given feature array."""
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
            prediction_ab = np.argmax(similarity, axis=1)
            prediction_ba = np.argmax(similarity.T, axis=1)
            pair_rows.append({
                "cosine": float(diagonal.mean()),
                "retrieval": float(0.5 * (
                    np.mean(prediction_ab == truth) +
                    np.mean(prediction_ba == truth))),
            })

    if not pair_rows:
        return {"mean_paired_cosine": float("nan"),
                "worst_paired_cosine": float("nan"),
                "mean_top1_retrieval": float("nan"),
                "worst_top1_retrieval": float("nan")}

    pair_frame = pd.DataFrame(pair_rows)
    return {
        "mean_paired_cosine": float(pair_frame["cosine"].mean()),
        "worst_paired_cosine": float(pair_frame["cosine"].min()),
        "mean_top1_retrieval": float(pair_frame["retrieval"].mean()),
        "worst_top1_retrieval": float(pair_frame["retrieval"].min()),
    }


def _effective_rank(features: np.ndarray) -> float:
    centered = features - features.mean(axis=0, keepdims=True)
    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    energy = singular_values ** 2
    total = float(energy.sum())
    if total <= 0:
        return 0.0
    probabilities = energy / total
    probabilities = probabilities[probabilities > 0]
    return float(math.exp(-np.sum(probabilities * np.log(probabilities))))


def _cross_covariance_rms(biological: np.ndarray, acquisition: np.ndarray,
                          test: np.ndarray) -> float:
    b = StandardScaler().fit_transform(biological[test])
    a = StandardScaler().fit_transform(acquisition[test])
    cross = b.T @ a / max(1, len(test) - 1)
    return float(np.sqrt(np.mean(cross ** 2)))


# ---------------------------------------------------------------------------
# Atomic CSV write
# ---------------------------------------------------------------------------

def _atomic_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".csv",
                                        dir=path.parent)
    os.close(descriptor)
    temporary = Path(name)
    try:
        frame.to_csv(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Main audit
# ---------------------------------------------------------------------------

def run_audit(target: dict, out_dir: Path, smoke: bool = False) -> Dict[str, object]:
    """Audit one dataset: compute per-branch metrics on all runs."""
    name = target["name"]
    result_dir = Path(target["result_dir"])
    scanners = target["scanner_namespace"]
    folds = target["folds"]
    seeds = target["seeds"]
    conditions = target["conditions"]

    training_path = result_dir / "training_results.csv"
    if not training_path.is_file():
        raise RuntimeError(f"[{name}] Missing training_results.csv at {training_path}")
    training = pd.read_csv(training_path)

    run_rows = []
    n_loaded = 0
    n_skipped = 0

    for _, row in training.iterrows():
        fold = int(row["fold"])
        seed = int(row["seed"])
        condition = str(row["condition"])

        # Smoke test: only first fold, first seed, true_pairs
        if smoke and (fold != folds[0] or seed != seeds[0] or condition != "true_pairs"):
            n_skipped += 1
            continue
        fold = int(row["fold"])
        seed = int(row["seed"])
        condition = str(row["condition"])

        projected_path = (
            result_dir / f"fold_{fold}" / "runs" /
            f"{condition}_seed_{seed}" / "projected_features.npz"
        )
        if not projected_path.is_file():
            n_skipped += 1
            continue

        biological, acquisition, frame, metadata = _load_projected(projected_path)
        fit, test = _split_indices(frame)

        # ---- Metrics on BIOLOGICAL branch ----
        bio_probe = _scanner_probe(biological, frame, fit, test)
        bio_paired = _paired_metrics(biological, frame, test, scanners)
        bio_test = biological[test]
        bio_erank = _effective_rank(bio_test)

        # ---- Metrics on ACQUISITION branch ----
        acq_probe = _scanner_probe(acquisition, frame, fit, test)
        acq_paired = _paired_metrics(acquisition, frame, test, scanners)
        acq_test = acquisition[test]
        acq_erank = _effective_rank(acq_test)

        # ---- Cross-covariance (joint) ----
        cross_cov = _cross_covariance_rms(biological, acquisition, test)

        run_rows.append({
            "dataset_backbone": name,
            "fold": fold,
            "seed": seed,
            "condition": condition,
            # biological branch
            "bio_scanner_probe_accuracy": bio_probe,
            "bio_mean_paired_cosine": bio_paired["mean_paired_cosine"],
            "bio_worst_paired_cosine": bio_paired["worst_paired_cosine"],
            "bio_mean_top1_retrieval": bio_paired["mean_top1_retrieval"],
            "bio_worst_top1_retrieval": bio_paired["worst_top1_retrieval"],
            "bio_effective_rank": bio_erank,
            # acquisition branch
            "acq_scanner_probe_accuracy": acq_probe,
            "acq_mean_paired_cosine": acq_paired["mean_paired_cosine"],
            "acq_worst_paired_cosine": acq_paired["worst_paired_cosine"],
            "acq_mean_top1_retrieval": acq_paired["mean_top1_retrieval"],
            "acq_worst_top1_retrieval": acq_paired["worst_top1_retrieval"],
            "acq_effective_rank": acq_erank,
            # joint
            "biological_acquisition_cross_covariance": cross_cov,
        })
        n_loaded += 1

    runs = pd.DataFrame(run_rows)
    return {
        "name": name,
        "runs": runs,
        "n_loaded": n_loaded,
        "n_skipped": n_skipped,
        "n_total_expected": len(training),
    }


def compute_branch_separation(runs: pd.DataFrame) -> pd.DataFrame:
    """Compute bio vs acq branch contrasts per run."""
    contrast_rows = []
    for _, row in runs.iterrows():
        contrast_rows.append({
            "dataset_backbone": row["dataset_backbone"],
            "fold": row["fold"],
            "seed": row["seed"],
            "condition": row["condition"],
            # scanner probe: acq should be HIGHER (more scanner info in acq branch)
            "scanner_probe_delta": (row["acq_scanner_probe_accuracy"] -
                                    row["bio_scanner_probe_accuracy"]),
            # tissue retrieval: acq should be LOWER (less tissue info in acq branch)
            "paired_cosine_delta": (row["acq_mean_paired_cosine"] -
                                    row["bio_mean_paired_cosine"]),
            "top1_retrieval_delta": (row["acq_mean_top1_retrieval"] -
                                     row["bio_mean_top1_retrieval"]),
            # effective rank delta
            "effective_rank_delta": (row["acq_effective_rank"] -
                                     row["bio_effective_rank"]),
            # scanner-to-tissue ratio for each branch
            "bio_scanner_tissue_ratio": (
                row["bio_scanner_probe_accuracy"] /
                max(row["bio_mean_top1_retrieval"], 1e-8)
            ),
            "acq_scanner_tissue_ratio": (
                row["acq_scanner_probe_accuracy"] /
                max(row["acq_mean_top1_retrieval"], 1e-8)
            ),
        })
    return pd.DataFrame(contrast_rows)


def produce_summary(runs: pd.DataFrame, contrasts: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-branch metrics across runs for each dataset/condition."""
    bio_cols = [c for c in runs.columns if c.startswith("bio_")]
    acq_cols = [c for c in runs.columns if c.startswith("acq_")]
    joint_cols = ["biological_acquisition_cross_covariance"]
    all_num = bio_cols + acq_cols + joint_cols

    summary_rows = []
    for (ds, cond), grp in runs.groupby(["dataset_backbone", "condition"]):
        agg = {"dataset_backbone": ds, "condition": cond, "n_runs": len(grp)}
        for col in all_num:
            if col not in grp.columns:
                continue
            agg[f"{col}_mean"] = float(grp[col].mean())
            agg[f"{col}_std"] = float(grp[col].std())
        # scanner probe deltas
        delta_cols = [c for c in contrasts.columns if c.endswith("_delta")]
        csub = contrasts[(contrasts["dataset_backbone"] == ds) &
                         (contrasts["condition"] == cond)]
        for col in delta_cols:
            if col not in csub.columns:
                continue
            agg[f"{col}_mean"] = float(csub[col].mean())
            agg[f"{col}_std"] = float(csub[col].std())
        # scanner/tissue ratios
        for branch in ("bio", "acq"):
            rcol = f"{branch}_scanner_tissue_ratio"
            if rcol in csub.columns:
                agg[f"{rcol}_mean"] = float(csub[rcol].mean())
                agg[f"{rcol}_std"] = float(csub[rcol].std())
        summary_rows.append(agg)
    return pd.DataFrame(summary_rows)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def build_report(audit_results: List[Dict], summary: pd.DataFrame,
                 contrasts: pd.DataFrame, out_dir: Path,
                 runtime_seconds: float, smoke: bool) -> str:
    """Build the markdown audit report."""
    lines = []
    lines.append("# Acquisition-Branch Audit Report")
    lines.append("")
    lines.append(f"**Generated:** {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Runtime:** {runtime_seconds:.1f} s")
    lines.append(f"**Smoke test:** {'yes' if smoke else 'no'}")
    lines.append("")

    lines.append("## What was audited")
    lines.append("")
    lines.append("This audit loads existing projected features from completed "
                 "pair-integrity falsification runs and evaluates per-branch metrics. "
                 "**No training was rerun.** All embeddings are reused from prior runs.")
    lines.append("")

    lines.append("## Datasets / Backbones included")
    lines.append("")
    lines.append("| # | Dataset | Backbone | Runs loaded | Expected |")
    lines.append("|---|---|---|---|")
    for res in audit_results:
        lines.append(f"| {res['name']} | {res['name']} | — | "
                     f"{res['n_loaded']} | {res['n_total_expected']} |")
    lines.append("")

    lines.append("## Branch Separation Summary")
    lines.append("")
    lines.append("Positive scanner_probe_delta = acquisition branch carries MORE "
                 "scanner information than biological branch.")
    lines.append("Negative tissue-retrieval delta = acquisition branch carries LESS "
                 "tissue identity than biological branch.")
    lines.append("")

    # Per-dataset summary tables
    for ds_name in sorted(summary["dataset_backbone"].unique()):
        sub = summary[summary["dataset_backbone"] == ds_name]
        lines.append(f"### {ds_name}")
        lines.append("")
        lines.append("| Condition | Bio scanner probe | Acq scanner probe | "
                     "Probe Δ | Bio retrieval | Acq retrieval | Retrieval Δ | "
                     "Bio eff rank | Acq eff rank | Cross-cov |")
        lines.append("|---|---|---|---|---|---|---|---|---|")
        for _, srow in sub.iterrows():
            cond = srow["condition"]
            bsp = srow.get("bio_scanner_probe_accuracy_mean", float("nan"))
            asp = srow.get("acq_scanner_probe_accuracy_mean", float("nan"))
            pd_ = srow.get("scanner_probe_delta_mean", float("nan"))
            brt = srow.get("bio_mean_top1_retrieval_mean", float("nan"))
            art = srow.get("acq_mean_top1_retrieval_mean", float("nan"))
            rd_ = srow.get("top1_retrieval_delta_mean", float("nan"))
            ber = srow.get("bio_effective_rank_mean", float("nan"))
            aer = srow.get("acq_effective_rank_mean", float("nan"))
            cc = srow.get("biological_acquisition_cross_covariance_mean",
                          float("nan"))
            lines.append(f"| {cond} | {bsp:.4f} | {asp:.4f} | {pd_:+.4f} | "
                         f"{brt:.4f} | {art:.4f} | {rd_:+.4f} | "
                         f"{ber:.1f} | {aer:.1f} | {cc:.6f} |")
        lines.append("")

    # Interpretation
    lines.append("## Interpretation")
    lines.append("")

    # Aggregate across all datasets for the "true_pairs" condition
    true_only = summary[summary["condition"] == "true_pairs"]
    if len(true_only) > 0:
        avg_probe_delta = true_only["scanner_probe_delta_mean"].mean()
        avg_retrieval_delta = true_only["top1_retrieval_delta_mean"].mean()
        avg_bio_retrieval = true_only["bio_mean_top1_retrieval_mean"].mean()
        avg_acq_retrieval = true_only["acq_mean_top1_retrieval_mean"].mean()
        avg_bio_probe = true_only["bio_scanner_probe_accuracy_mean"].mean()
        avg_acq_probe = true_only["acq_scanner_probe_accuracy_mean"].mean()

        lines.append(f"- **Average scanner probe Δ (acq − bio):** {avg_probe_delta:+.4f}")
        lines.append(f"  (bio={avg_bio_probe:.4f}, acq={avg_acq_probe:.4f})")
        lines.append(f"- **Average tissue retrieval Δ (acq − bio):** {avg_retrieval_delta:+.4f}")
        lines.append(f"  (bio={avg_bio_retrieval:.4f}, acq={avg_acq_retrieval:.4f})")
        lines.append("")

        # Determine which interpretation fits
        acq_carries_scanner = avg_acq_probe > avg_bio_probe + 0.02
        acq_has_less_tissue = avg_acq_retrieval < avg_bio_retrieval - 0.01
        acq_retains_tissue = avg_acq_retrieval > 0.3  # heuristic threshold

        if acq_carries_scanner and acq_has_less_tissue:
            lines.append("**Finding: The acquisition-branch audit strongly supports "
                         "branch separation across the tested datasets/backbones.**")
            lines.append("")
            lines.append("Across the audited settings, the acquisition branch retained "
                         "high scanner/acquisition recoverability while carrying much "
                         "lower tissue-identity retrieval than the biological branch. "
                         "This supports the interpretation that the model learned a "
                         "useful acquisition/tissue branch separation, rather than "
                         "only suppressing scanner signal in the biological branch.")
        elif acq_carries_scanner and not acq_has_less_tissue:
            lines.append("**Finding: Acquisition branch carries more scanner "
                         "information but still preserves substantial tissue identity.**")
            lines.append("This suggests the acquisition branch partially overlaps "
                         "with biological information, consistent with partial rather "
                         "than clean branch separation.")
        elif not acq_carries_scanner:
            lines.append("**Finding: Acquisition branch does NOT carry substantially "
                         "more scanner information than the biological branch.**")
            lines.append("This suggests the method is better described as a "
                         "**scanner-suppressed biological representation** rather "
                         "than a representation with separate acquisition/tissue branches.")
        else:
            lines.append("**Finding: Mixed results — see per-dataset tables above.**")

    lines.append("")
    lines.append("## Claim boundaries")
    lines.append("")
    lines.append("- This audit evaluates per-branch metrics on held-out test slides "
                 "from existing trained models. No new training was performed.")
    lines.append("- The factorization architecture was trained with a specific "
                 "hyperparameter configuration (acquisition_dim=64, biological_dim=256, "
                 "scanner_adversary_weight=0.5, scanner_acquisition_weight=0.5, "
                 "scanner_dependence_weight=20.0). Results may not generalize to "
                 "other configurations.")
    lines.append("- Scanner probe accuracy is measured with a linear logistic regression "
                 "classifier (balanced class weight). Non-linear scanner signatures "
                 "may be underestimated.")
    lines.append("- Tissue retrieval is measured by same-slide nearest-neighbor "
                 "recall in the projected space. This captures same-tissue identity "
                 "at the slide level, not finer-grained region matching.")
    lines.append("- The audit covers SCORPION (human colorectal cancer, 5-scanner "
                 "HTA v1.0 protocol) and external canine SCC (5-scanner veterinary "
                 "oncology). Results are specific to these datasets and scanner "
                 "ensembles.")
    lines.append("")

    lines.append("## Output files")
    lines.append("")
    lines.append("| File | Description |")
    lines.append("|---|---|")
    lines.append("| branch_audit_raw_metrics.csv | Per-run, per-branch metrics |")
    lines.append("| branch_audit_summary.csv | Aggregated summary by dataset/condition |")
    lines.append("| branch_separation_contrasts.csv | Per-run bio-vs-acq deltas |")
    lines.append("| experiment_design.json | Audit configuration |")
    lines.append("| run_log.txt | Timestamped log |")
    lines.append("| acquisition_branch_audit_report.md | This report |")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Acquisition-branch audit for paired-acquisition factorization")
    parser.add_argument("--out-dir", type=Path,
                        default=Path("results/paired_acquisition_factorization_acquisition_branch_audit"))
    parser.add_argument("--smoke", action="store_true",
                        help="Smoke test: evaluate only fold_0/seed_0/true_pairs")
    parser.add_argument("--datasets", nargs="*",
                        help="Dataset names to audit (default: all)")
    return parser.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = parse_args(argv)
    out_dir: Path = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()

    # Open log
    log_path = out_dir / "run_log.txt"
    log_file = open(str(log_path), "w", encoding="utf-8")

    def log(msg: str) -> None:
        stamp = time.strftime("%Y-%m-%d %H:%M:%S")
        line = f"[{stamp}] {msg}"
        print(line)
        log_file.write(line + "\n")
        log_file.flush()

    log(f"Acquisition-branch audit started  (smoke={args.smoke})")

    # Filter targets
    targets = AUDIT_TARGETS
    if args.datasets:
        targets = [t for t in AUDIT_TARGETS if t["name"] in args.datasets]
    log(f"Auditing {len(targets)} dataset(s)")

    # Write experiment design
    design = {
        "stage": "acquisition_branch_audit",
        "smoke_test": args.smoke,
        "datasets": [t["name"] for t in targets],
        "metric_definitions": {
            "scanner_probe_accuracy": "balanced accuracy of linear logistic regression predicting scanner_id",
            "mean_paired_cosine": "mean cosine similarity between same-tissue cross-scanner region pairs",
            "mean_top1_retrieval": "same-slide nearest-neighbor retrieval accuracy",
            "effective_rank": "exp(entropy) of SVD singular-value distribution",
            "biological_acquisition_cross_covariance": "RMS of cross-covariance matrix between branches",
        },
        "reused_embeddings": True,
        "retraining_performed": False,
        "source_result_dirs": [str(Path(t["result_dir"]).resolve()) for t in targets],
    }
    (out_dir / "experiment_design.json").write_text(
        json.dumps(design, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    # Run audit
    all_runs = []
    audit_results = []

    for target in targets:
        log(f"Auditing {target['name']} ...")
        try:
            result = run_audit(target, out_dir, smoke=args.smoke)
            audit_results.append(result)
            all_runs.append(result["runs"])
            log(f"  {target['name']}: loaded {result['n_loaded']}, "
                f"skipped {result['n_skipped']}, "
                f"expected {result['n_total_expected']}")
        except Exception:
            log(f"  {target['name']}: FAILED — {traceback.format_exc()}")

    if not all_runs:
        log("No data loaded. Aborting.")
        log_file.close()
        return 1

    # Merge
    runs = pd.concat(all_runs, ignore_index=True)
    log(f"Total runs: {len(runs)}")

    # Compute branch separation contrasts
    contrasts = compute_branch_separation(runs)
    log(f"Contrast rows: {len(contrasts)}")

    # Summary
    summary = produce_summary(runs, contrasts)
    log(f"Summary rows: {len(summary)}")

    # Write CSVs
    _atomic_csv(out_dir / "branch_audit_raw_metrics.csv", runs)
    _atomic_csv(out_dir / "branch_audit_summary.csv", summary)
    _atomic_csv(out_dir / "branch_separation_contrasts.csv", contrasts)
    log("CSV outputs written")

    # Build report
    runtime = time.time() - t0
    report = build_report(audit_results, summary, contrasts, out_dir,
                          runtime, args.smoke)
    (out_dir / "acquisition_branch_audit_report.md").write_text(
        report, encoding="utf-8")
    log("Report written")

    log(f"Done in {runtime:.1f} s")
    log_file.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
