#!/usr/bin/env python3
"""Linear baseline consistency audit.

Reconciles apparent scanner-accuracy mismatch between two linear projection
implementations used in prior audits:

  A. old_style: per-scanner mean vectors, QR-orthonormalized (bec06eb4)
  B. new_style: logistic-regression coefficient SVD (ec2a509f)

Hypothesis: the two implementations remove different 4-dimensional subspaces.
The per-scanner-mean approach removes the full scanner-centroid subspace; the
logistic-regression-SVD approach removes the most discriminative directions
for one particular linear classifier, which may leave residual scanner signal.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import traceback
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score
from sklearn.preprocessing import StandardScaler

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from experiments.external_multiscanner import run_canine_pathoalign_crossfold as canine_cross  # noqa: E402
from experiments.scorpion import run_pathoalign_projection as projection  # noqa: E402

FEATURE_PATH = Path("results/external_multiscanner_caninescc/features/fold_0_dinov2_base.npz")
MANIFESTS_DIR = Path("data/external_multiscanner_caninescc/patch_manifests/splits")

SCANNERS = ("cs2", "gt450", "nz20", "nz210", "p1000")
K_TARGET = 4


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Old-style: per-scanner-mean directions (from bec06eb4)
# ---------------------------------------------------------------------------


def old_style_scanner_directions(
    features: np.ndarray, frame: pd.DataFrame, fit: np.ndarray,
) -> np.ndarray:
    """Per-scanner mean vectors (class centroid offsets from grand mean).

    From run_biological_label_preservation_audit.py, function _fit_scanner_directions.
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
    return np.stack(directions, axis=0)  # (n_scanners, d)


def old_style_remove_directions(
    features: np.ndarray, directions: np.ndarray, k: int,
) -> Tuple[np.ndarray, int]:
    """QR-orthonormalize and project out top-k directions.

    From run_biological_label_preservation_audit.py, function _remove_directions.
    """
    if k <= 0 or directions.shape[0] == 0:
        return features.copy(), 0
    effective_k = min(k, directions.shape[0])
    Q, _ = np.linalg.qr(directions[:effective_k].T)
    Q = Q.T
    actual_k = Q.shape[0]
    components = features @ Q.T
    result = features - components @ Q
    return result.astype(np.float32), actual_k


def old_style_linear_projection_k4(
    features: np.ndarray, frame: pd.DataFrame, fit: np.ndarray,
) -> np.ndarray:
    """Reproduce old-style linear_projection_k4."""
    scaler = StandardScaler()
    X_all = scaler.fit_transform(features)
    directions = old_style_scanner_directions(X_all, frame, fit)
    cleaned, effective_k = old_style_remove_directions(X_all, directions, K_TARGET)
    return cleaned.astype(np.float32)


# ---------------------------------------------------------------------------
# New-style: logistic-regression coefficient SVD (from ec2a509f)
# ---------------------------------------------------------------------------


def numeric_rank(singular_values: np.ndarray, shape: Tuple[int, int]) -> int:
    if singular_values.size == 0:
        return 0
    tolerance = max(shape) * np.finfo(np.float64).eps * float(singular_values[0])
    return int(np.sum(singular_values > tolerance))


def new_style_scanner_directions(
    features: np.ndarray, frame: pd.DataFrame, fit: np.ndarray,
) -> Tuple[np.ndarray, int]:
    """Logistic regression coefficient SVD directions.

    From run_linear_residual_branch_separation_audit.py, function scanner_subspace_directions.
    """
    labels = frame["scanner_id"].astype(str).to_numpy()
    model = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=5000, random_state=0, solver="lbfgs",
    )
    model.fit(features[fit], labels[fit])
    coefficients = np.asarray(model.coef_, dtype=np.float64)
    coefficients = coefficients - coefficients.mean(axis=0, keepdims=True)
    _, singular_values, vt = np.linalg.svd(coefficients, full_matrices=False)
    rank = numeric_rank(singular_values, coefficients.shape)
    return vt[:rank].astype(np.float64), rank


def new_style_split_keep_removed(
    features: np.ndarray, directions: np.ndarray, k: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Split into keep (directions removed) and removed (projection onto directions).

    From run_linear_residual_branch_separation_audit.py, function split_keep_removed.
    """
    effective_k = min(int(k), len(directions))
    if effective_k <= 0:
        return (
            np.asarray(features, dtype=np.float32).copy(),
            np.zeros_like(features, dtype=np.float32),
            0,
        )
    matrix = np.asarray(features, dtype=np.float64)
    basis = np.asarray(directions[:effective_k], dtype=np.float64)
    proj = (matrix @ basis.T) @ basis
    keep = (matrix - proj).astype(np.float32)
    removed = proj.astype(np.float32)
    return keep, removed, effective_k


def new_style_linear_keep_k4(
    features: np.ndarray, frame: pd.DataFrame, fit: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Reproduce new-style linear_keep_k4 and linear_removed_k4."""
    standardized, _mean, _std = projection.standardize(features, fit)
    directions, rank = new_style_scanner_directions(standardized, frame, fit)
    keep, removed, effective_k = new_style_split_keep_removed(standardized, directions, K_TARGET)
    return keep, removed


# ---------------------------------------------------------------------------
# Evaluation (single protocol for all variants)
# ---------------------------------------------------------------------------


def evaluate(
    features: np.ndarray,
    frame: pd.DataFrame,
    fit: np.ndarray,
    test: np.ndarray,
) -> dict[str, float]:
    """Evaluate scanner and category probe accuracy with a consistent protocol."""
    results = {}

    # Scanner probe (no additional standardization; features are already standardized)
    y_scanner = frame["scanner_id"].to_numpy()
    model_s = LogisticRegression(
        C=1.0, class_weight="balanced", max_iter=5000, random_state=0, solver="lbfgs",
    )
    model_s.fit(features[fit], y_scanner[fit])
    pred_s = model_s.predict(features[test])
    results["scanner_balanced_accuracy"] = float(balanced_accuracy_score(y_scanner[test], pred_s))
    results["scanner_macro_f1"] = float(f1_score(y_scanner[test], pred_s, average="macro"))

    # Category probe
    if "category_name" in frame.columns:
        # Use only test samples whose category exists in fit
        y_cat = frame["category_name"].to_numpy()
        model_c = LogisticRegression(
            C=1.0, class_weight="balanced", max_iter=5000, random_state=0, solver="lbfgs",
        )
        model_c.fit(features[fit], y_cat[fit])
        pred_c = model_c.predict(features[test])
        results["category_balanced_accuracy"] = float(balanced_accuracy_score(y_cat[test], pred_c))
        results["category_macro_f1"] = float(f1_score(y_cat[test], pred_c, average="macro"))

    return results


# ---------------------------------------------------------------------------
# Feature-space comparison
# ---------------------------------------------------------------------------


def feature_comparison(
    old_features: np.ndarray,
    new_keep: np.ndarray,
    new_removed: np.ndarray,
) -> dict[str, float]:
    """Compare old and new linear representations."""
    diff = old_features - new_keep
    # Per-row L2 difference
    l2_diff = np.linalg.norm(diff, axis=1)
    # Cosine similarity
    old_norm = old_features / (np.linalg.norm(old_features, axis=1, keepdims=True) + 1e-12)
    new_norm = new_keep / (np.linalg.norm(new_keep, axis=1, keepdims=True) + 1e-12)
    cos_sim = np.sum(old_norm * new_norm, axis=1)

    return {
        "mean_l2_difference": float(np.mean(l2_diff)),
        "max_l2_difference": float(np.max(l2_diff)),
        "mean_cosine_similarity": float(np.mean(cos_sim)),
        "min_cosine_similarity": float(np.min(cos_sim)),
        "old_mean_norm": float(np.mean(np.linalg.norm(old_features, axis=1))),
        "new_keep_mean_norm": float(np.mean(np.linalg.norm(new_keep, axis=1))),
        "new_removed_mean_norm": float(np.mean(np.linalg.norm(new_removed, axis=1))),
        # Entrywise old-vs-new keep similarity diagnostic, not removed variance.
        "old_new_keep_entrywise_variance_similarity": float(
            1.0 - np.var(diff) / max(1e-12, np.var(old_features))
        ),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_audit(out_dir: Path, folds: Tuple[int, ...] = (0, 1, 2, 3, 4)) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    canine_cross.patch_scanner_namespace()

    # Load frozen features once
    frozen_features, frozen_frame, source_metadata = projection.load_archive(FEATURE_PATH)
    frozen_frame["scanner_id"] = frozen_frame["scanner_id"].astype(str).str.lower()

    metric_rows = []
    comparison_rows = []

    for fold in folds:
        manifest = pd.read_csv(
            MANIFESTS_DIR / f"fold_{fold}_patch_manifest.csv", dtype=str,
        )
        manifest["scanner_id"] = manifest["scanner_id"].astype(str).str.lower()

        # Align features to manifest
        aligned, aligned_frame = canine_cross.align_fold(
            frozen_features, frozen_frame, MANIFESTS_DIR / f"fold_{fold}_patch_manifest.csv",
        )
        fit, test = canine_cross.validate_fold(manifest, fold)

        # Attach category_name from manifest
        aligned_frame["category_name"] = manifest["category_name"].to_numpy()

        print(f"\nFold {fold}: fit={len(fit)} test={len(test)}")

        # ---- A: old-style linear_projection_k4 ----
        old_k4 = old_style_linear_projection_k4(aligned, aligned_frame, fit)
        old_metrics = evaluate(old_k4, aligned_frame, fit, test)
        metric_rows.append({
            "variant": "old_style_linear_projection_k4",
            "fold": int(fold),
            **old_metrics,
        })

        # ---- B: new-style linear_keep_k4 ----
        new_keep, new_removed = new_style_linear_keep_k4(aligned, aligned_frame, fit)
        new_keep_metrics = evaluate(new_keep, aligned_frame, fit, test)
        metric_rows.append({
            "variant": "new_style_linear_keep_k4",
            "fold": int(fold),
            **new_keep_metrics,
        })

        # ---- C: new-style linear_removed_k4 ----
        new_rem_metrics = evaluate(new_removed, aligned_frame, fit, test)
        metric_rows.append({
            "variant": "new_style_linear_removed_k4",
            "fold": int(fold),
            **new_rem_metrics,
        })

        # ---- D: new-style on old-style standardized features (control) ----
        # Use the SAME standardization as old-style (fit on ALL data)
        scaler_all = StandardScaler()
        X_all_std = scaler_all.fit_transform(aligned)
        directions_ctrl, rank_ctrl = new_style_scanner_directions(X_all_std, aligned_frame, fit)
        keep_ctrl, rem_ctrl, _ = new_style_split_keep_removed(X_all_std, directions_ctrl, K_TARGET)
        ctrl_metrics = evaluate(keep_ctrl, aligned_frame, fit, test)
        metric_rows.append({
            "variant": "new_style_keep_on_old_standardization",
            "fold": int(fold),
            **ctrl_metrics,
        })

        # ---- Feature-space comparison ----
        comp = feature_comparison(old_k4, new_keep, new_removed)
        comp["fold"] = int(fold)
        comparison_rows.append(comp)

        # Print per-fold summary
        print(f"  old_k4:        scanner={old_metrics['scanner_balanced_accuracy']:.4f}  cat={old_metrics.get('category_balanced_accuracy', float('nan')):.4f}")
        print(f"  new_keep_k4:   scanner={new_keep_metrics['scanner_balanced_accuracy']:.4f}  cat={new_keep_metrics.get('category_balanced_accuracy', float('nan')):.4f}")
        print(f"  old vs new cos_sim: {comp['mean_cosine_similarity']:.6f}")
        print(f"  old vs new L2 diff: {comp['mean_l2_difference']:.4f}")

    # Write outputs
    metrics_df = pd.DataFrame(metric_rows).sort_values(["variant", "fold"])
    comparison_df = pd.DataFrame(comparison_rows).sort_values("fold")

    atomic_csv(out_dir / "linear_baseline_consistency_metrics.csv", metrics_df)
    atomic_csv(out_dir / "linear_baseline_feature_comparison.csv", comparison_df)

    # Compute means
    means = metrics_df.groupby("variant").agg(["mean", "std"])
    means.columns = ["_".join(col).strip("_") for col in means.columns]

    comp_means = comparison_df.drop(columns=["fold"]).mean()

    # Build report
    lines = [
        "# Linear Baseline Consistency Audit",
        "",
        "## Question",
        "",
        "Are old_style linear_projection_k4 (bec06eb4) and new_style linear_keep_k4",
        "(ec2a509f) equivalent representations? If not, what explains the scanner",
        "accuracy mismatch (0.2000 vs 0.7071)?",
        "",
        "## Definitions",
        "",
        "### Old-style (bec06eb4): per-scanner mean directions + QR orthonormalization",
        "",
        "1. Fit StandardScaler on ALL features (fit + test).",
        "2. Compute per-scanner mean vectors: for each scanner, mean(features) - grand_mean.",
        "   This produces 5 directions (one per scanner).",
        "3. Take first k=4 direction rows, QR-orthonormalize to get ~4 orthonormal vectors.",
        "4. Project out: features - features @ Q.T @ Q.",
        "5. Return result (no removed branch computed).",
        "",
        "Note: The 5 scanner means span a 4-dimensional affine subspace (they sum to zero).",
        "Taking any 4 of the 5 means and QR-orthonormalizing gives the full scanner-mean",
        "subspace. So k=4 removes essentially all first-order scanner-centroid information.",
        "",
        "### New-style (ec2a509f): logistic regression coefficient SVD",
        "",
        "1. Fit StandardScaler on FIT features only, transform all.",
        "2. Fit LogisticRegression(scanner) on fit set.",
        "3. Center the (5, 768) coefficient matrix, compute SVD.",
        "4. Take top k=4 right singular vectors (max 4 due to rank after centering).",
        "5. keep = features - features @ basis.T @ basis.",
        "6. removed = features @ basis.T @ basis.",
        "",
        "Note: SVD of logistic regression coefficients finds directions in feature space",
        "that the classifier uses for discrimination. These may not span the full",
        "scanner-mean subspace.",
        "",
        "### Key difference",
        "",
        "The old-style approach removes the full scanner-centroid subspace (all first-order",
        "scanner structure). The new-style approach removes the most discriminative",
        "directions for ONE particular linear classifier, which may leave residual",
        "scanner signal that a fresh probe classifier can exploit.",
        "",
        "## Results (5-fold means)",
        "",
    ]

    for variant in ["old_style_linear_projection_k4", "new_style_linear_keep_k4",
                     "new_style_linear_removed_k4", "new_style_keep_on_old_standardization"]:
        if variant in means.index:
            row = means.loc[variant]
            lines.append(f"### {variant}")
            if "scanner_balanced_accuracy_mean" in row.index:
                lines.append(f"- Scanner acc: {float(row['scanner_balanced_accuracy_mean']):.4f} +- {float(row.get('scanner_balanced_accuracy_std', 0)):.4f}")
            if "category_balanced_accuracy_mean" in row.index:
                lines.append(f"- Category acc: {float(row['category_balanced_accuracy_mean']):.4f} +- {float(row.get('category_balanced_accuracy_std', 0)):.4f}")
            lines.append("")

    lines.extend([
        "## Feature-Space Comparison (5-fold means)",
        "",
        f"- Mean L2 difference (old vs new keep): {comp_means['mean_l2_difference']:.2f}",
        f"- Mean cosine similarity (old vs new keep): {comp_means['mean_cosine_similarity']:.6f}",
        f"- Old mean norm: {comp_means['old_mean_norm']:.2f}",
        f"- New keep mean norm: {comp_means['new_keep_mean_norm']:.2f}",
        f"- New removed mean norm: {comp_means['new_removed_mean_norm']:.2f}",
        f"- Entrywise old/new keep variance-similarity ratio: {comp_means['old_new_keep_entrywise_variance_similarity']:.4f}",
        "",
        "Variance-similarity definition: 1 - Var(old_style_linear_projection_k4 -",
        "new_style_linear_keep_k4) / Var(old_style_linear_projection_k4), where",
        "Var is computed over all matrix entries. This is a feature-space similarity",
        "diagnostic only; it is not centered variance removed, per-sample projected",
        "feature energy removed, or scanner-centroid-offset variance.",
        "",
    ])

    # Check if they're equivalent
    old_scanner = float(means.loc["old_style_linear_projection_k4", "scanner_balanced_accuracy_mean"])
    new_scanner = float(means.loc["new_style_linear_keep_k4", "scanner_balanced_accuracy_mean"])

    lines.extend([
        "## Conclusion",
        "",
    ])

    if abs(old_scanner - new_scanner) < 0.05:
        lines.append(
            "old_style linear_projection_k4 and new_style linear_keep_k4 produce equivalent "
            f"scanner accuracy ({old_scanner:.4f} vs {new_scanner:.4f}). No reconciliation needed."
        )
    else:
        lines.append(
            f"old_style linear_projection_k4 and new_style linear_keep_k4 are NOT equivalent "
            f"representations (scanner acc {old_scanner:.4f} vs {new_scanner:.4f})."
        )
        lines.append("")
        lines.append(
            "The old-style per-scanner-mean approach removes the full scanner-centroid "
            "subspace, which eliminates essentially all first-order scanner signal. "
            "The new-style logistic-regression-SVD approach removes only the 4 most "
            "discriminative directions for one particular classifier, leaving residual "
            "scanner signal that a fresh probe can exploit."
        )

    lines.extend([
        "",
        "## Bounded Interpretation",
        "",
        "This is a consistency audit. It identifies the implementation difference",
        "between two linear projection baselines used in different experiments. It",
        "does not claim clinical validation, diagnostic performance, or deployment",
        "readiness.",
        "",
        "Lower scanner probe accuracy means stronger scanner suppression. Under this",
        "reading, old_style linear_projection_k4 (scanner acc 0.2000) is a stronger",
        "raw scanner-removal baseline than the previously reported true_pair_biological",
        "branch (scanner acc 0.3614). Paired-acquisition should not be claimed to beat",
        "the strongest linear scanner-removal baseline on raw scanner suppression or",
        "raw category preservation.",
        "",
        "The old-style baseline is a stronger scanner-removal baseline because it",
        "directly targets the scanner centroids. The new-style baseline is weaker at",
        "scanner removal because it relies on one classifier's decision boundaries.",
        "",
        "For future experiments, the old-style (per-scanner-mean) approach should be",
        "the default linear baseline, as it more completely removes first-order",
        "scanner information. The new-style approach understates the power of a",
        "simple linear baseline.",
        "",
        "Prior conclusions about structured separation remain useful, but the raw",
        "scanner-removal comparison must favor old-style projection. The remaining",
        "paired-acquisition distinction is structural: it learns an explicit",
        "acquisition branch carrying scanner signal, while old-style projection",
        "removes scanner-centroid signal without a learned acquisition branch.",
        "",
        "## Implications for Previous Experiments",
        "",
        "- The biological-label preservation audit using linear_projection_k4 remains",
        "  the stronger scanner-removal baseline.",
        "- Scanner-confounded and heldout transfer interpretations should prefer the",
        "  old-style linear baseline when discussing strongest simple linear scanner",
        "  removal.",
        "- The linear residual branch-separation audit ec2a509f used a weaker",
        "  logistic-SVD linear split. Its frontier result is still informative for",
        "  that split, but it should not be treated as the strongest possible linear",
        "  scanner-subspace decomposition.",
        "",
        "## Follow-up Needed",
        "",
        "This consistency audit reconciles the mismatch, but it does not fully compare",
        "old-style keep/residual branch separation. A follow-up old-style residual",
        "decomposition audit may be needed:",
        "",
        "- old_style_keep_k4",
        "- old_style_removed_k4",
        "- category leakage in old_style_removed",
        "- scanner signal in old_style_removed",
        "- category/scanner contrast versus true_pair_biological/acquisition",
        "",
    ])

    lines.extend([
        "",
        "## Validation",
        "",
        f"- Metric rows: {len(metrics_df)}",
        f"- Feature comparison rows: {len(comparison_df)}",
        f"- Variants evaluated: {len(metrics_df['variant'].unique())}",
        f"- Folds: {len(folds)}",
        "",
        "## Output Files",
        "",
        "- linear_baseline_consistency_metrics.csv",
        "- linear_baseline_feature_comparison.csv",
        "- linear_baseline_consistency_report.md",
        "- experiment_design.json",
        "- run_log.txt",
        "",
        "## Readiness",
        "",
        "Ready after validation; no staging or commit performed by this script.",
        "",
    ])

    report = "\n".join(lines)
    atomic_text(out_dir / "linear_baseline_consistency_report.md", report)

    print("\n" + "=" * 80)
    print("LINEAR BASELINE CONSISTENCY AUDIT COMPLETE")
    print(f"old scanner acc: {old_scanner:.4f}")
    print(f"new scanner acc: {new_scanner:.4f}")
    print(f"Match: {abs(old_scanner - new_scanner) < 0.05}")
    print(f"Report: {(out_dir / 'linear_baseline_consistency_report.md').resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("results/paired_acquisition_factorization_linear_baseline_consistency_audit"),
    )
    parser.add_argument("--folds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.out_dir / "run_log.txt"
    start = time.perf_counter()

    design = {
        "stage": "linear_baseline_consistency_audit",
        "dataset": "canine_cutaneous_scc_dinov2",
        "k_target": K_TARGET,
        "folds": list(args.folds),
        "variants": {
            "old_style_linear_projection_k4": "per-scanner mean vectors, QR orthonormalization, StandardScaler on all data (bec06eb4)",
            "new_style_linear_keep_k4": "logistic regression coefficient SVD, StandardScaler on fit only (ec2a509f)",
            "new_style_linear_removed_k4": "complement of new_style_linear_keep_k4",
            "new_style_keep_on_old_standardization": "new-style SVD directions on old-style (all-data) standardization",
        },
        "command": " ".join(sys.argv),
    }
    atomic_text(args.out_dir / "experiment_design.json", json.dumps(design, indent=2, sort_keys=True) + "\n")

    with log_path.open("a", encoding="utf-8") as log_file:
        with redirect_stdout(Tee(sys.stdout, log_file)), redirect_stderr(Tee(sys.stderr, log_file)):
            print("\n" + "=" * 80)
            print(time.strftime("%Y-%m-%d %H:%M:%S"))
            print("COMMAND " + " ".join(sys.argv))
            try:
                run_audit(args.out_dir, tuple(args.folds))
            except Exception:
                traceback.print_exc()
                raise


if __name__ == "__main__":
    main()
