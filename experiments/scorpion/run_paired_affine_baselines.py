#!/usr/bin/env python3
"""Run prospective paired affine baselines on the frozen SCORPION folds.

This runner is deliberately separate from the promoted neural-factorization
evidence. It implements deterministic, source-to-reference embedding maps that
answer the paired affine harmonization question:

    can a single global map make one scanner's embeddings resemble a reference
    scanner's embeddings when exact same-region pairs are available?

All five scanners are used as references in turn. Ridge hyperparameters are
selected using train/validation slides only and are then refit on all non-test
slides. No test row is used to estimate a transform or choose a hyperparameter.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

SCANNERS = ("AT2", "B300", "DP200", "GT450", "P1000")
FOLDS = tuple(range(5))
VARIANTS = (
    "identity_standardized",
    "centroid_translation",
    "orthogonal_procrustes",
    "affine_least_squares",
    "ridge_affine",
)
RIDGE_ALPHAS = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0)
SCHEMA_VERSION = "scorpion-paired-affine-baselines/v1"


class BaselineError(ValueError):
    pass


@dataclass(frozen=True)
class AffineMap:
    source_mean: np.ndarray
    target_mean: np.ndarray
    matrix: np.ndarray

    def apply(self, values: np.ndarray) -> np.ndarray:
        projected = (values - self.source_mean) @ self.matrix + self.target_mean
        if not np.isfinite(projected).all():
            raise BaselineError("Affine projection produced non-finite values.")
        return projected


def fit_global_standardization(
    features: np.ndarray, fit_indices: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Use only non-test rows to put all compared methods on one input scale."""
    fit = np.asarray(features[fit_indices], dtype=np.float64)
    mean = fit.mean(axis=0, keepdims=True)
    std = fit.std(axis=0, keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    transformed = ((np.asarray(features, dtype=np.float64) - mean) / std).astype(
        np.float32
    )
    if not np.isfinite(transformed).all():
        raise BaselineError("Fit-only standardization produced non-finite values.")
    return transformed, mean.astype(np.float32), std.astype(np.float32)


def paired_matrices(
    features: np.ndarray,
    frame: pd.DataFrame,
    indices: np.ndarray,
    source_scanner: str,
    target_scanner: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Return same-region source and target matrices in a deterministic order."""
    if source_scanner == target_scanner:
        raise BaselineError("Source and target scanners must differ.")
    subset = frame.iloc[np.asarray(indices, dtype=np.int64)].copy()
    subset["_row_index"] = np.asarray(indices, dtype=np.int64)
    if subset.duplicated(["region_id", "scanner_id"]).any():
        raise BaselineError("Duplicate region/scanner rows entered paired fitting.")

    source = subset.loc[subset["scanner_id"] == source_scanner].set_index("region_id")
    target = subset.loc[subset["scanner_id"] == target_scanner].set_index("region_id")
    source_regions = set(source.index.astype(str))
    target_regions = set(target.index.astype(str))
    if source_regions != target_regions or not source_regions:
        raise BaselineError(
            f"Incomplete {source_scanner}->{target_scanner} pairs: "
            f"source={len(source_regions)}, target={len(target_regions)}"
        )
    regions = sorted(source_regions)
    source_rows = source.loc[regions, "_row_index"].to_numpy(dtype=np.int64)
    target_rows = target.loc[regions, "_row_index"].to_numpy(dtype=np.int64)
    x = np.asarray(features[source_rows], dtype=np.float64)
    y = np.asarray(features[target_rows], dtype=np.float64)
    if (
        x.shape != y.shape
        or x.ndim != 2
        or not np.isfinite(x).all()
        or not np.isfinite(y).all()
    ):
        raise BaselineError("Invalid paired matrices.")
    return x, y


def fit_centroid_translation(source: np.ndarray, target: np.ndarray) -> AffineMap:
    dimension = source.shape[1]
    return AffineMap(
        source_mean=source.mean(axis=0, keepdims=True),
        target_mean=target.mean(axis=0, keepdims=True),
        matrix=np.eye(dimension, dtype=np.float64),
    )


def fit_orthogonal_procrustes(source: np.ndarray, target: np.ndarray) -> AffineMap:
    source_mean = source.mean(axis=0, keepdims=True)
    target_mean = target.mean(axis=0, keepdims=True)
    cross = (source - source_mean).T @ (target - target_mean)
    left, _, right_t = np.linalg.svd(cross, full_matrices=False)
    matrix = left @ right_t
    return AffineMap(source_mean, target_mean, matrix)


def fit_affine(
    source: np.ndarray,
    target: np.ndarray,
    *,
    alpha: float,
) -> AffineMap:
    """Fit a centered global affine map using a stable primal/dual solution."""
    if alpha < 0:
        raise BaselineError("Ridge alpha cannot be negative.")
    source_mean = source.mean(axis=0, keepdims=True)
    target_mean = target.mean(axis=0, keepdims=True)
    x = source - source_mean
    y = target - target_mean
    if alpha == 0:
        matrix = np.linalg.pinv(x, rcond=1e-8) @ y
    elif len(x) <= x.shape[1]:
        gram = x @ x.T
        matrix = x.T @ np.linalg.solve(
            gram + alpha * np.eye(len(gram), dtype=np.float64),
            y,
        )
    else:
        gram = x.T @ x
        matrix = np.linalg.solve(
            gram + alpha * np.eye(len(gram), dtype=np.float64),
            x.T @ y,
        )
    if matrix.shape != (source.shape[1], target.shape[1]):
        raise BaselineError("Affine fit returned an unexpected matrix shape.")
    return AffineMap(source_mean, target_mean, matrix)


def select_ridge_alpha(
    features: np.ndarray,
    frame: pd.DataFrame,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    source_scanner: str,
    reference_scanner: str,
    candidates: Iterable[float] = RIDGE_ALPHAS,
) -> tuple[float, list[dict[str, float]]]:
    """Select alpha by paired validation MSE without looking at test rows."""
    train_source, train_target = paired_matrices(
        features,
        frame,
        train_indices,
        source_scanner,
        reference_scanner,
    )
    validation_source, validation_target = paired_matrices(
        features,
        frame,
        validation_indices,
        source_scanner,
        reference_scanner,
    )
    rows: list[dict[str, float]] = []
    for candidate in candidates:
        alpha = float(candidate)
        model = fit_affine(train_source, train_target, alpha=alpha)
        error = float(
            np.mean((model.apply(validation_source) - validation_target) ** 2)
        )
        if not np.isfinite(error):
            raise BaselineError(
                "Ridge selection produced a non-finite validation error."
            )
        rows.append({"alpha": alpha, "validation_pair_mse": error})
    rows.sort(key=lambda row: (row["validation_pair_mse"], row["alpha"]))
    return float(rows[0]["alpha"]), rows


def fit_variant(
    variant: str,
    source: np.ndarray,
    target: np.ndarray,
    *,
    ridge_alpha: float | None = None,
) -> AffineMap:
    if variant == "centroid_translation":
        return fit_centroid_translation(source, target)
    if variant == "orthogonal_procrustes":
        return fit_orthogonal_procrustes(source, target)
    if variant == "affine_least_squares":
        return fit_affine(source, target, alpha=0.0)
    if variant == "ridge_affine":
        if ridge_alpha is None:
            raise BaselineError("ridge_affine requires a selected alpha.")
        return fit_affine(source, target, alpha=ridge_alpha)
    raise BaselineError(f"Unknown fitted variant: {variant}")


def harmonize(
    features: np.ndarray,
    frame: pd.DataFrame,
    fit_indices: np.ndarray,
    reference_scanner: str,
    variant: str,
    *,
    ridge_alphas: dict[str, float] | None = None,
) -> np.ndarray:
    """Fit on non-test pairs and transform every source scanner into the reference."""
    if reference_scanner not in SCANNERS:
        raise BaselineError(f"Unknown reference scanner: {reference_scanner}")
    if variant not in VARIANTS:
        raise BaselineError(f"Unknown variant: {variant}")
    output = np.asarray(features, dtype=np.float64).copy()
    if variant == "identity_standardized":
        return output.astype(np.float32)

    labels = frame["scanner_id"].astype(str).to_numpy()
    for source_scanner in SCANNERS:
        if source_scanner == reference_scanner:
            continue
        source, target = paired_matrices(
            features,
            frame,
            fit_indices,
            source_scanner,
            reference_scanner,
        )
        alpha = None if ridge_alphas is None else ridge_alphas.get(source_scanner)
        model = fit_variant(variant, source, target, ridge_alpha=alpha)
        rows = np.flatnonzero(labels == source_scanner)
        output[rows] = model.apply(output[rows])
    if not np.isfinite(output).all():
        raise BaselineError("Harmonized representation contains non-finite values.")
    return output.astype(np.float32)


def _split_indices(frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    train = np.flatnonzero(frame["split"].astype(str).to_numpy() == "train")
    validation = np.flatnonzero(frame["split"].astype(str).to_numpy() == "val")
    test = np.flatnonzero(frame["split"].astype(str).to_numpy() == "test")
    if not len(train) or not len(validation) or not len(test):
        raise BaselineError(
            "Every fold requires non-empty train, validation, and test rows."
        )
    train_slides = set(frame.iloc[train]["slide_id"])
    validation_slides = set(frame.iloc[validation]["slide_id"])
    test_slides = set(frame.iloc[test]["slide_id"])
    if (
        train_slides & validation_slides
        or (train_slides | validation_slides) & test_slides
    ):
        raise BaselineError("Slide leakage detected across train/validation/test.")
    return train, validation, test


def run(args: argparse.Namespace) -> dict[str, object]:
    from experiments.scorpion.run_pathoalign_projection import atomic_npz, string_array

    from experiments.scorpion import run_pathoalign_capacity_matched_ablations as frozen

    root = frozen.repository_root()
    _, _, source_metadata, fold_data = frozen.validate_inputs(
        args.base_features.resolve(),
        args.manifests_dir.resolve(),
        root,
    )
    source = frozen.source_state(root)
    if not source["tracked_worktree_clean"]:
        raise BaselineError(
            "Refusing evidence execution from a tracked dirty worktree. "
            "Commit the reviewed comparison implementation first."
        )
    design = {
        "schema_version": SCHEMA_VERSION,
        "status": "prospective_comparison_execution",
        "source": source,
        "source_metadata": source_metadata,
        "folds": list(FOLDS),
        "reference_scanners": list(SCANNERS),
        "variants": list(VARIANTS),
        "ridge_alphas": list(RIDGE_ALPHAS),
        "ridge_selection": (
            "per source/reference map; minimum paired validation MSE; "
            "tie broken toward smaller alpha; refit on train+validation"
        ),
        "test_usage": "evaluation only",
        "primary_aggregation": "average the five reference-scanner outcomes within slide",
        "claim_boundaries": [
            "These are harmonization baselines, not explicit factorization models.",
            "Same-region retrieval and cosine agreement do not prove biological preservation.",
            "Five reference scanners are sensitivity conditions, not independent samples.",
            "No result is promoted until the complete output is validated and separately released.",
        ],
    }
    args.out_dir.mkdir(parents=True, exist_ok=True)
    frozen.ensure_immutable_json(args.out_dir / "comparison_design.json", design)

    completed = 0
    selection_rows: list[dict[str, object]] = []
    for fold in FOLDS:
        features, frame = fold_data[fold]
        train_indices, validation_indices, test_indices = _split_indices(frame)
        fit_indices = np.concatenate([train_indices, validation_indices])
        transformed, mean, std = fit_global_standardization(features, fit_indices)
        fold_dir = args.out_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(fold_dir / "fit_standardization.npz", mean=mean, std=std)

        for reference_scanner in SCANNERS:
            ridge_alphas: dict[str, float] = {}
            for source_scanner in SCANNERS:
                if source_scanner == reference_scanner:
                    continue
                alpha, trace = select_ridge_alpha(
                    transformed,
                    frame,
                    train_indices,
                    validation_indices,
                    source_scanner,
                    reference_scanner,
                )
                ridge_alphas[source_scanner] = alpha
                for row in trace:
                    selection_rows.append(
                        {
                            "fold": fold,
                            "source_scanner": source_scanner,
                            "reference_scanner": reference_scanner,
                            "selected": row["alpha"] == alpha,
                            **row,
                        }
                    )

            for variant in VARIANTS:
                projected = harmonize(
                    transformed,
                    frame,
                    fit_indices,
                    reference_scanner,
                    variant,
                    ridge_alphas=ridge_alphas,
                )
                metadata = {
                    "schema_version": SCHEMA_VERSION,
                    "fold": fold,
                    "variant": variant,
                    "reference_scanner": reference_scanner,
                    "fit_splits": ["train", "val"],
                    "evaluation_split": "test",
                    "contains_test_rows": True,
                    "transform_estimation_uses_test_rows": False,
                    "ridge_alphas": ridge_alphas if variant == "ridge_affine" else None,
                }
                text = json.dumps(metadata, sort_keys=True)
                arrays = {
                    "features": projected,
                    "slide_id": string_array(frame["slide_id"]),
                    "region_id": string_array(frame["region_id"]),
                    "scanner_id": string_array(frame["scanner_id"]),
                    "split": string_array(frame["split"]),
                    "path": string_array(frame["path"]),
                    "metadata_json": np.asarray(text, dtype=f"<U{len(text)}"),
                }
                output = (
                    fold_dir
                    / f"reference_{reference_scanner}"
                    / variant
                    / "projected_features.npz"
                )
                atomic_npz(output, arrays)
                completed += 1

        if set(frame.iloc[test_indices]["scanner_id"]) != set(SCANNERS):
            raise BaselineError(f"Fold {fold} test rows do not cover all scanners.")

    expected = len(FOLDS) * len(SCANNERS) * len(VARIANTS)
    if completed != expected:
        raise BaselineError(f"Expected {expected} projections, observed {completed}.")
    pd.DataFrame(selection_rows).sort_values(
        ["fold", "reference_scanner", "source_scanner", "alpha"]
    ).to_csv(args.out_dir / "ridge_selection.csv", index=False)
    summary = {
        "schema_version": SCHEMA_VERSION,
        "status": "complete",
        "expected_projections": expected,
        "completed_projections": completed,
        "ridge_selection_rows": len(selection_rows),
    }
    frozen.ensure_immutable_json(args.out_dir / "execution_summary.json", summary)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-features", type=Path, required=True)
    parser.add_argument("--manifests-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    print(json.dumps(run(parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    try:
        main()
    except (
        BaselineError,
        OSError,
        RuntimeError,
        ValueError,
        np.linalg.LinAlgError,
    ) as exc:
        print(f"SCORPION PAIRED AFFINE BASELINES FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
