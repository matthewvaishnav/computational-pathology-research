#!/usr/bin/env python3
"""Run frozen PathoAlign across all five canine SCC sample-blocked folds.

Fold-0 validation established that the locked dep20 objective transfers to the
external canine SCC benchmark. This stage now performs frozen five-fold testing:
for each fold, all non-test samples are used for fitting and the held-out fold
is projected exactly once. No hyperparameter selection occurs here.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.scorpion import run_pathoalign_projection as projection
from src.models.scorpion_pathoalign import ProjectionConfig


CANINE_SCANNERS = ("cs2", "gt450", "nz20", "nz210", "p1000")
FOLDS = tuple(range(5))
SEEDS = tuple(range(911, 916))
KEY_COLUMNS = ("slide_id", "region_id", "scanner_id", "path")
VARIANTS = {
    "paired_reference": {
        "method": "paired_consistency",
        "scanner_adversary_weight": 0.0,
        "scanner_acquisition_weight": 0.0,
        "scanner_dependence_weight": 0.0,
        "cross_covariance_weight": 0.0,
        "gradient_reversal_strength": 0.0,
    },
    "pathoalign_dep20": {
        "method": "pathoalign",
        "scanner_adversary_weight": 0.5,
        "scanner_acquisition_weight": 0.5,
        "scanner_dependence_weight": 20.0,
        "cross_covariance_weight": 0.05,
        "gradient_reversal_strength": 1.0,
    },
}


def patch_scanner_namespace() -> None:
    projection.SCANNERS = CANINE_SCANNERS
    projection.SCANNER_TO_INDEX = {
        scanner: index for index, scanner in enumerate(CANINE_SCANNERS)
    }


def row_keys(frame: pd.DataFrame) -> list[tuple[str, ...]]:
    return [tuple(str(row[column]) for column in KEY_COLUMNS) for _, row in frame.iterrows()]


def align_fold(
    base_features: np.ndarray,
    base_frame: pd.DataFrame,
    manifest_path: Path,
) -> tuple[np.ndarray, pd.DataFrame]:
    manifest = pd.read_csv(manifest_path, dtype=str)
    missing = [column for column in (*KEY_COLUMNS, "split") if column not in manifest.columns]
    if missing:
        raise projection.ExperimentError(f"{manifest_path} is missing columns: {missing}")
    manifest["scanner_id"] = manifest["scanner_id"].astype(str).str.lower()
    base_frame = base_frame.copy()
    base_frame["scanner_id"] = base_frame["scanner_id"].astype(str).str.lower()
    if manifest.duplicated(list(KEY_COLUMNS)).any():
        raise projection.ExperimentError(f"Duplicate manifest keys in {manifest_path}")

    base_lookup = {key: index for index, key in enumerate(row_keys(base_frame))}
    manifest_keys = row_keys(manifest)
    if set(manifest_keys) != set(base_lookup):
        missing_from_manifest = len(set(base_lookup) - set(manifest_keys))
        missing_from_features = len(set(manifest_keys) - set(base_lookup))
        raise projection.ExperimentError(
            f"Feature/manifest key mismatch for {manifest_path}: "
            f"missing_from_manifest={missing_from_manifest}, "
            f"missing_from_features={missing_from_features}"
        )
    order = np.asarray([base_lookup[key] for key in manifest_keys], dtype=np.int64)
    aligned_features = base_features[order]
    keep_columns = [column for column in (*KEY_COLUMNS, "split") if column in manifest.columns]
    aligned_frame = manifest.loc[:, keep_columns].reset_index(drop=True)
    if len(aligned_features) != 4025:
        raise projection.ExperimentError(
            f"Expected 4,025 aligned rows, observed {len(aligned_features)}"
        )
    if aligned_frame["region_id"].nunique() != 805:
        raise projection.ExperimentError("Expected 805 geometry-qualified regions")
    return aligned_features, aligned_frame


def validate_fold(frame: pd.DataFrame, fold: int) -> tuple[np.ndarray, np.ndarray]:
    frame["scanner_id"] = frame["scanner_id"].astype(str).str.lower()
    if set(frame["scanner_id"].unique()) != set(CANINE_SCANNERS):
        raise projection.ExperimentError("Unexpected scanner set")
    if not (frame.groupby("region_id").size() == len(CANINE_SCANNERS)).all():
        raise projection.ExperimentError("At least one region is missing a scanner view")
    test_indices = np.flatnonzero(frame["split"].to_numpy() == "test")
    fit_indices = np.flatnonzero(frame["split"].to_numpy() != "test")
    if len(test_indices) == 0 or len(fit_indices) == 0:
        raise projection.ExperimentError(f"Fold {fold} has an empty fit or test set.")
    fit_samples = set(frame.iloc[fit_indices]["slide_id"])
    test_samples = set(frame.iloc[test_indices]["slide_id"])
    overlap = sorted(fit_samples & test_samples)
    if overlap:
        raise projection.ExperimentError(f"Fold {fold} has sample leakage: {overlap[:20]}")
    if len(fit_samples | test_samples) != 44:
        raise projection.ExperimentError(f"Fold {fold} does not cover all 44 samples.")
    return fit_indices, test_indices


def config_for(input_dim: int, variant: dict[str, float | str]) -> ProjectionConfig:
    return ProjectionConfig(
        input_dim=input_dim,
        biological_dim=256,
        acquisition_dim=64,
        hidden_dim=512,
        temperature=0.1,
        reconstruction_weight=1.0,
        variance_weight=1.0,
        covariance_weight=0.01,
        scanner_adversary_weight=float(variant["scanner_adversary_weight"]),
        scanner_acquisition_weight=float(variant["scanner_acquisition_weight"]),
        scanner_dependence_weight=float(variant["scanner_dependence_weight"]),
        cross_covariance_weight=float(variant["cross_covariance_weight"]),
        gradient_reversal_strength=float(variant["gradient_reversal_strength"]),
    )


def load_existing(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    return pd.read_csv(path).to_dict("records")


def write_results(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    pd.DataFrame(rows).sort_values(["fold", "variant", "seed"]).to_csv(path, index=False)


def mark_frozen_test_projection(path: Path, fold: int) -> None:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    metadata.update(
        {
            "source": "External canine SCC DINOv2 frozen five-fold PathoAlign test",
            "evaluation_stage": "external_canine_frozen_five_fold_test",
            "contains_test_rows": True,
            "fold": int(fold),
            "fit_splits": ["train", "val"],
            "evaluation_split": "test",
            "hyperparameters_frozen": True,
            "scanner_namespace": list(CANINE_SCANNERS),
        }
    )
    text = json.dumps(metadata, sort_keys=True)
    arrays["metadata_json"] = np.asarray(text, dtype=f"<U{len(text)}")
    projection.atomic_npz(path, arrays)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-features", type=Path, required=True)
    parser.add_argument("--manifests-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--region-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.epochs != 75 or args.region_batch_size != 32:
        raise projection.ExperimentError("Use the frozen 75-epoch / batch-32 schedule.")
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    patch_scanner_namespace()

    base_features, base_frame, source_metadata = projection.load_archive(args.base_features)
    base_frame["scanner_id"] = base_frame["scanner_id"].astype(str).str.lower()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise projection.ExperimentError("CUDA requested but unavailable.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.out_dir / "crossfold_training_results.csv"
    rows = load_existing(results_path)
    completed = {
        (int(row["fold"]), str(row["variant"]), int(row["seed"])) for row in rows
    }

    design = {
        "stage": "external_canine_frozen_five_fold_test",
        "base_features": str(args.base_features.resolve()),
        "source_metadata": source_metadata,
        "manifest_directory": str(args.manifests_dir.resolve()),
        "scanners": list(CANINE_SCANNERS),
        "folds": list(FOLDS),
        "seeds": list(args.seeds),
        "variants": VARIANTS,
        "fold0_validation_seeds_excluded": list(range(901, 906)),
        "epochs": args.epochs,
        "region_batch_size": args.region_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "hyperparameters_frozen": True,
        "device": str(device),
    }
    (args.out_dir / "crossfold_design.json").write_text(
        json.dumps(design, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    for fold in FOLDS:
        manifest_path = args.manifests_dir / f"fold_{fold}_patch_manifest.csv"
        features, frame = align_fold(base_features, base_frame, manifest_path)
        fit_indices, test_indices = validate_fold(frame, fold)
        transformed, mean, std = projection.standardize(features, fit_indices)
        groups = projection.region_groups(frame, fit_indices)
        fold_dir = args.out_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(fold_dir / "fit_standardization.npz", mean=mean, std=std)

        for seed in args.seeds:
            for variant_name, variant in VARIANTS.items():
                key = (fold, variant_name, seed)
                if key in completed:
                    print(f"Skipping completed fold={fold} variant={variant_name} seed={seed}")
                    continue
                config = config_for(features.shape[1], variant)
                run_dir = fold_dir / "runs" / f"{variant_name}_seed_{seed}"
                result = projection.train_one(
                    method=str(variant["method"]),
                    seed=seed,
                    features=transformed,
                    frame=frame,
                    train_indices=fit_indices,
                    development_indices=np.arange(len(frame), dtype=np.int64),
                    groups=groups,
                    config=config,
                    device=device,
                    epochs=args.epochs,
                    region_batch_size=args.region_batch_size,
                    learning_rate=args.learning_rate,
                    weight_decay=args.weight_decay,
                    run_dir=run_dir,
                )
                projected = run_dir / "projected_features.npz"
                mark_frozen_test_projection(projected, fold)
                rows.append(
                    {
                        "fold": fold,
                        "variant": variant_name,
                        **result,
                        **asdict(config),
                        "n_fit_samples": int(frame.iloc[fit_indices]["slide_id"].nunique()),
                        "n_test_samples": int(frame.iloc[test_indices]["slide_id"].nunique()),
                        "n_fit_regions": int(frame.iloc[fit_indices]["region_id"].nunique()),
                        "n_test_regions": int(frame.iloc[test_indices]["region_id"].nunique()),
                    }
                )
                write_results(results_path, rows)
                completed.add(key)

    expected = len(FOLDS) * len(args.seeds) * len(VARIANTS)
    table = pd.read_csv(results_path)
    if len(table) != expected:
        raise projection.ExperimentError(f"Expected {expected} completed fits, observed {len(table)}")
    if table.duplicated(["fold", "variant", "seed"]).any():
        raise projection.ExperimentError("Duplicate fold/variant/seed rows found.")

    print("CANINE SCC FROZEN FIVE-FOLD TRAINING PASSED")
    print(f"Completed fits: {len(table)}")
    print(f"Results: {results_path.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (projection.ExperimentError, OSError, RuntimeError) as exc:
        print(f"CANINE SCC CROSS-FOLD TRAINING FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
