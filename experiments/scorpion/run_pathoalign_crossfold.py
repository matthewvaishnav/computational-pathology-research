#!/usr/bin/env python3
"""Run frozen PathoAlign across all five SCORPION test folds.

The dep20 objective and schedule are locked from fold-0 validation work. For each
rotating fold, all non-test slides are used for fitting and that fold's test
slides are projected exactly once. Five new matched seeds (601-605) are used.
No hyperparameter selection or checkpoint selection occurs in this stage.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.scorpion.run_pathoalign_projection import (
    ExperimentError,
    load_archive,
    region_groups,
    standardize,
    train_one,
    write_results,
)
from src.models.scorpion_pathoalign import ProjectionConfig


SEEDS = tuple(range(601, 606))
FOLDS = tuple(range(5))
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
        raise ExperimentError(f"{manifest_path} is missing columns: {missing}")
    if manifest.duplicated(list(KEY_COLUMNS)).any():
        raise ExperimentError(f"Duplicate manifest keys in {manifest_path}")

    base_lookup = {key: index for index, key in enumerate(row_keys(base_frame))}
    manifest_keys = row_keys(manifest)
    if set(manifest_keys) != set(base_lookup):
        missing_from_manifest = len(set(base_lookup) - set(manifest_keys))
        missing_from_features = len(set(manifest_keys) - set(base_lookup))
        raise ExperimentError(
            f"Feature/manifest key mismatch for {manifest_path}: "
            f"missing_from_manifest={missing_from_manifest}, "
            f"missing_from_features={missing_from_features}"
        )
    order = np.asarray([base_lookup[key] for key in manifest_keys], dtype=np.int64)
    aligned_features = base_features[order]
    aligned_frame = manifest.loc[:, [*KEY_COLUMNS, "split"]].reset_index(drop=True)
    if len(aligned_features) != 2400:
        raise ExperimentError(
            f"Expected 2,400 aligned rows, observed {len(aligned_features)}"
        )
    return aligned_features, aligned_frame


def validate_fold(frame: pd.DataFrame, fold: int) -> tuple[np.ndarray, np.ndarray]:
    test_indices = np.flatnonzero(frame["split"].to_numpy() == "test")
    fit_indices = np.flatnonzero(frame["split"].to_numpy() != "test")
    if len(test_indices) == 0 or len(fit_indices) == 0:
        raise ExperimentError(f"Fold {fold} has an empty fit or test set.")
    fit_slides = set(frame.iloc[fit_indices]["slide_id"])
    test_slides = set(frame.iloc[test_indices]["slide_id"])
    overlap = sorted(fit_slides & test_slides)
    if overlap:
        raise ExperimentError(f"Fold {fold} has slide leakage: {overlap[:20]}")
    if len(fit_slides | test_slides) != 48:
        raise ExperimentError(f"Fold {fold} does not cover all 48 slides.")
    return fit_indices, test_indices


def atomic_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".npz", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        np.savez_compressed(temporary, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def mark_frozen_test_projection(path: Path, fold: int) -> None:
    with np.load(path, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    metadata = json.loads(str(arrays["metadata_json"].item()))
    metadata.update(
        {
            "contains_test_rows": True,
            "evaluation_stage": "frozen_five_fold_test",
            "fold": int(fold),
            "fit_splits": ["train", "val"],
            "evaluation_split": "test",
            "hyperparameters_frozen": True,
        }
    )
    text = json.dumps(metadata, sort_keys=True)
    arrays["metadata_json"] = np.asarray(text, dtype=f"<U{len(text)}")
    atomic_npz(path, arrays)


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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-features", type=Path, required=True)
    parser.add_argument("--manifests-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--region-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    base_features, base_frame, source_metadata = load_archive(args.base_features)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ExperimentError("CUDA requested but unavailable.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    results_path = args.out_dir / "crossfold_training_results.csv"
    rows = load_existing(results_path)
    completed = {
        (int(row["fold"]), str(row["variant"]), int(row["seed"])) for row in rows
    }

    design = {
        "stage": "frozen_five_fold_test",
        "base_features": str(args.base_features.resolve()),
        "source_metadata": source_metadata,
        "manifest_directory": str(args.manifests_dir.resolve()),
        "folds": list(FOLDS),
        "seeds": list(SEEDS),
        "variants": VARIANTS,
        "calibration_seeds_excluded": [401, 402, 403],
        "confirmation_seeds_excluded": list(range(501, 511)),
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
        manifest_path = args.manifests_dir / f"fold_{fold}_manifest.csv"
        features, frame = align_fold(base_features, base_frame, manifest_path)
        fit_indices, test_indices = validate_fold(frame, fold)
        transformed, mean, std = standardize(features, fit_indices)
        groups = region_groups(frame, fit_indices)
        fold_dir = args.out_dir / f"fold_{fold}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(fold_dir / "fit_standardization.npz", mean=mean, std=std)

        for seed in SEEDS:
            for variant_name, variant in VARIANTS.items():
                key = (fold, variant_name, seed)
                if key in completed:
                    print(f"Skipping completed fold={fold} variant={variant_name} seed={seed}")
                    continue
                config = config_for(features.shape[1], variant)
                run_dir = fold_dir / "runs" / f"{variant_name}_seed_{seed}"
                result = train_one(
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
                        "n_fit_slides": int(frame.iloc[fit_indices]["slide_id"].nunique()),
                        "n_test_slides": int(frame.iloc[test_indices]["slide_id"].nunique()),
                    }
                )
                write_results(results_path, rows)
                completed.add(key)

    expected = len(FOLDS) * len(SEEDS) * len(VARIANTS)
    table = pd.DataFrame(rows)
    if len(table) != expected:
        raise ExperimentError(f"Expected {expected} completed fits, observed {len(table)}")
    if table.duplicated(["fold", "variant", "seed"]).any():
        raise ExperimentError("Duplicate fold/variant/seed rows found.")

    print("SCORPION FROZEN FIVE-FOLD TRAINING PASSED")
    print(f"Completed fits: {len(table)}")
    print(f"Results: {results_path.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (ExperimentError, OSError, RuntimeError) as exc:
        print(f"SCORPION CROSS-FOLD TRAINING FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
