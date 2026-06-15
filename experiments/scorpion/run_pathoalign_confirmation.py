#!/usr/bin/env python3
"""Confirm frozen SCORPION PathoAlign dep20 against paired consistency.

The objective was selected with seeds 401-403. This script uses independent
seeds 501-510, fixed hyperparameters, the same 75-epoch schedule, and train/val
only. Test rows are never projected.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from experiments.scorpion.run_pathoalign_projection import (
    ExperimentError,
    indices_for,
    load_archive,
    region_groups,
    standardize,
    train_one,
    validate_splits,
    write_results,
)
from src.models.scorpion_pathoalign import ProjectionConfig


SEEDS = tuple(range(501, 511))
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--region-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    features, frame, source_metadata = load_archive(args.features)
    validate_splits(frame)
    train_indices = indices_for(frame, "train")
    val_indices = indices_for(frame, "val")
    development_indices = np.concatenate([train_indices, val_indices])
    if np.any(frame.iloc[development_indices]["split"].to_numpy() == "test"):
        raise ExperimentError("Test rows entered the confirmation set.")

    transformed, mean, std = standardize(features, train_indices)
    groups = region_groups(frame, train_indices)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ExperimentError("CUDA requested but unavailable.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_dir / "train_standardization.npz", mean=mean, std=std)
    design = {
        "stage": "independent_seed_confirmation",
        "source_features": str(args.features.resolve()),
        "source_metadata": source_metadata,
        "variants": VARIANTS,
        "seeds": list(SEEDS),
        "calibration_seeds_excluded": [401, 402, 403],
        "epochs": args.epochs,
        "region_batch_size": args.region_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "n_train_slides": int(frame.iloc[train_indices]["slide_id"].nunique()),
        "n_val_slides": int(frame.iloc[val_indices]["slide_id"].nunique()),
        "n_test_rows_processed": 0,
        "device": str(device),
    }
    (args.out_dir / "confirmation_design.json").write_text(
        json.dumps(design, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        for variant_name, variant in VARIANTS.items():
            config = ProjectionConfig(
                input_dim=features.shape[1],
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
            run_dir = args.out_dir / "runs" / f"{variant_name}_seed_{seed}"
            result = train_one(
                method=str(variant["method"]),
                seed=seed,
                features=transformed,
                frame=frame,
                train_indices=train_indices,
                development_indices=development_indices,
                groups=groups,
                config=config,
                device=device,
                epochs=args.epochs,
                region_batch_size=args.region_batch_size,
                learning_rate=args.learning_rate,
                weight_decay=args.weight_decay,
                run_dir=run_dir,
            )
            rows.append({"variant": variant_name, **result, **asdict(config)})
            write_results(args.out_dir / "confirmation_training_results.csv", rows)

    expected = len(SEEDS) * len(VARIANTS)
    if len(rows) != expected:
        raise ExperimentError(f"Expected {expected} fits, observed {len(rows)}")
    table = pd.DataFrame(rows)
    if table.duplicated(["variant", "seed"]).any():
        raise ExperimentError("Duplicate variant/seed rows found.")

    print("SCORPION PATHOALIGN CONFIRMATION TRAINING PASSED")
    print(table.groupby("variant").mean(numeric_only=True).to_string())
    print(f"Results: {(args.out_dir / 'confirmation_training_results.csv').resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (ExperimentError, OSError, RuntimeError) as exc:
        print(f"SCORPION PATHOALIGN CONFIRMATION FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
