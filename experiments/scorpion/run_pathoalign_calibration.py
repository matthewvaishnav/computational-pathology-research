#!/usr/bin/env python3
"""Run a bounded train/validation-only PathoAlign objective calibration.

This is not an unrestricted hyperparameter search. It compares one fixed paired
consistency reference against four predefined PathoAlign settings over three
matched seeds. Test rows are never projected.
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


DEFAULT_SEEDS = (401, 402, 403)

VARIANTS = {
    "paired_reference": {
        "method": "paired_consistency",
        "scanner_adversary_weight": 0.0,
        "scanner_acquisition_weight": 0.0,
        "scanner_dependence_weight": 0.0,
        "cross_covariance_weight": 0.0,
        "gradient_reversal_strength": 0.0,
    },
    "pathoalign_dep1": {
        "method": "pathoalign",
        "scanner_adversary_weight": 0.5,
        "scanner_acquisition_weight": 0.5,
        "scanner_dependence_weight": 1.0,
        "cross_covariance_weight": 0.05,
        "gradient_reversal_strength": 1.0,
    },
    "pathoalign_dep5": {
        "method": "pathoalign",
        "scanner_adversary_weight": 0.5,
        "scanner_acquisition_weight": 0.5,
        "scanner_dependence_weight": 5.0,
        "cross_covariance_weight": 0.05,
        "gradient_reversal_strength": 1.0,
    },
    "pathoalign_dep20": {
        "method": "pathoalign",
        "scanner_adversary_weight": 0.5,
        "scanner_acquisition_weight": 0.5,
        "scanner_dependence_weight": 20.0,
        "cross_covariance_weight": 0.05,
        "gradient_reversal_strength": 1.0,
    },
    "pathoalign_dep5_strong_sep": {
        "method": "pathoalign",
        "scanner_adversary_weight": 1.0,
        "scanner_acquisition_weight": 0.5,
        "scanner_dependence_weight": 5.0,
        "cross_covariance_weight": 0.20,
        "gradient_reversal_strength": 2.0,
    },
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    parser.add_argument("--epochs", type=int, default=75)
    parser.add_argument("--region-batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--biological-dim", type=int, default=256)
    parser.add_argument("--acquisition-dim", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    args = parser.parse_args()

    if len(set(args.seeds)) != len(args.seeds):
        raise SystemExit("Seeds must be unique.")
    if args.epochs <= 0 or args.region_batch_size <= 1:
        raise SystemExit("Epochs must be positive and region batch size > 1.")

    features, frame, source_metadata = load_archive(args.features)
    validate_splits(frame)
    train_indices = indices_for(frame, "train")
    val_indices = indices_for(frame, "val")
    development_indices = np.concatenate([train_indices, val_indices])
    if np.any(frame.iloc[development_indices]["split"].to_numpy() == "test"):
        raise ExperimentError("Test rows entered the calibration set.")

    transformed, mean, std = standardize(features, train_indices)
    groups = region_groups(frame, train_indices)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ExperimentError("CUDA requested but unavailable.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.out_dir / "train_standardization.npz", mean=mean, std=std
    )
    design = {
        "source_features": str(args.features.resolve()),
        "source_metadata": source_metadata,
        "variants": VARIANTS,
        "seeds": list(args.seeds),
        "epochs": args.epochs,
        "region_batch_size": args.region_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "n_train_slides": int(frame.iloc[train_indices]["slide_id"].nunique()),
        "n_val_slides": int(frame.iloc[val_indices]["slide_id"].nunique()),
        "n_test_rows_processed": 0,
        "device": str(device),
    }
    (args.out_dir / "calibration_design.json").write_text(
        json.dumps(design, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    rows: list[dict[str, object]] = []
    for seed in args.seeds:
        for variant_name, variant in VARIANTS.items():
            config = ProjectionConfig(
                input_dim=features.shape[1],
                biological_dim=args.biological_dim,
                acquisition_dim=args.acquisition_dim,
                hidden_dim=args.hidden_dim,
                scanner_adversary_weight=float(
                    variant["scanner_adversary_weight"]
                ),
                scanner_acquisition_weight=float(
                    variant["scanner_acquisition_weight"]
                ),
                scanner_dependence_weight=float(
                    variant["scanner_dependence_weight"]
                ),
                cross_covariance_weight=float(
                    variant["cross_covariance_weight"]
                ),
                gradient_reversal_strength=float(
                    variant["gradient_reversal_strength"]
                ),
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
            result = {
                "variant": variant_name,
                **result,
                **{
                    key: value
                    for key, value in asdict(config).items()
                    if key != "input_dim"
                },
            }
            rows.append(result)
            write_results(args.out_dir / "calibration_training_results.csv", rows)

    expected = len(args.seeds) * len(VARIANTS)
    if len(rows) != expected:
        raise ExperimentError(f"Expected {expected} fits, observed {len(rows)}")
    results = pd.DataFrame(rows)
    if results.duplicated(["variant", "seed"]).any():
        raise ExperimentError("Duplicate variant/seed rows found.")

    print("SCORPION PATHOALIGN CALIBRATION TRAINING PASSED")
    print(results.groupby("variant").mean(numeric_only=True).to_string())
    print(
        f"Results: "
        f"{(args.out_dir / 'calibration_training_results.csv').resolve()}"
    )


if __name__ == "__main__":
    try:
        main()
    except (ExperimentError, OSError, RuntimeError) as exc:
        print(f"SCORPION PATHOALIGN CALIBRATION FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
