#!/usr/bin/env python3
"""Run frozen PathoAlign validation on canine SCC fold 0.

This is the first external paired-acquisition PathoAlign validation. It reuses
the locked SCORPION projection implementation, but patches the scanner namespace
to the five canine scanners before grouping or training. Test rows are never
projected; only train+validation rows are used for fold-0 validation analysis.
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
SEEDS = tuple(range(901, 906))
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


def validate_archive(frame: pd.DataFrame) -> None:
    scanners = set(frame["scanner_id"].astype(str).str.lower())
    if scanners != set(CANINE_SCANNERS):
        raise projection.ExperimentError(f"Unexpected scanner set: {sorted(scanners)}")
    frame["scanner_id"] = frame["scanner_id"].astype(str).str.lower()
    if frame["region_id"].nunique() != 805:
        raise projection.ExperimentError("Expected 805 geometry-qualified regions.")
    if len(frame) != 4025:
        raise projection.ExperimentError("Expected 4,025 five-view rows.")
    train_samples = set(frame.loc[frame["split"] == "train", "slide_id"])
    val_samples = set(frame.loc[frame["split"] == "val", "slide_id"])
    test_samples = set(frame.loc[frame["split"] == "test", "slide_id"])
    if train_samples & val_samples or train_samples & test_samples or val_samples & test_samples:
        raise projection.ExperimentError("Sample leakage across split labels.")
    for split in ("train", "val", "test"):
        subset = frame[frame["split"] == split]
        if subset.empty:
            raise projection.ExperimentError(f"Empty split: {split}")
        if not (subset.groupby("region_id").size() == len(CANINE_SCANNERS)).all():
            raise projection.ExperimentError(f"Split {split} contains incomplete regions.")


def load_existing(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    return pd.read_csv(path).to_dict("records")


def write_results(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        return
    pd.DataFrame(rows).sort_values(["variant", "seed"]).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--features", type=Path, required=True)
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
    patch_scanner_namespace()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    features, frame, source_metadata = projection.load_archive(args.features)
    frame["scanner_id"] = frame["scanner_id"].astype(str).str.lower()
    validate_archive(frame)
    train_indices = projection.indices_for(frame, "train")
    val_indices = projection.indices_for(frame, "val")
    development_indices = np.concatenate([train_indices, val_indices])
    if np.any(frame.iloc[development_indices]["split"].to_numpy() == "test"):
        raise projection.ExperimentError("Test rows entered the development set.")
    transformed, mean, std = projection.standardize(features, train_indices)
    groups = projection.region_groups(frame, train_indices)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise projection.ExperimentError("CUDA requested but unavailable.")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out_dir / "train_standardization.npz", mean=mean, std=std)
    results_path = args.out_dir / "training_results.csv"
    rows = load_existing(results_path)
    completed = {(str(row["variant"]), int(row["seed"])) for row in rows}

    design = {
        "stage": "external_canine_fold0_validation",
        "features": str(args.features.resolve()),
        "source_metadata": source_metadata,
        "scanners": list(CANINE_SCANNERS),
        "seeds": list(args.seeds),
        "variants": VARIANTS,
        "epochs": args.epochs,
        "region_batch_size": args.region_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "fit_split": "train",
        "evaluation_split": "val",
        "test_rows_projected": 0,
        "hyperparameters_frozen": True,
        "device": str(device),
    }
    (args.out_dir / "design.json").write_text(
        json.dumps(design, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    for seed in args.seeds:
        for variant_name, variant in VARIANTS.items():
            key = (variant_name, seed)
            if key in completed:
                print(f"Skipping completed variant={variant_name} seed={seed}")
                continue
            config = config_for(features.shape[1], variant)
            run_dir = args.out_dir / "runs" / f"{variant_name}_seed_{seed}"
            result = projection.train_one(
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
            projected = run_dir / "projected_features.npz"
            with np.load(projected, allow_pickle=False) as archive:
                arrays = {name: archive[name] for name in archive.files}
            metadata = json.loads(str(arrays["metadata_json"].item()))
            metadata.update(
                {
                    "source": "External canine SCC DINOv2 PathoAlign validation",
                    "model": f"canine_dinov2_{variant_name}",
                    "evaluation_stage": "external_canine_fold0_validation",
                    "contains_test_rows": False,
                    "fit_splits": ["train"],
                    "evaluation_split": "val",
                    "hyperparameters_frozen": True,
                    "scanner_namespace": list(CANINE_SCANNERS),
                }
            )
            text = json.dumps(metadata, sort_keys=True)
            arrays["metadata_json"] = np.asarray(text, dtype=f"<U{len(text)}")
            projection.atomic_npz(projected, arrays)
            rows.append(
                {
                    "variant": variant_name,
                    **result,
                    **asdict(config),
                    "n_train_samples": int(frame.iloc[train_indices]["slide_id"].nunique()),
                    "n_val_samples": int(frame.iloc[val_indices]["slide_id"].nunique()),
                    "n_train_regions": int(frame.iloc[train_indices]["region_id"].nunique()),
                    "n_val_regions": int(frame.iloc[val_indices]["region_id"].nunique()),
                }
            )
            write_results(results_path, rows)
            completed.add(key)

    expected = len(args.seeds) * len(VARIANTS)
    if len(pd.read_csv(results_path)) != expected:
        raise projection.ExperimentError(f"Expected {expected} completed fits.")
    print("CANINE SCC PATHOALIGN FOLD0 TRAINING PASSED")
    print(f"Completed fits: {expected}")
    print(f"Results: {results_path.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (projection.ExperimentError, OSError, RuntimeError) as exc:
        print(f"CANINE SCC PATHOALIGN FOLD0 TRAINING FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
