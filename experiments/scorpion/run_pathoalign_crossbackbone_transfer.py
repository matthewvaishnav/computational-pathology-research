#!/usr/bin/env python3
"""Run frozen PathoAlign transfer on SCORPION Phikon and ResNet50 features.

The objective, architecture, schedule, folds, and success criteria were fixed
using DINOv2 development. This runner applies them unchanged to two new frozen
representation families with shared new seeds 701-705.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# PyTorch requires this to be configured before CUDA/cuBLAS initialization when
# deterministic algorithms are requested. Respect an explicit user override.
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from experiments.scorpion import run_pathoalign_crossfold as crossfold
from experiments.scorpion.run_pathoalign_projection import ExperimentError, load_archive


SEEDS = tuple(range(701, 706))
BACKBONES = ("phikon", "resnet50")
EXPECTED_DIMENSIONS = {"phikon": 768, "resnet50": 2048}


def validate_archive(label: str, path: Path) -> dict[str, object]:
    features, frame, metadata = load_archive(path)
    if len(features) != 2400 or len(frame) != 2400:
        raise ExperimentError(
            f"{label} archive must contain 2,400 aligned rows; "
            f"observed features={len(features)}, metadata={len(frame)}"
        )
    expected_dim = EXPECTED_DIMENSIONS[label]
    if features.ndim != 2 or features.shape[1] != expected_dim:
        raise ExperimentError(
            f"{label} archive must have dimension {expected_dim}; "
            f"observed shape={features.shape}"
        )
    if not np.isfinite(features).all() or float(features.var(axis=0).mean()) <= 0:
        raise ExperimentError(f"{label} archive contains invalid or constant features.")

    model = str(metadata.get("model", "")).lower()
    if label == "phikon" and "phikon" not in model:
        raise ExperimentError(
            f"Expected Phikon metadata for {path}; observed model={model!r}"
        )
    if label == "resnet50" and "resnet50" not in model:
        raise ExperimentError(
            f"Expected ResNet50 metadata for {path}; observed model={model!r}"
        )

    return {
        "path": str(path.resolve()),
        "model": metadata.get("model"),
        "model_source": metadata.get("model_source"),
        "model_revision": metadata.get("model_revision"),
        "feature_dim": int(features.shape[1]),
        "n_images": int(len(features)),
        "feature_variance_mean": float(features.var(axis=0).mean()),
    }


def run_crossfold(
    *,
    features: Path,
    manifests_dir: Path,
    out_dir: Path,
    epochs: int,
    region_batch_size: int,
    learning_rate: float,
    weight_decay: float,
    device: str,
) -> None:
    original_argv = sys.argv[:]
    original_seeds = crossfold.SEEDS
    crossfold.SEEDS = SEEDS
    sys.argv = [
        str(Path(crossfold.__file__).resolve()),
        "--base-features",
        str(features),
        "--manifests-dir",
        str(manifests_dir),
        "--out-dir",
        str(out_dir),
        "--epochs",
        str(epochs),
        "--region-batch-size",
        str(region_batch_size),
        "--learning-rate",
        str(learning_rate),
        "--weight-decay",
        str(weight_decay),
        "--device",
        device,
    ]
    try:
        crossfold.main()
    finally:
        crossfold.SEEDS = original_seeds
        sys.argv = original_argv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phikon-features", type=Path, required=True)
    parser.add_argument("--resnet50-features", type=Path, required=True)
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

    if args.epochs != 75:
        raise ExperimentError("Cross-backbone transfer is frozen at 75 epochs.")
    if args.region_batch_size != 32:
        raise ExperimentError("Cross-backbone transfer is frozen at region batch size 32.")
    if args.learning_rate != 3e-4:
        raise ExperimentError("Cross-backbone transfer is frozen at learning rate 3e-4.")
    if args.weight_decay != 1e-4:
        raise ExperimentError("Cross-backbone transfer is frozen at weight decay 1e-4.")

    feature_paths = {
        "phikon": args.phikon_features,
        "resnet50": args.resnet50_features,
    }
    archives = {
        label: validate_archive(label, path)
        for label, path in feature_paths.items()
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    design = {
        "stage": "frozen_crossbackbone_transfer",
        "development_backbone": "dinov2_base",
        "transfer_backbones": list(BACKBONES),
        "seeds": list(SEEDS),
        "folds": list(crossfold.FOLDS),
        "variants": crossfold.VARIANTS,
        "archives": archives,
        "manifests_dir": str(args.manifests_dir.resolve()),
        "epochs": args.epochs,
        "region_batch_size": args.region_batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "device": args.device,
        "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        "hyperparameters_frozen": True,
        "no_backbone_specific_tuning": True,
        "protocol": "docs/research/scorpion-pathoalign-crossbackbone-protocol.md",
    }
    (args.out_dir / "crossbackbone_transfer_design.json").write_text(
        json.dumps(design, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    for label in BACKBONES:
        print(f"\n=== FROZEN TRANSFER: {label} ===")
        run_crossfold(
            features=feature_paths[label],
            manifests_dir=args.manifests_dir,
            out_dir=args.out_dir / label,
            epochs=args.epochs,
            region_batch_size=args.region_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            device=args.device,
        )

    print("\nSCORPION PATHOALIGN CROSS-BACKBONE TRANSFER TRAINING PASSED")
    print(f"Seeds: {list(SEEDS)}")
    print(f"Completed fits: {len(BACKBONES) * 5 * len(SEEDS) * 2}")
    print(f"Artifacts: {args.out_dir.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (ExperimentError, OSError, RuntimeError) as exc:
        print(f"SCORPION CROSS-BACKBONE TRANSFER FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
