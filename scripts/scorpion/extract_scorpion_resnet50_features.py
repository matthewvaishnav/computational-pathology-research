#!/usr/bin/env python3
"""Extract frozen ImageNet ResNet50 features for every SCORPION image.

This reproduces the feature family used in the SCORPION dataset analysis: a
ResNet50 pretrained on ImageNet. Each 1024 x 1024 JPEG is deterministically
resized/cropped with the torchvision IMAGENET1K_V2 preprocessing pipeline and
mapped to one 2,048-dimensional penultimate-layer embedding.

Input manifest columns
----------------------
Required:
    slide_id, region_id, scanner_id, path
Optional but preserved:
    split, fold, slide_number, sample_number, source_filename

Example
-------
python scripts/scorpion/extract_scorpion_resnet50_features.py \
    --manifest data/scorpion/splits/fold_0_manifest.csv \
    --data-root "$HOME/Downloads/SCORPION_dataset/SCORPION_dataset" \
    --output results/scorpion/features/fold_0_resnet50_imagenet.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.models import ResNet50_Weights, resnet50


REQUIRED_COLUMNS = ("slide_id", "region_id", "scanner_id", "path")
MODEL_NAME = "resnet50_imagenet1k_v2"
FEATURE_DIM = 2048


class ExtractionError(RuntimeError):
    """Raised when a manifest or image cannot be processed safely."""


def load_manifest(path: Path) -> pd.DataFrame:
    """Load and validate a SCORPION manifest while preserving row order."""
    frame = pd.read_csv(path, dtype=str)
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ExtractionError(f"Manifest is missing required columns: {missing}")
    if frame.empty:
        raise ExtractionError("Manifest contains no rows.")
    if frame.duplicated(["slide_id", "region_id", "scanner_id"]).any():
        raise ExtractionError("Manifest contains duplicate slide/region/scanner rows.")
    return frame.reset_index(drop=True)


def resolve_path(raw: str, data_root: Path) -> Path:
    """Resolve an absolute or dataset-root-relative image path."""
    path = Path(raw)
    if not path.is_absolute():
        path = data_root / path
    return path.resolve()


class ScorpionImageDataset(Dataset[tuple[torch.Tensor, int]]):
    """Deterministic image loader that returns tensors plus manifest row indices."""

    def __init__(self, frame: pd.DataFrame, data_root: Path, transform: Any):
        self.frame = frame
        self.data_root = data_root
        self.transform = transform

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        row = self.frame.iloc[index]
        path = resolve_path(str(row["path"]), self.data_root)
        if not path.is_file():
            raise ExtractionError(f"Image does not exist: {path}")
        try:
            with Image.open(path) as image:
                rgb = image.convert("RGB")
                tensor = self.transform(rgb)
        except Exception as exc:  # noqa: BLE001 - report corrupt input precisely
            raise ExtractionError(f"Could not read {path}: {type(exc).__name__}: {exc}") from exc
        return tensor, index


def create_model(device: torch.device) -> tuple[nn.Module, Any]:
    """Create the frozen ResNet50 and its canonical preprocessing transform."""
    weights = ResNet50_Weights.IMAGENET1K_V2
    model = resnet50(weights=weights)
    model.fc = nn.Identity()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad = False
    model = model.to(device)
    return model, weights.transforms()


def atomic_save_npz(path: Path, arrays: dict[str, np.ndarray]) -> None:
    """Write a compressed NPZ atomically to avoid leaving a valid-looking partial file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temp_name = tempfile.mkstemp(
        prefix=path.name + ".",
        suffix=".tmp.npz",
        dir=str(path.parent),
    )
    os.close(handle)
    temp_path = Path(temp_name)
    try:
        np.savez_compressed(temp_path, **arrays)
        with np.load(temp_path, allow_pickle=False) as check:
            if "features" not in check or check["features"].shape[1] != FEATURE_DIM:
                raise ExtractionError("Temporary NPZ failed post-write validation.")
        temp_path.replace(path)
    finally:
        temp_path.unlink(missing_ok=True)


def string_array(series: pd.Series) -> np.ndarray:
    """Create a non-object Unicode array safe for allow_pickle=False loading."""
    values = series.fillna("").astype(str).tolist()
    width = max(1, max(len(value) for value in values))
    return np.asarray(values, dtype=f"<U{width}")


def extract_features(
    manifest: Path,
    data_root: Path,
    output: Path,
    batch_size: int,
    num_workers: int,
    device_name: str,
) -> dict[str, object]:
    """Run frozen feature extraction and return a compact summary."""
    frame = load_manifest(manifest)
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ExtractionError("CUDA was requested but torch.cuda.is_available() is false.")

    model, transform = create_model(device)
    dataset = ScorpionImageDataset(frame, data_root, transform)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )

    features = np.empty((len(frame), FEATURE_DIM), dtype=np.float32)
    seen = np.zeros(len(frame), dtype=bool)

    with torch.inference_mode():
        for batch_number, (images, indices) in enumerate(loader, start=1):
            images = images.to(device, non_blocking=device.type == "cuda")
            batch_features = model(images)
            if batch_features.ndim != 2 or batch_features.shape[1] != FEATURE_DIM:
                raise ExtractionError(
                    f"Unexpected model output shape: {tuple(batch_features.shape)}"
                )
            indices_np = indices.numpy()
            features[indices_np] = batch_features.detach().cpu().numpy().astype(np.float32)
            seen[indices_np] = True
            print(
                f"\rProcessed {int(seen.sum()):,} / {len(frame):,} images "
                f"(batch {batch_number})",
                end="",
                flush=True,
            )
    print()

    if not seen.all():
        missing = np.flatnonzero(~seen).tolist()
        raise ExtractionError(f"Rows were not processed: {missing[:20]}")
    if not np.isfinite(features).all():
        raise ExtractionError("Features contain NaN or infinite values.")
    variances = features.var(axis=0)
    if float(variances.mean()) <= 0.0:
        raise ExtractionError("Feature matrix has zero mean variance.")

    arrays: dict[str, np.ndarray] = {
        "features": features,
        "slide_id": string_array(frame["slide_id"]),
        "region_id": string_array(frame["region_id"]),
        "scanner_id": string_array(frame["scanner_id"]),
        "path": string_array(frame["path"]),
    }
    for optional in ("split", "fold", "slide_number", "sample_number", "source_filename"):
        if optional in frame.columns:
            arrays[optional] = string_array(frame[optional])

    metadata = {
        "model": MODEL_NAME,
        "feature_dim": FEATURE_DIM,
        "manifest": str(manifest.resolve()),
        "data_root": str(data_root.resolve()),
        "n_images": len(frame),
        "device": str(device),
        "batch_size": batch_size,
        "num_workers": num_workers,
        "torch_version": torch.__version__,
    }
    metadata_text = json.dumps(metadata, sort_keys=True)
    arrays["metadata_json"] = np.asarray(metadata_text, dtype=f"<U{len(metadata_text)}")
    atomic_save_npz(output, arrays)

    summary = {
        **metadata,
        "output": str(output.resolve()),
        "feature_mean": float(features.mean()),
        "feature_std": float(features.std()),
        "mean_dimension_variance": float(variances.mean()),
        "output_size_bytes": output.stat().st_size,
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive.")
    if args.num_workers < 0:
        raise SystemExit("--num-workers cannot be negative.")

    try:
        summary = extract_features(
            manifest=args.manifest,
            data_root=args.data_root,
            output=args.output,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            device_name=args.device,
        )
    except (ExtractionError, OSError, RuntimeError) as exc:
        print(f"SCORPION FEATURE EXTRACTION FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc

    print("SCORPION RESNET50 FEATURE EXTRACTION PASSED")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
