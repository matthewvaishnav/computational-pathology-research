#!/usr/bin/env python3
"""Extract frozen Phikon, DINOv2-Base, or UNI features for SCORPION."""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset


SPECS = {
    "phikon": {"repo": "owkin/phikon", "dim": 768, "loader": "hf"},
    "dinov2_base": {"repo": "facebook/dinov2-base", "dim": 768, "loader": "hf"},
    "uni": {"repo": "MahmoodLab/UNI", "dim": 1024, "loader": "timm"},
}
REQUIRED = ("slide_id", "region_id", "scanner_id", "path")


class ExtractionError(RuntimeError):
    pass


def load_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str)
    missing = [column for column in REQUIRED if column not in frame.columns]
    if missing:
        raise ExtractionError(f"Missing manifest columns: {missing}")
    if frame.empty or frame.duplicated(["slide_id", "region_id", "scanner_id"]).any():
        raise ExtractionError("Manifest is empty or contains duplicate paired rows.")
    return frame.reset_index(drop=True)


def resolve_path(raw: str, root: Path) -> Path:
    path = Path(raw)
    return (path if path.is_absolute() else root / path).resolve()


class ImageDataset(Dataset):
    def __init__(self, frame: pd.DataFrame, root: Path, transform):
        self.frame, self.root, self.transform = frame, root, transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, index):
        path = resolve_path(str(self.frame.iloc[index]["path"]), self.root)
        if not path.is_file():
            raise ExtractionError(f"Missing image: {path}")
        with Image.open(path) as image:
            tensor = self.transform(image.convert("RGB"))
        return tensor, index


def load_hf(repo: str, device: torch.device):
    try:
        from transformers import AutoImageProcessor, AutoModel
    except ImportError as exc:
        raise ExtractionError("Install transformers: python -m pip install transformers") from exc
    processor = AutoImageProcessor.from_pretrained(repo)
    model = AutoModel.from_pretrained(repo).eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False

    def transform(image):
        return processor(images=image, return_tensors="pt")["pixel_values"].squeeze(0)

    def forward(batch):
        return model(pixel_values=batch).last_hidden_state[:, 0, :]

    revision = str(getattr(model.config, "_commit_hash", None) or "unrecorded")
    return model, transform, forward, revision


def load_uni(device: torch.device):
    try:
        import timm
        from timm.data import resolve_data_config
        from timm.data.transforms_factory import create_transform
    except ImportError as exc:
        raise ExtractionError(
            "Install UNI dependencies: python -m pip install timm huggingface_hub"
        ) from exc
    try:
        model = timm.create_model(
            "hf-hub:MahmoodLab/UNI",
            pretrained=True,
            init_values=1e-5,
            dynamic_img_size=True,
        ).eval().to(device)
    except Exception as exc:
        raise ExtractionError(
            "UNI access failed. Accept the model terms and authenticate with Hugging Face. "
            f"Original error: {exc}"
        ) from exc
    for parameter in model.parameters():
        parameter.requires_grad = False
    transform = create_transform(**resolve_data_config(model.pretrained_cfg, model=model))
    return model, transform, model, "hf-hub:MahmoodLab/UNI"


def string_array(series: pd.Series) -> np.ndarray:
    values = series.fillna("").astype(str).tolist()
    width = max(1, max(map(len, values)))
    return np.asarray(values, dtype=f"<U{width}")


def save_npz(path: Path, arrays: dict[str, np.ndarray], dim: int):
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=path.name + ".", suffix=".npz", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        np.savez_compressed(temporary, **arrays)
        with np.load(temporary, allow_pickle=False) as check:
            if check["features"].shape[1] != dim:
                raise ExtractionError("Saved feature dimension is incorrect.")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def run(args) -> dict[str, object]:
    frame = load_manifest(args.manifest)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ExtractionError("CUDA requested but unavailable.")
    spec = SPECS[args.encoder]
    if spec["loader"] == "hf":
        model, transform, forward, revision = load_hf(str(spec["repo"]), device)
    else:
        model, transform, forward, revision = load_uni(device)

    loader = DataLoader(
        ImageDataset(frame, args.data_root, transform),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    dim = int(spec["dim"])
    features = np.empty((len(frame), dim), dtype=np.float32)
    seen = np.zeros(len(frame), dtype=bool)
    amp = device.type == "cuda" and not args.no_amp

    with torch.inference_mode():
        for batch_number, (images, indices) in enumerate(loader, 1):
            images = images.to(device, non_blocking=amp)
            context = torch.autocast("cuda", dtype=torch.float16) if amp else contextlib.nullcontext()
            with context:
                output = forward(images)
            if output.ndim != 2 or output.shape[1] != dim:
                raise ExtractionError(f"Unexpected output shape: {tuple(output.shape)}")
            index_array = indices.numpy()
            features[index_array] = output.float().cpu().numpy()
            seen[index_array] = True
            print(f"\r{args.encoder}: {seen.sum():,}/{len(frame):,} images (batch {batch_number})", end="", flush=True)
    print()

    if not seen.all() or not np.isfinite(features).all() or float(features.var(axis=0).mean()) <= 0:
        raise ExtractionError("Feature validation failed.")

    metadata = {
        "model": args.encoder,
        "model_source": spec["repo"],
        "model_revision": revision,
        "feature_dim": dim,
        "manifest": str(args.manifest.resolve()),
        "data_root": str(args.data_root.resolve()),
        "n_images": len(frame),
        "device": str(device),
        "batch_size": args.batch_size,
        "amp": amp,
        "torch_version": torch.__version__,
    }
    text = json.dumps(metadata, sort_keys=True)
    arrays = {
        "features": features,
        "slide_id": string_array(frame["slide_id"]),
        "region_id": string_array(frame["region_id"]),
        "scanner_id": string_array(frame["scanner_id"]),
        "path": string_array(frame["path"]),
        "metadata_json": np.asarray(text, dtype=f"<U{len(text)}"),
    }
    for column in ("split", "fold", "slide_number", "sample_number", "source_filename"):
        if column in frame:
            arrays[column] = string_array(frame[column])
    save_npz(args.output, arrays, dim)

    summary = {
        **metadata,
        "output": str(args.output.resolve()),
        "feature_mean": float(features.mean()),
        "feature_std": float(features.std()),
        "mean_dimension_variance": float(features.var(axis=0).mean()),
        "output_size_bytes": args.output.stat().st_size,
    }
    args.output.with_suffix(".summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--encoder", choices=tuple(SPECS), required=True)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true")
    args = parser.parse_args()
    try:
        summary = run(args)
    except (ExtractionError, OSError, RuntimeError) as exc:
        print(f"SCORPION VIT FEATURE EXTRACTION FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
    print("SCORPION VIT FEATURE EXTRACTION PASSED")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
