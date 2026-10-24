#!/usr/bin/env python3
"""Identify and replay-validate the local Phikon snapshot used for PANDA features.

Before any SICAPv2 feature extraction, this script re-extracts fixed PANDA patch
rows from raw WSIs at the stored HDF5 coordinates and compares them against the
existing 768-D PANDA features. External extraction must use a snapshot that
passes this gate.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import openslide
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms
from transformers import ViTModel


DEFAULT_SPEC = Path(
    "experiments/transnnmil/transnnmil_sicap_external_transport_spec_20260925.json"
)
DEFAULT_MANIFEST = Path(
    "results/panda_transnnmil_matched_rerun/locked_split_20260923_readable.csv"
)
DEFAULT_OUTPUT = Path(
    "experiments/transnnmil/external/phikon_replay_verification_20260925.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spec", type=Path, default=DEFAULT_SPEC)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--panda-wsi-root", type=Path, default=Path(r"D:\panda\train_images")
    )
    parser.add_argument(
        "--snapshot",
        type=Path,
        default=None,
        help="Explicit local Hugging Face snapshot directory. If omitted, resolve refs/main.",
    )
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def hf_model_cache_root() -> Path:
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub" / "models--owkin--phikon"
    if os.environ.get("HUGGINGFACE_HUB_CACHE"):
        return Path(os.environ["HUGGINGFACE_HUB_CACHE"]) / "models--owkin--phikon"
    return Path.home() / ".cache" / "huggingface" / "hub" / "models--owkin--phikon"


def resolve_snapshot(explicit: Path | None) -> tuple[Path, str]:
    if explicit is not None:
        path = explicit.resolve()
        if not path.is_dir():
            raise FileNotFoundError(f"Phikon snapshot not found: {path}")
        return path, path.name

    root = hf_model_cache_root()
    ref_path = root / "refs" / "main"
    if not ref_path.is_file():
        raise FileNotFoundError(
            f"cannot resolve cached owkin/phikon refs/main at {ref_path}; "
            "pass --snapshot explicitly"
        )
    revision = ref_path.read_text(encoding="utf-8").strip()
    path = root / "snapshots" / revision
    if not path.is_dir():
        raise FileNotFoundError(f"cached Phikon snapshot missing: {path}")
    return path.resolve(), revision


def find_wsi(root: Path, image_id: str) -> Path:
    candidates = [
        root / f"{image_id}.tiff",
        root / f"{image_id}.tif",
        root / f"{image_id}.svs",
        root / f"{image_id}.ndpi",
    ]
    hits = [path for path in candidates if path.is_file()]
    if len(hits) != 1:
        raise FileNotFoundError(
            f"expected exactly one raw WSI for {image_id}; found {[str(x) for x in hits]}"
        )
    return hits[0]


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.dot(a, b) / max(denom, 1e-12))


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / max(float(np.linalg.norm(b)), 1e-12))


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(
            f"{args.output} already exists. Replay verification is frozen once external work begins."
        )

    spec = load_json(args.spec)
    replay = spec["feature_transport"]["replay_gate"]
    n_slides = int(replay["minimum_panda_slides"])
    rows_per_slide = int(replay["rows_per_slide"])
    minimum_cosine = float(replay["minimum_cosine_similarity"])
    maximum_relative_l2 = float(replay["maximum_relative_l2"])

    snapshot, revision = resolve_snapshot(args.snapshot)
    device = torch.device(args.device)
    model = ViTModel.from_pretrained(
        str(snapshot),
        local_files_only=True,
        add_pooling_layer=False,
    ).to(device)
    model.eval()

    transform = transforms.Compose(
        [
            transforms.Resize(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )

    frame = pd.read_csv(args.manifest).sort_values("image_id").reset_index(drop=True)
    selected = []
    for _, row in frame.iterrows():
        image_id = str(row["image_id"])
        try:
            wsi_path = find_wsi(args.panda_wsi_root, image_id)
        except FileNotFoundError:
            continue
        feature_path = Path(str(row["feature_path"]))
        if feature_path.is_file():
            selected.append((image_id, wsi_path, feature_path))
        if len(selected) >= n_slides:
            break
    if len(selected) != n_slides:
        raise RuntimeError(
            f"could identify only {len(selected)}/{n_slides} deterministic PANDA replay slides"
        )

    comparisons = []
    for image_id, wsi_path, feature_path in selected:
        with h5py.File(feature_path, "r") as handle:
            if "features" not in handle or "coordinates" not in handle:
                raise ValueError(f"missing PANDA HDF5 datasets: {feature_path}")
            refs = np.asarray(handle["features"][:rows_per_slide], dtype=np.float32)
            coords = np.asarray(handle["coordinates"][:rows_per_slide])
        if refs.shape != (rows_per_slide, int(spec["feature_transport"]["required_output_dim"])):
            raise ValueError(f"unexpected reference feature shape {refs.shape}: {feature_path}")
        if coords.shape[0] != rows_per_slide:
            raise ValueError(f"insufficient coordinates: {feature_path}")

        slide = openslide.OpenSlide(str(wsi_path))
        try:
            if slide.level_count <= 1:
                raise ValueError(f"PANDA WSI has no level 1: {wsi_path}")
            tensors = []
            for coord in coords:
                x, y = int(coord[0]), int(coord[1])
                patch = slide.read_region((x, y), 1, (224, 224)).convert("RGB")
                tensors.append(transform(patch))
        finally:
            slide.close()

        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            replay_features = (
                model(pixel_values=batch)
                .last_hidden_state[:, 0, :]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
            )

        for index in range(rows_per_slide):
            c = cosine(replay_features[index], refs[index])
            r = relative_l2(replay_features[index], refs[index])
            comparisons.append(
                {
                    "image_id": image_id,
                    "row_index": index,
                    "cosine_similarity": c,
                    "relative_l2": r,
                }
            )

    min_cosine = min(row["cosine_similarity"] for row in comparisons)
    max_relative = max(row["relative_l2"] for row in comparisons)
    passed = bool(min_cosine >= minimum_cosine and max_relative <= maximum_relative_l2)
    payload = {
        "schema_version": "phikon-panda-replay-verification/v1",
        "status": "passed" if passed else "failed",
        "snapshot_path": str(snapshot),
        "snapshot_revision": revision,
        "device": str(device),
        "panda_wsi_root": str(args.panda_wsi_root),
        "panda_manifest": str(args.manifest),
        "thresholds": {
            "minimum_cosine_similarity": minimum_cosine,
            "maximum_relative_l2": maximum_relative_l2,
        },
        "observed": {
            "minimum_cosine_similarity": min_cosine,
            "maximum_relative_l2": max_relative,
            "comparison_count": len(comparisons),
        },
        "comparisons": comparisons,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["observed"] | {"status": payload["status"], "snapshot_revision": revision}, indent=2))
    if not passed:
        raise RuntimeError(
            "local Phikon snapshot failed replay against existing PANDA features; "
            "do not extract SICAPv2 features"
        )


if __name__ == "__main__":
    main()
