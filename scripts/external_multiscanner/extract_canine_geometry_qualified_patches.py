#!/usr/bin/env python3
"""Extract orientation-normalized JPEG patches for the canine SCC benchmark.

Input is ``geometry_qualified_manifest.csv`` produced by
``build_canine_geometry_qualified_manifest.py``. The script reads each matched
TIFF region with the same adaptive crop geometry, applies the recorded P1000
orientation normalization, writes deterministic RGB JPEG patches, and emits
SCORPION-compatible full and rotating split manifests for frozen encoder runs.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.external_multiscanner.audit_canine_adaptive_crops_zarr3 import (
    read_adaptive_patch_zarr3,
)


SCANNERS = ("cs2", "gt450", "nz20", "nz210", "p1000")
REQUIRED = {
    "sample_id",
    "region_id",
    "scanner_id",
    "file_name",
    "category_name",
    "fold",
    "bbox_center_x",
    "bbox_center_y",
    "adaptive_crop_side_level0",
    "orientation_normalization_degrees",
}


def stable_patch_name(row: pd.Series) -> str:
    material = f"{row['region_id']}::{row['scanner_id']}::{row['file_name']}"
    digest = hashlib.sha1(material.encode("utf-8")).hexdigest()[:10]
    return f"{row['region_id']}__{row['scanner_id']}__{digest}.jpg"


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                fieldnames.append(key)
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def load_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    missing = sorted(REQUIRED - set(frame.columns))
    if missing:
        raise RuntimeError(f"Geometry-qualified manifest is missing columns: {missing}")
    frame["scanner_id"] = frame["scanner_id"].astype(str).str.lower()
    if set(frame["scanner_id"].unique()) != set(SCANNERS):
        raise RuntimeError("Unexpected scanner set")
    if frame.duplicated(["region_id", "scanner_id"]).any():
        raise RuntimeError("Duplicate region/scanner rows found")
    if not (frame.groupby("region_id").size() == len(SCANNERS)).all():
        raise RuntimeError("Each retained region must have exactly five scanner views")
    if frame["fold"].nunique() != 5:
        raise RuntimeError("Expected five sample-blocked folds")
    return frame.reset_index(drop=True)


def split_for_fold(row_fold: int, test_fold: int) -> str:
    val_fold = (test_fold + 1) % 5
    if row_fold == test_fold:
        return "test"
    if row_fold == val_fold:
        return "val"
    return "train"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--patch-root", type=Path, required=True)
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--target-read-size", type=int, default=768)
    parser.add_argument("--output-size", type=int, default=256)
    parser.add_argument("--jpeg-quality", type=int, default=94)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if not (1 <= args.jpeg_quality <= 100):
        raise ValueError("jpeg-quality must be between 1 and 100")
    frame = load_manifest(args.manifest)
    args.patch_root.mkdir(parents=True, exist_ok=True)
    args.manifest_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    pixel_rows: list[dict[str, Any]] = []
    for index, item in frame.iterrows():
        scanner = str(item["scanner_id"])
        scanner_dir = args.patch_root / scanner
        scanner_dir.mkdir(parents=True, exist_ok=True)
        patch_name = stable_patch_name(item)
        patch_path = scanner_dir / patch_name
        if args.overwrite or not patch_path.is_file():
            source_path = args.dataset_root / str(item["file_name"])
            if not source_path.is_file():
                raise FileNotFoundError(source_path)
            patch, metadata = read_adaptive_patch_zarr3(
                source_path,
                center_x_level0=float(item["bbox_center_x"]),
                center_y_level0=float(item["bbox_center_y"]),
                crop_side_level0=int(item["adaptive_crop_side_level0"]),
                target_read_size=args.target_read_size,
                output_size=args.output_size,
            )
            angle = float(item["orientation_normalization_degrees"])
            if abs(angle) > 1e-6:
                patch = patch.rotate(
                    angle,
                    resample=Image.Resampling.BICUBIC,
                    expand=False,
                    fillcolor="white",
                )
            patch.save(patch_path, quality=args.jpeg_quality, subsampling=0)
        else:
            metadata = {"read_skipped_existing_patch": True}

        rel_path = patch_path.relative_to(args.patch_root).as_posix()
        row = {
            "slide_id": str(item["sample_id"]),
            "sample_id": str(item["sample_id"]),
            "region_id": str(item["region_id"]),
            "scanner_id": scanner,
            "path": rel_path,
            "fold": int(item["fold"]),
            "source_filename": str(item["file_name"]),
            "category_name": str(item["category_name"]),
            "adaptive_crop_side_level0": int(item["adaptive_crop_side_level0"]),
            "orientation_normalization_degrees": float(item["orientation_normalization_degrees"]),
        }
        rows.append(row)
        pixel_rows.append({**row, **metadata})
        if (index + 1) % 100 == 0 or index + 1 == len(frame):
            print(f"\rExtracted/verified {index + 1:,} / {len(frame):,} patches", end="", flush=True)
    print()

    full_manifest = pd.DataFrame(rows)
    full_manifest.to_csv(args.manifest_dir / "full_patch_manifest.csv", index=False)
    pd.DataFrame(pixel_rows).to_csv(args.manifest_dir / "patch_pixel_audit.csv", index=False)

    split_summaries: dict[str, Any] = {}
    for test_fold in range(5):
        split = full_manifest.copy()
        split["split"] = split["fold"].map(lambda value: split_for_fold(int(value), test_fold))
        path = args.manifest_dir / "splits" / f"fold_{test_fold}_patch_manifest.csv"
        path.parent.mkdir(parents=True, exist_ok=True)
        split.to_csv(path, index=False)
        split_summaries[f"fold_{test_fold}"] = {
            name: {
                "rows": int((split["split"] == name).sum()),
                "regions": int(split.loc[split["split"] == name, "region_id"].nunique()),
                "samples": int(split.loc[split["split"] == name, "sample_id"].nunique()),
            }
            for name in ("train", "val", "test")
        }
        if any(split_summaries[f"fold_{test_fold}"][name]["rows"] == 0 for name in ("train", "val", "test")):
            raise RuntimeError(f"Rotating fold {test_fold} contains an empty split")

    summary = {
        "status": "canine_geometry_qualified_patch_extraction_complete",
        "source_manifest": str(args.manifest.resolve()),
        "patch_root": str(args.patch_root.resolve()),
        "manifest_dir": str(args.manifest_dir.resolve()),
        "n_rows": int(len(full_manifest)),
        "n_regions": int(full_manifest["region_id"].nunique()),
        "n_samples": int(full_manifest["sample_id"].nunique()),
        "n_scanners": int(full_manifest["scanner_id"].nunique()),
        "target_read_size": args.target_read_size,
        "output_size": args.output_size,
        "jpeg_quality": args.jpeg_quality,
        "rotating_split_summaries": split_summaries,
        "next_gate": "Extract frozen encoder features from the fold patch manifests.",
    }
    (args.manifest_dir / "patch_extraction_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n",
        encoding="utf-8",
    )

    print("CANINE SCC PATCH EXTRACTION PASSED")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"CANINE SCC PATCH EXTRACTION FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
