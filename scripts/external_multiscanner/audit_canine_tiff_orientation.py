#!/usr/bin/env python3
"""Audit TIFF orientation and annotation coordinate frames by scanner.

The adaptive montage shows strong alignment for CS2, GT450, NZ20, and NZ210,
but systematic displacement/rotation for P1000. This metadata-only audit checks
whether TIFF orientation tags, raw level-0 dimensions, or coordinate transforms
explain that scanner-specific failure before any feature extraction.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable

import pandas as pd
import tifffile


SCANNERS = ("cs2", "gt450", "nz20", "nz210", "p1000")


def parse_bbox(value: Any) -> tuple[float, float, float, float]:
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, list) or len(parsed) != 4:
        raise ValueError(f"Invalid COCO bbox: {value!r}")
    x, y, width, height = (float(item) for item in parsed)
    if width <= 0 or height <= 0:
        raise ValueError(f"Non-positive bbox: {value!r}")
    return x, y, width, height


def tag_value(page: Any, name: str) -> Any:
    tag = page.tags.get(name)
    if tag is None:
        return None
    value = tag.value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, tuple):
        return list(value)
    return value


def inspect_file(path: Path) -> dict[str, Any]:
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        level = list(getattr(series, "levels", [series]))[0]
        axes = level.axes
        shape = tuple(int(item) for item in level.shape)
        page = level.pages[0]
        height = int(shape[axes.index("Y")])
        width = int(shape[axes.index("X")])
        return {
            "raw_width": width,
            "raw_height": height,
            "axes": axes,
            "shape": "x".join(str(item) for item in shape),
            "orientation": tag_value(page, "Orientation"),
            "image_width_tag": tag_value(page, "ImageWidth"),
            "image_length_tag": tag_value(page, "ImageLength"),
            "software": tag_value(page, "Software"),
            "x_resolution": tag_value(page, "XResolution"),
            "y_resolution": tag_value(page, "YResolution"),
            "resolution_unit": tag_value(page, "ResolutionUnit"),
            "description_prefix": str(tag_value(page, "ImageDescription") or "")[:240],
        }


def in_bounds(x: float, y: float, width: float, height: float) -> bool:
    return 0.0 <= x < width and 0.0 <= y < height


def transforms(raw_width: float, raw_height: float) -> dict[str, Callable[[float, float], tuple[float, float]]]:
    """Eight square-image dihedral coordinate candidates in raw coordinates."""
    return {
        "identity": lambda x, y: (x, y),
        "flip_x": lambda x, y: (raw_width - 1.0 - x, y),
        "flip_y": lambda x, y: (x, raw_height - 1.0 - y),
        "rotate_180": lambda x, y: (raw_width - 1.0 - x, raw_height - 1.0 - y),
        "transpose": lambda x, y: (y, x),
        "rotate_90_cw": lambda x, y: (raw_width - 1.0 - y, x),
        "rotate_90_ccw": lambda x, y: (y, raw_height - 1.0 - x),
        "transverse": lambda x, y: (raw_width - 1.0 - y, raw_height - 1.0 - x),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()

    frame = pd.read_csv(args.manifest)
    frame["scanner_id"] = frame["scanner_id"].astype(str).str.lower()
    if set(frame["scanner_id"].unique()) != set(SCANNERS):
        raise RuntimeError("Unexpected scanner set")

    file_rows: list[dict[str, Any]] = []
    metadata_by_file: dict[str, dict[str, Any]] = {}
    for file_name in sorted(frame["file_name"].astype(str).unique()):
        path = args.dataset_root / file_name
        if not path.is_file():
            raise FileNotFoundError(path)
        metadata = inspect_file(path)
        scanner_id = str(frame.loc[frame["file_name"].astype(str) == file_name, "scanner_id"].iloc[0])
        coco_width = float(frame.loc[frame["file_name"].astype(str) == file_name, "image_width"].iloc[0])
        coco_height = float(frame.loc[frame["file_name"].astype(str) == file_name, "image_height"].iloc[0])
        row = {
            "file_name": file_name,
            "scanner_id": scanner_id,
            "coco_width": coco_width,
            "coco_height": coco_height,
            **metadata,
        }
        row["raw_matches_coco_dimensions"] = (
            math.isclose(metadata["raw_width"], coco_width)
            and math.isclose(metadata["raw_height"], coco_height)
        )
        row["raw_matches_swapped_coco_dimensions"] = (
            math.isclose(metadata["raw_width"], coco_height)
            and math.isclose(metadata["raw_height"], coco_width)
        )
        file_rows.append(row)
        metadata_by_file[file_name] = row

    center_rows: list[dict[str, Any]] = []
    for _, item in frame.iterrows():
        x, y, width, height = parse_bbox(item["bbox"])
        center_x = x + width / 2.0
        center_y = y + height / 2.0
        metadata = metadata_by_file[str(item["file_name"])]
        raw_width = float(metadata["raw_width"])
        raw_height = float(metadata["raw_height"])
        row: dict[str, Any] = {
            "sample_id": item["sample_id"],
            "region_id": item["region_id"],
            "scanner_id": item["scanner_id"],
            "file_name": item["file_name"],
            "center_x_coco": center_x,
            "center_y_coco": center_y,
            "raw_width": raw_width,
            "raw_height": raw_height,
            "orientation": metadata["orientation"],
        }
        for name, transform in transforms(raw_width, raw_height).items():
            transformed_x, transformed_y = transform(center_x, center_y)
            row[f"{name}_x"] = transformed_x
            row[f"{name}_y"] = transformed_y
            row[f"{name}_center_in_bounds"] = in_bounds(
                transformed_x, transformed_y, raw_width, raw_height
            )
        center_rows.append(row)

    files = pd.DataFrame(file_rows)
    centers = pd.DataFrame(center_rows)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    files.to_csv(args.out_dir / "tiff_file_metadata.csv", index=False)
    centers.to_csv(args.out_dir / "annotation_center_transform_audit.csv", index=False)

    summary_rows: list[dict[str, Any]] = []
    for scanner in SCANNERS:
        scanner_files = files[files["scanner_id"] == scanner]
        scanner_centers = centers[centers["scanner_id"] == scanner]
        row: dict[str, Any] = {
            "scanner_id": scanner,
            "n_files": int(len(scanner_files)),
            "orientation_values": "|".join(
                sorted(scanner_files["orientation"].fillna("missing").astype(str).unique())
            ),
            "raw_matches_coco_fraction": float(scanner_files["raw_matches_coco_dimensions"].mean()),
            "raw_matches_swapped_coco_fraction": float(scanner_files["raw_matches_swapped_coco_dimensions"].mean()),
        }
        for name in transforms(1.0, 1.0):
            row[f"{name}_center_in_bounds_fraction"] = float(
                scanner_centers[f"{name}_center_in_bounds"].mean()
            )
        summary_rows.append(row)

    summary_frame = pd.DataFrame(summary_rows)
    summary_frame.to_csv(args.out_dir / "scanner_orientation_summary.csv", index=False)

    p1000 = summary_frame[summary_frame["scanner_id"] == "p1000"].iloc[0]
    transform_columns = [
        column for column in summary_frame.columns if column.endswith("_center_in_bounds_fraction")
    ]
    best_transform_column = max(transform_columns, key=lambda column: float(p1000[column]))
    best_transform = best_transform_column.removesuffix("_center_in_bounds_fraction")

    summary = {
        "status": "canine_tiff_orientation_audit_complete",
        "n_files": int(len(files)),
        "n_annotation_views": int(len(centers)),
        "scanner_summaries": summary_rows,
        "p1000_best_in_bounds_transform": best_transform,
        "p1000_best_in_bounds_fraction": float(p1000[best_transform_column]),
        "interpretation_gate": (
            "If P1000 differs in TIFF orientation or swapped dimensions, apply the corresponding "
            "coordinate transform and regenerate the montage. If all metadata are identical, the "
            "remaining issue is scanner-specific registration rather than TIFF orientation."
        ),
    }
    (args.out_dir / "orientation_audit_summary.json").write_text(
        json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8"
    )

    print("CANINE SCC TIFF ORIENTATION AUDIT PASSED")
    print(json.dumps(summary, indent=2, default=str))
    print("\nSCANNER SUMMARY")
    print(summary_frame.to_string(index=False))
    print(f"Artifacts: {args.out_dir.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"CANINE SCC TIFF ORIENTATION AUDIT FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
