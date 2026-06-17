#!/usr/bin/env python3
"""Create an adaptive five-view crop plan and visual montage for canine SCC.

A single fixed crop retains fewer than half of the matched regions. This script
therefore defines one field of view per matched biological region from the
largest annotation box across its five scanner views, adds a fixed margin, and
uses that same level-0 field-of-view size for every scanner. It performs a
geometry audit for all 1,243 regions and reads only a representative subset of
TIFF pixels for a montage. No model features are extracted.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

try:
    import tifffile
    import zarr
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "Missing TIFF dependencies. Install with: "
        "python -m pip install --upgrade tifffile zarr imagecodecs pillow"
    ) from exc


SCANNERS = ("cs2", "gt450", "nz20", "nz210", "p1000")


def parse_bbox(value: Any) -> tuple[float, float, float, float]:
    parsed = json.loads(value) if isinstance(value, str) else value
    if not isinstance(parsed, list) or len(parsed) != 4:
        raise ValueError(f"Invalid COCO bbox: {value!r}")
    x, y, width, height = (float(item) for item in parsed)
    if width <= 0 or height <= 0:
        raise ValueError(f"Non-positive COCO bbox: {value!r}")
    return x, y, width, height


def quantize_side(value: float, quantum: int) -> int:
    return max(quantum, int(math.ceil(value / quantum) * quantum))


def build_crop_plan(
    frame: pd.DataFrame,
    *,
    margin: float,
    quantum: int,
    minimum_side: int,
) -> pd.DataFrame:
    parsed = frame["bbox"].map(parse_bbox)
    result = frame.copy()
    result[["bbox_x", "bbox_y", "bbox_width", "bbox_height"]] = pd.DataFrame(
        parsed.tolist(), index=result.index
    )
    result["bbox_center_x"] = result["bbox_x"] + result["bbox_width"] / 2.0
    result["bbox_center_y"] = result["bbox_y"] + result["bbox_height"] / 2.0
    result["bbox_max_side"] = result[["bbox_width", "bbox_height"]].max(axis=1)
    result["image_width"] = pd.to_numeric(result["image_width"], errors="raise")
    result["image_height"] = pd.to_numeric(result["image_height"], errors="raise")

    region_side = (
        result.groupby("region_id")["bbox_max_side"].max() * float(margin)
    ).map(lambda value: max(minimum_side, quantize_side(value, quantum)))
    result["adaptive_crop_side_level0"] = result["region_id"].map(region_side)

    half = result["adaptive_crop_side_level0"] / 2.0
    x0 = result["bbox_center_x"] - half
    y0 = result["bbox_center_y"] - half
    x1 = result["bbox_center_x"] + half
    y1 = result["bbox_center_y"] + half
    clipped_width = (np.minimum(x1, result["image_width"]) - np.maximum(x0, 0)).clip(lower=0)
    clipped_height = (np.minimum(y1, result["image_height"]) - np.maximum(y0, 0)).clip(lower=0)
    requested_area = result["adaptive_crop_side_level0"] ** 2
    result["inside_image_fraction"] = clipped_width * clipped_height / requested_area
    result["padding_fraction"] = 1.0 - result["inside_image_fraction"]
    result["crop_x0_level0"] = x0
    result["crop_y0_level0"] = y0
    return result


def level_metadata(path: Path) -> list[dict[str, Any]]:
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        levels = list(getattr(series, "levels", [series]))
        base_shape = levels[0].shape
        base_axes = levels[0].axes
        y_index = base_axes.index("Y")
        x_index = base_axes.index("X")
        base_height = int(base_shape[y_index])
        base_width = int(base_shape[x_index])
        rows: list[dict[str, Any]] = []
        for index, level in enumerate(levels):
            axes = level.axes
            shape = level.shape
            height = int(shape[axes.index("Y")])
            width = int(shape[axes.index("X")])
            rows.append(
                {
                    "level": index,
                    "axes": axes,
                    "shape": "x".join(str(item) for item in shape),
                    "height": height,
                    "width": width,
                    "downsample_x": base_width / width,
                    "downsample_y": base_height / height,
                    "dtype": str(level.dtype),
                }
            )
        return rows


def choose_level(
    levels: list[dict[str, Any]],
    crop_side_level0: int,
    target_read_size: int,
) -> dict[str, Any]:
    return min(
        levels,
        key=lambda item: abs(
            math.log(
                max(crop_side_level0 / float(item["downsample_x"]), 1.0)
                / target_read_size
            )
        ),
    )


def normalize_rgb(array: np.ndarray) -> np.ndarray:
    value = np.asarray(array)
    if value.ndim == 2:
        value = np.repeat(value[..., None], 3, axis=2)
    elif value.ndim > 3:
        value = value.reshape(value.shape[0], value.shape[1], -1)
    if value.shape[2] == 1:
        value = np.repeat(value, 3, axis=2)
    elif value.shape[2] > 3:
        value = value[..., :3]

    if value.dtype == np.uint8:
        return value
    if np.issubdtype(value.dtype, np.integer):
        maximum = float(np.iinfo(value.dtype).max)
        return np.clip(value.astype(np.float32) * (255.0 / maximum), 0, 255).astype(np.uint8)
    finite = value[np.isfinite(value)]
    if finite.size == 0:
        return np.full((*value.shape[:2], 3), 255, dtype=np.uint8)
    low, high = np.percentile(finite, [0.5, 99.5])
    if high <= low:
        high = low + 1.0
    return np.clip((value - low) * 255.0 / (high - low), 0, 255).astype(np.uint8)


def read_adaptive_patch(
    path: Path,
    *,
    center_x_level0: float,
    center_y_level0: float,
    crop_side_level0: int,
    target_read_size: int,
    output_size: int,
) -> tuple[Image.Image, dict[str, Any]]:
    with tifffile.TiffFile(path) as tif:
        series = tif.series[0]
        levels = list(getattr(series, "levels", [series]))
        metadata = level_metadata(path)
        selected = choose_level(metadata, crop_side_level0, target_read_size)
        level_index = int(selected["level"])
        level = levels[level_index]
        axes = level.axes
        y_axis = axes.index("Y")
        x_axis = axes.index("X")
        shape = level.shape
        level_height = int(shape[y_axis])
        level_width = int(shape[x_axis])
        downsample_x = float(selected["downsample_x"])
        downsample_y = float(selected["downsample_y"])

        crop_width = max(1, int(math.ceil(crop_side_level0 / downsample_x)))
        crop_height = max(1, int(math.ceil(crop_side_level0 / downsample_y)))
        center_x = center_x_level0 / downsample_x
        center_y = center_y_level0 / downsample_y
        request_x0 = int(math.floor(center_x - crop_width / 2.0))
        request_y0 = int(math.floor(center_y - crop_height / 2.0))
        request_x1 = request_x0 + crop_width
        request_y1 = request_y0 + crop_height

        source_x0 = max(0, request_x0)
        source_y0 = max(0, request_y0)
        source_x1 = min(level_width, request_x1)
        source_y1 = min(level_height, request_y1)

        slices: list[Any] = [slice(None)] * len(shape)
        slices[y_axis] = slice(source_y0, source_y1)
        slices[x_axis] = slice(source_x0, source_x1)
        store = level.aszarr()
        try:
            zarray = zarr.open(store, mode="r")
            raw = np.asarray(zarray[tuple(slices)])
        finally:
            close = getattr(store, "close", None)
            if callable(close):
                close()

        raw = np.moveaxis(raw, (y_axis, x_axis), (0, 1))
        raw = normalize_rgb(raw)
        canvas = np.full((crop_height, crop_width, 3), 255, dtype=np.uint8)
        destination_x0 = source_x0 - request_x0
        destination_y0 = source_y0 - request_y0
        destination_x1 = destination_x0 + raw.shape[1]
        destination_y1 = destination_y0 + raw.shape[0]
        canvas[destination_y0:destination_y1, destination_x0:destination_x1] = raw

    image = Image.fromarray(canvas, mode="RGB").resize(
        (output_size, output_size), Image.Resampling.LANCZOS
    )
    padding_fraction = 1.0 - (
        max(source_x1 - source_x0, 0) * max(source_y1 - source_y0, 0)
    ) / float(crop_width * crop_height)
    return image, {
        "selected_level": level_index,
        "downsample_x": downsample_x,
        "downsample_y": downsample_y,
        "read_width": crop_width,
        "read_height": crop_height,
        "padding_fraction_at_level": padding_fraction,
    }


def select_preview_regions(region_table: pd.DataFrame, count: int) -> list[str]:
    ordered = region_table.sort_values(
        ["adaptive_crop_side_level0", "region_id"]
    ).reset_index(drop=True)
    if count >= len(ordered):
        return ordered["region_id"].astype(str).tolist()

    selected: list[str] = []
    for category in sorted(ordered["category_name"].astype(str).unique()):
        group = ordered[ordered["category_name"].astype(str) == category]
        middle = group.iloc[len(group) // 2]
        selected.append(str(middle["region_id"]))
        if len(selected) >= count:
            return selected

    positions = np.linspace(0, len(ordered) - 1, count, dtype=int)
    for position in positions:
        region_id = str(ordered.iloc[int(position)]["region_id"])
        if region_id not in selected:
            selected.append(region_id)
        if len(selected) >= count:
            break
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--margin", type=float, default=1.25)
    parser.add_argument("--minimum-side", type=int, default=224)
    parser.add_argument("--quantum", type=int, default=16)
    parser.add_argument("--target-read-size", type=int, default=768)
    parser.add_argument("--output-size", type=int, default=256)
    parser.add_argument("--preview-regions", type=int, default=15)
    args = parser.parse_args()

    if args.margin < 1.0:
        raise ValueError("margin must be at least 1.0")
    frame = pd.read_csv(args.manifest)
    frame["scanner_id"] = frame["scanner_id"].astype(str).str.lower()
    if set(frame["scanner_id"].unique()) != set(SCANNERS):
        raise RuntimeError("Unexpected scanner set")
    plan = build_crop_plan(
        frame,
        margin=args.margin,
        quantum=args.quantum,
        minimum_side=args.minimum_side,
    )
    args.out_dir.mkdir(parents=True, exist_ok=True)
    plan.to_csv(args.out_dir / "adaptive_crop_plan.csv", index=False)

    region_table = (
        plan.sort_values(["region_id", "scanner_id"])
        .groupby("region_id", as_index=False)
        .agg(
            sample_id=("sample_id", "first"),
            category_name=("category_name", "first"),
            adaptive_crop_side_level0=("adaptive_crop_side_level0", "first"),
            maximum_padding_fraction=("padding_fraction", "max"),
        )
    )
    preview_ids = select_preview_regions(region_table, args.preview_regions)

    pyramid_rows: list[dict[str, Any]] = []
    for scanner in SCANNERS:
        first = plan[plan["scanner_id"] == scanner].iloc[0]
        path = args.dataset_root / str(first["file_name"])
        for item in level_metadata(path):
            pyramid_rows.append({"scanner_id": scanner, "example_file": str(path), **item})
    pd.DataFrame(pyramid_rows).to_csv(args.out_dir / "pyramid_metadata.csv", index=False)

    label_width = 290
    header_height = 44
    row_height = args.output_size + 28
    montage = Image.new(
        "RGB",
        (label_width + len(SCANNERS) * args.output_size, header_height + len(preview_ids) * row_height),
        "white",
    )
    draw = ImageDraw.Draw(montage)
    font = ImageFont.load_default()
    for column, scanner in enumerate(SCANNERS):
        draw.text(
            (label_width + column * args.output_size + 8, 14),
            scanner,
            fill="black",
            font=font,
        )

    pixel_rows: list[dict[str, Any]] = []
    for row_index, region_id in enumerate(preview_ids):
        group = plan[plan["region_id"].astype(str) == region_id].copy()
        if len(group) != len(SCANNERS):
            raise RuntimeError(f"Region {region_id} does not have five views")
        reference = group.iloc[0]
        y = header_height + row_index * row_height
        label = (
            f"{region_id}\n{reference['sample_id']} | {reference['category_name']}\n"
            f"side={int(reference['adaptive_crop_side_level0'])} px"
        )
        draw.multiline_text((8, y + 8), label, fill="black", font=font, spacing=3)

        for column, scanner in enumerate(SCANNERS):
            item = group[group["scanner_id"] == scanner].iloc[0]
            path = args.dataset_root / str(item["file_name"])
            if not path.is_file():
                raise FileNotFoundError(path)
            patch, metadata = read_adaptive_patch(
                path,
                center_x_level0=float(item["bbox_center_x"]),
                center_y_level0=float(item["bbox_center_y"]),
                crop_side_level0=int(item["adaptive_crop_side_level0"]),
                target_read_size=args.target_read_size,
                output_size=args.output_size,
            )
            montage.paste(patch, (label_width + column * args.output_size, y))
            pixel_rows.append(
                {
                    "region_id": region_id,
                    "sample_id": item["sample_id"],
                    "category_name": item["category_name"],
                    "scanner_id": scanner,
                    "file_name": item["file_name"],
                    "adaptive_crop_side_level0": int(item["adaptive_crop_side_level0"]),
                    "geometry_padding_fraction": float(item["padding_fraction"]),
                    **metadata,
                }
            )

    montage_path = args.out_dir / "adaptive_crop_montage.jpg"
    montage.save(montage_path, quality=94, subsampling=0)
    pd.DataFrame(pixel_rows).to_csv(args.out_dir / "preview_pixel_audit.csv", index=False)

    sides = region_table["adaptive_crop_side_level0"].to_numpy(dtype=float)
    padding = plan["padding_fraction"].to_numpy(dtype=float)
    summary = {
        "status": "adaptive_crop_audit_complete",
        "n_rows": int(len(plan)),
        "n_regions": int(plan["region_id"].nunique()),
        "n_samples": int(plan["sample_id"].nunique()),
        "n_scanners": int(plan["scanner_id"].nunique()),
        "margin": args.margin,
        "minimum_side": args.minimum_side,
        "quantum": args.quantum,
        "target_read_size": args.target_read_size,
        "output_size": args.output_size,
        "crop_side_quantiles": {
            str(q): float(np.quantile(sides, q))
            for q in (0.0, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0)
        },
        "views_requiring_any_padding_fraction": float(np.mean(padding > 0)),
        "views_requiring_more_than_10_percent_padding_fraction": float(np.mean(padding > 0.10)),
        "maximum_padding_fraction": float(np.max(padding)),
        "preview_regions": preview_ids,
        "montage": str(montage_path.resolve()),
        "next_gate": (
            "Visually verify five-view tissue correspondence and field-of-view "
            "coverage before freezing adaptive extraction."
        ),
    }
    (args.out_dir / "adaptive_crop_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print("CANINE SCC ADAPTIVE CROP AUDIT PASSED")
    print(json.dumps(summary, indent=2))
    print(f"Montage: {montage_path.resolve()}")
    print(f"Artifacts: {args.out_dir.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"CANINE SCC ADAPTIVE CROP AUDIT FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
