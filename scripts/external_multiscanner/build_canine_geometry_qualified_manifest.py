#!/usr/bin/env python3
"""Build a geometry-qualified canine SCC manifest and normalized montage.

The registration audit shows that matched annotation centers follow highly
accurate sample-specific affine maps. P1000 is consistently rotated by about
-90 degrees relative to CS2, while a minority of released P1000 regions extend
outside the raster. This script therefore:

1. keeps only matched regions whose five adaptive crops require no more than a
   configurable padding fraction;
2. records the inverse sample-specific P1000 rotation needed for canonical
   orientation normalization;
3. preserves the existing biological-sample folds; and
4. generates a corrected five-scanner montage before feature extraction.

No model features are extracted.
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.external_multiscanner.audit_canine_adaptive_crops_zarr3 import (
    read_adaptive_patch_zarr3,
)


SCANNERS = ("cs2", "gt450", "nz20", "nz210", "p1000")


def select_preview_regions(region_table: pd.DataFrame, count: int) -> list[str]:
    if count < 1:
        return []
    ordered = region_table.sort_values(
        ["maximum_padding_fraction", "adaptive_crop_side_level0", "region_id"]
    ).reset_index(drop=True)
    if count >= len(ordered):
        return ordered["region_id"].astype(str).tolist()

    selected: list[str] = []
    for category in sorted(ordered["category_name"].astype(str).unique()):
        group = ordered[ordered["category_name"].astype(str) == category]
        if group.empty:
            continue
        row = group.iloc[len(group) // 2]
        selected.append(str(row["region_id"]))
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


def load_rotation_map(path: Path) -> dict[tuple[str, str], float]:
    frame = pd.read_csv(path)
    required = {"sample_id", "scanner_id", "rotation_degrees"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise RuntimeError(f"Affine transform table is missing columns: {missing}")

    result: dict[tuple[str, str], float] = {}
    for _, row in frame.iterrows():
        sample_id = str(row["sample_id"])
        scanner_id = str(row["scanner_id"]).lower()
        angle = float(row["rotation_degrees"])
        if not math.isfinite(angle):
            raise RuntimeError(
                f"Non-finite rotation for sample={sample_id} scanner={scanner_id}"
            )
        result[(sample_id, scanner_id)] = angle
    return result


def make_montage(
    plan: pd.DataFrame,
    *,
    dataset_root: Path,
    preview_ids: list[str],
    target_read_size: int,
    output_size: int,
    out_path: Path,
) -> pd.DataFrame:
    label_width = 320
    header_height = 44
    row_height = output_size + 30
    montage = Image.new(
        "RGB",
        (
            label_width + len(SCANNERS) * output_size,
            header_height + len(preview_ids) * row_height,
        ),
        "white",
    )
    draw = ImageDraw.Draw(montage)
    font = ImageFont.load_default()
    for column, scanner in enumerate(SCANNERS):
        draw.text(
            (label_width + column * output_size + 8, 14),
            scanner,
            fill="black",
            font=font,
        )

    audit_rows: list[dict[str, Any]] = []
    for row_index, region_id in enumerate(preview_ids):
        group = plan[plan["region_id"].astype(str) == region_id].copy()
        if len(group) != len(SCANNERS):
            raise RuntimeError(f"Region {region_id} does not have five scanner views")
        reference = group.iloc[0]
        y = header_height + row_index * row_height
        label = (
            f"{region_id}\n{reference['sample_id']} | {reference['category_name']}\n"
            f"side={int(reference['adaptive_crop_side_level0'])} px | "
            f"maxpad={float(reference['region_max_padding_fraction']):.3f}"
        )
        draw.multiline_text((8, y + 8), label, fill="black", font=font, spacing=3)

        for column, scanner in enumerate(SCANNERS):
            item = group[group["scanner_id"] == scanner].iloc[0]
            path = dataset_root / str(item["file_name"])
            if not path.is_file():
                raise FileNotFoundError(path)
            patch, metadata = read_adaptive_patch_zarr3(
                path,
                center_x_level0=float(item["bbox_center_x"]),
                center_y_level0=float(item["bbox_center_y"]),
                crop_side_level0=int(item["adaptive_crop_side_level0"]),
                target_read_size=target_read_size,
                output_size=output_size,
            )
            angle = float(item["orientation_normalization_degrees"])
            if abs(angle) > 1e-6:
                patch = patch.rotate(
                    angle,
                    resample=Image.Resampling.BICUBIC,
                    expand=False,
                    fillcolor="white",
                )
            montage.paste(patch, (label_width + column * output_size, y))
            audit_rows.append(
                {
                    "region_id": region_id,
                    "sample_id": item["sample_id"],
                    "category_name": item["category_name"],
                    "scanner_id": scanner,
                    "file_name": item["file_name"],
                    "adaptive_crop_side_level0": int(item["adaptive_crop_side_level0"]),
                    "geometry_padding_fraction": float(item["padding_fraction"]),
                    "orientation_normalization_degrees": angle,
                    **metadata,
                }
            )

    montage.save(out_path, quality=94, subsampling=0)
    return pd.DataFrame(audit_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adaptive-plan", type=Path, required=True)
    parser.add_argument("--affine-transforms", type=Path, required=True)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--max-padding-fraction", type=float, default=0.10)
    parser.add_argument("--preview-regions", type=int, default=15)
    parser.add_argument("--target-read-size", type=int, default=768)
    parser.add_argument("--output-size", type=int, default=256)
    args = parser.parse_args()

    if not 0.0 <= args.max_padding_fraction < 1.0:
        raise ValueError("max-padding-fraction must be in [0, 1)")

    plan = pd.read_csv(args.adaptive_plan)
    required = {
        "sample_id",
        "region_id",
        "scanner_id",
        "file_name",
        "category_name",
        "fold",
        "bbox_center_x",
        "bbox_center_y",
        "adaptive_crop_side_level0",
        "padding_fraction",
    }
    missing = sorted(required - set(plan.columns))
    if missing:
        raise RuntimeError(f"Adaptive crop plan is missing columns: {missing}")

    plan["scanner_id"] = plan["scanner_id"].astype(str).str.lower()
    if set(plan["scanner_id"].unique()) != set(SCANNERS):
        raise RuntimeError("Unexpected scanner set")
    if plan["region_id"].nunique() != 1243 or len(plan) != 6215:
        raise RuntimeError("Unexpected adaptive-plan size")

    rotation_map = load_rotation_map(args.affine_transforms)
    normalization_angles: list[float] = []
    for _, row in plan.iterrows():
        key = (str(row["sample_id"]), str(row["scanner_id"]).lower())
        if key not in rotation_map:
            raise RuntimeError(f"Missing affine rotation for {key}")
        # Only P1000 requires material orientation correction. Other scanners
        # differ from CS2 by less than about 0.4 degrees and are left untouched
        # to avoid unnecessary resampling.
        normalization_angles.append(
            -rotation_map[key] if key[1] == "p1000" else 0.0
        )
    plan["orientation_normalization_degrees"] = normalization_angles

    region_padding = plan.groupby("region_id")["padding_fraction"].max()
    plan["region_max_padding_fraction"] = plan["region_id"].map(region_padding)
    qualified_ids = region_padding[
        region_padding <= args.max_padding_fraction + 1e-12
    ].index.astype(str)
    qualified_set = set(qualified_ids)
    qualified = plan[plan["region_id"].astype(str).isin(qualified_set)].copy()
    excluded = (
        plan.drop_duplicates("region_id")
        .loc[lambda frame: ~frame["region_id"].astype(str).isin(qualified_set)]
        .copy()
    )

    if qualified.empty:
        raise RuntimeError("No regions pass the geometry qualification threshold")
    if not (qualified.groupby("region_id").size() == len(SCANNERS)).all():
        raise RuntimeError("Qualified regions do not contain exactly five views")

    fold_summary = (
        qualified.drop_duplicates("region_id")
        .groupby("fold")
        .agg(
            regions=("region_id", "nunique"),
            samples=("sample_id", "nunique"),
        )
        .reindex(range(5), fill_value=0)
        .reset_index()
    )
    if (fold_summary["regions"] == 0).any():
        raise RuntimeError("Geometry qualification emptied at least one fold")

    sample_summary = (
        qualified.drop_duplicates("region_id")
        .groupby("sample_id")
        .agg(regions=("region_id", "nunique"), fold=("fold", "first"))
        .reset_index()
    )
    category_summary = (
        qualified.drop_duplicates("region_id")
        .groupby("category_name")
        .agg(regions=("region_id", "nunique"))
        .reset_index()
        .sort_values("regions", ascending=False)
    )

    region_table = (
        qualified.sort_values(["region_id", "scanner_id"])
        .groupby("region_id", as_index=False)
        .agg(
            sample_id=("sample_id", "first"),
            category_name=("category_name", "first"),
            fold=("fold", "first"),
            adaptive_crop_side_level0=("adaptive_crop_side_level0", "first"),
            maximum_padding_fraction=("padding_fraction", "max"),
        )
    )
    preview_ids = select_preview_regions(region_table, args.preview_regions)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    qualified.to_csv(args.out_dir / "geometry_qualified_manifest.csv", index=False)
    excluded.to_csv(args.out_dir / "excluded_regions.csv", index=False)
    fold_summary.to_csv(args.out_dir / "fold_summary.csv", index=False)
    sample_summary.to_csv(args.out_dir / "sample_summary.csv", index=False)
    category_summary.to_csv(args.out_dir / "category_summary.csv", index=False)

    montage_path = args.out_dir / "orientation_normalized_montage.jpg"
    preview_audit = make_montage(
        qualified,
        dataset_root=args.dataset_root,
        preview_ids=preview_ids,
        target_read_size=args.target_read_size,
        output_size=args.output_size,
        out_path=montage_path,
    )
    preview_audit.to_csv(args.out_dir / "preview_pixel_audit.csv", index=False)

    p1000_angles = qualified.loc[
        qualified["scanner_id"] == "p1000",
        "orientation_normalization_degrees",
    ].to_numpy(dtype=float)
    summary = {
        "status": "canine_geometry_qualified_manifest_complete",
        "source_rows": int(len(plan)),
        "source_regions": int(plan["region_id"].nunique()),
        "qualified_rows": int(len(qualified)),
        "qualified_regions": int(qualified["region_id"].nunique()),
        "qualified_region_fraction": float(
            qualified["region_id"].nunique() / plan["region_id"].nunique()
        ),
        "qualified_samples": int(qualified["sample_id"].nunique()),
        "max_padding_fraction": args.max_padding_fraction,
        "folds": fold_summary.to_dict(orient="records"),
        "p1000_normalization_angle_median": float(np.median(p1000_angles)),
        "p1000_normalization_angle_q05": float(np.quantile(p1000_angles, 0.05)),
        "p1000_normalization_angle_q95": float(np.quantile(p1000_angles, 0.95)),
        "preview_regions": preview_ids,
        "montage": str(montage_path.resolve()),
        "next_gate": (
            "Visually verify that P1000 now matches the canonical tissue orientation "
            "and that all five columns show the same biological region."
        ),
    }
    (args.out_dir / "geometry_qualified_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    print("CANINE SCC GEOMETRY-QUALIFIED MANIFEST PASSED")
    print(json.dumps(summary, indent=2))
    print("\nFOLD SUMMARY")
    print(fold_summary.to_string(index=False))
    print(f"Montage: {montage_path.resolve()}")
    print(f"Artifacts: {args.out_dir.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (OSError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"CANINE SCC GEOMETRY-QUALIFIED MANIFEST FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
