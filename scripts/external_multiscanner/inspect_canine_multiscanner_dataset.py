#!/usr/bin/env python3
"""Inventory and probe the external multi-scanner canine cSCC release.

This script intentionally does not guess biological pairings. It records the
release layout, file types, candidate scanner/sample path components, image
readability, dimensions, and optional SHA-256 hashes so the exact manifest
builder can be written against observed evidence rather than assumptions.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd


IMAGE_EXTENSIONS = {
    ".svs",
    ".ndpi",
    ".mrxs",
    ".scn",
    ".isyntax",
    ".tif",
    ".tiff",
    ".ome.tif",
    ".ome.tiff",
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
}


class InspectionError(ValueError):
    pass


def normalized_extension(path: Path) -> str:
    lower = path.name.lower()
    for compound in (".ome.tiff", ".ome.tif"):
        if lower.endswith(compound):
            return compound
    return path.suffix.lower() or "<none>"


def sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def try_openslide(path: Path) -> dict[str, object] | None:
    try:
        import openslide  # type: ignore
    except ImportError:
        return None

    try:
        slide = openslide.OpenSlide(str(path))
        try:
            properties = dict(slide.properties)
            return {
                "reader": "openslide",
                "readable": True,
                "width": int(slide.dimensions[0]),
                "height": int(slide.dimensions[1]),
                "level_count": int(slide.level_count),
                "mpp_x": properties.get("openslide.mpp-x"),
                "mpp_y": properties.get("openslide.mpp-y"),
                "vendor": properties.get("openslide.vendor"),
                "objective_power": properties.get("openslide.objective-power"),
                "error": None,
            }
        finally:
            slide.close()
    except Exception as exc:  # reader-specific exceptions vary
        return {
            "reader": "openslide",
            "readable": False,
            "width": None,
            "height": None,
            "level_count": None,
            "mpp_x": None,
            "mpp_y": None,
            "vendor": None,
            "objective_power": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def try_pillow(path: Path) -> dict[str, object]:
    try:
        from PIL import Image
    except ImportError as exc:
        return {
            "reader": "pillow-unavailable",
            "readable": False,
            "width": None,
            "height": None,
            "level_count": None,
            "mpp_x": None,
            "mpp_y": None,
            "vendor": None,
            "objective_power": None,
            "error": f"Pillow import failed: {exc}",
        }

    try:
        with Image.open(path) as image:
            image.verify()
        with Image.open(path) as image:
            width, height = image.size
            mode = image.mode
        return {
            "reader": "pillow",
            "readable": True,
            "width": int(width),
            "height": int(height),
            "level_count": 1,
            "mpp_x": None,
            "mpp_y": None,
            "vendor": None,
            "objective_power": None,
            "mode": mode,
            "error": None,
        }
    except Exception as exc:
        return {
            "reader": "pillow",
            "readable": False,
            "width": None,
            "height": None,
            "level_count": None,
            "mpp_x": None,
            "mpp_y": None,
            "vendor": None,
            "objective_power": None,
            "mode": None,
            "error": f"{type(exc).__name__}: {exc}",
        }


def probe_image(path: Path) -> dict[str, object]:
    extension = normalized_extension(path)
    if extension in IMAGE_EXTENSIONS:
        openslide_result = try_openslide(path)
        if openslide_result is not None and openslide_result["readable"]:
            return openslide_result
        pillow_result = try_pillow(path)
        if pillow_result["readable"]:
            return pillow_result
        if openslide_result is not None:
            pillow_result["error"] = (
                f"OpenSlide: {openslide_result['error']}; "
                f"Pillow: {pillow_result['error']}"
            )
        return pillow_result
    return {
        "reader": "not-an-image-candidate",
        "readable": False,
        "width": None,
        "height": None,
        "level_count": None,
        "mpp_x": None,
        "mpp_y": None,
        "vendor": None,
        "objective_power": None,
        "error": None,
    }


def component_tables(root: Path, files: list[Path]) -> tuple[pd.DataFrame, pd.DataFrame]:
    component_counts: Counter[tuple[int, str]] = Counter()
    component_extensions: defaultdict[tuple[int, str], Counter[str]] = defaultdict(Counter)
    for path in files:
        relative = path.relative_to(root)
        for depth, component in enumerate(relative.parts[:-1]):
            key = (depth, component)
            component_counts[key] += 1
            component_extensions[key][normalized_extension(path)] += 1

    rows = []
    for (depth, component), count in component_counts.most_common():
        rows.append(
            {
                "depth": depth,
                "component": component,
                "file_count_below": count,
                "extension_counts_json": json.dumps(
                    component_extensions[(depth, component)], sort_keys=True
                ),
            }
        )
    components = pd.DataFrame(rows)

    top_rows = []
    for path in files:
        relative = path.relative_to(root)
        top = relative.parts[0] if len(relative.parts) > 1 else "<root-files>"
        top_rows.append(
            {
                "top_component": top,
                "extension": normalized_extension(path),
                "size_bytes": path.stat().st_size,
            }
        )
    top = pd.DataFrame(top_rows)
    if not top.empty:
        top = (
            top.groupby("top_component", as_index=False)
            .agg(
                file_count=("extension", "size"),
                total_size_bytes=("size_bytes", "sum"),
                extension_count=("extension", "nunique"),
            )
            .sort_values(["file_count", "top_component"], ascending=[False, True])
        )
    return components, top


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument(
        "--probe-per-extension",
        type=int,
        default=5,
        help="Maximum image files opened per extension.",
    )
    parser.add_argument(
        "--hash-all",
        action="store_true",
        help="Compute SHA-256 for every file. This can be slow for WSIs.",
    )
    args = parser.parse_args()

    root = args.dataset_root.resolve()
    if not root.is_dir():
        raise InspectionError(f"Dataset root does not exist: {root}")
    if args.probe_per_extension < 0:
        raise InspectionError("--probe-per-extension must be nonnegative.")

    files = sorted(path for path in root.rglob("*") if path.is_file())
    if not files:
        raise InspectionError(f"No files found under {root}")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    inventory_rows = []
    for path in files:
        relative = path.relative_to(root)
        stat = path.stat()
        inventory_rows.append(
            {
                "relative_path": relative.as_posix(),
                "extension": normalized_extension(path),
                "size_bytes": int(stat.st_size),
                "depth": len(relative.parts) - 1,
                "parent": relative.parent.as_posix(),
                "stem": path.stem,
                "sha256": sha256(path) if args.hash_all else None,
            }
        )
    inventory = pd.DataFrame(inventory_rows)
    inventory.to_csv(args.out_dir / "file_inventory.csv", index=False)

    extension_summary = (
        inventory.groupby("extension", as_index=False)
        .agg(
            file_count=("relative_path", "size"),
            total_size_bytes=("size_bytes", "sum"),
            min_size_bytes=("size_bytes", "min"),
            median_size_bytes=("size_bytes", "median"),
            max_size_bytes=("size_bytes", "max"),
        )
        .sort_values(["file_count", "extension"], ascending=[False, True])
    )
    extension_summary.to_csv(args.out_dir / "extension_summary.csv", index=False)

    components, top = component_tables(root, files)
    components.to_csv(args.out_dir / "path_component_summary.csv", index=False)
    top.to_csv(args.out_dir / "top_level_summary.csv", index=False)

    probe_rows = []
    for extension, group in inventory.groupby("extension", sort=True):
        if extension not in IMAGE_EXTENSIONS or args.probe_per_extension == 0:
            continue
        for relative_path in group["relative_path"].head(args.probe_per_extension):
            path = root / relative_path
            result = probe_image(path)
            probe_rows.append(
                {
                    "relative_path": relative_path,
                    "extension": extension,
                    **result,
                }
            )
    probes = pd.DataFrame(probe_rows)
    probes.to_csv(args.out_dir / "image_probe.csv", index=False)

    image_candidates = inventory[inventory["extension"].isin(IMAGE_EXTENSIONS)]
    summary = {
        "dataset_root": str(root),
        "n_files": int(len(inventory)),
        "n_directories": int(sum(1 for path in root.rglob("*") if path.is_dir())),
        "total_size_bytes": int(inventory["size_bytes"].sum()),
        "n_extensions": int(inventory["extension"].nunique()),
        "extension_counts": {
            str(row.extension): int(row.file_count)
            for row in extension_summary.itertuples(index=False)
        },
        "n_image_candidates": int(len(image_candidates)),
        "n_probed_images": int(len(probes)),
        "n_readable_probes": (
            int(probes["readable"].sum()) if not probes.empty else 0
        ),
        "hash_all": bool(args.hash_all),
        "status": "inventory_complete",
        "next_gate": (
            "Review path_component_summary.csv, top_level_summary.csv, and "
            "image_probe.csv before defining scanner/sample/correspondence parsing."
        ),
    }
    (args.out_dir / "inspection_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    print("EXTERNAL MULTI-SCANNER DATASET INSPECTION PASSED")
    print(json.dumps(summary, indent=2, sort_keys=True))
    print(f"Artifacts: {args.out_dir.resolve()}")


if __name__ == "__main__":
    try:
        main()
    except (InspectionError, OSError, RuntimeError) as exc:
        print(f"EXTERNAL MULTI-SCANNER INSPECTION FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
