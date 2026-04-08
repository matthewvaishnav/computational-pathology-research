"""
Render CAMELYON heatmaps from tile-level scores.

This script turns tile coordinates + scores into slide-aligned heatmaps and
optional thumbnail overlays. It can also rasterize CAMELYON XML annotations
to produce annotation-aware summaries for the generated heatmap.

Supported tile score formats:
- JSON list of objects: [{"x": 0, "y": 0, "score": 0.8}, ...]
- JSON object with "tiles": {"tiles": [...]}
- CSV with columns: x,y,score (or row,col,score)
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.data.camelyon_annotations import (
    load_camelyon_annotations,
    rasterize_annotation_mask,
    save_heatmap_overlay,
    score_tiles_from_annotation_mask,
    tile_scores_to_heatmap,
)


def load_tile_scores(
    input_path: str | Path,
    coordinate_order: str = "xy",
) -> tuple[np.ndarray, np.ndarray]:
    """Load tile coordinates and scores from JSON or CSV."""
    input_path = Path(input_path)
    suffix = input_path.suffix.lower()

    if suffix == ".csv":
        return _load_tile_scores_csv(input_path, coordinate_order=coordinate_order)
    if suffix in {".json", ".jsonl"}:
        return _load_tile_scores_json(input_path, coordinate_order=coordinate_order)

    raise ValueError(f"Unsupported tile score format: {input_path.suffix}")


def build_camelyon_heatmap_artifacts(
    tile_scores_path: str | Path,
    slide_width: int,
    slide_height: int,
    patch_size: int,
    output_dir: str | Path,
    *,
    thumbnail_path: str | Path | None = None,
    annotation_xml_path: str | Path | None = None,
    downsample: int = 1,
    coordinate_order: str = "xy",
    positive_threshold: float = 0.25,
    alpha: float = 0.6,
) -> dict[str, Any]:
    """Generate CAMELYON heatmap artifacts and summary metadata."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    coordinates, scores = load_tile_scores(tile_scores_path, coordinate_order=coordinate_order)
    heatmap, counts = tile_scores_to_heatmap(
        coordinates=coordinates,
        scores=scores,
        slide_width=slide_width,
        slide_height=slide_height,
        patch_size=patch_size,
        downsample=downsample,
        coordinate_order=coordinate_order,
    )

    heatmap_path = output_dir / "camelyon_heatmap.npy"
    counts_path = output_dir / "camelyon_heatmap_counts.npy"
    np.save(heatmap_path, heatmap)
    np.save(counts_path, counts)

    summary: dict[str, Any] = {
        "tile_scores_path": Path(tile_scores_path).as_posix(),
        "slide_width": int(slide_width),
        "slide_height": int(slide_height),
        "patch_size": int(patch_size),
        "downsample": int(downsample),
        "num_tiles": int(len(scores)),
        "heatmap_shape": list(heatmap.shape),
        "heatmap_min": float(heatmap.min()) if heatmap.size else 0.0,
        "heatmap_max": float(heatmap.max()) if heatmap.size else 0.0,
        "artifacts": {
            "heatmap": heatmap_path.as_posix(),
            "counts": counts_path.as_posix(),
        },
    }

    if thumbnail_path is not None:
        thumbnail = np.asarray(Image.open(thumbnail_path).convert("RGB"))
        overlay_path = output_dir / "camelyon_heatmap_overlay.png"
        save_heatmap_overlay(
            thumbnail=thumbnail,
            heatmap=heatmap,
            output_path=overlay_path,
            alpha=alpha,
        )
        summary["artifacts"]["overlay"] = overlay_path.as_posix()
        summary["thumbnail_path"] = Path(thumbnail_path).as_posix()

    if annotation_xml_path is not None:
        polygons = load_camelyon_annotations(annotation_xml_path)
        annotation_mask = rasterize_annotation_mask(
            polygons=polygons,
            slide_width=slide_width,
            slide_height=slide_height,
            downsample=downsample,
        )
        mask_path = output_dir / "camelyon_annotation_mask.npy"
        mask_preview_path = output_dir / "camelyon_annotation_mask.png"
        np.save(mask_path, annotation_mask)
        Image.fromarray((annotation_mask.astype(np.uint8) * 255), mode="L").save(mask_preview_path)

        tile_labels, tile_coverage = score_tiles_from_annotation_mask(
            coordinates=coordinates,
            annotation_mask=annotation_mask,
            patch_size=patch_size,
            downsample=downsample,
            coordinate_order=coordinate_order,
            positive_threshold=positive_threshold,
        )
        tile_stats_path = output_dir / "camelyon_tile_annotation_stats.json"
        with open(tile_stats_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "positive_threshold": positive_threshold,
                    "tile_labels": tile_labels.tolist(),
                    "tile_coverage": [float(value) for value in tile_coverage],
                },
                handle,
                indent=2,
            )

        summary["artifacts"]["annotation_mask"] = mask_path.as_posix()
        summary["artifacts"]["annotation_mask_preview"] = mask_preview_path.as_posix()
        summary["artifacts"]["tile_annotation_stats"] = tile_stats_path.as_posix()
        summary["annotation_xml_path"] = Path(annotation_xml_path).as_posix()
        summary["num_annotation_polygons"] = len(polygons)
        summary["num_positive_tiles"] = int(tile_labels.sum())

    summary_path = output_dir / "camelyon_heatmap_summary.json"
    summary["summary_path"] = summary_path.as_posix()
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary


def _load_tile_scores_json(
    input_path: Path,
    coordinate_order: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load tile scores from JSON or JSONL."""
    if input_path.suffix.lower() == ".jsonl":
        with open(input_path, "r", encoding="utf-8") as handle:
            records = [json.loads(line) for line in handle if line.strip()]
    else:
        with open(input_path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
        records = data["tiles"] if isinstance(data, dict) and "tiles" in data else data

    return _records_to_arrays(records, coordinate_order=coordinate_order)


def _load_tile_scores_csv(
    input_path: Path,
    coordinate_order: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Load tile scores from CSV."""
    with open(input_path, "r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        records = list(reader)
    return _records_to_arrays(records, coordinate_order=coordinate_order)


def _records_to_arrays(
    records: list[dict[str, Any]],
    coordinate_order: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Convert score records into coordinate and score arrays."""
    coordinates = []
    scores = []

    for record in records:
        if coordinate_order == "xy":
            x = float(record["x"])
            y = float(record["y"])
            coordinates.append((x, y))
        elif coordinate_order == "row_col":
            row = float(record["row"])
            col = float(record["col"])
            coordinates.append((row, col))
        else:
            raise ValueError(f"Unsupported coordinate_order: {coordinate_order}")
        scores.append(float(record["score"]))

    return np.asarray(coordinates, dtype=np.float32), np.asarray(scores, dtype=np.float32)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Render CAMELYON heatmaps from tile-level scores",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/data/render_camelyon_heatmap.py ^
      --tile-scores results/camelyon/tile_scores.json ^
      --slide-width 80000 --slide-height 60000 --patch-size 256 ^
      --output-dir results/camelyon/heatmaps

  python scripts/data/render_camelyon_heatmap.py ^
      --tile-scores results/camelyon/tile_scores.csv ^
      --thumbnail data/camelyon/thumb.png ^
      --annotation-xml data/camelyon/annotations/slide_001.xml ^
      --slide-width 80000 --slide-height 60000 --patch-size 256 ^
      --downsample 32 --output-dir results/camelyon/heatmaps/slide_001
        """,
    )
    parser.add_argument("--tile-scores", required=True, help="Path to tile score JSON/JSONL/CSV")
    parser.add_argument("--slide-width", type=int, required=True, help="Base-level slide width")
    parser.add_argument("--slide-height", type=int, required=True, help="Base-level slide height")
    parser.add_argument("--patch-size", type=int, required=True, help="Tile size at base level")
    parser.add_argument("--output-dir", required=True, help="Directory for generated artifacts")
    parser.add_argument("--thumbnail", default=None, help="Optional RGB thumbnail image path")
    parser.add_argument(
        "--annotation-xml", default=None, help="Optional CAMELYON XML annotation path"
    )
    parser.add_argument(
        "--downsample", type=int, default=1, help="Downsample factor for heatmap grid"
    )
    parser.add_argument(
        "--coordinate-order",
        choices=["xy", "row_col"],
        default="xy",
        help="Coordinate convention in the tile score file",
    )
    parser.add_argument(
        "--positive-threshold",
        type=float,
        default=0.25,
        help="Annotation coverage threshold for positive tile labels",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.6,
        help="Maximum opacity for the heatmap overlay",
    )
    args = parser.parse_args()

    summary = build_camelyon_heatmap_artifacts(
        tile_scores_path=args.tile_scores,
        slide_width=args.slide_width,
        slide_height=args.slide_height,
        patch_size=args.patch_size,
        output_dir=args.output_dir,
        thumbnail_path=args.thumbnail,
        annotation_xml_path=args.annotation_xml,
        downsample=args.downsample,
        coordinate_order=args.coordinate_order,
        positive_threshold=args.positive_threshold,
        alpha=args.alpha,
    )

    print(f"Saved CAMELYON heatmap summary to {summary['summary_path']}")


if __name__ == "__main__":
    main()
