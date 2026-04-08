"""
Generate CAMELYON heatmaps from a trained slide-level feature baseline.

This script loads the lightweight logistic-regression baseline trained by
``run_camelyon_feature_baseline.py``, exports model-driven tile scores for one
feature-cache slide, and optionally renders heatmaps / overlays / annotation
statistics using the CAMELYON heatmap utilities.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any

from sklearn.linear_model import LogisticRegression

# Add repo root to path for direct script execution.
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.run_camelyon_feature_baseline import export_model_tile_scores
from scripts.data.render_camelyon_heatmap import build_camelyon_heatmap_artifacts


def load_baseline_model(model_path: str | Path) -> LogisticRegression:
    """Load a trained CAMELYON feature baseline pickle."""
    model_path = Path(model_path)
    with open(model_path, "rb") as handle:
        model = pickle.load(handle)
    if not isinstance(model, LogisticRegression):
        raise TypeError(
            f"Expected a scikit-learn LogisticRegression model, got {type(model).__name__}."
        )
    return model


def generate_camelyon_baseline_heatmap(
    *,
    model_path: str | Path,
    feature_file: str | Path,
    output_dir: str | Path,
    slide_width: int | None = None,
    slide_height: int | None = None,
    patch_size: int | None = None,
    thumbnail_path: str | Path | None = None,
    annotation_xml_path: str | Path | None = None,
    downsample: int = 1,
) -> dict[str, Any]:
    """Export model-driven tile scores and optional heatmap artifacts."""
    model = load_baseline_model(model_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tile_scores_path = output_dir / "model_tile_scores.json"
    tile_score_payload = export_model_tile_scores(
        feature_file=feature_file,
        model=model,
        output_path=tile_scores_path,
    )

    summary: dict[str, Any] = {
        "model_path": Path(model_path).as_posix(),
        "feature_file": Path(feature_file).as_posix(),
        "tile_scores_path": tile_scores_path.as_posix(),
        "num_tiles": tile_score_payload["num_tiles"],
    }

    if slide_width is not None or slide_height is not None or patch_size is not None:
        if slide_width is None or slide_height is None or patch_size is None:
            raise ValueError(
                "slide_width, slide_height, and patch_size must all be provided to render heatmaps."
            )
        heatmap_summary = build_camelyon_heatmap_artifacts(
            tile_scores_path=tile_scores_path,
            slide_width=slide_width,
            slide_height=slide_height,
            patch_size=patch_size,
            output_dir=output_dir / "heatmap",
            thumbnail_path=thumbnail_path,
            annotation_xml_path=annotation_xml_path,
            downsample=downsample,
        )
        summary["heatmap_summary"] = heatmap_summary

    summary_path = output_dir / "camelyon_baseline_heatmap_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    summary["summary_path"] = summary_path.as_posix()
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate CAMELYON heatmaps from a trained feature baseline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiments/generate_camelyon_baseline_heatmap.py ^
      --model-path results/camelyon/baseline/camelyon_feature_baseline.pkl ^
      --feature-file data/camelyon/features/tumor_001.h5 ^
      --output-dir results/camelyon/tumor_001_heatmap

  python experiments/generate_camelyon_baseline_heatmap.py ^
      --model-path results/camelyon/baseline/camelyon_feature_baseline.pkl ^
      --feature-file data/camelyon/features/tumor_001.h5 ^
      --output-dir results/camelyon/tumor_001_heatmap ^
      --slide-width 80000 --slide-height 60000 --patch-size 256 ^
      --thumbnail data/camelyon/thumbnails/tumor_001.png ^
      --annotation-xml data/camelyon/annotations/tumor_001.xml ^
      --downsample 32
        """,
    )
    parser.add_argument("--model-path", required=True, help="Path to trained baseline pickle")
    parser.add_argument(
        "--feature-file", required=True, help="Path to one CAMELYON HDF5 feature file"
    )
    parser.add_argument(
        "--output-dir", required=True, help="Directory for tile-score and heatmap artifacts"
    )
    parser.add_argument(
        "--slide-width", type=int, default=None, help="Optional slide width for heatmap rendering"
    )
    parser.add_argument(
        "--slide-height", type=int, default=None, help="Optional slide height for heatmap rendering"
    )
    parser.add_argument(
        "--patch-size", type=int, default=None, help="Patch size for heatmap rendering"
    )
    parser.add_argument(
        "--thumbnail", default=None, help="Optional thumbnail image for heatmap overlay"
    )
    parser.add_argument(
        "--annotation-xml", default=None, help="Optional CAMELYON XML annotation file"
    )
    parser.add_argument(
        "--downsample", type=int, default=1, help="Downsample factor for heatmap rendering"
    )
    args = parser.parse_args()

    summary = generate_camelyon_baseline_heatmap(
        model_path=args.model_path,
        feature_file=args.feature_file,
        output_dir=args.output_dir,
        slide_width=args.slide_width,
        slide_height=args.slide_height,
        patch_size=args.patch_size,
        thumbnail_path=args.thumbnail,
        annotation_xml_path=args.annotation_xml,
        downsample=args.downsample,
    )
    print(f"Saved CAMELYON baseline heatmap summary to {summary['summary_path']}")


if __name__ == "__main__":
    main()
