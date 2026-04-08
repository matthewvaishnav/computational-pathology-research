"""
Export CAMELYON tile scores from pre-extracted HDF5 patch features.

This script produces tile-level score files compatible with
`scripts/data/render_camelyon_heatmap.py` from the current CAMELYON
feature-cache format. It is a lightweight bridge until full slide-level
inference outputs exist.

Supported score methods:
- l2_norm: score by feature vector magnitude
- mean_activation: score by mean feature value
- feature_index: score by one selected feature dimension
"""

import argparse
import json
from pathlib import Path
from typing import Literal

import h5py
import numpy as np

ScoreMethod = Literal["l2_norm", "mean_activation", "feature_index"]


def compute_tile_scores(
    features: np.ndarray,
    method: ScoreMethod = "l2_norm",
    feature_index: int | None = None,
) -> np.ndarray:
    """Compute scalar tile scores from feature vectors."""
    if features.ndim != 2:
        raise ValueError("features must have shape [num_tiles, feature_dim].")

    if method == "l2_norm":
        return np.linalg.norm(features, axis=1).astype(np.float32)
    if method == "mean_activation":
        return np.mean(features, axis=1).astype(np.float32)
    if method == "feature_index":
        if feature_index is None:
            raise ValueError("feature_index must be provided when method='feature_index'.")
        if feature_index < 0 or feature_index >= features.shape[1]:
            raise ValueError("feature_index is out of bounds for the feature dimension.")
        return features[:, feature_index].astype(np.float32)

    raise ValueError(f"Unsupported score method: {method}")


def export_slide_tile_scores(
    feature_file: str | Path,
    output_path: str | Path,
    method: ScoreMethod = "l2_norm",
    feature_index: int | None = None,
    normalize: bool = True,
) -> dict:
    """Export tile coordinates and scores for one CAMELYON feature file."""
    feature_file = Path(feature_file)
    output_path = Path(output_path)

    with h5py.File(feature_file, "r") as handle:
        features = handle["features"][:]
        coordinates = handle["coordinates"][:]

    scores = compute_tile_scores(features, method=method, feature_index=feature_index)
    if normalize and scores.size > 0:
        min_score = float(scores.min())
        max_score = float(scores.max())
        if max_score > min_score:
            scores = ((scores - min_score) / (max_score - min_score)).astype(np.float32)
        else:
            scores = np.zeros_like(scores, dtype=np.float32)

    tiles = [
        {
            "x": float(coord[0]),
            "y": float(coord[1]),
            "score": float(score),
        }
        for coord, score in zip(coordinates, scores)
    ]

    summary = {
        "source_feature_file": feature_file.as_posix(),
        "method": method,
        "feature_index": feature_index,
        "num_tiles": int(len(tiles)),
        "normalized": bool(normalize),
        "tiles": tiles,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export CAMELYON tile scores from HDF5 feature caches",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/data/export_camelyon_tile_scores.py ^
      --feature-file data/camelyon/features/slide_001.h5 ^
      --output results/camelyon/tile_scores/slide_001.json

  python scripts/data/export_camelyon_tile_scores.py ^
      --feature-file data/camelyon/features/slide_001.h5 ^
      --output results/camelyon/tile_scores/slide_001_feature42.json ^
      --method feature_index --feature-index 42
        """,
    )
    parser.add_argument("--feature-file", required=True, help="Input HDF5 feature file")
    parser.add_argument("--output", required=True, help="Output JSON tile score path")
    parser.add_argument(
        "--method",
        choices=["l2_norm", "mean_activation", "feature_index"],
        default="l2_norm",
        help="Score heuristic to apply to each tile",
    )
    parser.add_argument(
        "--feature-index",
        type=int,
        default=None,
        help="Feature dimension to use when method=feature_index",
    )
    parser.add_argument(
        "--no-normalize",
        action="store_true",
        help="Disable min-max normalization of exported scores",
    )
    args = parser.parse_args()

    summary = export_slide_tile_scores(
        feature_file=args.feature_file,
        output_path=args.output,
        method=args.method,
        feature_index=args.feature_index,
        normalize=not args.no_normalize,
    )
    print(f"Exported {summary['num_tiles']} CAMELYON tile scores to {args.output}")


if __name__ == "__main__":
    main()
