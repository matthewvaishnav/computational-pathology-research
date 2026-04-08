"""Tests for exporting CAMELYON tile scores from HDF5 feature files."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest

from scripts.data.export_camelyon_tile_scores import (
    compute_tile_scores,
    export_slide_tile_scores,
)
from scripts.data.render_camelyon_heatmap import build_camelyon_heatmap_artifacts


def test_compute_tile_scores_l2_norm():
    features = np.array([[3.0, 4.0], [0.0, 5.0]], dtype=np.float32)
    scores = compute_tile_scores(features, method="l2_norm")
    assert np.allclose(scores, np.array([5.0, 5.0], dtype=np.float32))


def test_compute_tile_scores_feature_index_requires_index():
    features = np.array([[1.0, 2.0]], dtype=np.float32)
    with pytest.raises(ValueError, match="feature_index must be provided"):
        compute_tile_scores(features, method="feature_index")


def test_export_slide_tile_scores_writes_json(tmp_path):
    feature_file = tmp_path / "slide_001.h5"
    with h5py.File(feature_file, "w") as handle:
        handle.create_dataset(
            "features",
            data=np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
        )
        handle.create_dataset(
            "coordinates",
            data=np.array([[0, 0], [8, 8]], dtype=np.int32),
        )

    output_path = tmp_path / "tile_scores.json"
    summary = export_slide_tile_scores(feature_file, output_path, method="mean_activation")

    assert output_path.exists()
    assert summary["num_tiles"] == 2
    assert summary["method"] == "mean_activation"

    saved = json.loads(output_path.read_text(encoding="utf-8"))
    assert saved["tiles"][0]["x"] == 0.0
    assert saved["tiles"][1]["y"] == 8.0
    assert 0.0 <= saved["tiles"][0]["score"] <= 1.0


def test_exported_tile_scores_feed_into_heatmap_renderer(tmp_path):
    feature_file = tmp_path / "slide_001.h5"
    with h5py.File(feature_file, "w") as handle:
        handle.create_dataset(
            "features",
            data=np.array([[0.0, 1.0], [2.0, 3.0]], dtype=np.float32),
        )
        handle.create_dataset(
            "coordinates",
            data=np.array([[0, 0], [8, 8]], dtype=np.int32),
        )

    tile_scores_path = tmp_path / "tile_scores.json"
    export_slide_tile_scores(feature_file, tile_scores_path, method="l2_norm")

    summary = build_camelyon_heatmap_artifacts(
        tile_scores_path=tile_scores_path,
        slide_width=16,
        slide_height=16,
        patch_size=8,
        output_dir=tmp_path / "heatmap_artifacts",
    )

    assert Path(summary["summary_path"]).exists()
    assert Path(summary["artifacts"]["heatmap"]).exists()
    assert summary["num_tiles"] == 2
