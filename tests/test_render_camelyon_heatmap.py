"""Tests for CAMELYON heatmap rendering script helpers."""

import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.data.render_camelyon_heatmap import (
    build_camelyon_heatmap_artifacts,
    load_tile_scores,
)


def test_load_tile_scores_supports_json_tiles_key(tmp_path):
    tile_scores_path = tmp_path / "tiles.json"
    tile_scores_path.write_text(
        json.dumps({"tiles": [{"x": 0, "y": 0, "score": 0.2}, {"x": 8, "y": 8, "score": 0.9}]}),
        encoding="utf-8",
    )

    coordinates, scores = load_tile_scores(tile_scores_path)

    assert coordinates.shape == (2, 2)
    assert np.allclose(scores, np.array([0.2, 0.9], dtype=np.float32))


def test_load_tile_scores_supports_csv_row_col(tmp_path):
    tile_scores_path = tmp_path / "tiles.csv"
    with open(tile_scores_path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["row", "col", "score"])
        writer.writeheader()
        writer.writerow({"row": 4, "col": 8, "score": 0.5})

    coordinates, scores = load_tile_scores(tile_scores_path, coordinate_order="row_col")

    assert coordinates.tolist() == [[4.0, 8.0]]
    assert scores.tolist() == [0.5]


def test_build_camelyon_heatmap_artifacts_writes_summary_overlay_and_annotation_stats(tmp_path):
    tile_scores_path = tmp_path / "tiles.json"
    tile_scores_path.write_text(
        json.dumps(
            {
                "tiles": [
                    {"x": 0, "y": 0, "score": 0.2},
                    {"x": 8, "y": 8, "score": 0.9},
                ]
            }
        ),
        encoding="utf-8",
    )

    thumbnail_path = tmp_path / "thumb.png"
    Image.fromarray(np.full((8, 8, 3), 180, dtype=np.uint8)).save(thumbnail_path)

    annotation_path = tmp_path / "slide.xml"
    annotation_path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<ASAP_Annotations>
  <Annotations>
    <Annotation Name="Tumor" Type="Polygon">
      <Coordinates>
        <Coordinate Order="0" X="0" Y="0" />
        <Coordinate Order="1" X="8" Y="0" />
        <Coordinate Order="2" X="8" Y="8" />
        <Coordinate Order="3" X="0" Y="8" />
      </Coordinates>
    </Annotation>
  </Annotations>
</ASAP_Annotations>
""",
        encoding="utf-8",
    )

    summary = build_camelyon_heatmap_artifacts(
        tile_scores_path=tile_scores_path,
        slide_width=16,
        slide_height=16,
        patch_size=8,
        output_dir=tmp_path / "artifacts",
        thumbnail_path=thumbnail_path,
        annotation_xml_path=annotation_path,
        downsample=1,
    )

    assert Path(summary["summary_path"]).exists()
    assert Path(summary["artifacts"]["heatmap"]).exists()
    assert Path(summary["artifacts"]["counts"]).exists()
    assert Path(summary["artifacts"]["overlay"]).exists()
    assert Path(summary["artifacts"]["annotation_mask"]).exists()
    assert Path(summary["artifacts"]["annotation_mask_preview"]).exists()
    assert Path(summary["artifacts"]["tile_annotation_stats"]).exists()
    assert summary["num_annotation_polygons"] == 1
    assert summary["num_tiles"] == 2

    saved_summary = json.loads(Path(summary["summary_path"]).read_text(encoding="utf-8"))
    assert saved_summary["thumbnail_path"].endswith("thumb.png")
    assert saved_summary["annotation_xml_path"].endswith("slide.xml")
