"""Tests for CAMELYON annotation and heatmap utilities."""

import numpy as np
import pytest
from PIL import Image

from src.data.camelyon_annotations import (
    AnnotationPolygon,
    load_camelyon_annotations,
    overlay_heatmap_on_thumbnail,
    rasterize_annotation_mask,
    save_heatmap_overlay,
    score_tiles_from_annotation_mask,
    tile_scores_to_heatmap,
)


def test_load_camelyon_annotations_parses_xml(tmp_path):
    annotation_path = tmp_path / "slide.xml"
    annotation_path.write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<ASAP_Annotations>
  <Annotations>
    <Annotation Name="Tumor" Type="Polygon">
      <Coordinates>
        <Coordinate Order="0" X="10" Y="10" />
        <Coordinate Order="1" X="30" Y="10" />
        <Coordinate Order="2" X="30" Y="30" />
        <Coordinate Order="3" X="10" Y="30" />
      </Coordinates>
    </Annotation>
    <Annotation Name="Normal" Type="Polygon">
      <Coordinates>
        <Coordinate Order="0" X="40" Y="40" />
        <Coordinate Order="1" X="50" Y="40" />
        <Coordinate Order="2" X="50" Y="50" />
      </Coordinates>
    </Annotation>
  </Annotations>
</ASAP_Annotations>
""",
        encoding="utf-8",
    )

    polygons = load_camelyon_annotations(annotation_path)

    assert len(polygons) == 2
    assert polygons[0].name == "Tumor"
    assert polygons[0].annotation_type == "Polygon"
    assert polygons[0].coordinates[0] == (10.0, 10.0)
    assert polygons[0].bounds == (10.0, 10.0, 30.0, 30.0)


def test_rasterize_annotation_mask_creates_binary_mask():
    polygons = [
        AnnotationPolygon(
            name="Tumor",
            annotation_type="Polygon",
            coordinates=((10.0, 10.0), (30.0, 10.0), (30.0, 30.0), (10.0, 30.0)),
        )
    ]

    mask = rasterize_annotation_mask(polygons, slide_width=40, slide_height=40, downsample=2)

    assert mask.shape == (20, 20)
    assert mask.dtype == np.uint8
    assert mask[10, 10] == 1
    assert mask[2, 2] == 0


def test_tile_scores_to_heatmap_averages_overlapping_tiles():
    heatmap, counts = tile_scores_to_heatmap(
        coordinates=[(0, 0), (0, 0), (8, 8)],
        scores=[0.2, 0.8, 1.0],
        slide_width=24,
        slide_height=24,
        patch_size=8,
    )

    assert heatmap.shape == (24, 24)
    assert counts.shape == (24, 24)
    assert np.allclose(heatmap[0:8, 0:8], 0.5)
    assert np.all(counts[0:8, 0:8] == 2)
    assert np.allclose(heatmap[8:16, 8:16], 1.0)
    assert np.all(counts[8:16, 8:16] == 1)


def test_score_tiles_from_annotation_mask_returns_labels_and_coverage():
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[0:10, 0:10] = 1
    mask[10:15, 10:15] = 1

    labels, coverage = score_tiles_from_annotation_mask(
        coordinates=[(0, 0), (10, 10), (0, 10)],
        annotation_mask=mask,
        patch_size=10,
        positive_threshold=0.3,
    )

    assert labels.tolist() == [1, 0, 0]
    assert np.isclose(coverage[0], 1.0)
    assert np.isclose(coverage[1], 0.25)
    assert np.isclose(coverage[2], 0.0)


def test_tile_scores_to_heatmap_supports_row_col_coordinates():
    heatmap, counts = tile_scores_to_heatmap(
        coordinates=[(8, 0)],
        scores=[0.75],
        slide_width=24,
        slide_height=24,
        patch_size=8,
        coordinate_order="row_col",
    )

    assert np.allclose(heatmap[8:16, 0:8], 0.75)
    assert np.all(counts[8:16, 0:8] == 1)


def test_score_tiles_from_annotation_mask_validates_threshold():
    with pytest.raises(ValueError, match="positive_threshold"):
        score_tiles_from_annotation_mask(
            coordinates=[(0, 0)],
            annotation_mask=np.zeros((4, 4), dtype=np.uint8),
            patch_size=2,
            positive_threshold=1.5,
        )


def test_overlay_heatmap_on_thumbnail_blends_positive_regions():
    thumbnail = np.full((4, 4, 3), 200, dtype=np.uint8)
    heatmap = np.zeros((4, 4), dtype=np.float32)
    heatmap[1:3, 1:3] = 1.0

    overlay = overlay_heatmap_on_thumbnail(thumbnail, heatmap, alpha=0.8)

    assert overlay.shape == thumbnail.shape
    assert overlay.dtype == np.uint8
    assert np.array_equal(overlay[0, 0], thumbnail[0, 0])
    assert not np.array_equal(overlay[1, 1], thumbnail[1, 1])


def test_save_heatmap_overlay_resizes_heatmap_to_thumbnail(tmp_path):
    thumbnail = np.full((6, 6, 3), 180, dtype=np.uint8)
    heatmap = np.array([[0.0, 1.0], [0.5, 0.0]], dtype=np.float32)

    output_path = save_heatmap_overlay(
        thumbnail=thumbnail,
        heatmap=heatmap,
        output_path=tmp_path / "overlay.png",
    )

    saved = np.asarray(Image.open(output_path))
    assert saved.shape == thumbnail.shape
    assert saved.dtype == np.uint8
