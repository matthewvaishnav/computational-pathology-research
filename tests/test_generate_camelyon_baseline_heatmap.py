"""Tests for CAMELYON baseline heatmap generation."""

import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from PIL import Image

from experiments.generate_camelyon_baseline_heatmap import (
    generate_camelyon_baseline_heatmap,
    load_baseline_model,
)
from experiments.run_camelyon_feature_baseline import run_camelyon_feature_baseline
from src.data.camelyon_dataset import CAMELYONSlideIndex, SlideMetadata


@pytest.fixture
def camelyon_feature_setup(tmp_path):
    slides = [
        SlideMetadata("slide_train_0", "patient_0", "/slides/0.tif", 0, "train"),
        SlideMetadata("slide_train_1", "patient_1", "/slides/1.tif", 1, "train"),
        SlideMetadata("slide_val_0", "patient_2", "/slides/2.tif", 0, "val"),
        SlideMetadata("slide_val_1", "patient_3", "/slides/3.tif", 1, "val"),
    ]
    slide_index = CAMELYONSlideIndex(slides)
    slide_index_path = tmp_path / "slide_index.json"
    slide_index.save(slide_index_path)

    features_dir = tmp_path / "features"
    features_dir.mkdir()
    feature_map = {
        "slide_train_0": np.array([[0.1, 0.2], [0.2, 0.1]], dtype=np.float32),
        "slide_train_1": np.array([[2.0, 2.2], [2.1, 2.3]], dtype=np.float32),
        "slide_val_0": np.array([[0.0, 0.1], [0.2, 0.0]], dtype=np.float32),
        "slide_val_1": np.array([[2.3, 2.2], [2.4, 2.5]], dtype=np.float32),
    }
    for slide_id, features in feature_map.items():
        with h5py.File(features_dir / f"{slide_id}.h5", "w") as handle:
            handle.create_dataset("features", data=features)
            handle.create_dataset(
                "coordinates",
                data=np.array([[0, 0], [8, 8]], dtype=np.int32),
            )

    return slide_index_path, features_dir


@pytest.fixture
def trained_camelyon_baseline(camelyon_feature_setup, tmp_path):
    slide_index_path, features_dir = camelyon_feature_setup
    output_dir = tmp_path / "trained_baseline"
    results = run_camelyon_feature_baseline(
        slide_index_path=slide_index_path,
        features_dir=features_dir,
        output_dir=output_dir,
    )
    return {
        "slide_index_path": slide_index_path,
        "features_dir": features_dir,
        "output_dir": output_dir,
        "model_path": Path(results["artifacts"]["model"]),
    }


def test_load_baseline_model_returns_logistic_regression(trained_camelyon_baseline):
    model = load_baseline_model(trained_camelyon_baseline["model_path"])
    assert hasattr(model, "predict_proba")


def test_generate_camelyon_baseline_heatmap_exports_tile_scores(
    trained_camelyon_baseline, tmp_path
):
    summary = generate_camelyon_baseline_heatmap(
        model_path=trained_camelyon_baseline["model_path"],
        feature_file=trained_camelyon_baseline["features_dir"] / "slide_val_1.h5",
        output_dir=tmp_path / "slide_heatmap",
    )

    assert Path(summary["tile_scores_path"]).exists()
    assert Path(summary["summary_path"]).exists()

    saved_summary = json.loads(Path(summary["summary_path"]).read_text(encoding="utf-8"))
    assert saved_summary["num_tiles"] == 2
    assert Path(saved_summary["tile_scores_path"]).exists()


def test_generate_camelyon_baseline_heatmap_can_render_heatmap(trained_camelyon_baseline, tmp_path):
    thumbnail_path = tmp_path / "thumb.png"
    Image.fromarray(np.full((8, 8, 3), 160, dtype=np.uint8)).save(thumbnail_path)

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

    summary = generate_camelyon_baseline_heatmap(
        model_path=trained_camelyon_baseline["model_path"],
        feature_file=trained_camelyon_baseline["features_dir"] / "slide_val_1.h5",
        output_dir=tmp_path / "slide_heatmap_with_overlay",
        slide_width=16,
        slide_height=16,
        patch_size=8,
        thumbnail_path=thumbnail_path,
        annotation_xml_path=annotation_path,
        downsample=1,
    )

    heatmap_summary = summary["heatmap_summary"]
    assert Path(heatmap_summary["artifacts"]["heatmap"]).exists()
    assert Path(heatmap_summary["artifacts"]["overlay"]).exists()
    assert Path(heatmap_summary["artifacts"]["annotation_mask"]).exists()


def test_generate_camelyon_baseline_heatmap_requires_complete_heatmap_geometry(
    trained_camelyon_baseline, tmp_path
):
    with pytest.raises(ValueError, match="must all be provided"):
        generate_camelyon_baseline_heatmap(
            model_path=trained_camelyon_baseline["model_path"],
            feature_file=trained_camelyon_baseline["features_dir"] / "slide_val_1.h5",
            output_dir=tmp_path / "invalid_heatmap",
            slide_width=16,
        )
