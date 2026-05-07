"""Tests for CAMELYON slide-level feature baseline helpers."""

import csv
import json
from pathlib import Path

import h5py
import numpy as np
import pytest
from PIL import Image

from experiments.run_camelyon_feature_baseline import (
    PLOT_AVAILABLE,
    collect_slide_level_features,
    evaluate_slide_classifier,
    export_model_tile_scores,
    run_camelyon_feature_baseline,
    train_logistic_baseline,
)
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


def test_collect_slide_level_features_returns_arrays(camelyon_feature_setup):
    slide_index_path, features_dir = camelyon_feature_setup

    features, labels, slide_ids = collect_slide_level_features(
        slide_index_path=slide_index_path,
        features_dir=features_dir,
        split="train",
        aggregation="mean",
    )

    assert features.shape == (2, 2)
    assert labels.tolist() == [0, 1]
    assert slide_ids == ["slide_train_0", "slide_train_1"]


def test_train_logistic_baseline_requires_two_classes():
    with pytest.raises(ValueError, match="at least two classes"):
        train_logistic_baseline(
            features=np.array([[0.1, 0.2], [0.2, 0.1]], dtype=np.float32),
            labels=np.array([0, 0], dtype=np.int64),
        )


def test_run_camelyon_feature_baseline_writes_results_and_model(camelyon_feature_setup, tmp_path):
    slide_index_path, features_dir = camelyon_feature_setup

    results = run_camelyon_feature_baseline(
        slide_index_path=slide_index_path,
        features_dir=features_dir,
        output_dir=tmp_path / "baseline_results",
        export_tile_scores_for_slide="slide_val_1",
    )

    assert Path(results["artifacts"]["model"]).exists()
    assert Path(results["artifacts"]["results"]).exists()
    assert Path(results["artifacts"]["tile_scores"]).exists()
    assert Path(results["artifacts"]["eval_slide_predictions"]).exists()
    assert results["train_metrics"]["accuracy"] >= 0.5
    assert results["eval_metrics"]["num_slides"] == 2

    saved_results = json.loads(Path(results["artifacts"]["results"]).read_text(encoding="utf-8"))
    assert saved_results["aggregation"] == "mean"
    if PLOT_AVAILABLE:
        assert Path(results["artifacts"]["confusion_matrix"]).exists()
        assert Path(results["artifacts"]["roc_curve"]).exists()


def test_run_camelyon_feature_baseline_writes_eval_prediction_csv(camelyon_feature_setup, tmp_path):
    slide_index_path, features_dir = camelyon_feature_setup

    results = run_camelyon_feature_baseline(
        slide_index_path=slide_index_path,
        features_dir=features_dir,
        output_dir=tmp_path / "baseline_predictions",
    )

    prediction_rows = list(
        csv.DictReader(
            Path(results["artifacts"]["eval_slide_predictions"])
            .read_text(encoding="utf-8")
            .splitlines()
        )
    )
    assert len(prediction_rows) == 2
    assert prediction_rows[0]["slide_id"] == "slide_val_0"
    assert prediction_rows[1]["slide_id"] == "slide_val_1"
    assert prediction_rows[0]["label"] in {"0", "1"}
    assert 0.0 <= float(prediction_rows[0]["probability"]) <= 1.0


def test_run_camelyon_feature_baseline_can_render_heatmap(camelyon_feature_setup, tmp_path):
    slide_index_path, features_dir = camelyon_feature_setup

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

    results = run_camelyon_feature_baseline(
        slide_index_path=slide_index_path,
        features_dir=features_dir,
        output_dir=tmp_path / "baseline_results_with_heatmap",
        export_tile_scores_for_slide="slide_val_1",
        heatmap_slide_width=16,
        heatmap_slide_height=16,
        heatmap_patch_size=8,
        heatmap_thumbnail_path=thumbnail_path,
        heatmap_annotation_xml_path=annotation_path,
    )

    assert Path(results["artifacts"]["tile_scores"]).exists()
    assert Path(results["artifacts"]["heatmap_summary"]).exists()

    heatmap_summary = json.loads(
        Path(results["artifacts"]["heatmap_summary"]).read_text(encoding="utf-8")
    )
    assert Path(heatmap_summary["artifacts"]["heatmap"]).exists()
    assert Path(heatmap_summary["artifacts"]["overlay"]).exists()
    assert Path(heatmap_summary["artifacts"]["annotation_mask"]).exists()


def test_export_model_tile_scores_writes_renderer_compatible_json(camelyon_feature_setup, tmp_path):
    slide_index_path, features_dir = camelyon_feature_setup
    train_x, train_y, _ = collect_slide_level_features(
        slide_index_path, features_dir, split="train"
    )
    model = train_logistic_baseline(train_x, train_y)

    output_path = tmp_path / "slide_scores.json"
    payload = export_model_tile_scores(
        feature_file=Path(features_dir) / "slide_val_1.h5",
        model=model,
        output_path=output_path,
    )

    assert output_path.exists()
    assert payload["num_tiles"] == 2
    assert "tiles" in payload
    assert payload["tiles"][0]["x"] == 0.0
    assert 0.0 <= payload["tiles"][0]["score"] <= 1.0


def test_evaluate_slide_classifier_handles_single_class_eval(camelyon_feature_setup):
    slide_index_path, features_dir = camelyon_feature_setup
    train_x, train_y, _ = collect_slide_level_features(
        slide_index_path, features_dir, split="train"
    )
    model = train_logistic_baseline(train_x, train_y)

    features = np.array([[0.1, 0.1]], dtype=np.float32)
    labels = np.array([0], dtype=np.int64)
    metrics = evaluate_slide_classifier(model, features, labels, ["slide_only"])

    assert metrics["auc"] is None
    assert metrics["num_slides"] == 1
    assert metrics["confusion_matrix"] == [[1, 0], [0, 0]]
