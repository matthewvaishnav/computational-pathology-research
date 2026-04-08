"""
Tests for CAMELYON evaluation script.
"""

import json
import subprocess
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from experiments.evaluate_camelyon import (
    compute_patch_scores,
    evaluate_slide_level,
)
from src.data.camelyon_dataset import CAMELYONSlideIndex, SlideMetadata


def test_evaluate_camelyon_script_exists():
    """Test that CAMELYON evaluation script exists."""
    script_path = Path("experiments/evaluate_camelyon.py")
    assert script_path.exists(), f"Evaluation script not found: {script_path}"


def test_evaluate_camelyon_has_required_functions():
    """Test that evaluation script has required functions."""
    script_path = Path("experiments/evaluate_camelyon.py")

    with open(script_path, "r") as f:
        content = f.read()

    # Check for required functions
    assert "def load_checkpoint" in content
    assert "def evaluate_slide_level" in content
    assert "def compute_metrics" in content
    assert "def save_metrics" in content
    assert "def plot_confusion_matrix" in content
    assert "def plot_roc_curve" in content
    assert "def main()" in content


class DummySlideModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pooling = "mean"
        self.classifier = torch.nn.Linear(2, 1, bias=False)
        with torch.no_grad():
            self.classifier.weight.copy_(torch.tensor([[1.0, 0.0]]))

    def forward(self, patch_features, num_patches=None):
        if num_patches is not None:
            mask = (
                torch.arange(patch_features.size(1), device=patch_features.device)[None, :]
                < num_patches[:, None]
            ).unsqueeze(-1)
            slide_features = (patch_features * mask).sum(dim=1) / num_patches.unsqueeze(-1).float()
        else:
            slide_features = patch_features.mean(dim=1)
        return self.classifier(slide_features)


def test_compute_patch_scores_for_binary_model():
    """Patch scoring should emit one probability per patch."""
    model = DummySlideModel()
    patch_features = torch.tensor([[0.0, 5.0], [2.0, -1.0]], dtype=torch.float32)

    scores = compute_patch_scores(model, patch_features)

    assert scores.shape == (2,)
    assert 0.0 <= scores[0].item() <= 1.0
    assert 0.0 <= scores[1].item() <= 1.0
    assert scores[1].item() > scores[0].item()


def test_evaluate_slide_level_exports_tile_scores(tmp_path):
    """Slide-level evaluation can export heatmap-ready tile-score JSON."""
    slide = SlideMetadata(
        slide_id="tumor_001",
        patient_id="patient_001",
        file_path="/data/tumor_001.tif",
        label=1,
        split="test",
    )
    slide_index = CAMELYONSlideIndex([slide])
    features_dir = tmp_path / "features"
    features_dir.mkdir()

    with h5py.File(features_dir / "tumor_001.h5", "w") as handle:
        handle.create_dataset(
            "features",
            data=np.asarray([[0.0, 1.0], [2.0, 3.0], [4.0, 5.0]], dtype=np.float32),
        )
        handle.create_dataset(
            "coordinates",
            data=np.asarray([[0, 0], [256, 0], [512, 256]], dtype=np.int32),
        )

    tile_scores_dir = tmp_path / "tile_scores"
    metrics = evaluate_slide_level(
        model=DummySlideModel(),
        slide_index=slide_index,
        features_dir=features_dir,
        split="test",
        device="cpu",
        tile_scores_dir=tile_scores_dir,
    )

    assert metrics["num_slides"] == 1
    assert "slide_tile_score_paths" in metrics
    tile_score_path = Path(metrics["slide_tile_score_paths"]["tumor_001"])
    assert tile_score_path.exists()

    payload = json.loads(tile_score_path.read_text(encoding="utf-8"))
    assert payload["slide_id"] == "tumor_001"
    assert payload["num_tiles"] == 3
    assert payload["tiles"][1]["x"] == 256
    assert payload["tiles"][2]["y"] == 256
    assert 0.0 <= payload["tiles"][0]["score"] <= 1.0


def test_evaluate_camelyon_on_quick_test_checkpoint():
    """Test evaluation on the quick test checkpoint."""
    checkpoint_path = Path("checkpoints/camelyon_quick_test/best_model.pth")

    # Skip if checkpoint doesn't exist
    if not checkpoint_path.exists():
        pytest.skip("Quick test checkpoint not found")

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir) / "eval_results"

        result = subprocess.run(
            [
                "python",
                "experiments/evaluate_camelyon.py",
                "--checkpoint",
                str(checkpoint_path),
                "--split",
                "test",
                "--output-dir",
                str(output_dir),
                "--device",
                "cpu",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Evaluation failed: {result.stderr}"

        # Check that metrics file was created
        metrics_path = output_dir / "metrics.json"
        assert metrics_path.exists(), "Metrics file not created"

        # Load and verify metrics
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        # Check required metrics exist
        assert "accuracy" in metrics
        assert "auc" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "confusion_matrix" in metrics
        assert "per_class_metrics" in metrics
        assert "num_slides" in metrics

        # Check metrics are valid
        assert 0 <= metrics["accuracy"] <= 1
        if metrics["auc"] is not None:
            assert 0 <= metrics["auc"] <= 1
        assert 0 <= metrics["precision"] <= 1
        assert 0 <= metrics["recall"] <= 1
        assert 0 <= metrics["f1"] <= 1

        # Check confusion matrix shape
        cm = metrics["confusion_matrix"]
        assert len(cm) == 2
        assert len(cm[0]) == 2
        assert len(cm[1]) == 2


def test_evaluate_camelyon_creates_plots():
    """Test that evaluation creates confusion matrix and ROC curve plots."""
    checkpoint_path = Path("checkpoints/camelyon_quick_test/best_model.pth")

    # Skip if checkpoint doesn't exist
    if not checkpoint_path.exists():
        pytest.skip("Quick test checkpoint not found")

    # Skip if matplotlib not available
    try:
        import matplotlib
        import seaborn
    except ImportError:
        pytest.skip("matplotlib/seaborn not available")

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir) / "eval_results"

        result = subprocess.run(
            [
                "python",
                "experiments/evaluate_camelyon.py",
                "--checkpoint",
                str(checkpoint_path),
                "--split",
                "test",
                "--output-dir",
                str(output_dir),
                "--device",
                "cpu",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

        # Check that plots were created
        cm_path = output_dir / "confusion_matrix.png"
        roc_path = output_dir / "roc_curve.png"

        assert cm_path.exists(), "Confusion matrix plot not created"
        assert roc_path.exists(), "ROC curve plot not created"


def test_evaluate_camelyon_on_val_split():
    """Test evaluation on validation split."""
    checkpoint_path = Path("checkpoints/camelyon_quick_test/best_model.pth")

    # Skip if checkpoint doesn't exist
    if not checkpoint_path.exists():
        pytest.skip("Quick test checkpoint not found")

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir) / "eval_results"

        result = subprocess.run(
            [
                "python",
                "experiments/evaluate_camelyon.py",
                "--checkpoint",
                str(checkpoint_path),
                "--split",
                "val",
                "--output-dir",
                str(output_dir),
                "--device",
                "cpu",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Evaluation failed: {result.stderr}"

        # Check that metrics file was created
        metrics_path = output_dir / "metrics.json"
        assert metrics_path.exists()

        # Verify split is recorded
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        assert metrics["split"] == "val"


def test_evaluate_camelyon_aggregation_from_checkpoint():
    """Test that aggregation method is loaded from checkpoint, not CLI."""
    checkpoint_path = Path("checkpoints/camelyon_quick_test/best_model.pth")

    # Skip if checkpoint doesn't exist
    if not checkpoint_path.exists():
        pytest.skip("Quick test checkpoint not found")

    with tempfile.TemporaryDirectory() as tmp_dir:
        output_dir = Path(tmp_dir) / "eval_results"

        result = subprocess.run(
            [
                "python",
                "experiments/evaluate_camelyon.py",
                "--checkpoint",
                str(checkpoint_path),
                "--split",
                "test",
                "--output-dir",
                str(output_dir),
                "--device",
                "cpu",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0, f"Evaluation failed: {result.stderr}"

        # Verify aggregation method is recorded from checkpoint
        metrics_path = output_dir / "metrics.json"
        with open(metrics_path, "r") as f:
            metrics = json.load(f)

        # Aggregation method should be loaded from checkpoint config
        assert "aggregation_method" in metrics
        assert metrics["aggregation_method"] in ["mean", "max"]


def test_evaluate_camelyon_missing_checkpoint():
    """Test that evaluation fails gracefully with missing checkpoint."""
    result = subprocess.run(
        [
            "python",
            "experiments/evaluate_camelyon.py",
            "--checkpoint",
            "nonexistent_checkpoint.pth",
            "--split",
            "test",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0, "Should fail with missing checkpoint"
    assert "not found" in result.stderr.lower() or "not found" in result.stdout.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
