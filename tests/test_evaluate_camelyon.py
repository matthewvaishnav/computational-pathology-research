"""
Tests for CAMELYON evaluation script.
"""

import json
import subprocess
import tempfile
from pathlib import Path

import pytest


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


def test_evaluate_camelyon_aggregation_methods():
    """Test different aggregation methods (mean, max)."""
    checkpoint_path = Path("checkpoints/camelyon_quick_test/best_model.pth")

    # Skip if checkpoint doesn't exist
    if not checkpoint_path.exists():
        pytest.skip("Quick test checkpoint not found")

    for aggregation in ["mean", "max"]:
        with tempfile.TemporaryDirectory() as tmp_dir:
            output_dir = Path(tmp_dir) / f"eval_{aggregation}"

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
                    "--aggregation",
                    aggregation,
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0, f"Evaluation with {aggregation} failed: {result.stderr}"

            # Verify aggregation method is recorded
            metrics_path = output_dir / "metrics.json"
            with open(metrics_path, "r") as f:
                metrics = json.load(f)

            assert metrics["aggregation_method"] == aggregation


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
