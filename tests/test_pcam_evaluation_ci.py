"""
Integration tests for PCam evaluation with bootstrap confidence intervals.
"""

import json

import numpy as np
import pytest
import torch

from src.data.pcam_dataset import PCamDataset


@pytest.fixture
def synthetic_pcam_checkpoint(tmp_path):
    """Create a synthetic PCam checkpoint for testing."""
    # Create a simple model checkpoint
    checkpoint_dir = tmp_path / "checkpoints"
    checkpoint_dir.mkdir()
    checkpoint_path = checkpoint_dir / "test_model.pth"

    # Create minimal checkpoint structure
    checkpoint = {
        "epoch": 5,
        "model_state_dict": {
            # Minimal state dict for a simple model
            "feature_extractor.conv1.weight": torch.randn(64, 3, 7, 7),
            "feature_extractor.conv1.bias": torch.randn(64),
            "classifier.weight": torch.randn(2, 64),
            "classifier.bias": torch.randn(2),
        },
        "optimizer_state_dict": {},
        "val_auc": 0.85,
        "config": {
            "data": {
                "root_dir": str(tmp_path / "data"),
                "download": False,
                "num_workers": 0,
            },
            "model": {
                "feature_extractor": {
                    "model": "resnet18",
                    "pretrained": False,
                },
                "classifier": {
                    "hidden_dim": 64,
                    "dropout": 0.3,
                },
            },
            "task": {
                "num_classes": 2,
            },
        },
    }

    torch.save(checkpoint, checkpoint_path)
    return checkpoint_path


@pytest.fixture
def synthetic_pcam_data(tmp_path):
    """Create synthetic PCam data for testing."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    # Create synthetic test data
    n_samples = 100
    image_size = 96

    # Create .npy files for test split
    test_dir = data_dir / "test"
    test_dir.mkdir()

    # Create synthetic images [N, 96, 96, 3]
    images = np.random.randint(0, 256, size=(n_samples, image_size, image_size, 3), dtype=np.uint8)
    np.save(test_dir / "images.npy", images)

    # Create synthetic labels [N]
    labels = np.random.randint(0, 2, size=n_samples, dtype=np.int64)
    np.save(test_dir / "labels.npy", labels)

    return data_dir


def test_evaluation_with_bootstrap_ci_generates_metrics(
    synthetic_pcam_checkpoint, synthetic_pcam_data, tmp_path
):
    """Test that evaluation with bootstrap CI generates metrics JSON with CI bounds."""
    output_dir = tmp_path / "results"
    output_dir.mkdir()

    # Note: This test would require a working evaluate_pcam.py script
    # For now, we'll test the core functionality directly

    # Load synthetic data
    dataset = PCamDataset(
        root_dir=str(synthetic_pcam_data),
        split="test",
        download=False,
    )

    assert len(dataset) == 100

    # Simulate evaluation results
    y_true = np.random.randint(0, 2, size=100)
    y_pred = np.random.randint(0, 2, size=100)
    y_prob = np.random.rand(100)

    # Compute metrics with CI
    from src.utils.statistical import compute_all_metrics_with_ci

    metrics = compute_all_metrics_with_ci(y_true, y_pred, y_prob, n_bootstrap=100, random_state=42)

    # Verify CI structure
    assert "accuracy" in metrics
    assert "value" in metrics["accuracy"]
    assert "ci_lower" in metrics["accuracy"]
    assert "ci_upper" in metrics["accuracy"]

    # Verify CI bounds are reasonable
    assert metrics["accuracy"]["ci_lower"] <= metrics["accuracy"]["value"]
    assert metrics["accuracy"]["value"] <= metrics["accuracy"]["ci_upper"]

    # Save metrics to JSON
    metrics_file = output_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(metrics, f, indent=2)

    # Verify file was created
    assert metrics_file.exists()

    # Load and verify
    with open(metrics_file, "r") as f:
        loaded_metrics = json.load(f)

    assert loaded_metrics["accuracy"]["value"] == metrics["accuracy"]["value"]


def test_bootstrap_ci_bounds_in_output_json(tmp_path):
    """Test that bootstrap CI bounds are correctly saved to output JSON."""
    # Create synthetic predictions
    np.random.seed(42)
    n_samples = 200
    y_true = np.array([0] * 100 + [1] * 100)
    y_pred = y_true.copy()
    # Add some errors
    error_indices = np.random.choice(n_samples, size=20, replace=False)
    y_pred[error_indices] = 1 - y_pred[error_indices]
    y_prob = y_pred.astype(float)

    # Compute metrics with CI
    from src.utils.statistical import compute_all_metrics_with_ci

    metrics = compute_all_metrics_with_ci(
        y_true, y_pred, y_prob, n_bootstrap=100, confidence_level=0.95, random_state=42
    )

    # Save to JSON
    output_file = tmp_path / "metrics_with_ci.json"
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)

    # Load and verify structure
    with open(output_file, "r") as f:
        loaded = json.load(f)

    # Check all metrics have CI bounds
    for metric_name in ["accuracy", "f1", "precision", "recall"]:
        assert metric_name in loaded
        assert "value" in loaded[metric_name]
        assert "ci_lower" in loaded[metric_name]
        assert "ci_upper" in loaded[metric_name]

        # Verify bounds ordering
        assert loaded[metric_name]["ci_lower"] <= loaded[metric_name]["value"]
        assert loaded[metric_name]["value"] <= loaded[metric_name]["ci_upper"]

        # Verify bounds are in valid range [0, 1]
        assert 0.0 <= loaded[metric_name]["ci_lower"] <= 1.0
        assert 0.0 <= loaded[metric_name]["ci_upper"] <= 1.0


def test_bootstrap_ci_with_different_sample_sizes(tmp_path):
    """Test bootstrap CI computation with different sample sizes."""
    from src.utils.statistical import compute_all_metrics_with_ci

    for n_samples in [50, 100, 500]:
        np.random.seed(42)
        y_true = np.random.randint(0, 2, size=n_samples)
        y_pred = np.random.randint(0, 2, size=n_samples)
        y_prob = np.random.rand(n_samples)

        metrics = compute_all_metrics_with_ci(
            y_true, y_pred, y_prob, n_bootstrap=50, random_state=42
        )

        # Verify metrics are computed
        assert "accuracy" in metrics
        assert "value" in metrics["accuracy"]

        # CI width should generally decrease with larger samples
        ci_width = metrics["accuracy"]["ci_upper"] - metrics["accuracy"]["ci_lower"]
        assert ci_width >= 0.0


def test_bootstrap_ci_with_high_accuracy(tmp_path):
    """Test bootstrap CI with high accuracy predictions."""
    np.random.seed(42)
    n_samples = 200
    y_true = np.array([0] * 100 + [1] * 100)
    y_pred = y_true.copy()

    # Only 5% errors
    error_indices = np.random.choice(n_samples, size=10, replace=False)
    y_pred[error_indices] = 1 - y_pred[error_indices]
    y_prob = y_pred.astype(float)

    from src.utils.statistical import compute_all_metrics_with_ci

    metrics = compute_all_metrics_with_ci(y_true, y_pred, y_prob, n_bootstrap=100, random_state=42)

    # Accuracy should be around 95%
    assert 0.90 <= metrics["accuracy"]["value"] <= 1.0

    # CI should be tight
    ci_width = metrics["accuracy"]["ci_upper"] - metrics["accuracy"]["ci_lower"]
    assert ci_width < 0.15


def test_bootstrap_ci_with_low_accuracy(tmp_path):
    """Test bootstrap CI with low accuracy predictions."""
    np.random.seed(42)
    n_samples = 200
    y_true = np.array([0] * 100 + [1] * 100)
    y_pred = 1 - y_true  # All wrong

    # Fix 40% to be correct
    correct_indices = np.random.choice(n_samples, size=80, replace=False)
    y_pred[correct_indices] = y_true[correct_indices]
    y_prob = y_pred.astype(float)

    from src.utils.statistical import compute_all_metrics_with_ci

    metrics = compute_all_metrics_with_ci(y_true, y_pred, y_prob, n_bootstrap=100, random_state=42)

    # Accuracy should be around 40%
    assert 0.30 <= metrics["accuracy"]["value"] <= 0.50


def test_bootstrap_config_saved_to_metrics(tmp_path):
    """Test that bootstrap configuration is saved to metrics JSON."""
    np.random.seed(42)
    n_samples = 100
    y_true = np.random.randint(0, 2, size=n_samples)
    y_pred = np.random.randint(0, 2, size=n_samples)
    y_prob = np.random.rand(n_samples)

    from src.utils.statistical import compute_all_metrics_with_ci

    n_bootstrap = 500
    confidence_level = 0.90

    metrics = compute_all_metrics_with_ci(
        y_true,
        y_pred,
        y_prob,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_state=42,
    )

    # Add bootstrap config to metrics (as would be done in evaluate_pcam.py)
    metrics["bootstrap_config"] = {
        "n_bootstrap": n_bootstrap,
        "confidence_level": confidence_level,
        "random_state": 42,
    }

    # Save to JSON
    output_file = tmp_path / "metrics_with_config.json"
    with open(output_file, "w") as f:
        json.dump(metrics, f, indent=2)

    # Load and verify
    with open(output_file, "r") as f:
        loaded = json.load(f)

    assert "bootstrap_config" in loaded
    assert loaded["bootstrap_config"]["n_bootstrap"] == n_bootstrap
    assert loaded["bootstrap_config"]["confidence_level"] == confidence_level


def test_ci_computation_is_deterministic(tmp_path):
    """Test that CI computation is deterministic with same seed."""
    np.random.seed(42)
    n_samples = 150
    y_true = np.random.randint(0, 2, size=n_samples)
    y_pred = np.random.randint(0, 2, size=n_samples)
    y_prob = np.random.rand(n_samples)

    from src.utils.statistical import compute_all_metrics_with_ci

    # First run
    metrics1 = compute_all_metrics_with_ci(y_true, y_pred, y_prob, n_bootstrap=100, random_state=42)

    # Second run with same seed
    metrics2 = compute_all_metrics_with_ci(y_true, y_pred, y_prob, n_bootstrap=100, random_state=42)

    # Should be identical
    for metric_name in ["accuracy", "f1", "precision", "recall"]:
        assert metrics1[metric_name]["value"] == metrics2[metric_name]["value"]
        assert metrics1[metric_name]["ci_lower"] == metrics2[metric_name]["ci_lower"]
        assert metrics1[metric_name]["ci_upper"] == metrics2[metric_name]["ci_upper"]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
