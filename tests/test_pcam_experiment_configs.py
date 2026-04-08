"""
Regression tests for PCam train/eval config-driven behavior.

Tests focused on:
- Config-driven model construction
- Evaluation checkpoint reconstruction
- Metrics compatibility with comparison runner
"""

import json
import tempfile
from pathlib import Path

import pytest
import torch
import yaml

from experiments.evaluate_pcam import compute_metrics, count_model_parameters

# Import functions under test
from experiments.train_pcam import create_single_modality_model


class TestSimpleHeadConfig:
    """Test hidden_dims: [] (simple head) config works end-to-end."""

    @pytest.fixture
    def simple_head_config(self):
        """Config with no hidden layer in classification head."""
        return {
            "model": {
                "feature_extractor": {
                    "model": "resnet18",
                    "pretrained": False,
                    "feature_dim": 512,
                },
                "wsi": {
                    "input_dim": 512,
                    "hidden_dim": 256,
                    "num_heads": 4,
                    "num_layers": 1,
                    "pooling": "mean",
                },
                "embed_dim": 256,
            },
            "task": {
                "classification": {
                    "hidden_dims": [],  # No hidden layer
                    "dropout": 0.3,
                }
            },
            "training": {
                "dropout": 0.3,
            },
        }

    def test_simple_head_construction(self, simple_head_config):
        """Model creation works with empty hidden_dims."""
        feature_extractor, encoder, head = create_single_modality_model(simple_head_config)

        # Verify models were created
        assert feature_extractor is not None
        assert encoder is not None
        assert head is not None

        # Verify head has no hidden layer (direct projection)
        assert head.use_hidden_layer is False

    def test_simple_head_forward_pass(self, simple_head_config):
        """Simple head can perform forward pass."""
        feature_extractor, encoder, head = create_single_modality_model(simple_head_config)

        # Create dummy input
        batch_size = 4
        images = torch.randn(batch_size, 3, 96, 96)

        # Forward pass
        features = feature_extractor(images)
        features = features.unsqueeze(1)  # Add sequence dim
        encoded = encoder(features)
        logits = head(encoded)

        # Verify output shape (binary classification = 1 logit)
        assert logits.shape == (batch_size, 1)


class TestResNetConfigVariants:
    """Test ResNet-18 vs ResNet-50 config differences are reflected in models."""

    @pytest.fixture
    def resnet18_config(self):
        return {
            "model": {
                "feature_extractor": {
                    "model": "resnet18",
                    "pretrained": False,
                    "feature_dim": 512,
                },
                "wsi": {
                    "input_dim": 512,
                    "hidden_dim": 256,
                    "num_heads": 4,
                    "num_layers": 1,
                    "pooling": "mean",
                },
                "embed_dim": 256,
            },
            "task": {
                "classification": {
                    "hidden_dims": [128],
                    "dropout": 0.3,
                }
            },
            "training": {
                "dropout": 0.3,
            },
        }

    @pytest.fixture
    def resnet50_config(self):
        return {
            "model": {
                "feature_extractor": {
                    "model": "resnet50",
                    "pretrained": False,
                    "feature_dim": 2048,
                },
                "wsi": {
                    "input_dim": 2048,  # ResNet-50 has 2048-dim features
                    "hidden_dim": 256,
                    "num_heads": 4,
                    "num_layers": 1,
                    "pooling": "mean",
                },
                "embed_dim": 256,
            },
            "task": {
                "classification": {
                    "hidden_dims": [128],
                    "dropout": 0.3,
                }
            },
            "training": {
                "dropout": 0.3,
            },
        }

    def test_resnet18_feature_dim(self, resnet18_config):
        """ResNet-18 config produces 512-dim features."""
        feature_extractor, encoder, head = create_single_modality_model(resnet18_config)

        images = torch.randn(2, 3, 96, 96)
        features = feature_extractor(images)

        assert features.shape == (2, 512)
        assert encoder.input_dim == 512

    def test_resnet50_feature_dim(self, resnet50_config):
        """ResNet-50 config produces 2048-dim features."""
        feature_extractor, encoder, head = create_single_modality_model(resnet50_config)

        images = torch.randn(2, 3, 96, 96)
        features = feature_extractor(images)

        assert features.shape == (2, 2048)
        assert encoder.input_dim == 2048

    def test_resnet50_larger_than_resnet18(self, resnet18_config, resnet50_config):
        """ResNet-50 model has more parameters than ResNet-18."""
        fe18, enc18, head18 = create_single_modality_model(resnet18_config)
        fe50, enc50, head50 = create_single_modality_model(resnet50_config)

        params18 = count_model_parameters(fe18)
        params50 = count_model_parameters(fe50)

        assert params50 > params18, "ResNet-50 should have more parameters than ResNet-18"


class TestCheckpointReconstruction:
    """Test evaluation can reconstruct correct dimensions from checkpoint config."""

    @pytest.fixture
    def simple_head_checkpoint(self, tmp_path):
        """Create a minimal checkpoint with simple head config."""
        config = {
            "model": {
                "feature_extractor": {
                    "model": "resnet18",
                    "pretrained": False,
                    "feature_dim": 512,
                },
                "wsi": {
                    "input_dim": 512,
                    "hidden_dim": 256,
                    "num_heads": 4,
                    "num_layers": 1,
                    "pooling": "mean",
                },
                "embed_dim": 256,
            },
            "task": {
                "classification": {
                    "hidden_dims": [],  # Simple head
                    "dropout": 0.3,
                }
            },
            "training": {
                "dropout": 0.3,
            },
        }

        # Create models
        from experiments.train_pcam import create_single_modality_model

        feature_extractor, encoder, head = create_single_modality_model(config)

        # Create checkpoint
        checkpoint = {
            "epoch": 1,
            "config": config,
            "metrics": {"val_auc": 0.95},
            "feature_extractor_state_dict": feature_extractor.state_dict(),
            "encoder_state_dict": encoder.state_dict(),
            "head_state_dict": head.state_dict(),
        }

        checkpoint_path = tmp_path / "simple_head_checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)

        return str(checkpoint_path), config

    @pytest.fixture
    def hidden_layer_checkpoint(self, tmp_path):
        """Create a minimal checkpoint with hidden layer head config."""
        config = {
            "model": {
                "feature_extractor": {
                    "model": "resnet18",
                    "pretrained": False,
                    "feature_dim": 512,
                },
                "wsi": {
                    "input_dim": 512,
                    "hidden_dim": 256,
                    "num_heads": 4,
                    "num_layers": 1,
                    "pooling": "mean",
                },
                "embed_dim": 256,
            },
            "task": {
                "classification": {
                    "hidden_dims": [128],  # Hidden layer
                    "dropout": 0.3,
                }
            },
            "training": {
                "dropout": 0.3,
            },
        }

        # Create models
        from experiments.train_pcam import create_single_modality_model

        feature_extractor, encoder, head = create_single_modality_model(config)

        # Create checkpoint
        checkpoint = {
            "epoch": 1,
            "config": config,
            "metrics": {"val_auc": 0.95},
            "feature_extractor_state_dict": feature_extractor.state_dict(),
            "encoder_state_dict": encoder.state_dict(),
            "head_state_dict": head.state_dict(),
        }

        checkpoint_path = tmp_path / "hidden_layer_checkpoint.pth"
        torch.save(checkpoint, checkpoint_path)

        return str(checkpoint_path), config

    def test_simple_head_reconstruction(self, simple_head_checkpoint):
        """Evaluation can reconstruct simple head from checkpoint."""
        checkpoint_path, config = simple_head_checkpoint

        # Load checkpoint like evaluate_pcam.py does
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        loaded_config = checkpoint["config"]

        # Reconstruct models
        from experiments.train_pcam import create_single_modality_model

        feature_extractor, encoder, head = create_single_modality_model(loaded_config)

        # Load state dicts
        feature_extractor.load_state_dict(checkpoint["feature_extractor_state_dict"])
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
        head.load_state_dict(checkpoint["head_state_dict"])

        # Verify head is simple (no hidden layer)
        assert head.use_hidden_layer is False

        # Verify forward pass works
        images = torch.randn(2, 3, 96, 96)
        features = feature_extractor(images)
        features = features.unsqueeze(1)
        encoded = encoder(features)
        logits = head(encoded)

        assert logits.shape == (2, 1)

    def test_hidden_layer_reconstruction(self, hidden_layer_checkpoint):
        """Evaluation can reconstruct hidden layer head from checkpoint."""
        checkpoint_path, config = hidden_layer_checkpoint

        # Load checkpoint like evaluate_pcam.py does
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        loaded_config = checkpoint["config"]

        # Reconstruct models
        from experiments.train_pcam import create_single_modality_model

        feature_extractor, encoder, head = create_single_modality_model(loaded_config)

        # Load state dicts
        feature_extractor.load_state_dict(checkpoint["feature_extractor_state_dict"])
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
        head.load_state_dict(checkpoint["head_state_dict"])

        # Verify head has hidden layer
        assert head.use_hidden_layer is True

        # Verify forward pass works
        images = torch.randn(2, 3, 96, 96)
        features = feature_extractor(images)
        features = features.unsqueeze(1)
        encoded = encoder(features)
        logits = head(encoded)

        assert logits.shape == (2, 1)


class TestComparisonRunnerMetrics:
    """Test evaluation metrics include fields comparison runner expects."""

    def test_compute_metrics_has_required_fields(self):
        """compute_metrics returns all fields needed by comparison runner."""
        # Create dummy predictions
        predictions = torch.tensor([0, 1, 0, 1, 0, 1]).numpy()
        probabilities = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.3, 0.7]).numpy()
        labels = torch.tensor([0, 1, 0, 1, 1, 0]).numpy()

        metrics = compute_metrics(predictions, probabilities, labels)

        # Check all comparison runner fields are present
        required_fields = [
            "accuracy",
            "auc",
            "f1",
            "precision",
            "recall",
        ]

        for field in required_fields:
            assert field in metrics, f"Missing required field: {field}"
            assert isinstance(metrics[field], float), f"Field {field} should be a float"

    def test_metrics_include_confusion_matrix(self):
        """compute_metrics includes confusion matrix."""
        predictions = torch.tensor([0, 1, 0, 1]).numpy()
        probabilities = torch.tensor([0.1, 0.9, 0.2, 0.8]).numpy()
        labels = torch.tensor([0, 1, 0, 1]).numpy()

        metrics = compute_metrics(predictions, probabilities, labels)

        assert "confusion_matrix" in metrics
        # Confusion matrix should be 2x2: [[TN, FP], [FN, TP]]
        cm = metrics["confusion_matrix"]
        assert len(cm) == 2
        assert len(cm[0]) == 2

    def test_metrics_include_per_class_metrics(self):
        """compute_metrics includes per-class precision/recall/f1."""
        predictions = torch.tensor([0, 1, 0, 1, 0, 1]).numpy()
        probabilities = torch.tensor([0.1, 0.9, 0.2, 0.8, 0.3, 0.7]).numpy()
        labels = torch.tensor([0, 1, 0, 1, 1, 0]).numpy()

        metrics = compute_metrics(predictions, probabilities, labels)

        assert "per_class_metrics" in metrics
        assert "class_0" in metrics["per_class_metrics"]
        assert "class_1" in metrics["per_class_metrics"]

        for cls in ["class_0", "class_1"]:
            cls_metrics = metrics["per_class_metrics"][cls]
            assert "precision" in cls_metrics
            assert "recall" in cls_metrics
            assert "f1" in cls_metrics


class TestConfigYamlLoading:
    """Test actual YAML configs can be loaded and used."""

    @pytest.fixture
    def configs_dir(self):
        """Get path to configs directory."""
        return Path(__file__).parent.parent / "experiments" / "configs"

    def test_simple_head_yaml_loads(self, configs_dir):
        """simple_head.yaml can be loaded and used for model creation."""
        config_path = configs_dir / "pcam_comparison" / "simple_head.yaml"
        if not config_path.exists():
            pytest.skip(f"Config not found: {config_path}")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Verify hidden_dims is empty
        assert config["task"]["classification"]["hidden_dims"] == []

        # Model creation should work
        feature_extractor, encoder, head = create_single_modality_model(config)
        assert head.use_hidden_layer is False

    def test_resnet50_yaml_loads(self, configs_dir):
        """resnet50.yaml can be loaded and produces 2048-dim features."""
        config_path = configs_dir / "pcam_comparison" / "resnet50.yaml"
        if not config_path.exists():
            pytest.skip(f"Config not found: {config_path}")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Verify ResNet-50 is specified
        assert config["model"]["feature_extractor"]["model"] == "resnet50"
        assert config["model"]["feature_extractor"]["feature_dim"] == 2048

        # Model creation should work
        feature_extractor, encoder, head = create_single_modality_model(config)

        # Verify dimensions
        images = torch.randn(2, 3, 96, 96)
        features = feature_extractor(images)
        assert features.shape == (2, 2048)

    def test_baseline_resnet18_yaml_loads(self, configs_dir):
        """baseline_resnet18.yaml can be loaded and produces 512-dim features."""
        config_path = configs_dir / "pcam_comparison" / "baseline_resnet18.yaml"
        if not config_path.exists():
            pytest.skip(f"Config not found: {config_path}")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        # Verify ResNet-18 is specified
        assert config["model"]["feature_extractor"]["model"] == "resnet18"

        # Model creation should work
        feature_extractor, encoder, head = create_single_modality_model(config)

        # Verify dimensions
        images = torch.randn(2, 3, 96, 96)
        features = feature_extractor(images)
        assert features.shape == (2, 512)
