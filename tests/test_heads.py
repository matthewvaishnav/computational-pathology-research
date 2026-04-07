"""
Unit tests for task-specific prediction heads.
"""

import pytest
import torch

from src.models.heads import ClassificationHead, MultiTaskHead, SurvivalPredictionHead


class TestClassificationHead:
    """Tests for ClassificationHead module."""

    def test_default_forward_shape(self):
        """Test default classification head output shape."""
        head = ClassificationHead(input_dim=256, num_classes=5)
        embeddings = torch.randn(16, 256)
        logits = head(embeddings)
        assert logits.shape == (16, 5)

    def test_binary_classification_shape(self):
        """Test binary classification output shape."""
        head = ClassificationHead(input_dim=128, num_classes=2)
        embeddings = torch.randn(8, 128)
        logits = head(embeddings)
        assert logits.shape == (8, 2)

    def test_without_hidden_layer(self):
        """Test classification head without hidden layer."""
        head = ClassificationHead(input_dim=64, num_classes=3, use_hidden_layer=False)
        embeddings = torch.randn(4, 64)
        logits = head(embeddings)
        assert logits.shape == (4, 3)
        # Verify it's just dropout + linear (2 layers in Sequential)
        assert len(head.classifier) == 2

    def test_with_hidden_layer(self):
        """Test classification head with hidden layer."""
        head = ClassificationHead(input_dim=64, num_classes=3, use_hidden_layer=True)
        # Verify structure: Linear, LayerNorm, GELU, Dropout, Linear (5 layers)
        assert len(head.classifier) == 5

    def test_different_input_dimensions(self):
        """Test with various input dimensions."""
        for input_dim in [64, 128, 256, 512]:
            head = ClassificationHead(input_dim=input_dim, num_classes=10)
            embeddings = torch.randn(2, input_dim)
            logits = head(embeddings)
            assert logits.shape == (2, 10)

    def test_dropout_effect(self):
        """Test that dropout is applied during training."""
        head = ClassificationHead(input_dim=64, num_classes=2, dropout=0.5)
        head.train()
        embeddings = torch.randn(4, 64)

        # Run multiple forward passes - outputs should differ due to dropout
        outputs = [head(embeddings) for _ in range(10)]
        # At least some outputs should differ (highly probable with dropout=0.5)
        all_same = all(torch.allclose(outputs[0], o) for o in outputs[1:])
        assert not all_same, "Dropout should cause variation in training mode"

    def test_eval_mode_consistency(self):
        """Test that eval mode produces consistent outputs."""
        head = ClassificationHead(input_dim=64, num_classes=2, dropout=0.5)
        head.eval()
        embeddings = torch.randn(4, 64)

        with torch.no_grad():
            output1 = head(embeddings)
            output2 = head(embeddings)

        assert torch.allclose(output1, output2)

    def test_batch_size_one(self):
        """Test with batch size of 1."""
        head = ClassificationHead(input_dim=128, num_classes=4)
        embeddings = torch.randn(1, 128)
        logits = head(embeddings)
        assert logits.shape == (1, 4)


class TestSurvivalPredictionHead:
    """Tests for SurvivalPredictionHead module."""

    def test_risk_score_output_shape(self):
        """Test risk score output shape (no time bins)."""
        head = SurvivalPredictionHead(input_dim=256, num_time_bins=None)
        embeddings = torch.randn(16, 256)
        risk_scores = head(embeddings)
        assert risk_scores.shape == (16, 1)

    def test_discrete_hazard_output_shape(self):
        """Test discrete hazard output shape."""
        head = SurvivalPredictionHead(input_dim=256, num_time_bins=10)
        embeddings = torch.randn(8, 256)
        hazards = head(embeddings)
        assert hazards.shape == (8, 10)

    def test_hazard_probability_range(self):
        """Test that hazard probabilities are in [0, 1] range."""
        head = SurvivalPredictionHead(input_dim=128, num_time_bins=5)
        embeddings = torch.randn(4, 128)
        hazards = head(embeddings, return_hazards=True)
        assert torch.all(hazards >= 0) and torch.all(hazards <= 1)

    def test_logits_vs_hazards(self):
        """Test that logits can be any value but hazards are sigmoid-activated."""
        head = SurvivalPredictionHead(input_dim=64, num_time_bins=3)
        embeddings = torch.randn(2, 64)

        logits = head(embeddings, return_hazards=False)
        hazards = head(embeddings, return_hazards=True)

        # Logits should not be bounded to [0, 1]
        assert not torch.all((logits >= 0) & (logits <= 1))
        # Hazards should be in [0, 1]
        assert torch.all((hazards >= 0) & (hazards <= 1))

    def test_compute_survival_curve(self):
        """Test survival curve computation from discrete hazards."""
        head = SurvivalPredictionHead(input_dim=128, num_time_bins=4)
        embeddings = torch.randn(2, 128)
        survival_probs = head.compute_survival_curve(embeddings)

        assert survival_probs.shape == (2, 4)
        # Survival probabilities should be in [0, 1]
        assert torch.all(survival_probs >= 0) and torch.all(survival_probs <= 1)
        # Survival should be non-increasing over time
        for i in range(survival_probs.shape[0]):
            diffs = survival_probs[i, 1:] - survival_probs[i, :-1]
            assert torch.all(diffs <= 1e-6), "Survival curve should be non-increasing"

    def test_compute_survival_curve_without_time_bins(self):
        """Test that survival curve raises error without time bins."""
        head = SurvivalPredictionHead(input_dim=128, num_time_bins=None)
        embeddings = torch.randn(2, 128)

        with pytest.raises(ValueError, match="discrete hazard prediction mode"):
            head.compute_survival_curve(embeddings)

    def test_without_hidden_layer(self):
        """Test survival head without hidden layer."""
        head = SurvivalPredictionHead(
            input_dim=64, num_time_bins=5, use_hidden_layer=False
        )
        assert len(head.predictor) == 2  # Dropout + Linear

    def test_prediction_mode_attribute(self):
        """Test prediction_mode is set correctly."""
        head_risk = SurvivalPredictionHead(input_dim=64)
        assert head_risk.prediction_mode == "risk_score"

        head_hazard = SurvivalPredictionHead(input_dim=64, num_time_bins=10)
        assert head_hazard.prediction_mode == "discrete_hazard"


class TestMultiTaskHead:
    """Tests for MultiTaskHead module."""

    def test_default_forward(self):
        """Test default multi-task head forward pass."""
        head = MultiTaskHead(
            input_dim=256,
            classification_config={"num_classes": 5},
            survival_config={"num_time_bins": 10},
        )
        embeddings = torch.randn(16, 256)
        class_logits, survival_output = head(embeddings)

        assert class_logits.shape == (16, 5)
        assert survival_output.shape == (16, 10)

    def test_risk_score_survival(self):
        """Test multi-task with risk score survival (no time bins)."""
        head = MultiTaskHead(
            input_dim=128,
            classification_config={"num_classes": 2},
            survival_config={"num_time_bins": None},
        )
        embeddings = torch.randn(8, 128)
        class_logits, survival_output = head(embeddings)

        assert class_logits.shape == (8, 2)
        assert survival_output.shape == (8, 1)

    def test_classification_only(self):
        """Test multi-task head with only classification config."""
        head = MultiTaskHead(
            input_dim=128,
            classification_config={"num_classes": 3},
            survival_config=None,
        )
        embeddings = torch.randn(4, 128)
        class_logits, survival_output = head(embeddings)

        assert class_logits.shape == (4, 3)
        assert survival_output.shape == (4, 1)  # Default risk score

    def test_survival_only(self):
        """Test multi-task head with only survival config."""
        head = MultiTaskHead(
            input_dim=128,
            classification_config=None,
            survival_config={"num_time_bins": 5},
        )
        embeddings = torch.randn(4, 128)
        class_logits, survival_output = head(embeddings)

        assert class_logits.shape == (4, 2)  # Default num_classes=2
        assert survival_output.shape == (4, 5)

    def test_with_shared_hidden_layer(self):
        """Test multi-task head with shared hidden layer."""
        head = MultiTaskHead(
            input_dim=256,
            classification_config={"num_classes": 4},
            survival_config={"num_time_bins": 3},
            shared_hidden_dim=128,
        )
        # Verify shared layer exists
        assert head.shared_layer is not None

        embeddings = torch.randn(8, 256)
        class_logits, survival_output = head(embeddings)

        assert class_logits.shape == (8, 4)
        assert survival_output.shape == (8, 3)

    def test_without_shared_hidden_layer(self):
        """Test multi-task head without shared hidden layer."""
        head = MultiTaskHead(
            input_dim=128,
            classification_config={"num_classes": 3},
            survival_config={"num_time_bins": 4},
            shared_hidden_dim=None,
        )
        # Verify no shared layer
        assert head.shared_layer is None

        embeddings = torch.randn(4, 128)
        class_logits, survival_output = head(embeddings)

        assert class_logits.shape == (4, 3)
        assert survival_output.shape == (4, 4)

    def test_return_survival_hazards(self):
        """Test return_survival_hazards flag."""
        head = MultiTaskHead(
            input_dim=64,
            classification_config={"num_classes": 2},
            survival_config={"num_time_bins": 5},
        )
        embeddings = torch.randn(4, 64)

        # Without return_hazards (logits)
        _, survival_logits = head(embeddings, return_survival_hazards=False)
        assert not torch.all((survival_logits >= 0) & (survival_logits <= 1))

        # With return_hazards (probabilities)
        _, survival_hazards = head(embeddings, return_survival_hazards=True)
        assert torch.all((survival_hazards >= 0) & (survival_hazards <= 1))

    def test_default_configs(self):
        """Test that default configs work when None passed."""
        head = MultiTaskHead(input_dim=128)
        embeddings = torch.randn(2, 128)
        class_logits, survival_output = head(embeddings)

        assert class_logits.shape == (2, 2)  # Default num_classes
        assert survival_output.shape == (2, 1)  # Default risk score

    def test_input_dim_injection_does_not_mutate_caller_configs(self):
        """Caller-provided config dicts should not be mutated in-place."""
        classification_config = {"num_classes": 3}
        survival_config = {"num_time_bins": 5}

        head = MultiTaskHead(
            input_dim=128,
            classification_config=classification_config,
            survival_config=survival_config,
            shared_hidden_dim=64,
        )

        assert classification_config == {"num_classes": 3}
        assert survival_config == {"num_time_bins": 5}
        assert head.classification_head.input_dim == 64
        assert head.survival_head.input_dim == 64
