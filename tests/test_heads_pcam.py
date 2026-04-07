"""
Tests for ClassificationHead behavior specific to PCam binary classification.

PCam uses binary classification with a single output logit and BCEWithLogitsLoss,
unlike the multi-class classification used in other parts of the codebase.
"""

import pytest
import torch
import torch.nn as nn

from src.models.heads import ClassificationHead


class TestClassificationHeadPCamBinary:
    """Tests for ClassificationHead in PCam binary classification setup."""

    def test_binary_single_logit_output_shape(self):
        """Test PCam binary classification outputs single logit."""
        head = ClassificationHead(input_dim=256, num_classes=1)  # Binary: single logit
        embeddings = torch.randn(8, 256)
        logits = head(embeddings)
        assert logits.shape == (8, 1)

    def test_binary_single_logit_with_bce_loss(self):
        """Test single logit works with BCEWithLogitsLoss (PCam setup)."""
        head = ClassificationHead(input_dim=128, num_classes=1)
        criterion = nn.BCEWithLogitsLoss()

        embeddings = torch.randn(4, 128)
        logits = head(embeddings)
        targets = torch.randn(4, 1).sigmoid()  # Binary targets in [0, 1]

        loss = criterion(logits, targets)
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_binary_batch_size_one(self):
        """Test PCam binary classification with batch size 1."""
        head = ClassificationHead(input_dim=64, num_classes=1)
        embeddings = torch.randn(1, 64)
        logits = head(embeddings)
        assert logits.shape == (1, 1)

    def test_binary_probability_range(self):
        """Test that sigmoid of logit gives valid probabilities."""
        head = ClassificationHead(input_dim=128, num_classes=1)
        embeddings = torch.randn(10, 128)
        logits = head(embeddings)
        probs = torch.sigmoid(logits)

        assert torch.all(probs >= 0) and torch.all(probs <= 1)

    def test_binary_prediction_consistency(self):
        """Test binary predictions are consistent across eval calls."""
        head = ClassificationHead(input_dim=128, num_classes=1, dropout=0.0)
        head.eval()
        embeddings = torch.randn(5, 128)

        with torch.no_grad():
            logits1 = head(embeddings)
            logits2 = head(embeddings)

        assert torch.allclose(logits1, logits2, atol=1e-6)

    def test_binary_with_hidden_layer(self):
        """Test binary head with hidden layer (PCam default)."""
        head = ClassificationHead(
            input_dim=256,
            num_classes=1,
            hidden_dim=128,
            use_hidden_layer=True,
        )
        embeddings = torch.randn(4, 256)
        logits = head(embeddings)
        assert logits.shape == (4, 1)

    def test_binary_without_hidden_layer(self):
        """Test binary head without hidden layer."""
        head = ClassificationHead(
            input_dim=128,
            num_classes=1,
            use_hidden_layer=False,
        )
        embeddings = torch.randn(4, 128)
        logits = head(embeddings)
        assert logits.shape == (4, 1)

    def test_binary_gradient_flow(self):
        """Test gradients flow properly in binary classification setup."""
        head = ClassificationHead(input_dim=64, num_classes=1)
        criterion = nn.BCEWithLogitsLoss()

        embeddings = torch.randn(4, 64, requires_grad=True)
        logits = head(embeddings)
        targets = torch.randint(0, 2, (4, 1)).float()

        loss = criterion(logits, targets)
        loss.backward()

        assert embeddings.grad is not None
        assert not torch.all(embeddings.grad == 0)

    def test_binary_different_embed_dims(self):
        """Test binary head with various embedding dimensions."""
        for embed_dim in [64, 128, 256, 512]:
            head = ClassificationHead(input_dim=embed_dim, num_classes=1)
            embeddings = torch.randn(2, embed_dim)
            logits = head(embeddings)
            assert logits.shape == (2, 1)

    def test_binary_different_hidden_dims(self):
        """Test binary head with various hidden dimensions."""
        for hidden_dim in [64, 128, 256]:
            head = ClassificationHead(
                input_dim=256,
                num_classes=1,
                hidden_dim=hidden_dim,
                use_hidden_layer=True,
            )
            embeddings = torch.randn(2, 256)
            logits = head(embeddings)
            assert logits.shape == (2, 1)
