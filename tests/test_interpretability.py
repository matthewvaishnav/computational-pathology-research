"""
Tests for src/utils/interpretability.py

Tests cover:
- AttentionVisualizer
- SaliencyMap
- EmbeddingAnalyzer
"""

import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.utils.interpretability import AttentionVisualizer, EmbeddingAnalyzer, SaliencyMap


# ============================================================================
# AttentionVisualizer Tests
# ============================================================================


class TestAttentionVisualizer:
    """Tests for AttentionVisualizer."""

    def test_initialization(self):
        """Test visualizer initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            viz = AttentionVisualizer(output_dir=tmpdir)

            assert viz.output_dir == Path(tmpdir)
            assert viz.output_dir.exists()

    def test_plot_modality_attention(self):
        """Test plotting cross-modal attention."""
        with tempfile.TemporaryDirectory() as tmpdir:
            viz = AttentionVisualizer(output_dir=tmpdir)

            # Create sample embeddings
            embeddings = {
                "wsi": torch.randn(4, 128),
                "genomic": torch.randn(4, 128),
                "clinical": torch.randn(4, 128),
            }

            filepath = viz.plot_modality_attention(embeddings)

            assert Path(filepath).exists()
            assert "modality_attention.png" in filepath

    def test_plot_modality_attention_with_masks(self):
        """Test plotting with modality masks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            viz = AttentionVisualizer(output_dir=tmpdir)

            embeddings = {
                "wsi": torch.randn(4, 128),
                "genomic": torch.randn(4, 128),
            }

            masks = {
                "wsi": torch.ones(4, dtype=torch.bool),
                "genomic": torch.tensor([True, True, False, False]),
            }

            filepath = viz.plot_modality_attention(embeddings, modality_masks=masks)

            assert Path(filepath).exists()

    def test_plot_modality_attention_with_none_embeddings(self):
        """Test plotting with some None embeddings."""
        with tempfile.TemporaryDirectory() as tmpdir:
            viz = AttentionVisualizer(output_dir=tmpdir)

            embeddings = {
                "wsi": torch.randn(4, 128),
                "genomic": None,
                "clinical": torch.randn(4, 128),
            }

            filepath = viz.plot_modality_attention(embeddings)

            assert Path(filepath).exists()

    def test_plot_temporal_attention(self):
        """Test plotting temporal attention."""
        with tempfile.TemporaryDirectory() as tmpdir:
            viz = AttentionVisualizer(output_dir=tmpdir)

            # Create sample slide embeddings
            slide_embeddings = torch.randn(2, 10, 128)

            filepath = viz.plot_temporal_attention(slide_embeddings)

            assert Path(filepath).exists()
            assert "temporal_attention.png" in filepath

    def test_plot_temporal_attention_2d_input(self):
        """Test temporal attention with 2D input (adds sequence dim)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            viz = AttentionVisualizer(output_dir=tmpdir)

            # 2D input: [batch_size, embed_dim]
            slide_embeddings = torch.randn(4, 128)

            filepath = viz.plot_temporal_attention(slide_embeddings)

            assert Path(filepath).exists()

    def test_plot_temporal_attention_with_timestamps(self):
        """Test temporal attention with timestamps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            viz = AttentionVisualizer(output_dir=tmpdir)

            slide_embeddings = torch.randn(2, 10, 128)
            timestamps = torch.arange(10).unsqueeze(0).expand(2, -1).float()

            filepath = viz.plot_temporal_attention(slide_embeddings, timestamps=timestamps)

            assert Path(filepath).exists()


# ============================================================================
# SaliencyMap Tests
# ============================================================================


class DummyModel(nn.Module):
    """Dummy model for testing saliency."""

    def __init__(self, output_dim=2):
        super().__init__()
        self.wsi_proj = nn.Linear(128, 64)
        self.genomic_proj = nn.Linear(100, 64)
        self.clinical_proj = nn.Embedding(1000, 64)
        self.classifier = nn.Linear(64, output_dim)

    def forward(self, batch):
        features = []

        if "wsi_features" in batch and batch["wsi_features"] is not None:
            wsi = self.wsi_proj(batch["wsi_features"])
            if wsi.dim() == 3:
                wsi = wsi.mean(dim=1)
            features.append(wsi)

        if "genomic" in batch and batch["genomic"] is not None:
            genomic = self.genomic_proj(batch["genomic"])
            features.append(genomic)

        if "clinical_text" in batch and batch["clinical_text"] is not None:
            clinical = self.clinical_proj(batch["clinical_text"])
            if clinical.dim() == 3:
                clinical = clinical.mean(dim=1)
            features.append(clinical)

        if not features:
            raise ValueError("No valid features")

        combined = torch.stack(features).mean(dim=0)
        return self.classifier(combined)


class TestSaliencyMap:
    """Tests for SaliencyMap."""

    def test_initialization(self):
        """Test saliency map initialization."""
        saliency = SaliencyMap()

        assert saliency.device is not None

    def test_initialization_with_device(self):
        """Test initialization with specific device."""
        device = torch.device("cpu")
        saliency = SaliencyMap(device=device)

        assert saliency.device == device

    def test_batch_key_for_modality(self):
        """Test modality to batch key mapping."""
        assert SaliencyMap._batch_key_for_modality("wsi") == "wsi_features"
        assert SaliencyMap._batch_key_for_modality("genomic") == "genomic"
        assert SaliencyMap._batch_key_for_modality("clinical") == "clinical_text"

    def test_select_target_logits_with_target_idx(self):
        """Test target logit selection with target index."""
        device = torch.device("cpu")

        # Multi-class case
        output = torch.randn(4, 3)
        result = SaliencyMap._select_target_logits(output, None, target_idx=1, device=device)

        assert result.shape == (4, 1)

    def test_select_target_logits_with_labels(self):
        """Test target logit selection with labels."""
        device = torch.device("cpu")

        output = torch.randn(4, 3)
        labels = torch.tensor([0, 1, 2, 1])

        result = SaliencyMap._select_target_logits(output, labels, None, device=device)

        assert result.shape == (4, 1)

    def test_select_target_logits_binary_output(self):
        """Test target logit selection with binary single-logit output."""
        device = torch.device("cpu")

        # Binary output [B, 1]
        output = torch.randn(4, 1)

        # Target class 1 (positive)
        result = SaliencyMap._select_target_logits(output, None, target_idx=1, device=device)
        assert result.shape == (4, 1)

        # Target class 0 (negative)
        result = SaliencyMap._select_target_logits(output, None, target_idx=0, device=device)
        assert result.shape == (4, 1)

    def test_compute_gradient_saliency(self):
        """Test gradient-based saliency computation."""
        model = DummyModel(output_dim=2)
        saliency = SaliencyMap(device=torch.device("cpu"))

        batch = {
            "wsi_features": torch.randn(2, 10, 128),
            "genomic": torch.randn(2, 100),
            "wsi_mask": torch.ones(2, 10, dtype=torch.bool),
        }

        saliency_maps = saliency.compute_gradient_saliency(model, batch, target_idx=1)

        assert "wsi" in saliency_maps
        assert "genomic" in saliency_maps
        assert isinstance(saliency_maps["wsi"], np.ndarray)
        assert isinstance(saliency_maps["genomic"], np.ndarray)

    def test_compute_gradient_saliency_with_labels(self):
        """Test gradient saliency with labels."""
        model = DummyModel(output_dim=2)
        saliency = SaliencyMap(device=torch.device("cpu"))

        batch = {
            "wsi_features": torch.randn(2, 10, 128),
            "genomic": torch.randn(2, 100),
            "label": torch.tensor([0, 1]),
        }

        saliency_maps = saliency.compute_gradient_saliency(model, batch)

        assert "wsi" in saliency_maps
        assert "genomic" in saliency_maps
        # Label should be restored to batch
        assert "label" in batch

    def test_compute_gradient_saliency_skips_non_float(self):
        """Test that saliency skips non-floating point tensors."""
        model = DummyModel(output_dim=2)
        saliency = SaliencyMap(device=torch.device("cpu"))

        batch = {
            "wsi_features": torch.randn(2, 10, 128),
            "clinical_text": torch.randint(0, 1000, (2, 20)),  # Integer tensor
        }

        saliency_maps = saliency.compute_gradient_saliency(model, batch)

        assert "wsi" in saliency_maps
        assert "clinical" not in saliency_maps  # Skipped because non-float

    def test_compute_integrated_gradients(self):
        """Test integrated gradients computation."""
        model = DummyModel(output_dim=2)
        saliency = SaliencyMap(device=torch.device("cpu"))

        batch = {
            "wsi_features": torch.randn(2, 10, 128),
            "genomic": torch.randn(2, 100),
        }

        integrated_grads = saliency.compute_integrated_gradients(
            model, batch, num_steps=10, target_idx=1
        )

        assert "wsi" in integrated_grads
        assert "genomic" in integrated_grads
        assert isinstance(integrated_grads["wsi"], np.ndarray)

    def test_compute_integrated_gradients_with_baseline(self):
        """Test integrated gradients with custom baseline."""
        model = DummyModel(output_dim=2)
        saliency = SaliencyMap(device=torch.device("cpu"))

        batch = {
            "wsi_features": torch.randn(2, 10, 128),
            "genomic": torch.randn(2, 100),
        }

        baseline = {
            "wsi": torch.zeros(2, 10, 128),
            "genomic": torch.zeros(2, 100),
        }

        integrated_grads = saliency.compute_integrated_gradients(
            model, batch, baseline=baseline, num_steps=10
        )

        assert "wsi" in integrated_grads
        assert "genomic" in integrated_grads

    def test_compute_integrated_gradients_with_labels(self):
        """Test integrated gradients with labels."""
        model = DummyModel(output_dim=2)
        saliency = SaliencyMap(device=torch.device("cpu"))

        batch = {
            "wsi_features": torch.randn(2, 10, 128),
            "genomic": torch.randn(2, 100),
            "label": torch.tensor([0, 1]),
        }

        integrated_grads = saliency.compute_integrated_gradients(model, batch, num_steps=5)

        assert "wsi" in integrated_grads
        # Label should be restored
        assert "label" in batch


# ============================================================================
# EmbeddingAnalyzer Tests
# ============================================================================


class TestEmbeddingAnalyzer:
    """Tests for EmbeddingAnalyzer."""

    def test_initialization(self):
        """Test analyzer initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = EmbeddingAnalyzer(output_dir=tmpdir)

            assert analyzer.output_dir == Path(tmpdir)
            assert analyzer.output_dir.exists()

    def test_plot_tsne(self):
        """Test t-SNE visualization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = EmbeddingAnalyzer(output_dir=tmpdir)

            embeddings = np.random.randn(50, 128)
            labels = np.random.randint(0, 3, size=50)

            filepath = analyzer.plot_tsne(embeddings, labels)

            assert Path(filepath).exists()
            assert "tsne_visualization.png" in filepath

    def test_plot_tsne_custom_params(self):
        """Test t-SNE with custom parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = EmbeddingAnalyzer(output_dir=tmpdir)

            embeddings = np.random.randn(50, 128)
            labels = np.random.randint(0, 3, size=50)

            filepath = analyzer.plot_tsne(
                embeddings,
                labels,
                title="Custom t-SNE",
                filename="custom_tsne.png",
                perplexity=20,
                colormap="plasma",
            )

            assert Path(filepath).exists()
            assert "custom_tsne.png" in filepath

    def test_plot_tsne_with_nan(self):
        """Test t-SNE with NaN values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = EmbeddingAnalyzer(output_dir=tmpdir)

            embeddings = np.random.randn(50, 128)
            embeddings[0, 0] = np.nan
            labels = np.random.randint(0, 3, size=50)

            filepath = analyzer.plot_tsne(embeddings, labels)

            assert Path(filepath).exists()

    def test_plot_tsne_too_few_samples(self):
        """Test t-SNE with too few samples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = EmbeddingAnalyzer(output_dir=tmpdir)

            embeddings = np.random.randn(1, 128)
            labels = np.array([0])

            with pytest.raises(ValueError, match="at least 2 samples"):
                analyzer.plot_tsne(embeddings, labels)

    def test_plot_pca(self):
        """Test PCA visualization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = EmbeddingAnalyzer(output_dir=tmpdir)

            embeddings = np.random.randn(50, 128)
            labels = np.random.randint(0, 3, size=50)

            filepath, explained_variance = analyzer.plot_pca(embeddings, labels)

            assert Path(filepath).exists()
            assert "pca_visualization.png" in filepath
            assert "PC1" in explained_variance
            assert "PC2" in explained_variance
            assert "total" in explained_variance
            assert 0 <= explained_variance["total"] <= 1

    def test_plot_pca_custom_params(self):
        """Test PCA with custom parameters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = EmbeddingAnalyzer(output_dir=tmpdir)

            embeddings = np.random.randn(50, 128)
            labels = np.random.randint(0, 3, size=50)

            filepath, _ = analyzer.plot_pca(
                embeddings, labels, title="Custom PCA", filename="custom_pca.png", colormap="plasma"
            )

            assert Path(filepath).exists()
            assert "custom_pca.png" in filepath

    def test_plot_pca_with_nan(self):
        """Test PCA with NaN values."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = EmbeddingAnalyzer(output_dir=tmpdir)

            embeddings = np.random.randn(50, 128)
            embeddings[0, 0] = np.nan
            labels = np.random.randint(0, 3, size=50)

            filepath, _ = analyzer.plot_pca(embeddings, labels)

            assert Path(filepath).exists()

    def test_compute_modality_correlation(self):
        """Test modality correlation computation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = EmbeddingAnalyzer(output_dir=tmpdir)

            embeddings = {
                "wsi": torch.randn(50, 128),
                "genomic": torch.randn(50, 100),
                "clinical": torch.randn(50, 64),
            }

            filepath, correlation_matrix = analyzer.compute_modality_correlation(embeddings)

            assert Path(filepath).exists()
            assert "modality_correlation.png" in filepath
            assert correlation_matrix.shape == (3, 3)
            # Diagonal should be 1.0
            assert np.allclose(np.diag(correlation_matrix), 1.0)

    def test_compute_modality_correlation_numpy(self):
        """Test modality correlation with numpy arrays."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = EmbeddingAnalyzer(output_dir=tmpdir)

            embeddings = {
                "wsi": np.random.randn(50, 128),
                "genomic": np.random.randn(50, 100),
            }

            filepath, correlation_matrix = analyzer.compute_modality_correlation(embeddings)

            assert Path(filepath).exists()
            assert correlation_matrix.shape == (2, 2)

    def test_compute_modality_correlation_edge_cases(self):
        """Test modality correlation with edge cases."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = EmbeddingAnalyzer(output_dir=tmpdir)

            # Single sample (correlation undefined)
            embeddings = {
                "wsi": torch.randn(1, 128),
                "genomic": torch.randn(1, 100),
            }

            filepath, correlation_matrix = analyzer.compute_modality_correlation(embeddings)

            assert Path(filepath).exists()
            # Should handle gracefully (zeros for undefined correlations)
            assert correlation_matrix.shape == (2, 2)

    def test_compute_modality_correlation_zero_std(self):
        """Test modality correlation with zero standard deviation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            analyzer = EmbeddingAnalyzer(output_dir=tmpdir)

            # Constant embeddings (zero std)
            embeddings = {
                "wsi": torch.ones(50, 128),
                "genomic": torch.randn(50, 100),
            }

            filepath, correlation_matrix = analyzer.compute_modality_correlation(embeddings)

            assert Path(filepath).exists()
            # Should handle gracefully
            assert correlation_matrix.shape == (2, 2)


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests for interpretability tools."""

    def test_full_interpretability_pipeline(self):
        """Test full interpretability pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Setup
            model = DummyModel(output_dim=2)
            viz = AttentionVisualizer(output_dir=tmpdir)
            saliency = SaliencyMap(device=torch.device("cpu"))
            analyzer = EmbeddingAnalyzer(output_dir=tmpdir)

            # Create sample data
            batch = {
                "wsi_features": torch.randn(4, 10, 128),
                "genomic": torch.randn(4, 100),
            }

            embeddings = {
                "wsi": torch.randn(4, 128),
                "genomic": torch.randn(4, 128),
            }

            # Test attention visualization
            attn_path = viz.plot_modality_attention(embeddings)
            assert Path(attn_path).exists()

            # Test saliency computation
            saliency_maps = saliency.compute_gradient_saliency(model, batch)
            assert len(saliency_maps) > 0

            # Test embedding analysis
            emb_array = np.random.randn(50, 128)
            labels = np.random.randint(0, 2, size=50)

            tsne_path = analyzer.plot_tsne(emb_array, labels)
            assert Path(tsne_path).exists()

            pca_path, _ = analyzer.plot_pca(emb_array, labels)
            assert Path(pca_path).exists()

            corr_path, _ = analyzer.compute_modality_correlation(embeddings)
            assert Path(corr_path).exists()
