"""Regression tests for interpretability utilities."""

import numpy as np
import torch
import torch.nn as nn

from src.utils.interpretability import EmbeddingAnalyzer, SaliencyMap


class _ToyMultimodalClassifier(nn.Module):
    """Minimal multimodal classifier for saliency regression tests."""

    def forward(self, batch):
        wsi_term = batch["wsi_features"].mean(dim=(1, 2))
        genomic_term = batch["genomic"].mean(dim=1)
        logits = torch.stack((wsi_term + genomic_term, wsi_term - genomic_term), dim=1)
        return logits


class _ToyBinaryClassifier(nn.Module):
    """Minimal single-logit classifier for BCE-style attribution tests."""

    def forward(self, batch):
        wsi_term = batch["wsi_features"].mean(dim=(1, 2))
        genomic_term = batch["genomic"].mean(dim=1)
        return (wsi_term + genomic_term).unsqueeze(-1)


def _make_batch(batch_size=2):
    return {
        "wsi_features": torch.randn(batch_size, 6, 4),
        "genomic": torch.randn(batch_size, 8),
        "clinical_text": torch.randint(0, 100, (batch_size, 5), dtype=torch.long),
        "label": torch.tensor([0, 1], dtype=torch.long)[:batch_size],
    }


def test_gradient_saliency_uses_wsi_features_key_and_skips_integer_clinical_inputs():
    model = _ToyMultimodalClassifier()
    saliency = SaliencyMap(device=torch.device("cpu"))

    maps = saliency.compute_gradient_saliency(model, _make_batch())

    assert set(maps) == {"wsi", "genomic"}
    assert maps["wsi"].shape == (2, 6)
    assert maps["genomic"].shape == (2,)
    assert np.all(maps["wsi"] >= 0)
    assert np.all(maps["genomic"] >= 0)


def test_integrated_gradients_uses_wsi_features_key_and_skips_integer_clinical_inputs():
    model = _ToyMultimodalClassifier()
    saliency = SaliencyMap(device=torch.device("cpu"))

    attributions = saliency.compute_integrated_gradients(model, _make_batch(), num_steps=8)

    assert set(attributions) == {"wsi", "genomic"}
    assert attributions["wsi"].shape == (2, 6, 4)
    assert attributions["genomic"].shape == (2, 8)
    assert np.isfinite(attributions["wsi"]).all()
    assert np.isfinite(attributions["genomic"]).all()


def test_integrated_gradients_supports_single_logit_binary_outputs_with_labels():
    model = _ToyBinaryClassifier()
    saliency = SaliencyMap(device=torch.device("cpu"))

    attributions = saliency.compute_integrated_gradients(model, _make_batch(), num_steps=8)

    assert set(attributions) == {"wsi", "genomic"}
    assert attributions["wsi"].shape == (2, 6, 4)
    assert attributions["genomic"].shape == (2, 8)
    assert np.isfinite(attributions["wsi"]).all()
    assert np.isfinite(attributions["genomic"]).all()


def test_plot_tsne_handles_two_samples_with_auto_perplexity(tmp_path):
    analyzer = EmbeddingAnalyzer(output_dir=str(tmp_path))

    embeddings = np.array([[0.0, 1.0, 2.0], [2.0, 1.0, 0.0]], dtype=np.float32)
    labels = np.array([0, 1])

    output_path = analyzer.plot_tsne(embeddings, labels)

    assert output_path.endswith(".png")
    assert (tmp_path / "tsne_visualization.png").exists()


def test_modality_correlation_avoids_nan_with_constant_embeddings(tmp_path):
    analyzer = EmbeddingAnalyzer(output_dir=str(tmp_path))

    embeddings = {
        "wsi": torch.ones(3, 4),
        "genomic": torch.ones(3, 4),
        "clinical": torch.tensor([[1.0, 2.0, 3.0, 4.0]] * 3),
    }

    output_path, correlation = analyzer.compute_modality_correlation(embeddings)

    assert output_path.endswith(".png")
    assert (tmp_path / "modality_correlation.png").exists()
    assert np.isfinite(correlation).all()
    assert np.allclose(np.diag(correlation), 1.0)
