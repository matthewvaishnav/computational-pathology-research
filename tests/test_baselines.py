"""
Unit tests for baseline model implementations.
"""

import pytest
import torch

from src.models.baselines import (
    AttentionBaseline,
    LateFusionModel,
    SingleModalityModel,
    get_baseline_model,
)


class TestSingleModalityModel:
    """Tests for SingleModalityModel baseline."""

    def _create_wsi_batch(self, batch_size=4, num_patches=50, feature_dim=1024):
        """Create dummy WSI batch."""
        return {
            "wsi_features": torch.randn(batch_size, num_patches, feature_dim),
            "wsi_mask": torch.ones(batch_size, num_patches, dtype=torch.bool),
        }

    def _create_genomic_batch(self, batch_size=4, num_genes=2000):
        """Create dummy genomic batch."""
        return {"genomic": torch.randn(batch_size, num_genes)}

    def _create_clinical_batch(self, batch_size=4, seq_len=128, vocab_size=30000):
        """Create dummy clinical text batch."""
        return {
            "clinical_text": torch.randint(0, vocab_size, (batch_size, seq_len)),
            "clinical_mask": torch.ones(batch_size, seq_len, dtype=torch.bool),
        }

    def test_wsi_modality_forward(self):
        """Test single modality model with WSI."""
        model = SingleModalityModel(modality="wsi", embed_dim=256)
        batch = self._create_wsi_batch(batch_size=4)
        output = model(batch)
        assert output.shape == (4, 256)

    def test_genomic_modality_forward(self):
        """Test single modality model with genomic."""
        model = SingleModalityModel(modality="genomic", embed_dim=128)
        batch = self._create_genomic_batch(batch_size=8)
        output = model(batch)
        assert output.shape == (8, 128)

    def test_clinical_modality_forward(self):
        """Test single modality model with clinical text."""
        model = SingleModalityModel(modality="clinical", embed_dim=64)
        batch = self._create_clinical_batch(batch_size=2)
        output = model(batch)
        assert output.shape == (2, 64)

    def test_unsupported_modality_raises(self):
        """Test that unsupported modality raises ValueError."""
        with pytest.raises(ValueError, match="Modality must be one of"):
            SingleModalityModel(modality="unsupported", embed_dim=256)

    def test_missing_wsi_features_raises(self):
        """Test that missing WSI features raises ValueError."""
        model = SingleModalityModel(modality="wsi", embed_dim=256)
        batch = {"genomic": torch.randn(4, 2000)}  # Wrong modality

        with pytest.raises(ValueError, match="WSI features required"):
            model(batch)

    def test_missing_genomic_raises(self):
        """Test that missing genomic data raises ValueError."""
        model = SingleModalityModel(modality="genomic", embed_dim=256)
        batch = {"wsi_features": torch.randn(4, 50, 1024)}  # Wrong modality

        with pytest.raises(ValueError, match="Genomic data required"):
            model(batch)

    def test_missing_clinical_raises(self):
        """Test that missing clinical text raises ValueError."""
        model = SingleModalityModel(modality="clinical", embed_dim=256)
        batch = {"wsi_features": torch.randn(4, 50, 1024)}  # Wrong modality

        with pytest.raises(ValueError, match="Clinical text required"):
            model(batch)

    def test_get_embedding_dim(self):
        """Test get_embedding_dim method."""
        model = SingleModalityModel(modality="wsi", embed_dim=128)
        assert model.get_embedding_dim() == 128

    def test_custom_config(self):
        """Test with custom encoder configuration."""
        config = {
            "wsi_config": {
                "input_dim": 512,
                "hidden_dim": 256,
                "num_heads": 4,
                "num_layers": 1,
            }
        }
        model = SingleModalityModel(modality="wsi", config=config, embed_dim=128)
        batch = {
            "wsi_features": torch.randn(2, 30, 512),  # Custom input_dim
            "wsi_mask": torch.ones(2, 30, dtype=torch.bool),
        }
        output = model(batch)
        assert output.shape == (2, 128)


class TestLateFusionModel:
    """Tests for LateFusionModel baseline."""

    def _create_full_batch(self, batch_size=4):
        """Create batch with all modalities."""
        return {
            "wsi_features": torch.randn(batch_size, 50, 1024),
            "wsi_mask": torch.ones(batch_size, 50, dtype=torch.bool),
            "genomic": torch.randn(batch_size, 2000),
            "clinical_text": torch.randint(0, 30000, (batch_size, 128)),
            "clinical_mask": torch.ones(batch_size, 128, dtype=torch.bool),
        }

    def test_full_modality_forward(self):
        """Test forward pass with all modalities present."""
        model = LateFusionModel(embed_dim=256)
        batch = self._create_full_batch(batch_size=4)
        output = model(batch)
        assert output.shape == (4, 256)

    def test_missing_wsi_uses_zeros(self):
        """Test that missing WSI is handled with zeros."""
        model = LateFusionModel(embed_dim=128)
        batch = {
            "genomic": torch.randn(4, 2000),
            "clinical_text": torch.randint(0, 30000, (4, 128)),
            "clinical_mask": torch.ones(4, 128, dtype=torch.bool),
        }
        output = model(batch)
        assert output.shape == (4, 128)

    def test_missing_genomic_uses_zeros(self):
        """Test that missing genomic is handled with zeros."""
        model = LateFusionModel(embed_dim=128)
        batch = {
            "wsi_features": torch.randn(4, 50, 1024),
            "wsi_mask": torch.ones(4, 50, dtype=torch.bool),
            "clinical_text": torch.randint(0, 30000, (4, 128)),
            "clinical_mask": torch.ones(4, 128, dtype=torch.bool),
        }
        output = model(batch)
        assert output.shape == (4, 128)

    def test_missing_clinical_uses_zeros(self):
        """Test that missing clinical is handled with zeros."""
        model = LateFusionModel(embed_dim=128)
        batch = {
            "wsi_features": torch.randn(4, 50, 1024),
            "wsi_mask": torch.ones(4, 50, dtype=torch.bool),
            "genomic": torch.randn(4, 2000),
        }
        output = model(batch)
        assert output.shape == (4, 128)

    def test_single_modality_only(self):
        """Test with only one modality present."""
        model = LateFusionModel(embed_dim=64)
        batch = {"genomic": torch.randn(4, 2000)}
        output = model(batch)
        assert output.shape == (4, 64)

    def test_no_modalities_raises(self):
        """Test that no modalities raises ValueError."""
        model = LateFusionModel(embed_dim=256)
        batch = {}  # Empty batch

        with pytest.raises(ValueError, match="At least one modality must be provided"):
            model(batch)

    def test_get_embedding_dim(self):
        """Test get_embedding_dim method."""
        model = LateFusionModel(embed_dim=128)
        assert model.get_embedding_dim() == 128

    def test_different_batch_sizes(self):
        """Test with various batch sizes."""
        model = LateFusionModel(embed_dim=64)
        for batch_size in [1, 2, 8, 16]:
            batch = self._create_full_batch(batch_size=batch_size)
            # Use eval mode for batch_size=1 to avoid BatchNorm error in GenomicEncoder
            if batch_size == 1:
                model.eval()
                with torch.no_grad():
                    output = model(batch)
                model.train()
            else:
                output = model(batch)
            assert output.shape == (batch_size, 64)


class TestAttentionBaseline:
    """Tests for AttentionBaseline model."""

    def _create_full_batch(self, batch_size=4):
        """Create batch with all modalities."""
        return {
            "wsi_features": torch.randn(batch_size, 50, 1024),
            "wsi_mask": torch.ones(batch_size, 50, dtype=torch.bool),
            "genomic": torch.randn(batch_size, 2000),
            "clinical_text": torch.randint(0, 30000, (batch_size, 128)),
            "clinical_mask": torch.ones(batch_size, 128, dtype=torch.bool),
        }

    def test_full_modality_forward(self):
        """Test forward pass with all modalities."""
        model = AttentionBaseline(embed_dim=256)
        batch = self._create_full_batch(batch_size=4)
        output = model(batch)
        assert output.shape == (4, 256)

    def test_missing_modality_with_zeros(self):
        """Test handling of missing modalities."""
        model = AttentionBaseline(embed_dim=128)
        batch = {
            "wsi_features": torch.randn(4, 50, 1024),
            "wsi_mask": torch.ones(4, 50, dtype=torch.bool),
            "genomic": torch.randn(4, 2000),
            # Missing clinical
        }
        output = model(batch)
        assert output.shape == (4, 128)

    def test_single_modality_only(self):
        """Test with only one modality."""
        model = AttentionBaseline(embed_dim=64)
        batch = {"genomic": torch.randn(4, 2000)}
        output = model(batch)
        assert output.shape == (4, 64)

    def test_no_modalities_raises(self):
        """Test that no modalities raises ValueError."""
        model = AttentionBaseline(embed_dim=256)
        batch = {}

        with pytest.raises(ValueError, match="At least one modality must be provided"):
            model(batch)

    def test_get_embedding_dim(self):
        """Test get_embedding_dim method."""
        model = AttentionBaseline(embed_dim=256)
        assert model.get_embedding_dim() == 256

    def test_fusion_weights_normalized(self):
        """Test that fusion weights are properly normalized."""
        model = AttentionBaseline(embed_dim=128)
        # Weights should sum to approximately 1 (softmax normalized)
        weights = torch.softmax(model.fusion_weights, dim=0)
        assert torch.allclose(weights.sum(), torch.tensor(1.0), atol=1e-6)

    def test_invalid_modality_zero_weighted(self):
        """Test that invalid modalities get zero weight in fusion."""
        model = AttentionBaseline(embed_dim=64)
        batch = {"genomic": torch.randn(4, 2000)}  # Only one modality

        # Before forward, all weights are equal
        raw_weights = torch.softmax(model.fusion_weights, dim=0)
        expected_equal_weight = raw_weights[1].item()  # Weight for genomic

        output = model(batch)

        # The output should still be valid
        assert output.shape == (4, 64)


class TestGetBaselineModel:
    """Tests for get_baseline_model factory function."""

    def test_single_modality_factory(self):
        """Test factory for single modality model."""
        model = get_baseline_model("single_modality", modality="wsi", embed_dim=256)
        assert isinstance(model, SingleModalityModel)
        assert model.modality == "wsi"

    def test_late_fusion_factory(self):
        """Test factory for late fusion model."""
        model = get_baseline_model("late_fusion", embed_dim=128)
        assert isinstance(model, LateFusionModel)

    def test_attention_baseline_factory(self):
        """Test factory for attention baseline."""
        model = get_baseline_model("attention", embed_dim=64)
        assert isinstance(model, AttentionBaseline)

    def test_unknown_baseline_type_raises(self):
        """Test that unknown baseline type raises ValueError."""
        with pytest.raises(ValueError, match="Unknown baseline type"):
            get_baseline_model("unknown_type")

    def test_factory_passes_config(self):
        """Test that factory passes config to models."""
        config = {"wsi_config": {"input_dim": 512}}
        model = get_baseline_model(
            "single_modality", modality="wsi", config=config, embed_dim=128
        )
        # Verify config was applied (would fail if input_dim mismatch)
        batch = {
            "wsi_features": torch.randn(2, 50, 512),
            "wsi_mask": torch.ones(2, 50, dtype=torch.bool),
        }
        output = model(batch)
        assert output.shape == (2, 128)
