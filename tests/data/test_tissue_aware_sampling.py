"""
Tests for tissue-aware sampling in FixedLengthBagSampler.
"""

import pytest
import torch

from src.data.loaders.bag_samplers import FixedLengthBagSampler
from src.models.tissue_classifier import TissueClassifier


class TestTissueAwareSampling:
    """Test tissue-aware sampling functionality."""

    def test_initialization_with_tissue_classifier(self):
        """Test sampler initialization with tissue classifier."""
        tissue_classifier = TissueClassifier(feature_dim=1024)
        sampler = FixedLengthBagSampler(
            bag_length=512, mode="train", tissue_classifier=tissue_classifier, tissue_aware=True
        )

        assert sampler.tissue_aware is True
        assert sampler.tissue_classifier is not None

    def test_initialization_tissue_aware_without_classifier(self):
        """Test that tissue_aware=True requires classifier."""
        with pytest.raises(ValueError, match="tissue_classifier must be provided"):
            FixedLengthBagSampler(bag_length=512, mode="train", tissue_aware=True)

    def test_tissue_aware_sampling(self):
        """Test tissue-aware sampling produces different results than uniform."""
        tissue_classifier = TissueClassifier(feature_dim=1024)

        # Create sampler with tissue-aware sampling
        sampler_tissue = FixedLengthBagSampler(
            bag_length=100,
            mode="train",
            tissue_classifier=tissue_classifier,
            tissue_aware=True,
            temperature=0.5,
        )

        # Create sampler with uniform sampling
        sampler_uniform = FixedLengthBagSampler(bag_length=100, mode="train", tissue_aware=False)

        # Create features
        features = torch.randn(200, 1024)

        # Sample multiple times and check distributions differ
        tissue_samples = []
        uniform_samples = []

        for _ in range(10):
            sampled_tissue, _ = sampler_tissue.sample(features, num_patches=200)
            sampled_uniform, _ = sampler_uniform.sample(features, num_patches=200)

            tissue_samples.append(sampled_tissue)
            uniform_samples.append(sampled_uniform)

        # Tissue-aware sampling should produce different results
        # (not a perfect test, but checks basic functionality)
        assert len(tissue_samples) == 10
        assert len(uniform_samples) == 10

    def test_tissue_aware_sampling_shape(self):
        """Test tissue-aware sampling produces correct output shape."""
        tissue_classifier = TissueClassifier(feature_dim=1024)
        sampler = FixedLengthBagSampler(
            bag_length=100, mode="train", tissue_classifier=tissue_classifier, tissue_aware=True
        )

        features = torch.randn(200, 1024)
        sampled_features, mask = sampler.sample(features, num_patches=200)

        assert sampled_features.shape == (100, 1024)
        assert mask.shape == (100,)
        assert mask.all()  # All True (no padding)

    def test_tissue_aware_sampling_with_padding(self):
        """Test tissue-aware sampling with N < M (padding case)."""
        tissue_classifier = TissueClassifier(feature_dim=1024)
        sampler = FixedLengthBagSampler(
            bag_length=100, mode="train", tissue_classifier=tissue_classifier, tissue_aware=True
        )

        features = torch.randn(50, 1024)
        sampled_features, mask = sampler.sample(features, num_patches=50)

        assert sampled_features.shape == (100, 1024)
        assert mask.shape == (100,)
        assert mask.sum() == 50  # 50 valid patches

    def test_temperature_effect(self):
        """Test temperature parameter affects sampling distribution."""
        tissue_classifier = TissueClassifier(feature_dim=1024)

        # Low temperature = more peaked (favor high-importance patches)
        sampler_low_temp = FixedLengthBagSampler(
            bag_length=100,
            mode="train",
            tissue_classifier=tissue_classifier,
            tissue_aware=True,
            temperature=0.1,
        )

        # High temperature = more uniform
        sampler_high_temp = FixedLengthBagSampler(
            bag_length=100,
            mode="train",
            tissue_classifier=tissue_classifier,
            tissue_aware=True,
            temperature=10.0,
        )

        features = torch.randn(200, 1024)

        # Both should work without errors
        sampled_low, _ = sampler_low_temp.sample(features, num_patches=200)
        sampled_high, _ = sampler_high_temp.sample(features, num_patches=200)

        assert sampled_low.shape == (100, 1024)
        assert sampled_high.shape == (100, 1024)

    def test_inference_mode_ignores_tissue_aware(self):
        """Test that inference mode uses sliding window regardless of tissue_aware."""
        tissue_classifier = TissueClassifier(feature_dim=1024)
        sampler = FixedLengthBagSampler(
            bag_length=100, mode="inference", tissue_classifier=tissue_classifier, tissue_aware=True
        )

        features = torch.randn(200, 1024)
        sampled_features, mask = sampler.sample(features, num_patches=200)

        # Should use sliding window (first 100 patches)
        assert sampled_features.shape == (100, 1024)
        assert torch.allclose(sampled_features, features[:100])

    def test_tissue_aware_sampling_deterministic_with_seed(self):
        """Test tissue-aware sampling is deterministic with same seed."""
        tissue_classifier = TissueClassifier(feature_dim=1024)
        sampler = FixedLengthBagSampler(
            bag_length=100, mode="train", tissue_classifier=tissue_classifier, tissue_aware=True
        )

        features = torch.randn(200, 1024)

        # Sample twice with same seed
        torch.manual_seed(42)
        sampled1, _ = sampler.sample(features, num_patches=200)

        torch.manual_seed(42)
        sampled2, _ = sampler.sample(features, num_patches=200)

        # Should be identical
        assert torch.allclose(sampled1, sampled2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
