"""Tests for pretrained model loading and feature extraction."""

import pytest
import torch

from src.models import (
    PretrainedFeatureExtractor,
    create_wsi_encoder_with_pretrained,
    get_recommended_model,
    list_pretrained_models,
)


def test_list_pretrained_models():
    """Should return list of available models."""
    models = list_pretrained_models()
    assert isinstance(models, list)
    assert len(models) > 0

    for model_info in models:
        assert "name" in model_info
        assert "full_name" in model_info
        assert "description" in model_info
        assert "output_dim" in model_info


def test_get_recommended_model():
    """Should return model name for valid tasks."""
    for task in ["general", "gigapixel", "fast", "baseline"]:
        model = get_recommended_model(task)
        assert isinstance(model, str)
        assert len(model) > 0


def test_get_recommended_model_invalid():
    """Should raise error for invalid task."""
    with pytest.raises(ValueError):
        get_recommended_model("invalid_task")


def test_pretrained_feature_extractor_invalid_model():
    """Should raise error for unknown model."""
    with pytest.raises(ValueError, match="Unknown model"):
        PretrainedFeatureExtractor("nonexistent_model")


def test_create_wsi_encoder_with_pretrained():
    """Should create extractor + encoder combo."""
    # Use ResNet50 as it doesn't require external downloads
    extractor, encoder = create_wsi_encoder_with_pretrained(
        pretrained_model="resnet50_imagenet",
        output_dim=256,
        freeze_pretrained=True,
    )

    assert isinstance(extractor, PretrainedFeatureExtractor)
    assert extractor.output_dim == 2048  # ResNet50 output
    assert extractor.freeze is True


def test_pretrained_feature_extractor_resnet_forward():
    """Should extract features from image patches."""
    extractor = PretrainedFeatureExtractor(
        "resnet50_imagenet",
        freeze=True,
    )

    # Create dummy patches [batch, 3, 224, 224]
    patches = torch.randn(4, 3, 224, 224)

    features = extractor(patches)

    assert features.shape == (4, 2048)  # ResNet50 outputs 2048-dim
    assert not torch.isnan(features).any()
    assert not torch.isinf(features).any()


def test_pretrained_feature_extractor_freeze():
    """Should freeze weights when freeze=True."""
    extractor = PretrainedFeatureExtractor("resnet50_imagenet", freeze=True)

    for param in extractor.backbone.parameters():
        assert not param.requires_grad


def test_pretrained_feature_extractor_unfreeze():
    """Should allow training when freeze=False."""
    extractor = PretrainedFeatureExtractor("resnet50_imagenet", freeze=False)

    # At least some parameters should be trainable
    trainable = any(p.requires_grad for p in extractor.backbone.parameters())
    assert trainable
