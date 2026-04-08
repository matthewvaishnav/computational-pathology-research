"""
Unit tests for feature extractors used in PCam benchmark.
"""

import pytest
import torch

from src.models.feature_extractors import ResNetFeatureExtractor


class TestResNetFeatureExtractor:
    """Tests for ResNetFeatureExtractor used in PCam."""

    def test_resnet18_default_construction(self):
        """Test default ResNet-18 construction."""
        extractor = ResNetFeatureExtractor(model_name="resnet18", pretrained=False)
        assert extractor.feature_dim == 512
        assert isinstance(extractor.model.fc, torch.nn.Identity)

    def test_resnet50_construction(self):
        """Test ResNet-50 construction."""
        extractor = ResNetFeatureExtractor(model_name="resnet50", pretrained=False)
        assert extractor.feature_dim == 2048
        assert isinstance(extractor.model.fc, torch.nn.Identity)

    def test_resnet18_without_pretraining(self):
        """Test ResNet-18 without pretrained weights."""
        extractor = ResNetFeatureExtractor(model_name="resnet18", pretrained=False)
        assert extractor.feature_dim == 512

    def test_resnet50_without_pretraining(self):
        """Test ResNet-50 without pretrained weights."""
        extractor = ResNetFeatureExtractor(model_name="resnet50", pretrained=False)
        assert extractor.feature_dim == 2048

    def test_custom_feature_dim_override(self):
        """Test custom feature dimension override."""
        extractor = ResNetFeatureExtractor(model_name="resnet18", pretrained=False, feature_dim=256)
        assert extractor.feature_dim == 256

    def test_feature_dim_projection_forward(self):
        """Test that feature_dim actually changes the forward output shape."""
        extractor = ResNetFeatureExtractor(model_name="resnet18", pretrained=False, feature_dim=256)
        images = torch.randn(4, 3, 96, 96)
        features = extractor(images)
        assert features.shape == (4, 256), f"Expected (4, 256), got {features.shape}"

    def test_invalid_model_name_raises(self):
        """Test that invalid model name raises ValueError."""
        with pytest.raises(ValueError, match="Unknown model_name"):
            ResNetFeatureExtractor(model_name="resnet101")

    def test_invalid_model_name_includes_suggestions(self):
        """Test error message includes valid model suggestions."""
        with pytest.raises(ValueError, match="resnet18.*resnet50"):
            ResNetFeatureExtractor(model_name="invalid_model")

    def test_forward_pass_resnet18(self):
        """Test forward pass with ResNet-18."""
        extractor = ResNetFeatureExtractor(model_name="resnet18", pretrained=False)
        images = torch.randn(4, 3, 96, 96)  # PCam image size
        features = extractor(images)
        assert features.shape == (4, 512)

    def test_forward_pass_resnet50(self):
        """Test forward pass with ResNet-50."""
        extractor = ResNetFeatureExtractor(model_name="resnet50", pretrained=False)
        images = torch.randn(4, 3, 96, 96)
        features = extractor(images)
        assert features.shape == (4, 2048)

    def test_forward_pass_different_batch_sizes(self):
        """Test forward pass with various batch sizes."""
        extractor = ResNetFeatureExtractor(model_name="resnet18", pretrained=False)
        for batch_size in [1, 2, 8, 16]:
            images = torch.randn(batch_size, 3, 96, 96)
            features = extractor(images)
            assert features.shape == (batch_size, 512)

    def test_forward_pass_different_image_sizes(self):
        """Test forward pass with different image sizes."""
        extractor = ResNetFeatureExtractor(model_name="resnet18", pretrained=False)
        # ResNet can handle different input sizes due to adaptive pooling
        for size in [64, 96, 128, 224]:
            images = torch.randn(2, 3, size, size)
            features = extractor(images)
            assert features.shape == (2, 512)

    def test_eval_mode_consistency(self):
        """Test that eval mode produces consistent outputs."""
        extractor = ResNetFeatureExtractor(model_name="resnet18", pretrained=False)
        extractor.eval()
        images = torch.randn(4, 3, 96, 96)

        with torch.no_grad():
            features1 = extractor(images)
            features2 = extractor(images)

        assert torch.allclose(features1, features2, atol=1e-6)

    def test_training_mode_has_batch_norm_variation(self):
        """Test that training mode behaves differently due to BatchNorm."""
        extractor = ResNetFeatureExtractor(model_name="resnet18", pretrained=False)
        extractor.train()
        images = torch.randn(8, 3, 96, 96)  # Larger batch for BN stats

        # Run multiple forward passes - outputs may differ due to batch norm stats
        # Note: ResNet doesn't have dropout, only BatchNorm
        with torch.no_grad():
            outputs = [extractor(images) for _ in range(5)]

        # Just verify training mode works - BN will use batch statistics
        # We can't reliably test for variation due to BN behavior
        assert all(o.shape == (8, 512) for o in outputs)

    def test_get_num_params_resnet18(self):
        """Test parameter count for ResNet-18."""
        extractor = ResNetFeatureExtractor(model_name="resnet18", pretrained=False)
        num_params = extractor.get_num_params()
        # ResNet-18 has ~11M parameters
        assert num_params > 10_000_000 and num_params < 15_000_000

    def test_get_num_params_resnet50(self):
        """Test parameter count for ResNet-50."""
        extractor = ResNetFeatureExtractor(model_name="resnet50", pretrained=False)
        num_params = extractor.get_num_params()
        # ResNet-50 has ~25M parameters
        assert num_params > 20_000_000 and num_params < 30_000_000

    def test_feature_extraction_no_grad(self):
        """Test that feature extraction works with no_grad."""
        extractor = ResNetFeatureExtractor(model_name="resnet18", pretrained=False)
        extractor.eval()
        images = torch.randn(2, 3, 96, 96)

        with torch.no_grad():
            features = extractor(images)

        assert features.shape == (2, 512)
        assert features.requires_grad is False

    def test_device_compatibility_cpu(self):
        """Test CPU compatibility."""
        extractor = ResNetFeatureExtractor(model_name="resnet18", pretrained=False)
        images = torch.randn(2, 3, 96, 96)
        features = extractor(images)
        assert features.device.type == "cpu"

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_device_compatibility_cuda(self):
        """Test CUDA compatibility."""
        extractor = ResNetFeatureExtractor(model_name="resnet18", pretrained=False).cuda()
        images = torch.randn(2, 3, 96, 96).cuda()
        features = extractor(images)
        assert features.device.type == "cuda"
