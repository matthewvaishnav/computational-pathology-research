"""Unit tests for foundation model encoders."""

import pytest
import torch

from src.models.foundation import (
    FeatureProjector,
    PhikonEncoder,
    load_foundation_model,
)


class TestPhikonEncoder:
    """Tests for Phikon encoder."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Requires GPU for foundation model tests"
    )
    def test_phikon_load(self):
        """Test Phikon model loading."""
        encoder = PhikonEncoder(freeze=True)
        assert encoder.feature_dim == 768
        assert encoder.freeze is True

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Requires GPU for foundation model tests"
    )
    def test_phikon_forward(self):
        """Test Phikon forward pass."""
        encoder = PhikonEncoder(freeze=True)
        encoder.eval()

        # 224x224 input (Phikon expects this size)
        x = torch.randn(2, 3, 224, 224)
        features = encoder(x)

        assert features.shape == (2, 768)
        assert features.requires_grad is False  # frozen

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Requires GPU for foundation model tests"
    )
    def test_phikon_extract_features(self):
        """Test legacy extract_features interface."""
        encoder = PhikonEncoder(freeze=True)
        encoder.eval()

        x = torch.randn(2, 3, 224, 224)
        features = encoder.extract_features(x)

        assert features.shape == (2, 768)


class TestFeatureProjector:
    """Tests for feature projector."""

    def test_projector_init(self):
        """Test projector initialization."""
        projector = FeatureProjector(input_dim=768, output_dim=256)
        assert projector.input_dim == 768
        assert projector.output_dim == 256

    def test_projector_forward(self):
        """Test projector forward pass."""
        projector = FeatureProjector(input_dim=768, output_dim=256)

        x = torch.randn(4, 768)
        output = projector(x)

        assert output.shape == (4, 256)

    def test_projector_different_dims(self):
        """Test projector with different dimensions."""
        # UNI: 1024 → 256
        projector_uni = FeatureProjector(input_dim=1024, output_dim=256)
        x_uni = torch.randn(2, 1024)
        output_uni = projector_uni(x_uni)
        assert output_uni.shape == (2, 256)

        # CONCH: 512 → 256
        projector_conch = FeatureProjector(input_dim=512, output_dim=256)
        x_conch = torch.randn(2, 512)
        output_conch = projector_conch(x_conch)
        assert output_conch.shape == (2, 256)

    def test_projector_trainable(self):
        """Test that projector parameters are trainable."""
        projector = FeatureProjector(input_dim=768, output_dim=256)

        trainable_params = sum(
            p.numel() for p in projector.parameters() if p.requires_grad
        )
        assert trainable_params > 0

    def test_projector_get_num_params(self):
        """Test parameter counting."""
        projector = FeatureProjector(input_dim=768, output_dim=256)
        num_params = projector.get_num_params()

        assert num_params > 0
        assert isinstance(num_params, int)


class TestLoadFoundationModel:
    """Tests for load_foundation_model factory function."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Requires GPU for foundation model tests"
    )
    def test_load_phikon(self):
        """Test loading Phikon via factory."""
        encoder = load_foundation_model("phikon", freeze=True)

        assert isinstance(encoder, PhikonEncoder)
        assert encoder.feature_dim == 768

    def test_load_invalid_model(self):
        """Test loading invalid model name."""
        with pytest.raises(ValueError, match="Unknown model"):
            load_foundation_model("invalid_model")

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Requires GPU for foundation model tests"
    )
    def test_load_with_freeze_false(self):
        """Test loading with trainable weights."""
        encoder = load_foundation_model("phikon", freeze=False)

        assert encoder.freeze is False


class TestFoundationModelIntegration:
    """Integration tests for foundation model pipeline."""

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Requires GPU for foundation model tests"
    )
    def test_encoder_projector_pipeline(self):
        """Test full encoder + projector pipeline."""
        # Load encoder
        encoder = load_foundation_model("phikon", freeze=True)
        encoder.eval()

        # Create projector
        projector = FeatureProjector(
            input_dim=encoder.feature_dim,
            output_dim=256
        )

        # Forward pass
        x = torch.randn(4, 3, 224, 224)

        with torch.no_grad():
            raw_features = encoder(x)  # [4, 768] - frozen

        features = projector(raw_features)  # [4, 256] - trainable

        assert raw_features.shape == (4, 768)
        assert features.shape == (4, 256)
        assert raw_features.requires_grad is False
        assert features.requires_grad is True

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="Requires GPU for foundation model tests"
    )
    def test_batch_processing(self):
        """Test processing multiple batches."""
        encoder = load_foundation_model("phikon", freeze=True)
        encoder.eval()

        projector = FeatureProjector(input_dim=768, output_dim=256)

        batch_sizes = [1, 2, 4, 8]
        for bs in batch_sizes:
            x = torch.randn(bs, 3, 224, 224)

            with torch.no_grad():
                raw_features = encoder(x)

            features = projector(raw_features)

            assert features.shape == (bs, 256)
