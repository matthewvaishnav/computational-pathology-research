"""
Unit tests for modality-specific encoders.
"""

import pytest
import torch
from src.models.encoders import (
    WSIEncoder,
    GenomicEncoder,
    ClinicalTextEncoder
)


class TestWSIEncoder:
    """Tests for WSIEncoder module."""
    
    def test_wsi_encoder_forward(self):
        """Test WSI encoder with various input dimensions."""
        encoder = WSIEncoder(input_dim=1024, output_dim=256)
        
        # Test with batch of patch features
        patches = torch.randn(4, 100, 1024)  # [batch, num_patches, feature_dim]
        output = encoder(patches)
        
        assert output.shape == (4, 256)
        
    def test_wsi_encoder_variable_patches(self):
        """Test WSI encoder with different numbers of patches."""
        encoder = WSIEncoder(input_dim=512, output_dim=128)
        
        # Test with different patch counts
        patches1 = torch.randn(2, 50, 512)
        output1 = encoder(patches1)
        assert output1.shape == (2, 128)
        
        patches2 = torch.randn(2, 200, 512)
        output2 = encoder(patches2)
        assert output2.shape == (2, 128)
        
    def test_wsi_encoder_with_mask(self):
        """Test WSI encoder with padding mask."""
        encoder = WSIEncoder(input_dim=1024, output_dim=256, pooling='attention')
        
        patches = torch.randn(2, 100, 1024)
        # Create mask: first sample has 80 valid patches, second has 60
        mask = torch.zeros(2, 100, dtype=torch.bool)
        mask[0, :80] = True
        mask[1, :60] = True
        
        output = encoder(patches, mask=mask)
        
        assert output.shape == (2, 256)
        
    def test_wsi_encoder_pooling_strategies(self):
        """Test different pooling strategies."""
        patches = torch.randn(2, 50, 512)
        
        # Test attention pooling
        encoder_attn = WSIEncoder(input_dim=512, output_dim=128, pooling='attention')
        output_attn = encoder_attn(patches)
        assert output_attn.shape == (2, 128)
        
        # Test mean pooling
        encoder_mean = WSIEncoder(input_dim=512, output_dim=128, pooling='mean')
        output_mean = encoder_mean(patches)
        assert output_mean.shape == (2, 128)
        
        # Test max pooling
        encoder_max = WSIEncoder(input_dim=512, output_dim=128, pooling='max')
        output_max = encoder_max(patches)
        assert output_max.shape == (2, 128)
        
    def test_wsi_encoder_invalid_pooling(self):
        """Test that invalid pooling strategy raises error."""
        encoder = WSIEncoder(input_dim=512, output_dim=128, pooling='invalid')
        patches = torch.randn(2, 50, 512)
        
        with pytest.raises(ValueError):
            encoder(patches)


class TestGenomicEncoder:
    """Tests for GenomicEncoder module."""
    
    def test_genomic_encoder_forward(self):
        """Test genomic encoder forward pass."""
        encoder = GenomicEncoder(input_dim=2000, output_dim=256)
        
        genomic_data = torch.randn(8, 2000)  # [batch, num_genes]
        output = encoder(genomic_data)
        
        assert output.shape == (8, 256)
        
    def test_genomic_encoder_different_dimensions(self):
        """Test genomic encoder with different input/output dimensions."""
        encoder = GenomicEncoder(input_dim=5000, output_dim=512)
        
        genomic_data = torch.randn(4, 5000)
        output = encoder(genomic_data)
        
        assert output.shape == (4, 512)
        
    def test_genomic_encoder_custom_hidden_dims(self):
        """Test genomic encoder with custom hidden dimensions."""
        encoder = GenomicEncoder(
            input_dim=1000,
            hidden_dims=[512, 256, 128],
            output_dim=64
        )
        
        genomic_data = torch.randn(2, 1000)
        output = encoder(genomic_data)
        
        assert output.shape == (2, 64)
        
    def test_genomic_encoder_without_batch_norm(self):
        """Test genomic encoder without batch normalization."""
        encoder = GenomicEncoder(
            input_dim=2000,
            output_dim=256,
            use_batch_norm=False
        )
        
        genomic_data = torch.randn(4, 2000)
        output = encoder(genomic_data)
        
        assert output.shape == (4, 256)
        
    def test_genomic_encoder_single_sample(self):
        """Test genomic encoder with single sample (batch size 1)."""
        encoder = GenomicEncoder(input_dim=1000, output_dim=128)
        encoder.eval()  # Set to eval mode to avoid batch norm issues with batch size 1
        
        genomic_data = torch.randn(1, 1000)
        output = encoder(genomic_data)
        
        assert output.shape == (1, 128)


class TestClinicalTextEncoder:
    """Tests for ClinicalTextEncoder module."""
    
    def test_clinical_text_encoder_forward(self):
        """Test clinical text encoder forward pass."""
        encoder = ClinicalTextEncoder(vocab_size=30000, output_dim=256)
        
        token_ids = torch.randint(0, 30000, (4, 128))  # [batch, seq_len]
        output = encoder(token_ids)
        
        assert output.shape == (4, 256)
        
    def test_clinical_text_encoder_variable_length(self):
        """Test clinical text encoder with different sequence lengths."""
        encoder = ClinicalTextEncoder(vocab_size=10000, output_dim=128)
        
        # Test with different sequence lengths
        tokens1 = torch.randint(1, 10000, (2, 50))
        output1 = encoder(tokens1)
        assert output1.shape == (2, 128)
        
        tokens2 = torch.randint(1, 10000, (2, 200))
        output2 = encoder(tokens2)
        assert output2.shape == (2, 128)
        
    def test_clinical_text_encoder_with_padding(self):
        """Test clinical text encoder with padded sequences."""
        encoder = ClinicalTextEncoder(vocab_size=10000, output_dim=256, pooling='mean')
        
        # Create sequences with padding (0 = padding token)
        token_ids = torch.randint(1, 10000, (2, 100))
        token_ids[0, 50:] = 0  # Pad second half of first sequence
        token_ids[1, 80:] = 0  # Pad last 20 tokens of second sequence
        
        output = encoder(token_ids)
        
        assert output.shape == (2, 256)
        
    def test_clinical_text_encoder_with_attention_mask(self):
        """Test clinical text encoder with explicit attention mask."""
        encoder = ClinicalTextEncoder(vocab_size=10000, output_dim=256)
        
        token_ids = torch.randint(1, 10000, (2, 100))
        attention_mask = torch.ones(2, 100, dtype=torch.bool)
        attention_mask[0, 60:] = False  # Mask last 40 tokens of first sequence
        attention_mask[1, 70:] = False  # Mask last 30 tokens of second sequence
        
        output = encoder(token_ids, attention_mask=attention_mask)
        
        assert output.shape == (2, 256)
        
    def test_clinical_text_encoder_pooling_strategies(self):
        """Test different pooling strategies."""
        token_ids = torch.randint(1, 5000, (2, 64))
        
        # Test CLS pooling
        encoder_cls = ClinicalTextEncoder(
            vocab_size=5000,
            output_dim=128,
            pooling='cls'
        )
        output_cls = encoder_cls(token_ids)
        assert output_cls.shape == (2, 128)
        
        # Test mean pooling
        encoder_mean = ClinicalTextEncoder(
            vocab_size=5000,
            output_dim=128,
            pooling='mean'
        )
        output_mean = encoder_mean(token_ids)
        assert output_mean.shape == (2, 128)
        
        # Test max pooling
        encoder_max = ClinicalTextEncoder(
            vocab_size=5000,
            output_dim=128,
            pooling='max'
        )
        output_max = encoder_max(token_ids)
        assert output_max.shape == (2, 128)
        
    def test_clinical_text_encoder_long_sequence(self):
        """Test clinical text encoder with sequence longer than max_seq_length."""
        encoder = ClinicalTextEncoder(
            vocab_size=10000,
            output_dim=256,
            max_seq_length=128
        )
        
        # Create sequence longer than max_seq_length
        token_ids = torch.randint(1, 10000, (2, 256))
        output = encoder(token_ids)
        
        # Should truncate and still produce valid output
        assert output.shape == (2, 256)
        
    def test_clinical_text_encoder_invalid_pooling(self):
        """Test that invalid pooling strategy raises error."""
        encoder = ClinicalTextEncoder(
            vocab_size=10000,
            output_dim=256,
            pooling='invalid'
        )
        token_ids = torch.randint(1, 10000, (2, 64))
        
        with pytest.raises(ValueError):
            encoder(token_ids)
            
    def test_clinical_text_encoder_output_consistency(self):
        """Test that encoder produces consistent output shapes across batches."""
        encoder = ClinicalTextEncoder(vocab_size=10000, output_dim=256)
        
        batch_sizes = [1, 2, 4, 8]
        for batch_size in batch_sizes:
            token_ids = torch.randint(1, 10000, (batch_size, 100))
            output = encoder(token_ids)
            assert output.shape == (batch_size, 256)


class TestEncoderIntegration:
    """Integration tests for all encoders."""
    
    def test_all_encoders_same_output_dim(self):
        """Test that all encoders can produce same output dimension."""
        output_dim = 256
        
        # Create encoders
        wsi_encoder = WSIEncoder(input_dim=1024, output_dim=output_dim)
        genomic_encoder = GenomicEncoder(input_dim=2000, output_dim=output_dim)
        text_encoder = ClinicalTextEncoder(vocab_size=10000, output_dim=output_dim)
        
        # Create inputs
        wsi_input = torch.randn(2, 100, 1024)
        genomic_input = torch.randn(2, 2000)
        text_input = torch.randint(1, 10000, (2, 128))
        
        # Encode
        wsi_emb = wsi_encoder(wsi_input)
        genomic_emb = genomic_encoder(genomic_input)
        text_emb = text_encoder(text_input)
        
        # All should have same output dimension
        assert wsi_emb.shape == (2, output_dim)
        assert genomic_emb.shape == (2, output_dim)
        assert text_emb.shape == (2, output_dim)
        
    def test_encoders_gradient_flow(self):
        """Test that gradients flow through all encoders."""
        # Create encoders
        wsi_encoder = WSIEncoder(input_dim=512, output_dim=128)
        genomic_encoder = GenomicEncoder(input_dim=1000, output_dim=128)
        text_encoder = ClinicalTextEncoder(vocab_size=5000, output_dim=128)
        
        # Create inputs
        wsi_input = torch.randn(2, 50, 512, requires_grad=True)
        genomic_input = torch.randn(2, 1000, requires_grad=True)
        text_input = torch.randint(1, 5000, (2, 64))
        
        # Forward pass
        wsi_emb = wsi_encoder(wsi_input)
        genomic_emb = genomic_encoder(genomic_input)
        text_emb = text_encoder(text_input)
        
        # Compute loss and backward
        loss = wsi_emb.sum() + genomic_emb.sum() + text_emb.sum()
        loss.backward()
        
        # Check gradients exist
        assert wsi_input.grad is not None
        assert genomic_input.grad is not None

