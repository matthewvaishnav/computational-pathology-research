"""
Unit tests for FixedLengthBagSampler.

Tests cover:
- Padding when N < M
- Random sampling when N > M (train mode)
- Sliding window when N > M (inference mode)
- Attention mask correctness
- Configuration validation
- Edge cases
"""

import pytest
import torch

from src.data.bag_samplers import FixedLengthBagSampler


class TestFixedLengthBagSamplerInitialization:
    """Test sampler initialization and configuration validation."""
    
    def test_valid_initialization(self):
        """Test initialization with valid parameters."""
        sampler = FixedLengthBagSampler(bag_length=512, mode='train')
        assert sampler.bag_length == 512
        assert sampler.mode == 'train'
        assert sampler.stride == 512  # Default stride equals bag_length
    
    def test_custom_stride(self):
        """Test initialization with custom stride."""
        sampler = FixedLengthBagSampler(bag_length=512, mode='inference', stride=256)
        assert sampler.stride == 256
    
    def test_bag_length_too_small(self):
        """Test that bag_length < 100 raises ValueError."""
        with pytest.raises(ValueError, match="bag_length must be in range"):
            FixedLengthBagSampler(bag_length=99, mode='train')
    
    def test_bag_length_too_large(self):
        """Test that bag_length > 10000 raises ValueError."""
        with pytest.raises(ValueError, match="bag_length must be in range"):
            FixedLengthBagSampler(bag_length=10001, mode='train')
    
    def test_invalid_mode(self):
        """Test that invalid mode raises ValueError."""
        with pytest.raises(ValueError, match="mode must be 'train' or 'inference'"):
            FixedLengthBagSampler(bag_length=512, mode='invalid')
    
    def test_negative_stride(self):
        """Test that negative stride raises ValueError."""
        with pytest.raises(ValueError, match="stride must be positive"):
            FixedLengthBagSampler(bag_length=512, mode='inference', stride=-1)
    
    def test_zero_stride(self):
        """Test that zero stride raises ValueError."""
        with pytest.raises(ValueError, match="stride must be positive"):
            FixedLengthBagSampler(bag_length=512, mode='inference', stride=0)
    
    def test_boundary_bag_lengths(self):
        """Test boundary values for bag_length (100 and 10000)."""
        sampler_min = FixedLengthBagSampler(bag_length=100, mode='train')
        assert sampler_min.bag_length == 100
        
        sampler_max = FixedLengthBagSampler(bag_length=10000, mode='train')
        assert sampler_max.bag_length == 10000


class TestPaddingCase:
    """Test padding when N < M."""
    
    def test_padding_basic(self):
        """Test basic padding functionality."""
        sampler = FixedLengthBagSampler(bag_length=100, mode='train')
        features = torch.randn(50, 512)
        
        sampled_features, mask = sampler.sample(features, num_patches=50)
        
        # Check shapes
        assert sampled_features.shape == (100, 512)
        assert mask.shape == (100,)
        
        # Check mask: first 50 True, rest False
        assert mask[:50].all()
        assert not mask[50:].any()
        
        # Check that first 50 patches match original
        assert torch.allclose(sampled_features[:50], features[:50])
        
        # Check that padding is zeros
        assert torch.allclose(sampled_features[50:], torch.zeros(50, 512))
    
    def test_padding_preserves_dtype(self):
        """Test that padding preserves feature dtype."""
        sampler = FixedLengthBagSampler(bag_length=100, mode='train')
        features = torch.randn(50, 512, dtype=torch.float16)
        
        sampled_features, mask = sampler.sample(features, num_patches=50)
        
        assert sampled_features.dtype == torch.float16
    
    def test_padding_preserves_device(self):
        """Test that padding preserves feature device."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        sampler = FixedLengthBagSampler(bag_length=100, mode='train')
        features = torch.randn(50, 512, device='cuda')
        
        sampled_features, mask = sampler.sample(features, num_patches=50)
        
        assert sampled_features.device.type == 'cuda'
        assert mask.device.type == 'cuda'
    
    def test_padding_extreme_case(self):
        """Test padding when only 1 patch available."""
        sampler = FixedLengthBagSampler(bag_length=100, mode='train')
        features = torch.randn(1, 512)
        
        sampled_features, mask = sampler.sample(features, num_patches=1)
        
        assert sampled_features.shape == (100, 512)
        assert mask.sum() == 1
        assert mask[0]
        assert not mask[1:].any()


class TestRandomSamplingCase:
    """Test random sampling when N > M in train mode."""
    
    def test_random_sampling_basic(self):
        """Test basic random sampling functionality."""
        sampler = FixedLengthBagSampler(bag_length=100, mode='train')
        features = torch.randn(200, 512)
        
        sampled_features, mask = sampler.sample(features, num_patches=200)
        
        # Check shapes
        assert sampled_features.shape == (100, 512)
        assert mask.shape == (100,)
        
        # All patches should be valid (no padding)
        assert mask.all()
    
    def test_random_sampling_without_replacement(self):
        """Test that sampling is without replacement (no duplicates)."""
        sampler = FixedLengthBagSampler(bag_length=100, mode='train')
        
        # Create features with unique values for each patch
        features = torch.arange(200).unsqueeze(1).float()  # [200, 1]
        
        sampled_features, mask = sampler.sample(features, num_patches=200)
        
        # Check that all sampled values are unique
        sampled_values = sampled_features.squeeze()
        unique_values = torch.unique(sampled_values)
        assert len(unique_values) == 100  # All 100 sampled patches should be unique
    
    def test_random_sampling_randomness(self):
        """Test that sampling produces different results across calls."""
        sampler = FixedLengthBagSampler(bag_length=100, mode='train')
        features = torch.randn(200, 512)
        
        # Sample twice
        sampled1, _ = sampler.sample(features, num_patches=200)
        sampled2, _ = sampler.sample(features, num_patches=200)
        
        # Results should be different (with very high probability)
        assert not torch.allclose(sampled1, sampled2)
    
    def test_random_sampling_all_from_valid_range(self):
        """Test that sampled patches come from valid range [0, N)."""
        sampler = FixedLengthBagSampler(bag_length=100, mode='train')
        
        # Create features where each patch has a unique identifier
        N = 200
        features = torch.zeros(N, 512)
        for i in range(N):
            features[i, 0] = i  # First dimension is patch ID
        
        sampled_features, _ = sampler.sample(features, num_patches=N)
        
        # Check that all sampled IDs are in valid range
        sampled_ids = sampled_features[:, 0]
        assert (sampled_ids >= 0).all()
        assert (sampled_ids < N).all()


class TestSlidingWindowCase:
    """Test sliding window sampling when N > M in inference mode."""
    
    def test_sliding_window_basic(self):
        """Test basic sliding window functionality."""
        sampler = FixedLengthBagSampler(bag_length=100, mode='inference')
        features = torch.randn(200, 512)
        
        sampled_features, mask = sampler.sample(features, num_patches=200)
        
        # Check shapes
        assert sampled_features.shape == (100, 512)
        assert mask.shape == (100,)
        
        # All patches should be valid (no padding)
        assert mask.all()
    
    def test_sliding_window_first_window(self):
        """Test that sliding window returns first M patches."""
        sampler = FixedLengthBagSampler(bag_length=100, mode='inference')
        features = torch.randn(200, 512)
        
        sampled_features, _ = sampler.sample(features, num_patches=200)
        
        # Should return first 100 patches
        assert torch.allclose(sampled_features, features[:100])
    
    def test_sliding_window_deterministic(self):
        """Test that sliding window is deterministic."""
        sampler = FixedLengthBagSampler(bag_length=100, mode='inference')
        features = torch.randn(200, 512)
        
        # Sample twice
        sampled1, _ = sampler.sample(features, num_patches=200)
        sampled2, _ = sampler.sample(features, num_patches=200)
        
        # Results should be identical
        assert torch.allclose(sampled1, sampled2)
    
    def test_get_num_windows(self):
        """Test calculation of number of windows."""
        sampler = FixedLengthBagSampler(bag_length=100, mode='inference', stride=50)
        
        # Test various cases
        assert sampler.get_num_windows(50) == 1   # N < M
        assert sampler.get_num_windows(100) == 1  # N == M
        assert sampler.get_num_windows(150) == 2  # One full stride
        assert sampler.get_num_windows(200) == 3  # Two full strides
        assert sampler.get_num_windows(250) == 4  # Three full strides
    
    def test_get_num_windows_non_overlapping(self):
        """Test window count with non-overlapping windows (stride == bag_length)."""
        sampler = FixedLengthBagSampler(bag_length=100, mode='inference', stride=100)
        
        assert sampler.get_num_windows(100) == 1
        assert sampler.get_num_windows(200) == 2
        assert sampler.get_num_windows(300) == 3


class TestExactMatchCase:
    """Test when N == M (no padding or sampling needed)."""
    
    def test_exact_match_train_mode(self):
        """Test exact match in train mode."""
        sampler = FixedLengthBagSampler(bag_length=100, mode='train')
        features = torch.randn(100, 512)
        
        sampled_features, mask = sampler.sample(features, num_patches=100)
        
        # Should return features as-is
        assert torch.allclose(sampled_features, features)
        assert mask.all()
    
    def test_exact_match_inference_mode(self):
        """Test exact match in inference mode."""
        sampler = FixedLengthBagSampler(bag_length=100, mode='inference')
        features = torch.randn(100, 512)
        
        sampled_features, mask = sampler.sample(features, num_patches=100)
        
        # Should return features as-is
        assert torch.allclose(sampled_features, features)
        assert mask.all()


class TestInputValidation:
    """Test input validation in sample() method."""
    
    def test_invalid_tensor_dimension(self):
        """Test that non-2D tensor raises ValueError."""
        sampler = FixedLengthBagSampler(bag_length=100, mode='train')
        
        # 1D tensor
        with pytest.raises(ValueError, match="Expected 2D tensor"):
            sampler.sample(torch.randn(100), num_patches=100)
        
        # 3D tensor
        with pytest.raises(ValueError, match="Expected 2D tensor"):
            sampler.sample(torch.randn(10, 10, 512), num_patches=100)
    
    def test_negative_num_patches(self):
        """Test that negative num_patches raises ValueError."""
        sampler = FixedLengthBagSampler(bag_length=100, mode='train')
        features = torch.randn(100, 512)
        
        with pytest.raises(ValueError, match="num_patches must be in range"):
            sampler.sample(features, num_patches=-1)
    
    def test_num_patches_exceeds_features(self):
        """Test that num_patches > features.shape[0] raises ValueError."""
        sampler = FixedLengthBagSampler(bag_length=100, mode='train')
        features = torch.randn(100, 512)
        
        with pytest.raises(ValueError, match="num_patches must be in range"):
            sampler.sample(features, num_patches=101)


class TestAttentionMaskCorrectness:
    """Test attention mask correctness across all cases."""
    
    def test_mask_all_true_no_padding(self):
        """Test mask is all True when no padding needed."""
        sampler = FixedLengthBagSampler(bag_length=100, mode='train')
        features = torch.randn(200, 512)
        
        _, mask = sampler.sample(features, num_patches=200)
        
        assert mask.all()
        assert mask.dtype == torch.bool
    
    def test_mask_partial_true_with_padding(self):
        """Test mask has True for valid patches, False for padding."""
        sampler = FixedLengthBagSampler(bag_length=100, mode='train')
        features = torch.randn(60, 512)
        
        _, mask = sampler.sample(features, num_patches=60)
        
        # First 60 should be True
        assert mask[:60].all()
        # Remaining 40 should be False
        assert not mask[60:].any()
        assert mask.sum() == 60
    
    def test_mask_dtype_is_bool(self):
        """Test that mask dtype is always bool."""
        sampler = FixedLengthBagSampler(bag_length=100, mode='train')
        
        # Test with padding
        _, mask1 = sampler.sample(torch.randn(50, 512), num_patches=50)
        assert mask1.dtype == torch.bool
        
        # Test without padding
        _, mask2 = sampler.sample(torch.randn(200, 512), num_patches=200)
        assert mask2.dtype == torch.bool


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    def test_minimum_bag_length(self):
        """Test with minimum allowed bag_length (100)."""
        sampler = FixedLengthBagSampler(bag_length=100, mode='train')
        features = torch.randn(50, 512)
        
        sampled_features, mask = sampler.sample(features, num_patches=50)
        
        assert sampled_features.shape == (100, 512)
        assert mask.sum() == 50
    
    def test_maximum_bag_length(self):
        """Test with maximum allowed bag_length (10000)."""
        sampler = FixedLengthBagSampler(bag_length=10000, mode='train')
        features = torch.randn(5000, 512)
        
        sampled_features, mask = sampler.sample(features, num_patches=5000)
        
        assert sampled_features.shape == (10000, 512)
        assert mask.sum() == 5000
    
    def test_single_patch(self):
        """Test with single patch input."""
        sampler = FixedLengthBagSampler(bag_length=100, mode='train')
        features = torch.randn(1, 512)
        
        sampled_features, mask = sampler.sample(features, num_patches=1)
        
        assert sampled_features.shape == (100, 512)
        assert mask.sum() == 1
    
    def test_different_feature_dimensions(self):
        """Test with various feature dimensions."""
        sampler = FixedLengthBagSampler(bag_length=100, mode='train')
        
        # Test common foundation model dimensions
        for feat_dim in [512, 768, 1024, 2048]:
            features = torch.randn(50, feat_dim)
            sampled_features, mask = sampler.sample(features, num_patches=50)
            
            assert sampled_features.shape == (100, feat_dim)
            assert mask.shape == (100,)


class TestReproducibility:
    """Test reproducibility with manual seed."""
    
    def test_reproducible_with_seed(self):
        """Test that results are reproducible when seed is set."""
        sampler = FixedLengthBagSampler(bag_length=100, mode='train')
        features = torch.randn(200, 512)
        
        # First run
        torch.manual_seed(42)
        sampled1, _ = sampler.sample(features, num_patches=200)
        
        # Second run with same seed
        torch.manual_seed(42)
        sampled2, _ = sampler.sample(features, num_patches=200)
        
        # Should be identical
        assert torch.allclose(sampled1, sampled2)


class TestStringRepresentation:
    """Test string representation of sampler."""
    
    def test_repr(self):
        """Test __repr__ method."""
        sampler = FixedLengthBagSampler(bag_length=512, mode='train', stride=256)
        repr_str = repr(sampler)
        
        assert 'FixedLengthBagSampler' in repr_str
        assert 'bag_length=512' in repr_str
        assert "mode='train'" in repr_str
        assert 'stride=256' in repr_str
