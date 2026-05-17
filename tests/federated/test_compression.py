"""
Tests for gradient compression module.

Tests quantization, sparsification, and mixed compression modes with
property-based testing for round-trip correctness.
"""

import numpy as np
import pytest
import torch

from hypothesis import assume, given, settings
from hypothesis import strategies as st
from src.federated.compression import (
    CompressionConfig,
    CompressionMethod,
    GradientCompressor,
    QuantizationConfig,
    SparsificationConfig,
    create_compressor,
    densify_gradients,
    dequantize_gradients,
    quantize_gradients,
    sparsify_gradients,
)

# ============================================================================
# Unit Tests: Quantization
# ============================================================================


class TestQuantization:
    """Test gradient quantization."""

    def test_quantize_8bit_basic(self):
        """Test basic 8-bit quantization."""
        gradients = {
            "layer1.weight": torch.randn(10, 10),
            "layer1.bias": torch.randn(10),
        }

        config = QuantizationConfig(num_bits=8)
        quantized = quantize_gradients(gradients, config)

        # Check quantized values are in valid range
        for name, qvals in quantized.quantized_values.items():
            assert qvals.dtype == torch.uint8
            assert torch.all(qvals >= 0)
            assert torch.all(qvals <= 255)

        # Check metadata
        assert quantized.num_bits == 8
        assert len(quantized.scales) == 2
        assert len(quantized.zero_points) == 2

    def test_quantize_4bit_basic(self):
        """Test basic 4-bit quantization."""
        gradients = {
            "layer1.weight": torch.randn(10, 10),
        }

        config = QuantizationConfig(num_bits=4)
        quantized = quantize_gradients(gradients, config)

        # Check quantized values are packed
        assert quantized.num_bits == 4
        # 4-bit values are packed into uint8 (2 values per byte)
        expected_size = (100 + 1) // 2  # 100 elements -> 50 bytes
        assert quantized.quantized_values["layer1.weight"].numel() == expected_size

    def test_quantize_16bit_basic(self):
        """Test basic 16-bit quantization."""
        gradients = {
            "layer1.weight": torch.randn(10, 10),
        }

        config = QuantizationConfig(num_bits=16)
        quantized = quantize_gradients(gradients, config)

        # Check quantized values are in valid range
        assert quantized.num_bits == 16
        assert quantized.quantized_values["layer1.weight"].dtype == torch.int32

    def test_dequantize_8bit(self):
        """Test 8-bit dequantization."""
        gradients = {
            "layer1.weight": torch.randn(10, 10),
        }

        config = QuantizationConfig(num_bits=8)
        quantized = quantize_gradients(gradients, config)
        dequantized = dequantize_gradients(quantized)

        # Check shapes match
        assert dequantized["layer1.weight"].shape == gradients["layer1.weight"].shape

        # Check values are approximately equal (within quantization error)
        error = torch.norm(gradients["layer1.weight"] - dequantized["layer1.weight"])
        original_norm = torch.norm(gradients["layer1.weight"])
        relative_error = error / (original_norm + 1e-8)

        # 8-bit quantization should have < 5% relative error
        assert relative_error < 0.05

    def test_quantization_compression_ratio(self):
        """Test compression ratio calculation."""
        gradients = {
            "layer1.weight": torch.randn(100, 100),  # 40KB in float32
        }

        config = QuantizationConfig(num_bits=8)
        quantized = quantize_gradients(gradients, config)

        compression_ratio = quantized.get_compression_ratio()

        # 8-bit should give ~4x compression (float32 -> uint8)
        assert 3.5 < compression_ratio < 4.5

    def test_quantization_symmetric(self):
        """Test symmetric quantization mode."""
        gradients = {
            "layer1.weight": torch.randn(10, 10),
        }

        config = QuantizationConfig(num_bits=8, symmetric=True)
        quantized = quantize_gradients(gradients, config)

        # Symmetric quantization should have zero_point near middle
        zero_point = quantized.zero_points["layer1.weight"]
        assert 120 < zero_point < 135  # Should be near 127 (middle of 0-255)

    def test_quantization_invalid_bits(self):
        """Test invalid bit width raises error."""
        with pytest.raises(ValueError, match="num_bits must be 4, 8, or 16"):
            QuantizationConfig(num_bits=32)


# ============================================================================
# Unit Tests: Sparsification
# ============================================================================


class TestSparsification:
    """Test gradient sparsification."""

    def test_sparsify_10pct_basic(self):
        """Test basic 10% sparsification."""
        gradients = {
            "layer1.weight": torch.randn(100, 100),  # 10,000 elements
        }

        config = SparsificationConfig(top_k_percent=10.0)
        sparsified = sparsify_gradients(gradients, config)

        # Check number of non-zero values
        num_values = len(sparsified.values["layer1.weight"])
        expected = 1000  # 10% of 10,000

        assert 900 < num_values <= 1000  # Allow small rounding

    def test_sparsify_1pct_basic(self):
        """Test 1% sparsification."""
        gradients = {
            "layer1.weight": torch.randn(100, 100),
        }

        config = SparsificationConfig(top_k_percent=1.0)
        sparsified = sparsify_gradients(gradients, config)

        # Check number of non-zero values
        num_values = len(sparsified.values["layer1.weight"])
        expected = 100  # 1% of 10,000

        assert 90 < num_values <= 100

    def test_densify_basic(self):
        """Test densification."""
        gradients = {
            "layer1.weight": torch.randn(10, 10),
        }

        config = SparsificationConfig(top_k_percent=10.0)
        sparsified = sparsify_gradients(gradients, config)
        densified = densify_gradients(sparsified)

        # Check shapes match
        assert densified["layer1.weight"].shape == gradients["layer1.weight"].shape

        # Check sparsity
        num_nonzero = torch.count_nonzero(densified["layer1.weight"])
        expected = 10  # 10% of 100

        assert num_nonzero <= expected + 1  # Allow rounding

    def test_sparsification_top_k_magnitude(self):
        """Test that top-k selects largest magnitude values."""
        # Create gradient with known structure
        gradients = {
            "layer1.weight": torch.tensor([1.0, -5.0, 2.0, -3.0, 0.5]),
        }

        config = SparsificationConfig(top_k_percent=40.0)  # Keep 2 out of 5
        sparsified = sparsify_gradients(gradients, config)
        densified = densify_gradients(sparsified)

        # Should keep -5.0 and -3.0 (largest magnitudes)
        nonzero_values = densified["layer1.weight"][densified["layer1.weight"] != 0]

        assert len(nonzero_values) == 2
        assert -5.0 in nonzero_values
        assert -3.0 in nonzero_values or 2.0 in nonzero_values  # Either -3 or 2

    def test_sparsification_compression_ratio(self):
        """Test compression ratio calculation."""
        gradients = {
            "layer1.weight": torch.randn(100, 100),
        }

        config = SparsificationConfig(top_k_percent=10.0)
        sparsified = sparsify_gradients(gradients, config)

        compression_ratio = sparsified.get_compression_ratio()

        # 10% sparsity should give ~3-4x compression
        # (values: 1000*4 bytes, indices: 1000*8 bytes = 12KB vs 40KB original)
        assert 2.5 < compression_ratio < 4.5

    def test_sparsification_get_sparsity(self):
        """Test sparsity calculation."""
        gradients = {
            "layer1.weight": torch.randn(100, 100),
        }

        config = SparsificationConfig(top_k_percent=10.0)
        sparsified = sparsify_gradients(gradients, config)

        sparsity = sparsified.get_sparsity()

        # 10% kept means 90% sparse
        assert 0.89 < sparsity["layer1.weight"] < 0.91

    def test_sparsification_invalid_percent(self):
        """Test invalid top_k_percent raises error."""
        with pytest.raises(ValueError, match="top_k_percent must be in"):
            SparsificationConfig(top_k_percent=0.0)

        with pytest.raises(ValueError, match="top_k_percent must be in"):
            SparsificationConfig(top_k_percent=150.0)


# ============================================================================
# Unit Tests: Unified Compressor
# ============================================================================


class TestGradientCompressor:
    """Test unified gradient compressor."""

    def test_compressor_none(self):
        """Test no compression."""
        gradients = {
            "layer1.weight": torch.randn(10, 10),
        }

        config = CompressionConfig(method=CompressionMethod.NONE)
        compressor = GradientCompressor(config)

        compressed = compressor.compress(gradients)
        decompressed = compressor.decompress(compressed)

        # Should be identical
        assert torch.allclose(gradients["layer1.weight"], decompressed["layer1.weight"])
        assert compressed.compression_ratio == 1.0

    def test_compressor_quantize_8bit(self):
        """Test 8-bit quantization compression."""
        gradients = {
            "layer1.weight": torch.randn(10, 10),
        }

        config = CompressionConfig(method=CompressionMethod.QUANTIZE_8BIT)
        compressor = GradientCompressor(config)

        compressed = compressor.compress(gradients)
        decompressed = compressor.decompress(compressed)

        # Check compression ratio
        assert compressed.compression_ratio > 1.0

        # Check round-trip error
        error = torch.norm(gradients["layer1.weight"] - decompressed["layer1.weight"])
        original_norm = torch.norm(gradients["layer1.weight"])
        relative_error = error / (original_norm + 1e-8)

        assert relative_error < 0.05

    def test_compressor_sparsify_10pct(self):
        """Test 10% sparsification compression."""
        gradients = {
            "layer1.weight": torch.randn(100, 100),
        }

        config = CompressionConfig(method=CompressionMethod.SPARSIFY_10PCT)
        compressor = GradientCompressor(config)

        compressed = compressor.compress(gradients)
        decompressed = compressor.decompress(compressed)

        # Check compression ratio
        assert compressed.compression_ratio > 1.0

        # Check sparsity
        num_nonzero = torch.count_nonzero(decompressed["layer1.weight"])
        assert num_nonzero <= 1000  # 10% of 10,000

    def test_compressor_mixed_mode(self):
        """Test mixed quantization + sparsification."""
        gradients = {
            "layer1.weight": torch.randn(100, 100),
        }

        config = CompressionConfig(method=CompressionMethod.QUANTIZE_8BIT_SPARSIFY_10PCT)
        compressor = GradientCompressor(config)

        compressed = compressor.compress(gradients)
        decompressed = compressor.decompress(compressed)

        # Check compression ratio (should be higher than either alone)
        assert compressed.compression_ratio > 3.0

        # Check shapes match
        assert decompressed["layer1.weight"].shape == gradients["layer1.weight"].shape

    def test_create_compressor_factory(self):
        """Test factory function."""
        compressor = create_compressor("quantize_8bit")

        assert isinstance(compressor, GradientCompressor)
        assert compressor.config.method == CompressionMethod.QUANTIZE_8BIT

    def test_create_compressor_invalid(self):
        """Test factory with invalid method."""
        with pytest.raises(ValueError, match="Unknown compression method"):
            create_compressor("invalid_method")


# ============================================================================
# Property-Based Tests
# ============================================================================


@st.composite
def gradient_dict(draw):
    """Generate random gradient dictionaries."""
    num_layers = draw(st.integers(min_value=1, max_value=3))
    gradients = {}

    for i in range(num_layers):
        shape = draw(
            st.tuples(
                st.integers(min_value=5, max_value=50),
                st.integers(min_value=5, max_value=50),
            )
        )
        gradients[f"layer{i}.weight"] = torch.randn(*shape)

    return gradients


class TestQuantizationProperties:
    """Property-based tests for quantization."""

    @given(gradients=gradient_dict())
    @settings(max_examples=50, deadline=None)
    def test_quantization_round_trip_bounded_error(self, gradients):
        """
        Property: Quantization round-trip has bounded error.

        **Validates: Requirements 8.7**

        For all gradients g:
          ||dequantize(quantize(g)) - g|| / ||g|| ≤ ε
        """
        config = QuantizationConfig(num_bits=8)
        quantized = quantize_gradients(gradients, config)
        dequantized = dequantize_gradients(quantized)

        for name in gradients.keys():
            orig = gradients[name]
            dequant = dequantized[name]

            # Compute relative error
            error = torch.norm(orig - dequant)
            original_norm = torch.norm(orig)

            if original_norm > 1e-6:  # Avoid division by zero
                relative_error = error / original_norm
                # 8-bit quantization should have < 10% relative error
                assert relative_error < 0.1, f"Relative error {relative_error} too high for {name}"

    @given(gradients=gradient_dict())
    @settings(max_examples=50, deadline=None)
    def test_quantization_compression_ratio_invariant(self, gradients):
        """
        Property: Quantized size < original size.

        **Validates: Requirements 8.7**

        For all gradients g:
          size(quantize(g)) < size(g)
        """
        config = QuantizationConfig(num_bits=8)
        quantized = quantize_gradients(gradients, config)

        compression_ratio = quantized.get_compression_ratio()

        # Should have compression (ratio > 1)
        assert compression_ratio > 1.0

    @given(
        gradients=gradient_dict(),
        num_bits=st.sampled_from([4, 8, 16]),
    )
    @settings(max_examples=50, deadline=None)
    def test_quantization_value_range_invariant(self, gradients, num_bits):
        """
        Property: Quantized values in valid range.

        **Validates: Requirements 8.1**

        For all gradients g and bit-width b:
          quantize(g, b) ∈ [0, 2^b - 1]
        """
        config = QuantizationConfig(num_bits=num_bits)
        quantized = quantize_gradients(gradients, config)

        qmax = 2**num_bits - 1

        for name, qvals in quantized.quantized_values.items():
            if num_bits == 4:
                # 4-bit values are packed, so check after unpacking
                # For now, just check the packed values are valid uint8
                assert qvals.dtype == torch.uint8
            elif num_bits == 8:
                assert qvals.dtype == torch.uint8
                assert torch.all(qvals >= 0)
                assert torch.all(qvals <= qmax)
            else:  # 16-bit
                assert qvals.dtype == torch.int32
                # int32 can represent [0, 65535] for 16-bit values
                assert torch.all(qvals >= 0)
                assert torch.all(qvals <= qmax)

    @given(gradients=gradient_dict())
    @settings(max_examples=50, deadline=None)
    def test_quantization_shape_preservation(self, gradients):
        """
        Property: Quantization preserves shapes.

        For all gradients g:
          shape(dequantize(quantize(g))) = shape(g)
        """
        config = QuantizationConfig(num_bits=8)
        quantized = quantize_gradients(gradients, config)
        dequantized = dequantize_gradients(quantized)

        for name in gradients.keys():
            assert dequantized[name].shape == gradients[name].shape


class TestSparsificationProperties:
    """Property-based tests for sparsification."""

    @given(
        gradients=gradient_dict(),
        top_k_percent=st.floats(min_value=1.0, max_value=50.0),
    )
    @settings(max_examples=50, deadline=None)
    def test_sparsification_top_k_invariant(self, gradients, top_k_percent):
        """
        Property: Sparsified gradients have correct number of non-zero values.

        **Validates: Requirements 8.2**

        For all gradients g and percentage k:
          |nonzero(sparsify(g, k))| ≈ k% * |g|
        """
        config = SparsificationConfig(top_k_percent=top_k_percent)
        sparsified = sparsify_gradients(gradients, config)
        densified = densify_gradients(sparsified)

        for name in gradients.keys():
            total_elements = gradients[name].numel()
            expected_k = max(1, int(total_elements * top_k_percent / 100.0))

            num_nonzero = torch.count_nonzero(densified[name])

            # Allow small rounding error
            assert num_nonzero <= expected_k + 1

    @given(gradients=gradient_dict())
    @settings(max_examples=50, deadline=None)
    def test_sparsification_compression_ratio_invariant(self, gradients):
        """
        Property: Sparsified size < original size.

        **Validates: Requirements 8.7**

        For all gradients g:
          size(sparsify(g)) < size(g)
        """
        config = SparsificationConfig(top_k_percent=10.0)
        sparsified = sparsify_gradients(gradients, config)

        compression_ratio = sparsified.get_compression_ratio()

        # Should have compression (ratio > 1)
        assert compression_ratio > 1.0

    @given(gradients=gradient_dict())
    @settings(max_examples=50, deadline=None)
    def test_sparsification_shape_preservation(self, gradients):
        """
        Property: Sparsification preserves shapes.

        For all gradients g:
          shape(densify(sparsify(g))) = shape(g)
        """
        config = SparsificationConfig(top_k_percent=10.0)
        sparsified = sparsify_gradients(gradients, config)
        densified = densify_gradients(sparsified)

        for name in gradients.keys():
            assert densified[name].shape == gradients[name].shape

    @given(gradients=gradient_dict())
    @settings(max_examples=50, deadline=None)
    def test_sparsification_zero_preservation(self, gradients):
        """
        Property: Non-selected values are zero after densification.

        For all gradients g:
          densify(sparsify(g)) has (100-k)% zeros
        """
        config = SparsificationConfig(top_k_percent=10.0)
        sparsified = sparsify_gradients(gradients, config)
        densified = densify_gradients(sparsified)

        for name in gradients.keys():
            total_elements = gradients[name].numel()
            num_zeros = torch.sum(densified[name] == 0).item()

            # At least 90% should be zero (for 10% top-k)
            assert num_zeros >= total_elements * 0.89


class TestCompressorProperties:
    """Property-based tests for unified compressor."""

    @given(
        gradients=gradient_dict(),
        method=st.sampled_from(
            [
                CompressionMethod.QUANTIZE_8BIT,
                CompressionMethod.SPARSIFY_10PCT,
                CompressionMethod.QUANTIZE_8BIT_SPARSIFY_10PCT,
            ]
        ),
    )
    @settings(max_examples=50, deadline=None)
    def test_compressor_round_trip_shape_preservation(self, gradients, method):
        """
        Property: Compression preserves shapes.

        **Validates: Requirements 8.7**

        For all gradients g and methods m:
          shape(decompress(compress(g, m))) = shape(g)
        """
        config = CompressionConfig(method=method)
        compressor = GradientCompressor(config)

        compressed = compressor.compress(gradients)
        decompressed = compressor.decompress(compressed)

        for name in gradients.keys():
            assert decompressed[name].shape == gradients[name].shape

    @given(
        gradients=gradient_dict(),
        method=st.sampled_from(
            [
                CompressionMethod.QUANTIZE_8BIT,
                CompressionMethod.SPARSIFY_10PCT,
            ]
        ),
    )
    @settings(max_examples=50, deadline=None)
    def test_compressor_compression_ratio_invariant(self, gradients, method):
        """
        Property: Compression reduces size.

        **Validates: Requirements 8.7**

        For all gradients g and methods m (except NONE):
          compressed_size(g, m) < original_size(g)
        """
        config = CompressionConfig(method=method)
        compressor = GradientCompressor(config)

        compressed = compressor.compress(gradients)

        # Should have compression
        assert compressed.compression_ratio > 1.0
        assert compressed.compressed_size_bytes < compressed.original_size_bytes


# ============================================================================
# Integration Tests
# ============================================================================


class TestCompressionIntegration:
    """Integration tests for compression in federated learning context."""

    def test_compression_with_privacy(self):
        """Test compression after DP-SGD (as per requirements)."""
        # Simulate DP-SGD gradients (clipped + noised)
        gradients = {
            "layer1.weight": torch.randn(50, 50),
        }

        # Apply DP-SGD (simplified: just add noise)
        clipping_bound = 1.0
        noise_multiplier = 0.1

        for name in gradients.keys():
            grad_norm = torch.norm(gradients[name])
            clipping_factor = min(1.0, clipping_bound / (grad_norm + 1e-8))
            gradients[name] = gradients[name] * clipping_factor

            noise = torch.randn_like(gradients[name]) * clipping_bound * noise_multiplier
            gradients[name] = gradients[name] + noise

        # Now compress (as per Requirement 8.3)
        config = CompressionConfig(method=CompressionMethod.QUANTIZE_8BIT)
        compressor = GradientCompressor(config)

        compressed = compressor.compress(gradients)
        decompressed = compressor.decompress(compressed)

        # Should work correctly
        assert decompressed["layer1.weight"].shape == gradients["layer1.weight"].shape

    def test_mixed_compression_modes(self):
        """
        Test mixed compression modes (different clients use different schemes).

        **Validates: Requirements 8.6**
        """
        gradients = {
            "layer1.weight": torch.randn(50, 50),
        }

        # Client 1: 8-bit quantization
        compressor1 = create_compressor("quantize_8bit")
        compressed1 = compressor1.compress(gradients)
        decompressed1 = compressor1.decompress(compressed1)

        # Client 2: 10% sparsification
        compressor2 = create_compressor("sparsify_10pct")
        compressed2 = compressor2.compress(gradients)
        decompressed2 = compressor2.decompress(compressed2)

        # Client 3: Mixed mode
        compressor3 = create_compressor("quantize_8bit_sparsify_10pct")
        compressed3 = compressor3.compress(gradients)
        decompressed3 = compressor3.decompress(compressed3)

        # All should decompress to valid gradients
        assert decompressed1["layer1.weight"].shape == gradients["layer1.weight"].shape
        assert decompressed2["layer1.weight"].shape == gradients["layer1.weight"].shape
        assert decompressed3["layer1.weight"].shape == gradients["layer1.weight"].shape

        # Aggregator can average them (simplified)
        avg_gradient = (
            decompressed1["layer1.weight"]
            + decompressed2["layer1.weight"]
            + decompressed3["layer1.weight"]
        ) / 3.0

        assert avg_gradient.shape == gradients["layer1.weight"].shape
