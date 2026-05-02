"""
Unified gradient compression interface.

Provides a single interface for applying quantization, sparsification,
or mixed compression modes to gradients.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Dict, Optional, Union
import torch

from .quantization import (
    quantize_gradients,
    dequantize_gradients,
    QuantizationConfig,
    QuantizedGradients,
)
from .sparsification import (
    sparsify_gradients,
    densify_gradients,
    SparsificationConfig,
    SparsifiedGradients,
)


class CompressionMethod(Enum):
    """Supported compression methods."""
    
    NONE = "none"
    QUANTIZE_4BIT = "quantize_4bit"
    QUANTIZE_8BIT = "quantize_8bit"
    QUANTIZE_16BIT = "quantize_16bit"
    SPARSIFY_1PCT = "sparsify_1pct"
    SPARSIFY_5PCT = "sparsify_5pct"
    SPARSIFY_10PCT = "sparsify_10pct"
    QUANTIZE_8BIT_SPARSIFY_10PCT = "quantize_8bit_sparsify_10pct"  # Mixed mode


@dataclass
class CompressionConfig:
    """Configuration for gradient compression."""
    
    method: CompressionMethod = CompressionMethod.QUANTIZE_8BIT
    quantization_config: Optional[QuantizationConfig] = None
    sparsification_config: Optional[SparsificationConfig] = None
    
    def __post_init__(self):
        """Initialize default configs based on method."""
        if self.method == CompressionMethod.QUANTIZE_4BIT:
            if self.quantization_config is None:
                self.quantization_config = QuantizationConfig(num_bits=4)
        
        elif self.method == CompressionMethod.QUANTIZE_8BIT:
            if self.quantization_config is None:
                self.quantization_config = QuantizationConfig(num_bits=8)
        
        elif self.method == CompressionMethod.QUANTIZE_16BIT:
            if self.quantization_config is None:
                self.quantization_config = QuantizationConfig(num_bits=16)
        
        elif self.method == CompressionMethod.SPARSIFY_1PCT:
            if self.sparsification_config is None:
                self.sparsification_config = SparsificationConfig(top_k_percent=1.0)
        
        elif self.method == CompressionMethod.SPARSIFY_5PCT:
            if self.sparsification_config is None:
                self.sparsification_config = SparsificationConfig(top_k_percent=5.0)
        
        elif self.method == CompressionMethod.SPARSIFY_10PCT:
            if self.sparsification_config is None:
                self.sparsification_config = SparsificationConfig(top_k_percent=10.0)
        
        elif self.method == CompressionMethod.QUANTIZE_8BIT_SPARSIFY_10PCT:
            if self.quantization_config is None:
                self.quantization_config = QuantizationConfig(num_bits=8)
            if self.sparsification_config is None:
                self.sparsification_config = SparsificationConfig(top_k_percent=10.0)


@dataclass
class CompressedGradients:
    """Container for compressed gradients."""
    
    data: Union[QuantizedGradients, SparsifiedGradients, Dict[str, torch.Tensor]]
    method: CompressionMethod
    compression_ratio: float
    original_size_bytes: int
    compressed_size_bytes: int


class GradientCompressor:
    """
    Unified interface for gradient compression.
    
    Supports quantization, sparsification, and mixed compression modes.
    
    **Validates: Requirements 8.1-8.7**
    """
    
    def __init__(self, config: Optional[CompressionConfig] = None):
        """
        Initialize gradient compressor.
        
        Args:
            config: Compression configuration (default: 8-bit quantization)
        """
        self.config = config or CompressionConfig()
    
    def compress(
        self,
        gradients: Dict[str, torch.Tensor],
    ) -> CompressedGradients:
        """
        Compress gradients using configured method.
        
        Args:
            gradients: Dictionary of parameter name -> gradient tensor
        
        Returns:
            CompressedGradients object with compressed data and metadata
        
        **Validates: Requirements 8.3, 8.6**
        
        Properties:
        - Invariant: Compressed size < original size (for non-NONE methods)
        - Invariant: Compression preserves gradient structure
        """
        # Compute original size
        original_size = self._compute_size(gradients)
        
        if self.config.method == CompressionMethod.NONE:
            return CompressedGradients(
                data=gradients,
                method=self.config.method,
                compression_ratio=1.0,
                original_size_bytes=original_size,
                compressed_size_bytes=original_size,
            )
        
        elif self.config.method in [
            CompressionMethod.QUANTIZE_4BIT,
            CompressionMethod.QUANTIZE_8BIT,
            CompressionMethod.QUANTIZE_16BIT,
        ]:
            # Quantization only
            quantized = quantize_gradients(gradients, self.config.quantization_config)
            compressed_size = self._compute_quantized_size(quantized)
            
            return CompressedGradients(
                data=quantized,
                method=self.config.method,
                compression_ratio=quantized.get_compression_ratio(),
                original_size_bytes=original_size,
                compressed_size_bytes=compressed_size,
            )
        
        elif self.config.method in [
            CompressionMethod.SPARSIFY_1PCT,
            CompressionMethod.SPARSIFY_5PCT,
            CompressionMethod.SPARSIFY_10PCT,
        ]:
            # Sparsification only
            sparsified = sparsify_gradients(gradients, self.config.sparsification_config)
            compressed_size = self._compute_sparsified_size(sparsified)
            
            return CompressedGradients(
                data=sparsified,
                method=self.config.method,
                compression_ratio=sparsified.get_compression_ratio(),
                original_size_bytes=original_size,
                compressed_size_bytes=compressed_size,
            )
        
        elif self.config.method == CompressionMethod.QUANTIZE_8BIT_SPARSIFY_10PCT:
            # Mixed mode: sparsify first, then quantize
            sparsified = sparsify_gradients(gradients, self.config.sparsification_config)
            dense = densify_gradients(sparsified)
            quantized = quantize_gradients(dense, self.config.quantization_config)
            
            compressed_size = self._compute_quantized_size(quantized)
            compression_ratio = original_size / compressed_size if compressed_size > 0 else 1.0
            
            return CompressedGradients(
                data=(sparsified, quantized),  # Store both for decompression
                method=self.config.method,
                compression_ratio=compression_ratio,
                original_size_bytes=original_size,
                compressed_size_bytes=compressed_size,
            )
        
        else:
            raise ValueError(f"Unsupported compression method: {self.config.method}")
    
    def decompress(
        self,
        compressed: CompressedGradients,
    ) -> Dict[str, torch.Tensor]:
        """
        Decompress gradients for aggregation.
        
        Args:
            compressed: CompressedGradients object from compress()
        
        Returns:
            Dictionary of parameter name -> decompressed gradient tensor
        
        **Validates: Requirements 8.4, 8.7**
        
        Properties:
        - Round-trip: decompress(compress(g)) ≈ g within compression error
        - Invariant: Output shapes match original shapes
        """
        if compressed.method == CompressionMethod.NONE:
            return compressed.data
        
        elif compressed.method in [
            CompressionMethod.QUANTIZE_4BIT,
            CompressionMethod.QUANTIZE_8BIT,
            CompressionMethod.QUANTIZE_16BIT,
        ]:
            return dequantize_gradients(compressed.data)
        
        elif compressed.method in [
            CompressionMethod.SPARSIFY_1PCT,
            CompressionMethod.SPARSIFY_5PCT,
            CompressionMethod.SPARSIFY_10PCT,
        ]:
            return densify_gradients(compressed.data)
        
        elif compressed.method == CompressionMethod.QUANTIZE_8BIT_SPARSIFY_10PCT:
            # Mixed mode: dequantize first, then densify
            sparsified, quantized = compressed.data
            dequantized = dequantize_gradients(quantized)
            # Note: In mixed mode, we already have sparse structure from sparsified
            # The quantized data is the dense version, so we just return it
            return dequantized
        
        else:
            raise ValueError(f"Unsupported compression method: {compressed.method}")
    
    def _compute_size(self, gradients: Dict[str, torch.Tensor]) -> int:
        """Compute total size of gradients in bytes."""
        total_size = 0
        for grad in gradients.values():
            if grad is not None:
                total_size += grad.numel() * grad.element_size()
        return total_size
    
    def _compute_quantized_size(self, quantized: QuantizedGradients) -> int:
        """Compute size of quantized gradients in bytes."""
        total_size = 0
        
        for name in quantized.quantized_values.keys():
            # Quantized values
            total_size += quantized.quantized_values[name].numel() * quantized.quantized_values[name].element_size()
            # Scale and zero_point (float32 each)
            total_size += 8
        
        return total_size
    
    def _compute_sparsified_size(self, sparsified: SparsifiedGradients) -> int:
        """Compute size of sparsified gradients in bytes."""
        total_size = 0
        
        for name in sparsified.values.keys():
            # Values (float32)
            total_size += len(sparsified.values[name]) * 4
            # Indices (int64)
            total_size += len(sparsified.indices[name]) * 8
        
        return total_size


def create_compressor(method: str) -> GradientCompressor:
    """
    Factory function to create compressor from method string.
    
    Args:
        method: Compression method name (e.g., "quantize_8bit", "sparsify_10pct")
    
    Returns:
        GradientCompressor instance
    """
    try:
        compression_method = CompressionMethod(method)
    except ValueError:
        raise ValueError(f"Unknown compression method: {method}")
    
    config = CompressionConfig(method=compression_method)
    return GradientCompressor(config)
