"""
Gradient compression module for federated learning.

Provides quantization and sparsification techniques to reduce bandwidth
usage during federated training.
"""

from .compressor import (
    CompressionConfig,
    CompressionMethod,
    GradientCompressor,
    create_compressor,
)
from .quantization import (
    QuantizationConfig,
    dequantize_gradients,
    quantize_gradients,
)
from .sparsification import (
    SparsificationConfig,
    densify_gradients,
    sparsify_gradients,
)

__all__ = [
    "quantize_gradients",
    "dequantize_gradients",
    "QuantizationConfig",
    "sparsify_gradients",
    "densify_gradients",
    "SparsificationConfig",
    "GradientCompressor",
    "CompressionConfig",
    "CompressionMethod",
    "create_compressor",
]
