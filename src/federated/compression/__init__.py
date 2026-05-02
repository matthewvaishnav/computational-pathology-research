"""
Gradient compression module for federated learning.

Provides quantization and sparsification techniques to reduce bandwidth
usage during federated training.
"""

from .quantization import (
    quantize_gradients,
    dequantize_gradients,
    QuantizationConfig,
)
from .sparsification import (
    sparsify_gradients,
    densify_gradients,
    SparsificationConfig,
)
from .compressor import (
    GradientCompressor,
    CompressionConfig,
    CompressionMethod,
    create_compressor,
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
