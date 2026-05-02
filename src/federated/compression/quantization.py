"""
Gradient quantization for bandwidth reduction.

Implements quantization to 4-bit, 8-bit, and 16-bit precision with
scale/zero-point calculation for accurate reconstruction.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import torch
import numpy as np


@dataclass
class QuantizationConfig:
    """Configuration for gradient quantization."""
    
    num_bits: int = 8  # 4, 8, or 16
    symmetric: bool = False  # Symmetric vs asymmetric quantization
    per_tensor: bool = True  # Per-tensor vs per-channel quantization
    
    def __post_init__(self):
        """Validate configuration."""
        if self.num_bits not in [4, 8, 16]:
            raise ValueError(f"num_bits must be 4, 8, or 16, got {self.num_bits}")


@dataclass
class QuantizedGradients:
    """Container for quantized gradients with metadata."""
    
    quantized_values: Dict[str, torch.Tensor]  # Quantized to uint8/uint16
    scales: Dict[str, torch.Tensor]  # Scale factors
    zero_points: Dict[str, torch.Tensor]  # Zero points
    original_shapes: Dict[str, torch.Size]  # Original tensor shapes
    original_dtypes: Dict[str, torch.dtype]  # Original data types
    num_bits: int  # Bit width used
    
    def get_compression_ratio(self) -> float:
        """Calculate compression ratio."""
        original_size = 0
        compressed_size = 0
        
        for name in self.quantized_values.keys():
            # Original size (assuming float32)
            original_size += np.prod(self.original_shapes[name]) * 4
            
            # Compressed size
            if self.num_bits == 4:
                # 4-bit stored in uint8 (2 values per byte)
                compressed_size += (np.prod(self.quantized_values[name].shape) + 1) // 2
            elif self.num_bits == 8:
                compressed_size += np.prod(self.quantized_values[name].shape)
            else:  # 16-bit (stored in int32 but only uses 16 bits)
                compressed_size += np.prod(self.quantized_values[name].shape) * 2
            
            # Add scale and zero_point overhead (float32 each)
            compressed_size += 8
        
        return original_size / compressed_size if compressed_size > 0 else 1.0


def quantize_gradients(
    gradients: Dict[str, torch.Tensor],
    config: Optional[QuantizationConfig] = None,
) -> QuantizedGradients:
    """
    Quantize gradients to reduce transmission size.
    
    Args:
        gradients: Dictionary of parameter name -> gradient tensor
        config: Quantization configuration (default: 8-bit asymmetric)
    
    Returns:
        QuantizedGradients object with quantized values and metadata
    
    **Validates: Requirements 8.1, 8.7**
    
    Properties:
    - Invariant: Quantized values in range [0, 2^num_bits - 1]
    - Invariant: Compressed size < original size
    - Round-trip: ||dequantize(quantize(g)) - g|| ≤ scale/2
    """
    if config is None:
        config = QuantizationConfig()
    
    quantized_values = {}
    scales = {}
    zero_points = {}
    original_shapes = {}
    original_dtypes = {}
    
    qmin = 0
    qmax = 2 ** config.num_bits - 1
    
    for name, grad in gradients.items():
        if grad is None:
            continue
        
        # Store original metadata
        original_shapes[name] = grad.shape
        original_dtypes[name] = grad.dtype
        
        # Flatten for quantization
        grad_flat = grad.flatten().float()
        
        if config.symmetric:
            # Symmetric quantization: [-max_val, max_val] -> [0, qmax]
            max_val = torch.max(torch.abs(grad_flat))
            scale = (2 * max_val) / qmax if max_val > 0 else torch.tensor(1.0)
            zero_point = torch.tensor(qmax // 2, dtype=torch.float32)
        else:
            # Asymmetric quantization: [min_val, max_val] -> [0, qmax]
            min_val = torch.min(grad_flat)
            max_val = torch.max(grad_flat)
            
            # Compute scale and zero_point
            scale = (max_val - min_val) / qmax if max_val > min_val else torch.tensor(1.0)
            zero_point = qmin - min_val / scale if scale > 0 else torch.tensor(0.0)
        
        # Quantize
        quantized = torch.clamp(
            torch.round(grad_flat / scale + zero_point),
            qmin,
            qmax
        )
        
        # Store as appropriate dtype
        if config.num_bits == 4:
            # Pack 4-bit values into uint8 (2 values per byte)
            quantized_packed = _pack_4bit(quantized)
            quantized_values[name] = quantized_packed
        elif config.num_bits == 8:
            quantized_values[name] = quantized.to(torch.uint8)
        else:  # 16-bit
            # Use int32 to store 16-bit values (torch doesn't have uint16)
            quantized_values[name] = quantized.to(torch.int32)
        
        scales[name] = scale
        zero_points[name] = zero_point
    
    return QuantizedGradients(
        quantized_values=quantized_values,
        scales=scales,
        zero_points=zero_points,
        original_shapes=original_shapes,
        original_dtypes=original_dtypes,
        num_bits=config.num_bits,
    )


def dequantize_gradients(
    quantized: QuantizedGradients,
) -> Dict[str, torch.Tensor]:
    """
    Dequantize gradients for aggregation.
    
    Args:
        quantized: QuantizedGradients object from quantize_gradients
    
    Returns:
        Dictionary of parameter name -> dequantized gradient tensor
    
    **Validates: Requirements 8.4, 8.7**
    
    Properties:
    - Round-trip: dequantize(quantize(g)) ≈ g within quantization error
    - Invariant: Output shapes match original shapes
    """
    dequantized = {}
    
    for name in quantized.quantized_values.keys():
        quantized_vals = quantized.quantized_values[name]
        scale = quantized.scales[name]
        zero_point = quantized.zero_points[name]
        original_shape = quantized.original_shapes[name]
        
        # Unpack if 4-bit
        if quantized.num_bits == 4:
            quantized_vals = _unpack_4bit(quantized_vals, original_shape)
        
        # Dequantize: x = (x_q - zero_point) * scale
        dequantized_flat = (quantized_vals.float() - zero_point) * scale
        
        # Reshape to original shape
        dequantized[name] = dequantized_flat.reshape(original_shape)
    
    return dequantized


def _pack_4bit(values: torch.Tensor) -> torch.Tensor:
    """
    Pack 4-bit values into uint8 (2 values per byte).
    
    Args:
        values: Tensor of values in range [0, 15]
    
    Returns:
        Packed tensor with half the size
    """
    values = values.to(torch.uint8)
    
    # Pad to even length
    if len(values) % 2 == 1:
        values = torch.cat([values, torch.zeros(1, dtype=torch.uint8)])
    
    # Pack pairs: (high << 4) | low
    even_vals = values[::2]
    odd_vals = values[1::2]
    packed = (even_vals << 4) | odd_vals
    
    return packed


def _unpack_4bit(packed: torch.Tensor, original_shape: torch.Size) -> torch.Tensor:
    """
    Unpack 4-bit values from uint8.
    
    Args:
        packed: Packed tensor from _pack_4bit
        original_shape: Original tensor shape before packing
    
    Returns:
        Unpacked tensor with original number of elements
    """
    # Unpack: high = (packed >> 4), low = (packed & 0x0F)
    high = (packed >> 4) & 0x0F
    low = packed & 0x0F
    
    # Interleave
    unpacked = torch.stack([high, low], dim=1).flatten()
    
    # Trim to original size
    original_size = np.prod(original_shape)
    unpacked = unpacked[:original_size]
    
    return unpacked


def compute_quantization_error(
    original: Dict[str, torch.Tensor],
    quantized: QuantizedGradients,
) -> Dict[str, float]:
    """
    Compute quantization error metrics.
    
    Args:
        original: Original gradients
        quantized: Quantized gradients
    
    Returns:
        Dictionary of parameter name -> relative error
    """
    dequantized = dequantize_gradients(quantized)
    errors = {}
    
    for name in original.keys():
        if name not in dequantized:
            continue
        
        orig = original[name]
        dequant = dequantized[name]
        
        # Compute relative L2 error
        error = torch.norm(orig - dequant) / (torch.norm(orig) + 1e-8)
        errors[name] = error.item()
    
    return errors
