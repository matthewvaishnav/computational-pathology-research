"""
Gradient sparsification for bandwidth reduction.

Implements top-k sparsification where only the largest k% of gradient
values are transmitted, with the rest set to zero.
"""

from dataclasses import dataclass
from typing import Dict, Tuple, Optional
import torch
import numpy as np


@dataclass
class SparsificationConfig:
    """Configuration for gradient sparsification."""
    
    top_k_percent: float = 10.0  # Percentage of gradients to keep (1%, 5%, 10%)
    threshold_mode: str = "magnitude"  # "magnitude" or "random"
    
    def __post_init__(self):
        """Validate configuration."""
        if not 0.0 < self.top_k_percent <= 100.0:
            raise ValueError(f"top_k_percent must be in (0, 100], got {self.top_k_percent}")
        
        if self.threshold_mode not in ["magnitude", "random"]:
            raise ValueError(f"threshold_mode must be 'magnitude' or 'random', got {self.threshold_mode}")


@dataclass
class SparsifiedGradients:
    """Container for sparsified gradients with metadata."""
    
    values: Dict[str, torch.Tensor]  # Non-zero values
    indices: Dict[str, torch.Tensor]  # Indices of non-zero values
    original_shapes: Dict[str, torch.Size]  # Original tensor shapes
    original_dtypes: Dict[str, torch.dtype]  # Original data types
    top_k_percent: float  # Percentage of values kept
    
    def get_compression_ratio(self) -> float:
        """Calculate compression ratio."""
        original_size = 0
        compressed_size = 0
        
        for name in self.values.keys():
            # Original size (assuming float32)
            original_size += np.prod(self.original_shapes[name]) * 4
            
            # Compressed size: values (float32) + indices (int64)
            compressed_size += len(self.values[name]) * 4  # values
            compressed_size += len(self.indices[name]) * 8  # indices
        
        return original_size / compressed_size if compressed_size > 0 else 1.0
    
    def get_sparsity(self) -> Dict[str, float]:
        """Calculate sparsity for each parameter."""
        sparsity = {}
        
        for name in self.values.keys():
            original_size = np.prod(self.original_shapes[name])
            non_zero_size = len(self.values[name])
            sparsity[name] = 1.0 - (non_zero_size / original_size)
        
        return sparsity


def sparsify_gradients(
    gradients: Dict[str, torch.Tensor],
    config: Optional[SparsificationConfig] = None,
) -> SparsifiedGradients:
    """
    Sparsify gradients by keeping only top-k values.
    
    Args:
        gradients: Dictionary of parameter name -> gradient tensor
        config: Sparsification configuration (default: 10% top-k)
    
    Returns:
        SparsifiedGradients object with sparse values and indices
    
    **Validates: Requirements 8.2, 8.7**
    
    Properties:
    - Invariant: Number of non-zero values ≈ top_k_percent * total_values
    - Invariant: Compressed size < original size
    - Round-trip: densify(sparsify(g)) has same support as sparsified
    """
    if config is None:
        config = SparsificationConfig()
    
    values = {}
    indices = {}
    original_shapes = {}
    original_dtypes = {}
    
    for name, grad in gradients.items():
        if grad is None:
            continue
        
        # Store original metadata
        original_shapes[name] = grad.shape
        original_dtypes[name] = grad.dtype
        
        # Flatten for sparsification
        grad_flat = grad.flatten().float()
        total_elements = len(grad_flat)
        
        # Compute number of elements to keep
        k = max(1, int(total_elements * config.top_k_percent / 100.0))
        
        if config.threshold_mode == "magnitude":
            # Select top-k by absolute magnitude
            abs_grad = torch.abs(grad_flat)
            top_k_values, top_k_indices = torch.topk(abs_grad, k, largest=True)
            
            # Get actual values (with sign)
            selected_values = grad_flat[top_k_indices]
        else:
            # Random sampling
            perm = torch.randperm(total_elements)
            top_k_indices = perm[:k]
            selected_values = grad_flat[top_k_indices]
        
        values[name] = selected_values
        indices[name] = top_k_indices
    
    return SparsifiedGradients(
        values=values,
        indices=indices,
        original_shapes=original_shapes,
        original_dtypes=original_dtypes,
        top_k_percent=config.top_k_percent,
    )


def densify_gradients(
    sparsified: SparsifiedGradients,
) -> Dict[str, torch.Tensor]:
    """
    Densify sparsified gradients for aggregation.
    
    Args:
        sparsified: SparsifiedGradients object from sparsify_gradients
    
    Returns:
        Dictionary of parameter name -> dense gradient tensor
    
    **Validates: Requirements 8.4, 8.7**
    
    Properties:
    - Round-trip: densify(sparsify(g)) preserves top-k values
    - Invariant: Output shapes match original shapes
    - Invariant: Non-selected values are zero
    """
    densified = {}
    
    for name in sparsified.values.keys():
        sparse_values = sparsified.values[name]
        sparse_indices = sparsified.indices[name]
        original_shape = sparsified.original_shapes[name]
        
        # Create zero tensor
        total_elements = np.prod(original_shape)
        dense_flat = torch.zeros(total_elements, dtype=torch.float32)
        
        # Fill in sparse values
        dense_flat[sparse_indices] = sparse_values
        
        # Reshape to original shape
        densified[name] = dense_flat.reshape(original_shape)
    
    return densified


def compute_sparsification_error(
    original: Dict[str, torch.Tensor],
    sparsified: SparsifiedGradients,
) -> Dict[str, float]:
    """
    Compute sparsification error metrics.
    
    Args:
        original: Original gradients
        sparsified: Sparsified gradients
    
    Returns:
        Dictionary of parameter name -> relative error
    """
    densified = densify_gradients(sparsified)
    errors = {}
    
    for name in original.keys():
        if name not in densified:
            continue
        
        orig = original[name]
        dense = densified[name]
        
        # Compute relative L2 error
        error = torch.norm(orig - dense) / (torch.norm(orig) + 1e-8)
        errors[name] = error.item()
    
    return errors


def adaptive_top_k(
    gradients: Dict[str, torch.Tensor],
    target_compression_ratio: float = 10.0,
) -> SparsificationConfig:
    """
    Compute adaptive top-k percentage to achieve target compression ratio.
    
    Args:
        gradients: Dictionary of parameter name -> gradient tensor
        target_compression_ratio: Desired compression ratio (e.g., 10x)
    
    Returns:
        SparsificationConfig with computed top_k_percent
    """
    # Estimate total size
    total_elements = sum(grad.numel() for grad in gradients.values() if grad is not None)
    
    # Compute required sparsity
    # compressed_size = (values + indices) = k * (4 + 8) bytes
    # original_size = total_elements * 4 bytes
    # ratio = original_size / compressed_size
    # ratio = (total_elements * 4) / (k * 12)
    # k = (total_elements * 4) / (ratio * 12)
    
    k = (total_elements * 4) / (target_compression_ratio * 12)
    top_k_percent = (k / total_elements) * 100.0
    
    # Clamp to valid range
    top_k_percent = max(0.1, min(100.0, top_k_percent))
    
    return SparsificationConfig(top_k_percent=top_k_percent)
