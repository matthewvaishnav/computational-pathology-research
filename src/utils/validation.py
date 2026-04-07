"""
Input validation module for computational pathology framework.

This module provides comprehensive validation functions for tensor operations,
model inputs, and multimodal batch data. Validation can be disabled for production
use via environment variable DISABLE_VALIDATION=1.

Example:
    >>> import torch
    >>> from src.utils.validation import validate_tensor_shape, validate_wsi_features
    >>> 
    >>> # Validate tensor shape
    >>> tensor = torch.randn(16, 100, 1024)
    >>> validate_tensor_shape(tensor, (16, 100, 1024), "wsi_features")
    >>> 
    >>> # Validate WSI features
    >>> validate_wsi_features(tensor)
    >>> 
    >>> # Use decorator for automatic validation
    >>> @validate_inputs
    >>> def forward(self, batch):
    ...     return self.model(batch)
"""

import functools
import os
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch

# Check if validation is disabled
_VALIDATION_ENABLED = os.environ.get("DISABLE_VALIDATION", "0") != "1"


def is_validation_enabled() -> bool:
    """Check if validation is currently enabled."""
    return _VALIDATION_ENABLED


def set_validation_enabled(enabled: bool) -> None:
    """
    Enable or disable validation globally.
    
    Args:
        enabled: Whether to enable validation
        
    Example:
        >>> set_validation_enabled(False)  # Disable for production
        >>> set_validation_enabled(True)   # Re-enable for debugging
    """
    global _VALIDATION_ENABLED
    _VALIDATION_ENABLED = enabled


class ValidationError(Exception):
    """Custom exception for validation errors with helpful messages."""
    pass


# ============================================================================
# Core Tensor Validation Functions
# ============================================================================


def validate_tensor_shape(
    tensor: torch.Tensor,
    expected_shape: Tuple[Optional[int], ...],
    name: str = "tensor"
) -> None:
    """
    Validate that a tensor has the expected shape.
    
    Args:
        tensor: Input tensor to validate
        expected_shape: Expected shape tuple. Use None for dimensions that can vary.
                       Example: (None, 100, 1024) allows variable batch size
        name: Name of the tensor for error messages
        
    Raises:
        ValidationError: If shape doesn't match expected shape
        
    Example:
        >>> tensor = torch.randn(16, 100, 1024)
        >>> validate_tensor_shape(tensor, (None, 100, 1024), "wsi_features")
        >>> # Passes - batch size can vary
        >>> 
        >>> validate_tensor_shape(tensor, (16, 100, 512), "wsi_features")
        >>> # Raises ValidationError - feature dim mismatch
    """
    if not _VALIDATION_ENABLED:
        return
        
    if not isinstance(tensor, torch.Tensor):
        raise ValidationError(
            f"Expected {name} to be a torch.Tensor, but got {type(tensor).__name__}.\n"
            f"Suggestion: Ensure input is converted to tensor using torch.tensor() or torch.from_numpy()"
        )
    
    actual_shape = tuple(tensor.shape)
    
    if len(actual_shape) != len(expected_shape):
        raise ValidationError(
            f"Shape mismatch for {name}:\n"
            f"  Expected: {len(expected_shape)} dimensions {expected_shape}\n"
            f"  Received: {len(actual_shape)} dimensions {actual_shape}\n"
            f"Suggestion: Check that your input has the correct number of dimensions"
        )
    
    for i, (actual, expected) in enumerate(zip(actual_shape, expected_shape)):
        if expected is not None and actual != expected:
            raise ValidationError(
                f"Shape mismatch for {name} at dimension {i}:\n"
                f"  Expected: {expected_shape}\n"
                f"  Received: {actual_shape}\n"
                f"  Dimension {i}: expected {expected}, got {actual}\n"
                f"Suggestion: Verify the input preprocessing and data loading pipeline"
            )


def validate_tensor_range(
    tensor: torch.Tensor,
    min_val: Optional[float] = None,
    max_val: Optional[float] = None,
    name: str = "tensor"
) -> None:
    """
    Validate that tensor values are within expected range.
    
    Args:
        tensor: Input tensor to validate
        min_val: Minimum allowed value (inclusive), None to skip
        max_val: Maximum allowed value (inclusive), None to skip
        name: Name of the tensor for error messages
        
    Raises:
        ValidationError: If values are outside expected range
        
    Example:
        >>> tensor = torch.randn(16, 1024) * 0.1  # Small values
        >>> validate_tensor_range(tensor, -1.0, 1.0, "normalized_features")
        >>> # Passes
        >>> 
        >>> tensor = torch.randn(16, 1024) * 10  # Large values
        >>> validate_tensor_range(tensor, -1.0, 1.0, "normalized_features")
        >>> # Raises ValidationError
    """
    if not _VALIDATION_ENABLED:
        return
        
    if not isinstance(tensor, torch.Tensor):
        raise ValidationError(
            f"Expected {name} to be a torch.Tensor, but got {type(tensor).__name__}"
        )
    
    actual_min = tensor.min().item()
    actual_max = tensor.max().item()
    
    if min_val is not None and actual_min < min_val:
        raise ValidationError(
            f"Value range error for {name}:\n"
            f"  Expected minimum: {min_val}\n"
            f"  Actual minimum: {actual_min:.6f}\n"
            f"  Actual range: [{actual_min:.6f}, {actual_max:.6f}]\n"
            f"Suggestion: Check normalization or scaling of input data"
        )
    
    if max_val is not None and actual_max > max_val:
        raise ValidationError(
            f"Value range error for {name}:\n"
            f"  Expected maximum: {max_val}\n"
            f"  Actual maximum: {actual_max:.6f}\n"
            f"  Actual range: [{actual_min:.6f}, {actual_max:.6f}]\n"
            f"Suggestion: Check normalization or scaling of input data"
        )


def validate_no_nan_inf(tensor: torch.Tensor, name: str = "tensor") -> None:
    """
    Validate that tensor contains no NaN or Inf values.
    
    Args:
        tensor: Input tensor to validate
        name: Name of the tensor for error messages
        
    Raises:
        ValidationError: If tensor contains NaN or Inf values
        
    Example:
        >>> tensor = torch.randn(16, 1024)
        >>> validate_no_nan_inf(tensor, "features")
        >>> # Passes
        >>> 
        >>> tensor[0, 0] = float('nan')
        >>> validate_no_nan_inf(tensor, "features")
        >>> # Raises ValidationError
    """
    if not _VALIDATION_ENABLED:
        return
        
    if not isinstance(tensor, torch.Tensor):
        raise ValidationError(
            f"Expected {name} to be a torch.Tensor, but got {type(tensor).__name__}"
        )
    
    has_nan = torch.isnan(tensor).any().item()
    has_inf = torch.isinf(tensor).any().item()
    
    if has_nan or has_inf:
        nan_count = torch.isnan(tensor).sum().item()
        inf_count = torch.isinf(tensor).sum().item()
        total_elements = tensor.numel()
        
        error_msg = f"Invalid values detected in {name}:\n"
        if has_nan:
            error_msg += f"  NaN values: {nan_count} / {total_elements} ({100*nan_count/total_elements:.2f}%)\n"
        if has_inf:
            error_msg += f"  Inf values: {inf_count} / {total_elements} ({100*inf_count/total_elements:.2f}%)\n"
        error_msg += (
            f"Suggestions:\n"
            f"  - Check for division by zero in preprocessing\n"
            f"  - Verify data loading pipeline for corrupted data\n"
            f"  - Add gradient clipping if this occurs during training\n"
            f"  - Check for numerical instability in model computations"
        )
        raise ValidationError(error_msg)


def validate_batch_size(
    tensor: torch.Tensor,
    expected_batch_size: int,
    name: str = "tensor"
) -> None:
    """
    Validate that tensor has expected batch size (first dimension).
    
    Args:
        tensor: Input tensor to validate
        expected_batch_size: Expected batch size
        name: Name of the tensor for error messages
        
    Raises:
        ValidationError: If batch size doesn't match
        
    Example:
        >>> tensor = torch.randn(16, 1024)
        >>> validate_batch_size(tensor, 16, "features")
        >>> # Passes
        >>> 
        >>> validate_batch_size(tensor, 32, "features")
        >>> # Raises ValidationError
    """
    if not _VALIDATION_ENABLED:
        return
        
    if not isinstance(tensor, torch.Tensor):
        raise ValidationError(
            f"Expected {name} to be a torch.Tensor, but got {type(tensor).__name__}"
        )
    
    if len(tensor.shape) == 0:
        raise ValidationError(
            f"Cannot validate batch size for {name}: tensor is scalar (0 dimensions)\n"
            f"Suggestion: Ensure tensor has at least 1 dimension"
        )
    
    actual_batch_size = tensor.shape[0]
    if actual_batch_size != expected_batch_size:
        raise ValidationError(
            f"Batch size mismatch for {name}:\n"
            f"  Expected: {expected_batch_size}\n"
            f"  Received: {actual_batch_size}\n"
            f"  Tensor shape: {tuple(tensor.shape)}\n"
            f"Suggestion: Check DataLoader batch_size and collate_fn configuration"
        )


# ============================================================================
# Modality-Specific Validation Functions
# ============================================================================


def validate_wsi_features(
    features: torch.Tensor,
    expected_feature_dim: int = 1024,
    name: str = "wsi_features"
) -> None:
    """
    Validate WSI (Whole Slide Image) features tensor.
    
    Expected shape: [batch_size, num_patches, feature_dim]
    
    Args:
        features: WSI features tensor
        expected_feature_dim: Expected feature dimension (default: 1024)
        name: Name for error messages
        
    Raises:
        ValidationError: If features don't match expected format
        
    Example:
        >>> wsi_features = torch.randn(16, 100, 1024)
        >>> validate_wsi_features(wsi_features)
        >>> # Passes
        >>> 
        >>> wsi_features = torch.randn(16, 100, 512)
        >>> validate_wsi_features(wsi_features)
        >>> # Raises ValidationError - wrong feature dim
    """
    if not _VALIDATION_ENABLED:
        return
    
    # Check it's a tensor
    if not isinstance(features, torch.Tensor):
        raise ValidationError(
            f"Expected {name} to be a torch.Tensor, but got {type(features).__name__}"
        )
    
    # Check number of dimensions
    if len(features.shape) != 3:
        raise ValidationError(
            f"Invalid shape for {name}:\n"
            f"  Expected: 3 dimensions [batch_size, num_patches, feature_dim]\n"
            f"  Received: {len(features.shape)} dimensions {tuple(features.shape)}\n"
            f"Suggestion: WSI features should be 3D tensor with shape [B, N, D]"
        )
    
    batch_size, num_patches, feature_dim = features.shape
    
    # Check feature dimension
    if feature_dim != expected_feature_dim:
        raise ValidationError(
            f"Feature dimension mismatch for {name}:\n"
            f"  Expected: [batch_size, num_patches, {expected_feature_dim}]\n"
            f"  Received: {tuple(features.shape)}\n"
            f"  Feature dim: expected {expected_feature_dim}, got {feature_dim}\n"
            f"Suggestion: Verify feature extractor output dimension (e.g., ResNet, ViT)"
        )
    
    # Check for valid values
    validate_no_nan_inf(features, name)
    
    # Check reasonable number of patches
    if num_patches == 0:
        raise ValidationError(
            f"Invalid number of patches for {name}:\n"
            f"  Received: {num_patches} patches\n"
            f"Suggestion: WSI should have at least 1 patch"
        )


def validate_genomic_features(
    features: torch.Tensor,
    expected_feature_dim: int = 2000,
    name: str = "genomic_features"
) -> None:
    """
    Validate genomic features tensor.
    
    Expected shape: [batch_size, num_genes]
    
    Args:
        features: Genomic features tensor
        expected_feature_dim: Expected number of genes (default: 2000)
        name: Name for error messages
        
    Raises:
        ValidationError: If features don't match expected format
        
    Example:
        >>> genomic = torch.randn(16, 2000)
        >>> validate_genomic_features(genomic)
        >>> # Passes
        >>> 
        >>> genomic = torch.randn(16, 1000)
        >>> validate_genomic_features(genomic)
        >>> # Raises ValidationError - wrong gene count
    """
    if not _VALIDATION_ENABLED:
        return
    
    if not isinstance(features, torch.Tensor):
        raise ValidationError(
            f"Expected {name} to be a torch.Tensor, but got {type(features).__name__}"
        )
    
    # Check number of dimensions
    if len(features.shape) != 2:
        raise ValidationError(
            f"Invalid shape for {name}:\n"
            f"  Expected: 2 dimensions [batch_size, num_genes]\n"
            f"  Received: {len(features.shape)} dimensions {tuple(features.shape)}\n"
            f"Suggestion: Genomic features should be 2D tensor with shape [B, G]"
        )
    
    batch_size, num_genes = features.shape
    
    # Check feature dimension
    if num_genes != expected_feature_dim:
        raise ValidationError(
            f"Feature dimension mismatch for {name}:\n"
            f"  Expected: [batch_size, {expected_feature_dim}]\n"
            f"  Received: {tuple(features.shape)}\n"
            f"  Num genes: expected {expected_feature_dim}, got {num_genes}\n"
            f"Suggestion: Verify genomic data preprocessing and feature selection"
        )
    
    # Check for valid values
    validate_no_nan_inf(features, name)


def validate_clinical_text(
    tokens: torch.Tensor,
    max_seq_length: Optional[int] = None,
    name: str = "clinical_text"
) -> None:
    """
    Validate clinical text token IDs tensor.
    
    Expected shape: [batch_size, seq_length]
    
    Args:
        tokens: Clinical text token IDs tensor
        max_seq_length: Maximum sequence length (optional)
        name: Name for error messages
        
    Raises:
        ValidationError: If tokens don't match expected format
        
    Example:
        >>> tokens = torch.randint(0, 30000, (16, 128))
        >>> validate_clinical_text(tokens, max_seq_length=512)
        >>> # Passes
        >>> 
        >>> tokens = torch.randint(0, 30000, (16, 1000))
        >>> validate_clinical_text(tokens, max_seq_length=512)
        >>> # Raises ValidationError - sequence too long
    """
    if not _VALIDATION_ENABLED:
        return
    
    if not isinstance(tokens, torch.Tensor):
        raise ValidationError(
            f"Expected {name} to be a torch.Tensor, but got {type(tokens).__name__}"
        )
    
    # Check number of dimensions
    if len(tokens.shape) != 2:
        raise ValidationError(
            f"Invalid shape for {name}:\n"
            f"  Expected: 2 dimensions [batch_size, seq_length]\n"
            f"  Received: {len(tokens.shape)} dimensions {tuple(tokens.shape)}\n"
            f"Suggestion: Clinical text should be 2D tensor with shape [B, L]"
        )
    
    batch_size, seq_length = tokens.shape
    
    # Check sequence length if specified
    if max_seq_length is not None and seq_length > max_seq_length:
        raise ValidationError(
            f"Sequence length exceeds maximum for {name}:\n"
            f"  Maximum allowed: {max_seq_length}\n"
            f"  Received: {seq_length}\n"
            f"  Tensor shape: {tuple(tokens.shape)}\n"
            f"Suggestion: Truncate sequences during preprocessing or increase max_seq_length"
        )
    
    # Check that tokens are integers
    if not tokens.dtype in [torch.int32, torch.int64, torch.long]:
        raise ValidationError(
            f"Invalid dtype for {name}:\n"
            f"  Expected: integer type (torch.long, torch.int32, torch.int64)\n"
            f"  Received: {tokens.dtype}\n"
            f"Suggestion: Token IDs should be integers, use .long() to convert"
        )
    
    # Check for negative token IDs
    if (tokens < 0).any():
        num_negative = (tokens < 0).sum().item()
        raise ValidationError(
            f"Invalid token IDs in {name}:\n"
            f"  Found {num_negative} negative token IDs\n"
            f"  Token range: [{tokens.min().item()}, {tokens.max().item()}]\n"
            f"Suggestion: Token IDs should be non-negative integers"
        )


def validate_multimodal_batch(
    batch: Dict[str, Any],
    expected_batch_size: Optional[int] = None,
    require_all_modalities: bool = False
) -> None:
    """
    Validate complete multimodal batch dictionary.
    
    Expected keys:
        - 'wsi_features': Optional[Tensor] [batch_size, num_patches, 1024]
        - 'genomic': Optional[Tensor] [batch_size, 2000]
        - 'clinical_text': Optional[Tensor] [batch_size, seq_len]
        - 'labels': Tensor [batch_size]
    
    Args:
        batch: Batch dictionary from dataloader
        expected_batch_size: Expected batch size (optional)
        require_all_modalities: If True, all modalities must be present
        
    Raises:
        ValidationError: If batch doesn't match expected format
        
    Example:
        >>> batch = {
        ...     'wsi_features': torch.randn(16, 100, 1024),
        ...     'genomic': torch.randn(16, 2000),
        ...     'clinical_text': torch.randint(0, 30000, (16, 128)),
        ...     'labels': torch.randint(0, 2, (16,))
        ... }
        >>> validate_multimodal_batch(batch)
        >>> # Passes
    """
    if not _VALIDATION_ENABLED:
        return
    
    if not isinstance(batch, dict):
        raise ValidationError(
            f"Expected batch to be a dictionary, but got {type(batch).__name__}\n"
            f"Suggestion: Ensure collate_fn returns a dictionary"
        )
    
    # Determine batch size from first available modality
    batch_size = None
    for key in ['wsi_features', 'genomic', 'clinical_text', 'labels']:
        if key in batch and batch[key] is not None:
            if isinstance(batch[key], torch.Tensor):
                batch_size = batch[key].shape[0]
                break
            elif isinstance(batch[key], list) and len(batch[key]) > 0:
                batch_size = len(batch[key])
                break
    
    if batch_size is None:
        raise ValidationError(
            "Cannot determine batch size: no valid modality data found\n"
            f"Available keys: {list(batch.keys())}\n"
            f"Suggestion: Ensure at least one modality is present in batch"
        )
    
    # Check expected batch size if provided
    if expected_batch_size is not None and batch_size != expected_batch_size:
        raise ValidationError(
            f"Batch size mismatch:\n"
            f"  Expected: {expected_batch_size}\n"
            f"  Received: {batch_size}\n"
            f"Suggestion: Check DataLoader configuration"
        )
    
    # Validate WSI features
    if 'wsi_features' in batch and batch['wsi_features'] is not None:
        wsi = batch['wsi_features']
        if isinstance(wsi, list):
            # Variable length patches - validate each sample
            if len(wsi) != batch_size:
                raise ValidationError(
                    f"WSI features list length mismatch:\n"
                    f"  Expected: {batch_size} samples\n"
                    f"  Received: {len(wsi)} samples"
                )
            for i, sample in enumerate(wsi):
                if sample is not None:
                    if len(sample.shape) != 2:
                        raise ValidationError(
                            f"Invalid WSI features shape for sample {i}:\n"
                            f"  Expected: 2D [num_patches, feature_dim]\n"
                            f"  Received: {tuple(sample.shape)}"
                        )
                    if sample.shape[1] != 1024:
                        raise ValidationError(
                            f"Invalid WSI feature dimension for sample {i}:\n"
                            f"  Expected: 1024\n"
                            f"  Received: {sample.shape[1]}"
                        )
        else:
            validate_wsi_features(wsi)
            validate_batch_size(wsi, batch_size, "wsi_features")
    elif require_all_modalities:
        raise ValidationError(
            "Missing required modality: wsi_features\n"
            f"Available keys: {list(batch.keys())}"
        )
    
    # Validate genomic features
    if 'genomic' in batch and batch['genomic'] is not None:
        validate_genomic_features(batch['genomic'])
        validate_batch_size(batch['genomic'], batch_size, "genomic")
    elif require_all_modalities:
        raise ValidationError(
            "Missing required modality: genomic\n"
            f"Available keys: {list(batch.keys())}"
        )
    
    # Validate clinical text
    if 'clinical_text' in batch and batch['clinical_text'] is not None:
        validate_clinical_text(batch['clinical_text'])
        validate_batch_size(batch['clinical_text'], batch_size, "clinical_text")
    elif require_all_modalities:
        raise ValidationError(
            "Missing required modality: clinical_text\n"
            f"Available keys: {list(batch.keys())}"
        )
    
    # Validate labels if present
    if 'labels' in batch:
        labels = batch['labels']
        if not isinstance(labels, torch.Tensor):
            raise ValidationError(
                f"Expected labels to be a torch.Tensor, but got {type(labels).__name__}"
            )
        validate_batch_size(labels, batch_size, "labels")


# ============================================================================
# Decorator for Automatic Validation
# ============================================================================


def validate_inputs(func: Callable) -> Callable:
    """
    Decorator for automatic input validation on model forward methods.
    
    Validates batch dictionary before passing to the wrapped function.
    Can be disabled by setting DISABLE_VALIDATION=1 environment variable.
    
    Args:
        func: Function to wrap (typically a forward method)
        
    Returns:
        Wrapped function with input validation
        
    Example:
        >>> class MyModel(nn.Module):
        ...     @validate_inputs
        ...     def forward(self, batch):
        ...         # batch is automatically validated
        ...         return self.process(batch)
        >>> 
        >>> model = MyModel()
        >>> output = model(batch)  # Validation happens automatically
    """
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        if not _VALIDATION_ENABLED:
            return func(*args, **kwargs)
        
        # Find batch argument (first arg after self, or in kwargs)
        batch = None
        if len(args) > 1:
            batch = args[1]
        elif 'batch' in kwargs:
            batch = kwargs['batch']
        
        # Validate if batch is a dict
        if batch is not None and isinstance(batch, dict):
            try:
                validate_multimodal_batch(batch, require_all_modalities=False)
            except ValidationError as e:
                # Add context about where validation failed
                raise ValidationError(
                    f"Validation failed in {func.__name__}:\n{str(e)}"
                ) from e
        
        return func(*args, **kwargs)
    
    return wrapper


# ============================================================================
# Utility Functions
# ============================================================================


def get_validation_summary(batch: Dict[str, Any]) -> str:
    """
    Generate a summary of batch contents for debugging.
    
    Args:
        batch: Batch dictionary
        
    Returns:
        String summary of batch structure and shapes
        
    Example:
        >>> summary = get_validation_summary(batch)
        >>> print(summary)
        Batch Summary:
          wsi_features: torch.Tensor [16, 100, 1024]
          genomic: torch.Tensor [16, 2000]
          clinical_text: torch.Tensor [16, 128]
          labels: torch.Tensor [16]
    """
    lines = ["Batch Summary:"]
    
    for key, value in batch.items():
        if value is None:
            lines.append(f"  {key}: None")
        elif isinstance(value, torch.Tensor):
            lines.append(f"  {key}: torch.Tensor {list(value.shape)} dtype={value.dtype}")
        elif isinstance(value, list):
            if len(value) > 0 and isinstance(value[0], torch.Tensor):
                shapes = [list(v.shape) if v is not None else None for v in value[:3]]
                lines.append(f"  {key}: List[Tensor] length={len(value)} shapes={shapes}...")
            else:
                lines.append(f"  {key}: List length={len(value)}")
        else:
            lines.append(f"  {key}: {type(value).__name__}")
    
    return "\n".join(lines)
