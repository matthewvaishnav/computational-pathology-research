"""Shared utility functions for interpretability module."""

import torch
import numpy as np
from typing import Optional, Union
import logging

logger = logging.getLogger(__name__)


def get_device(device: Optional[str] = None) -> torch.device:
    """Get PyTorch device with automatic GPU detection and fallback.

    Args:
        device: Device string ('cuda', 'cpu', or None for auto-detection)

    Returns:
        torch.device object

    Examples:
        >>> device = get_device()  # Auto-detect
        >>> device = get_device('cuda')  # Force GPU
        >>> device = get_device('cpu')  # Force CPU
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu" and torch.cuda.is_available():
            logger.warning(
                "GPU available but not used. Set device='cuda' to enable GPU acceleration."
            )

    torch_device = torch.device(device)

    if torch_device.type == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available. Falling back to CPU.")
        torch_device = torch.device("cpu")

    return torch_device


def to_numpy(tensor: Union[torch.Tensor, np.ndarray]) -> np.ndarray:
    """Convert PyTorch tensor to NumPy array.

    Args:
        tensor: PyTorch tensor or NumPy array

    Returns:
        NumPy array

    Examples:
        >>> arr = to_numpy(torch.tensor([1, 2, 3]))
        >>> arr = to_numpy(np.array([1, 2, 3]))  # No-op
    """
    if isinstance(tensor, torch.Tensor):
        return tensor.detach().cpu().numpy()
    return tensor


def to_tensor(
    array: Union[np.ndarray, torch.Tensor], device: Optional[torch.device] = None
) -> torch.Tensor:
    """Convert NumPy array to PyTorch tensor.

    Args:
        array: NumPy array or PyTorch tensor
        device: Target device (None for CPU)

    Returns:
        PyTorch tensor

    Examples:
        >>> tensor = to_tensor(np.array([1, 2, 3]))
        >>> tensor = to_tensor(torch.tensor([1, 2, 3]))  # No-op
    """
    if isinstance(array, np.ndarray):
        tensor = torch.from_numpy(array)
    else:
        tensor = array

    if device is not None:
        tensor = tensor.to(device)

    return tensor


def normalize_array(array: np.ndarray, min_val: float = 0.0, max_val: float = 1.0) -> np.ndarray:
    """Normalize array to specified range.

    Args:
        array: Input array
        min_val: Minimum value of output range
        max_val: Maximum value of output range

    Returns:
        Normalized array in range [min_val, max_val]

    Examples:
        >>> normalized = normalize_array(np.array([0, 5, 10]))  # [0.0, 0.5, 1.0]
        >>> normalized = normalize_array(np.array([0, 5, 10]), 0, 255)  # [0, 127.5, 255]
    """
    array_min = array.min()
    array_max = array.max()

    if array_max == array_min:
        # Constant array - return array filled with middle value
        return np.full_like(array, (min_val + max_val) / 2, dtype=np.float32)

    # Normalize to [0, 1] then scale to [min_val, max_val]
    normalized = (array - array_min) / (array_max - array_min)
    scaled = normalized * (max_val - min_val) + min_val

    return scaled.astype(np.float32)


def ensure_4d_tensor(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure tensor is 4D [batch, channels, height, width].

    Args:
        tensor: Input tensor (2D, 3D, or 4D)

    Returns:
        4D tensor with batch and channel dimensions

    Examples:
        >>> t = ensure_4d_tensor(torch.randn(224, 224))  # [1, 1, 224, 224]
        >>> t = ensure_4d_tensor(torch.randn(3, 224, 224))  # [1, 3, 224, 224]
        >>> t = ensure_4d_tensor(torch.randn(8, 3, 224, 224))  # [8, 3, 224, 224]
    """
    if tensor.ndim == 2:
        # [H, W] -> [1, 1, H, W]
        return tensor.unsqueeze(0).unsqueeze(0)
    elif tensor.ndim == 3:
        # [C, H, W] -> [1, C, H, W]
        return tensor.unsqueeze(0)
    elif tensor.ndim == 4:
        return tensor
    else:
        raise ValueError(f"Expected 2D, 3D, or 4D tensor, got {tensor.ndim}D")


def clear_gpu_cache():
    """Clear GPU cache to free memory."""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        logger.debug("GPU cache cleared")
