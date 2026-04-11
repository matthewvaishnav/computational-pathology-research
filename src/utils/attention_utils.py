"""
Utilities for attention weight extraction and storage.

This module provides functions to save and load attention weights from
attention-based MIL models. Attention weights are stored in HDF5 format
along with patch coordinates and slide metadata for visualization and analysis.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple

import h5py
import numpy as np
import torch

logger = logging.getLogger(__name__)


def save_attention_weights(
    attention_weights: torch.Tensor,
    coordinates: torch.Tensor,
    slide_id: str,
    output_dir: Path,
) -> None:
    """Save attention weights and coordinates to HDF5 file.
    
    This function saves attention weights along with their corresponding patch
    coordinates to an HDF5 file for later visualization and analysis. The file
    is named using the slide_id and stored in the output_dir.
    
    Args:
        attention_weights: Attention weights for each patch [num_patches]
        coordinates: Patch coordinates [num_patches, 2] in (x, y) format
        slide_id: Unique identifier for the slide
        output_dir: Directory where HDF5 file will be saved
        
    Raises:
        ValueError: If attention_weights and coordinates have different lengths
        
    Example:
        >>> attention = torch.tensor([0.1, 0.3, 0.6])
        >>> coords = torch.tensor([[0, 0], [256, 0], [0, 256]])
        >>> save_attention_weights(attention, coords, "slide_001", Path("./output"))
    """
    # Validate inputs
    if attention_weights.shape[0] != coordinates.shape[0]:
        raise ValueError(
            f"Attention weights ({attention_weights.shape[0]}) and coordinates "
            f"({coordinates.shape[0]}) must have same length"
        )
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Define output path
    output_path = output_dir / f"{slide_id}.h5"
    
    try:
        # Save to HDF5
        with h5py.File(output_path, "w") as f:
            # Create datasets
            f.create_dataset(
                "attention_weights",
                data=attention_weights.cpu().numpy(),
                compression="gzip",
                compression_opts=4,
            )
            f.create_dataset(
                "coordinates",
                data=coordinates.cpu().numpy(),
                compression="gzip",
                compression_opts=4,
            )
            
            # Add slide_id as attribute
            f.attrs["slide_id"] = slide_id
        
        logger.info(f"Saved attention weights to {output_path}")
        
    except Exception as e:
        logger.error(f"Error saving attention weights for {slide_id}: {e}")
        raise


def load_attention_weights(
    slide_id: str,
    attention_dir: Path,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Load attention weights and coordinates from HDF5 file.
    
    This function loads previously saved attention weights and patch coordinates
    from an HDF5 file. Returns None if the file is not found.
    
    Args:
        slide_id: Unique identifier for the slide
        attention_dir: Directory containing attention weight HDF5 files
        
    Returns:
        Tuple of (attention_weights, coordinates) as numpy arrays, or None if
        file not found. attention_weights has shape [num_patches] and
        coordinates has shape [num_patches, 2].
        
    Example:
        >>> weights, coords = load_attention_weights("slide_001", Path("./output"))
        >>> if weights is not None:
        ...     print(f"Loaded {len(weights)} attention weights")
    """
    # Define file path
    attention_path = Path(attention_dir) / f"{slide_id}.h5"
    
    # Check if file exists
    if not attention_path.exists():
        logger.warning(
            f"Attention weights file not found for {slide_id}: {attention_path}"
        )
        return None
    
    try:
        # Load from HDF5
        with h5py.File(attention_path, "r") as f:
            attention_weights = f["attention_weights"][:]
            coordinates = f["coordinates"][:]
        
        logger.info(
            f"Loaded attention weights for {slide_id}: "
            f"{len(attention_weights)} patches"
        )
        
        return attention_weights, coordinates
        
    except Exception as e:
        logger.error(f"Error loading attention weights for {slide_id}: {e}")
        return None
