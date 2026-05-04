"""
Metadata and batch structures for WSI streaming.

This module defines data structures for WSI streaming configuration
and tile batch processing.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import numpy as np


@dataclass
class StreamingMetadata:
    """Metadata for WSI streaming configuration."""

    slide_id: str
    dimensions: Tuple[int, int]  # (width, height) at level 0
    estimated_patches: int
    tile_size: int
    memory_budget_gb: float
    target_processing_time: float
    confidence_threshold: float
    level_count: int
    level_dimensions: List[Tuple[int, int]]
    detected_format: str  # Format detected by WSI reader ('openslide', 'dicom', etc.)
    file_extension: str  # Original file extension
    format_compatibility: Dict[str, Union[bool, str, List[str]]]  # Compatibility validation results
    magnification: Optional[float] = None
    mpp: Optional[Tuple[float, float]] = None


@dataclass
class TileBatch:
    """Batch of tiles for processing."""

    tiles: np.ndarray  # [batch_size, height, width, channels]
    coordinates: np.ndarray  # [batch_size, 2] - (x, y) coordinates
    level: int
    batch_id: int
    total_batches: int
    processing_priority: float = 1.0
