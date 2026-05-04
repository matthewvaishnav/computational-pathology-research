"""
Type definitions for inference module.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class InferenceResult:
    """Result of model inference."""
    
    prediction: float
    confidence: float
    class_probabilities: Dict[str, float]
    attention_map: Optional[np.ndarray] = None
    processing_time: Optional[float] = None
    model_version: Optional[str] = None
    metadata: Optional[Dict] = None


@dataclass
class BatchInferenceResult:
    """Result of batch inference."""
    
    results: List[InferenceResult]
    batch_size: int
    total_processing_time: float
    average_processing_time: float
    metadata: Optional[Dict] = None


@dataclass
class StreamingInferenceResult:
    """Result of streaming inference."""
    
    tile_results: List[InferenceResult]
    aggregated_result: InferenceResult
    tile_coordinates: List[Tuple[int, int, int, int]]
    processing_stats: Dict[str, float]
    metadata: Optional[Dict] = None