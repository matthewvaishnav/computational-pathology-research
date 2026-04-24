"""Data models for WSI processing pipeline.

This module defines dataclasses for slide metadata, processing results,
and other data structures used throughout the pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass
class SlideMetadata:
    """Metadata for a whole slide image.

    Attributes:
        slide_id: Unique identifier for the slide
        patient_id: Patient identifier
        file_path: Path to the WSI file
        label: Slide label (e.g., 0 for negative, 1 for positive)
        split: Dataset split (train, val, or test)
        annotation_path: Optional path to annotation file
        width: Slide width in pixels at level 0
        height: Slide height in pixels at level 0
        magnification: Objective magnification (e.g., 20.0 for 20x)
        mpp: Microns per pixel
        scanner_model: Scanner manufacturer and model
        scan_date: Date the slide was scanned
        processing_timestamp: Timestamp when slide was processed
    """

    slide_id: str
    patient_id: str
    file_path: str
    label: int
    split: str
    annotation_path: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    magnification: Optional[float] = None
    mpp: Optional[float] = None
    scanner_model: Optional[str] = None
    scan_date: Optional[str] = None
    processing_timestamp: Optional[str] = None


@dataclass
class ProcessingResult:
    """Result of processing a single slide.

    Attributes:
        slide_id: Unique identifier for the slide
        success: Whether processing completed successfully
        feature_file: Path to the generated HDF5 feature file
        num_patches: Number of patches extracted and processed
        processing_time: Total processing time in seconds
        error_message: Error message if processing failed
        qc_metrics: Quality control metrics dictionary
    """

    slide_id: str
    success: bool
    feature_file: Optional[Path] = None
    num_patches: int = 0
    processing_time: float = 0.0
    error_message: Optional[str] = None
    qc_metrics: Optional[Dict[str, Any]] = None
