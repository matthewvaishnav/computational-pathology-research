"""
WSI Processing Pipeline

This module provides a complete pipeline for processing Whole Slide Images (WSI)
in clinical formats (.svs, .tiff, .ndpi, DICOM) for computational pathology.

Components:
- WSIReader: Read WSI files using OpenSlide
- PatchExtractor: Extract patches at multiple magnification levels
- TissueDetector: Detect tissue regions and filter background (Otsu + deep learning)
- FeatureGenerator: Generate feature embeddings using pretrained encoders
- FeatureCache: Cache processed features in HDF5 format with optimizations
- BatchProcessor: Orchestrate batch processing with memory optimization
- QualityControl: Quality control checks for processed slides

Validation and Benchmarking:
- PerformanceBenchmark: Benchmark pipeline performance
- WSIPipelineValidator: Comprehensive validation suite

Utilities:
- ConfigValidator: Configuration validation and documentation
- ProgressTracker: Progress tracking and monitoring
- LoggingUtils: Standardized logging configuration

Exceptions:
- WSIProcessingError: Base exception for WSI processing errors
- FileFormatError: Raised when file format is unsupported or corrupted
- ResourceError: Raised when system resources are insufficient
- ProcessingError: Raised when processing step fails

Example Usage:
    >>> from data.wsi_pipeline import BatchProcessor, ProcessingConfig
    >>>
    >>> # Configure pipeline
    >>> config = ProcessingConfig(
    ...     patch_size=256,
    ...     encoder_name="resnet50",
    ...     batch_size=32
    ... )
    >>>
    >>> # Process slides
    >>> processor = BatchProcessor(config, num_workers=4)
    >>> result = processor.process_slide("slide.svs")
    >>>
    >>> # Run validation
    >>> from data.wsi_pipeline.validation import run_comprehensive_validation
    >>> validation_results = run_comprehensive_validation()

CLI Usage:
    # Process WSI files
    python -m data.wsi_pipeline.cli process *.svs --output-dir ./features

    # Run benchmarks
    python -m data.wsi_pipeline.cli benchmark --quick

    # Validate installation
    python -m data.wsi_pipeline.cli validate
"""

from .batch_processor import BatchProcessor

# Import validation and benchmarking utilities
from .benchmarks import PerformanceBenchmark, run_performance_benchmarks
from .cache import FeatureCache
from .config import ProcessingConfig

# Import new utilities
from .config_validator import ConfigValidator, get_recommended_config, validate_config
from .exceptions import (
    FileFormatError,
    ProcessingError,
    ResourceError,
    WSIProcessingError,
)
from .extractor import PatchExtractor
from .feature_generator import FeatureGenerator
from .logging_utils import get_logger, setup_logging
from .models import ProcessingResult, SlideMetadata
from .progress_tracker import BatchProgressMonitor, ProgressTracker
from .quality_control import QualityControl
from .reader import WSIReader
from .tissue_detector import TissueDetector
from .validation import WSIPipelineValidator, run_comprehensive_validation

__all__ = [
    # Core components
    "BatchProcessor",
    "FeatureCache",
    "ProcessingConfig",
    "PatchExtractor",
    "FeatureGenerator",
    "SlideMetadata",
    "ProcessingResult",
    "QualityControl",
    "WSIReader",
    "TissueDetector",
    # Exceptions
    "WSIProcessingError",
    "FileFormatError",
    "ResourceError",
    "ProcessingError",
    # Validation and benchmarking
    "PerformanceBenchmark",
    "run_performance_benchmarks",
    "WSIPipelineValidator",
    "run_comprehensive_validation",
    # Utilities
    "ConfigValidator",
    "validate_config",
    "get_recommended_config",
    "ProgressTracker",
    "BatchProgressMonitor",
    "setup_logging",
    "get_logger",
]

__version__ = "1.0.0"
