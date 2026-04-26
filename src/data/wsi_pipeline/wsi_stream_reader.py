"""
WSI Stream Reader for Real-Time WSI Streaming.

This module implements a streaming WSI reader that progressively loads tiles
with adaptive sizing based on available memory, enabling real-time processing
of gigapixel whole-slide images.

Key Features:
- Progressive tile loading with configurable buffer sizes
- Adaptive tile sizing based on available memory
- Memory-efficient streaming without full slide loading
- Integration with TileBufferPool for optimal caching
- Support for multiple WSI formats (.svs, .tiff, .ndpi, DICOM)
- Automatic format detection and appropriate reader selection
- Consistent streaming API across all supported formats
- Error handling for unsupported or corrupted formats

Requirements Addressed:
- REQ-1.1.1: Support for standard WSI formats (.svs, .tiff, .ndpi, DICOM)
- REQ-1.1.2: Progressive tile streaming with configurable buffer sizes
- REQ-1.1.3: Adaptive tile sizing based on available memory
- REQ-2.2.1: Memory usage below 2GB during processing
- REQ-3.2.2: Handle various WSI formats and scanner types
"""

import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union, Callable
from collections import deque
from datetime import datetime, timedelta

import numpy as np
import psutil
from PIL import Image

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .exceptions import ProcessingError, ResourceError
from .reader import WSIReader
from .tile_buffer_pool import TileBufferConfig, TileBufferPool

logger = logging.getLogger(__name__)


@dataclass
class ProgressCallback:
    """Callback configuration for progress updates."""
    
    callback_func: Callable[[StreamingProgress], None]
    update_interval: float = 1.0  # Update interval in seconds
    min_progress_delta: float = 0.01  # Minimum progress change to trigger update (1%)


class StreamingProgressTracker:
    """
    Comprehensive progress tracking for WSI streaming operations.
    
    Provides detailed progress tracking, ETA estimation, and performance monitoring
    with support for confidence-based early stopping and real-time callbacks.
    
    Features:
    - Multi-stage progress tracking (loading, processing, aggregating)
    - Adaptive ETA estimation with confidence intervals
    - Performance monitoring (throughput, memory, CPU/GPU usage)
    - Confidence tracking with early stopping recommendations
    - Real-time progress callbacks for visualization
    - Quality metrics tracking (failed tiles, data quality)
    """
    
    def __init__(
        self,
        total_tiles: int,
        confidence_threshold: float = 0.95,
        target_processing_time: float = 30.0,
        progress_callbacks: Optional[List[ProgressCallback]] = None
    ):
        """
        Initialize progress tracker.
        
        Args:
            total_tiles: Total number of tiles to process
            confidence_threshold: Confidence threshold for early stopping
            target_processing_time: Target processing time in seconds
            progress_callbacks: List of progress callback configurations
        """
        self.total_tiles = total_tiles
        self.confidence_threshold = confidence_threshold
        self.target_processing_time = target_processing_time
        self.progress_callbacks = progress_callbacks or []
        
        # Timing tracking
        self.start_time: Optional[float] = None
        self.stage_start_times: Dict[str, float] = {}
        self.stage_durations: Dict[str, float] = {}
        
        # Progress tracking
        self.tiles_processed = 0
        self.tiles_skipped = 0
        self.tiles_failed = 0
        self.current_stage = "initializing"
        self.stage_progress = 0.0
        
        # Performance tracking
        self.processing_times: deque = deque(maxlen=100)  # Keep last 100 processing times
        self.throughput_history: deque = deque(maxlen=50)  # Keep last 50 throughput measurements
        self.memory_usage_history: deque = deque(maxlen=100)  # Memory usage history
        
        # Confidence tracking
        self.confidence_history: List[Tuple[float, float]] = []  # (timestamp, confidence)
        self.current_confidence = 0.0
        self.confidence_delta = 0.0
        self.early_stop_recommended = False
        
        # Resource monitoring
        self.peak_memory_usage_gb = 0.0
        self.last_callback_time = 0.0
        self.last_callback_progress = 0.0
        
        logger.info(
            f"Initialized StreamingProgressTracker: {total_tiles} tiles, "
            f"confidence_threshold={confidence_threshold:.3f}, "
            f"target_time={target_processing_time:.1f}s"
        )
    
    def start_processing(self) -> None:
        """Start progress tracking."""
        self.start_time = time.time()
        self.current_stage = "streaming"
        self.stage_start_times[self.current_stage] = self.start_time
        
        logger.info("Started WSI streaming progress tracking")
    
    def start_stage(self, stage_name: str) -> None:
        """
        Start a new processing stage.
        
        Args:
            stage_name: Name of the stage ("streaming", "processing", "aggregating", "finalizing")
        """
        current_time = time.time()
        
        # Finish previous stage
        if self.current_stage in self.stage_start_times:
            stage_duration = current_time - self.stage_start_times[self.current_stage]
            self.stage_durations[self.current_stage] = stage_duration
        
        # Start new stage
        self.current_stage = stage_name
        self.stage_start_times[stage_name] = current_time
        self.stage_progress = 0.0
        
        logger.debug(f"Started processing stage: {stage_name}")
    
    def update_stage_progress(self, progress: float) -> None:
        """
        Update progress within current stage.
        
        Args:
            progress: Progress within stage (0.0 to 1.0)
        """
        self.stage_progress = max(0.0, min(1.0, progress))
    
    def record_tile_processed(
        self,
        processing_time: float,
        tile_size: int,
        success: bool = True,
        skipped: bool = False
    ) -> None:
        """
        Record processing of a tile.
        
        Args:
            processing_time: Time taken to process the tile
            tile_size: Size of the processed tile
            success: Whether processing was successful
            skipped: Whether tile was skipped (e.g., no tissue)
        """
        if success and not skipped:
            self.tiles_processed += 1
            self.processing_times.append(processing_time)
            
            # Calculate throughput (tiles per second)
            if processing_time > 0:
                throughput = 1.0 / processing_time
                self.throughput_history.append(throughput)
        elif skipped:
            self.tiles_skipped += 1
        else:
            self.tiles_failed += 1
        
        # Update memory usage tracking
        current_memory = self._get_current_memory_usage()
        self.memory_usage_history.append(current_memory)
        self.peak_memory_usage_gb = max(self.peak_memory_usage_gb, current_memory)
    
    def update_confidence(self, confidence: float) -> None:
        """
        Update confidence tracking.
        
        Args:
            confidence: Current confidence value (0.0 to 1.0)
        """
        current_time = time.time()
        
        # Calculate confidence delta
        if self.confidence_history:
            self.confidence_delta = confidence - self.current_confidence
        else:
            self.confidence_delta = 0.0
        
        self.current_confidence = confidence
        self.confidence_history.append((current_time, confidence))
        
        # Check for early stopping recommendation
        self._update_early_stopping_recommendation()
    
    def _update_early_stopping_recommendation(self) -> None:
        """Update early stopping recommendation based on confidence and time."""
        # Recommend early stopping if confidence threshold is reached
        confidence_met = self.current_confidence >= self.confidence_threshold
        
        # Also consider if we're making good progress and confidence is stable
        confidence_stable = False
        if len(self.confidence_history) >= 10:
            recent_confidences = [c for _, c in self.confidence_history[-10:]]
            confidence_variance = np.var(recent_confidences)
            confidence_stable = confidence_variance < 0.001  # Low variance indicates stability
        
        # Check if we're approaching time limit
        time_pressure = False
        if self.start_time is not None:
            elapsed_time = time.time() - self.start_time
            time_pressure = elapsed_time > (self.target_processing_time * 0.8)  # 80% of target time
        
        # Recommend early stopping if confidence is met or if confidence is high and stable with time pressure
        self.early_stop_recommended = (
            confidence_met or 
            (self.current_confidence > 0.9 and confidence_stable and time_pressure)
        )
    
    def _get_current_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / (1024**3)  # Convert to GB
        except Exception:
            return 0.0
    
    def _get_gpu_memory_usage(self) -> float:
        """Get current GPU memory usage in GB."""
        if not TORCH_AVAILABLE or not torch.cuda.is_available():
            return 0.0
        
        try:
            return torch.cuda.memory_allocated() / (1024**3)
        except Exception:
            return 0.0
    
    def _get_cpu_utilization(self) -> float:
        """Get current CPU utilization percentage."""
        try:
            return psutil.cpu_percent(interval=None)
        except Exception:
            return 0.0
    
    def _calculate_eta(self) -> Tuple[float, float]:
        """
        Calculate estimated time remaining and total time.
        
        Returns:
            Tuple of (estimated_time_remaining, estimated_total_time)
        """
        if self.start_time is None or self.tiles_processed == 0:
            return 0.0, 0.0
        
        elapsed_time = time.time() - self.start_time
        
        # Use different estimation strategies based on available data
        if len(self.processing_times) >= 10:
            # Use recent processing times for more accurate estimation
            recent_times = list(self.processing_times)[-20:]  # Last 20 tiles
            avg_time_per_tile = np.mean(recent_times)
        else:
            # Fall back to overall average
            avg_time_per_tile = elapsed_time / self.tiles_processed
        
        # Account for early stopping possibility
        effective_remaining_tiles = self.total_tiles - self.tiles_processed
        if self.early_stop_recommended and self.current_confidence > 0.9:
            # Estimate we might stop early - reduce remaining tiles estimate
            confidence_factor = min(1.0, self.current_confidence / self.confidence_threshold)
            early_stop_factor = 0.5 + 0.5 * (1.0 - confidence_factor)  # 0.5 to 1.0
            effective_remaining_tiles = int(effective_remaining_tiles * early_stop_factor)
        
        estimated_time_remaining = effective_remaining_tiles * avg_time_per_tile
        estimated_total_time = elapsed_time + estimated_time_remaining
        
        return estimated_time_remaining, estimated_total_time
    
    def get_current_progress(self) -> StreamingProgress:
        """
        Get current progress information.
        
        Returns:
            StreamingProgress object with current status
        """
        if self.start_time is None:
            # Not started yet
            return StreamingProgress(
                tiles_processed=0,
                total_tiles=self.total_tiles,
                progress_ratio=0.0,
                elapsed_time=0.0,
                estimated_time_remaining=0.0,
                estimated_total_time=0.0,
                current_tile_size=0,
                memory_usage_gb=0.0,
                throughput_tiles_per_second=0.0,
                average_processing_time_per_tile=0.0,
                current_confidence=0.0,
                confidence_delta=0.0,
                early_stop_recommended=False,
                confidence_threshold=self.confidence_threshold,
                current_stage=self.current_stage,
                stage_progress=self.stage_progress,
                time_spent_loading=0.0,
                time_spent_processing=0.0,
                time_spent_aggregating=0.0,
                peak_memory_usage_gb=0.0,
                gpu_memory_usage_gb=0.0,
                cpu_utilization_percent=0.0,
                tiles_skipped=0,
                tiles_failed=0,
                data_quality_score=1.0,
            )
        
        elapsed_time = time.time() - self.start_time
        progress_ratio = self.tiles_processed / self.total_tiles if self.total_tiles > 0 else 0.0
        
        # Calculate ETA
        estimated_time_remaining, estimated_total_time = self._calculate_eta()
        
        # Calculate performance metrics
        avg_throughput = np.mean(self.throughput_history) if self.throughput_history else 0.0
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0.0
        
        # Calculate data quality score
        total_tiles_attempted = self.tiles_processed + self.tiles_failed + self.tiles_skipped
        if total_tiles_attempted > 0:
            data_quality_score = self.tiles_processed / total_tiles_attempted
        else:
            data_quality_score = 1.0
        
        # Get current resource usage
        current_memory = self._get_current_memory_usage()
        gpu_memory = self._get_gpu_memory_usage()
        cpu_utilization = self._get_cpu_utilization()
        
        # Calculate time spent in different stages
        time_spent_loading = self.stage_durations.get("streaming", 0.0)
        time_spent_processing = self.stage_durations.get("processing", 0.0)
        time_spent_aggregating = self.stage_durations.get("aggregating", 0.0)
        
        # Add current stage time if still in progress
        if self.current_stage in self.stage_start_times:
            current_stage_time = elapsed_time - self.stage_start_times[self.current_stage]
            if self.current_stage == "streaming":
                time_spent_loading += current_stage_time
            elif self.current_stage == "processing":
                time_spent_processing += current_stage_time
            elif self.current_stage == "aggregating":
                time_spent_aggregating += current_stage_time
        
        progress = StreamingProgress(
            tiles_processed=self.tiles_processed,
            total_tiles=self.total_tiles,
            progress_ratio=progress_ratio,
            elapsed_time=elapsed_time,
            estimated_time_remaining=estimated_time_remaining,
            estimated_total_time=estimated_total_time,
            current_tile_size=0,  # Will be set by caller
            memory_usage_gb=current_memory,
            throughput_tiles_per_second=avg_throughput,
            average_processing_time_per_tile=avg_processing_time,
            current_confidence=self.current_confidence,
            confidence_delta=self.confidence_delta,
            early_stop_recommended=self.early_stop_recommended,
            confidence_threshold=self.confidence_threshold,
            current_stage=self.current_stage,
            stage_progress=self.stage_progress,
            time_spent_loading=time_spent_loading,
            time_spent_processing=time_spent_processing,
            time_spent_aggregating=time_spent_aggregating,
            peak_memory_usage_gb=self.peak_memory_usage_gb,
            gpu_memory_usage_gb=gpu_memory,
            cpu_utilization_percent=cpu_utilization,
            tiles_skipped=self.tiles_skipped,
            tiles_failed=self.tiles_failed,
            data_quality_score=data_quality_score,
        )
        
        # Trigger progress callbacks if needed
        self._trigger_progress_callbacks(progress)
        
        return progress
    
    def _trigger_progress_callbacks(self, progress: StreamingProgress) -> None:
        """
        Trigger progress callbacks if update conditions are met.
        
        Args:
            progress: Current progress information
        """
        current_time = time.time()
        
        for callback_config in self.progress_callbacks:
            # Check if enough time has passed since last callback
            time_condition = (current_time - self.last_callback_time) >= callback_config.update_interval
            
            # Check if progress has changed significantly
            progress_condition = abs(progress.progress_ratio - self.last_callback_progress) >= callback_config.min_progress_delta
            
            if time_condition or progress_condition:
                try:
                    callback_config.callback_func(progress)
                    self.last_callback_time = current_time
                    self.last_callback_progress = progress.progress_ratio
                except Exception as e:
                    logger.warning(f"Progress callback failed: {e}")
    
    def finish_processing(self) -> StreamingProgress:
        """
        Finish progress tracking and return final statistics.
        
        Returns:
            Final progress information
        """
        if self.current_stage in self.stage_start_times:
            # Finish current stage
            current_time = time.time()
            stage_duration = current_time - self.stage_start_times[self.current_stage]
            self.stage_durations[self.current_stage] = stage_duration
        
        self.current_stage = "completed"
        final_progress = self.get_current_progress()
        
        logger.info(
            f"WSI streaming completed: {self.tiles_processed}/{self.total_tiles} tiles processed "
            f"({final_progress.progress_ratio:.1%}) in {final_progress.elapsed_time:.1f}s, "
            f"final confidence: {self.current_confidence:.3f}"
        )
        
        return final_progress


logger = logging.getLogger(__name__)


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


@dataclass
class StreamingProgress:
    """Progress information for streaming with comprehensive tracking."""
    
    # Basic progress metrics
    tiles_processed: int
    total_tiles: int
    progress_ratio: float
    
    # Time and ETA metrics
    elapsed_time: float
    estimated_time_remaining: float
    estimated_total_time: float
    
    # Performance metrics
    current_tile_size: int
    memory_usage_gb: float
    throughput_tiles_per_second: float
    average_processing_time_per_tile: float
    
    # Confidence and early stopping metrics
    current_confidence: float
    confidence_delta: float
    early_stop_recommended: bool
    confidence_threshold: float
    
    # Processing stage information
    current_stage: str  # "initializing", "streaming", "processing", "aggregating", "finalizing"
    stage_progress: float  # Progress within current stage (0.0 to 1.0)
    
    # Detailed timing breakdown
    time_spent_loading: float
    time_spent_processing: float
    time_spent_aggregating: float
    
    # Memory and resource metrics
    peak_memory_usage_gb: float
    gpu_memory_usage_gb: float
    cpu_utilization_percent: float
    
    # Quality metrics
    tiles_skipped: int
    tiles_failed: int
    data_quality_score: float
    
    def to_dict(self) -> Dict[str, Union[int, float, str, bool]]:
        """Convert progress to dictionary for serialization."""
        return {
            'tiles_processed': self.tiles_processed,
            'total_tiles': self.total_tiles,
            'progress_ratio': self.progress_ratio,
            'elapsed_time': self.elapsed_time,
            'estimated_time_remaining': self.estimated_time_remaining,
            'estimated_total_time': self.estimated_total_time,
            'current_tile_size': self.current_tile_size,
            'memory_usage_gb': self.memory_usage_gb,
            'throughput_tiles_per_second': self.throughput_tiles_per_second,
            'average_processing_time_per_tile': self.average_processing_time_per_tile,
            'current_confidence': self.current_confidence,
            'confidence_delta': self.confidence_delta,
            'early_stop_recommended': self.early_stop_recommended,
            'confidence_threshold': self.confidence_threshold,
            'current_stage': self.current_stage,
            'stage_progress': self.stage_progress,
            'time_spent_loading': self.time_spent_loading,
            'time_spent_processing': self.time_spent_processing,
            'time_spent_aggregating': self.time_spent_aggregating,
            'peak_memory_usage_gb': self.peak_memory_usage_gb,
            'gpu_memory_usage_gb': self.gpu_memory_usage_gb,
            'cpu_utilization_percent': self.cpu_utilization_percent,
            'tiles_skipped': self.tiles_skipped,
            'tiles_failed': self.tiles_failed,
            'data_quality_score': self.data_quality_score,
        }
    
    def get_eta_string(self) -> str:
        """Get ETA as formatted string."""
        if self.estimated_time_remaining <= 0:
            return "Complete"
        
        total_seconds = int(self.estimated_time_remaining)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}"
        else:
            return f"{minutes:02d}:{seconds:02d}"
    
    def get_progress_percentage(self) -> str:
        """Get progress as formatted percentage string."""
        return f"{self.progress_ratio * 100:.1f}%"


class WSIStreamReader:
    """
    Streaming WSI reader with adaptive tile sizing and memory management.
    
    This class provides progressive tile loading capabilities for gigapixel WSI files,
    with adaptive tile sizing based on available memory and processing performance.
    
    Features:
    - Progressive tile loading without full slide loading
    - Adaptive tile sizing (64-4096 pixels) based on memory pressure
    - Integration with TileBufferPool for efficient caching
    - Memory pressure-aware sizing adjustments
    - Support for multiple WSI formats
    
    Example:
        >>> config = TileBufferConfig(max_memory_gb=2.0, adaptive_sizing_enabled=True)
        >>> reader = WSIStreamReader("slide.svs", config)
        >>> metadata = reader.initialize_streaming()
        >>> 
        >>> for batch in reader.stream_tiles():
        ...     # Process batch of tiles
        ...     features = process_tiles(batch.tiles)
        ...     
        ...     # Check progress
        ...     progress = reader.get_progress()
        ...     print(f"Progress: {progress.progress_ratio:.1%}")
    """
    
    def __init__(
        self, 
        wsi_path: Union[str, Path], 
        config: Optional[TileBufferConfig] = None,
        level: int = 0,
        overlap: int = 0,
        progress_callbacks: Optional[List[ProgressCallback]] = None
    ):
        """
        Initialize WSI stream reader.
        
        Args:
            wsi_path: Path to WSI file
            config: Tile buffer configuration (uses defaults if None)
            level: Pyramid level to read from (default: 0 for highest resolution)
            overlap: Tile overlap in pixels (default: 0)
            progress_callbacks: List of progress callback configurations for real-time updates
            
        Raises:
            ProcessingError: If WSI file cannot be opened
            ResourceError: If insufficient memory available
            FileFormatError: If file format is unsupported
        """
        self.wsi_path = Path(wsi_path)
        self.config = config or TileBufferConfig()
        self.level = level
        self.overlap = overlap
        self.progress_callbacks = progress_callbacks or []
        
        # Validate file exists and format is supported
        self._validate_wsi_file()
        
        # Initialize WSI reader with format detection
        try:
            self.wsi_reader = WSIReader(self.wsi_path)
            self._detected_format = self._get_detected_format()
        except Exception as e:
            raise ProcessingError(f"Failed to open WSI file {wsi_path}: {e}")
        
        # Initialize tile buffer pool
        self.tile_pool = TileBufferPool(self.config)
        
        # Streaming state
        self._metadata: Optional[StreamingMetadata] = None
        self._tile_coordinates: List[Tuple[int, int]] = []
        self._progress_tracker: Optional[StreamingProgressTracker] = None
        
        # Legacy compatibility - will be removed in favor of progress tracker
        self._tiles_processed = 0
        self._start_time: Optional[float] = None
        self._throughput_history: List[float] = []
        
        logger.info(
            f"Initialized WSIStreamReader for {self.wsi_path.name} "
            f"(format={self._detected_format}, level={level}, "
            f"adaptive_sizing={self.config.adaptive_sizing_enabled}, "
            f"progress_callbacks={len(self.progress_callbacks)})"
        )
    
    def initialize_streaming(
        self, 
        target_processing_time: float = 30.0,
        confidence_threshold: float = 0.95
    ) -> StreamingMetadata:
        """
        Initialize streaming with slide metadata and memory optimization.
        
        Args:
            target_processing_time: Target processing time in seconds
            confidence_threshold: Confidence threshold for early stopping
            
        Returns:
            StreamingMetadata with slide information and configuration
            
        Raises:
            ResourceError: If insufficient memory for streaming
        """
        # Get slide dimensions and properties
        dimensions = self.wsi_reader.dimensions
        level_count = self.wsi_reader.level_count
        level_dimensions = self.wsi_reader.level_dimensions
        
        # Get current adaptive tile size
        current_tile_size = self.tile_pool.get_current_tile_size()
        
        # Calculate tile coordinates for the specified level
        level_dims = level_dimensions[self.level]
        self._tile_coordinates = self._calculate_tile_coordinates(
            level_dims, current_tile_size, self.overlap
        )
        
        # Estimate total patches
        estimated_patches = len(self._tile_coordinates)
        
        # Initialize progress tracker
        self._progress_tracker = StreamingProgressTracker(
            total_tiles=estimated_patches,
            confidence_threshold=confidence_threshold,
            target_processing_time=target_processing_time,
            progress_callbacks=self.progress_callbacks
        )
        
        # Validate memory requirements
        estimated_memory_gb = self._estimate_memory_requirements(
            estimated_patches, current_tile_size
        )
        
        if estimated_memory_gb > self.config.max_memory_gb:
            # Try to reduce tile size to fit memory budget
            self.tile_pool.update_adaptive_tile_size()
            new_tile_size = self.tile_pool.get_current_tile_size()
            
            if new_tile_size < current_tile_size:
                # Recalculate with new tile size
                self._tile_coordinates = self._calculate_tile_coordinates(
                    level_dims, new_tile_size, self.overlap
                )
                estimated_patches = len(self._tile_coordinates)
                estimated_memory_gb = self._estimate_memory_requirements(
                    estimated_patches, new_tile_size
                )
                current_tile_size = new_tile_size
                
                # Update progress tracker with new tile count
                self._progress_tracker.total_tiles = estimated_patches
                
                logger.info(
                    f"Reduced tile size to {new_tile_size}px to fit memory budget "
                    f"(estimated memory: {estimated_memory_gb:.2f}GB)"
                )
        
        if estimated_memory_gb > self.config.max_memory_gb:
            raise ResourceError(
                f"Estimated memory usage ({estimated_memory_gb:.2f}GB) exceeds "
                f"maximum allowed ({self.config.max_memory_gb:.2f}GB). "
                f"Consider reducing tile size or increasing memory limit."
            )
        
        # Validate format compatibility
        compatibility = self.validate_format_compatibility()
        if not compatibility['is_compatible']:
            logger.warning(
                f"Format compatibility issues detected: {compatibility['compatibility_issues']}"
            )
        
        # Create metadata
        self._metadata = StreamingMetadata(
            slide_id=self.wsi_path.stem,
            dimensions=dimensions,
            estimated_patches=estimated_patches,
            tile_size=current_tile_size,
            memory_budget_gb=self.config.max_memory_gb,
            target_processing_time=target_processing_time,
            confidence_threshold=confidence_threshold,
            level_count=level_count,
            level_dimensions=level_dimensions,
            detected_format=self._detected_format,
            file_extension=self.wsi_path.suffix.lower(),
            format_compatibility=compatibility,
            magnification=self.wsi_reader.get_magnification(),
            mpp=self.wsi_reader.get_mpp()
        )
        
        # Reset legacy streaming state for backward compatibility
        self._tiles_processed = 0
        self._start_time = None
        self._throughput_history = []
        
        logger.info(
            f"Initialized streaming: {estimated_patches} tiles, "
            f"tile_size={current_tile_size}px, "
            f"estimated_memory={estimated_memory_gb:.2f}GB, "
            f"target_time={target_processing_time:.1f}s, "
            f"confidence_threshold={confidence_threshold:.3f}"
        )
        
        return self._metadata
    
    def stream_tiles(self, batch_size: int = 16) -> Iterator[TileBatch]:
        """
        Stream WSI tiles progressively with adaptive sizing and comprehensive progress tracking.
        
        Args:
            batch_size: Number of tiles per batch
            
        Yields:
            TileBatch objects containing tiles and metadata
            
        Raises:
            ProcessingError: If tile reading fails
        """
        if self._metadata is None:
            raise ProcessingError("Streaming not initialized. Call initialize_streaming() first.")
        
        if self._progress_tracker is None:
            raise ProcessingError("Progress tracker not initialized.")
        
        # Start progress tracking
        self._progress_tracker.start_processing()
        self._start_time = time.time()  # Legacy compatibility
        total_batches = (len(self._tile_coordinates) + batch_size - 1) // batch_size
        
        logger.info(f"Starting tile streaming: {len(self._tile_coordinates)} tiles in {total_batches} batches")
        
        for batch_id in range(total_batches):
            batch_start_time = time.time()
            
            # Update stage progress
            batch_progress = batch_id / total_batches
            self._progress_tracker.update_stage_progress(batch_progress)
            
            # Get batch coordinates
            start_idx = batch_id * batch_size
            end_idx = min(start_idx + batch_size, len(self._tile_coordinates))
            batch_coordinates = self._tile_coordinates[start_idx:end_idx]
            
            # Check for adaptive tile size updates
            if self.config.adaptive_sizing_enabled and batch_id % 10 == 0:
                # Update tile size every 10 batches
                tile_size_changed = self.tile_pool.update_adaptive_tile_size()
                if tile_size_changed:
                    new_tile_size = self.tile_pool.get_current_tile_size()
                    logger.info(f"Tile size adapted to {new_tile_size}px during streaming")
            
            # Load tiles for this batch
            current_tile_size = self.tile_pool.get_current_tile_size()
            tiles = []
            valid_coordinates = []
            
            for coord in batch_coordinates:
                tile_start_time = time.time()
                success = False
                skipped = False
                
                try:
                    # Check if tile is already in buffer pool
                    cached_tile = self.tile_pool.get_tile(coord, self.level)
                    
                    if cached_tile is not None:
                        tile_data = cached_tile
                    else:
                        # Read tile from WSI
                        tile_data = self._read_tile(coord, current_tile_size)
                        
                        # Check if tile contains tissue (basic quality check)
                        if self._is_tile_valid(tile_data):
                            # Store in buffer pool for future use
                            self.tile_pool.store_tile(coord, self.level, tile_data)
                        else:
                            # Skip tiles with no tissue or poor quality
                            skipped = True
                            logger.debug(f"Skipped tile at {coord}: no tissue detected")
                    
                    if not skipped:
                        tiles.append(tile_data)
                        valid_coordinates.append(coord)
                        success = True
                    
                except Exception as e:
                    logger.warning(f"Failed to read tile at {coord}: {e}")
                    success = False
                
                # Record tile processing in progress tracker
                tile_processing_time = time.time() - tile_start_time
                self._progress_tracker.record_tile_processed(
                    processing_time=tile_processing_time,
                    tile_size=current_tile_size,
                    success=success,
                    skipped=skipped
                )
                
                # Update legacy counters for backward compatibility
                if success and not skipped:
                    self._tiles_processed += 1
            
            if not tiles:
                logger.warning(f"No valid tiles in batch {batch_id}")
                continue
            
            # Convert to numpy arrays
            tiles_array = np.stack(tiles, axis=0)
            coordinates_array = np.array(valid_coordinates)
            
            # Calculate batch throughput for legacy compatibility
            batch_time = time.time() - batch_start_time
            if batch_time > 0:
                throughput = len(tiles) / batch_time
                self._throughput_history.append(throughput)
                
                # Keep only recent throughput measurements
                if len(self._throughput_history) > 50:
                    self._throughput_history = self._throughput_history[-50:]
            
            # Create batch object
            batch = TileBatch(
                tiles=tiles_array,
                coordinates=coordinates_array,
                level=self.level,
                batch_id=batch_id,
                total_batches=total_batches,
                processing_priority=1.0
            )
            
            yield batch
            
            # Memory optimization check
            if batch_id % 20 == 0:  # Every 20 batches
                self.tile_pool.optimize_memory_usage()
            
            # Check for early stopping recommendation
            current_progress = self._progress_tracker.get_current_progress()
            if current_progress.early_stop_recommended:
                logger.info(
                    f"Early stopping recommended at batch {batch_id}/{total_batches} "
                    f"(confidence: {current_progress.current_confidence:.3f})"
                )
                break
        
        # Finish streaming stage
        self._progress_tracker.start_stage("completed")
        logger.info(f"Completed tile streaming: {self._progress_tracker.tiles_processed} tiles processed")
    
    def _is_tile_valid(self, tile_data: np.ndarray) -> bool:
        """
        Check if tile contains valid tissue data.
        
        Args:
            tile_data: Tile data as numpy array
            
        Returns:
            True if tile contains tissue, False if mostly background
        """
        try:
            # Simple tissue detection based on color variance
            # This is a basic implementation - more sophisticated methods could be used
            if len(tile_data.shape) == 3 and tile_data.shape[2] == 3:
                # RGB image - check if it's not mostly white/background
                mean_intensity = np.mean(tile_data)
                std_intensity = np.std(tile_data)
                
                # Tissue typically has lower mean intensity and higher variance than background
                is_tissue = mean_intensity < 240 and std_intensity > 10
                return is_tissue
            else:
                # For non-RGB images, assume valid
                return True
        except Exception:
            # If validation fails, assume tile is valid
            return True
    
    def get_progress(self) -> StreamingProgress:
        """
        Get current streaming progress information with comprehensive tracking.
        
        Returns:
            StreamingProgress with current status
        """
        if self._progress_tracker is None:
            # Fall back to legacy progress calculation if tracker not initialized
            return self._get_legacy_progress()
        
        # Get progress from comprehensive tracker
        progress = self._progress_tracker.get_current_progress()
        
        # Update current tile size from tile pool
        progress.current_tile_size = self.tile_pool.get_current_tile_size()
        
        return progress
    
    def _get_legacy_progress(self) -> StreamingProgress:
        """
        Legacy progress calculation for backward compatibility.
        
        Returns:
            StreamingProgress with basic information
        """
        if self._metadata is None:
            return StreamingProgress(
                tiles_processed=0,
                total_tiles=0,
                progress_ratio=0.0,
                elapsed_time=0.0,
                estimated_time_remaining=0.0,
                estimated_total_time=0.0,
                current_tile_size=0,
                memory_usage_gb=0.0,
                throughput_tiles_per_second=0.0,
                average_processing_time_per_tile=0.0,
                current_confidence=0.0,
                confidence_delta=0.0,
                early_stop_recommended=False,
                confidence_threshold=0.95,
                current_stage="unknown",
                stage_progress=0.0,
                time_spent_loading=0.0,
                time_spent_processing=0.0,
                time_spent_aggregating=0.0,
                peak_memory_usage_gb=0.0,
                gpu_memory_usage_gb=0.0,
                cpu_utilization_percent=0.0,
                tiles_skipped=0,
                tiles_failed=0,
                data_quality_score=1.0,
            )
        
        total_tiles = self._metadata.estimated_patches
        progress_ratio = self._tiles_processed / total_tiles if total_tiles > 0 else 0.0
        
        # Calculate estimated time remaining
        if self._start_time is not None and self._tiles_processed > 0:
            elapsed_time = time.time() - self._start_time
            avg_time_per_tile = elapsed_time / self._tiles_processed
            remaining_tiles = total_tiles - self._tiles_processed
            estimated_time_remaining = remaining_tiles * avg_time_per_tile
        else:
            elapsed_time = 0.0
            estimated_time_remaining = 0.0
        
        # Calculate average throughput
        avg_throughput = np.mean(self._throughput_history) if self._throughput_history else 0.0
        
        return StreamingProgress(
            tiles_processed=self._tiles_processed,
            total_tiles=total_tiles,
            progress_ratio=progress_ratio,
            elapsed_time=elapsed_time,
            estimated_time_remaining=estimated_time_remaining,
            estimated_total_time=elapsed_time + estimated_time_remaining,
            current_tile_size=self.tile_pool.get_current_tile_size(),
            memory_usage_gb=self.tile_pool.get_memory_usage(),
            throughput_tiles_per_second=avg_throughput,
            average_processing_time_per_tile=elapsed_time / self._tiles_processed if self._tiles_processed > 0 else 0.0,
            current_confidence=0.0,
            confidence_delta=0.0,
            early_stop_recommended=False,
            confidence_threshold=0.95,
            current_stage="streaming",
            stage_progress=progress_ratio,
            time_spent_loading=elapsed_time,
            time_spent_processing=0.0,
            time_spent_aggregating=0.0,
            peak_memory_usage_gb=self.tile_pool.get_memory_usage(),
            gpu_memory_usage_gb=0.0,
            cpu_utilization_percent=0.0,
            tiles_skipped=0,
            tiles_failed=0,
            data_quality_score=1.0,
        )
    
    def update_confidence(self, confidence: float) -> None:
        """
        Update confidence tracking for early stopping decisions.
        
        Args:
            confidence: Current confidence value (0.0 to 1.0)
        """
        if self._progress_tracker is not None:
            self._progress_tracker.update_confidence(confidence)
    
    def add_progress_callback(self, callback_func: Callable[[StreamingProgress], None], 
                            update_interval: float = 1.0, 
                            min_progress_delta: float = 0.01) -> None:
        """
        Add a progress callback for real-time updates.
        
        Args:
            callback_func: Function to call with progress updates
            update_interval: Minimum time between callbacks (seconds)
            min_progress_delta: Minimum progress change to trigger callback
        """
        callback_config = ProgressCallback(
            callback_func=callback_func,
            update_interval=update_interval,
            min_progress_delta=min_progress_delta
        )
        self.progress_callbacks.append(callback_config)
        
        # Add to existing progress tracker if available
        if self._progress_tracker is not None:
            self._progress_tracker.progress_callbacks.append(callback_config)
    
    def get_detailed_progress_stats(self) -> Dict[str, Union[int, float, str, bool]]:
        """
        Get detailed progress statistics for monitoring and debugging.
        
        Returns:
            Dictionary with comprehensive progress information
        """
        if self._progress_tracker is None:
            return {"error": "Progress tracker not initialized"}
        
        progress = self._progress_tracker.get_current_progress()
        stats = progress.to_dict()
        
        # Add additional streaming-specific stats
        stats.update({
            'slide_path': str(self.wsi_path),
            'slide_id': self._metadata.slide_id if self._metadata else "unknown",
            'level': self.level,
            'adaptive_sizing_enabled': self.config.adaptive_sizing_enabled,
            'buffer_pool_stats': self.tile_pool.get_buffer_stats(),
            'adaptive_sizing_stats': self.tile_pool.get_adaptive_sizing_stats(),
        })
        
        return stats
    
    def get_performance_summary(self) -> Dict[str, Union[int, float, str]]:
        """
        Get performance summary for benchmarking and optimization.
        
        Returns:
            Dictionary with performance metrics
        """
        if self._progress_tracker is None:
            return {"error": "Progress tracker not initialized"}
        
        progress = self._progress_tracker.get_current_progress()
        
        return {
            'total_processing_time': progress.elapsed_time,
            'tiles_per_second': progress.throughput_tiles_per_second,
            'average_tile_time': progress.average_processing_time_per_tile,
            'memory_efficiency': progress.memory_usage_gb / self.config.max_memory_gb,
            'data_quality_score': progress.data_quality_score,
            'early_stopping_triggered': progress.early_stop_recommended,
            'confidence_achieved': progress.current_confidence,
            'target_confidence': progress.confidence_threshold,
            'processing_stages': {
                'loading_time': progress.time_spent_loading,
                'processing_time': progress.time_spent_processing,
                'aggregating_time': progress.time_spent_aggregating,
            },
            'resource_usage': {
                'peak_memory_gb': progress.peak_memory_usage_gb,
                'gpu_memory_gb': progress.gpu_memory_usage_gb,
                'cpu_utilization': progress.cpu_utilization_percent,
            }
        }
    
    def finish_streaming(self) -> StreamingProgress:
        """
        Finish streaming and get final progress statistics.
        
        Returns:
            Final progress information
        """
        if self._progress_tracker is not None:
            return self._progress_tracker.finish_processing()
        else:
            return self.get_progress()
        """
        Estimate total number of patches for progress tracking.
        
        Returns:
            Estimated number of patches
        """
        if self._metadata is not None:
            return self._metadata.estimated_patches
        
        # Quick estimation without full initialization
        dimensions = self.wsi_reader.level_dimensions[self.level]
        current_tile_size = self.tile_pool.get_current_tile_size()
        
        tiles_x = (dimensions[0] + current_tile_size - 1) // current_tile_size
        tiles_y = (dimensions[1] + current_tile_size - 1) // current_tile_size
        
        return tiles_x * tiles_y
    
    def estimate_total_patches(self) -> int:
        """
        Estimate total number of patches for progress tracking.
        
        Returns:
            Estimated number of patches
        """
        if self._metadata is not None:
            return self._metadata.estimated_patches
        
        # Quick estimation without full initialization
        dimensions = self.wsi_reader.level_dimensions[self.level]
        current_tile_size = self.tile_pool.get_current_tile_size()
        
        tiles_x = (dimensions[0] + current_tile_size - 1) // current_tile_size
        tiles_y = (dimensions[1] + current_tile_size - 1) // current_tile_size
        
        return tiles_x * tiles_y
    
    def _calculate_tile_coordinates(
        self, 
        level_dimensions: Tuple[int, int], 
        tile_size: int, 
        overlap: int
    ) -> List[Tuple[int, int]]:
        """
        Calculate tile coordinates for the given level and tile size.
        
        Args:
            level_dimensions: (width, height) of the level
            tile_size: Size of tiles in pixels
            overlap: Overlap between tiles in pixels
            
        Returns:
            List of (x, y) coordinates for tiles
        """
        width, height = level_dimensions
        stride = tile_size - overlap
        
        coordinates = []
        
        y = 0
        while y < height:
            x = 0
            while x < width:
                coordinates.append((x, y))
                x += stride
            y += stride
        
        return coordinates
    
    def _read_tile(self, coordinate: Tuple[int, int], tile_size: int) -> np.ndarray:
        """
        Read a single tile from the WSI with format-specific optimizations.
        
        Args:
            coordinate: (x, y) coordinate for the tile
            tile_size: Size of the tile in pixels
            
        Returns:
            Tile data as numpy array
            
        Raises:
            ProcessingError: If tile reading fails
        """
        try:
            x, y = coordinate
            
            # Apply format-specific optimizations
            if self._detected_format == 'openslide':
                # OpenSlide optimization: use read_region directly
                tile_data = self.wsi_reader.read_region(
                    location=(x, y),
                    level=self.level,
                    size=(tile_size, tile_size)
                )
            elif self._detected_format == 'dicom':
                # DICOM optimization: may need special handling for large tiles
                try:
                    tile_data = self.wsi_reader.read_region(
                        location=(x, y),
                        level=self.level,
                        size=(tile_size, tile_size)
                    )
                except Exception as e:
                    # Fallback for DICOM reading issues
                    logger.warning(f"DICOM tile reading failed, trying smaller size: {e}")
                    # Try with smaller tile size for problematic DICOM files
                    smaller_size = min(tile_size, 512)
                    tile_data = self.wsi_reader.read_region(
                        location=(x, y),
                        level=self.level,
                        size=(smaller_size, smaller_size)
                    )
                    # Pad to requested size if needed
                    if tile_data.shape[:2] != (tile_size, tile_size):
                        tile_data = self._normalize_tile_size(tile_data, tile_size)
            else:
                # Generic reading for unknown formats
                tile_data = self.wsi_reader.read_region(
                    location=(x, y),
                    level=self.level,
                    size=(tile_size, tile_size)
                )
            
            # Ensure consistent shape
            if tile_data.shape[:2] != (tile_size, tile_size):
                tile_data = self._normalize_tile_size(tile_data, tile_size)
            
            # Format-specific post-processing
            tile_data = self._apply_format_specific_processing(tile_data)
            
            return tile_data
            
        except Exception as e:
            raise ProcessingError(f"Failed to read tile at {coordinate}: {e}")
    
    def _apply_format_specific_processing(self, tile_data: np.ndarray) -> np.ndarray:
        """
        Apply format-specific post-processing to tile data.
        
        Args:
            tile_data: Raw tile data
            
        Returns:
            Processed tile data
        """
        try:
            if self._detected_format == 'openslide':
                # OpenSlide typically returns RGB data, ensure it's in correct format
                if len(tile_data.shape) == 3 and tile_data.shape[2] == 4:
                    # Convert RGBA to RGB if needed
                    tile_data = tile_data[:, :, :3]
            
            elif self._detected_format == 'dicom':
                # DICOM may need color space conversion or normalization
                if len(tile_data.shape) == 2:
                    # Convert grayscale to RGB if needed
                    tile_data = np.stack([tile_data] * 3, axis=2)
                elif len(tile_data.shape) == 3 and tile_data.shape[2] == 1:
                    # Convert single channel to RGB
                    tile_data = np.repeat(tile_data, 3, axis=2)
            
            # Ensure data type is uint8
            if tile_data.dtype != np.uint8:
                if tile_data.dtype in [np.uint16, np.int16]:
                    # Scale down from 16-bit to 8-bit
                    tile_data = (tile_data / 256).astype(np.uint8)
                elif tile_data.dtype in [np.float32, np.float64]:
                    # Scale float data to uint8 range
                    tile_data = (tile_data * 255).astype(np.uint8)
                else:
                    tile_data = tile_data.astype(np.uint8)
            
            return tile_data
            
        except Exception as e:
            logger.warning(f"Format-specific processing failed: {e}")
            return tile_data  # Return original data if processing fails
    
    def _normalize_tile_size(self, tile_data: np.ndarray, target_size: int) -> np.ndarray:
        """
        Normalize tile to target size by padding or cropping.
        
        Args:
            tile_data: Input tile data
            target_size: Target tile size
            
        Returns:
            Normalized tile data
        """
        h, w = tile_data.shape[:2]
        
        if h == target_size and w == target_size:
            return tile_data
        
        # Create target array
        if len(tile_data.shape) == 3:
            normalized = np.zeros((target_size, target_size, tile_data.shape[2]), dtype=tile_data.dtype)
        else:
            normalized = np.zeros((target_size, target_size), dtype=tile_data.dtype)
        
        # Copy data (crop if larger, pad if smaller)
        copy_h = min(h, target_size)
        copy_w = min(w, target_size)
        
        normalized[:copy_h, :copy_w] = tile_data[:copy_h, :copy_w]
        
        return normalized
    
    def _validate_wsi_file(self) -> None:
        """
        Validate WSI file exists and format is supported.
        
        Raises:
            FileFormatError: If file doesn't exist or format is unsupported
        """
        if not self.wsi_path.exists():
            raise FileFormatError(f"WSI file not found: {self.wsi_path}")
        
        if not self.wsi_path.is_file():
            raise FileFormatError(f"Path is not a file: {self.wsi_path}")
        
        # Check file size (basic validation)
        file_size = self.wsi_path.stat().st_size
        if file_size == 0:
            raise FileFormatError(f"WSI file is empty: {self.wsi_path}")
        
        # Validate format is supported
        supported_formats = self._get_supported_formats()
        file_extension = self.wsi_path.suffix.lower()
        
        # Special handling for DICOM files (may not have .dcm extension)
        if file_extension not in supported_formats and not self._is_likely_dicom():
            raise FileFormatError(
                f"Unsupported file format: {file_extension}. "
                f"Supported formats: {', '.join(supported_formats)}"
            )
    
    def _get_supported_formats(self) -> List[str]:
        """
        Get list of supported WSI file formats.
        
        Returns:
            List of supported file extensions
        """
        supported = []
        
        # OpenSlide formats
        try:
            from openslide import OpenSlide
            supported.extend(['.svs', '.tiff', '.tif', '.ndpi', '.vms', '.vmu', '.scn', '.mrxs', '.bif'])
        except ImportError:
            logger.warning("OpenSlide not available - .svs, .tiff, .ndpi formats not supported")
        
        # DICOM formats
        try:
            import wsidicom
            supported.extend(['.dcm', ''])  # DICOM files may not have extension
        except ImportError:
            logger.warning("wsidicom not available - DICOM format not supported")
        
        return supported
    
    def _is_likely_dicom(self) -> bool:
        """
        Check if file is likely DICOM format by reading magic bytes.
        
        Returns:
            True if file appears to be DICOM format
        """
        try:
            with open(self.wsi_path, "rb") as f:
                # DICOM files have 'DICM' at byte 128
                f.seek(128)
                magic = f.read(4)
                return magic == b"DICM"
        except Exception:
            return False
    
    def _get_detected_format(self) -> str:
        """
        Get the detected format from the WSI reader.
        
        Returns:
            String indicating the detected format ('openslide', 'dicom', 'unknown')
        """
        if hasattr(self.wsi_reader, '_format'):
            return self.wsi_reader._format
        
        # Fallback format detection based on file extension
        suffix = self.wsi_path.suffix.lower()
        if suffix in ['.svs', '.tiff', '.tif', '.ndpi', '.vms', '.vmu', '.scn', '.mrxs', '.bif']:
            return 'openslide'
        elif suffix in ['.dcm'] or self._is_likely_dicom():
            return 'dicom'
        else:
            return 'unknown'
    
    def get_format_info(self) -> Dict[str, Union[str, bool, List[str]]]:
        """
        Get comprehensive format information for the loaded WSI.
        
        Returns:
            Dictionary with format details including:
            - detected_format: The detected format type
            - file_extension: File extension
            - supported_formats: List of all supported formats
            - format_specific_properties: Format-specific metadata
        """
        format_info = {
            'detected_format': self._detected_format,
            'file_extension': self.wsi_path.suffix.lower(),
            'supported_formats': self._get_supported_formats(),
            'file_size_mb': self.wsi_path.stat().st_size / (1024 * 1024),
        }
        
        # Add format-specific properties
        try:
            properties = self.wsi_reader.properties
            scanner_info = self.wsi_reader.get_scanner_info()
            
            format_info['format_specific_properties'] = {
                'scanner_model': scanner_info.get('model'),
                'scan_date': scanner_info.get('date'),
                'magnification': self.wsi_reader.get_magnification(),
                'mpp': self.wsi_reader.get_mpp(),
                'property_count': len(properties),
            }
            
            # Add format-specific details
            if self._detected_format == 'openslide':
                format_info['openslide_vendor'] = properties.get('openslide.vendor', 'unknown')
                format_info['openslide_quickhash'] = properties.get('openslide.quickhash-1', 'unknown')
            elif self._detected_format == 'dicom':
                format_info['dicom_patient_id'] = properties.get('patient_id', 'unknown')
                format_info['dicom_study_date'] = properties.get('study_date', 'unknown')
                format_info['dicom_manufacturer'] = properties.get('manufacturer', 'unknown')
        
        except Exception as e:
            logger.warning(f"Failed to extract format-specific properties: {e}")
            format_info['format_specific_properties'] = {}
        
        return format_info
    
    def validate_format_compatibility(self) -> Dict[str, Union[bool, str, List[str]]]:
        """
        Validate format compatibility and streaming capabilities.
        
        Returns:
            Dictionary with compatibility information:
            - is_compatible: Whether format is fully compatible with streaming
            - compatibility_issues: List of any compatibility issues found
            - recommended_actions: Suggested actions for any issues
            - streaming_optimizations: Available optimizations for this format
        """
        compatibility = {
            'is_compatible': True,
            'compatibility_issues': [],
            'recommended_actions': [],
            'streaming_optimizations': [],
        }
        
        try:
            # Check pyramid structure
            level_count = self.wsi_reader.level_count
            level_dimensions = self.wsi_reader.level_dimensions
            
            if level_count < 2:
                compatibility['compatibility_issues'].append(
                    "Single-level image detected - may impact streaming performance"
                )
                compatibility['recommended_actions'].append(
                    "Consider creating pyramid structure for better streaming performance"
                )
            
            # Check for very large dimensions at level 0
            width, height = level_dimensions[0]
            total_pixels = width * height
            
            if total_pixels > 1e9:  # 1 billion pixels
                compatibility['streaming_optimizations'].append(
                    "Large slide detected - adaptive tile sizing recommended"
                )
            
            # Format-specific checks
            if self._detected_format == 'openslide':
                # Check for known problematic formats
                properties = self.wsi_reader.properties
                vendor = properties.get('openslide.vendor', '').lower()
                
                if 'aperio' in vendor:
                    compatibility['streaming_optimizations'].append(
                        "Aperio format detected - optimized for tile-based access"
                    )
                elif 'hamamatsu' in vendor:
                    compatibility['streaming_optimizations'].append(
                        "Hamamatsu format detected - may benefit from larger tile sizes"
                    )
            
            elif self._detected_format == 'dicom':
                compatibility['streaming_optimizations'].append(
                    "DICOM format detected - ensure wsidicom library is optimized"
                )
            
            # Check for potential memory issues
            estimated_memory_gb = self._estimate_memory_requirements(
                len(self._tile_coordinates) if self._tile_coordinates else 1000,
                self.tile_pool.get_current_tile_size()
            )
            
            if estimated_memory_gb > self.config.max_memory_gb:
                compatibility['compatibility_issues'].append(
                    f"Estimated memory usage ({estimated_memory_gb:.2f}GB) exceeds limit ({self.config.max_memory_gb:.2f}GB)"
                )
                compatibility['recommended_actions'].append(
                    "Enable adaptive tile sizing or increase memory limit"
                )
                compatibility['is_compatible'] = False
        
        except Exception as e:
            logger.warning(f"Error during compatibility validation: {e}")
            compatibility['compatibility_issues'].append(f"Validation error: {str(e)}")
            compatibility['is_compatible'] = False
        
        return compatibility
    
    def _estimate_memory_requirements(self, num_patches: int, tile_size: int) -> float:
        """
        Estimate memory requirements for streaming.
        
        Args:
            num_patches: Number of patches to process
            tile_size: Size of tiles in pixels
            
        Returns:
            Estimated memory usage in GB
        """
        # Memory per tile (RGB, uint8)
        bytes_per_tile = tile_size * tile_size * 3
        
        # Buffer pool memory (for cached tiles)
        buffer_memory = self.config.max_buffer_size * bytes_per_tile
        
        # Processing overhead (temporary arrays, etc.)
        processing_overhead = buffer_memory * 0.5
        
        total_memory_bytes = buffer_memory + processing_overhead
        
        return total_memory_bytes / (1024**3)
    
    def get_adaptive_sizing_stats(self) -> Dict:
        """
        Get adaptive sizing statistics from the tile pool.
        
        Returns:
            Dictionary with adaptive sizing statistics
        """
        return self.tile_pool.get_adaptive_sizing_stats()
    
    def get_format_statistics(self) -> Dict[str, Union[str, int, float, bool]]:
        """
        Get comprehensive format-specific statistics and performance metrics.
        
        Returns:
            Dictionary with format statistics including:
            - Format detection results
            - Performance characteristics
            - Compatibility information
            - Optimization recommendations
        """
        stats = {
            'detected_format': self._detected_format,
            'file_extension': self.wsi_path.suffix.lower(),
            'file_size_mb': self.wsi_path.stat().st_size / (1024 * 1024),
        }
        
        if self._metadata is not None:
            stats.update({
                'slide_dimensions': self._metadata.dimensions,
                'level_count': self._metadata.level_count,
                'estimated_patches': self._metadata.estimated_patches,
                'magnification': self._metadata.magnification,
                'mpp': self._metadata.mpp,
                'compatibility_status': self._metadata.format_compatibility['is_compatible'],
                'compatibility_issues_count': len(self._metadata.format_compatibility['compatibility_issues']),
            })
        
        # Add format-specific performance metrics
        try:
            if self._detected_format == 'openslide':
                properties = self.wsi_reader.properties
                stats.update({
                    'openslide_vendor': properties.get('openslide.vendor', 'unknown'),
                    'openslide_comment': properties.get('openslide.comment', 'none'),
                    'has_associated_images': len(getattr(self.wsi_reader._reader, 'associated_images', {})) > 0,
                })
            elif self._detected_format == 'dicom':
                properties = self.wsi_reader.properties
                stats.update({
                    'dicom_manufacturer': properties.get('manufacturer', 'unknown'),
                    'dicom_model': properties.get('manufacturer_model', 'unknown'),
                    'dicom_patient_id_available': 'patient_id' in properties,
                })
        except Exception as e:
            logger.warning(f"Failed to extract format-specific statistics: {e}")
        
        # Add streaming performance recommendations
        stats['streaming_recommendations'] = self._get_streaming_recommendations()
        
        return stats
    
    def _get_streaming_recommendations(self) -> List[str]:
        """
        Get format-specific streaming performance recommendations.
        
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        try:
            if self._metadata is None:
                return ["Initialize streaming to get recommendations"]
            
            # Format-specific recommendations
            if self._detected_format == 'openslide':
                properties = self.wsi_reader.properties
                vendor = properties.get('openslide.vendor', '').lower()
                
                if 'aperio' in vendor:
                    recommendations.append("Aperio format: Use tile sizes 256-1024px for optimal performance")
                elif 'hamamatsu' in vendor:
                    recommendations.append("Hamamatsu format: Consider larger tile sizes (1024-2048px)")
                elif 'leica' in vendor:
                    recommendations.append("Leica format: Standard tile sizes work well")
                
                # Check for pyramid structure
                if self._metadata.level_count < 3:
                    recommendations.append("Limited pyramid levels: Consider enabling adaptive tile sizing")
            
            elif self._detected_format == 'dicom':
                recommendations.append("DICOM format: Ensure wsidicom library is up to date")
                recommendations.append("DICOM format: Monitor memory usage closely")
                
                # Check for large dimensions
                width, height = self._metadata.dimensions
                if width * height > 5e8:  # 500M pixels
                    recommendations.append("Large DICOM slide: Use smaller initial tile sizes")
            
            # General recommendations based on slide characteristics
            total_pixels = self._metadata.dimensions[0] * self._metadata.dimensions[1]
            if total_pixels > 1e9:  # 1B pixels
                recommendations.append("Very large slide: Enable adaptive tile sizing and memory monitoring")
            
            if self._metadata.estimated_patches > 50000:
                recommendations.append("High patch count: Consider early stopping with confidence thresholds")
            
            # Memory-based recommendations
            if self._metadata.memory_budget_gb < 2.0:
                recommendations.append("Limited memory: Use smaller tile sizes and enable aggressive cleanup")
            elif self._metadata.memory_budget_gb > 8.0:
                recommendations.append("Ample memory: Can use larger tile sizes for better performance")
        
        except Exception as e:
            logger.warning(f"Failed to generate streaming recommendations: {e}")
            recommendations.append("Error generating recommendations - check logs")
        
        return recommendations if recommendations else ["No specific recommendations for this format"]
    
    def close(self) -> None:
        """Close the reader and release resources."""
        # Finish progress tracking if active
        if self._progress_tracker is not None:
            try:
                final_progress = self._progress_tracker.finish_processing()
                logger.info(
                    f"WSI streaming session completed: "
                    f"{final_progress.tiles_processed} tiles processed, "
                    f"final confidence: {final_progress.current_confidence:.3f}"
                )
            except Exception as e:
                logger.warning(f"Error finishing progress tracking: {e}")
        
        if hasattr(self, 'wsi_reader'):
            self.wsi_reader.close()
        
        if hasattr(self, 'tile_pool'):
            self.tile_pool.clear()
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
    
    def __del__(self):
        """Cleanup on deletion."""
        try:
            self.close()
        except Exception:
            pass  # Ignore errors during cleanup