"""
Progress tracking for WSI streaming operations.

This module provides comprehensive progress tracking, ETA estimation, and
performance monitoring for WSI streaming with support for confidence-based
early stopping and real-time callbacks.
"""

import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
import psutil

try:
    import torch

    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class StreamingProgress:
    """Progress information for streaming with comprehensive tracking."""

    tiles_processed: int
    total_tiles: int
    progress_ratio: float
    elapsed_time: float
    estimated_time_remaining: float
    estimated_total_time: float
    current_tile_size: int
    memory_usage_gb: float
    throughput_tiles_per_second: float
    average_processing_time_per_tile: float
    current_confidence: float
    confidence_delta: float
    early_stop_recommended: bool
    confidence_threshold: float
    current_stage: str
    stage_progress: float
    time_spent_loading: float
    time_spent_processing: float
    time_spent_aggregating: float
    peak_memory_usage_gb: float
    gpu_memory_usage_gb: float
    cpu_utilization_percent: float
    tiles_skipped: int
    tiles_failed: int
    data_quality_score: float


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
        progress_callbacks: Optional[List[ProgressCallback]] = None,
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
        self.processing_times: deque = deque(maxlen=100)
        self.throughput_history: deque = deque(maxlen=50)
        self.memory_usage_history: deque = deque(maxlen=100)

        # Confidence tracking
        self.confidence_history: List[Tuple[float, float]] = []
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
        """Start a new processing stage."""
        current_time = time.time()

        if self.current_stage in self.stage_start_times:
            stage_duration = current_time - self.stage_start_times[self.current_stage]
            self.stage_durations[self.current_stage] = stage_duration

        self.current_stage = stage_name
        self.stage_start_times[stage_name] = current_time
        self.stage_progress = 0.0
        logger.debug(f"Started processing stage: {stage_name}")

    def update_stage_progress(self, progress: float) -> None:
        """Update progress within current stage."""
        self.stage_progress = max(0.0, min(1.0, progress))

    def record_tile_processed(
        self, processing_time: float, tile_size: int, success: bool = True, skipped: bool = False
    ) -> None:
        """Record processing of a tile."""
        if success and not skipped:
            self.tiles_processed += 1
            self.processing_times.append(processing_time)
            if processing_time > 0:
                throughput = 1.0 / processing_time
                self.throughput_history.append(throughput)
        elif skipped:
            self.tiles_skipped += 1
        else:
            self.tiles_failed += 1

        current_memory = self._get_current_memory_usage()
        self.memory_usage_history.append(current_memory)
        self.peak_memory_usage_gb = max(self.peak_memory_usage_gb, current_memory)

    def update_confidence(self, confidence: float) -> None:
        """Update confidence tracking."""
        current_time = time.time()

        if self.confidence_history:
            self.confidence_delta = confidence - self.current_confidence
        else:
            self.confidence_delta = 0.0

        self.current_confidence = confidence
        self.confidence_history.append((current_time, confidence))
        self._update_early_stopping_recommendation()

    def _update_early_stopping_recommendation(self) -> None:
        """Update early stopping recommendation based on confidence and time."""
        confidence_met = self.current_confidence >= self.confidence_threshold

        confidence_stable = False
        if len(self.confidence_history) >= 10:
            recent_confidences = [c for _, c in self.confidence_history[-10:]]
            confidence_variance = np.var(recent_confidences)
            confidence_stable = confidence_variance < 0.001

        time_pressure = False
        if self.start_time is not None:
            elapsed_time = time.time() - self.start_time
            time_pressure = elapsed_time > (self.target_processing_time * 0.8)

        self.early_stop_recommended = confidence_met or (
            self.current_confidence > 0.9 and confidence_stable and time_pressure
        )

    def _get_current_memory_usage(self) -> float:
        """Get current memory usage in GB."""
        try:
            process = psutil.Process()
            memory_info = process.memory_info()
            return memory_info.rss / (1024**3)
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
        """Calculate estimated time remaining and total time."""
        if self.start_time is None or self.tiles_processed == 0:
            return 0.0, 0.0

        elapsed_time = time.time() - self.start_time

        if len(self.processing_times) >= 10:
            recent_times = list(self.processing_times)[-20:]
            avg_time_per_tile = np.mean(recent_times)
        else:
            avg_time_per_tile = elapsed_time / self.tiles_processed

        effective_remaining_tiles = self.total_tiles - self.tiles_processed
        if self.early_stop_recommended and self.current_confidence > 0.9:
            confidence_factor = min(1.0, self.current_confidence / self.confidence_threshold)
            early_stop_factor = 0.5 + 0.5 * (1.0 - confidence_factor)
            effective_remaining_tiles = int(effective_remaining_tiles * early_stop_factor)

        estimated_time_remaining = effective_remaining_tiles * avg_time_per_tile
        estimated_total_time = elapsed_time + estimated_time_remaining

        return estimated_time_remaining, estimated_total_time

    def get_current_progress(self) -> StreamingProgress:
        """Get current progress information."""
        if self.start_time is None:
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

        estimated_time_remaining, estimated_total_time = self._calculate_eta()

        avg_throughput = np.mean(self.throughput_history) if self.throughput_history else 0.0
        avg_processing_time = np.mean(self.processing_times) if self.processing_times else 0.0

        total_tiles_attempted = self.tiles_processed + self.tiles_failed + self.tiles_skipped
        if total_tiles_attempted > 0:
            data_quality_score = self.tiles_processed / total_tiles_attempted
        else:
            data_quality_score = 1.0

        current_memory = self._get_current_memory_usage()
        gpu_memory = self._get_gpu_memory_usage()
        cpu_utilization = self._get_cpu_utilization()

        time_spent_loading = self.stage_durations.get("streaming", 0.0)
        time_spent_processing = self.stage_durations.get("processing", 0.0)
        time_spent_aggregating = self.stage_durations.get("aggregating", 0.0)

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
            current_tile_size=0,
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

        self._trigger_progress_callbacks(progress)
        return progress

    def _trigger_progress_callbacks(self, progress: StreamingProgress) -> None:
        """Trigger progress callbacks if update conditions are met."""
        current_time = time.time()

        for callback_config in self.progress_callbacks:
            time_condition = (
                current_time - self.last_callback_time
            ) >= callback_config.update_interval

            progress_condition = (
                abs(progress.progress_ratio - self.last_callback_progress)
                >= callback_config.min_progress_delta
            )

            if time_condition or progress_condition:
                try:
                    callback_config.callback_func(progress)
                    self.last_callback_time = current_time
                    self.last_callback_progress = progress.progress_ratio
                except Exception as e:
                    logger.warning(f"Progress callback failed: {e}")

    def finish_processing(self) -> StreamingProgress:
        """Finish progress tracking and return final statistics."""
        if self.current_stage in self.stage_start_times:
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
