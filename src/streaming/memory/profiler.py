"""Memory profiling for WSI streaming.

This module provides memory profiling capabilities for tracking CPU and GPU
memory usage during whole slide image (WSI) processing.

Components:
    - MemorySnapshot: Point-in-time memory usage snapshot
    - MemoryProfiler: Profile and track memory usage over time

Usage:
    from streaming.memory.profiler import MemoryProfiler, MemorySnapshot

    profiler = MemoryProfiler(device=torch.device("cuda"))
    snapshot = profiler.take_snapshot()
    print(f"GPU Memory: {snapshot.allocated_gb:.2f}GB")

    # Track peak usage
    peak = profiler.get_peak_usage()
    print(f"Peak GPU: {peak['gpu_gb']:.2f}GB")
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List

import torch

logger = logging.getLogger(__name__)


# ============================================================================
# Enums
# ============================================================================


class MemoryPressureLevel(Enum):
    """Memory pressure levels."""

    NORMAL = "normal"  # < 60% usage
    MODERATE = "moderate"  # 60-75% usage
    HIGH = "high"  # 75-90% usage
    CRITICAL = "critical"  # > 90% usage


# ============================================================================
# Data Models
# ============================================================================


@dataclass
class MemorySnapshot:
    """Point-in-time memory usage snapshot."""

    timestamp: float
    allocated_gb: float
    reserved_gb: float
    total_gb: float
    pressure_level: MemoryPressureLevel

    @property
    def utilization_percent(self) -> float:
        """Calculate memory utilization percentage."""
        if self.total_gb == 0:
            return 0.0
        return (self.allocated_gb / self.total_gb) * 100.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "timestamp": self.timestamp,
            "allocated_gb": self.allocated_gb,
            "reserved_gb": self.reserved_gb,
            "total_gb": self.total_gb,
            "utilization_percent": self.utilization_percent,
            "pressure_level": self.pressure_level.value,
        }


# ============================================================================
# Memory Profiler
# ============================================================================


class MemoryProfiler:
    """Profile and track memory usage for CPU and GPU.

    Features:
    - Real-time memory usage tracking
    - Peak usage tracking
    - Memory pressure calculation
    - Historical snapshot storage
    - Support for both CPU and GPU devices

    Example:
        >>> profiler = MemoryProfiler(device=torch.device("cuda"))
        >>> snapshot = profiler.take_snapshot()
        >>> print(f"Memory: {snapshot.allocated_gb:.2f}GB")
        >>> peak = profiler.get_peak_usage()
    """

    def __init__(
        self,
        device: torch.device,
        memory_limit_gb: float = 2.0,
        max_snapshots: int = 1000,
    ):
        """Initialize memory profiler.

        Args:
            device: Target device to profile (CPU or CUDA)
            memory_limit_gb: Memory limit in GB for pressure calculation
            max_snapshots: Maximum number of snapshots to keep in history
        """
        self.device = device
        self.memory_limit_gb = memory_limit_gb
        self.max_snapshots = max_snapshots

        # Get total device memory
        if device.type == "cuda":
            self.total_memory_gb = torch.cuda.get_device_properties(device).total_memory / (1024**3)
        else:
            self.total_memory_gb = memory_limit_gb

        # Snapshot history
        self.snapshots: List[MemorySnapshot] = []

        # Peak usage tracking
        self.peak_allocated_gb = 0.0
        self.peak_reserved_gb = 0.0

        # Pressure thresholds
        self.pressure_thresholds = {
            MemoryPressureLevel.NORMAL: 0.60,
            MemoryPressureLevel.MODERATE: 0.75,
            MemoryPressureLevel.HIGH: 0.90,
            MemoryPressureLevel.CRITICAL: 0.95,
        }

        logger.info(
            f"MemoryProfiler initialized: device={device}, "
            f"total={self.total_memory_gb:.2f}GB, limit={memory_limit_gb:.2f}GB"
        )

    def _get_current_memory_usage(self) -> tuple[float, float]:
        """Get current memory usage.

        Returns:
            Tuple of (allocated_gb, reserved_gb)
        """
        if self.device.type == "cuda":
            allocated = torch.cuda.memory_allocated(self.device) / (1024**3)
            reserved = torch.cuda.memory_reserved(self.device) / (1024**3)
            return allocated, reserved
        else:
            # For CPU, return zeros (CPU memory tracking not implemented)
            return 0.0, 0.0

    def _calculate_pressure_level(self, allocated_gb: float) -> MemoryPressureLevel:
        """Calculate memory pressure level.

        Args:
            allocated_gb: Current allocated memory in GB

        Returns:
            Memory pressure level
        """
        if self.memory_limit_gb == 0:
            return MemoryPressureLevel.NORMAL

        utilization = allocated_gb / self.memory_limit_gb

        # Check thresholds from highest to lowest
        if utilization >= self.pressure_thresholds[MemoryPressureLevel.CRITICAL]:
            return MemoryPressureLevel.CRITICAL
        elif utilization >= self.pressure_thresholds[MemoryPressureLevel.HIGH]:
            return MemoryPressureLevel.HIGH
        elif utilization >= self.pressure_thresholds[MemoryPressureLevel.MODERATE]:
            return MemoryPressureLevel.MODERATE
        else:
            return MemoryPressureLevel.NORMAL

    def take_snapshot(self) -> MemorySnapshot:
        """Take a memory usage snapshot.

        Returns:
            Memory snapshot with current usage
        """
        allocated_gb, reserved_gb = self._get_current_memory_usage()
        pressure_level = self._calculate_pressure_level(allocated_gb)

        snapshot = MemorySnapshot(
            timestamp=time.time(),
            allocated_gb=allocated_gb,
            reserved_gb=reserved_gb,
            total_gb=self.total_memory_gb,
            pressure_level=pressure_level,
        )

        # Update peak usage
        if allocated_gb > self.peak_allocated_gb:
            self.peak_allocated_gb = allocated_gb
        if reserved_gb > self.peak_reserved_gb:
            self.peak_reserved_gb = reserved_gb

        # Store snapshot (maintain max size)
        self.snapshots.append(snapshot)
        if len(self.snapshots) > self.max_snapshots:
            self.snapshots.pop(0)

        return snapshot

    def get_peak_usage(self) -> Dict[str, float]:
        """Get peak memory usage.

        Returns:
            Dictionary with peak allocated and reserved memory in GB
        """
        return {
            "allocated_gb": self.peak_allocated_gb,
            "reserved_gb": self.peak_reserved_gb,
        }

    def get_recent_snapshots(self, count: int = 10) -> List[MemorySnapshot]:
        """Get recent memory snapshots.

        Args:
            count: Number of recent snapshots to return

        Returns:
            List of recent snapshots (most recent last)
        """
        return self.snapshots[-count:]

    def get_average_usage(self) -> Dict[str, float]:
        """Get average memory usage across all snapshots.

        Returns:
            Dictionary with average allocated and reserved memory in GB
        """
        if not self.snapshots:
            return {"allocated_gb": 0.0, "reserved_gb": 0.0}

        avg_allocated = sum(s.allocated_gb for s in self.snapshots) / len(self.snapshots)
        avg_reserved = sum(s.reserved_gb for s in self.snapshots) / len(self.snapshots)

        return {
            "allocated_gb": avg_allocated,
            "reserved_gb": avg_reserved,
        }

    def get_pressure_distribution(self) -> Dict[str, float]:
        """Get distribution of time spent in each pressure level.

        Returns:
            Dictionary mapping pressure level to percentage of time
        """
        if not self.snapshots:
            return {}

        pressure_counts = {}
        for snapshot in self.snapshots:
            level = snapshot.pressure_level.value
            pressure_counts[level] = pressure_counts.get(level, 0) + 1

        total_snapshots = len(self.snapshots)
        return {
            level: (count / total_snapshots) * 100.0 for level, count in pressure_counts.items()
        }

    def reset(self):
        """Reset profiler state and clear history."""
        self.snapshots.clear()
        self.peak_allocated_gb = 0.0
        self.peak_reserved_gb = 0.0
        logger.info("Memory profiler reset")

    def set_pressure_threshold(self, level: MemoryPressureLevel, threshold: float):
        """Set custom pressure threshold.

        Args:
            level: Pressure level to configure
            threshold: Threshold value (0.0 to 1.0)

        Raises:
            ValueError: If threshold is not between 0.0 and 1.0
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError("Threshold must be between 0.0 and 1.0")

        self.pressure_thresholds[level] = threshold
        logger.info(f"Updated pressure threshold: {level.value} = {threshold:.2f}")

    def __repr__(self) -> str:
        """String representation of profiler."""
        return (
            f"MemoryProfiler(device={self.device}, "
            f"total={self.total_memory_gb:.2f}GB, "
            f"snapshots={len(self.snapshots)})"
        )
