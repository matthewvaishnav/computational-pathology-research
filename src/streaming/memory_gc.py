"""Smart garbage collection for memory optimization."""

import gc
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict

import torch

logger = logging.getLogger(__name__)


@dataclass
class GCStats:
    """Garbage collection statistics."""

    collections_triggered: int
    memory_freed_gb: float
    avg_collection_time_ms: float
    last_collection_time: float
    blocks_collected: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "collections_triggered": self.collections_triggered,
            "memory_freed_gb": self.memory_freed_gb,
            "avg_collection_time_ms": self.avg_collection_time_ms,
            "last_collection_time": self.last_collection_time,
            "blocks_collected": self.blocks_collected,
        }


class SmartGarbageCollector:
    """Smart garbage collection with adaptive strategies.

    Features:
    - Pressure-based collection triggers
    - Generational collection strategies
    - Adaptive collection thresholds
    - Performance-aware scheduling
    """

    def __init__(
        self,
        device: torch.device,
        memory_pressure_threshold: float = 0.8,
        collection_interval_seconds: float = 10.0,
        enable_adaptive: bool = True,
    ):
        """Initialize smart garbage collector.

        Args:
            device: Target device
            memory_pressure_threshold: Trigger collection above this memory usage
            collection_interval_seconds: Minimum time between collections
            enable_adaptive: Enable adaptive threshold adjustment
        """
        self.device = device
        self.memory_pressure_threshold = memory_pressure_threshold
        self.collection_interval_seconds = collection_interval_seconds
        self.enable_adaptive = enable_adaptive

        # Statistics
        self.collections_triggered = 0
        self.total_memory_freed_gb = 0.0
        self.collection_times = deque(maxlen=100)
        self.last_collection_time = 0.0

        # Adaptive thresholds
        self.min_threshold = 0.6
        self.max_threshold = 0.95
        self.threshold_adjustment_rate = 0.05

        logger.info(
            f"SmartGarbageCollector initialized: threshold={memory_pressure_threshold:.2f}, "
            f"interval={collection_interval_seconds}s"
        )

    def should_collect(self, current_memory_gb: float, total_memory_gb: float) -> bool:
        """Determine if garbage collection should be triggered.

        Args:
            current_memory_gb: Current memory usage in GB
            total_memory_gb: Total available memory in GB

        Returns:
            True if collection should be triggered
        """
        # Check time since last collection
        time_since_last = time.time() - self.last_collection_time
        if time_since_last < self.collection_interval_seconds:
            return False

        # Check memory pressure
        if total_memory_gb == 0:
            return False

        memory_pressure = current_memory_gb / total_memory_gb

        return memory_pressure >= self.memory_pressure_threshold

    def collect(self, aggressive: bool = False) -> float:
        """Perform garbage collection.

        Args:
            aggressive: If True, perform aggressive collection

        Returns:
            Memory freed in GB
        """
        start_time = time.time()

        # Get memory before collection
        if self.device.type == "cuda":
            memory_before = torch.cuda.memory_allocated(self.device) / (1024**3)
        else:
            memory_before = 0.0

        # Python garbage collection
        if aggressive:
            # Aggressive: collect all generations
            gc.collect(2)
        else:
            # Normal: collect generation 0
            gc.collect(0)

        # CUDA cache cleanup
        if self.device.type == "cuda":
            torch.cuda.empty_cache()
            if aggressive:
                torch.cuda.synchronize()

        # Get memory after collection
        if self.device.type == "cuda":
            memory_after = torch.cuda.memory_allocated(self.device) / (1024**3)
        else:
            memory_after = 0.0

        memory_freed = max(0.0, memory_before - memory_after)

        # Update statistics
        collection_time = time.time() - start_time
        self.collections_triggered += 1
        self.total_memory_freed_gb += memory_freed
        self.collection_times.append(collection_time * 1000)  # Convert to ms
        self.last_collection_time = time.time()

        # Adaptive threshold adjustment
        if self.enable_adaptive:
            self._adjust_threshold(memory_freed, collection_time)

        logger.debug(f"GC completed: freed {memory_freed:.3f}GB in {collection_time*1000:.1f}ms")

        return memory_freed

    def _adjust_threshold(self, memory_freed: float, collection_time: float):
        """Adjust collection threshold based on effectiveness.

        Args:
            memory_freed: Memory freed by last collection
            collection_time: Time taken for collection
        """
        # If collection freed significant memory, we can be more conservative
        if memory_freed > 0.5:  # Freed >500MB
            self.memory_pressure_threshold = min(
                self.max_threshold, self.memory_pressure_threshold + self.threshold_adjustment_rate
            )

        # If collection freed little memory, be more aggressive
        elif memory_freed < 0.1:  # Freed <100MB
            self.memory_pressure_threshold = max(
                self.min_threshold, self.memory_pressure_threshold - self.threshold_adjustment_rate
            )

        # If collection took too long, be more conservative
        if collection_time > 0.5:  # >500ms
            self.memory_pressure_threshold = min(
                self.max_threshold, self.memory_pressure_threshold + self.threshold_adjustment_rate
            )

    def get_stats(self) -> GCStats:
        """Get garbage collection statistics."""
        import numpy as np

        avg_time_ms = np.mean(self.collection_times) if self.collection_times else 0.0

        return GCStats(
            collections_triggered=self.collections_triggered,
            memory_freed_gb=self.total_memory_freed_gb,
            avg_collection_time_ms=avg_time_ms,
            last_collection_time=self.last_collection_time,
            blocks_collected=self.collections_triggered,  # Simplified
        )
