"""Configuration management for memory optimization.

This module provides configuration dataclasses for memory optimization components.

Classes:
    OptimizerConfig: Configuration for memory optimization system

Example:
    >>> config = OptimizerConfig(
    ...     cache_size_mb=1000,
    ...     memory_limit_gb=8.0,
    ...     enable_monitoring=True
    ... )
    >>> coordinator = MemoryCoordinator(config)
"""

from dataclasses import dataclass, field
from typing import Optional

import torch


@dataclass
class OptimizerConfig:
    """Configuration for memory optimization system.

    Attributes:
        device: Target device for memory management
        memory_limit_gb: Memory limit in GB
        cache_size_mb: Cache size in MB
        enable_monitoring: Enable real-time monitoring
        enable_profiling: Enable memory profiling
        sampling_interval_ms: Monitoring sampling interval in ms
        alert_threshold_percent: Alert threshold as percentage of limit
        batch_size_range: Tuple of (min, max) batch sizes
        tile_size_range: Tuple of (min, max) tile sizes
    """

    device: torch.device = field(default_factory=lambda: torch.device("cpu"))
    memory_limit_gb: float = 8.0
    cache_size_mb: float = 1000.0
    enable_monitoring: bool = True
    enable_profiling: bool = True
    sampling_interval_ms: float = 100.0
    alert_threshold_percent: float = 90.0
    batch_size_range: tuple = (1, 64)
    tile_size_range: tuple = (224, 512)

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.memory_limit_gb <= 0:
            raise ValueError("memory_limit_gb must be positive")

        if self.cache_size_mb <= 0:
            raise ValueError("cache_size_mb must be positive")

        if not 0 < self.alert_threshold_percent <= 100:
            raise ValueError("alert_threshold_percent must be between 0 and 100")

        if self.sampling_interval_ms <= 0:
            raise ValueError("sampling_interval_ms must be positive")

        min_batch, max_batch = self.batch_size_range
        if min_batch <= 0 or max_batch < min_batch:
            raise ValueError("Invalid batch_size_range")

        min_tile, max_tile = self.tile_size_range
        if min_tile <= 0 or max_tile < min_tile:
            raise ValueError("Invalid tile_size_range")

    @classmethod
    def from_dict(cls, config_dict: dict) -> "OptimizerConfig":
        """Create config from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            OptimizerConfig instance
        """
        # Handle device string conversion
        if "device" in config_dict and isinstance(config_dict["device"], str):
            config_dict["device"] = torch.device(config_dict["device"])

        return cls(**config_dict)

    def to_dict(self) -> dict:
        """Convert config to dictionary.

        Returns:
            Configuration dictionary
        """
        return {
            "device": str(self.device),
            "memory_limit_gb": self.memory_limit_gb,
            "cache_size_mb": self.cache_size_mb,
            "enable_monitoring": self.enable_monitoring,
            "enable_profiling": self.enable_profiling,
            "sampling_interval_ms": self.sampling_interval_ms,
            "alert_threshold_percent": self.alert_threshold_percent,
            "batch_size_range": self.batch_size_range,
            "tile_size_range": self.tile_size_range,
        }
