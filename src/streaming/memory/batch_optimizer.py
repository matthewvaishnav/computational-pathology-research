"""Batch and tile size optimization for WSI streaming.

Optimize batch/tile sizes based on available memory to maximize throughput
while avoiding OOM.
"""

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class OptimalSizes:
    """Optimal batch and tile sizes."""

    batch_size: int
    tile_size: int
    estimated_memory_gb: float
    throughput_estimate: float  # patches/sec


class BatchOptimizer:
    """Optimize batch and tile sizes for memory constraints.

    Features:
    - Calculate optimal batch size for available memory
    - Calculate optimal tile size for slide characteristics
    - Balance memory usage vs throughput
    - Adaptive sizing based on runtime feedback
    """

    def __init__(
        self,
        available_memory_gb: float = 2.0,
        safety_margin: float = 0.8,
        min_batch_size: int = 1,
        max_batch_size: int = 64,
    ):
        """Init batch optimizer.

        Args:
            available_memory_gb: Available memory in GB
            safety_margin: Use only this fraction of available memory (0-1)
            min_batch_size: Minimum batch size
            max_batch_size: Maximum batch size
        """
        self.available_memory_gb = available_memory_gb
        self.safety_margin = safety_margin
        self.min_batch_size = min_batch_size
        self.max_batch_size = max_batch_size

        # Memory overhead estimates (GB)
        self.base_overhead_gb = 0.5
        self.per_patch_overhead_mb = 0.1

        # Tile size options (pixels)
        self.tile_size_options = [96, 224, 256, 512]

        logger.info(
            f"BatchOptimizer init: {available_memory_gb:.2f}GB, " f"margin={safety_margin:.2f}"
        )

    def optimize_batch_size(
        self,
        tile_size: int = 224,
        feature_dim: int = 512,
        num_patches: int = 1000,
    ) -> int:
        """Calculate optimal batch size.

        Args:
            tile_size: Tile size in pixels
            feature_dim: Feature dimension
            num_patches: Number of patches to process

        Returns:
            Optimal batch size
        """
        # Memory per patch (MB)
        # Tile: channels * tile_size^2 * bytes_per_element
        tile_memory_mb = (3 * tile_size * tile_size * 4) / (1024 * 1024)

        # Feature: feature_dim * bytes_per_element
        feature_memory_mb = (feature_dim * 4) / (1024 * 1024)

        # Total per patch
        per_patch_mb = tile_memory_mb + feature_memory_mb + self.per_patch_overhead_mb

        # Available memory after base overhead
        available_mb = (
            self.available_memory_gb * self.safety_margin - self.base_overhead_gb
        ) * 1024

        # Calculate batch size
        batch_size = int(available_mb / per_patch_mb)

        # Clamp to valid range
        batch_size = max(self.min_batch_size, min(batch_size, self.max_batch_size))

        # Round down to power of 2 for efficiency
        batch_size = 2 ** int(batch_size.bit_length() - 1) if batch_size > 0 else 1

        logger.debug(
            f"Optimal batch: {batch_size} (tile={tile_size}, "
            f"feat={feature_dim}, mem={per_patch_mb:.2f}MB/patch)"
        )

        return batch_size

    def optimize_tile_size(
        self,
        slide_dimensions: Tuple[int, int],
        magnification: float = 20.0,
        target_mpp: float = 0.5,
    ) -> int:
        """Calculate optimal tile size.

        Args:
            slide_dimensions: (width, height) in pixels
            magnification: Slide magnification
            target_mpp: Target microns per pixel

        Returns:
            Optimal tile size in pixels
        """
        width, height = slide_dimensions

        # Estimate slide MPP from magnification
        # Rough approximation: 40x ≈ 0.25 MPP, 20x ≈ 0.5 MPP
        slide_mpp = 10.0 / magnification

        # Calculate downsample factor
        downsample = target_mpp / slide_mpp if slide_mpp > 0 else 1.0

        # Prefer larger tiles for high-res slides
        if downsample < 1.5:
            preferred_size = 512
        elif downsample < 3.0:
            preferred_size = 256
        else:
            preferred_size = 224

        # Find closest available tile size
        tile_size = min(self.tile_size_options, key=lambda x: abs(x - preferred_size))

        logger.debug(
            f"Optimal tile: {tile_size}px (mag={magnification}x, "
            f"mpp={slide_mpp:.3f}, downsample={downsample:.2f})"
        )

        return tile_size

    def optimize_for_workload(
        self,
        slide_dimensions: Tuple[int, int] = (50000, 50000),
        magnification: float = 20.0,
        target_mpp: float = 0.5,
        feature_dim: int = 512,
        estimated_patches: Optional[int] = None,
    ) -> OptimalSizes:
        """Optimize both batch and tile sizes for workload.

        Args:
            slide_dimensions: (width, height) in pixels
            magnification: Slide magnification
            target_mpp: Target microns per pixel
            feature_dim: Feature dimension
            estimated_patches: Estimated number of patches (auto-calc if None)

        Returns:
            Optimal sizes with estimates
        """
        # Optimize tile size
        tile_size = self.optimize_tile_size(slide_dimensions, magnification, target_mpp)

        # Estimate patches if not provided
        if estimated_patches is None:
            width, height = slide_dimensions
            # Assume 50% tissue coverage, 50% overlap
            patches_per_row = (width // tile_size) * 2
            patches_per_col = (height // tile_size) * 2
            estimated_patches = int(patches_per_row * patches_per_col * 0.5)

        # Optimize batch size
        batch_size = self.optimize_batch_size(tile_size, feature_dim, estimated_patches)

        # Estimate memory usage
        tile_memory_gb = (batch_size * 3 * tile_size * tile_size * 4) / (1024**3)
        feature_memory_gb = (estimated_patches * feature_dim * 4) / (1024**3)
        estimated_memory_gb = self.base_overhead_gb + tile_memory_gb + feature_memory_gb

        # Estimate throughput (rough)
        # Assume 10ms per patch at batch_size=1, scales sublinearly
        base_time_per_patch = 0.01  # seconds
        batch_efficiency = 0.7  # 70% efficiency from batching
        time_per_batch = base_time_per_patch * batch_size * (1.0 - batch_efficiency)
        throughput_estimate = batch_size / time_per_batch if time_per_batch > 0 else 0.0

        result = OptimalSizes(
            batch_size=batch_size,
            tile_size=tile_size,
            estimated_memory_gb=estimated_memory_gb,
            throughput_estimate=throughput_estimate,
        )

        logger.info(
            f"Optimized: batch={batch_size}, tile={tile_size}px, "
            f"mem={estimated_memory_gb:.2f}GB, throughput={throughput_estimate:.1f} patches/s"
        )

        return result

    def adjust_for_oom(self, current_batch_size: int, current_tile_size: int) -> Tuple[int, int]:
        """Adjust sizes after OOM event.

        Args:
            current_batch_size: Current batch size
            current_tile_size: Current tile size

        Returns:
            (new_batch_size, new_tile_size)
        """
        # Reduce batch size first (more effective)
        new_batch_size = max(self.min_batch_size, current_batch_size // 2)

        # If batch already at minimum, reduce tile size
        new_tile_size = current_tile_size
        if new_batch_size == self.min_batch_size and current_tile_size > min(
            self.tile_size_options
        ):
            # Find next smaller tile size
            smaller_sizes = [s for s in self.tile_size_options if s < current_tile_size]
            new_tile_size = max(smaller_sizes) if smaller_sizes else current_tile_size

        logger.warning(
            f"OOM adjust: batch {current_batch_size}→{new_batch_size}, "
            f"tile {current_tile_size}→{new_tile_size}"
        )

        return new_batch_size, new_tile_size

    def get_memory_estimate(
        self, batch_size: int, tile_size: int, feature_dim: int, num_patches: int
    ) -> float:
        """Estimate memory usage for config.

        Args:
            batch_size: Batch size
            tile_size: Tile size in pixels
            feature_dim: Feature dimension
            num_patches: Number of patches

        Returns:
            Estimated memory in GB
        """
        tile_memory_gb = (batch_size * 3 * tile_size * tile_size * 4) / (1024**3)
        feature_memory_gb = (num_patches * feature_dim * 4) / (1024**3)
        total_gb = self.base_overhead_gb + tile_memory_gb + feature_memory_gb

        return total_gb

    def __repr__(self) -> str:
        """String repr."""
        return (
            f"BatchOptimizer(mem={self.available_memory_gb:.2f}GB, "
            f"margin={self.safety_margin:.2f}, "
            f"batch_range=[{self.min_batch_size}, {self.max_batch_size}])"
        )
