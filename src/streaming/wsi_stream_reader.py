"""WSI Stream Reader for progressive tile loading without full slide loading."""

import gc
import logging
import os
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from queue import Empty, Queue
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import openslide
import torch
from openslide import OpenSlide
from openslide.deepzoom import DeepZoomGenerator
from PIL import Image

from .format_handlers import get_supported_wsi_formats, get_wsi_handler, validate_wsi_format

logger = logging.getLogger(__name__)


class SlideWrapper:
    """Wrapper to make non-OpenSlide formats compatible with OpenSlide interface."""

    def __init__(self, slide_obj, handler):
        """Initialize wrapper with slide object and format handler."""
        self.slide_obj = slide_obj
        self.handler = handler
        self._dimensions = handler.get_dimensions(slide_obj)
        self._properties = handler.get_properties(slide_obj)

    @property
    def dimensions(self):
        """Get slide dimensions."""
        return self._dimensions

    @property
    def properties(self):
        """Get slide properties."""
        return self._properties

    def read_region(self, location, level, size):
        """Read a region from the slide."""
        return self.handler.read_region(self.slide_obj, location, level, size)

    def get_best_level_for_downsample(self, downsample):
        """Get best level for given downsample factor."""
        return self.handler.get_best_level_for_downsample(self.slide_obj, downsample)

    def close(self):
        """Close the slide."""
        if hasattr(self.handler, "close"):
            self.handler.close(self.slide_obj)
        elif hasattr(self.slide_obj, "close"):
            self.slide_obj.close()


@dataclass
class StreamingMetadata:
    """Metadata for WSI streaming configuration."""

    slide_id: str
    dimensions: Tuple[int, int]
    estimated_patches: int
    tile_size: int
    memory_budget_gb: float
    target_processing_time: float
    confidence_threshold: float
    magnification: Optional[float] = None
    vendor: Optional[str] = None

    def __post_init__(self):
        """Validate metadata parameters."""
        if self.dimensions[0] <= 0 or self.dimensions[1] <= 0:
            raise ValueError("Dimensions must be positive integers")
        if self.estimated_patches <= 0:
            raise ValueError("Estimated patches must be > 0")
        if not (0.5 <= self.memory_budget_gb <= 32.0):
            raise ValueError("Memory budget must be between 0.5 and 32.0 GB")
        if not (5.0 <= self.target_processing_time <= 300.0):
            raise ValueError("Target processing time must be between 5.0 and 300.0 seconds")


@dataclass
class StreamingProgress:
    """Progress tracking for WSI streaming."""

    patches_processed: int
    total_patches: int
    elapsed_time: float
    estimated_remaining_time: float
    current_confidence: float
    memory_usage_gb: float
    throughput_patches_per_sec: float

    @property
    def progress_percentage(self) -> float:
        """Calculate progress as percentage."""
        if self.total_patches == 0:
            return 0.0
        return min(100.0, (self.patches_processed / self.total_patches) * 100.0)


@dataclass
class TileBatch:
    """Batch of tiles for processing."""

    tiles: torch.Tensor  # [batch_size, channels, height, width]
    coordinates: np.ndarray  # [batch_size, 2] - (x, y) coordinates
    batch_id: int
    total_batches: int
    processing_priority: float = 1.0

    def __post_init__(self):
        """Validate tile batch parameters."""
        if len(self.tiles.shape) != 4:
            raise ValueError("Tiles tensor must have 4 dimensions")
        if self.coordinates.shape[0] != self.tiles.shape[0]:
            raise ValueError("Coordinates must match batch size")
        if self.batch_id > self.total_batches:
            raise ValueError("Batch ID must be <= total batches")
        if not (0.0 <= self.processing_priority <= 1.0):
            raise ValueError("Processing priority must be between 0.0 and 1.0")


class TileBufferPool:
    """Memory-managed tile buffer pool with configurable limits."""

    def __init__(self, max_memory_gb: float = 1.0, tile_size: int = 1024):
        self.max_memory_bytes = int(max_memory_gb * 1024**3)
        self.tile_size = tile_size
        self.current_memory = 0
        self.buffers: Queue = Queue()
        self.lock = threading.Lock()

        # Pre-allocate some buffers
        self._preallocate_buffers()

    def _preallocate_buffers(self):
        """Pre-allocate tile buffers for efficiency."""
        # Estimate bytes per tile (RGB, uint8)
        bytes_per_tile = self.tile_size * self.tile_size * 3
        max_tiles = min(100, self.max_memory_bytes // bytes_per_tile)

        for _ in range(min(10, max_tiles)):  # Start with 10 buffers
            buffer = np.empty((self.tile_size, self.tile_size, 3), dtype=np.uint8)
            self.buffers.put(buffer)
            self.current_memory += buffer.nbytes

    def get_buffer(self) -> Optional[np.ndarray]:
        """Get a buffer from the pool."""
        try:
            return self.buffers.get_nowait()
        except Empty:
            # Try to allocate new buffer if under memory limit
            bytes_per_tile = self.tile_size * self.tile_size * 3
            if self.current_memory + bytes_per_tile <= self.max_memory_bytes:
                with self.lock:
                    buffer = np.empty((self.tile_size, self.tile_size, 3), dtype=np.uint8)
                    self.current_memory += buffer.nbytes
                    return buffer
            return None

    def return_buffer(self, buffer: np.ndarray):
        """Return a buffer to the pool."""
        self.buffers.put(buffer)

    def cleanup(self):
        """Clean up buffer pool."""
        with self.lock:
            while not self.buffers.empty():
                try:
                    self.buffers.get_nowait()
                except Empty:
                    break
            self.current_memory = 0
            gc.collect()


class WSIStreamReader:
    """Progressive WSI tile streaming without loading entire slide into memory.

    Implements Iterator[TileBatch] interface for memory-efficient tile iteration.
    """

    def __init__(
        self,
        wsi_path: str,
        tile_size: int = 1024,
        buffer_size: int = 16,
        overlap: int = 0,
        stride: Optional[int] = None,
    ):
        """Initialize streaming reader with configurable buffer and tiling parameters.

        Args:
            wsi_path: Path to WSI file
            tile_size: Size of tiles to extract
            buffer_size: Number of tiles to buffer in memory
            overlap: Pixel overlap between adjacent tiles (default: 0)
            stride: Stride for tile extraction (default: tile_size - overlap)
        """
        self.wsi_path = Path(wsi_path)
        self.tile_size = tile_size
        self.buffer_size = buffer_size
        self.overlap = overlap
        self.stride = stride if stride is not None else (tile_size - overlap)

        # Validate inputs
        if not self.wsi_path.exists():
            raise FileNotFoundError(f"WSI file not found: {wsi_path}")

        if not validate_wsi_format(str(self.wsi_path)):
            raise ValueError(f"Unsupported WSI format: {self.wsi_path.suffix}")

        if self.overlap < 0 or self.overlap >= tile_size:
            raise ValueError(f"Overlap must be in [0, {tile_size})")

        if self.stride <= 0 or self.stride > tile_size:
            raise ValueError(f"Stride must be in (0, {tile_size}]")

        logger.info(f"Supported formats: {get_supported_wsi_formats()}")

        # Initialize components
        self.slide: Optional[OpenSlide] = None
        self.deepzoom: Optional[DeepZoomGenerator] = None
        self.metadata: Optional[StreamingMetadata] = None
        self.buffer_pool: Optional[TileBufferPool] = None

        # Progress tracking
        self.start_time: Optional[float] = None
        self.patches_processed = 0
        self.total_patches = 0

        # Adaptive parameters
        self.current_tile_size = tile_size
        self.memory_pressure = False

        logger.info(
            f"Initialized WSIStreamReader for {self.wsi_path} (tile_size={tile_size}, overlap={overlap}, stride={self.stride})"
        )

    def initialize_streaming(self) -> StreamingMetadata:
        """Setup streaming with slide metadata."""
        try:
            # Get appropriate format handler
            handler = get_wsi_handler(str(self.wsi_path))
            if not handler:
                raise ValueError(f"Unsupported WSI format: {self.wsi_path.suffix}")

            # Open slide using format handler
            slide_obj = handler.open_slide(str(self.wsi_path))

            # For OpenSlide compatibility, wrap in OpenSlide if needed
            if hasattr(slide_obj, "dimensions"):
                self.slide = slide_obj
            else:
                # For non-OpenSlide formats, create a wrapper
                self.slide = self._create_slide_wrapper(slide_obj, handler)

            # Get slide properties
            dimensions = handler.get_dimensions(slide_obj)
            slide_id = self.wsi_path.stem

            # Extract magnification if available
            properties = handler.get_properties(slide_obj)
            magnification = None
            if "openslide.objective-power" in properties:
                try:
                    magnification = float(properties["openslide.objective-power"])
                except (ValueError, KeyError):
                    pass
            elif "aperio.AppMag" in properties:
                try:
                    magnification = float(properties["aperio.AppMag"])
                except (ValueError, KeyError):
                    pass
            elif "tiff.XResolution" in properties and "tiff.YResolution" in properties:
                # Estimate from resolution (microns per pixel)
                try:
                    x_res = float(properties["tiff.XResolution"])
                    # Common magnifications: 40x ≈ 0.25 µm/px, 20x ≈ 0.5 µm/px
                    if x_res > 0:
                        mpp = 1.0 / x_res  # microns per pixel
                        if mpp < 0.3:
                            magnification = 40.0
                        elif mpp < 0.6:
                            magnification = 20.0
                        elif mpp < 1.2:
                            magnification = 10.0
                except (ValueError, KeyError):
                    pass

            # Get vendor info
            vendor = properties.get("openslide.vendor", properties.get("manufacturer", "unknown"))

            # Log extracted metadata
            logger.info(
                f"Extracted metadata - Dimensions: {dimensions}, Magnification: {magnification}x, Vendor: {vendor}"
            )

            # Create DeepZoom generator for efficient tile access (OpenSlide only)
            if isinstance(self.slide, OpenSlide):
                self.deepzoom = DeepZoomGenerator(
                    self.slide,
                    tile_size=self.tile_size,
                    overlap=self.overlap,  # Use configured overlap
                    limit_bounds=True,
                )
            else:
                # For other formats, we'd need custom tiling logic
                raise NotImplementedError("DeepZoom only supported for OpenSlide formats")

            # Estimate total patches with background filtering
            level = self.deepzoom.level_count - 1  # Highest resolution
            level_dimensions = self.deepzoom.level_dimensions[level]
            tiles_x, tiles_y = self.deepzoom.level_tiles[level]
            raw_patch_count = tiles_x * tiles_y

            # Estimate actual patches after background filtering (typically 30-50% of raw)
            # This is a heuristic - actual count determined during streaming
            background_filter_ratio = 0.4  # Assume 40% are tissue (60% background)
            estimated_patches = int(raw_patch_count * background_filter_ratio)

            logger.info(
                f"Patch estimation - Raw: {raw_patch_count}, Estimated (after filtering): {estimated_patches}"
            )

            # Calculate memory budget (adaptive based on slide size)
            # Formula: base_budget + (patches * bytes_per_patch_estimate)
            base_budget_gb = 0.5  # Minimum 500MB
            bytes_per_patch = self.tile_size * self.tile_size * 3  # RGB uint8
            estimated_memory_gb = base_budget_gb + (estimated_patches * bytes_per_patch) / (1024**3)
            memory_budget_gb = min(2.0, max(0.5, estimated_memory_gb))

            # Calculate optimal buffer size based on memory budget
            # Target: use 50% of budget for buffering
            buffer_memory_gb = memory_budget_gb * 0.5
            optimal_buffer_size = int((buffer_memory_gb * 1024**3) / bytes_per_patch)
            optimal_buffer_size = max(4, min(64, optimal_buffer_size))  # Clamp to [4, 64]

            # Update instance buffer size
            self.buffer_size = optimal_buffer_size

            logger.info(
                f"Buffer optimization - Memory budget: {memory_budget_gb:.2f}GB, Buffer size: {optimal_buffer_size} tiles"
            )

            # Initialize buffer pool
            self.buffer_pool = TileBufferPool(
                max_memory_gb=memory_budget_gb * 0.5,  # 50% for buffer pool
                tile_size=self.tile_size,
            )

            # Create metadata
            self.metadata = StreamingMetadata(
                slide_id=slide_id,
                dimensions=dimensions,
                estimated_patches=estimated_patches,
                tile_size=self.tile_size,
                memory_budget_gb=memory_budget_gb,
                target_processing_time=30.0,  # Default 30 seconds
                confidence_threshold=0.95,
                magnification=magnification,
                vendor=vendor,
            )

            self.total_patches = estimated_patches
            self.start_time = time.time()

            logger.info(
                f"Streaming initialized: {dimensions} pixels, {estimated_patches} patches, format: {self.wsi_path.suffix}"
            )
            return self.metadata

        except Exception as e:
            logger.error(f"Failed to initialize streaming: {e}")
            self._cleanup()
            raise

    def stream_tiles(self, spatial_order: bool = True) -> Iterator[TileBatch]:
        """Yield tile batches for processing with spatial locality optimization.

        Args:
            spatial_order: If True, process tiles in spatial order (row-major) for better
                          attention computation locality. If False, process in arbitrary order.
        """
        if not self.deepzoom or not self.metadata:
            raise RuntimeError("Streaming not initialized. Call initialize_streaming() first.")

        level = self.deepzoom.level_count - 1  # Highest resolution
        tiles_x, tiles_y = self.deepzoom.level_tiles[level]

        batch_tiles = []
        batch_coords = []
        batch_id = 0
        total_batches = (tiles_x * tiles_y + self.buffer_size - 1) // self.buffer_size

        try:
            # Spatial order: row-major traversal (default)
            # This ensures tiles in same batch are spatially close
            # Benefits attention computation with spatial locality
            for y in range(tiles_y):
                for x in range(tiles_x):
                    # Check memory pressure and adapt
                    if self._check_memory_pressure():
                        self._adapt_to_memory_pressure()

                    # Get tile from DeepZoom
                    try:
                        tile_pil = self.deepzoom.get_tile(level, (x, y))

                        # Skip if tile is mostly background
                        if self._is_background_tile(tile_pil):
                            continue

                        # Convert to tensor
                        tile_array = np.array(
                            tile_pil.resize((self.current_tile_size, self.current_tile_size))
                        )
                        if len(tile_array.shape) == 2:  # Grayscale
                            tile_array = np.stack([tile_array] * 3, axis=-1)
                        elif tile_array.shape[2] == 4:  # RGBA
                            tile_array = tile_array[:, :, :3]  # Drop alpha

                        # Normalize to [0, 1]
                        tile_tensor = torch.from_numpy(tile_array).float() / 255.0
                        tile_tensor = tile_tensor.permute(2, 0, 1)  # HWC -> CHW

                        batch_tiles.append(tile_tensor)
                        batch_coords.append([x, y])

                        # Yield batch when full
                        if len(batch_tiles) >= self.buffer_size:
                            yield self._create_tile_batch(
                                batch_tiles, batch_coords, batch_id, total_batches
                            )
                            batch_tiles = []
                            batch_coords = []
                            batch_id += 1

                    except Exception as e:
                        logger.warning(f"Failed to process tile ({x}, {y}): {e}")
                        continue

            # Yield remaining tiles
            if batch_tiles:
                yield self._create_tile_batch(batch_tiles, batch_coords, batch_id, total_batches)

        except Exception as e:
            logger.error(f"Error during tile streaming: {e}")
            raise
        finally:
            self._cleanup()

    def _create_tile_batch(
        self, tiles: List[torch.Tensor], coords: List[List[int]], batch_id: int, total_batches: int
    ) -> TileBatch:
        """Create a TileBatch from accumulated tiles."""
        tiles_tensor = torch.stack(tiles)
        coords_array = np.array(coords)

        # Update progress
        self.patches_processed += len(tiles)

        return TileBatch(
            tiles=tiles_tensor,
            coordinates=coords_array,
            batch_id=batch_id,
            total_batches=total_batches,
            processing_priority=1.0,
        )

    def _is_background_tile(self, tile_pil: Image.Image, threshold: float = 0.8) -> bool:
        """Check if tile is mostly background (white/empty)."""
        # Convert to grayscale and check if mostly white
        gray = tile_pil.convert("L")
        gray_array = np.array(gray)
        white_ratio = np.sum(gray_array > 240) / gray_array.size
        return white_ratio > threshold

    def _check_memory_pressure(self) -> bool:
        """Check if system is under memory pressure."""
        try:
            import psutil

            memory = psutil.virtual_memory()
            self.memory_pressure = memory.percent > 80.0
            return self.memory_pressure
        except ImportError:
            # Fallback: assume no pressure if psutil not available
            return False

    def _adapt_to_memory_pressure(self):
        """Adapt tile size and buffer size under memory pressure."""
        if self.memory_pressure:
            # Reduce tile size by 25%
            self.current_tile_size = max(256, int(self.current_tile_size * 0.75))
            # Reduce buffer size
            self.buffer_size = max(4, self.buffer_size // 2)
            logger.info(
                f"Adapted to memory pressure: tile_size={self.current_tile_size}, buffer_size={self.buffer_size}"
            )

    def get_progress(self) -> StreamingProgress:
        """Get current streaming progress."""
        if not self.start_time:
            return StreamingProgress(0, 0, 0.0, 0.0, 0.0, 0.0, 0.0)

        elapsed_time = time.time() - self.start_time

        # Calculate throughput
        throughput = self.patches_processed / elapsed_time if elapsed_time > 0 else 0.0

        # Estimate remaining time
        remaining_patches = max(0, self.total_patches - self.patches_processed)
        estimated_remaining = remaining_patches / throughput if throughput > 0 else 0.0

        # Get memory usage
        memory_usage = 0.0
        if self.buffer_pool:
            memory_usage = self.buffer_pool.current_memory / (1024**3)

        return StreamingProgress(
            patches_processed=self.patches_processed,
            total_patches=self.total_patches,
            elapsed_time=elapsed_time,
            estimated_remaining_time=estimated_remaining,
            current_confidence=0.0,  # Will be updated by aggregator
            memory_usage_gb=memory_usage,
            throughput_patches_per_sec=throughput,
        )

    def estimate_total_patches(self) -> int:
        """Estimate total patches for progress tracking."""
        if self.metadata:
            return self.metadata.estimated_patches

        # Fallback estimation if not initialized
        if self.wsi_path.exists():
            try:
                with OpenSlide(str(self.wsi_path)) as slide:
                    w, h = slide.dimensions
                    # Rough estimate: slide_area / tile_area
                    return (w * h) // (self.tile_size * self.tile_size)
            except Exception:
                pass

        return 1000  # Default fallback

    def _cleanup(self):
        """Clean up resources."""
        if self.buffer_pool:
            self.buffer_pool.cleanup()

        if self.slide:
            try:
                self.slide.close()
            except Exception:
                pass
            self.slide = None

        self.deepzoom = None
        gc.collect()

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self._cleanup()


def get_supported_formats() -> List[str]:
    """Get list of supported WSI formats."""
    return get_supported_wsi_formats()


def validate_wsi_format_compat(wsi_path: str) -> bool:
    """Validate if WSI format is supported (compatibility function)."""
    return validate_wsi_format(wsi_path)
