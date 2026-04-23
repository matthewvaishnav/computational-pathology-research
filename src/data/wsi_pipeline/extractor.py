"""
Patch Extractor for WSI processing.

This module provides patch extraction functionality for whole-slide images,
supporting grid-based coordinate generation, configurable stride, and
multi-resolution pyramid level support.
"""

import logging
from typing import Iterator, List, Optional, Tuple

import numpy as np

from .exceptions import ProcessingError
from .reader import WSIReader

logger = logging.getLogger(__name__)


class PatchExtractor:
    """
    Extract patches from WSI at specified coordinates, sizes, and pyramid levels.

    Supports:
    - Grid-based coordinate generation
    - Configurable stride for overlapping/non-overlapping patches
    - Coordinate conversion between pyramid levels
    - Streaming extraction for memory efficiency

    Args:
        patch_size: Size of patches to extract (width and height in pixels)
        stride: Step size between patches. If None, defaults to patch_size (non-overlapping)
        level: Pyramid level to extract patches from (0 = highest resolution)
        target_mpp: Target microns per pixel for resolution-independent extraction (optional)

    Example:
        >>> extractor = PatchExtractor(patch_size=256, stride=128, level=0)
        >>> coords = extractor.generate_coordinates((10000, 10000))
        >>> for patch, coord in extractor.extract_patches_streaming(reader, coords):
        ...     process(patch)
    """

    def __init__(
        self,
        patch_size: int = 256,
        stride: Optional[int] = None,
        level: int = 0,
        target_mpp: Optional[float] = None,
    ):
        """
        Initialize patch extractor with sampling parameters.

        Args:
            patch_size: Size of patches to extract (width and height in pixels)
            stride: Step size between patches. If None, defaults to patch_size
            level: Pyramid level to extract from (0 = highest resolution)
            target_mpp: Target microns per pixel (optional, for resolution-independent extraction)

        Raises:
            ValueError: If patch_size or stride are invalid
        """
        if patch_size < 64 or patch_size > 2048:
            raise ValueError(f"patch_size must be between 64 and 2048, got {patch_size}")

        self.patch_size = patch_size
        self.stride = stride if stride is not None else patch_size
        self.level = level
        self.target_mpp = target_mpp

        if self.stride <= 0:
            raise ValueError(f"stride must be positive, got {self.stride}")

        logger.debug(
            f"Initialized PatchExtractor: patch_size={patch_size}, "
            f"stride={self.stride}, level={level}, target_mpp={target_mpp}"
        )

    def generate_coordinates(
        self,
        slide_dimensions: Tuple[int, int],
        tissue_mask: Optional[np.ndarray] = None,
        precompute_all: bool = True,
    ) -> List[Tuple[int, int]]:
        """
        Generate grid coordinates for patch extraction with optimization.

        Generates a regular grid of coordinates based on patch_size and stride.
        Optionally filters coordinates using a tissue mask. Uses vectorized
        operations for faster coordinate generation.

        Args:
            slide_dimensions: Slide dimensions at level 0 (width, height)
            tissue_mask: Optional binary mask indicating tissue regions (height, width)
                        If provided, only coordinates in tissue regions are returned
            precompute_all: Use vectorized coordinate generation for speed

        Returns:
            List of (x, y) coordinates at level 0

        Example:
            >>> extractor = PatchExtractor(patch_size=256, stride=256)
            >>> coords = extractor.generate_coordinates((10000, 10000))
            >>> len(coords)  # Number of non-overlapping patches
            1600
        """
        width, height = slide_dimensions

        if precompute_all:
            # Vectorized coordinate generation (faster for large slides)
            coordinates = self._generate_coordinates_vectorized(width, height)
        else:
            # Original loop-based generation
            coordinates = []
            for y in range(0, height - self.patch_size + 1, self.stride):
                for x in range(0, width - self.patch_size + 1, self.stride):
                    coordinates.append((x, y))

        logger.debug(
            f"Generated {len(coordinates)} grid coordinates for "
            f"{width}x{height} slide with stride={self.stride}"
        )

        # Filter by tissue mask if provided
        if tissue_mask is not None:
            coordinates = self._filter_by_tissue_mask_optimized(
                coordinates, tissue_mask, slide_dimensions
            )
            logger.debug(
                f"Filtered to {len(coordinates)} coordinates using tissue mask"
            )

        return coordinates

    def _generate_coordinates_vectorized(
        self, 
        width: int, 
        height: int
    ) -> List[Tuple[int, int]]:
        """
        Generate coordinates using vectorized operations for speed.
        
        Args:
            width: Slide width
            height: Slide height
            
        Returns:
            List of (x, y) coordinates
        """
        # Generate x and y ranges
        x_coords = np.arange(0, width - self.patch_size + 1, self.stride)
        y_coords = np.arange(0, height - self.patch_size + 1, self.stride)
        
        # Create meshgrid and flatten
        xx, yy = np.meshgrid(x_coords, y_coords)
        coordinates = list(zip(xx.flatten(), yy.flatten()))
        
        return coordinates

    def _filter_by_tissue_mask_optimized(
        self,
        coordinates: List[Tuple[int, int]],
        tissue_mask: np.ndarray,
        slide_dimensions: Tuple[int, int],
    ) -> List[Tuple[int, int]]:
        """
        Filter coordinates using tissue mask with vectorized operations.

        Args:
            coordinates: List of (x, y) coordinates at level 0
            tissue_mask: Binary mask (height, width) at some resolution
            slide_dimensions: Original slide dimensions at level 0

        Returns:
            Filtered list of coordinates
        """
        if not coordinates:
            return coordinates
            
        mask_height, mask_width = tissue_mask.shape
        slide_width, slide_height = slide_dimensions

        # Calculate scale factors
        scale_x = mask_width / slide_width
        scale_y = mask_height / slide_height

        # Convert to numpy arrays for vectorized operations
        coords_array = np.array(coordinates)
        
        # Convert coordinates to mask space
        mask_coords = coords_array * np.array([scale_x, scale_y])
        patch_size_mask = np.array([
            max(1, int(self.patch_size * scale_x)),
            max(1, int(self.patch_size * scale_y))
        ])
        
        # Calculate patch centers
        centers = mask_coords + patch_size_mask // 2
        
        # Clip to mask bounds
        centers[:, 0] = np.clip(centers[:, 0], 0, mask_width - 1)
        centers[:, 1] = np.clip(centers[:, 1], 0, mask_height - 1)
        
        # Check tissue mask at centers (vectorized)
        center_indices = centers.astype(int)
        tissue_flags = tissue_mask[center_indices[:, 1], center_indices[:, 0]]
        
        # Filter coordinates
        filtered_coords = [
            (int(x), int(y)) for (x, y), flag in zip(coordinates, tissue_flags) if flag
        ]

        return filtered_coords

    def convert_coordinates_to_level(
        self,
        coordinates: List[Tuple[int, int]],
        from_level: int,
        to_level: int,
        level_downsamples: List[float],
    ) -> List[Tuple[int, int]]:
        """
        Convert coordinates between pyramid levels.

        Args:
            coordinates: List of (x, y) coordinates at from_level
            from_level: Source pyramid level
            to_level: Target pyramid level
            level_downsamples: Downsample factors for all pyramid levels

        Returns:
            List of (x, y) coordinates at to_level

        Example:
            >>> coords_level0 = [(0, 0), (256, 0), (512, 0)]
            >>> downsamples = [1.0, 2.0, 4.0]
            >>> coords_level1 = extractor.convert_coordinates_to_level(
            ...     coords_level0, from_level=0, to_level=1, level_downsamples=downsamples
            ... )
            >>> coords_level1
            [(0, 0), (128, 0), (256, 0)]
        """
        if from_level == to_level:
            return coordinates

        if from_level >= len(level_downsamples) or to_level >= len(level_downsamples):
            raise ValueError(
                f"Invalid level: from_level={from_level}, to_level={to_level}, "
                f"available levels={len(level_downsamples)}"
            )

        # Calculate conversion factor
        from_downsample = level_downsamples[from_level]
        to_downsample = level_downsamples[to_level]
        scale_factor = from_downsample / to_downsample

        # Convert coordinates
        converted_coords = [
            (int(x * scale_factor), int(y * scale_factor)) for x, y in coordinates
        ]

        return converted_coords

    def extract_patch(
        self,
        reader: WSIReader,
        location: Tuple[int, int],
    ) -> np.ndarray:
        """
        Extract a single patch at specified location.

        Args:
            reader: WSIReader instance
            location: (x, y) coordinates at level 0

        Returns:
            RGB numpy array of shape (patch_size, patch_size, 3)

        Raises:
            ProcessingError: If patch extraction fails
        """
        try:
            # Validate coordinates are within bounds
            slide_width, slide_height = reader.dimensions
            x, y = location

            if x < 0 or y < 0:
                raise ProcessingError(
                    f"Coordinates out of bounds: ({x}, {y}) must be non-negative"
                )

            if x + self.patch_size > slide_width or y + self.patch_size > slide_height:
                raise ProcessingError(
                    f"Patch at ({x}, {y}) with size {self.patch_size} "
                    f"exceeds slide dimensions ({slide_width}, {slide_height})"
                )

            # Extract patch
            patch = reader.read_region(
                location=location,
                level=self.level,
                size=(self.patch_size, self.patch_size),
            )

            # Validate patch dimensions
            if patch.shape[:2] != (self.patch_size, self.patch_size):
                raise ProcessingError(
                    f"Extracted patch has incorrect dimensions: "
                    f"expected ({self.patch_size}, {self.patch_size}), "
                    f"got {patch.shape[:2]}"
                )

            return patch

        except Exception as e:
            raise ProcessingError(f"Failed to extract patch at {location}: {e}")

    def extract_patches_streaming(
        self,
        reader: WSIReader,
        coordinates: List[Tuple[int, int]],
    ) -> Iterator[Tuple[np.ndarray, Tuple[int, int]]]:
        """
        Stream patches without loading all into memory.

        Yields patches one at a time to avoid memory accumulation.

        Args:
            reader: WSIReader instance
            coordinates: List of (x, y) coordinates at level 0

        Yields:
            Tuple of (patch, coordinate) where patch is RGB numpy array

        Example:
            >>> extractor = PatchExtractor(patch_size=256)
            >>> coords = extractor.generate_coordinates((10000, 10000))
            >>> for patch, coord in extractor.extract_patches_streaming(reader, coords):
            ...     features = model(patch)
            ...     save_features(features, coord)
        """
        for coord in coordinates:
            try:
                patch = self.extract_patch(reader, coord)
                yield patch, coord
            except ProcessingError as e:
                logger.warning(f"Skipping patch at {coord}: {e}")
                continue
