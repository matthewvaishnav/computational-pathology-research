"""
OpenSlide utilities for reading whole-slide images (.svs, .tiff, etc.).

This module provides utilities for working with whole-slide images using OpenSlide.
"""

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from PIL import Image

try:
    import openslide
    from openslide import OpenSlide

    OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False
    OpenSlide = None

logger = logging.getLogger(__name__)


def check_openslide_available() -> bool:
    """Check if OpenSlide is available."""
    if not OPENSLIDE_AVAILABLE:
        logger.warning(
            "OpenSlide is not installed. Install with: pip install openslide-python"
        )
    return OPENSLIDE_AVAILABLE


class WSIReader:
    """
    Whole-slide image reader using OpenSlide.

    Provides methods for reading WSI files, extracting patches, and getting metadata.

    Args:
        wsi_path: Path to the whole-slide image file (.svs, .tiff, etc.)

    Example:
        >>> reader = WSIReader("slide.svs")
        >>> thumbnail = reader.get_thumbnail(size=(512, 512))
        >>> patch = reader.read_region((1000, 1000), level=0, size=(256, 256))
        >>> reader.close()
    """

    def __init__(self, wsi_path: str):
        """Initialize WSI reader.

        Args:
            wsi_path: Path to WSI file

        Raises:
            ImportError: If OpenSlide is not installed
            FileNotFoundError: If WSI file doesn't exist
        """
        if not OPENSLIDE_AVAILABLE:
            raise ImportError(
                "OpenSlide is not installed. Install with: pip install openslide-python"
            )

        self.wsi_path = Path(wsi_path)
        if not self.wsi_path.exists():
            raise FileNotFoundError(f"WSI file not found: {wsi_path}")

        self.slide = OpenSlide(str(self.wsi_path))
        logger.info(f"Opened WSI: {self.wsi_path.name}")

    @property
    def dimensions(self) -> Tuple[int, int]:
        """Get slide dimensions at level 0 (width, height)."""
        return self.slide.dimensions

    @property
    def level_count(self) -> int:
        """Get number of pyramid levels."""
        return self.slide.level_count

    @property
    def level_dimensions(self) -> List[Tuple[int, int]]:
        """Get dimensions for all pyramid levels."""
        return self.slide.level_dimensions

    @property
    def level_downsamples(self) -> List[float]:
        """Get downsample factors for all pyramid levels."""
        return self.slide.level_downsamples

    @property
    def properties(self) -> dict:
        """Get slide properties/metadata."""
        return dict(self.slide.properties)

    def get_thumbnail(self, size: Tuple[int, int] = (512, 512)) -> Image.Image:
        """
        Get thumbnail of the slide.

        Args:
            size: Target thumbnail size (width, height)

        Returns:
            PIL Image thumbnail
        """
        return self.slide.get_thumbnail(size)

    def read_region(
        self,
        location: Tuple[int, int],
        level: int = 0,
        size: Tuple[int, int] = (256, 256),
    ) -> Image.Image:
        """
        Read a region from the slide.

        Args:
            location: (x, y) coordinates at level 0
            level: Pyramid level to read from
            size: Size of region to read (width, height)

        Returns:
            PIL Image of the region (RGBA format)
        """
        return self.slide.read_region(location, level, size)

    def read_region_rgb(
        self,
        location: Tuple[int, int],
        level: int = 0,
        size: Tuple[int, int] = (256, 256),
    ) -> np.ndarray:
        """
        Read a region and convert to RGB numpy array.

        Args:
            location: (x, y) coordinates at level 0
            level: Pyramid level to read from
            size: Size of region to read (width, height)

        Returns:
            RGB numpy array of shape (height, width, 3)
        """
        region = self.read_region(location, level, size)
        # Convert RGBA to RGB
        rgb = region.convert("RGB")
        return np.array(rgb)

    def extract_patches(
        self,
        patch_size: int = 256,
        level: int = 0,
        stride: Optional[int] = None,
        tissue_threshold: float = 0.5,
    ) -> List[Tuple[np.ndarray, Tuple[int, int]]]:
        """
        Extract patches from the slide using a grid.

        Args:
            patch_size: Size of patches to extract
            level: Pyramid level to extract from
            stride: Stride between patches (defaults to patch_size for non-overlapping)
            tissue_threshold: Minimum tissue percentage to keep patch (0-1)

        Returns:
            List of (patch_array, (x, y)) tuples
        """
        if stride is None:
            stride = patch_size

        width, height = self.level_dimensions[level]
        downsample = self.level_downsamples[level]

        patches = []

        for y in range(0, height - patch_size, stride):
            for x in range(0, width - patch_size, stride):
                # Convert to level 0 coordinates
                x0 = int(x * downsample)
                y0 = int(y * downsample)

                # Read patch
                patch = self.read_region_rgb(
                    (x0, y0), level=level, size=(patch_size, patch_size)
                )

                # Check tissue content
                if self._has_tissue(patch, tissue_threshold):
                    patches.append((patch, (x0, y0)))

        logger.info(f"Extracted {len(patches)} patches from {self.wsi_path.name}")
        return patches

    def _has_tissue(self, patch: np.ndarray, threshold: float = 0.5) -> bool:
        """
        Check if patch contains sufficient tissue.

        Simple heuristic: check if patch is not mostly white/background.

        Args:
            patch: RGB patch array
            threshold: Minimum non-background percentage

        Returns:
            True if patch has sufficient tissue
        """
        # Convert to grayscale
        gray = np.mean(patch, axis=2)

        # Background is typically white (high intensity)
        # Tissue is darker (lower intensity)
        tissue_mask = gray < 200  # Threshold for non-background

        tissue_percentage = np.mean(tissue_mask)
        return tissue_percentage >= threshold

    def close(self):
        """Close the slide."""
        if hasattr(self, "slide") and self.slide is not None:
            self.slide.close()
            logger.debug(f"Closed WSI: {self.wsi_path.name}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()


def get_slide_info(wsi_path: str) -> dict:
    """
    Get information about a whole-slide image.

    Args:
        wsi_path: Path to WSI file

    Returns:
        Dictionary with slide information
    """
    with WSIReader(wsi_path) as reader:
        return {
            "path": str(reader.wsi_path),
            "dimensions": reader.dimensions,
            "level_count": reader.level_count,
            "level_dimensions": reader.level_dimensions,
            "level_downsamples": reader.level_downsamples,
            "properties": reader.properties,
        }


if __name__ == "__main__":
    # Example usage
    import sys

    if len(sys.argv) < 2:
        print("Usage: python openslide_utils.py <path_to_wsi>")
        sys.exit(1)

    wsi_path = sys.argv[1]

    # Get slide info
    info = get_slide_info(wsi_path)
    print(f"\nSlide Information:")
    print(f"  Path: {info['path']}")
    print(f"  Dimensions: {info['dimensions']}")
    print(f"  Levels: {info['level_count']}")
    print(f"  Level dimensions: {info['level_dimensions']}")
    print(f"  Downsamples: {info['level_downsamples']}")

    # Extract some patches
    with WSIReader(wsi_path) as reader:
        thumbnail = reader.get_thumbnail((512, 512))
        thumbnail.save("thumbnail.png")
        print(f"\nSaved thumbnail to thumbnail.png")

        patches = reader.extract_patches(patch_size=256, level=1, stride=512)
        print(f"Extracted {len(patches)} patches")
