"""
WSI Reader with OpenSlide and DICOM support.

This module provides an enhanced WSI reader that supports multiple clinical formats
including .svs, .tiff, .ndpi, and DICOM WSI files.
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image

from .exceptions import FileFormatError, ProcessingError

try:
    from openslide import OpenSlide

    OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False
    OpenSlide = None

try:
    import wsidicom

    WSIDICOM_AVAILABLE = True
except ImportError:
    WSIDICOM_AVAILABLE = False
    wsidicom = None

logger = logging.getLogger(__name__)


class WSIReader:
    """
    Enhanced WSI reader with OpenSlide and DICOM support.

    Supports reading whole-slide images in multiple clinical formats:
    - .svs (Aperio)
    - .tiff (Generic TIFF)
    - .ndpi (Hamamatsu)
    - DICOM WSI

    Args:
        wsi_path: Path to the whole-slide image file

    Example:
        >>> with WSIReader("slide.svs") as reader:
        ...     dimensions = reader.dimensions
        ...     patch = reader.read_region((1000, 1000), level=0, size=(256, 256))
    """

    def __init__(self, wsi_path: Union[str, Path]):
        """
        Initialize WSI reader.

        Args:
            wsi_path: Path to WSI file

        Raises:
            FileFormatError: If file format is unsupported or file doesn't exist
            ProcessingError: If file cannot be opened
        """
        self.wsi_path = Path(wsi_path)

        if not self.wsi_path.exists():
            raise FileFormatError(f"WSI file not found: {wsi_path}")

        # Detect format and initialize appropriate reader
        self._init_reader()

        logger.info(f"Opened WSI: {self.wsi_path.name} (format: {self._format})")

    def _init_reader(self) -> None:
        """Initialize the appropriate reader based on file format."""
        suffix = self.wsi_path.suffix.lower()

        # Try DICOM first if file has .dcm extension or no extension
        if suffix in [".dcm", ""] or self._is_dicom():
            if not WSIDICOM_AVAILABLE:
                raise FileFormatError(
                    "DICOM WSI support requires wsidicom library. "
                    "Install with: pip install wsidicom"
                )
            try:
                self._reader = wsidicom.WsiDicom.open(str(self.wsi_path))
                self._format = "dicom"
                return
            except Exception as e:
                logger.debug(f"Failed to open as DICOM: {e}")

        # Try OpenSlide for standard formats
        if suffix in [".svs", ".tiff", ".tif", ".ndpi", ".vms", ".vmu", ".scn", ".mrxs", ".bif"]:
            if not OPENSLIDE_AVAILABLE:
                raise FileFormatError(
                    "OpenSlide is not installed. Install with: pip install openslide-python"
                )
            try:
                self._reader = OpenSlide(str(self.wsi_path))
                self._format = "openslide"
                return
            except Exception as e:
                raise ProcessingError(f"Failed to open WSI file: {e}")

        # Unsupported format
        raise FileFormatError(
            f"Unsupported file format: {suffix}. "
            f"Supported formats: .svs, .tiff, .ndpi, .dcm"
        )

    def _is_dicom(self) -> bool:
        """Check if file is DICOM format by reading magic bytes."""
        try:
            with open(self.wsi_path, "rb") as f:
                # DICOM files have 'DICM' at byte 128
                f.seek(128)
                magic = f.read(4)
                return magic == b"DICM"
        except Exception:
            return False

    @property
    def dimensions(self) -> Tuple[int, int]:
        """
        Get slide dimensions at level 0 (width, height).

        Returns:
            Tuple of (width, height) in pixels
        """
        if self._format == "openslide":
            return self._reader.dimensions
        elif self._format == "dicom":
            # DICOM: get dimensions from first level
            level = self._reader.levels[0]
            return (level.width, level.height)

    @property
    def level_count(self) -> int:
        """
        Get number of pyramid levels.

        Returns:
            Number of pyramid levels available
        """
        if self._format == "openslide":
            return self._reader.level_count
        elif self._format == "dicom":
            return len(self._reader.levels)

    @property
    def level_dimensions(self) -> List[Tuple[int, int]]:
        """
        Get dimensions for all pyramid levels.

        Returns:
            List of (width, height) tuples for each level
        """
        if self._format == "openslide":
            return list(self._reader.level_dimensions)
        elif self._format == "dicom":
            return [(level.width, level.height) for level in self._reader.levels]

    @property
    def level_downsamples(self) -> List[float]:
        """
        Get downsample factors for all pyramid levels.

        Returns:
            List of downsample factors (level 0 = 1.0)
        """
        if self._format == "openslide":
            return list(self._reader.level_downsamples)
        elif self._format == "dicom":
            # Calculate downsamples relative to level 0
            base_width = self._reader.levels[0].width
            return [base_width / level.width for level in self._reader.levels]

    @property
    def properties(self) -> Dict[str, str]:
        """
        Get slide properties/metadata.

        Returns:
            Dictionary of slide properties
        """
        if self._format == "openslide":
            return dict(self._reader.properties)
        elif self._format == "dicom":
            # Extract relevant DICOM metadata
            metadata = {}
            try:
                dataset = self._reader.dataset
                if hasattr(dataset, "PatientID"):
                    metadata["patient_id"] = str(dataset.PatientID)
                if hasattr(dataset, "StudyDate"):
                    metadata["study_date"] = str(dataset.StudyDate)
                if hasattr(dataset, "Manufacturer"):
                    metadata["manufacturer"] = str(dataset.Manufacturer)
                if hasattr(dataset, "ManufacturerModelName"):
                    metadata["manufacturer_model"] = str(dataset.ManufacturerModelName)
            except Exception as e:
                logger.warning(f"Failed to extract DICOM metadata: {e}")
            return metadata

    def get_magnification(self) -> Optional[float]:
        """
        Extract magnification from slide metadata.

        Returns:
            Magnification value (e.g., 20.0 for 20x) or None if not available
        """
        props = self.properties

        # Try common OpenSlide property keys
        mag_keys = [
            "openslide.objective-power",
            "aperio.AppMag",
            "hamamatsu.SourceLens",
        ]

        for key in mag_keys:
            if key in props:
                try:
                    return float(props[key])
                except (ValueError, TypeError):
                    pass

        logger.warning(f"Magnification not found in metadata for {self.wsi_path.name}")
        return None

    def get_mpp(self) -> Optional[Tuple[float, float]]:
        """
        Extract microns per pixel (MPP) from slide metadata.

        Returns:
            Tuple of (mpp_x, mpp_y) or None if not available
        """
        props = self.properties

        # Try common OpenSlide property keys
        mpp_x_keys = ["openslide.mpp-x", "aperio.MPP"]
        mpp_y_keys = ["openslide.mpp-y", "aperio.MPP"]

        mpp_x = None
        mpp_y = None

        for key in mpp_x_keys:
            if key in props:
                try:
                    mpp_x = float(props[key])
                    break
                except (ValueError, TypeError):
                    pass

        for key in mpp_y_keys:
            if key in props:
                try:
                    mpp_y = float(props[key])
                    break
                except (ValueError, TypeError):
                    pass

        if mpp_x is not None and mpp_y is not None:
            return (mpp_x, mpp_y)

        logger.warning(f"MPP not found in metadata for {self.wsi_path.name}")
        return None

    def get_scanner_info(self) -> Dict[str, Optional[str]]:
        """
        Extract scanner model and scan date from metadata.

        Returns:
            Dictionary with 'model' and 'date' keys
        """
        props = self.properties

        scanner_info = {"model": None, "date": None}

        # Scanner model
        model_keys = [
            "openslide.vendor",
            "aperio.ScanScope ID",
            "hamamatsu.SourceLens",
            "manufacturer",
            "manufacturer_model",
        ]

        for key in model_keys:
            if key in props:
                scanner_info["model"] = props[key]
                break

        # Scan date
        date_keys = [
            "aperio.Date",
            "hamamatsu.Date",
            "study_date",
        ]

        for key in date_keys:
            if key in props:
                scanner_info["date"] = props[key]
                break

        if scanner_info["model"] is None:
            logger.warning(f"Scanner model not found in metadata for {self.wsi_path.name}")
        if scanner_info["date"] is None:
            logger.warning(f"Scan date not found in metadata for {self.wsi_path.name}")

        return scanner_info

    def read_region(
        self,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int],
    ) -> np.ndarray:
        """
        Read a region from the slide as RGB numpy array.

        Args:
            location: (x, y) coordinates at level 0
            level: Pyramid level to read from
            size: Size of region to read (width, height)

        Returns:
            RGB numpy array of shape (height, width, 3)

        Raises:
            ProcessingError: If region cannot be read
        """
        try:
            if self._format == "openslide":
                # OpenSlide returns RGBA PIL Image
                region = self._reader.read_region(location, level, size)
                # Convert to RGB numpy array
                rgb = region.convert("RGB")
                return np.array(rgb)

            elif self._format == "dicom":
                # DICOM: read region from specified level
                x, y = location
                w, h = size

                # Convert level 0 coordinates to target level coordinates
                downsample = self.level_downsamples[level]
                x_level = int(x / downsample)
                y_level = int(y / downsample)

                # Read region
                region = self._reader.read_region((x_level, y_level), level, (w, h))
                return np.array(region)

        except Exception as e:
            raise ProcessingError(f"Failed to read region at {location}: {e}")

    def get_thumbnail(self, size: Tuple[int, int] = (512, 512)) -> Image.Image:
        """
        Get thumbnail of the slide.

        Args:
            size: Target thumbnail size (width, height)

        Returns:
            PIL Image thumbnail
        """
        if self._format == "openslide":
            return self._reader.get_thumbnail(size)
        elif self._format == "dicom":
            # DICOM: read from lowest resolution level and resize
            lowest_level = len(self._reader.levels) - 1
            level_dims = self.level_dimensions[lowest_level]

            # Read entire lowest level
            region = self._reader.read_region((0, 0), lowest_level, level_dims)
            img = Image.fromarray(region)

            # Resize to target size
            img.thumbnail(size, Image.Resampling.LANCZOS)
            return img

    def close(self) -> None:
        """Close the slide and release resources."""
        if hasattr(self, "_reader") and self._reader is not None:
            try:
                if self._format == "openslide":
                    self._reader.close()
                elif self._format == "dicom":
                    self._reader.close()
                logger.debug(f"Closed WSI: {self.wsi_path.name}")
            except Exception as e:
                logger.warning(f"Error closing WSI {self.wsi_path.name}: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()

    def __del__(self):
        """Cleanup on deletion."""
        self.close()
