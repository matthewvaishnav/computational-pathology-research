"""WSI format handlers for multiple file types."""

import logging
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import openslide
from openslide import OpenSlide
from PIL import Image

logger = logging.getLogger(__name__)

try:
    import pydicom
    from pydicom.dataset import Dataset

    DICOM_AVAILABLE = True
except ImportError:
    DICOM_AVAILABLE = False
    logger.warning("pydicom not available - DICOM support disabled")


class WSIFormatHandler(ABC):
    """Abstract base class for WSI format handlers."""

    @abstractmethod
    def can_handle(self, file_path: str) -> bool:
        """Check if this handler can process the given file."""
        pass

    @abstractmethod
    def open_slide(self, file_path: str) -> Any:
        """Open the slide file and return slide object."""
        pass

    @abstractmethod
    def get_dimensions(self, slide_obj: Any) -> Tuple[int, int]:
        """Get slide dimensions (width, height)."""
        pass

    @abstractmethod
    def get_properties(self, slide_obj: Any) -> Dict[str, str]:
        """Get slide properties/metadata."""
        pass

    @abstractmethod
    def read_region(
        self, slide_obj: Any, location: Tuple[int, int], level: int, size: Tuple[int, int]
    ) -> Image.Image:
        """Read a region from the slide."""
        pass

    @abstractmethod
    def get_level_count(self, slide_obj: Any) -> int:
        """Get number of pyramid levels."""
        pass

    @abstractmethod
    def get_level_dimensions(self, slide_obj: Any) -> list:
        """Get dimensions for each pyramid level."""
        pass

    @abstractmethod
    def close_slide(self, slide_obj: Any):
        """Close the slide and clean up resources."""
        pass


class OpenSlideHandler(WSIFormatHandler):
    """Handler for OpenSlide-supported formats (.svs, .tiff, .ndpi, etc.)."""

    SUPPORTED_EXTENSIONS = {
        ".svs",
        ".tiff",
        ".tif",
        ".ndpi",
        ".vms",
        ".vmu",
        ".scn",
        ".mrxs",
        ".bif",
    }

    def can_handle(self, file_path: str) -> bool:
        """Check if OpenSlide can handle this file."""
        path = Path(file_path)
        if path.suffix.lower() not in self.SUPPORTED_EXTENSIONS:
            return False

        try:
            # Quick check if OpenSlide can detect the format
            return openslide.detect_format(file_path) is not None
        except Exception:
            return False

    def open_slide(self, file_path: str) -> OpenSlide:
        """Open slide with OpenSlide."""
        return OpenSlide(file_path)

    def get_dimensions(self, slide_obj: OpenSlide) -> Tuple[int, int]:
        """Get slide dimensions."""
        return slide_obj.dimensions

    def get_properties(self, slide_obj: OpenSlide) -> Dict[str, str]:
        """Get slide properties."""
        return dict(slide_obj.properties)

    def read_region(
        self, slide_obj: OpenSlide, location: Tuple[int, int], level: int, size: Tuple[int, int]
    ) -> Image.Image:
        """Read region from slide."""
        return slide_obj.read_region(location, level, size)

    def get_level_count(self, slide_obj: OpenSlide) -> int:
        """Get number of pyramid levels."""
        return slide_obj.level_count

    def get_level_dimensions(self, slide_obj: OpenSlide) -> list:
        """Get dimensions for each level."""
        return slide_obj.level_dimensions

    def close_slide(self, slide_obj: OpenSlide):
        """Close the slide."""
        slide_obj.close()


class DICOMHandler(WSIFormatHandler):
    """Handler for DICOM WSI files."""

    def __init__(self):
        if not DICOM_AVAILABLE:
            raise ImportError("pydicom required for DICOM support")

    def can_handle(self, file_path: str) -> bool:
        """Check if this is a DICOM WSI file."""
        if not DICOM_AVAILABLE:
            return False

        path = Path(file_path)
        if path.suffix.lower() not in {".dcm", ".dicom"}:
            return False

        try:
            # Quick check if it's a valid DICOM file
            ds = pydicom.dcmread(file_path, stop_before_pixels=True)
            # Check if it's a WSI (VL Whole Slide Microscopy Image)
            return hasattr(ds, "SOPClassUID") and ds.SOPClassUID == "1.2.840.10008.5.1.4.1.1.77.1.6"
        except Exception:
            return False

    def open_slide(self, file_path: str) -> Dataset:
        """Open DICOM slide."""
        return pydicom.dcmread(file_path)

    def get_dimensions(self, slide_obj: Dataset) -> Tuple[int, int]:
        """Get slide dimensions from DICOM."""
        # DICOM WSI dimensions are in TotalPixelMatrixColumns/Rows
        width = getattr(slide_obj, "TotalPixelMatrixColumns", 0)
        height = getattr(slide_obj, "TotalPixelMatrixRows", 0)

        if width == 0 or height == 0:
            # Fallback to regular image dimensions
            width = getattr(slide_obj, "Columns", 0)
            height = getattr(slide_obj, "Rows", 0)

        return (width, height)

    def get_properties(self, slide_obj: Dataset) -> Dict[str, str]:
        """Get DICOM properties."""
        properties = {}

        # Extract relevant DICOM tags
        tag_mapping = {
            "PatientID": "patient_id",
            "StudyInstanceUID": "study_uid",
            "SeriesInstanceUID": "series_uid",
            "SOPInstanceUID": "instance_uid",
            "Manufacturer": "manufacturer",
            "ManufacturerModelName": "model",
            "SoftwareVersions": "software_version",
            "ImageType": "image_type",
            "PhotometricInterpretation": "photometric_interpretation",
            "SamplesPerPixel": "samples_per_pixel",
            "BitsAllocated": "bits_allocated",
            "BitsStored": "bits_stored",
            "HighBit": "high_bit",
            "PixelRepresentation": "pixel_representation",
        }

        for dicom_tag, prop_name in tag_mapping.items():
            if hasattr(slide_obj, dicom_tag):
                value = getattr(slide_obj, dicom_tag)
                properties[prop_name] = str(value)

        return properties

    def read_region(
        self, slide_obj: Dataset, location: Tuple[int, int], level: int, size: Tuple[int, int]
    ) -> Image.Image:
        """Read region from DICOM slide."""
        # This is a simplified implementation
        # Full DICOM WSI support would require handling tiled images, pyramids, etc.

        if hasattr(slide_obj, "pixel_array"):
            pixel_array = slide_obj.pixel_array

            # Extract region
            x, y = location
            w, h = size

            # Ensure bounds
            img_h, img_w = pixel_array.shape[:2]
            x = max(0, min(x, img_w - 1))
            y = max(0, min(y, img_h - 1))
            w = min(w, img_w - x)
            h = min(h, img_h - y)

            region = pixel_array[y : y + h, x : x + w]

            # Convert to PIL Image
            if len(region.shape) == 2:  # Grayscale
                return Image.fromarray(region, mode="L")
            elif len(region.shape) == 3:  # RGB
                return Image.fromarray(region, mode="RGB")
            else:
                raise ValueError(f"Unsupported pixel array shape: {region.shape}")

        else:
            raise ValueError("DICOM file does not contain pixel data")

    def get_level_count(self, slide_obj: Dataset) -> int:
        """Get number of pyramid levels (simplified for DICOM)."""
        # Most DICOM WSI files have pyramid structure, but this is simplified
        return 1

    def get_level_dimensions(self, slide_obj: Dataset) -> list:
        """Get dimensions for each level."""
        dimensions = self.get_dimensions(slide_obj)
        return [dimensions]  # Simplified - single level

    def close_slide(self, slide_obj: Dataset):
        """Close DICOM slide (no explicit close needed)."""
        pass


class WSIFormatManager:
    """Manager for different WSI format handlers."""

    def __init__(self):
        self.handlers = []
        self._register_default_handlers()

    def _register_default_handlers(self):
        """Register default format handlers."""
        # Register OpenSlide handler
        self.handlers.append(OpenSlideHandler())

        # Register DICOM handler if available
        if DICOM_AVAILABLE:
            try:
                self.handlers.append(DICOMHandler())
            except ImportError:
                logger.warning("Failed to initialize DICOM handler")

    def register_handler(self, handler: WSIFormatHandler):
        """Register a custom format handler."""
        self.handlers.append(handler)

    def get_handler(self, file_path: str) -> Optional[WSIFormatHandler]:
        """Get appropriate handler for the given file."""
        for handler in self.handlers:
            if handler.can_handle(file_path):
                return handler
        return None

    def get_supported_formats(self) -> list:
        """Get list of all supported formats."""
        formats = set()

        # OpenSlide formats
        formats.update(OpenSlideHandler.SUPPORTED_EXTENSIONS)

        # DICOM formats
        if DICOM_AVAILABLE:
            formats.update({".dcm", ".dicom"})

        return sorted(list(formats))

    def validate_format(self, file_path: str) -> bool:
        """Validate if file format is supported."""
        return self.get_handler(file_path) is not None


# Global format manager instance
format_manager = WSIFormatManager()


def get_wsi_handler(file_path: str) -> Optional[WSIFormatHandler]:
    """Get WSI format handler for the given file."""
    return format_manager.get_handler(file_path)


def get_supported_wsi_formats() -> list:
    """Get list of supported WSI formats."""
    return format_manager.get_supported_formats()


def validate_wsi_format(file_path: str) -> bool:
    """Validate if WSI format is supported."""
    return format_manager.validate_format(file_path)
