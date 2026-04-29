"""
Extended Format Support for WSI

Integrates python-bioformats for 165+ slide formats beyond OpenSlide.
"""

import logging
from pathlib import Path
from typing import Optional, Tuple, Union
import numpy as np

logger = logging.getLogger(__name__)

try:
    import javabridge
    import bioformats
    BIOFORMATS_AVAILABLE = True
except ImportError:
    BIOFORMATS_AVAILABLE = False
    logger.warning("python-bioformats not available - limited format support")

try:
    import openslide
    OPENSLIDE_AVAILABLE = True
except ImportError:
    OPENSLIDE_AVAILABLE = False
    logger.warning("openslide not available")


# OpenSlide formats (fast, native)
OPENSLIDE_FORMATS = {
    '.svs', '.tif', '.tiff', '.ndpi', '.vms', '.vmu', 
    '.scn', '.mrxs', '.bif', '.svslide'
}

# Bio-Formats exclusive formats (165 total, key ones listed)
BIOFORMATS_FORMATS = {
    '.czi',      # Zeiss CZI
    '.lif',      # Leica LIF
    '.vsi',      # Olympus VSI
    '.scn',      # Leica SCN
    '.mrxs',     # MIRAX
    '.nd2',      # Nikon ND2
    '.oib',      # Olympus OIB
    '.oif',      # Olympus OIF
    '.oir',      # Olympus OIR
    '.lsm',      # Zeiss LSM
    '.zvi',      # Zeiss ZVI
    '.ims',      # Imaris
    '.dv',       # DeltaVision
    '.r3d',      # DeltaVision
    '.dcm',      # DICOM
    '.dicom',    # DICOM
    '.jp2',      # JPEG 2000
    '.jpx',      # JPX
    '.qptiff',   # Vectra QPTIFF
}


class BioFormatsReader:
    """
    Bio-Formats reader for extended format support.
    
    Wraps python-bioformats to read 165+ slide formats.
    """
    
    def __init__(self, path: Union[str, Path]):
        """
        Initialize Bio-Formats reader.
        
        Args:
            path: Path to slide file
        """
        if not BIOFORMATS_AVAILABLE:
            raise ImportError("python-bioformats required for extended formats")
        
        self.path = str(path)
        self._jvm_started = False
        self._metadata = None
    
    def __enter__(self):
        """Start JVM and return self."""
        self.start_jvm()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Stop JVM."""
        self.stop_jvm()
    
    def start_jvm(self):
        """Start Java Virtual Machine."""
        if not self._jvm_started:
            javabridge.start_vm(class_path=bioformats.JARS)
            self._jvm_started = True
    
    def stop_jvm(self):
        """Stop Java Virtual Machine."""
        if self._jvm_started:
            javabridge.kill_vm()
            self._jvm_started = False
    
    @property
    def metadata(self):
        """Get slide metadata."""
        if self._metadata is None:
            self._metadata = bioformats.get_omexml_metadata(self.path)
        return self._metadata
    
    @property
    def dimensions(self) -> Tuple[int, int]:
        """Get slide dimensions (width, height)."""
        with bioformats.ImageReader(self.path) as reader:
            return (reader.rdr.getSizeX(), reader.rdr.getSizeY())
    
    @property
    def level_count(self) -> int:
        """Get number of pyramid levels."""
        with bioformats.ImageReader(self.path) as reader:
            return reader.rdr.getResolutionCount()
    
    def read_region(
        self,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Read region from slide.
        
        Args:
            location: (x, y) top-left corner
            level: Pyramid level
            size: (width, height) of region
        
        Returns:
            RGB image array (H, W, 3)
        """
        x, y = location
        w, h = size
        
        with bioformats.ImageReader(self.path) as reader:
            # Set resolution level
            reader.rdr.setResolution(level)
            
            # Read region
            image = reader.read(
                c=None,  # All channels
                z=0,     # Z-slice
                t=0,     # Timepoint
                series=0,
                index=None,
                rescale=False,
                XYWH=(x, y, w, h)
            )
        
        return image


class UniversalSlideReader:
    """
    Universal slide reader supporting 165+ formats.
    
    Automatically selects OpenSlide (fast) or Bio-Formats (comprehensive).
    """
    
    def __init__(self, path: Union[str, Path]):
        """
        Initialize universal reader.
        
        Args:
            path: Path to slide file
        """
        self.path = Path(path)
        self.backend = self._select_backend()
        self.reader = self._create_reader()
    
    def _select_backend(self) -> str:
        """Select optimal backend for format."""
        ext = self.path.suffix.lower()
        
        # Prefer OpenSlide for supported formats (faster)
        if ext in OPENSLIDE_FORMATS and OPENSLIDE_AVAILABLE:
            return "openslide"
        
        # Fall back to Bio-Formats for everything else
        if BIOFORMATS_AVAILABLE:
            return "bioformats"
        
        raise ValueError(
            f"No backend available for {ext}. "
            f"Install openslide-python or python-bioformats."
        )
    
    def _create_reader(self):
        """Create backend-specific reader."""
        if self.backend == "openslide":
            return openslide.OpenSlide(str(self.path))
        else:
            return BioFormatsReader(self.path)
    
    @property
    def dimensions(self) -> Tuple[int, int]:
        """Get slide dimensions (width, height)."""
        if self.backend == "openslide":
            return self.reader.dimensions
        else:
            return self.reader.dimensions
    
    @property
    def level_count(self) -> int:
        """Get number of pyramid levels."""
        if self.backend == "openslide":
            return self.reader.level_count
        else:
            return self.reader.level_count
    
    def read_region(
        self,
        location: Tuple[int, int],
        level: int,
        size: Tuple[int, int]
    ) -> np.ndarray:
        """
        Read region from slide.
        
        Args:
            location: (x, y) top-left corner
            level: Pyramid level
            size: (width, height) of region
        
        Returns:
            RGB image array (H, W, 3)
        """
        if self.backend == "openslide":
            # OpenSlide returns RGBA PIL Image
            pil_img = self.reader.read_region(location, level, size)
            return np.array(pil_img.convert('RGB'))
        else:
            return self.reader.read_region(location, level, size)
    
    def close(self):
        """Close reader."""
        if self.backend == "openslide":
            self.reader.close()
        else:
            self.reader.stop_jvm()
    
    def __enter__(self):
        """Context manager entry."""
        if self.backend == "bioformats":
            self.reader.start_jvm()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()


def open_slide(path: Union[str, Path]) -> UniversalSlideReader:
    """
    Open slide with automatic format detection.
    
    Args:
        path: Path to slide file
    
    Returns:
        Universal slide reader
    
    Example:
        >>> with open_slide("slide.czi") as slide:
        ...     dims = slide.dimensions
        ...     region = slide.read_region((0, 0), 0, (512, 512))
    """
    return UniversalSlideReader(path)


def get_supported_formats() -> dict:
    """
    Get supported formats by backend.
    
    Returns:
        Dict mapping backend to format list
    """
    formats = {}
    
    if OPENSLIDE_AVAILABLE:
        formats['openslide'] = sorted(OPENSLIDE_FORMATS)
    
    if BIOFORMATS_AVAILABLE:
        formats['bioformats'] = sorted(BIOFORMATS_FORMATS)
    
    return formats
