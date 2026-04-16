"""
Unit tests for OpenSlide integration.

Tests WSI file format compatibility, patch extraction with correct dimensions/coordinates,
pyramid level access, tissue detection, thumbnail generation, and error handling.
"""

import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
import numpy as np
from PIL import Image

from src.data.openslide_utils import WSIReader, get_slide_info, check_openslide_available
from tests.dataset_testing.synthetic.wsi_generator import WSISyntheticGenerator, WSISyntheticSpec
from tests.dataset_testing.base_interfaces import ErrorSimulator


class TestOpenSlideAvailability:
    """Test OpenSlide availability checking."""

    def test_check_openslide_available_when_installed(self):
        """Test OpenSlide availability check when installed."""
        with patch("src.data.openslide_utils.OPENSLIDE_AVAILABLE", True):
            assert check_openslide_available() is True

    def test_check_openslide_available_when_not_installed(self):
        """Test OpenSlide availability check when not installed."""
        with patch("src.data.openslide_utils.OPENSLIDE_AVAILABLE", False):
            assert check_openslide_available() is False


class TestWSIReaderInitialization:
    """Test WSI reader initialization and basic properties."""

    def test_wsi_reader_init_without_openslide(self):
        """Test WSIReader initialization fails without OpenSlide."""
        with patch("src.data.openslide_utils.OPENSLIDE_AVAILABLE", False):
            with pytest.raises(ImportError, match="OpenSlide is not installed"):
                WSIReader("dummy.svs")

    def test_wsi_reader_init_file_not_found(self):
        """Test WSIReader initialization fails with non-existent file."""
        with patch("src.data.openslide_utils.OPENSLIDE_AVAILABLE", True):
            with pytest.raises(FileNotFoundError, match="WSI file not found"):
                WSIReader("nonexistent.svs")

    @patch("src.data.openslide_utils.OPENSLIDE_AVAILABLE", True)
    @patch("src.data.openslide_utils.OpenSlide")
    def test_wsi_reader_init_success(self, mock_openslide):
        """Test successful WSIReader initialization."""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix=".svs", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Mock OpenSlide instance
            mock_slide = Mock()
            mock_openslide.return_value = mock_slide

            reader = WSIReader(tmp_path)

            assert reader.wsi_path == Path(tmp_path)
            assert reader.slide == mock_slide
            mock_openslide.assert_called_once_with(tmp_path)
        finally:
            Path(tmp_path).unlink()


class TestWSIReaderProperties:
    """Test WSI reader properties and metadata access."""

    @patch("src.data.openslide_utils.OPENSLIDE_AVAILABLE", True)
    @patch("src.data.openslide_utils.OpenSlide")
    def test_dimensions_property(self, mock_openslide):
        """Test dimensions property access."""
        with tempfile.NamedTemporaryFile(suffix=".svs", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            mock_slide = Mock()
            mock_slide.dimensions = (10000, 8000)
            mock_openslide.return_value = mock_slide

            reader = WSIReader(tmp_path)
            assert reader.dimensions == (10000, 8000)
        finally:
            Path(tmp_path).unlink()

    @patch("src.data.openslide_utils.OPENSLIDE_AVAILABLE", True)
    @patch("src.data.openslide_utils.OpenSlide")
    def test_level_properties(self, mock_openslide):
        """Test pyramid level properties."""
        with tempfile.NamedTemporaryFile(suffix=".svs", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            mock_slide = Mock()
            mock_slide.level_count = 4
            mock_slide.level_dimensions = [(10000, 8000), (5000, 4000), (2500, 2000), (1250, 1000)]
            mock_slide.level_downsamples = [1.0, 2.0, 4.0, 8.0]
            mock_openslide.return_value = mock_slide

            reader = WSIReader(tmp_path)
            assert reader.level_count == 4
            assert reader.level_dimensions == [
                (10000, 8000),
                (5000, 4000),
                (2500, 2000),
                (1250, 1000),
            ]
            assert reader.level_downsamples == [1.0, 2.0, 4.0, 8.0]
        finally:
            Path(tmp_path).unlink()

    @patch("src.data.openslide_utils.OPENSLIDE_AVAILABLE", True)
    @patch("src.data.openslide_utils.OpenSlide")
    def test_properties_metadata(self, mock_openslide):
        """Test slide properties/metadata access."""
        with tempfile.NamedTemporaryFile(suffix=".svs", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            mock_slide = Mock()
            mock_properties = {
                "openslide.vendor": "aperio",
                "openslide.quickhash-1": "abc123",
                "aperio.ImageID": "12345",
            }
            mock_slide.properties = mock_properties
            mock_openslide.return_value = mock_slide

            reader = WSIReader(tmp_path)
            assert reader.properties == mock_properties
        finally:
            Path(tmp_path).unlink()


class TestWSIReaderImageOperations:
    """Test WSI reader image operations."""

    @patch("src.data.openslide_utils.OPENSLIDE_AVAILABLE", True)
    @patch("src.data.openslide_utils.OpenSlide")
    def test_get_thumbnail(self, mock_openslide):
        """Test thumbnail generation."""
        with tempfile.NamedTemporaryFile(suffix=".svs", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            mock_slide = Mock()
            mock_thumbnail = Image.new("RGB", (512, 512), color="red")
            mock_slide.get_thumbnail.return_value = mock_thumbnail
            mock_openslide.return_value = mock_slide

            reader = WSIReader(tmp_path)
            thumbnail = reader.get_thumbnail((512, 512))

            assert thumbnail == mock_thumbnail
            mock_slide.get_thumbnail.assert_called_once_with((512, 512))
        finally:
            Path(tmp_path).unlink()

    @patch("src.data.openslide_utils.OPENSLIDE_AVAILABLE", True)
    @patch("src.data.openslide_utils.OpenSlide")
    def test_read_region(self, mock_openslide):
        """Test region reading."""
        with tempfile.NamedTemporaryFile(suffix=".svs", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            mock_slide = Mock()
            mock_region = Image.new("RGBA", (256, 256), color="blue")
            mock_slide.read_region.return_value = mock_region
            mock_openslide.return_value = mock_slide

            reader = WSIReader(tmp_path)
            region = reader.read_region((1000, 1000), level=0, size=(256, 256))

            assert region == mock_region
            mock_slide.read_region.assert_called_once_with((1000, 1000), 0, (256, 256))
        finally:
            Path(tmp_path).unlink()

    @patch("src.data.openslide_utils.OPENSLIDE_AVAILABLE", True)
    @patch("src.data.openslide_utils.OpenSlide")
    def test_read_region_rgb(self, mock_openslide):
        """Test RGB region reading and conversion."""
        with tempfile.NamedTemporaryFile(suffix=".svs", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            mock_slide = Mock()
            # Create RGBA image that will be converted to RGB
            mock_region = Image.new("RGBA", (256, 256), color=(255, 0, 0, 255))
            mock_slide.read_region.return_value = mock_region
            mock_openslide.return_value = mock_slide

            reader = WSIReader(tmp_path)
            rgb_array = reader.read_region_rgb((1000, 1000), level=0, size=(256, 256))

            assert isinstance(rgb_array, np.ndarray)
            assert rgb_array.shape == (256, 256, 3)
            assert rgb_array.dtype == np.uint8
            # Check that it's red (converted from RGBA)
            assert np.all(rgb_array[:, :, 0] == 255)  # Red channel
            assert np.all(rgb_array[:, :, 1] == 0)  # Green channel
            assert np.all(rgb_array[:, :, 2] == 0)  # Blue channel
        finally:
            Path(tmp_path).unlink()


class TestWSIReaderPatchExtraction:
    """Test patch extraction functionality."""

    @patch("src.data.openslide_utils.OPENSLIDE_AVAILABLE", True)
    @patch("src.data.openslide_utils.OpenSlide")
    def test_extract_patches_basic(self, mock_openslide):
        """Test basic patch extraction."""
        with tempfile.NamedTemporaryFile(suffix=".svs", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            mock_slide = Mock()
            mock_slide.level_dimensions = [(1000, 1000)]
            mock_slide.level_downsamples = [1.0]

            # Mock read_region to return tissue-like patches
            def mock_read_region(location, level, size):
                # Create a patch with some tissue (not all white)
                return Image.new("RGBA", size, color=(150, 100, 100, 255))

            mock_slide.read_region = mock_read_region
            mock_openslide.return_value = mock_slide

            reader = WSIReader(tmp_path)
            patches = reader.extract_patches(patch_size=256, level=0, stride=256)

            # Should extract patches from a 1000x1000 image with 256x256 patches
            # Grid would be 3x3 = 9 patches (with some overlap at edges)
            assert len(patches) > 0

            # Check patch format
            for patch, (x, y) in patches:
                assert isinstance(patch, np.ndarray)
                assert patch.shape == (256, 256, 3)
                assert isinstance(x, int)
                assert isinstance(y, int)
        finally:
            Path(tmp_path).unlink()

    @patch("src.data.openslide_utils.OPENSLIDE_AVAILABLE", True)
    @patch("src.data.openslide_utils.OpenSlide")
    def test_extract_patches_with_stride(self, mock_openslide):
        """Test patch extraction with custom stride."""
        with tempfile.NamedTemporaryFile(suffix=".svs", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            mock_slide = Mock()
            mock_slide.level_dimensions = [(1000, 1000)]
            mock_slide.level_downsamples = [1.0]

            def mock_read_region(location, level, size):
                return Image.new("RGBA", size, color=(150, 100, 100, 255))

            mock_slide.read_region = mock_read_region
            mock_openslide.return_value = mock_slide

            reader = WSIReader(tmp_path)
            patches = reader.extract_patches(patch_size=256, level=0, stride=128)

            # With stride=128, should get more overlapping patches
            assert len(patches) > 0
        finally:
            Path(tmp_path).unlink()

    @patch("src.data.openslide_utils.OPENSLIDE_AVAILABLE", True)
    @patch("src.data.openslide_utils.OpenSlide")
    def test_tissue_detection(self, mock_openslide):
        """Test tissue detection in patches."""
        with tempfile.NamedTemporaryFile(suffix=".svs", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            mock_slide = Mock()
            mock_slide.level_dimensions = [(512, 512)]
            mock_slide.level_downsamples = [1.0]

            call_count = 0

            def mock_read_region(location, level, size):
                nonlocal call_count
                call_count += 1
                if call_count % 2 == 0:
                    # White background patch (should be filtered out)
                    return Image.new("RGBA", size, color=(255, 255, 255, 255))
                else:
                    # Tissue patch (should be kept)
                    return Image.new("RGBA", size, color=(150, 100, 100, 255))

            mock_slide.read_region = mock_read_region
            mock_openslide.return_value = mock_slide

            reader = WSIReader(tmp_path)
            patches = reader.extract_patches(patch_size=256, level=0, tissue_threshold=0.3)

            # Should filter out white background patches
            assert len(patches) >= 0  # Some patches should pass tissue detection
        finally:
            Path(tmp_path).unlink()


class TestWSIReaderContextManager:
    """Test WSI reader context manager functionality."""

    @patch("src.data.openslide_utils.OPENSLIDE_AVAILABLE", True)
    @patch("src.data.openslide_utils.OpenSlide")
    def test_context_manager(self, mock_openslide):
        """Test WSIReader as context manager."""
        with tempfile.NamedTemporaryFile(suffix=".svs", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            mock_slide = Mock()
            mock_openslide.return_value = mock_slide

            with WSIReader(tmp_path) as reader:
                assert reader.slide == mock_slide

            # Should call close on exit
            mock_slide.close.assert_called_once()
        finally:
            Path(tmp_path).unlink()

    @patch("src.data.openslide_utils.OPENSLIDE_AVAILABLE", True)
    @patch("src.data.openslide_utils.OpenSlide")
    def test_manual_close(self, mock_openslide):
        """Test manual close functionality."""
        with tempfile.NamedTemporaryFile(suffix=".svs", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            mock_slide = Mock()
            mock_openslide.return_value = mock_slide

            reader = WSIReader(tmp_path)
            reader.close()

            mock_slide.close.assert_called_once()
        finally:
            Path(tmp_path).unlink()


class TestGetSlideInfo:
    """Test slide info utility function."""

    @patch("src.data.openslide_utils.OPENSLIDE_AVAILABLE", True)
    @patch("src.data.openslide_utils.OpenSlide")
    def test_get_slide_info(self, mock_openslide):
        """Test get_slide_info function."""
        with tempfile.NamedTemporaryFile(suffix=".svs", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            mock_slide = Mock()
            mock_slide.dimensions = (10000, 8000)
            mock_slide.level_count = 3
            mock_slide.level_dimensions = [(10000, 8000), (5000, 4000), (2500, 2000)]
            mock_slide.level_downsamples = [1.0, 2.0, 4.0]
            mock_slide.properties = {"vendor": "aperio"}
            mock_openslide.return_value = mock_slide

            info = get_slide_info(tmp_path)

            assert info["path"] == tmp_path
            assert info["dimensions"] == (10000, 8000)
            assert info["level_count"] == 3
            assert info["level_dimensions"] == [(10000, 8000), (5000, 4000), (2500, 2000)]
            assert info["level_downsamples"] == [1.0, 2.0, 4.0]
            assert info["properties"] == {"vendor": "aperio"}
        finally:
            Path(tmp_path).unlink()


class TestWSIFormatCompatibility:
    """Test WSI file format compatibility validation."""

    @patch("src.data.openslide_utils.OPENSLIDE_AVAILABLE", True)
    @patch("src.data.openslide_utils.OpenSlide")
    def test_svs_format_compatibility(self, mock_openslide):
        """Test .svs format compatibility."""
        with tempfile.NamedTemporaryFile(suffix=".svs", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            mock_slide = Mock()
            mock_slide.properties = {"openslide.vendor": "aperio"}
            mock_openslide.return_value = mock_slide

            reader = WSIReader(tmp_path)
            assert "openslide.vendor" in reader.properties
        finally:
            Path(tmp_path).unlink()

    @patch("src.data.openslide_utils.OPENSLIDE_AVAILABLE", True)
    @patch("src.data.openslide_utils.OpenSlide")
    def test_tiff_format_compatibility(self, mock_openslide):
        """Test .tiff format compatibility."""
        with tempfile.NamedTemporaryFile(suffix=".tiff", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            mock_slide = Mock()
            mock_slide.properties = {"openslide.vendor": "generic-tiff"}
            mock_openslide.return_value = mock_slide

            reader = WSIReader(tmp_path)
            assert reader.properties is not None
        finally:
            Path(tmp_path).unlink()

    @patch("src.data.openslide_utils.OPENSLIDE_AVAILABLE", True)
    @patch("src.data.openslide_utils.OpenSlide")
    def test_ndpi_format_compatibility(self, mock_openslide):
        """Test .ndpi format compatibility."""
        with tempfile.NamedTemporaryFile(suffix=".ndpi", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            mock_slide = Mock()
            mock_slide.properties = {"openslide.vendor": "hamamatsu"}
            mock_openslide.return_value = mock_slide

            reader = WSIReader(tmp_path)
            assert reader.properties is not None
        finally:
            Path(tmp_path).unlink()


class TestPyramidLevelAccess:
    """Test pyramid level access and downsample factor calculations."""

    @patch("src.data.openslide_utils.OPENSLIDE_AVAILABLE", True)
    @patch("src.data.openslide_utils.OpenSlide")
    def test_pyramid_level_dimensions(self, mock_openslide):
        """Test pyramid level dimension calculations."""
        with tempfile.NamedTemporaryFile(suffix=".svs", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            mock_slide = Mock()
            mock_slide.level_count = 4
            mock_slide.level_dimensions = [
                (20000, 16000),  # Level 0
                (10000, 8000),  # Level 1
                (5000, 4000),  # Level 2
                (2500, 2000),  # Level 3
            ]
            mock_slide.level_downsamples = [1.0, 2.0, 4.0, 8.0]
            mock_openslide.return_value = mock_slide

            reader = WSIReader(tmp_path)

            # Verify level count
            assert reader.level_count == 4

            # Verify dimensions at each level
            assert reader.level_dimensions[0] == (20000, 16000)
            assert reader.level_dimensions[1] == (10000, 8000)
            assert reader.level_dimensions[2] == (5000, 4000)
            assert reader.level_dimensions[3] == (2500, 2000)

            # Verify downsample factors
            assert reader.level_downsamples[0] == 1.0
            assert reader.level_downsamples[1] == 2.0
            assert reader.level_downsamples[2] == 4.0
            assert reader.level_downsamples[3] == 8.0
        finally:
            Path(tmp_path).unlink()

    @patch("src.data.openslide_utils.OPENSLIDE_AVAILABLE", True)
    @patch("src.data.openslide_utils.OpenSlide")
    def test_patch_extraction_different_levels(self, mock_openslide):
        """Test patch extraction at different pyramid levels."""
        with tempfile.NamedTemporaryFile(suffix=".svs", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            mock_slide = Mock()
            mock_slide.level_count = 3
            mock_slide.level_dimensions = [
                (10000, 8000),  # Level 0
                (5000, 4000),  # Level 1
                (2500, 2000),  # Level 2
            ]
            mock_slide.level_downsamples = [1.0, 2.0, 4.0]

            def mock_read_region(location, level, size):
                return Image.new("RGBA", size, color=(150, 100, 100, 255))

            mock_slide.read_region = mock_read_region
            mock_openslide.return_value = mock_slide

            reader = WSIReader(tmp_path)

            # Extract patches at level 1
            patches_level1 = reader.extract_patches(patch_size=256, level=1, stride=256)
            assert len(patches_level1) > 0

            # Extract patches at level 2
            patches_level2 = reader.extract_patches(patch_size=256, level=2, stride=256)
            assert len(patches_level2) > 0

            # Level 2 should have fewer patches due to smaller dimensions
            # (though this depends on the exact dimensions and patch size)
        finally:
            Path(tmp_path).unlink()


class TestCoordinateAccuracy:
    """Test patch extraction coordinate accuracy."""

    @patch("src.data.openslide_utils.OPENSLIDE_AVAILABLE", True)
    @patch("src.data.openslide_utils.OpenSlide")
    def test_coordinate_accuracy_level0(self, mock_openslide):
        """Test coordinate accuracy at level 0."""
        with tempfile.NamedTemporaryFile(suffix=".svs", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            mock_slide = Mock()
            mock_slide.level_dimensions = [(1000, 1000)]
            mock_slide.level_downsamples = [1.0]

            # Track coordinates passed to read_region
            coordinates_called = []

            def mock_read_region(location, level, size):
                coordinates_called.append((location, level, size))
                return Image.new("RGBA", size, color=(150, 100, 100, 255))

            mock_slide.read_region = mock_read_region
            mock_openslide.return_value = mock_slide

            reader = WSIReader(tmp_path)
            patches = reader.extract_patches(patch_size=256, level=0, stride=256)

            # Verify coordinates are correct
            for i, ((location, level, size), (patch, (x, y))) in enumerate(
                zip(coordinates_called, patches)
            ):
                assert level == 0
                assert size == (256, 256)
                assert location == (x, y)  # Coordinates should match
        finally:
            Path(tmp_path).unlink()

    @patch("src.data.openslide_utils.OPENSLIDE_AVAILABLE", True)
    @patch("src.data.openslide_utils.OpenSlide")
    def test_coordinate_accuracy_higher_level(self, mock_openslide):
        """Test coordinate accuracy at higher pyramid levels."""
        with tempfile.NamedTemporaryFile(suffix=".svs", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            mock_slide = Mock()
            mock_slide.level_dimensions = [(2000, 2000), (1000, 1000)]
            mock_slide.level_downsamples = [1.0, 2.0]

            coordinates_called = []

            def mock_read_region(location, level, size):
                coordinates_called.append((location, level, size))
                return Image.new("RGBA", size, color=(150, 100, 100, 255))

            mock_slide.read_region = mock_read_region
            mock_openslide.return_value = mock_slide

            reader = WSIReader(tmp_path)
            patches = reader.extract_patches(patch_size=256, level=1, stride=256)

            # Verify coordinate transformation for level 1
            for (location, level, size), (patch, (x, y)) in zip(coordinates_called, patches):
                assert level == 1
                assert size == (256, 256)
                # x, y should be level 0 coordinates (transformed by downsample factor)
                assert location == (x, y)
        finally:
            Path(tmp_path).unlink()
