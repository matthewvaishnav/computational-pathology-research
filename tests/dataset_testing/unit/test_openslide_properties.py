"""
OpenSlide properties and metadata tests.

This module provides comprehensive tests for OpenSlide slide properties,
metadata access, and WSI file format compatibility validation.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
from typing import Dict, Any, Optional, Tuple, List
from PIL import Image

from src.data.openslide_utils import WSIReader, get_slide_info, check_openslide_available
from tests.dataset_testing.synthetic.wsi_generator import WSISyntheticGenerator, WSISyntheticSpec
from tests.dataset_testing.base_interfaces import ErrorSimulator


class TestOpenSlideProperties:
    """Test OpenSlide slide properties and metadata access."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())
        self.generator = WSISyntheticGenerator(random_seed=42)
        self.error_simulator = ErrorSimulator(self.temp_dir)

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @patch("src.data.openslide_utils.OPENSLIDE_AVAILABLE", True)
    @patch("src.data.openslide_utils.OpenSlide")
    def test_slide_vendor_properties(self, mock_openslide):
        """Test slide vendor-specific properties access."""
        with tempfile.NamedTemporaryFile(suffix=".svs", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Test Aperio vendor properties
            mock_slide = Mock()
            aperio_properties = {
                "openslide.vendor": "aperio",
                "openslide.quickhash-1": "abc123def456",
                "aperio.ImageID": "12345",
                "aperio.Date": "2023/01/15",
                "aperio.Time": "14:30:25",
                "aperio.ScanScope ID": "SS1234",
                "aperio.Filename": "slide001.svs",
                "aperio.MPP": "0.25",
                "aperio.AppMag": "20",
                "aperio.StripeWidth": "2040",
                "aperio.TotalWidth": "46000",
                "aperio.TotalHeight": "32914",
            }
            mock_slide.properties = aperio_properties
            mock_openslide.return_value = mock_slide

            reader = WSIReader(tmp_path)
            properties = reader.properties

            # Verify vendor identification
            assert properties["openslide.vendor"] == "aperio"
            assert "aperio.ImageID" in properties
            assert "aperio.MPP" in properties

            # Verify metadata completeness
            assert properties["aperio.Date"] == "2023/01/15"
            assert properties["aperio.Time"] == "14:30:25"
            assert float(properties["aperio.MPP"]) == 0.25

        finally:
            Path(tmp_path).unlink()

    @patch("src.data.openslide_utils.OPENSLIDE_AVAILABLE", True)
    @patch("src.data.openslide_utils.OpenSlide")
    def test_hamamatsu_vendor_properties(self, mock_openslide):
        """Test Hamamatsu vendor-specific properties."""
        with tempfile.NamedTemporaryFile(suffix=".ndpi", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            mock_slide = Mock()
            hamamatsu_properties = {
                "openslide.vendor": "hamamatsu",
                "openslide.quickhash-1": "xyz789abc123",
                "hamamatsu.SourceLens": "20",
                "hamamatsu.XResolution": "227600",
                "hamamatsu.YResolution": "227600",
                "hamamatsu.Reference": "slide_ref_001",
                "hamamatsu.AuthCode": "auth123456",
                "hamamatsu.SerialNumber": "SN987654321",
            }
            mock_slide.properties = hamamatsu_properties
            mock_openslide.return_value = mock_slide

            reader = WSIReader(tmp_path)
            properties = reader.properties

            # Verify vendor identification
            assert properties["openslide.vendor"] == "hamamatsu"
            assert "hamamatsu.SourceLens" in properties
            assert "hamamatsu.XResolution" in properties

            # Verify resolution properties
            assert int(properties["hamamatsu.XResolution"]) == 227600
            assert int(properties["hamamatsu.YResolution"]) == 227600

        finally:
            Path(tmp_path).unlink()

    @patch("src.data.openslide_utils.OPENSLIDE_AVAILABLE", True)
    @patch("src.data.openslide_utils.OpenSlide")
    def test_generic_tiff_properties(self, mock_openslide):
        """Test generic TIFF properties."""
        with tempfile.NamedTemporaryFile(suffix=".tiff", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            mock_slide = Mock()
            tiff_properties = {
                "openslide.vendor": "generic-tiff",
                "openslide.quickhash-1": "tiff123hash456",
                "tiff.ImageDescription": "Histopathology slide",
                "tiff.XResolution": "96",
                "tiff.YResolution": "96",
                "tiff.ResolutionUnit": "2",  # inches
                "tiff.Software": "ScanSoft v2.1",
                "tiff.DateTime": "2023:01:15 14:30:25",
            }
            mock_slide.properties = tiff_properties
            mock_openslide.return_value = mock_slide

            reader = WSIReader(tmp_path)
            properties = reader.properties

            # Verify vendor identification
            assert properties["openslide.vendor"] == "generic-tiff"
            assert "tiff.ImageDescription" in properties
            assert "tiff.XResolution" in properties

            # Verify TIFF-specific properties
            assert properties["tiff.ImageDescription"] == "Histopathology slide"
            assert properties["tiff.DateTime"] == "2023:01:15 14:30:25"

        finally:
            Path(tmp_path).unlink()

    @patch("src.data.openslide_utils.OPENSLIDE_AVAILABLE", True)
    @patch("src.data.openslide_utils.OpenSlide")
    def test_slide_hash_properties(self, mock_openslide):
        """Test slide hash and identification properties."""
        with tempfile.NamedTemporaryFile(suffix=".svs", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            mock_slide = Mock()
            hash_properties = {
                "openslide.vendor": "aperio",
                "openslide.quickhash-1": "a1b2c3d4e5f6g7h8",
                "openslide.background-color": "ffffff",
                "openslide.bounds-height": "32914",
                "openslide.bounds-width": "46000",
                "openslide.bounds-x": "0",
                "openslide.bounds-y": "0",
                "openslide.comment": "Test slide for validation",
                "openslide.mpp-x": "0.25",
                "openslide.mpp-y": "0.25",
            }
            mock_slide.properties = hash_properties
            mock_openslide.return_value = mock_slide

            reader = WSIReader(tmp_path)
            properties = reader.properties

            # Verify hash properties
            assert "openslide.quickhash-1" in properties
            assert len(properties["openslide.quickhash-1"]) > 0

            # Verify bounds properties
            assert int(properties["openslide.bounds-width"]) == 46000
            assert int(properties["openslide.bounds-height"]) == 32914
            assert int(properties["openslide.bounds-x"]) == 0
            assert int(properties["openslide.bounds-y"]) == 0

            # Verify microns per pixel
            assert float(properties["openslide.mpp-x"]) == 0.25
            assert float(properties["openslide.mpp-y"]) == 0.25

            # Verify background color
            assert properties["openslide.background-color"] == "ffffff"

        finally:
            Path(tmp_path).unlink()

    @patch("src.data.openslide_utils.OPENSLIDE_AVAILABLE", True)
    @patch("src.data.openslide_utils.OpenSlide")
    def test_slide_objective_properties(self, mock_openslide):
        """Test slide objective and magnification properties."""
        with tempfile.NamedTemporaryFile(suffix=".svs", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            mock_slide = Mock()
            objective_properties = {
                "openslide.vendor": "aperio",
                "openslide.objective-power": "20",
                "aperio.AppMag": "20",
                "aperio.MPP": "0.25",
                "aperio.OriginalWidth": "184320",
                "aperio.OriginalHeight": "131656",
                "aperio.Exposure Time": "8",
                "aperio.Exposure Scale": "0.000001",
                "aperio.DisplayColor": "0",
                "aperio.ICC Profile": "ScanScope v1",
            }
            mock_slide.properties = objective_properties
            mock_openslide.return_value = mock_slide

            reader = WSIReader(tmp_path)
            properties = reader.properties

            # Verify objective properties
            assert "openslide.objective-power" in properties
            assert int(properties["openslide.objective-power"]) == 20
            assert int(properties["aperio.AppMag"]) == 20

            # Verify resolution consistency
            assert float(properties["aperio.MPP"]) == 0.25

            # Verify original dimensions
            assert int(properties["aperio.OriginalWidth"]) == 184320
            assert int(properties["aperio.OriginalHeight"]) == 131656

            # Verify exposure properties
            assert int(properties["aperio.Exposure Time"]) == 8
            assert float(properties["aperio.Exposure Scale"]) == 0.000001

        finally:
            Path(tmp_path).unlink()

    @patch("src.data.openslide_utils.OPENSLIDE_AVAILABLE", True)
    @patch("src.data.openslide_utils.OpenSlide")
    def test_missing_properties_handling(self, mock_openslide):
        """Test handling of slides with missing or incomplete properties."""
        with tempfile.NamedTemporaryFile(suffix=".svs", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            mock_slide = Mock()
            minimal_properties = {
                "openslide.vendor": "unknown",
                "openslide.quickhash-1": "minimal123",
            }
            mock_slide.properties = minimal_properties
            mock_openslide.return_value = mock_slide

            reader = WSIReader(tmp_path)
            properties = reader.properties

            # Verify minimal properties are accessible
            assert properties["openslide.vendor"] == "unknown"
            assert "openslide.quickhash-1" in properties

            # Verify graceful handling of missing properties
            assert properties.get("aperio.MPP") is None
            assert properties.get("hamamatsu.SourceLens") is None
            assert properties.get("nonexistent.property") is None

        finally:
            Path(tmp_path).unlink()

    @patch("src.data.openslide_utils.OPENSLIDE_AVAILABLE", True)
    @patch("src.data.openslide_utils.OpenSlide")
    def test_property_data_types(self, mock_openslide):
        """Test proper handling of different property data types."""
        with tempfile.NamedTemporaryFile(suffix=".svs", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            mock_slide = Mock()
            typed_properties = {
                "openslide.vendor": "aperio",
                "string.property": "text_value",
                "integer.property": "12345",
                "float.property": "3.14159",
                "boolean.property": "true",
                "empty.property": "",
                "whitespace.property": "  spaces  ",
            }
            mock_slide.properties = typed_properties
            mock_openslide.return_value = mock_slide

            reader = WSIReader(tmp_path)
            properties = reader.properties

            # Verify all properties are returned as strings (OpenSlide behavior)
            assert isinstance(properties["string.property"], str)
            assert isinstance(properties["integer.property"], str)
            assert isinstance(properties["float.property"], str)
            assert isinstance(properties["boolean.property"], str)

            # Verify values are preserved
            assert properties["string.property"] == "text_value"
            assert properties["integer.property"] == "12345"
            assert properties["float.property"] == "3.14159"
            assert properties["boolean.property"] == "true"

            # Verify empty and whitespace handling
            assert properties["empty.property"] == ""
            assert properties["whitespace.property"] == "  spaces  "

        finally:
            Path(tmp_path).unlink()


class TestOpenSlideFormatCompatibility:
    """Test WSI file format compatibility validation."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @patch("src.data.openslide_utils.OPENSLIDE_AVAILABLE", True)
    @patch("src.data.openslide_utils.OpenSlide")
    def test_svs_format_validation(self, mock_openslide):
        """Test .svs format compatibility and validation."""
        with tempfile.NamedTemporaryFile(suffix=".svs", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            mock_slide = Mock()
            mock_slide.properties = {
                "openslide.vendor": "aperio",
                "openslide.quickhash-1": "svs123hash",
                "aperio.ImageID": "SVS001",
            }
            mock_slide.dimensions = (46000, 32914)
            mock_slide.level_count = 4
            mock_openslide.return_value = mock_slide

            # Test format validation function
            def validate_wsi_format(file_path: Path) -> Dict[str, Any]:
                """Validate WSI file format compatibility."""
                validation_result = {
                    "compatible": True,
                    "format": None,
                    "vendor": None,
                    "issues": [],
                    "recommendations": [],
                }

                try:
                    reader = WSIReader(str(file_path))
                    properties = reader.properties

                    # Determine format from file extension
                    file_ext = file_path.suffix.lower()
                    validation_result["format"] = file_ext

                    # Determine vendor
                    vendor = properties.get("openslide.vendor", "unknown")
                    validation_result["vendor"] = vendor

                    # Validate SVS format
                    if file_ext == ".svs":
                        if vendor != "aperio":
                            validation_result["issues"].append(
                                f"SVS file has unexpected vendor: {vendor}"
                            )

                        if "aperio.ImageID" not in properties:
                            validation_result["issues"].append("SVS file missing Aperio ImageID")

                    reader.close()

                except Exception as e:
                    validation_result["compatible"] = False
                    validation_result["issues"].append(f"Format validation error: {str(e)}")

                return validation_result

            result = validate_wsi_format(Path(tmp_path))

            assert result["compatible"] is True
            assert result["format"] == ".svs"
            assert result["vendor"] == "aperio"
            assert len(result["issues"]) == 0

        finally:
            Path(tmp_path).unlink()

    @patch("src.data.openslide_utils.OPENSLIDE_AVAILABLE", True)
    @patch("src.data.openslide_utils.OpenSlide")
    def test_ndpi_format_validation(self, mock_openslide):
        """Test .ndpi format compatibility and validation."""
        with tempfile.NamedTemporaryFile(suffix=".ndpi", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            mock_slide = Mock()
            mock_slide.properties = {
                "openslide.vendor": "hamamatsu",
                "openslide.quickhash-1": "ndpi456hash",
                "hamamatsu.SourceLens": "20",
            }
            mock_slide.dimensions = (92160, 65792)
            mock_slide.level_count = 8
            mock_openslide.return_value = mock_slide

            def validate_wsi_format(file_path: Path) -> Dict[str, Any]:
                """Validate WSI file format compatibility."""
                validation_result = {
                    "compatible": True,
                    "format": None,
                    "vendor": None,
                    "issues": [],
                    "recommendations": [],
                }

                try:
                    reader = WSIReader(str(file_path))
                    properties = reader.properties

                    file_ext = file_path.suffix.lower()
                    validation_result["format"] = file_ext

                    vendor = properties.get("openslide.vendor", "unknown")
                    validation_result["vendor"] = vendor

                    # Validate NDPI format
                    if file_ext == ".ndpi":
                        if vendor != "hamamatsu":
                            validation_result["issues"].append(
                                f"NDPI file has unexpected vendor: {vendor}"
                            )

                        if "hamamatsu.SourceLens" not in properties:
                            validation_result["issues"].append(
                                "NDPI file missing Hamamatsu SourceLens"
                            )

                    reader.close()

                except Exception as e:
                    validation_result["compatible"] = False
                    validation_result["issues"].append(f"Format validation error: {str(e)}")

                return validation_result

            result = validate_wsi_format(Path(tmp_path))

            assert result["compatible"] is True
            assert result["format"] == ".ndpi"
            assert result["vendor"] == "hamamatsu"
            assert len(result["issues"]) == 0

        finally:
            Path(tmp_path).unlink()

    @patch("src.data.openslide_utils.OPENSLIDE_AVAILABLE", True)
    @patch("src.data.openslide_utils.OpenSlide")
    def test_tiff_format_validation(self, mock_openslide):
        """Test .tiff format compatibility and validation."""
        with tempfile.NamedTemporaryFile(suffix=".tiff", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            mock_slide = Mock()
            mock_slide.properties = {
                "openslide.vendor": "generic-tiff",
                "openslide.quickhash-1": "tiff789hash",
                "tiff.ImageDescription": "Generic WSI",
            }
            mock_slide.dimensions = (20480, 15360)
            mock_slide.level_count = 1
            mock_openslide.return_value = mock_slide

            def validate_wsi_format(file_path: Path) -> Dict[str, Any]:
                """Validate WSI file format compatibility."""
                validation_result = {
                    "compatible": True,
                    "format": None,
                    "vendor": None,
                    "issues": [],
                    "recommendations": [],
                }

                try:
                    reader = WSIReader(str(file_path))
                    properties = reader.properties

                    file_ext = file_path.suffix.lower()
                    validation_result["format"] = file_ext

                    vendor = properties.get("openslide.vendor", "unknown")
                    validation_result["vendor"] = vendor

                    # Validate TIFF format
                    if file_ext in [".tiff", ".tif"]:
                        if vendor == "generic-tiff":
                            validation_result["recommendations"].append(
                                "Generic TIFF may have limited pyramid support"
                            )

                        if reader.level_count == 1:
                            validation_result["recommendations"].append(
                                "Single-level TIFF may impact performance for large images"
                            )

                    reader.close()

                except Exception as e:
                    validation_result["compatible"] = False
                    validation_result["issues"].append(f"Format validation error: {str(e)}")

                return validation_result

            result = validate_wsi_format(Path(tmp_path))

            assert result["compatible"] is True
            assert result["format"] == ".tiff"
            assert result["vendor"] == "generic-tiff"
            assert len(result["recommendations"]) > 0

        finally:
            Path(tmp_path).unlink()

    def test_unsupported_format_handling(self):
        """Test handling of unsupported file formats."""
        # Create unsupported file
        unsupported_file = self.temp_dir / "test.jpg"
        with open(unsupported_file, "wb") as f:
            f.write(b"fake jpeg content")

        def validate_wsi_format(file_path: Path) -> Dict[str, Any]:
            """Validate WSI file format compatibility."""
            validation_result = {
                "compatible": True,
                "format": None,
                "vendor": None,
                "issues": [],
                "recommendations": [],
            }

            file_ext = file_path.suffix.lower()
            validation_result["format"] = file_ext

            # Check for supported formats
            supported_formats = [".svs", ".ndpi", ".tiff", ".tif", ".vms", ".vmu", ".scn"]

            if file_ext not in supported_formats:
                validation_result["compatible"] = False
                validation_result["issues"].append(f"Unsupported file format: {file_ext}")
                validation_result["recommendations"].append(
                    f"Convert to supported format: {', '.join(supported_formats)}"
                )

            return validation_result

        result = validate_wsi_format(unsupported_file)

        assert result["compatible"] is False
        assert result["format"] == ".jpg"
        assert len(result["issues"]) > 0
        assert len(result["recommendations"]) > 0


class TestSlideInfoUtility:
    """Test slide info utility function comprehensive functionality."""

    def setup_method(self):
        """Set up test fixtures."""
        self.temp_dir = Path(tempfile.mkdtemp())

    def teardown_method(self):
        """Clean up test fixtures."""
        import shutil

        if self.temp_dir.exists():
            shutil.rmtree(self.temp_dir)

    @patch("src.data.openslide_utils.OPENSLIDE_AVAILABLE", True)
    @patch("src.data.openslide_utils.OpenSlide")
    def test_comprehensive_slide_info(self, mock_openslide):
        """Test comprehensive slide information extraction."""
        with tempfile.NamedTemporaryFile(suffix=".svs", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            mock_slide = Mock()
            mock_slide.dimensions = (46000, 32914)
            mock_slide.level_count = 4
            mock_slide.level_dimensions = [
                (46000, 32914),
                (23000, 16457),
                (11500, 8228),
                (5750, 4114),
            ]
            mock_slide.level_downsamples = [1.0, 2.0, 4.0, 8.0]
            mock_slide.properties = {
                "openslide.vendor": "aperio",
                "openslide.quickhash-1": "comprehensive123",
                "openslide.mpp-x": "0.25",
                "openslide.mpp-y": "0.25",
                "aperio.ImageID": "COMP001",
                "aperio.Date": "2023/01/15",
                "aperio.AppMag": "20",
            }
            mock_openslide.return_value = mock_slide

            info = get_slide_info(tmp_path)

            # Verify basic information
            assert info["path"] == tmp_path
            assert info["dimensions"] == (46000, 32914)
            assert info["level_count"] == 4

            # Verify pyramid information
            assert len(info["level_dimensions"]) == 4
            assert info["level_dimensions"][0] == (46000, 32914)
            assert info["level_dimensions"][3] == (5750, 4114)

            assert len(info["level_downsamples"]) == 4
            assert info["level_downsamples"][0] == 1.0
            assert info["level_downsamples"][3] == 8.0

            # Verify properties
            assert info["properties"]["openslide.vendor"] == "aperio"
            assert info["properties"]["aperio.ImageID"] == "COMP001"
            assert info["properties"]["aperio.Date"] == "2023/01/15"

        finally:
            Path(tmp_path).unlink()

    @patch("src.data.openslide_utils.OPENSLIDE_AVAILABLE", True)
    @patch("src.data.openslide_utils.OpenSlide")
    def test_slide_info_with_calculations(self, mock_openslide):
        """Test slide info with derived calculations."""
        with tempfile.NamedTemporaryFile(suffix=".svs", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            mock_slide = Mock()
            mock_slide.dimensions = (40000, 30000)
            mock_slide.level_count = 3
            mock_slide.level_dimensions = [(40000, 30000), (20000, 15000), (10000, 7500)]
            mock_slide.level_downsamples = [1.0, 2.0, 4.0]
            mock_slide.properties = {
                "openslide.vendor": "aperio",
                "openslide.mpp-x": "0.5",
                "openslide.mpp-y": "0.5",
                "aperio.AppMag": "10",
            }
            mock_openslide.return_value = mock_slide

            def get_enhanced_slide_info(wsi_path: str) -> Dict[str, Any]:
                """Get enhanced slide information with calculations."""
                base_info = get_slide_info(wsi_path)

                # Add calculated fields
                width, height = base_info["dimensions"]
                mpp_x = float(base_info["properties"].get("openslide.mpp-x", 0))
                mpp_y = float(base_info["properties"].get("openslide.mpp-y", 0))

                enhanced_info = base_info.copy()
                enhanced_info["calculated"] = {
                    "total_pixels": width * height,
                    "aspect_ratio": width / height if height > 0 else 0,
                    "physical_size_mm": {
                        "width": (width * mpp_x) / 1000 if mpp_x > 0 else None,
                        "height": (height * mpp_y) / 1000 if mpp_y > 0 else None,
                    },
                    "pyramid_reduction_factors": base_info["level_downsamples"],
                    "level_pixel_counts": [w * h for w, h in base_info["level_dimensions"]],
                }

                return enhanced_info

            enhanced_info = get_enhanced_slide_info(tmp_path)

            # Verify calculations
            assert enhanced_info["calculated"]["total_pixels"] == 40000 * 30000
            assert enhanced_info["calculated"]["aspect_ratio"] == 40000 / 30000

            # Verify physical size calculations
            physical_size = enhanced_info["calculated"]["physical_size_mm"]
            assert physical_size["width"] == (40000 * 0.5) / 1000  # 20 mm
            assert physical_size["height"] == (30000 * 0.5) / 1000  # 15 mm

            # Verify pyramid calculations
            level_counts = enhanced_info["calculated"]["level_pixel_counts"]
            assert level_counts[0] == 40000 * 30000
            assert level_counts[1] == 20000 * 15000
            assert level_counts[2] == 10000 * 7500

        finally:
            Path(tmp_path).unlink()
