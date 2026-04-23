"""
Integration tests for WSIReader and PatchExtractor.

This test verifies that the core reading and extraction components work together
correctly for end-to-end patch extraction from WSI files.
"""

import numpy as np
import pytest
from unittest.mock import Mock, patch, MagicMock
from PIL import Image

from src.data.wsi_pipeline.reader import WSIReader
from src.data.wsi_pipeline.extractor import PatchExtractor
from src.data.wsi_pipeline.exceptions import FileFormatError, ProcessingError


class TestWSIReaderExtractorIntegration:
    """Test integration between WSIReader and PatchExtractor."""

    @patch("src.data.wsi_pipeline.reader.OPENSLIDE_AVAILABLE", True)
    @patch("src.data.wsi_pipeline.reader.OpenSlide")
    def test_end_to_end_patch_extraction(self, mock_openslide_class):
        """Test complete workflow: open slide, extract patches, verify RGB format."""
        # Create mock OpenSlide instance
        mock_slide = Mock()
        mock_slide.dimensions = (2048, 2048)
        mock_slide.level_count = 3
        mock_slide.level_dimensions = [(2048, 2048), (1024, 1024), (512, 512)]
        mock_slide.level_downsamples = [1.0, 2.0, 4.0]
        mock_slide.properties = {
            "openslide.objective-power": "40",
            "openslide.mpp-x": "0.25",
            "openslide.mpp-y": "0.25",
        }
        
        # Mock read_region to return RGB patches
        def mock_read_region(location, level, size):
            # Create a mock RGBA image
            img = Image.new("RGBA", size, color=(128, 64, 32, 255))
            return img
        
        mock_slide.read_region = mock_read_region
        mock_openslide_class.return_value = mock_slide
        
        # Create a temporary file path (won't actually be read)
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".svs", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            # Step 1: Open slide with WSIReader
            with WSIReader(tmp_path) as reader:
                # Verify slide properties
                assert reader.dimensions == (2048, 2048)
                assert reader.level_count == 3
                assert len(reader.level_dimensions) == 3
                
                # Step 2: Create PatchExtractor
                extractor = PatchExtractor(patch_size=256, stride=256, level=0)
                
                # Step 3: Generate coordinates
                coords = extractor.generate_coordinates(reader.dimensions)
                assert len(coords) > 0
                
                # Step 4: Extract patches
                patches_extracted = []
                for patch, coord in extractor.extract_patches_streaming(reader, coords[:5]):
                    # Verify patch is RGB format
                    assert patch.shape == (256, 256, 3)
                    assert patch.dtype == np.uint8
                    
                    # Verify patch contains expected values (from our mock)
                    assert np.all(patch[:, :, 0] == 128)  # Red channel
                    assert np.all(patch[:, :, 1] == 64)   # Green channel
                    assert np.all(patch[:, :, 2] == 32)   # Blue channel
                    
                    patches_extracted.append((patch, coord))
                
                # Verify we extracted patches
                assert len(patches_extracted) == 5
                
        finally:
            # Cleanup
            import os
            try:
                os.unlink(tmp_path)
            except:
                pass

    @patch("src.data.wsi_pipeline.reader.OPENSLIDE_AVAILABLE", True)
    @patch("src.data.wsi_pipeline.reader.OpenSlide")
    def test_multi_level_extraction(self, mock_openslide_class):
        """Test extracting patches at different pyramid levels."""
        # Create mock OpenSlide instance
        mock_slide = Mock()
        mock_slide.dimensions = (4096, 4096)
        mock_slide.level_count = 3
        mock_slide.level_dimensions = [(4096, 4096), (2048, 2048), (1024, 1024)]
        mock_slide.level_downsamples = [1.0, 2.0, 4.0]
        mock_slide.properties = {}
        
        # Mock read_region
        def mock_read_region(location, level, size):
            img = Image.new("RGBA", size, color=(100, 100, 100, 255))
            return img
        
        mock_slide.read_region = mock_read_region
        mock_openslide_class.return_value = mock_slide
        
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".svs", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            with WSIReader(tmp_path) as reader:
                # Extract at level 0
                extractor_l0 = PatchExtractor(patch_size=256, level=0)
                coords_l0 = extractor_l0.generate_coordinates(reader.dimensions)
                
                # Extract at level 1
                extractor_l1 = PatchExtractor(patch_size=256, level=1)
                coords_l1 = extractor_l1.generate_coordinates(reader.level_dimensions[1])
                
                # Level 1 should have fewer patches (smaller dimensions)
                assert len(coords_l1) < len(coords_l0)
                
                # Extract one patch from each level
                patch_l0 = extractor_l0.extract_patch(reader, coords_l0[0])
                patch_l1 = extractor_l1.extract_patch(reader, coords_l1[0])
                
                # Both should be RGB with same patch size
                assert patch_l0.shape == (256, 256, 3)
                assert patch_l1.shape == (256, 256, 3)
                
        finally:
            import os
            try:
                os.unlink(tmp_path)
            except:
                pass

    @patch("src.data.wsi_pipeline.reader.OPENSLIDE_AVAILABLE", True)
    @patch("src.data.wsi_pipeline.reader.OpenSlide")
    def test_coordinate_conversion_integration(self, mock_openslide_class):
        """Test coordinate conversion between levels works correctly."""
        # Create mock OpenSlide instance
        mock_slide = Mock()
        mock_slide.dimensions = (2048, 2048)
        mock_slide.level_count = 3
        mock_slide.level_dimensions = [(2048, 2048), (1024, 1024), (512, 512)]
        mock_slide.level_downsamples = [1.0, 2.0, 4.0]
        mock_slide.properties = {}
        
        mock_slide.read_region = lambda loc, lvl, size: Image.new("RGBA", size, color=(50, 50, 50, 255))
        mock_openslide_class.return_value = mock_slide
        
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".svs", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            with WSIReader(tmp_path) as reader:
                extractor = PatchExtractor(patch_size=256)
                
                # Generate coordinates at level 0
                coords_l0 = [(0, 0), (512, 512), (1024, 1024)]
                
                # Convert to level 1
                coords_l1 = extractor.convert_coordinates_to_level(
                    coords_l0,
                    from_level=0,
                    to_level=1,
                    level_downsamples=reader.level_downsamples
                )
                
                # Verify conversion
                assert coords_l1 == [(0, 0), (256, 256), (512, 512)]
                
                # Convert to level 2
                coords_l2 = extractor.convert_coordinates_to_level(
                    coords_l0,
                    from_level=0,
                    to_level=2,
                    level_downsamples=reader.level_downsamples
                )
                
                # Verify conversion
                assert coords_l2 == [(0, 0), (128, 128), (256, 256)]
                
        finally:
            import os
            try:
                os.unlink(tmp_path)
            except:
                pass

    @patch("src.data.wsi_pipeline.reader.OPENSLIDE_AVAILABLE", True)
    @patch("src.data.wsi_pipeline.reader.OpenSlide")
    def test_metadata_extraction(self, mock_openslide_class):
        """Test that WSIReader correctly extracts metadata."""
        # Create mock OpenSlide instance with metadata
        mock_slide = Mock()
        mock_slide.dimensions = (1024, 1024)
        mock_slide.level_count = 2
        mock_slide.level_dimensions = [(1024, 1024), (512, 512)]
        mock_slide.level_downsamples = [1.0, 2.0]
        mock_slide.properties = {
            "openslide.objective-power": "20",
            "openslide.mpp-x": "0.5",
            "openslide.mpp-y": "0.5",
            "openslide.vendor": "Aperio",
            "aperio.Date": "2024-01-15",
        }
        
        mock_openslide_class.return_value = mock_slide
        
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".svs", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            with WSIReader(tmp_path) as reader:
                # Test magnification extraction
                mag = reader.get_magnification()
                assert mag == 20.0
                
                # Test MPP extraction
                mpp = reader.get_mpp()
                assert mpp == (0.5, 0.5)
                
                # Test scanner info extraction
                scanner_info = reader.get_scanner_info()
                assert scanner_info["model"] == "Aperio"
                assert scanner_info["date"] == "2024-01-15"
                
        finally:
            import os
            try:
                os.unlink(tmp_path)
            except:
                pass

    def test_file_not_found_error(self):
        """Test that WSIReader raises appropriate error for missing files."""
        with pytest.raises(FileFormatError, match="WSI file not found"):
            WSIReader("/nonexistent/path/to/slide.svs")

    @patch("src.data.wsi_pipeline.reader.OPENSLIDE_AVAILABLE", False)
    def test_openslide_not_available(self):
        """Test that WSIReader raises error when OpenSlide is not available."""
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".svs", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            with pytest.raises(FileFormatError, match="OpenSlide is not installed"):
                WSIReader(tmp_path)
        finally:
            import os
            try:
                os.unlink(tmp_path)
            except:
                pass


class TestStreamingMemoryEfficiency:
    """Test that streaming extraction is memory efficient."""

    @patch("src.data.wsi_pipeline.reader.OPENSLIDE_AVAILABLE", True)
    @patch("src.data.wsi_pipeline.reader.OpenSlide")
    def test_streaming_does_not_accumulate_patches(self, mock_openslide_class):
        """Test that streaming extraction doesn't accumulate all patches in memory."""
        # Create mock OpenSlide instance
        mock_slide = Mock()
        mock_slide.dimensions = (2048, 2048)
        mock_slide.level_count = 1
        mock_slide.level_dimensions = [(2048, 2048)]
        mock_slide.level_downsamples = [1.0]
        mock_slide.properties = {}
        
        # Track number of patches created
        patches_created = []
        
        def mock_read_region(location, level, size):
            img = Image.new("RGBA", size, color=(100, 100, 100, 255))
            patches_created.append(location)
            return img
        
        mock_slide.read_region = mock_read_region
        mock_openslide_class.return_value = mock_slide
        
        import tempfile
        with tempfile.NamedTemporaryFile(suffix=".svs", delete=False) as tmp:
            tmp_path = tmp.name
        
        try:
            with WSIReader(tmp_path) as reader:
                extractor = PatchExtractor(patch_size=256, stride=256)
                coords = extractor.generate_coordinates(reader.dimensions)
                
                # Process patches one at a time using streaming
                processed_count = 0
                for patch, coord in extractor.extract_patches_streaming(reader, coords[:10]):
                    # Process patch (in real scenario, this would be feature extraction)
                    processed_count += 1
                    # Patch goes out of scope here, can be garbage collected
                
                # Verify we processed patches
                assert processed_count == 10
                assert len(patches_created) == 10
                
        finally:
            import os
            try:
                os.unlink(tmp_path)
            except:
                pass
