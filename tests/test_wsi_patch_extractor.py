"""
Unit tests for PatchExtractor class.

Tests cover:
- Initialization with various parameters
- Grid coordinate generation
- Coordinate conversion between pyramid levels
- Patch extraction
- Streaming extraction
- Edge cases and error handling
"""

import numpy as np
import pytest

from src.data.wsi_pipeline.extractor import PatchExtractor
from src.data.wsi_pipeline.exceptions import ProcessingError


class TestPatchExtractorInitialization:
    """Test PatchExtractor initialization."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        extractor = PatchExtractor()
        assert extractor.patch_size == 256
        assert extractor.stride == 256  # Defaults to patch_size
        assert extractor.level == 0
        assert extractor.target_mpp is None

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        extractor = PatchExtractor(
            patch_size=512, stride=256, level=1, target_mpp=0.5
        )
        assert extractor.patch_size == 512
        assert extractor.stride == 256
        assert extractor.level == 1
        assert extractor.target_mpp == 0.5

    def test_init_stride_defaults_to_patch_size(self):
        """Test that stride defaults to patch_size when not specified."""
        extractor = PatchExtractor(patch_size=512)
        assert extractor.stride == 512

    def test_init_invalid_patch_size_too_small(self):
        """Test that patch_size below 64 raises ValueError."""
        with pytest.raises(ValueError, match="patch_size must be between 64 and 2048"):
            PatchExtractor(patch_size=32)

    def test_init_invalid_patch_size_too_large(self):
        """Test that patch_size above 2048 raises ValueError."""
        with pytest.raises(ValueError, match="patch_size must be between 64 and 2048"):
            PatchExtractor(patch_size=4096)

    def test_init_invalid_stride_zero(self):
        """Test that stride of 0 raises ValueError."""
        with pytest.raises(ValueError, match="stride must be positive"):
            PatchExtractor(patch_size=256, stride=0)

    def test_init_invalid_stride_negative(self):
        """Test that negative stride raises ValueError."""
        with pytest.raises(ValueError, match="stride must be positive"):
            PatchExtractor(patch_size=256, stride=-128)


class TestCoordinateGeneration:
    """Test coordinate generation functionality."""

    def test_generate_coordinates_non_overlapping(self):
        """Test grid generation with non-overlapping patches."""
        extractor = PatchExtractor(patch_size=256, stride=256)
        coords = extractor.generate_coordinates((1024, 1024))

        # Should generate 4x4 grid = 16 patches
        assert len(coords) == 16

        # Check first and last coordinates
        assert coords[0] == (0, 0)
        assert coords[-1] == (768, 768)

    def test_generate_coordinates_overlapping(self):
        """Test grid generation with 50% overlapping patches."""
        extractor = PatchExtractor(patch_size=256, stride=128)
        coords = extractor.generate_coordinates((1024, 1024))

        # With stride=128, should generate 7x7 grid = 49 patches
        # (0, 128, 256, 384, 512, 640, 768) in each dimension
        assert len(coords) == 49

        # Check stride between consecutive patches
        assert coords[1][0] - coords[0][0] == 128

    def test_generate_coordinates_small_slide(self):
        """Test coordinate generation for small slide."""
        extractor = PatchExtractor(patch_size=256, stride=256)
        coords = extractor.generate_coordinates((512, 512))

        # Should generate 2x2 grid = 4 patches
        assert len(coords) == 4
        assert coords == [(0, 0), (256, 0), (0, 256), (256, 256)]

    def test_generate_coordinates_exact_fit(self):
        """Test coordinate generation when slide is exact multiple of patch_size."""
        extractor = PatchExtractor(patch_size=256, stride=256)
        coords = extractor.generate_coordinates((768, 768))

        # Should generate 3x3 grid = 9 patches
        assert len(coords) == 9

    def test_generate_coordinates_with_tissue_mask(self):
        """Test coordinate filtering with tissue mask."""
        extractor = PatchExtractor(patch_size=256, stride=256)

        # Create tissue mask with tissue only in top-left quadrant
        tissue_mask = np.zeros((100, 100), dtype=bool)
        tissue_mask[:50, :50] = True

        coords = extractor.generate_coordinates((1024, 1024), tissue_mask=tissue_mask)

        # Should only generate coordinates in top-left quadrant
        assert len(coords) > 0
        assert len(coords) < 16  # Less than full grid

        # All coordinates should be in top-left region
        for x, y in coords:
            assert x < 512 and y < 512

    def test_generate_coordinates_empty_tissue_mask(self):
        """Test coordinate generation with empty tissue mask."""
        extractor = PatchExtractor(patch_size=256, stride=256)

        # Create empty tissue mask (all background)
        tissue_mask = np.zeros((100, 100), dtype=bool)

        coords = extractor.generate_coordinates((1024, 1024), tissue_mask=tissue_mask)

        # Should generate no coordinates
        assert len(coords) == 0

    def test_generate_coordinates_full_tissue_mask(self):
        """Test coordinate generation with full tissue mask."""
        extractor = PatchExtractor(patch_size=256, stride=256)

        # Create full tissue mask (all tissue)
        tissue_mask = np.ones((100, 100), dtype=bool)

        coords = extractor.generate_coordinates((1024, 1024), tissue_mask=tissue_mask)

        # Should generate all coordinates (same as without mask)
        assert len(coords) == 16


class TestCoordinateConversion:
    """Test coordinate conversion between pyramid levels."""

    def test_convert_coordinates_same_level(self):
        """Test conversion when from_level equals to_level."""
        extractor = PatchExtractor()
        coords = [(0, 0), (256, 256), (512, 512)]
        downsamples = [1.0, 2.0, 4.0]

        converted = extractor.convert_coordinates_to_level(
            coords, from_level=0, to_level=0, level_downsamples=downsamples
        )

        # Should return unchanged coordinates
        assert converted == coords

    def test_convert_coordinates_level0_to_level1(self):
        """Test conversion from level 0 to level 1 (2x downsample)."""
        extractor = PatchExtractor()
        coords = [(0, 0), (256, 0), (512, 0)]
        downsamples = [1.0, 2.0, 4.0]

        converted = extractor.convert_coordinates_to_level(
            coords, from_level=0, to_level=1, level_downsamples=downsamples
        )

        # Coordinates should be halved
        assert converted == [(0, 0), (128, 0), (256, 0)]

    def test_convert_coordinates_level0_to_level2(self):
        """Test conversion from level 0 to level 2 (4x downsample)."""
        extractor = PatchExtractor()
        coords = [(0, 0), (256, 0), (512, 0), (1024, 1024)]
        downsamples = [1.0, 2.0, 4.0]

        converted = extractor.convert_coordinates_to_level(
            coords, from_level=0, to_level=2, level_downsamples=downsamples
        )

        # Coordinates should be quartered
        assert converted == [(0, 0), (64, 0), (128, 0), (256, 256)]

    def test_convert_coordinates_level1_to_level0(self):
        """Test conversion from level 1 to level 0 (upsampling)."""
        extractor = PatchExtractor()
        coords = [(0, 0), (128, 0), (256, 0)]
        downsamples = [1.0, 2.0, 4.0]

        converted = extractor.convert_coordinates_to_level(
            coords, from_level=1, to_level=0, level_downsamples=downsamples
        )

        # Coordinates should be doubled
        assert converted == [(0, 0), (256, 0), (512, 0)]

    def test_convert_coordinates_invalid_from_level(self):
        """Test conversion with invalid from_level."""
        extractor = PatchExtractor()
        coords = [(0, 0)]
        downsamples = [1.0, 2.0, 4.0]

        with pytest.raises(ValueError, match="Invalid level"):
            extractor.convert_coordinates_to_level(
                coords, from_level=5, to_level=0, level_downsamples=downsamples
            )

    def test_convert_coordinates_invalid_to_level(self):
        """Test conversion with invalid to_level."""
        extractor = PatchExtractor()
        coords = [(0, 0)]
        downsamples = [1.0, 2.0, 4.0]

        with pytest.raises(ValueError, match="Invalid level"):
            extractor.convert_coordinates_to_level(
                coords, from_level=0, to_level=5, level_downsamples=downsamples
            )


class TestPatchExtraction:
    """Test patch extraction functionality."""

    def test_extract_patch_validates_negative_coordinates(self):
        """Test that negative coordinates raise ProcessingError."""
        extractor = PatchExtractor(patch_size=256)

        # Create mock reader
        class MockReader:
            @property
            def dimensions(self):
                return (1024, 1024)

        reader = MockReader()

        with pytest.raises(ProcessingError, match="Coordinates out of bounds"):
            extractor.extract_patch(reader, (-10, 0))

    def test_extract_patch_validates_out_of_bounds(self):
        """Test that out-of-bounds coordinates raise ProcessingError."""
        extractor = PatchExtractor(patch_size=256)

        # Create mock reader
        class MockReader:
            @property
            def dimensions(self):
                return (1024, 1024)

        reader = MockReader()

        # Patch would extend beyond slide boundary
        with pytest.raises(ProcessingError, match="exceeds slide dimensions"):
            extractor.extract_patch(reader, (900, 900))

    def test_extract_patch_success(self):
        """Test successful patch extraction."""
        extractor = PatchExtractor(patch_size=256)

        # Create mock reader
        class MockReader:
            @property
            def dimensions(self):
                return (1024, 1024)

            def read_region(self, location, level, size):
                # Return mock patch
                return np.zeros((size[1], size[0], 3), dtype=np.uint8)

        reader = MockReader()
        patch = extractor.extract_patch(reader, (0, 0))

        assert patch.shape == (256, 256, 3)

    def test_extract_patch_validates_dimensions(self):
        """Test that incorrect patch dimensions raise ProcessingError."""
        extractor = PatchExtractor(patch_size=256)

        # Create mock reader that returns wrong size
        class MockReader:
            @property
            def dimensions(self):
                return (1024, 1024)

            def read_region(self, location, level, size):
                # Return wrong size
                return np.zeros((128, 128, 3), dtype=np.uint8)

        reader = MockReader()

        with pytest.raises(ProcessingError, match="incorrect dimensions"):
            extractor.extract_patch(reader, (0, 0))


class TestStreamingExtraction:
    """Test streaming patch extraction."""

    def test_extract_patches_streaming_yields_patches(self):
        """Test that streaming extraction yields patches correctly."""
        extractor = PatchExtractor(patch_size=256)

        # Create mock reader
        class MockReader:
            @property
            def dimensions(self):
                return (1024, 1024)

            def read_region(self, location, level, size):
                return np.zeros((size[1], size[0], 3), dtype=np.uint8)

        reader = MockReader()
        coords = [(0, 0), (256, 0), (512, 0)]

        patches = list(extractor.extract_patches_streaming(reader, coords))

        assert len(patches) == 3
        for patch, coord in patches:
            assert patch.shape == (256, 256, 3)
            assert coord in coords

    def test_extract_patches_streaming_skips_failed_patches(self):
        """Test that streaming extraction skips patches that fail."""
        extractor = PatchExtractor(patch_size=256)

        # Create mock reader that fails on second patch
        class MockReader:
            call_count = 0

            @property
            def dimensions(self):
                return (1024, 1024)

            def read_region(self, location, level, size):
                self.call_count += 1
                if self.call_count == 2:
                    raise Exception("Simulated read failure")
                return np.zeros((size[1], size[0], 3), dtype=np.uint8)

        reader = MockReader()
        coords = [(0, 0), (256, 0), (512, 0)]

        patches = list(extractor.extract_patches_streaming(reader, coords))

        # Should yield 2 patches (skipping the failed one)
        assert len(patches) == 2

    def test_extract_patches_streaming_empty_coordinates(self):
        """Test streaming extraction with empty coordinate list."""
        extractor = PatchExtractor(patch_size=256)

        class MockReader:
            @property
            def dimensions(self):
                return (1024, 1024)

        reader = MockReader()
        coords = []

        patches = list(extractor.extract_patches_streaming(reader, coords))

        assert len(patches) == 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_large_stride_generates_sparse_grid(self):
        """Test that large stride generates sparse coordinate grid."""
        extractor = PatchExtractor(patch_size=256, stride=512)
        coords = extractor.generate_coordinates((2048, 2048))

        # With stride=512, should generate fewer patches
        # (0, 512, 1024, 1536) in each dimension = 4x4 = 16 patches
        assert len(coords) == 16

    def test_small_stride_generates_dense_grid(self):
        """Test that small stride generates dense coordinate grid."""
        extractor = PatchExtractor(patch_size=256, stride=64)
        coords = extractor.generate_coordinates((1024, 1024))

        # With stride=64, should generate many patches
        # (1024 - 256) / 64 + 1 = 13 positions per dimension
        # 13 x 13 = 169 patches
        assert len(coords) == 169

    def test_patch_size_larger_than_slide(self):
        """Test coordinate generation when patch_size > slide dimensions."""
        extractor = PatchExtractor(patch_size=512, stride=512)
        coords = extractor.generate_coordinates((256, 256))

        # Should generate no coordinates (patch doesn't fit)
        assert len(coords) == 0

    def test_non_square_slide_dimensions(self):
        """Test coordinate generation with non-square slide."""
        extractor = PatchExtractor(patch_size=256, stride=256)
        coords = extractor.generate_coordinates((1024, 512))

        # Should generate 4x2 grid = 8 patches
        assert len(coords) == 8

        # Check that coordinates respect boundaries
        for x, y in coords:
            assert x <= 1024 - 256
            assert y <= 512 - 256
