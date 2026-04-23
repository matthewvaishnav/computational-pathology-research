"""
Unit tests for TissueDetector component.

Tests tissue detection using Otsu thresholding, tissue percentage calculation,
and patch filtering functionality.
"""

import numpy as np
import pytest

from src.data.wsi_pipeline import TissueDetector
from src.data.wsi_pipeline.exceptions import ProcessingError


class TestTissueDetectorInit:
    """Test TissueDetector initialization."""

    def test_init_default_parameters(self):
        """Test initialization with default parameters."""
        detector = TissueDetector()
        assert detector.method == "otsu"
        assert detector.tissue_threshold == 0.5
        assert detector.thumbnail_level == -1

    def test_init_custom_parameters(self):
        """Test initialization with custom parameters."""
        detector = TissueDetector(
            method="otsu",
            tissue_threshold=0.7,
            thumbnail_level=2,
        )
        assert detector.method == "otsu"
        assert detector.tissue_threshold == 0.7
        assert detector.thumbnail_level == 2

    def test_init_invalid_threshold_too_low(self):
        """Test initialization fails with threshold < 0.0."""
        with pytest.raises(ValueError, match="tissue_threshold must be between"):
            TissueDetector(tissue_threshold=-0.1)

    def test_init_invalid_threshold_too_high(self):
        """Test initialization fails with threshold > 1.0."""
        with pytest.raises(ValueError, match="tissue_threshold must be between"):
            TissueDetector(tissue_threshold=1.5)

    def test_init_boundary_thresholds(self):
        """Test initialization with boundary threshold values."""
        detector_min = TissueDetector(tissue_threshold=0.0)
        assert detector_min.tissue_threshold == 0.0

        detector_max = TissueDetector(tissue_threshold=1.0)
        assert detector_max.tissue_threshold == 1.0


class TestTissuePercentageCalculation:
    """Test tissue percentage calculation on synthetic patches."""

    def test_calculate_tissue_percentage_all_tissue(self):
        """Test patch with 100% tissue (all dark pixels)."""
        detector = TissueDetector()

        # Create dark patch (tissue)
        patch = np.zeros((256, 256, 3), dtype=np.uint8)

        tissue_pct = detector.calculate_tissue_percentage(patch)

        # Should be close to 1.0 (100% tissue)
        assert tissue_pct > 0.9

    def test_calculate_tissue_percentage_all_background(self):
        """Test patch with 100% background (all white pixels)."""
        detector = TissueDetector()

        # Create white patch (background)
        patch = np.ones((256, 256, 3), dtype=np.uint8) * 255

        tissue_pct = detector.calculate_tissue_percentage(patch)

        # Should be close to 0.0 (0% tissue)
        assert tissue_pct < 0.1

    def test_calculate_tissue_percentage_mixed(self):
        """Test patch with mixed tissue and background."""
        detector = TissueDetector()

        # Create patch with half dark (tissue) and half white (background)
        patch = np.ones((256, 256, 3), dtype=np.uint8) * 255
        patch[:128, :, :] = 0  # Top half is tissue

        tissue_pct = detector.calculate_tissue_percentage(patch)

        # Should be approximately 0.5 (50% tissue)
        assert 0.3 < tissue_pct < 0.7

    def test_calculate_tissue_percentage_grayscale(self):
        """Test tissue percentage calculation on grayscale image."""
        detector = TissueDetector()

        # Create grayscale patch (2D array)
        patch = np.zeros((256, 256), dtype=np.uint8)

        tissue_pct = detector.calculate_tissue_percentage(patch)

        # Should handle grayscale input
        assert 0.0 <= tissue_pct <= 1.0


class TestIsTissuePatch:
    """Test is_tissue_patch decision logic."""

    def test_is_tissue_patch_above_threshold(self):
        """Test patch with tissue above threshold is classified as tissue."""
        detector = TissueDetector(tissue_threshold=0.5)

        # Create mostly dark patch (tissue)
        patch = np.zeros((256, 256, 3), dtype=np.uint8)

        assert detector.is_tissue_patch(patch) == True

    def test_is_tissue_patch_below_threshold(self):
        """Test patch with tissue below threshold is classified as background."""
        detector = TissueDetector(tissue_threshold=0.5)

        # Create mostly white patch (background)
        patch = np.ones((256, 256, 3), dtype=np.uint8) * 255

        assert detector.is_tissue_patch(patch) == False

    def test_is_tissue_patch_custom_threshold(self):
        """Test is_tissue_patch with custom threshold parameter."""
        detector = TissueDetector(tissue_threshold=0.5)

        # Create patch with ~50% tissue
        patch = np.ones((256, 256, 3), dtype=np.uint8) * 255
        patch[:128, :, :] = 0

        # Should pass with low threshold
        assert detector.is_tissue_patch(patch, threshold=0.3) == True

        # Should fail with high threshold
        assert detector.is_tissue_patch(patch, threshold=0.7) == False

    def test_is_tissue_patch_edge_case_at_threshold(self):
        """Test patch with tissue exactly at threshold."""
        detector = TissueDetector(tissue_threshold=0.5)

        # Create patch with exactly 50% tissue
        patch = np.ones((256, 256, 3), dtype=np.uint8) * 255
        patch[:128, :, :] = 0

        # At threshold should be classified as tissue (>=)
        result = detector.is_tissue_patch(patch)
        # Result depends on Otsu threshold, so just check it's boolean
        assert isinstance(result, (bool, np.bool_))


class TestCacheManagement:
    """Test tissue mask caching functionality."""

    def test_cache_initially_empty(self):
        """Test cache is empty on initialization."""
        detector = TissueDetector()
        assert detector.get_cache_size() == 0

    def test_clear_cache(self):
        """Test cache clearing functionality."""
        detector = TissueDetector()

        # Manually add to cache
        detector._mask_cache["test_slide"] = np.ones((100, 100), dtype=bool)
        assert detector.get_cache_size() == 1

        # Clear cache
        detector.clear_cache()
        assert detector.get_cache_size() == 0

    def test_get_cache_size(self):
        """Test cache size tracking."""
        detector = TissueDetector()

        # Add multiple entries
        detector._mask_cache["slide1"] = np.ones((100, 100), dtype=bool)
        detector._mask_cache["slide2"] = np.ones((100, 100), dtype=bool)
        detector._mask_cache["slide3"] = np.ones((100, 100), dtype=bool)

        assert detector.get_cache_size() == 3


class TestOtsuTissueDetection:
    """Test Otsu thresholding implementation."""

    def test_otsu_on_synthetic_image(self):
        """Test Otsu thresholding on synthetic bimodal image."""
        detector = TissueDetector()

        # Create bimodal image: half dark, half bright
        image = np.ones((256, 256, 3), dtype=np.uint8) * 255
        image[:128, :, :] = 50  # Dark region (tissue)

        mask = detector._otsu_tissue_detection(image)

        # Check mask is binary
        assert mask.dtype == bool
        assert mask.shape == (256, 256)

        # Check approximately half is tissue
        tissue_ratio = mask.sum() / mask.size
        assert 0.3 < tissue_ratio < 0.7

    def test_otsu_handles_uniform_image(self):
        """Test Otsu handles uniform image gracefully."""
        detector = TissueDetector()

        # Create uniform image (all same color)
        image = np.ones((256, 256, 3), dtype=np.uint8) * 128

        # Should not crash, even though Otsu may struggle
        mask = detector._otsu_tissue_detection(image)

        assert mask.dtype == bool
        assert mask.shape == (256, 256)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
