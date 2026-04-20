"""
Property-based tests for OpenSlide integration.

This module provides property-based tests for OpenSlide functionality,
focusing on coordinate and alignment preservation during patch extraction.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch as mock_patch

import numpy as np
from hypothesis import assume, given, settings
from hypothesis import strategies as st
from PIL import Image

from src.data.openslide_utils import WSIReader
from tests.dataset_testing.hypothesis_strategies import (
    patch_size_strategy,
)


def is_ci_environment():
    """
    Detect if running in a CI environment.

    Returns:
        bool: True if running in CI, False otherwise
    """
    ci_indicators = [
        os.getenv("CI") == "true",
        os.getenv("GITHUB_ACTIONS") == "true",
        os.getenv("TRAVIS") == "true",
        os.getenv("JENKINS_URL") is not None,
        os.getenv("BUILDKITE") == "true",
        os.getenv("CIRCLECI") == "true",
        os.getenv("RUNNER_OS") is not None,  # GitHub Actions specific
    ]
    return any(ci_indicators)


def get_test_config():
    """
    Get test configuration based on environment.

    Returns:
        dict: Configuration parameters for the current environment
    """
    if is_ci_environment():
        return {
            "max_examples": 20,
            "max_slide_dimension": 10000,
            "deadline": 30000,  # 30 seconds
            "environment": "ci",
        }
    else:
        return {
            "max_examples": 100,
            "max_slide_dimension": 50000,
            "deadline": 60000,  # 60 seconds
            "environment": "local",
        }


def ci_aware_settings(func):
    """
    Decorator to apply CI-aware Hypothesis settings to property-based tests.

    This decorator automatically configures test parameters based on the
    execution environment to prevent CI timeouts while maintaining
    comprehensive testing in local development.
    """
    config = get_test_config()

    # Apply settings with environment-specific parameters
    return settings(max_examples=config["max_examples"], deadline=config["deadline"])(func)


class TestOpenSlideProperties:
    """Property-based tests for OpenSlide integration."""

    @mock_patch("src.data.openslide_utils.OPENSLIDE_AVAILABLE", True)
    @mock_patch("src.data.openslide_utils.OpenSlide")
    @given(
        slide_width=st.integers(min_value=1000, max_value=get_test_config()["max_slide_dimension"]),
        slide_height=st.integers(
            min_value=1000, max_value=get_test_config()["max_slide_dimension"]
        ),
        patch_size=patch_size_strategy(),
        num_levels=st.integers(min_value=1, max_value=5),
    )
    @ci_aware_settings
    def test_patch_extraction_coordinate_consistency(
        self, mock_openslide, slide_width, slide_height, patch_size, num_levels
    ):
        """
        **Property 4: Coordinate and Alignment Preservation**

        For any valid coordinates and patch size, patch extraction SHALL preserve
        spatial relationships and return consistent coordinates.

        **Validates: Requirements 4.3, 4.4**
        """
        with tempfile.NamedTemporaryFile(suffix=".svs", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Set up mock slide with pyramid levels
            mock_slide = Mock()

            # Generate pyramid levels
            level_dimensions = []
            level_downsamples = []

            for level in range(num_levels):
                downsample = 2**level
                level_width = slide_width // downsample
                level_height = slide_height // downsample
                level_dimensions.append((level_width, level_height))
                level_downsamples.append(float(downsample))

            mock_slide.level_dimensions = level_dimensions
            mock_slide.level_downsamples = level_downsamples
            mock_slide.level_count = num_levels

            # Track all read_region calls
            read_region_calls = []

            def mock_read_region(location, level, size):
                read_region_calls.append((location, level, size))
                # Create a patch with coordinate-based pattern for verification
                x, y = location
                img_array = np.zeros((size[1], size[0], 4), dtype=np.uint8)

                # Encode coordinates in the image for verification
                img_array[:, :, 0] = min(255, (x // 100) % 256)  # Red encodes x
                img_array[:, :, 1] = min(255, (y // 100) % 256)  # Green encodes y
                img_array[:, :, 2] = 100  # Blue constant
                img_array[:, :, 3] = 255  # Alpha

                return Image.fromarray(img_array, "RGBA")

            mock_slide.read_region = mock_read_region
            mock_openslide.return_value = mock_slide

            reader = WSIReader(tmp_path)

            # Test patch extraction at level 0
            level = 0
            stride = patch_size  # Non-overlapping for simplicity

            # Skip if patch size is larger than slide dimensions
            assume(patch_size < slide_width and patch_size < slide_height)

            patches = reader.extract_patches(
                patch_size=patch_size,
                level=level,
                stride=stride,
                tissue_threshold=0.0,  # Accept all patches
            )

            # Property 1: All patches should have correct dimensions
            for patch, (x, y) in patches:
                assert patch.shape == (
                    patch_size,
                    patch_size,
                    3,
                ), f"Patch at ({x}, {y}) has incorrect shape: {patch.shape}"

            # Property 2: Coordinates should be within slide bounds
            for patch, (x, y) in patches:
                assert x >= 0 and y >= 0, f"Patch coordinates ({x}, {y}) are negative"
                assert (
                    x + patch_size <= slide_width
                ), f"Patch at ({x}, {y}) extends beyond slide width"
                assert (
                    y + patch_size <= slide_height
                ), f"Patch at ({y}, {y}) extends beyond slide height"

            # Property 3: Coordinates should match read_region calls
            assert len(patches) == len(
                read_region_calls
            ), "Number of patches should match number of read_region calls"

            for (patch, (patch_x, patch_y)), (call_location, call_level, call_size) in zip(
                patches, read_region_calls
            ):
                assert call_location == (
                    patch_x,
                    patch_y,
                ), f"Coordinate mismatch: patch ({patch_x}, {patch_y}) vs call {call_location}"
                assert call_level == level, f"Level mismatch: expected {level}, got {call_level}"
                assert call_size == (
                    patch_size,
                    patch_size,
                ), f"Size mismatch: expected ({patch_size}, {patch_size}), got {call_size}"

            # Property 4: Grid alignment should be consistent
            if len(patches) > 1:
                # Check that patches follow expected grid pattern
                patch_coords = [(x, y) for _, (x, y) in patches]
                patch_coords.sort()

                # Verify grid spacing
                for i in range(1, len(patch_coords)):
                    prev_x, prev_y = patch_coords[i - 1]
                    curr_x, curr_y = patch_coords[i]

                    # Either same row (x increases by stride) or next row
                    if curr_y == prev_y:
                        # Same row
                        assert (
                            curr_x == prev_x + stride
                        ), f"Horizontal spacing incorrect: {curr_x} vs {prev_x + stride}"
                    else:
                        # Different row - should be grid-aligned
                        assert curr_y >= prev_y, f"Row order incorrect: {curr_y} vs {prev_y}"

        finally:
            Path(tmp_path).unlink()

    @mock_patch("src.data.openslide_utils.OPENSLIDE_AVAILABLE", True)
    @mock_patch("src.data.openslide_utils.OpenSlide")
    @given(
        slide_width=st.integers(
            min_value=2000, max_value=min(20000, get_test_config()["max_slide_dimension"])
        ),
        slide_height=st.integers(
            min_value=2000, max_value=min(20000, get_test_config()["max_slide_dimension"])
        ),
        patch_size=patch_size_strategy(),
        level=st.integers(min_value=0, max_value=3),
    )
    @ci_aware_settings
    def test_pyramid_level_coordinate_transformation(
        self, mock_openslide, slide_width, slide_height, patch_size, level
    ):
        """
        **Property 4: Coordinate and Alignment Preservation**

        For any pyramid level, coordinate transformations SHALL preserve
        spatial relationships between levels.

        **Validates: Requirements 4.3, 4.4**
        """
        with tempfile.NamedTemporaryFile(suffix=".svs", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Set up mock slide with pyramid levels
            mock_slide = Mock()

            num_levels = level + 1
            level_dimensions = []
            level_downsamples = []

            for level in range(num_levels):
                downsample = 2**level
                level_width = slide_width // downsample
                level_height = slide_height // downsample
                level_dimensions.append((level_width, level_height))
                level_downsamples.append(float(downsample))

            mock_slide.level_dimensions = level_dimensions
            mock_slide.level_downsamples = level_downsamples
            mock_slide.level_count = num_levels

            # Skip if patch size is too large for the level
            level_width, level_height = level_dimensions[level]
            assume(patch_size < level_width and patch_size < level_height)

            read_region_calls = []

            def mock_read_region(location, call_level, size):
                read_region_calls.append((location, call_level, size))
                return Image.new("RGBA", size, (150, 100, 100, 255))

            mock_slide.read_region = mock_read_region
            mock_openslide.return_value = mock_slide

            reader = WSIReader(tmp_path)

            # Extract patches at the specified level
            patches = reader.extract_patches(
                patch_size=patch_size, level=level, stride=patch_size, tissue_threshold=0.0
            )

            downsample = level_downsamples[level]

            # Property 1: All coordinates should be level 0 coordinates
            for patch, (x, y) in patches:
                # Coordinates returned should be level 0 coordinates
                assert x >= 0 and y >= 0, f"Level 0 coordinates ({x}, {y}) should be non-negative"

                # Level 0 coordinates should be within slide bounds
                assert (
                    x + patch_size * downsample <= slide_width
                ), f"Level 0 coordinates ({x}, {y}) extend beyond slide width"
                assert (
                    y + patch_size * downsample <= slide_height
                ), f"Level 0 coordinates ({x}, {y}) extend beyond slide height"

            # Property 2: read_region should be called with correct level
            for call_location, call_level, call_size in read_region_calls:
                assert (
                    call_level == level
                ), f"read_region called with wrong level: {call_level} vs {level}"
                assert call_size == (
                    patch_size,
                    patch_size,
                ), f"read_region called with wrong size: {call_size}"

            # Property 3: Coordinate transformation should be consistent
            for (patch, (patch_x, patch_y)), (call_location, call_level, call_size) in zip(
                patches, read_region_calls
            ):
                assert call_location == (
                    patch_x,
                    patch_y,
                ), (
                    f"Coordinate transformation inconsistent: {call_location} "
                    f"vs ({patch_x}, {patch_y})"
                )

                # Verify that level coordinates are within level bounds
                level_x = patch_x / downsample
                level_y = patch_y / downsample

                assert (
                    level_x + patch_size <= level_width
                ), f"Level {level} coordinates ({level_x}, {level_y}) extend beyond level bounds"
                assert (
                    level_y + patch_size <= level_height
                ), f"Level {level} coordinates ({level_x}, {level_y}) extend beyond level bounds"

        finally:
            Path(tmp_path).unlink()

    @mock_patch("src.data.openslide_utils.OPENSLIDE_AVAILABLE", True)
    @mock_patch("src.data.openslide_utils.OpenSlide")
    @given(
        slide_width=st.integers(
            min_value=1000, max_value=min(10000, get_test_config()["max_slide_dimension"])
        ),
        slide_height=st.integers(
            min_value=1000, max_value=min(10000, get_test_config()["max_slide_dimension"])
        ),
        patch_size=patch_size_strategy(),
        stride=st.integers(min_value=64, max_value=512),
    )
    @ci_aware_settings
    def test_overlapping_patch_consistency(
        self, mock_openslide, slide_width, slide_height, patch_size, stride
    ):
        """
        **Property 4: Coordinate and Alignment Preservation**

        For overlapping patches, the overlapping regions SHALL contain
        consistent data when extracted from the same slide coordinates.

        **Validates: Requirements 4.4**
        """
        with tempfile.NamedTemporaryFile(suffix=".svs", delete=False) as tmp:
            tmp_path = tmp.name

        try:
            # Skip if stride is larger than patch size (no overlap)
            assume(stride < patch_size)
            assume(patch_size < slide_width and patch_size < slide_height)

            mock_slide = Mock()
            mock_slide.level_dimensions = [(slide_width, slide_height)]
            mock_slide.level_downsamples = [1.0]
            mock_slide.level_count = 1

            # Create a deterministic pattern based on coordinates
            def mock_read_region(location, level, size):
                x, y = location
                img_array = np.zeros((size[1], size[0], 4), dtype=np.uint8)

                # Create a pattern that varies with position
                for row in range(size[1]):
                    for col in range(size[0]):
                        global_x = x + col
                        global_y = y + row

                        # Create a repeating pattern
                        img_array[row, col, 0] = global_x % 256  # Red varies with x
                        img_array[row, col, 1] = global_y % 256  # Green varies with y
                        img_array[row, col, 2] = (global_x + global_y) % 256  # Blue varies with sum
                        img_array[row, col, 3] = 255  # Alpha

                return Image.fromarray(img_array, "RGBA")

            mock_slide.read_region = mock_read_region
            mock_openslide.return_value = mock_slide

            reader = WSIReader(tmp_path)

            # Extract overlapping patches
            patches = reader.extract_patches(
                patch_size=patch_size, level=0, stride=stride, tissue_threshold=0.0
            )

            # Property: Overlapping regions should have consistent pixel values
            patch_dict = {(x, y): patch for patch, (x, y) in patches}

            overlap_found = False
            for (x1, y1), patch1 in patch_dict.items():
                for (x2, y2), patch2 in patch_dict.items():
                    if x1 == x2 and y1 == y2:
                        continue  # Same patch

                    # Check for overlap
                    overlap_left = max(x1, x2)
                    overlap_top = max(y1, y2)
                    overlap_right = min(x1 + patch_size, x2 + patch_size)
                    overlap_bottom = min(y1 + patch_size, y2 + patch_size)

                    if overlap_right > overlap_left and overlap_bottom > overlap_top:
                        overlap_found = True

                        # Calculate overlap regions in each patch
                        patch1_left = overlap_left - x1
                        patch1_top = overlap_top - y1
                        patch1_right = overlap_right - x1
                        patch1_bottom = overlap_bottom - y1

                        patch2_left = overlap_left - x2
                        patch2_top = overlap_top - y2
                        patch2_right = overlap_right - x2
                        patch2_bottom = overlap_bottom - y2

                        # Extract overlapping regions
                        region1 = patch1[patch1_top:patch1_bottom, patch1_left:patch1_right]
                        region2 = patch2[patch2_top:patch2_bottom, patch2_left:patch2_right]

                        # Regions should be identical
                        assert region1.shape == region2.shape, (
                            f"Overlapping regions have different shapes: "
                            f"{region1.shape} vs {region2.shape}"
                        )

                        # Allow small differences due to potential rounding
                        diff = np.abs(region1.astype(np.int16) - region2.astype(np.int16))
                        max_diff = np.max(diff)

                        assert (
                            max_diff <= 1
                        ), f"Overlapping regions differ by more than 1: max_diff={max_diff}"

            # Ensure we actually tested overlapping patches
            if len(patches) > 1:
                assert overlap_found, "No overlapping patches found to test consistency"

        finally:
            Path(tmp_path).unlink()
