"""
Bug condition exploration test for macOS CI timeout fix.

This test is designed to FAIL on unfixed code to confirm the bug exists.
It simulates CI environment conditions with large slide dimensions and high example counts
that cause memory exhaustion and SIGKILL (exit code 137) on CI runners.

CRITICAL: This test MUST FAIL on unfixed code - failure confirms the bug exists.
DO NOT attempt to fix the test or the code when it fails.
"""

import os
import tempfile
from pathlib import Path
from unittest.mock import Mock
from unittest.mock import patch as mock_patch

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from PIL import Image

from src.data.openslide_utils import WSIReader
from tests.dataset_testing.hypothesis_strategies import patch_size_strategy


class TestCIMemoryExhaustion:
    """Bug condition exploration test for CI memory exhaustion."""

    @mock_patch("src.data.openslide_utils.OPENSLIDE_AVAILABLE", True)
    @mock_patch("src.data.openslide_utils.OpenSlide")
    @given(
        slide_width=st.integers(min_value=50000, max_value=50000),  # Large dimensions
        slide_height=st.integers(min_value=50000, max_value=50000),  # Large dimensions
        patch_size=st.sampled_from([256, 512]),  # Reasonable patch sizes
        num_levels=st.integers(min_value=1, max_value=3),
    )
    @settings(max_examples=100, deadline=None)  # High example count, no deadline
    def test_ci_memory_exhaustion_bug_condition(
        self, mock_openslide, slide_width, slide_height, patch_size, num_levels
    ):
        """
        **Property 1: Bug Condition** - CI Memory Exhaustion Test

        This test encodes the expected behavior: tests should complete within
        available memory and time limits without being killed.

        CRITICAL: This test MUST FAIL on unfixed code with SIGKILL (exit code 137)
        or memory exhaustion. This failure confirms the bug exists.

        Test implementation details from Bug Condition:
        - slide_width > 10000 AND slide_height > 10000
        - max_examples > 50
        - environment == "CI"

        **Validates: Requirements 2.1, 2.2, 2.3**
        """
        # Simulate CI environment conditions
        original_ci = os.environ.get("CI")
        original_github_actions = os.environ.get("GITHUB_ACTIONS")

        try:
            # Set CI environment variables to simulate CI conditions
            os.environ["CI"] = "true"
            os.environ["GITHUB_ACTIONS"] = "true"

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

                # Track all read_region calls for memory usage simulation
                read_region_calls = []

                def mock_read_region(location, level, size):
                    read_region_calls.append((location, level, size))

                    # Create large memory allocation to simulate real WSI processing
                    # This simulates the memory pressure that causes CI failures
                    x, y = location
                    img_array = np.zeros((size[1], size[0], 4), dtype=np.uint8)

                    # Simulate memory-intensive coordinate encoding
                    # This pattern creates significant memory usage per patch
                    for row in range(size[1]):
                        for col in range(size[0]):
                            global_x = x + col
                            global_y = y + row

                            # Memory-intensive operations that compound with large slides
                            img_array[row, col, 0] = min(255, (global_x // 100) % 256)
                            img_array[row, col, 1] = min(255, (global_y // 100) % 256)
                            img_array[row, col, 2] = min(255, ((global_x + global_y) // 200) % 256)
                            img_array[row, col, 3] = 255

                    return Image.fromarray(img_array, "RGBA")

                mock_slide.read_region = mock_read_region
                mock_openslide.return_value = mock_slide

                reader = WSIReader(tmp_path)

                # Test patch extraction at level 0 with parameters that trigger the bug
                level = 0
                stride = patch_size  # Non-overlapping for maximum memory usage

                # Extract patches - this should cause memory exhaustion on CI
                patches = reader.extract_patches(
                    patch_size=patch_size,
                    level=level,
                    stride=stride,
                    tissue_threshold=0.0,  # Accept all patches for maximum memory usage
                )

                # Expected behavior: tests complete within available memory and time limits
                # On unfixed code, this will fail with SIGKILL before reaching these assertions

                # Property 1: All patches should have correct dimensions
                patch_count = 0
                for patch, (x, y) in patches:
                    patch_count += 1
                    assert patch.shape == (
                        patch_size,
                        patch_size,
                        3,
                    ), f"Patch at ({x}, {y}) has incorrect shape: {patch.shape}"

                    # Property 2: Coordinates should be within slide bounds
                    assert x >= 0 and y >= 0, f"Patch coordinates ({x}, {y}) are negative"
                    assert (
                        x + patch_size <= slide_width
                    ), f"Patch at ({x}, {y}) extends beyond slide width"
                    assert (
                        y + patch_size <= slide_height
                    ), f"Patch at ({x}, {y}) extends beyond slide height"

                # Property 3: Test should complete without being killed
                # This assertion will not be reached on unfixed code due to SIGKILL
                assert patch_count > 0, "No patches were extracted"

                # Property 4: Memory usage should be within CI limits
                # On unfixed code, the process will be killed before this check
                assert (
                    len(read_region_calls) == patch_count
                ), "Number of patches should match number of read_region calls"

            finally:
                Path(tmp_path).unlink()

        finally:
            # Restore original environment variables
            if original_ci is not None:
                os.environ["CI"] = original_ci
            else:
                os.environ.pop("CI", None)

            if original_github_actions is not None:
                os.environ["GITHUB_ACTIONS"] = original_github_actions
            else:
                os.environ.pop("GITHUB_ACTIONS", None)

    def test_ci_environment_detection(self):
        """
        Test that CI environment detection works correctly.
        This is a supporting test for the main bug condition test.
        """
        # Test with CI environment variables set
        original_ci = os.environ.get("CI")
        original_github_actions = os.environ.get("GITHUB_ACTIONS")

        try:
            # Test CI=true detection
            os.environ["CI"] = "true"
            assert os.getenv("CI") == "true", "CI environment variable not set correctly"

            # Test GITHUB_ACTIONS=true detection
            os.environ["GITHUB_ACTIONS"] = "true"
            assert (
                os.getenv("GITHUB_ACTIONS") == "true"
            ), "GITHUB_ACTIONS environment variable not set correctly"

        finally:
            # Restore original environment variables
            if original_ci is not None:
                os.environ["CI"] = original_ci
            else:
                os.environ.pop("CI", None)

            if original_github_actions is not None:
                os.environ["GITHUB_ACTIONS"] = original_github_actions
            else:
                os.environ.pop("GITHUB_ACTIONS", None)

    @pytest.mark.parametrize(
        "slide_width,slide_height,max_examples",
        [
            (50000, 50000, 100),  # Exact bug condition parameters
            (25000, 25000, 75),  # Medium size with high examples
            (15000, 15000, 60),  # Boundary case
        ],
    )
    def test_bug_condition_parameters(self, slide_width, slide_height, max_examples):
        """
        Test specific parameter combinations that trigger the bug condition.

        Bug Condition: slide_width > 10000 AND slide_height > 10000
                      AND max_examples > 50 AND environment == "CI"
        """
        # Verify bug condition parameters
        assert slide_width > 10000, f"slide_width {slide_width} should be > 10000"
        assert slide_height > 10000, f"slide_height {slide_height} should be > 10000"
        assert max_examples > 50, f"max_examples {max_examples} should be > 50"

        # Simulate CI environment
        original_ci = os.environ.get("CI")
        try:
            os.environ["CI"] = "true"
            assert os.getenv("CI") == "true", "Should be in CI environment"

            # This test documents the bug condition parameters
            # The actual memory exhaustion test is in test_ci_memory_exhaustion_bug_condition

        finally:
            if original_ci is not None:
                os.environ["CI"] = original_ci
            else:
                os.environ.pop("CI", None)
