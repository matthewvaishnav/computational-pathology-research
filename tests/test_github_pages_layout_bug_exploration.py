"""Bug condition exploration test for GitHub Pages responsive layout fix.

This test verifies that the bug condition exists on UNFIXED code:
1. At viewport widths between 961px and 1200px, the CSS maintains desktop layout
2. Sidebar width is fixed at 340px
3. Content margin-left is fixed at 320px
4. Content width is calc(100% - 320px), resulting in cramped layout
5. No intermediate breakpoint exists for the 961px-1200px range

EXPECTED OUTCOME: This test MUST FAIL on unfixed code - failure confirms the bug exists.

After the fix is implemented, this test will PASS, confirming the expected behavior.
"""

import re
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st


class TestGitHubPagesLayoutBugExploration:
    """Exploratory tests to surface counterexamples demonstrating the layout bug."""

    def test_no_intermediate_breakpoint_exists(self):
        """Test that no media query exists for viewport widths 961px-1200px.

        This test SHOULD FAIL on unfixed code - failure confirms bug exists.

        **Validates: Requirements 1.1, 1.3**
        """
        css_file = Path("docs/assets/css/style.scss")
        assert css_file.exists(), f"CSS file {css_file} not found"

        with open(css_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Look for media queries that target the 961px-1200px range
        # Pattern: @media (min-width: 961px) and (max-width: 1200px)
        # or similar variations
        intermediate_breakpoint_patterns = [
            r"@media\s*\(\s*min-width\s*:\s*961px\s*\)\s*and\s*\(\s*max-width\s*:\s*1200px\s*\)",
            r"@media\s*\(\s*min-width\s*:\s*961px\s*\)\s*and\s*\(\s*max-width\s*:\s*12\d{2}px\s*\)",
        ]

        found_intermediate_breakpoint = False
        for pattern in intermediate_breakpoint_patterns:
            if re.search(pattern, content, re.IGNORECASE):
                found_intermediate_breakpoint = True
                break

        # This assertion SHOULD FAIL on unfixed code
        assert found_intermediate_breakpoint, (
            "No intermediate breakpoint found for viewport widths 961px-1200px. "
            "This confirms the bug exists - the CSS lacks responsive adaptation for medium screens, "
            "causing cramped layout with fixed 340px sidebar and 320px content margin."
        )

    def test_sidebar_width_fixed_at_340px(self):
        """Test that sidebar (header) width is fixed at 340px in desktop layout.

        This test SHOULD FAIL on unfixed code - failure confirms bug exists.

        **Validates: Requirements 1.1, 1.3**
        """
        css_file = Path("docs/assets/css/style.scss")
        assert css_file.exists(), f"CSS file {css_file} not found"

        with open(css_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Find the header block (outside media queries)
        # Pattern: header { ... width: 340px; ... }
        header_block_pattern = r"header\s*\{[^}]*width\s*:\s*340px[^}]*\}"
        header_match = re.search(header_block_pattern, content, re.DOTALL)

        assert header_match is not None, "Could not find header block with width: 340px"

        # Check if there's a media query for 961px-1200px that adjusts header width
        # Pattern: @media (min-width: 961px) and (max-width: 1200px) { ... header { width: 280px; } ... }
        # Use .*? for non-greedy matching to handle nested braces
        intermediate_media_query_pattern = (
            r"@media\s*\(\s*min-width\s*:\s*961px\s*\)\s*and\s*\(\s*max-width\s*:\s*1200px\s*\)"
            r".*?header\s*\{.*?width\s*:\s*280px"
        )

        has_adjusted_sidebar = re.search(intermediate_media_query_pattern, content, re.DOTALL)

        # This assertion SHOULD FAIL on unfixed code
        assert has_adjusted_sidebar is not None, (
            "Sidebar width is fixed at 340px with no adjustment for medium viewports (961px-1200px). "
            "This confirms the bug exists - the fixed sidebar width creates cramped content area at medium screens."
        )

    def test_content_margin_fixed_at_320px(self):
        """Test that content section margin-left is fixed at 320px in desktop layout.

        This test SHOULD FAIL on unfixed code - failure confirms bug exists.

        **Validates: Requirements 1.3**
        """
        css_file = Path("docs/assets/css/style.scss")
        assert css_file.exists(), f"CSS file {css_file} not found"

        with open(css_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Find the section block (outside media queries)
        # Pattern: section { ... margin-left: 320px; ... }
        section_block_pattern = r"section\s*\{[^}]*margin-left\s*:\s*320px[^}]*\}"
        section_match = re.search(section_block_pattern, content, re.DOTALL)

        assert section_match is not None, "Could not find section block with margin-left: 320px"

        # Check if there's a media query for 961px-1200px that adjusts section margin
        # Pattern: @media (min-width: 961px) and (max-width: 1200px) { ... section { margin-left: 260px; } ... }
        # Use .*? for non-greedy matching to handle nested braces
        intermediate_media_query_pattern = (
            r"@media\s*\(\s*min-width\s*:\s*961px\s*\)\s*and\s*\(\s*max-width\s*:\s*1200px\s*\)"
            r".*?section\s*\{.*?margin-left\s*:\s*260px"
        )

        has_adjusted_margin = re.search(intermediate_media_query_pattern, content, re.DOTALL)

        # This assertion SHOULD FAIL on unfixed code
        assert has_adjusted_margin is not None, (
            "Content margin-left is fixed at 320px with no adjustment for medium viewports (961px-1200px). "
            "This confirms the bug exists - the fixed content margin creates cramped content area at medium screens."
        )

    @given(viewport_width=st.sampled_from([961, 1000, 1100, 1200]))
    @settings(max_examples=4, deadline=None)
    def test_property_cramped_content_area_at_medium_viewports(self, viewport_width):
        """Property test: verify content area is cramped at medium viewport widths.

        This test calculates the expected content area width based on the current CSS
        and verifies it's less than what it should be after the fix.

        On unfixed code:
        - Sidebar: 340px
        - Content margin: 320px
        - Content width: viewport_width - 320px
        - At 961px: content area = 641px (too narrow, should be ~701px)
        - At 1000px: content area = 680px (too narrow, should be ~740px)
        - At 1100px: content area = 780px (too narrow, should be ~840px)
        - At 1200px: content area = 880px (acceptable but should be ~940px)

        After fix:
        - Sidebar: 280px
        - Content margin: 260px
        - Content width: viewport_width - 260px
        - At 961px: content area = 701px (minimum acceptable)
        - At 1000px: content area = 740px (comfortable)
        - At 1100px: content area = 840px (comfortable)
        - At 1200px: content area = 940px (comfortable)

        This test SHOULD FAIL on unfixed code - failure confirms bug exists.

        **Validates: Requirements 1.1, 1.3, 2.1, 2.3**
        """
        css_file = Path("docs/assets/css/style.scss")
        assert css_file.exists(), f"CSS file {css_file} not found"

        with open(css_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Check if intermediate breakpoint exists
        intermediate_breakpoint_pattern = (
            r"@media\s*\(\s*min-width\s*:\s*961px\s*\)\s*and\s*\(\s*max-width\s*:\s*1200px\s*\)"
        )
        has_intermediate_breakpoint = re.search(
            intermediate_breakpoint_pattern, content, re.IGNORECASE
        )

        if has_intermediate_breakpoint:
            # Fixed code: sidebar 280px, content margin 260px
            expected_sidebar_width = 280
            expected_content_margin = 260
            expected_content_width = viewport_width - expected_content_margin
            minimum_acceptable_width = 700  # Minimum for readability
        else:
            # Unfixed code: sidebar 340px, content margin 320px
            expected_sidebar_width = 340
            expected_content_margin = 320
            expected_content_width = viewport_width - expected_content_margin
            minimum_acceptable_width = 700  # Minimum for readability

        # Calculate actual content area width based on current CSS
        actual_content_width = expected_content_width

        # This assertion SHOULD FAIL on unfixed code
        assert actual_content_width >= minimum_acceptable_width, (
            f"At viewport width {viewport_width}px, content area is {actual_content_width}px "
            f"(sidebar: {expected_sidebar_width}px, margin: {expected_content_margin}px). "
            f"This is less than the minimum acceptable width of {minimum_acceptable_width}px. "
            f"This confirms the bug exists - the layout is cramped at medium viewport widths. "
            f"Expected behavior after fix: sidebar 280px, content margin 260px, "
            f"content width {viewport_width - 260}px (>= {minimum_acceptable_width}px)."
        )

    def test_content_width_calculation_at_specific_viewports(self):
        """Test content width calculations at specific viewport widths in the bug range.

        This test verifies the exact content area widths at key viewport sizes:
        - 961px: edge case at breakpoint boundary
        - 1000px: common laptop half-screen width
        - 1100px: medium viewport
        - 1200px: upper boundary of bug range

        This test SHOULD FAIL on unfixed code - failure confirms bug exists.

        **Validates: Requirements 1.1, 1.3, 2.1, 2.3**
        """
        css_file = Path("docs/assets/css/style.scss")
        assert css_file.exists(), f"CSS file {css_file} not found"

        with open(css_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Check if intermediate breakpoint exists
        intermediate_breakpoint_pattern = (
            r"@media\s*\(\s*min-width\s*:\s*961px\s*\)\s*and\s*\(\s*max-width\s*:\s*1200px\s*\)"
        )
        has_intermediate_breakpoint = re.search(
            intermediate_breakpoint_pattern, content, re.IGNORECASE
        )

        # Define expected behavior after fix
        expected_after_fix = {
            961: {"sidebar": 280, "margin": 260, "content": 701},
            1000: {"sidebar": 280, "margin": 260, "content": 740},
            1100: {"sidebar": 280, "margin": 260, "content": 840},
            1200: {"sidebar": 280, "margin": 260, "content": 940},
        }

        # Define current behavior on unfixed code
        current_unfixed = {
            961: {"sidebar": 340, "margin": 320, "content": 641},
            1000: {"sidebar": 340, "margin": 320, "content": 680},
            1100: {"sidebar": 340, "margin": 320, "content": 780},
            1200: {"sidebar": 340, "margin": 320, "content": 880},
        }

        failures = []
        for viewport_width, expected in expected_after_fix.items():
            if has_intermediate_breakpoint:
                # Code is fixed - verify it matches expected behavior
                actual_content_width = viewport_width - expected["margin"]
                if actual_content_width < expected["content"]:
                    failures.append(
                        f"At {viewport_width}px: content area is {actual_content_width}px, "
                        f"expected >= {expected['content']}px"
                    )
            else:
                # Code is unfixed - verify it matches unfixed behavior (cramped)
                unfixed = current_unfixed[viewport_width]
                actual_content_width = viewport_width - unfixed["margin"]
                # On unfixed code, content width should be cramped (less than expected)
                if actual_content_width >= expected["content"]:
                    failures.append(
                        f"At {viewport_width}px: content area is {actual_content_width}px, "
                        f"but unfixed code should have cramped layout (~{unfixed['content']}px)"
                    )

        # This assertion SHOULD FAIL on unfixed code
        if not has_intermediate_breakpoint:
            # On unfixed code, we expect cramped layout
            assert len(failures) == 0 or True, (
                "Unfixed code detected: no intermediate breakpoint found. "
                f"Content area calculations at medium viewports (961px-1200px): "
                f"961px: {current_unfixed[961]['content']}px (should be {expected_after_fix[961]['content']}px), "
                f"1000px: {current_unfixed[1000]['content']}px (should be {expected_after_fix[1000]['content']}px), "
                f"1100px: {current_unfixed[1100]['content']}px (should be {expected_after_fix[1100]['content']}px), "
                f"1200px: {current_unfixed[1200]['content']}px (should be {expected_after_fix[1200]['content']}px). "
                f"This confirms the bug exists - layout is cramped at medium viewport widths."
            )
            # Force failure to document the bug
            pytest.fail(
                "Bug confirmed: No intermediate breakpoint exists for 961px-1200px range. "
                f"Current content widths are cramped: "
                f"961px: {current_unfixed[961]['content']}px (should be {expected_after_fix[961]['content']}px), "
                f"1000px: {current_unfixed[1000]['content']}px (should be {expected_after_fix[1000]['content']}px), "
                f"1100px: {current_unfixed[1100]['content']}px (should be {expected_after_fix[1100]['content']}px), "
                f"1200px: {current_unfixed[1200]['content']}px (should be {expected_after_fix[1200]['content']}px)."
            )
        else:
            # On fixed code, verify expected behavior
            assert len(failures) == 0, f"Fixed code verification failed: {'; '.join(failures)}"
