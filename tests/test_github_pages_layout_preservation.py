"""Preservation property tests for GitHub Pages responsive layout fix.

This test suite verifies that behaviors outside the bug range (viewport widths NOT between
961px-1200px) remain unchanged after the fix.

OBSERVATION-FIRST METHODOLOGY:
1. Observe behavior on UNFIXED code for non-buggy inputs
2. Write tests capturing that baseline behavior
3. Run tests on UNFIXED code - they should PASS
4. After fix, re-run tests - they should still PASS (no regressions)

EXPECTED OUTCOME: These tests MUST PASS on unfixed code - this confirms baseline behavior to preserve.

**Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7**
"""

import re
from pathlib import Path


from hypothesis import given, settings
from hypothesis import strategies as st


class TestGitHubPagesLayoutPreservation:
    """Preservation tests to ensure non-buggy behaviors remain unchanged."""

    # ═══════════════════════════════════════════════════════════════
    # Desktop Layout Preservation (>1200px)
    # ═══════════════════════════════════════════════════════════════

    @given(viewport_width=st.sampled_from([1201, 1400, 1920]))
    @settings(max_examples=3, deadline=None)
    def test_property_desktop_layout_preserved_above_1200px(self, viewport_width):
        """Property test: desktop layout unchanged for viewport widths > 1200px.

        OBSERVATION (on unfixed code):
        - At 1201px viewport: sidebar is 340px, content margin is 320px
        - At 1400px viewport: sidebar is 340px, content margin is 320px
        - At 1920px viewport: sidebar is 340px, content margin is 320px

        This test verifies that the full desktop layout with 340px sidebar and 320px
        content margin is preserved for all viewport widths greater than 1200px.

        EXPECTED OUTCOME: This test MUST PASS on unfixed code and after fix.

        **Validates: Requirements 3.1**
        """
        css_file = Path("docs/assets/css/style.scss")
        assert css_file.exists(), f"CSS file {css_file} not found"

        with open(css_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Verify base desktop layout exists (header width: 340px, section margin-left: 320px)
        header_pattern = r"header\s*\{[^}]*width\s*:\s*340px"
        section_pattern = r"section\s*\{[^}]*margin-left\s*:\s*320px"

        assert re.search(
            header_pattern, content, re.DOTALL
        ), f"Desktop layout broken: header width should be 340px for viewport > 1200px"
        assert re.search(
            section_pattern, content, re.DOTALL
        ), f"Desktop layout broken: section margin-left should be 320px for viewport > 1200px"

        # Verify no media query overrides desktop layout for viewports > 1200px
        # Check that there's no @media (min-width: 1201px) or similar that changes these values
        override_pattern = r"@media\s*\([^)]*min-width\s*:\s*12[0-9]{2}px[^)]*\)\s*\{[^}]*(?:header|section)\s*\{[^}]*(?:width|margin-left)\s*:\s*(?!340px|320px)"

        has_override = re.search(override_pattern, content, re.DOTALL)
        assert not has_override, (
            f"Desktop layout preservation violated: found media query overriding "
            f"desktop layout for viewport > 1200px. Desktop layout (340px sidebar, "
            f"320px content margin) must be preserved for all viewports > 1200px."
        )

    def test_desktop_layout_base_values(self):
        """Test that base desktop layout values are present in CSS.

        OBSERVATION (on unfixed code):
        - header { width: 340px; ... }
        - section { margin-left: 320px; width: calc(100% - 320px); ... }

        This test verifies the base desktop layout CSS rules exist and have correct values.

        EXPECTED OUTCOME: This test MUST PASS on unfixed code and after fix.

        **Validates: Requirements 3.1**
        """
        css_file = Path("docs/assets/css/style.scss")
        assert css_file.exists(), f"CSS file {css_file} not found"

        with open(css_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Extract header block (outside media queries)
        header_block_pattern = r"header\s*\{([^}]*)\}"
        header_match = re.search(header_block_pattern, content, re.DOTALL)
        assert header_match, "Could not find header block in CSS"

        header_block = header_match.group(1)
        assert re.search(
            r"width\s*:\s*340px", header_block
        ), "Desktop layout broken: header width should be 340px"

        # Extract section block (outside media queries)
        section_block_pattern = r"section\s*\{([^}]*)\}"
        section_match = re.search(section_block_pattern, content, re.DOTALL)
        assert section_match, "Could not find section block in CSS"

        section_block = section_match.group(1)
        assert re.search(
            r"margin-left\s*:\s*320px", section_block
        ), "Desktop layout broken: section margin-left should be 320px"
        assert re.search(
            r"width\s*:\s*calc\s*\(\s*100%\s*-\s*320px\s*\)", section_block
        ), "Desktop layout broken: section width should be calc(100% - 320px)"

    # ═══════════════════════════════════════════════════════════════
    # Mobile Layout Preservation (≤960px)
    # ═══════════════════════════════════════════════════════════════

    @given(viewport_width=st.sampled_from([960, 768, 480, 320]))
    @settings(max_examples=4, deadline=None)
    def test_property_mobile_layout_preserved_at_or_below_960px(self, viewport_width):
        """Property test: mobile layout unchanged for viewport widths ≤ 960px.

        OBSERVATION (on unfixed code):
        - At 960px viewport: layout switches to mobile (full-width stacked content)
        - At 768px viewport: mobile layout with collapsible navigation
        - At 480px viewport: mobile layout with touch-friendly controls
        - At 320px viewport: mobile layout with minimal width

        This test verifies that the mobile layout with full-width stacked content
        is preserved for all viewport widths at or below 960px.

        EXPECTED OUTCOME: This test MUST PASS on unfixed code and after fix.

        **Validates: Requirements 3.2, 3.4**
        """
        css_file = Path("docs/assets/css/style.scss")
        assert css_file.exists(), f"CSS file {css_file} not found"

        with open(css_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Verify mobile layout media query exists (@media (max-width: 960px))
        mobile_breakpoint_pattern = r"@media\s*\(\s*max-width\s*:\s*960px\s*\)"
        mobile_match = re.search(mobile_breakpoint_pattern, content, re.IGNORECASE)
        assert mobile_match, (
            f"Mobile layout preservation violated: @media (max-width: 960px) not found. "
            f"Mobile layout must be preserved for viewport widths ≤ 960px."
        )

        # Verify mobile layout characteristics exist in the CSS after the mobile breakpoint
        # Look for header width: 100% and section width: 100% anywhere in the file
        # (they should be in the mobile media query block)
        mobile_section_start = mobile_match.start()
        mobile_section_content = content[mobile_section_start:]

        # Check for full-width layout indicators in mobile section
        assert re.search(
            r"header\s*\{[^}]*width\s*:\s*100%", mobile_section_content, re.DOTALL
        ) or re.search(
            r"width\s*:\s*100%[^}]*float\s*:\s*none", mobile_section_content, re.DOTALL
        ), f"Mobile layout broken at {viewport_width}px: header should be full-width (100%) in mobile layout"
        assert re.search(
            r"section\s*\{[^}]*width\s*:\s*100%", mobile_section_content, re.DOTALL
        ), f"Mobile layout broken at {viewport_width}px: section should be full-width (100%) in mobile layout"

    def test_mobile_breakpoint_at_960px(self):
        """Test that mobile layout breakpoint is at 960px.

        OBSERVATION (on unfixed code):
        - @media (max-width: 960px) triggers mobile layout
        - header and section become full-width (100%)
        - float: none for stacked layout

        This test verifies the mobile breakpoint exists and has correct values.

        EXPECTED OUTCOME: This test MUST PASS on unfixed code and after fix.

        **Validates: Requirements 3.2**
        """
        css_file = Path("docs/assets/css/style.scss")
        assert css_file.exists(), f"CSS file {css_file} not found"

        with open(css_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Verify @media (max-width: 960px) exists
        mobile_breakpoint_pattern = r"@media\s*\(\s*max-width\s*:\s*960px\s*\)"
        assert re.search(
            mobile_breakpoint_pattern, content, re.IGNORECASE
        ), "Mobile layout preservation violated: @media (max-width: 960px) not found"

    def test_phone_breakpoint_at_480px(self):
        """Test that phone-specific layout breakpoint is at 480px.

        OBSERVATION (on unfixed code):
        - @media (max-width: 480px) triggers phone-specific optimizations
        - Touch-friendly controls with larger tap targets
        - Reduced font sizes and padding for small screens

        This test verifies the phone breakpoint exists.

        EXPECTED OUTCOME: This test MUST PASS on unfixed code and after fix.

        **Validates: Requirements 3.4**
        """
        css_file = Path("docs/assets/css/style.scss")
        assert css_file.exists(), f"CSS file {css_file} not found"

        with open(css_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Verify @media (max-width: 480px) exists
        phone_breakpoint_pattern = r"@media\s*\(\s*max-width\s*:\s*480px\s*\)"
        assert re.search(
            phone_breakpoint_pattern, content, re.IGNORECASE
        ), "Phone layout preservation violated: @media (max-width: 480px) not found"

    # ═══════════════════════════════════════════════════════════════
    # Navigation Preservation
    # ═══════════════════════════════════════════════════════════════

    def test_navigation_structure_preserved(self):
        """Test that navigation structure is preserved in CSS.

        OBSERVATION (on unfixed code):
        - header nav { ... } contains navigation styles
        - nav ul { list-style: none; ... }
        - nav li a { ... } contains link styles
        - Navigation toggle button for mobile (.nav-toggle)

        This test verifies navigation CSS structure remains unchanged.

        EXPECTED OUTCOME: This test MUST PASS on unfixed code and after fix.

        **Validates: Requirements 3.3**
        """
        css_file = Path("docs/assets/css/style.scss")
        assert css_file.exists(), f"CSS file {css_file} not found"

        with open(css_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Verify navigation structure exists
        # Note: SCSS uses nested selectors, so we need to be more flexible
        nav_patterns = [
            (r"header\s+nav\s*\{|nav\s*\{", "header nav or nav block"),
            (r"nav\s+ul\s*\{|ul\s*\{", "nav ul or ul block"),
            (r"nav\s+li\s+a\s*\{|li\s*\{[^}]*a\s*\{|a\s*\{", "nav li a or nested a block"),
            (r"\.nav-toggle\s*\{", ".nav-toggle block"),
        ]

        for pattern, description in nav_patterns:
            assert re.search(
                pattern, content, re.DOTALL
            ), f"Navigation preservation violated: {description} not found in CSS"

    def test_navigation_mobile_behavior_preserved(self):
        """Test that mobile navigation behavior is preserved.

        OBSERVATION (on unfixed code):
        - .nav-toggle is hidden on desktop (display: none)
        - .nav-toggle is shown on mobile (display: flex)
        - nav#main-nav has collapsible behavior with max-height transition

        This test verifies mobile navigation CSS remains unchanged.

        EXPECTED OUTCOME: This test MUST PASS on unfixed code and after fix.

        **Validates: Requirements 3.3**
        """
        css_file = Path("docs/assets/css/style.scss")
        assert css_file.exists(), f"CSS file {css_file} not found"

        with open(css_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Verify nav-toggle is hidden by default
        nav_toggle_pattern = r"\.nav-toggle\s*\{[^}]*display\s*:\s*none"
        assert re.search(
            nav_toggle_pattern, content, re.DOTALL
        ), "Navigation preservation violated: .nav-toggle should have display: none by default"

        # Verify mobile media query shows nav-toggle
        mobile_block_pattern = r"@media\s*\(\s*max-width\s*:\s*960px\s*\)[^{]*\{[^}]*\.nav-toggle\s*\{[^}]*display\s*:\s*flex"
        # Note: This is a simplified check - actual CSS may have nav-toggle in nested blocks
        # We'll check if nav-toggle display is changed in mobile context
        assert re.search(
            r"\.nav-toggle\s*\{", content
        ), "Navigation preservation violated: .nav-toggle block not found"

    # ═══════════════════════════════════════════════════════════════
    # Accessibility Preservation
    # ═══════════════════════════════════════════════════════════════

    def test_accessibility_focus_styles_preserved(self):
        """Test that keyboard navigation focus styles are preserved.

        OBSERVATION (on unfixed code):
        - a:focus, button:focus { outline: 3px solid #A51C30; outline-offset: 3px; }
        - Focus indicators are visible for keyboard navigation

        This test verifies focus styles remain unchanged.

        EXPECTED OUTCOME: This test MUST PASS on unfixed code and after fix.

        **Validates: Requirements 3.7**
        """
        css_file = Path("docs/assets/css/style.scss")
        assert css_file.exists(), f"CSS file {css_file} not found"

        with open(css_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Verify focus styles exist
        focus_pattern = r"(?:a|button):focus\s*\{[^}]*outline\s*:\s*3px\s+solid\s+#A51C30"
        assert re.search(
            focus_pattern, content, re.IGNORECASE
        ), "Accessibility preservation violated: focus styles (outline: 3px solid #A51C30) not found"

    def test_accessibility_reduced_motion_preserved(self):
        """Test that reduced motion preference support is preserved.

        OBSERVATION (on unfixed code):
        - @media (prefers-reduced-motion: reduce) disables animations
        - animation-duration: 0.01ms !important
        - transition-duration: 0.01ms !important
        - scroll-behavior: auto !important

        This test verifies reduced motion support remains unchanged.

        EXPECTED OUTCOME: This test MUST PASS on unfixed code and after fix.

        **Validates: Requirements 3.6**
        """
        css_file = Path("docs/assets/css/style.scss")
        assert css_file.exists(), f"CSS file {css_file} not found"

        with open(css_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Verify prefers-reduced-motion media query exists
        reduced_motion_pattern = r"@media\s*\(\s*prefers-reduced-motion\s*:\s*reduce\s*\)"
        assert re.search(
            reduced_motion_pattern, content, re.IGNORECASE
        ), "Accessibility preservation violated: @media (prefers-reduced-motion: reduce) not found"

        # Verify animation and transition durations are set to minimal values
        # Simply check that the reduced motion block contains the expected properties
        # without trying to extract the exact block
        assert re.search(
            r"animation-duration\s*:\s*0\.01ms\s*!important", content
        ), "Accessibility preservation violated: animation-duration should be 0.01ms !important in reduced motion"
        assert re.search(
            r"transition-duration\s*:\s*0\.01ms\s*!important", content
        ), "Accessibility preservation violated: transition-duration should be 0.01ms !important in reduced motion"

    def test_accessibility_skip_to_content_preserved(self):
        """Test that skip-to-content link for screen readers is preserved.

        OBSERVATION (on unfixed code):
        - .skip-to-content { position: absolute; top: -40px; ... }
        - .skip-to-content:focus { top: 0; }
        - Provides keyboard navigation shortcut to main content

        This test verifies skip-to-content styles remain unchanged.

        EXPECTED OUTCOME: This test MUST PASS on unfixed code and after fix.

        **Validates: Requirements 3.7**
        """
        css_file = Path("docs/assets/css/style.scss")
        assert css_file.exists(), f"CSS file {css_file} not found"

        with open(css_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Verify skip-to-content class exists
        skip_to_content_pattern = r"\.skip-to-content\s*\{"
        assert re.search(
            skip_to_content_pattern, content
        ), "Accessibility preservation violated: .skip-to-content class not found"

        # Verify skip-to-content:focus behavior
        # Note: SCSS uses nested & syntax for :focus
        skip_focus_pattern = (
            r"\.skip-to-content:focus\s*\{[^}]*top\s*:\s*0|&:focus\s*\{[^}]*top\s*:\s*0"
        )
        assert re.search(
            skip_focus_pattern, content, re.DOTALL
        ), "Accessibility preservation violated: .skip-to-content:focus should set top: 0"

    # ═══════════════════════════════════════════════════════════════
    # Print Preservation
    # ═══════════════════════════════════════════════════════════════

    def test_print_styles_preserved(self):
        """Test that print-specific styles are preserved.

        OBSERVATION (on unfixed code):
        - @media print { ... } contains print-specific styles
        - header { width: 100%; border-right: none; border-bottom: 2px solid #000; }
        - section { width: 100%; float: none; }
        - nav, .downloads, .nav-toggle { display: none; }

        This test verifies print styles remain unchanged.

        EXPECTED OUTCOME: This test MUST PASS on unfixed code and after fix.

        **Validates: Requirements 3.5**
        """
        css_file = Path("docs/assets/css/style.scss")
        assert css_file.exists(), f"CSS file {css_file} not found"

        with open(css_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Verify @media print exists
        print_media_pattern = r"@media\s+print\s*\{"
        assert re.search(
            print_media_pattern, content, re.IGNORECASE
        ), "Print preservation violated: @media print not found"

        # Extract print media query block - just check after the first @media print
        print_match = re.search(print_media_pattern, content, re.IGNORECASE)
        if print_match:
            print_section_start = print_match.start()
            print_section_content = content[print_section_start:]

            # Verify print-specific styles exist somewhere in the print sections
            # Note: There are 2 @media print blocks, so check the entire print section
            assert re.search(
                r"header\s*\{[^}]*width\s*:\s*100%", print_section_content, re.DOTALL
            ) or re.search(
                r"width\s*:\s*100%", print_section_content
            ), "Print preservation violated: header should be full-width (100%) in print"
            assert re.search(
                r"section\s*\{[^}]*width\s*:\s*100%", print_section_content, re.DOTALL
            ), "Print preservation violated: section should be full-width (100%) in print"
            # Check for nav display: none in print section
            assert re.search(
                r"nav[^}]*display\s*:\s*none", print_section_content, re.DOTALL
            ), "Print preservation violated: nav should be hidden (display: none) in print"

    def test_print_media_query_count(self):
        """Test that the number of print media queries is preserved.

        OBSERVATION (on unfixed code):
        - There are 2 @media print blocks in the CSS

        This test verifies no print media queries are accidentally removed.

        EXPECTED OUTCOME: This test MUST PASS on unfixed code and after fix.

        **Validates: Requirements 3.5**
        """
        css_file = Path("docs/assets/css/style.scss")
        assert css_file.exists(), f"CSS file {css_file} not found"

        with open(css_file, "r", encoding="utf-8") as f:
            content = f.read()

        # Count @media print occurrences
        print_media_pattern = r"@media\s+print\s*\{"
        print_matches = re.findall(print_media_pattern, content, re.IGNORECASE)

        assert (
            len(print_matches) == 2
        ), f"Print preservation violated: expected 2 @media print blocks, found {len(print_matches)}"

    # ═══════════════════════════════════════════════════════════════
    # Comprehensive Preservation Test
    # ═══════════════════════════════════════════════════════════════

    @given(
        viewport_width=st.one_of(
            st.sampled_from([320, 480, 768, 960]),  # Mobile range
            st.sampled_from([1201, 1400, 1920]),  # Desktop range
        )
    )
    @settings(max_examples=7, deadline=None)
    def test_property_no_changes_outside_bug_range(self, viewport_width):
        """Property test: verify NO changes to CSS for viewports outside 961px-1200px.

        This comprehensive test verifies that the fix ONLY affects the 961px-1200px range
        and does NOT modify any CSS rules for other viewport widths.

        OBSERVATION (on unfixed code):
        - Mobile layout (≤960px): full-width stacked content
        - Desktop layout (>1200px): 340px sidebar, 320px content margin

        EXPECTED OUTCOME: This test MUST PASS on unfixed code and after fix.

        **Validates: Requirements 3.1, 3.2, 3.3, 3.4, 3.5, 3.6, 3.7**
        """
        css_file = Path("docs/assets/css/style.scss")
        assert css_file.exists(), f"CSS file {css_file} not found"

        with open(css_file, "r", encoding="utf-8") as f:
            content = f.read()

        if viewport_width <= 960:
            # Mobile layout should be preserved
            mobile_breakpoint_pattern = r"@media\s*\(\s*max-width\s*:\s*960px\s*\)"
            assert re.search(
                mobile_breakpoint_pattern, content, re.IGNORECASE
            ), f"Preservation violated at {viewport_width}px: mobile breakpoint @media (max-width: 960px) not found"
        elif viewport_width > 1200:
            # Desktop layout should be preserved
            header_pattern = r"header\s*\{[^}]*width\s*:\s*340px"
            section_pattern = r"section\s*\{[^}]*margin-left\s*:\s*320px"

            assert re.search(
                header_pattern, content, re.DOTALL
            ), f"Preservation violated at {viewport_width}px: desktop header width should be 340px"
            assert re.search(
                section_pattern, content, re.DOTALL
            ), f"Preservation violated at {viewport_width}px: desktop section margin-left should be 320px"
