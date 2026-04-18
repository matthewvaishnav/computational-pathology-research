#!/usr/bin/env python3
"""
Preservation property tests for documentation styling fix.

These tests capture the current styling behavior that should remain unchanged
after implementing the white-on-white text fix.

IMPORTANT: Follow observation-first methodology - these tests observe behavior
on UNFIXED code and should PASS, establishing the baseline to preserve.
"""

import os
import re
from pathlib import Path
import pytest


def test_preserve_professional_academic_styling():
    """
    Property 2: Preservation - Professional Academic Typography
    
    Test that professional academic styling with Crimson Text and Lato fonts is preserved.
    
    EXPECTED OUTCOME: Tests PASS (confirms baseline behavior to preserve)
    """
    css_file = Path("docs/assets/css/style.scss")
    css_content = css_file.read_text()
    
    # Check for Crimson Text font family for body text
    has_crimson_text = "font-family: 'Crimson Text'" in css_content
    
    # Check for Lato font family for headings
    has_lato_headings = "font-family: 'Lato'" in css_content
    
    # Check for professional typography settings
    has_proper_line_height = "line-height: 1.8" in css_content
    has_letter_spacing = "letter-spacing: -0.02em" in css_content
    
    assert has_crimson_text, "Crimson Text font family must be preserved for body text"
    assert has_lato_headings, "Lato font family must be preserved for headings"
    assert has_proper_line_height, "Professional line-height (1.8) must be preserved"
    assert has_letter_spacing, "Letter spacing for headings must be preserved"
    
    print("✅ Professional academic typography preserved")


def test_preserve_harvard_crimson_branding():
    """
    Test that Harvard crimson color scheme (#A51C30) for headers and branding remains intact.
    """
    css_file = Path("docs/assets/css/style.scss")
    css_content = css_file.read_text()
    
    # Count occurrences of Harvard crimson color
    crimson_occurrences = css_content.count("#A51C30")
    
    # Should appear in multiple places: headers, links, borders, etc.
    assert crimson_occurrences >= 8, f"Harvard crimson (#A51C30) should appear at least 8 times, found {crimson_occurrences}"
    
    # Check specific branding elements
    has_crimson_headers = "color: #A51C30" in css_content
    has_crimson_links = "a {" in css_content and "#A51C30" in css_content
    has_crimson_borders = "border" in css_content and "#A51C30" in css_content
    
    assert has_crimson_headers, "Harvard crimson must be preserved for headers"
    assert has_crimson_links, "Harvard crimson must be preserved for links"
    assert has_crimson_borders, "Harvard crimson must be preserved for borders"
    
    print("✅ Harvard crimson branding preserved")


def test_preserve_responsive_design():
    """
    Test that responsive design breakpoints and mobile layout behavior continues working.
    """
    css_file = Path("docs/assets/css/style.scss")
    css_content = css_file.read_text()
    
    # Check for responsive breakpoint
    has_mobile_breakpoint = "@media (max-width: 960px)" in css_content
    
    # Check for responsive layout changes
    has_responsive_wrapper = "width: 90%" in css_content
    has_responsive_positioning = "position: static" in css_content
    has_responsive_grid = "grid-template-columns: 1fr" in css_content
    
    assert has_mobile_breakpoint, "Mobile breakpoint (960px) must be preserved"
    assert has_responsive_wrapper, "Responsive wrapper width must be preserved"
    assert has_responsive_positioning, "Responsive positioning changes must be preserved"
    assert has_responsive_grid, "Responsive grid layout must be preserved"
    
    print("✅ Responsive design behavior preserved")


def test_preserve_navigation_styling():
    """
    Test that navigation styling and sidebar layout remain unchanged.
    """
    css_file = Path("docs/assets/css/style.scss")
    css_content = css_file.read_text()
    
    # Check header/sidebar styling
    has_fixed_header = "position: fixed" in css_content
    has_header_width = "width: 270px" in css_content
    has_nav_styling = "nav {" in css_content
    
    # Check navigation link styling
    has_nav_background = "background: #f8f8f8" in css_content
    has_nav_border = "border-left: 4px solid #A51C30" in css_content
    
    assert has_fixed_header, "Fixed header positioning must be preserved"
    assert has_header_width, "Header width (270px) must be preserved"
    assert has_nav_styling, "Navigation styling must be preserved"
    assert has_nav_background, "Navigation background color must be preserved"
    assert has_nav_border, "Navigation border styling must be preserved"
    
    print("✅ Navigation styling preserved")


def test_preserve_grid_layouts():
    """
    Test that grid layouts for feature cards maintain current appearance.
    """
    css_file = Path("docs/assets/css/style.scss")
    css_content = css_file.read_text()
    
    # Check for feature grid
    has_features_grid = ".features-grid {" in css_content
    has_grid_template = "grid-template-columns: repeat(auto-fit, minmax(280px, 1fr))" in css_content
    has_grid_gap = "gap: 20px" in css_content
    
    # Check for feature card styling
    has_feature_cards = ".feature-card {" in css_content
    has_card_background = "background: #f9f9f9" in css_content
    has_card_border = "border: 1px solid #e0e0e0" in css_content
    
    assert has_features_grid, "Features grid layout must be preserved"
    assert has_grid_template, "Grid template columns must be preserved"
    assert has_grid_gap, "Grid gap spacing must be preserved"
    assert has_feature_cards, "Feature card styling must be preserved"
    assert has_card_background, "Feature card background must be preserved"
    assert has_card_border, "Feature card borders must be preserved"
    
    print("✅ Grid layouts preserved")


def test_preserve_print_styles():
    """
    Test that print styles continue to provide appropriate formatting.
    """
    css_file = Path("docs/assets/css/style.scss")
    css_content = css_file.read_text()
    
    # Check for print media query
    has_print_media = "@media print {" in css_content
    
    # Check for print-specific styling
    has_print_hero = "border-bottom: 2px solid #000" in css_content
    has_print_cards = "break-inside: avoid" in css_content
    has_print_nav_hidden = "display: none" in css_content
    
    assert has_print_media, "Print media query must be preserved"
    assert has_print_hero, "Print hero styling must be preserved"
    assert has_print_cards, "Print card behavior must be preserved"
    assert has_print_nav_hidden, "Print navigation hiding must be preserved"
    
    print("✅ Print styles preserved")


def test_preserve_non_code_text_colors():
    """
    Test that all non-code text colors remain unchanged after the fix.
    """
    css_file = Path("docs/assets/css/style.scss")
    css_content = css_file.read_text()
    
    # Check body text color
    has_body_color = "color: #2c2c2c" in css_content
    
    # Check heading colors
    has_heading_color = "color: #1a1a1a" in css_content
    
    # Check various text colors
    has_paragraph_colors = "color: #666" in css_content
    has_subtitle_color = "color: #555" in css_content
    has_footer_color = "color: #777" in css_content
    
    assert has_body_color, "Body text color (#2c2c2c) must be preserved"
    assert has_heading_color, "Heading color (#1a1a1a) must be preserved"
    assert has_paragraph_colors, "Paragraph colors must be preserved"
    assert has_subtitle_color, "Subtitle colors must be preserved"
    assert has_footer_color, "Footer colors must be preserved"
    
    print("✅ Non-code text colors preserved")


def test_preserve_layout_dimensions():
    """
    Test that layout dimensions and positioning remain unchanged.
    """
    css_file = Path("docs/assets/css/style.scss")
    css_content = css_file.read_text()
    
    # Check wrapper width
    has_wrapper_width = "width: 960px" in css_content
    
    # Check section width
    has_section_width = "width: 650px" in css_content
    
    # Check margins and padding
    has_proper_margins = "margin: 0 auto" in css_content
    has_section_padding = "padding-bottom: 50px" in css_content
    
    assert has_wrapper_width, "Wrapper width (960px) must be preserved"
    assert has_section_width, "Section width (650px) must be preserved"
    assert has_proper_margins, "Auto margins must be preserved"
    assert has_section_padding, "Section padding must be preserved"
    
    print("✅ Layout dimensions preserved")


if __name__ == "__main__":
    # Run all preservation tests - they should PASS on unfixed code
    print("Running preservation property tests...")
    print("EXPECTED: These tests should PASS on unfixed code, establishing baseline to preserve")
    
    tests = [
        test_preserve_professional_academic_styling,
        test_preserve_harvard_crimson_branding,
        test_preserve_responsive_design,
        test_preserve_navigation_styling,
        test_preserve_grid_layouts,
        test_preserve_print_styles,
        test_preserve_non_code_text_colors,
        test_preserve_layout_dimensions
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"❌ FAILED: {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"❌ ERROR: {test.__name__}: {e}")
            failed += 1
    
    print(f"\n📊 Preservation Tests Results: {passed} passed, {failed} failed")
    
    if failed == 0:
        print("✅ All preservation tests passed - baseline behavior captured")
    else:
        print("⚠️  Some preservation tests failed - review baseline behavior")