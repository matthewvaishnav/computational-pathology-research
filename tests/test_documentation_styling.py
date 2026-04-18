#!/usr/bin/env python3
"""
Bug condition exploration test for documentation styling fix.

This test demonstrates the white-on-white text visibility issue in the GitHub Pages
documentation website. It should FAIL on unfixed code, proving the bug exists.

CRITICAL: This test MUST FAIL on unfixed code - failure confirms the bug exists.
DO NOT attempt to fix the test or the code when it fails.
"""

import os
import subprocess
import tempfile
from pathlib import Path
import pytest


def test_bug_condition_white_on_white_text():
    """
    Property 1: Bug Condition - White-on-White Text Visibility
    
    Test that Jekyll minimal theme import order causes CSS specificity issues
    that lead to white-on-white text in code blocks.
    
    EXPECTED OUTCOME: Test FAILS (this is correct - it proves the bug exists)
    """
    # Path to the current CSS file
    css_file = Path("docs/assets/css/style.scss")
    
    # Verify the CSS file exists
    assert css_file.exists(), f"CSS file not found: {css_file}"
    
    # Read the current CSS content
    css_content = css_file.read_text()
    
    # Check for the actual bug condition based on the screenshot evidence
    
    # 1. Jekyll theme is imported at the top
    has_theme_import = "@import \"{{ site.theme }}\"" in css_content
    
    # 2. Custom pre/code styles exist but may be overridden by theme
    has_custom_pre_styles = "pre {" in css_content and "background: #2d2d2d" in css_content
    has_custom_code_color = "color: #f8f8f2" in css_content
    
    # 3. Check if styles lack sufficient specificity or !important to override theme
    lacks_important_declarations = "color: #f8f8f2" in css_content and "!important" not in css_content.split("color: #f8f8f2")[1].split(";")[0]
    
    # 4. Check for Jekyll minimal theme specificity issues
    # The theme likely has more specific selectors that override our custom styles
    has_defensive_selectors = any([
        "section pre code" in css_content,
        ".wrapper pre code" in css_content,
        "div.highlight" in css_content
    ])
    
    # Bug condition: Theme import + custom styles + insufficient specificity = overridden styles
    bug_condition_detected = (
        has_theme_import and 
        has_custom_pre_styles and 
        has_custom_code_color and
        lacks_important_declarations and
        not has_defensive_selectors
    )
    
    # Document the counterexample
    if bug_condition_detected:
        print("\n=== BUG CONDITION DETECTED ===")
        print("Jekyll minimal theme imported BEFORE custom styles")
        print("Custom pre/code styles exist but lack sufficient specificity")
        print("Theme's more specific selectors override custom colors")
        print("Result: Code text becomes invisible (white on white/light background)")
        print("Evidence: User screenshot shows invisible text until highlighted")
        
    # This assertion should FAIL on unfixed code, proving the bug exists
    assert not bug_condition_detected, (
        "Bug condition confirmed: Jekyll minimal theme import order and CSS specificity issues "
        "cause custom code colors to be overridden, resulting in invisible white-on-white text. "
        "Theme has higher specificity selectors that override custom pre/code styling."
    )


def test_inline_code_color_conflicts():
    """
    Test that inline <code> elements may inherit conflicting colors from parent elements.
    
    EXPECTED OUTCOME: May FAIL on unfixed code if color conflicts exist.
    """
    css_file = Path("docs/assets/css/style.scss")
    css_content = css_file.read_text()
    
    # Check if inline code has explicit color that won't be overridden by theme
    inline_code_has_color = "code {" in css_content and "color:" in css_content
    inline_code_has_important = "code {" in css_content and "!important" in css_content
    
    # Bug condition: Inline code styling exists but may be overridden by theme
    has_inline_code_styling = "code {" in css_content
    theme_override_protection = inline_code_has_important or "section code" in css_content
    
    if has_inline_code_styling and not theme_override_protection:
        print("\n=== INLINE CODE ISSUE DETECTED ===")
        print("Inline code styling exists but lacks theme override protection")
        print("Jekyll minimal theme may override custom colors")
    
    # This may fail if inline code lacks proper theme override protection
    assert not (has_inline_code_styling and not theme_override_protection), (
        "Inline code styling lacks theme override protection. "
        "Jekyll minimal theme may override custom colors causing visibility issues."
    )


def test_css_specificity_conflicts():
    """
    Test that custom styles have sufficient specificity to override Jekyll minimal theme.
    
    EXPECTED OUTCOME: May FAIL if CSS specificity is insufficient.
    """
    css_file = Path("docs/assets/css/style.scss")
    css_content = css_file.read_text()
    
    # Check for Jekyll theme import
    has_theme_import = "{{ site.theme }}" in css_content
    
    # Check for high-specificity selectors that can override theme
    has_specific_selectors = any([
        "section pre code" in css_content,
        ".wrapper pre code" in css_content,
        "section code" in css_content,
        ".wrapper code" in css_content
    ])
    
    # Check for !important declarations
    has_important_declarations = "!important" in css_content
    
    if has_theme_import and not (has_specific_selectors or has_important_declarations):
        print("\n=== CSS SPECIFICITY ISSUE DETECTED ===")
        print("Jekyll theme imported but insufficient specificity to override defaults")
        print("Custom styles may be overridden by theme styles")
    
    # This may fail if CSS lacks sufficient specificity
    specificity_sufficient = not has_theme_import or has_specific_selectors or has_important_declarations
    
    assert specificity_sufficient, (
        "CSS specificity insufficient to override Jekyll minimal theme. "
        "Theme import detected but no high-specificity selectors or !important declarations found."
    )


if __name__ == "__main__":
    # Run the tests and expect them to fail on unfixed code
    print("Running bug condition exploration tests...")
    print("EXPECTED: These tests should FAIL on unfixed code, proving the bug exists")
    
    try:
        test_bug_condition_white_on_white_text()
        print("❌ UNEXPECTED: Bug condition test passed - bug may not exist or test is incorrect")
    except AssertionError as e:
        print(f"✅ EXPECTED: Bug condition test failed - bug confirmed: {e}")
    
    try:
        test_inline_code_color_conflicts()
        print("✅ Inline code test passed")
    except AssertionError as e:
        print(f"⚠️  Inline code issue detected: {e}")
    
    try:
        test_css_specificity_conflicts()
        print("✅ CSS specificity test passed")
    except AssertionError as e:
        print(f"⚠️  CSS specificity issue detected: {e}")