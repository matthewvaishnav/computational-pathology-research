"""Bug condition exploration test for Windows CI integration test timeout fix.

This test verifies that the bug condition exists on UNFIXED code:
- Integration tests in test_camelyon_training_integration.py execute in CI
- Tests lack @pytest.mark.slow decorator
- Tests would cause Windows CI timeout due to cumulative execution time

CRITICAL: This test MUST FAIL on unfixed code - failure confirms the bug exists.
DO NOT attempt to fix the test or the code when it fails.

After the fix is applied, this same test will PASS, confirming the bug is fixed.
"""

import subprocess
import sys
from pathlib import Path

import pytest


def test_integration_tests_missing_slow_marker():
    """
    Test that integration tests in test_camelyon_training_integration.py
    are missing @pytest.mark.slow decorator.
    
    This test encodes the EXPECTED behavior (tests should have slow marker).
    On UNFIXED code, this test will FAIL (confirming bug exists).
    On FIXED code, this test will PASS (confirming bug is fixed).
    """
    # Integration test names that should have @pytest.mark.slow
    integration_tests = [
        "test_end_to_end_training",
        "test_end_to_end_evaluation",
        "test_training_with_max_pooling",
        "test_evaluation_generates_plots",
        "test_training_validates_config",
    ]
    
    # Collect tests with CI marker expression (what CI uses)
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_camelyon_training_integration.py",
            "--collect-only",
            "-q",
            "-m",
            "not property and not slow",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    
    output = result.stdout + result.stderr
    
    # Check which integration tests are collected (would execute in CI)
    collected_integration_tests = []
    for test_name in integration_tests:
        if test_name in output:
            collected_integration_tests.append(test_name)
    
    # EXPECTED BEHAVIOR: No integration tests should be collected
    # (they should all be skipped due to slow marker)
    # On UNFIXED code: This assertion will FAIL (bug exists)
    # On FIXED code: This assertion will PASS (bug is fixed)
    assert len(collected_integration_tests) == 0, (
        f"Bug confirmed: {len(collected_integration_tests)} integration tests "
        f"would execute in CI (missing @pytest.mark.slow): {collected_integration_tests}. "
        f"These tests run subprocess calls with 390+ seconds cumulative timeout, "
        f"causing Windows CI to timeout after 17-21 minutes."
    )


def test_integration_tests_not_collected_with_slow_marker():
    """
    Test that integration tests are collected when using -m slow marker.
    
    On UNFIXED code: No tests collected (they lack slow marker) - FAIL
    On FIXED code: All 5 tests collected (they have slow marker) - PASS
    """
    # Collect tests with slow marker
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_camelyon_training_integration.py",
            "--collect-only",
            "-q",
            "-m",
            "slow",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    
    output = result.stdout + result.stderr
    
    # Count how many integration tests are collected
    integration_tests = [
        "test_end_to_end_training",
        "test_end_to_end_evaluation",
        "test_training_with_max_pooling",
        "test_evaluation_generates_plots",
        "test_training_validates_config",
    ]
    
    collected_count = sum(1 for test in integration_tests if test in output)
    
    # EXPECTED BEHAVIOR: All 5 integration tests should be collected with -m slow
    # On UNFIXED code: This assertion will FAIL (tests lack slow marker)
    # On FIXED code: This assertion will PASS (tests have slow marker)
    assert collected_count == 5, (
        f"Bug confirmed: Only {collected_count}/5 integration tests collected with -m slow. "
        f"Expected all 5 tests to have @pytest.mark.slow decorator."
    )


def test_ci_marker_expression_skips_integration_tests():
    """
    Test that CI marker expression 'not property and not slow' skips integration tests.
    
    This is the core bug condition test - verifies tests are skipped in CI.
    """
    # Run pytest with CI marker expression
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_camelyon_training_integration.py",
            "-v",
            "-m",
            "not property and not slow",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    
    output = result.stdout + result.stderr
    
    # Check for "SKIPPED" in output for each integration test
    integration_tests = [
        "test_end_to_end_training",
        "test_end_to_end_evaluation",
        "test_training_with_max_pooling",
        "test_evaluation_generates_plots",
        "test_training_validates_config",
    ]
    
    # Count how many tests were skipped vs executed
    skipped_count = 0
    executed_count = 0
    
    for test_name in integration_tests:
        if f"{test_name}" in output:
            if "SKIPPED" in output or "skipped" in output:
                skipped_count += 1
            else:
                executed_count += 1
    
    # EXPECTED BEHAVIOR: All 5 tests should be skipped
    # On UNFIXED code: Tests execute (not skipped) - FAIL
    # On FIXED code: Tests are skipped - PASS
    assert executed_count == 0, (
        f"Bug confirmed: {executed_count} integration tests would EXECUTE in CI "
        f"(should be SKIPPED). These tests run subprocess calls with 390+ seconds "
        f"cumulative timeout, causing Windows CI to timeout."
    )
    
    assert skipped_count == 5, (
        f"Expected all 5 integration tests to be SKIPPED in CI, "
        f"but only {skipped_count} were skipped."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
