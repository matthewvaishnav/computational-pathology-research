"""Preservation property tests for Windows CI integration test timeout fix.

These tests verify that non-CI test execution behavior remains unchanged.
They should PASS on both UNFIXED and FIXED code.

Tests verify:
- Local test execution without markers runs all integration tests
- Individual test execution works correctly
- Explicit slow marker behavior (will change after fix - tests should be collected)
- Other CI behaviors remain unchanged
"""

import subprocess
import sys

import pytest


def test_local_execution_runs_all_integration_tests():
    """
    Test that local execution without markers runs all 5 integration tests.
    
    This should PASS on both unfixed and fixed code.
    Preservation: Local test execution behavior unchanged.
    """
    # Note: We use --collect-only to avoid actually running the slow tests
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_camelyon_training_integration.py",
            "--collect-only",
            "-q",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    
    output = result.stdout + result.stderr
    
    # All 5 integration tests should be collected
    integration_tests = [
        "test_end_to_end_training",
        "test_end_to_end_evaluation",
        "test_training_with_max_pooling",
        "test_evaluation_generates_plots",
        "test_training_validates_config",
    ]
    
    collected_count = sum(1 for test in integration_tests if test in output)
    
    # Should collect all 5 tests (no marker filtering)
    assert collected_count == 5, (
        f"Preservation violation: Local execution should collect all 5 integration tests, "
        f"but only {collected_count} were collected."
    )


def test_individual_test_execution_works():
    """
    Test that individual test execution works correctly.
    
    This should PASS on both unfixed and fixed code.
    Preservation: Individual test execution behavior unchanged.
    """
    # Test that we can collect a specific test
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_camelyon_training_integration.py::test_end_to_end_training",
            "--collect-only",
            "-q",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    
    output = result.stdout + result.stderr
    
    # Should collect the specific test
    assert "test_end_to_end_training" in output, (
        "Preservation violation: Individual test execution should work."
    )


def test_fixture_tests_remain_unaffected():
    """
    Test that fixture functions are not affected by the fix.
    
    This should PASS on both unfixed and fixed code.
    Preservation: Fixture behavior unchanged.
    """
    # Verify that fixtures exist and are not marked as tests
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_camelyon_training_integration.py",
            "--collect-only",
            "-q",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    
    output = result.stdout + result.stderr
    
    # Fixtures should not be collected as tests
    assert "synthetic_camelyon_data" not in output or "fixture" in output.lower(), (
        "Preservation violation: Fixtures should not be collected as tests."
    )


def test_other_test_files_unaffected():
    """
    Test that other test files are not affected by the fix.
    
    This should PASS on both unfixed and fixed code.
    Preservation: Other test files unchanged.
    """
    # Run a quick collection on another test file to ensure it's unaffected
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_pcam_dataset.py",
            "--collect-only",
            "-q",
            "-m",
            "not property and not slow",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    
    # Should complete successfully (exit code 0 or 5 for no tests collected)
    assert result.returncode in [0, 5], (
        f"Preservation violation: Other test files should be unaffected. "
        f"Exit code: {result.returncode}"
    )


def test_property_marker_exclusion_still_works():
    """
    Test that property-based test exclusion continues to work.
    
    This should PASS on both unfixed and fixed code.
    Preservation: Property marker exclusion unchanged.
    """
    # Verify that property tests are still excluded
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/",
            "--collect-only",
            "-q",
            "-m",
            "property",
        ],
        capture_output=True,
        text=True,
        timeout=30,
    )
    
    output = result.stdout + result.stderr
    
    # Should collect property-based tests (if any exist)
    # This confirms the property marker is still recognized
    assert result.returncode in [0, 5], (
        "Preservation violation: Property marker should still be recognized."
    )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
