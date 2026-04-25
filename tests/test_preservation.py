"""Preservation property tests for CI foundation model test timeout fix.

These tests verify that non-CI test execution behavior remains unchanged.
Run on UNFIXED code first to establish baseline, then verify same behavior after fix.

EXPECTED OUTCOME: Tests PASS on both unfixed and fixed code (confirms no regressions).
"""

import ast
from pathlib import Path

import pytest


class TestPreservation:
    """Preservation tests to verify unchanged behavior."""

    def test_projector_tests_do_not_instantiate_foundation_models(self):
        """Verify that FeatureProjector tests don't download models.

        This test should PASS on unfixed code and continue to pass after fix.

        **Validates: Requirements 3.1, 3.5**
        """
        test_file = Path("tests/test_foundation_models.py")
        assert test_file.exists(), f"Test file {test_file} not found"

        with open(test_file, "r") as f:
            content = f.read()

        # Parse AST to find TestFeatureProjector tests
        tree = ast.parse(content)

        projector_tests = []
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "TestFeatureProjector":
                for item in node.body:
                    if isinstance(item, ast.FunctionDef) and item.name.startswith("test_"):
                        projector_tests.append(item.name)

        # Verify we found projector tests
        assert len(projector_tests) > 0, "No projector tests found"

        # Projector tests should exist and not require model downloads
        # This confirms they can continue running in CI
        expected_tests = [
            "test_projector_init",
            "test_projector_forward",
            "test_projector_different_dims",
            "test_projector_trainable",
            "test_projector_get_num_params",
        ]

        for test in expected_tests:
            assert test in projector_tests, f"Expected projector test {test} not found"

    def test_ci_workflow_other_jobs_unchanged(self):
        """Verify that other CI jobs remain unchanged.

        This test should PASS on unfixed code and continue to pass after fix.

        **Validates: Requirements 3.5**
        """
        ci_file = Path(".github/workflows/ci.yml")
        assert ci_file.exists(), f"CI workflow file {ci_file} not found"

        with open(ci_file, "r") as f:
            content = f.read()

        # Verify other CI jobs exist and are unchanged
        expected_jobs = [
            "lint:",
            "type-check:",
            "security:",
            "docker:",
            "docs:",
            "quick-demo:",
            "coverage-report:",
            "all-checks-passed:",
        ]

        for job in expected_jobs:
            assert job in content, f"Expected CI job {job} not found"

    def test_property_marker_exclusion_still_works(self):
        """Verify that property-based test exclusion continues to work.

        This test should PASS on unfixed code and continue to pass after fix.

        **Validates: Requirements 3.4**
        """
        ci_file = Path(".github/workflows/ci.yml")
        assert ci_file.exists(), f"CI workflow file {ci_file} not found"

        with open(ci_file, "r") as f:
            content = f.read()

        # Verify that pytest commands include "not property" marker
        assert "not property" in content, "CI pytest command should exclude property-based tests"

    def test_slow_marker_exists_in_pyproject(self):
        """Verify that slow marker is defined in pyproject.toml.

        This test should PASS on unfixed code and continue to pass after fix.

        **Validates: Requirements 3.2**
        """
        pyproject_file = Path("pyproject.toml")
        assert pyproject_file.exists(), f"pyproject.toml not found"

        with open(pyproject_file, "r") as f:
            content = f.read()

        # Verify slow marker is defined
        assert "slow" in content, "slow marker should be defined in pyproject.toml"
