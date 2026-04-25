"""Bug condition exploration test for CI foundation model test timeout fix.

This test verifies that the bug condition exists on UNFIXED code:
1. Foundation model tests in tests/test_foundation_models.py do NOT have @pytest.mark.slow decorator
2. CI pytest command in .github/workflows/ci.yml uses -m "not property" (missing "and not slow")
3. Foundation model tests are collected when running pytest with -m "not property"

EXPECTED OUTCOME: This test MUST FAIL on unfixed code - failure confirms the bug exists.
"""

import ast
import re
import subprocess
from pathlib import Path

import pytest

from hypothesis import given, settings
from hypothesis import strategies as st


class TestBugConditionExploration:
    """Exploratory tests to surface counterexamples demonstrating the bug."""

    def test_foundation_model_tests_missing_slow_marker(self):
        """Test that foundation model tests do NOT have @pytest.mark.slow decorator.

        This test SHOULD FAIL on unfixed code - failure confirms bug exists.

        **Validates: Requirements 1.3**
        """
        test_file = Path("tests/test_foundation_models.py")
        assert test_file.exists(), f"Test file {test_file} not found"

        with open(test_file, "r") as f:
            content = f.read()

        # Parse the AST to find test methods that instantiate foundation models
        tree = ast.parse(content)

        foundation_model_tests = []

        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef):
                # Check classes that test foundation models (not FeatureProjector)
                if node.name in [
                    "TestPhikonEncoder",
                    "TestLoadFoundationModel",
                    "TestFoundationModelIntegration",
                ]:
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef) and item.name.startswith("test_"):
                            # Check if this test has @pytest.mark.slow decorator
                            has_slow_marker = any(
                                isinstance(dec, ast.Attribute)
                                and isinstance(dec.value, ast.Attribute)
                                and dec.value.attr == "mark"
                                and dec.attr == "slow"
                                for dec in item.decorator_list
                            )

                            foundation_model_tests.append(
                                {
                                    "class": node.name,
                                    "method": item.name,
                                    "has_slow_marker": has_slow_marker,
                                }
                            )

        # Collect tests that are missing the slow marker
        missing_slow_marker = [
            f"{t['class']}.{t['method']}"
            for t in foundation_model_tests
            if not t["has_slow_marker"]
        ]

        # This assertion SHOULD FAIL on unfixed code
        assert len(missing_slow_marker) == 0, (
            f"Found {len(missing_slow_marker)} foundation model tests missing @pytest.mark.slow decorator: "
            f"{missing_slow_marker}. This confirms the bug exists - these tests will execute in CI and "
            f"attempt to download large models, causing timeouts."
        )

    def test_ci_pytest_command_missing_slow_exclusion(self):
        """Test that CI pytest command uses -m 'not property' (missing 'and not slow').

        This test SHOULD FAIL on unfixed code - failure confirms bug exists.

        **Validates: Requirements 1.4**
        """
        ci_file = Path(".github/workflows/ci.yml")
        assert ci_file.exists(), f"CI workflow file {ci_file} not found"

        with open(ci_file, "r") as f:
            content = f.read()

        # Find pytest commands in the test job
        # Pattern: pytest tests/ -v -m "not property"
        pytest_pattern = r'pytest\s+tests/\s+-v\s+-m\s+"([^"]+)"'
        matches = re.findall(pytest_pattern, content)

        assert len(matches) > 0, "No pytest commands found in CI workflow"

        # Check if any pytest command excludes slow tests
        commands_missing_slow = []
        for marker_expr in matches:
            if "not slow" not in marker_expr:
                commands_missing_slow.append(marker_expr)

        # This assertion SHOULD FAIL on unfixed code
        assert len(commands_missing_slow) == 0, (
            f"Found {len(commands_missing_slow)} pytest commands missing 'and not slow' exclusion: "
            f"{commands_missing_slow}. This confirms the bug exists - CI will execute slow foundation "
            f"model tests that download large models, causing timeouts."
        )

    def test_foundation_model_tests_collected_with_not_property_marker(self):
        """Test that foundation model tests are collected when running pytest with -m 'not property'.

        This test SHOULD FAIL on unfixed code - failure confirms bug exists.

        **Validates: Requirements 1.1**
        """
        # Run pytest --collect-only to see which tests would be collected
        result = subprocess.run(
            [
                "pytest",
                "tests/test_foundation_models.py",
                "--collect-only",
                "-m",
                "not property",
                "-q",
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )

        # Parse the output to find collected tests
        output = result.stdout + result.stderr

        # Look for foundation model test classes
        foundation_model_classes = [
            "TestPhikonEncoder",
            "TestLoadFoundationModel",
            "TestFoundationModelIntegration",
        ]

        collected_foundation_tests = []
        for line in output.split("\n"):
            for cls in foundation_model_classes:
                if cls in line and "test_" in line:
                    collected_foundation_tests.append(line.strip())

        # This assertion SHOULD FAIL on unfixed code
        assert len(collected_foundation_tests) == 0, (
            f"Found {len(collected_foundation_tests)} foundation model tests collected with -m 'not property': "
            f"{collected_foundation_tests}. This confirms the bug exists - these tests will execute in CI "
            f"and attempt to download large models, causing timeouts."
        )

    @given(
        marker_expr=st.sampled_from(["not property", "not property and not slow", "not slow", ""])
    )
    @settings(max_examples=4, deadline=None)
    def test_property_marker_combinations_collect_foundation_tests(self, marker_expr):
        """Property test: verify which marker expressions collect foundation model tests.

        This test explores different marker combinations to understand collection behavior.
        On unfixed code, 'not property' SHOULD collect foundation tests (bug condition).

        **Validates: Requirements 1.1, 1.4**
        """
        # Build pytest command
        cmd = ["pytest", "tests/test_foundation_models.py", "--collect-only", "-q"]
        if marker_expr:
            cmd.extend(["-m", marker_expr])

        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)

        output = result.stdout + result.stderr

        # Count foundation model tests collected
        foundation_model_classes = [
            "TestPhikonEncoder",
            "TestLoadFoundationModel",
            "TestFoundationModelIntegration",
        ]

        collected_count = 0
        for line in output.split("\n"):
            for cls in foundation_model_classes:
                if cls in line and "test_" in line:
                    collected_count += 1

        # On unfixed code:
        # - "not property" SHOULD collect foundation tests (bug!)
        # - "not property and not slow" SHOULD collect foundation tests (slow marker not applied)
        # - "not slow" SHOULD collect foundation tests (slow marker not applied)
        # - "" (no marker) SHOULD collect foundation tests

        if marker_expr == "not property":
            # This is the bug condition - foundation tests should NOT be collected
            # but they ARE collected on unfixed code
            assert collected_count == 0, (
                f"Bug confirmed: marker expression '{marker_expr}' collected {collected_count} "
                f"foundation model tests. These tests will execute in CI and cause timeouts."
            )
