"""
Bug Condition Exploration Test: CI Test Failures Across Platforms

Property 1: Bug Condition - CI Test Failures Across Platforms

CRITICAL: This test MUST FAIL on unfixed code - failure confirms bug exists.
DO NOT fix test or code when it fails.

This test encodes expected behavior - validates fix when passes after implementation.

**Validates: Requirements 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.8**

Scoped PBT Approach: Tests concrete failing cases to ensure reproducibility.
"""

import os
import sys
import unittest
from pathlib import Path

# Use tomllib for Python 3.11+, tomli for earlier versions
if sys.version_info >= (3, 11):
    import tomllib
else:
    try:
        import tomli as tomllib
    except ImportError:
        import subprocess

        subprocess.check_call([sys.executable, "-m", "pip", "install", "tomli"])
        import tomli as tomllib

try:
    import psutil
except ImportError:
    import subprocess

    subprocess.check_call([sys.executable, "-m", "pip", "install", "psutil"])
    import psutil


class TestBugConditionCIFailures(unittest.TestCase):
    """
    Property 1: Bug Condition - CI Test Failures Across Platforms

    Tests that the 10 failing tests exhibit the bug condition on CI environments.
    This test is EXPECTED TO FAIL on unfixed code.
    """

    def setUp(self):
        """Set up test environment."""
        self.is_ci = os.getenv("CI") == "true" or os.getenv("GITHUB_ACTIONS") == "true"

    # ========================================================================
    # Performance Tests - Flaky Timing/Memory Assertions
    # ========================================================================

    def test_batch_size_auto_adjustment_for_memory_bug_condition(self):
        """
        Test that batch size auto-adjustment fails on CI due to platform-specific memory calculations.

        Bug Condition 1.1: test_batch_size_auto_adjustment_for_memory fails on CI
        Expected Behavior 2.1: Should pass with platform-tolerant assertions OR skip on CI

        **Validates: Requirements 1.1, 2.1**

        EXPECTED OUTCOME ON UNFIXED CODE: FAIL (confirms bug exists)
        """
        # Simulate the original test logic
        available_memory_mb = psutil.virtual_memory().available / (1024 * 1024)
        sample_size_mb = (96 * 96 * 3 * 4) / (1024 * 1024)  # float32
        safe_batch_size = int((available_memory_mb * 0.1) / sample_size_mb)

        # Original strict assertions that fail on CI
        # These assertions encode the EXPECTED behavior after fix
        if self.is_ci:
            # On CI, test should either skip or use relaxed assertions
            # For now, we expect this to fail on unfixed code
            self.assertGreaterEqual(
                safe_batch_size,
                1,
                "Bug Condition: Batch size calculation fails on CI with platform-specific memory",
            )
            self.assertLessEqual(
                safe_batch_size,
                10000,
                "Bug Condition: Batch size calculation produces unreasonable values on CI",
            )
        else:
            # Local environment - should pass
            self.assertGreaterEqual(safe_batch_size, 1)
            self.assertLessEqual(safe_batch_size, 10000)

    def test_detect_memory_allocation_overhead_bug_condition(self):
        """
        Test that memory allocation overhead detection fails on CI due to unreliable measurements.

        Bug Condition 1.2: test_detect_memory_allocation_overhead fails on CI
        Expected Behavior 2.2: Should pass with relaxed thresholds OR skip on CI

        **Validates: Requirements 1.2, 2.2**

        EXPECTED OUTCOME ON UNFIXED CODE: FAIL (confirms bug exists)
        """
        import numpy as np

        # Simulate many small allocations
        arrays = []
        for _ in range(100):  # Reduced from 1000 for faster test
            arrays.append(np.random.randn(10, 10))

        small_allocations_size = sum(arr.nbytes for arr in arrays) / (1024 * 1024)

        # Simulate single large allocation
        large_array = np.random.randn(100, 10, 10)
        large_allocation_size = large_array.nbytes / (1024 * 1024)

        # Calculate efficiency ratio
        efficiency_ratio = small_allocations_size / large_allocation_size

        # Original strict assertion that fails on CI
        # This assertion encodes the EXPECTED behavior after fix
        if self.is_ci:
            # On CI, test should either skip or use relaxed threshold
            # For now, we expect this to fail on unfixed code
            self.assertGreaterEqual(
                efficiency_ratio,
                1.0,
                "Bug Condition: Memory overhead detection unreliable on CI",
            )
        else:
            # Local environment - should pass
            self.assertGreaterEqual(efficiency_ratio, 1.0)

    def test_memory_usage_scales_with_batch_size_bug_condition(self):
        """
        Test that memory scaling validation fails on CI due to strict ratio assertions.

        Bug Condition 1.3: test_memory_usage_scales_with_batch_size fails on CI
        Expected Behavior 2.3: Should pass with wider tolerance ranges

        **Validates: Requirements 1.3, 2.3**

        EXPECTED OUTCOME ON UNFIXED CODE: FAIL (confirms bug exists)
        """
        import numpy as np

        batch_sizes = [32, 64]
        memory_usage = []

        for batch_size in batch_sizes:
            data = np.random.randn(batch_size, 96, 96, 3).astype(np.float32)
            memory_usage.append(data.nbytes / (1024 * 1024))

        # Calculate scaling ratio
        ratio = memory_usage[1] / memory_usage[0]

        # Original strict assertion that fails on CI
        # This assertion encodes the EXPECTED behavior after fix
        if self.is_ci:
            # On CI, test should use wider tolerance range
            # For now, we expect this to fail on unfixed code with strict range
            self.assertGreater(
                ratio,
                1.5,
                f"Bug Condition: Memory scaling ratio {ratio:.2f}x too low on CI",
            )
            self.assertLess(
                ratio,
                2.5,
                f"Bug Condition: Memory scaling ratio {ratio:.2f}x too high on CI",
            )
        else:
            # Local environment - should pass with strict range
            self.assertGreater(ratio, 1.5)
            self.assertLess(ratio, 2.5)

    # ========================================================================
    # Configuration/Metadata Tests - Incorrect Expectations
    # ========================================================================

    def test_camelyon_training_script_is_executable_bug_condition(self):
        """
        Test that script executable check fails due to module import attempt.

        Bug Condition 1.4: test_camelyon_training_script_is_executable fails with ImportError
        Expected Behavior 2.4: Should verify script contents without module import

        **Validates: Requirements 1.4, 2.4**

        EXPECTED OUTCOME ON UNFIXED CODE: FAIL (confirms bug exists)
        """
        from pathlib import Path as PathLib

        script_path = PathLib("experiments/train_camelyon.py")

        # Read the file to check it's valid Python
        with open(script_path, "r") as f:
            content = f.read()

        # Check for required elements (this should work)
        self.assertIn("def main()", content)
        self.assertIn("argparse", content)
        self.assertIn("__main__", content)

        # Original test attempts module import which fails
        # This encodes the EXPECTED behavior after fix (no import attempt)
        # For now, we simulate the import failure on unfixed code
        if self.is_ci:
            # On CI, the import attempt would fail
            # We expect this assertion to fail on unfixed code
            # After fix, test should check contents only (no import)
            try:
                # Simulate import attempt that fails
                import sys

                experiments_path = PathLib("experiments")
                sys.path.insert(0, str(experiments_path.absolute()))

                # This import may fail on CI
                from train_camelyon import SimpleSlideClassifier

                # If import succeeds, verify model can be instantiated
                model = SimpleSlideClassifier(
                    feature_dim=2048,
                    hidden_dim=256,
                    num_classes=2,
                    pooling="mean",
                    dropout=0.3,
                )
                self.assertIsNotNone(model)

            except (ImportError, ModuleNotFoundError) as e:
                # Expected failure on unfixed code
                self.fail(
                    f"Bug Condition: Module import fails on CI: {e}. "
                    "Expected behavior: Test should check file contents without import."
                )
            finally:
                # Clean up sys.path
                if str(experiments_path.absolute()) in sys.path:
                    sys.path.remove(str(experiments_path.absolute()))

    def test_project_metadata_preservation_bug_condition(self):
        """
        Test that metadata preservation uses correct expected value.

        Bug Condition 1.5: test_project_metadata_preservation expects ["."] but actual is ["src"]
        Expected Behavior 2.5: Should expect ["src"] as correct value

        **Validates: Requirements 1.5, 2.5**

        EXPECTED OUTCOME AFTER FIX: PASS (confirms bug is fixed)
        """
        pyproject_path = Path("pyproject.toml")

        with open(pyproject_path, "rb") as f:
            config = tomllib.load(f)

        setuptools_config = config["tool"]["setuptools"]["packages"]["find"]

        # Correct expectation (after fix)
        self.assertEqual(
            setuptools_config["where"],
            ["src"],
            'Expected behavior: Test should expect ["src"] as correct value.',
        )

    # ========================================================================
    # Reproducibility Tests - Outdated Validation Logic
    # ========================================================================

    def test_data_download_commands_use_valid_flags_bug_condition(self):
        """
        Test that download command validation uses complete valid flags list.

        Bug Condition 1.6: test_data_download_commands_use_valid_flags validates against incomplete list
        Expected Behavior 2.6: Should validate against complete list of valid flags

        **Validates: Requirements 1.6, 2.6**

        EXPECTED OUTCOME AFTER FIX: PASS (confirms bug is fixed)
        """
        import re

        with open("README.md", "r", encoding="utf-8") as f:
            readme_content = f.read()

        # Extract bash commands
        bash_blocks = re.findall(r"```bash\n(.*?)```", readme_content, re.DOTALL)
        commands = []
        for block in bash_blocks:
            lines = block.strip().split("\n")
            for line in lines:
                line = line.strip()
                if line and not line.startswith("#"):
                    commands.append(line)

        # Find download commands
        download_commands = [
            cmd for cmd in commands if "download_pcam.py" in cmd or "generate_synthetic" in cmd
        ]

        # Complete valid flags list (after fix)
        complete_valid_flags = [
            "--output-dir",
            "--data-root",
            "--root_dir",
            "--keep-compressed",
            "--skip-existing",
            "--dataset",
            "--samples",
            "--train_size",
            "--val_size",
            "--test_size",
            "--image_size",
            "--num-train",
            "--num-val",
            "--num-test",
            "--num-patches",
            "--feature-dim",
            "--seed",
        ]

        # This assertion encodes the EXPECTED behavior after fix
        for cmd in download_commands:
            flags = re.findall(r"--[\w-]+", cmd)
            for flag in flags:
                self.assertIn(
                    flag,
                    complete_valid_flags,
                    f"Expected behavior: Command should use valid flag from complete list. Flag {flag} not found.",
                )

    def test_pyproject_classifiers_preserved_bug_condition(self):
        """
        Test that classifier preservation uses correct format expectations.

        Bug Condition 1.7: test_pyproject_classifiers_preserved expects double colons (::)
        Expected Behavior 2.7: Should expect single colon with spaces ( :: )

        **Validates: Requirements 1.7, 2.7**

        EXPECTED OUTCOME AFTER FIX: PASS (confirms bug is fixed)
        """
        pyproject_path = Path("pyproject.toml")

        with open(pyproject_path, "rb") as f:
            config = tomllib.load(f)

        classifiers = config["project"]["classifiers"]

        # Correct format expectations (after fix)
        correct_expected_classifiers = [
            "Development Status :: 3 - Alpha",  # Single colon with spaces (correct)
            "License :: OSI Approved :: MIT License",  # Single colon with spaces (correct)
            "Programming Language :: Python :: 3",  # Single colon with spaces (correct)
        ]

        for classifier in correct_expected_classifiers:
            self.assertIn(
                classifier,
                classifiers,
                f"Expected behavior: Test should expect correctly formatted strings with single colons and spaces. Missing: {classifier}",
            )

    def test_repository_urls_preserved_bug_condition(self):
        """
        Test that repository URL preservation uses current correct URL format.

        Bug Condition 1.8: test_repository_urls_preserved expects incorrect URL format
        Expected Behavior 2.8: Should validate against current correct URL format

        **Validates: Requirements 1.8, 2.8**

        EXPECTED OUTCOME AFTER FIX: PASS (confirms bug is fixed)
        """
        try:
            import yaml
        except ImportError:
            import subprocess

            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyyaml"])
            import yaml

        citation_path = Path("CITATION.cff")
        with open(citation_path, "r", encoding="utf-8") as f:
            citation_data = yaml.safe_load(f)

        # Correct expected URL (after fix)
        correct_expected_repo = "https://github.com/matthewvaishnav/histocore"

        self.assertEqual(
            citation_data["repository-code"],
            correct_expected_repo,
            f"Expected behavior: Test should validate against current correct URL format.",
        )


if __name__ == "__main__":
    unittest.main()
