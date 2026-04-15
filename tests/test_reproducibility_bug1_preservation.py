"""
Bug 1 preservation tests: Verify non-buggy paths still work.

IMPORTANT: These tests MUST PASS on unfixed code.
Captures baseline behavior to preserve after fix.
"""

import os
import tempfile
import unittest

from src.utils.benchmark_manifest import BenchmarkManifest


class TestBug1Preservation(unittest.TestCase):
    """Property 2: Preservation - Directory-Prefixed and Absolute Paths."""

    def setUp(self):
        """Set up test with temp directory."""
        self.original_cwd = os.getcwd()
        self.temp_dir = tempfile.mkdtemp()
        os.chdir(self.temp_dir)

    def tearDown(self):
        """Clean up temp directory."""
        os.chdir(self.original_cwd)
        import shutil

        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_directory_prefixed_relative_path(self):
        """Test BenchmarkManifest with directory-prefixed relative path."""
        # Non-buggy input: has directory component
        manifest_path = "benchmarks/manifest.jsonl"
        self.assertNotEqual(os.path.dirname(manifest_path), "")

        # Should work on unfixed code
        manifest = BenchmarkManifest(manifest_path)

        # Verify parent directory created
        self.assertTrue(os.path.exists("benchmarks"))
        self.assertTrue(os.path.isdir("benchmarks"))

        # Verify manifest path set correctly
        self.assertEqual(manifest.manifest_path, manifest_path)

    def test_absolute_path(self):
        """Test BenchmarkManifest with absolute path."""
        # Non-buggy input: absolute path
        abs_path = os.path.join(self.temp_dir, "results", "manifest.jsonl")
        self.assertNotEqual(os.path.dirname(abs_path), "")
        self.assertTrue(os.path.isabs(abs_path))

        # Should work on unfixed code
        manifest = BenchmarkManifest(abs_path)

        # Verify parent directory created
        parent_dir = os.path.dirname(abs_path)
        self.assertTrue(os.path.exists(parent_dir))
        self.assertTrue(os.path.isdir(parent_dir))

        # Verify manifest path set correctly
        self.assertEqual(manifest.manifest_path, abs_path)

    def test_default_path_none_parameter(self):
        """Test BenchmarkManifest with None (default path)."""
        # Non-buggy input: None uses default
        manifest = BenchmarkManifest(None)

        # Should use default path
        expected_default = "benchmarks/manifest.jsonl"
        self.assertEqual(manifest.manifest_path, expected_default)

        # Verify parent directory created
        self.assertTrue(os.path.exists("benchmarks"))
        self.assertTrue(os.path.isdir("benchmarks"))

    def test_nested_directory_path(self):
        """Test BenchmarkManifest with nested directory path."""
        manifest_path = "results/experiments/run1/manifest.jsonl"
        self.assertNotEqual(os.path.dirname(manifest_path), "")

        manifest = BenchmarkManifest(manifest_path)

        # Verify all parent directories created
        self.assertTrue(os.path.exists("results/experiments/run1"))
        self.assertEqual(manifest.manifest_path, manifest_path)

    def test_single_directory_prefix(self):
        """Test BenchmarkManifest with single directory prefix."""
        manifest_path = "output/manifest.jsonl"
        self.assertNotEqual(os.path.dirname(manifest_path), "")

        manifest = BenchmarkManifest(manifest_path)

        self.assertTrue(os.path.exists("output"))
        self.assertEqual(manifest.manifest_path, manifest_path)


if __name__ == "__main__":
    unittest.main()
