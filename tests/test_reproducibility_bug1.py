"""
Bug 1 exploration test: BenchmarkManifest simple filename handling.

CRITICAL: This test MUST FAIL on unfixed code.
Expected failure: FileNotFoundError when dirname is empty string.
"""

import os
import tempfile
import unittest

from src.utils.benchmark_manifest import BenchmarkManifest


class TestBug1SimpleFilenameHandling(unittest.TestCase):
    """Property 1: Bug Condition - Simple Filename Crashes."""

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

    def test_simple_filename_manifest_jsonl(self):
        """Test BenchmarkManifest with 'manifest.jsonl' (simple filename)."""
        # Bug condition: os.path.dirname("manifest.jsonl") == ""
        simple_filename = "manifest.jsonl"
        self.assertEqual(os.path.dirname(simple_filename), "")

        # Expected behavior (after fix): Should NOT raise FileNotFoundError
        # Current behavior (unfixed): WILL raise FileNotFoundError
        manifest = BenchmarkManifest(simple_filename)

        # Verify manifest created successfully (path set, no crash)
        self.assertEqual(manifest.manifest_path, simple_filename)
        # File doesn't exist yet (only created on add_entry), but no crash is the fix

    def test_simple_filename_results_jsonl(self):
        """Test BenchmarkManifest with 'results.jsonl' (another simple filename)."""
        simple_filename = "results.jsonl"
        self.assertEqual(os.path.dirname(simple_filename), "")

        # Expected behavior (after fix): Should NOT raise FileNotFoundError
        manifest = BenchmarkManifest(simple_filename)

        self.assertEqual(manifest.manifest_path, simple_filename)
        # File doesn't exist yet, but no crash is the fix


if __name__ == "__main__":
    unittest.main()
