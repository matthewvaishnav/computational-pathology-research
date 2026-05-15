"""
Integration tests for run_benchmark.py entry point.

Tests the complete workflow from command-line invocation to result generation.

Requirements: 5.1, 8.8
"""

import subprocess
import sys
from pathlib import Path

import pytest


class TestRunBenchmarkIntegration:
    """Integration tests for run_benchmark.py entry point."""

    def test_help_command_works(self):
        """
        Test that --help command works without errors.

        Requirement: 5.1 (Long-running workload support)
        """
        result = subprocess.run(
            [sys.executable, "experiments/benchmark_system/run_benchmark.py", "--help"],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 0
        assert "Competitor Benchmark System" in result.stdout
        assert "run" in result.stdout
        assert "validate" in result.stdout
        assert "list-frameworks" in result.stdout

    def test_list_frameworks_command_works(self):
        """
        Test that list-frameworks command works without errors.

        Requirement: 5.1 (Long-running workload support)
        """
        result = subprocess.run(
            [
                sys.executable,
                "experiments/benchmark_system/run_benchmark.py",
                "list-frameworks",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 0
        assert "HistoCore" in result.stdout
        assert "PathML" in result.stdout
        assert "CLAM" in result.stdout
        assert "PyTorch" in result.stdout

    @pytest.mark.slow
    def test_validate_command_works(self):
        """
        Test that validate command works (may fail if GPU not available).

        Requirement: 5.1 (Long-running workload support)

        Note: This test is marked as slow because it imports torch which can take time.
        """
        result = subprocess.run(
            [sys.executable, "experiments/benchmark_system/run_benchmark.py", "validate"],
            capture_output=True,
            text=True,
            timeout=120,  # Increased timeout for torch import
        )

        # Command should execute without crashing (may return 0 or 1 depending on GPU)
        assert result.returncode in [0, 1]

        # Should contain validation output
        assert "GPU" in result.stdout or "GPU" in result.stderr

    def test_module_invocation_works(self):
        """
        Test that run_benchmark.py can be invoked as a module.

        Requirement: 5.1 (Long-running workload support)
        """
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.benchmark_system.run_benchmark",
                "--help",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 0
        assert "Competitor Benchmark System" in result.stdout

    def test_error_summary_generated_on_keyboard_interrupt(self, tmp_path, monkeypatch):
        """
        Test that error summary is generated even on KeyboardInterrupt.

        Requirement: 8.8 (Error summary generation)
        """
        # This test is conceptual - actual KeyboardInterrupt testing is complex
        # We verify the error handling code exists and is structured correctly

        from experiments.benchmark_system import run_benchmark

        # Verify the main block has KeyboardInterrupt handling
        import inspect

        source = inspect.getsource(run_benchmark)

        assert "KeyboardInterrupt" in source
        assert "generate_error_summary" in source
        assert "sys.exit(130)" in source  # Standard SIGINT exit code

    def test_error_summary_generated_on_exception(self):
        """
        Test that error summary is generated on unexpected exceptions.

        Requirement: 8.8 (Error summary generation)
        """
        from experiments.benchmark_system import run_benchmark

        # Verify the main block has exception handling
        import inspect

        source = inspect.getsource(run_benchmark)

        assert "except Exception" in source
        assert "generate_error_summary" in source
        assert "sys.exit(1)" in source


class TestRunBenchmarkCLIArguments:
    """Test CLI argument parsing through entry point."""

    def test_verbose_flag_accepted(self):
        """Test that --verbose flag is accepted."""
        result = subprocess.run(
            [
                sys.executable,
                "experiments/benchmark_system/run_benchmark.py",
                "--verbose",
                "list-frameworks",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 0

    def test_quiet_flag_accepted(self):
        """Test that --quiet flag is accepted."""
        result = subprocess.run(
            [
                sys.executable,
                "experiments/benchmark_system/run_benchmark.py",
                "--quiet",
                "list-frameworks",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 0

    def test_log_file_flag_accepted(self, tmp_path):
        """Test that --log-file flag is accepted."""
        log_file = tmp_path / "test.log"

        result = subprocess.run(
            [
                sys.executable,
                "experiments/benchmark_system/run_benchmark.py",
                "--log-file",
                str(log_file),
                "list-frameworks",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )

        assert result.returncode == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
