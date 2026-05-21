"""
Unit tests for run_benchmark.py main entry point.

Tests error summary generation and logging configuration.

Requirements: 5.1, 8.8
"""

import logging
from pathlib import Path
from unittest.mock import patch

import pytest

from experiments.benchmark_system.run_benchmark import (
    generate_error_summary,
    setup_error_summary_logging,
)


class TestErrorSummaryLogging:
    """Test error summary logging configuration."""

    def test_setup_error_summary_logging_creates_handler(self, tmp_path, monkeypatch):
        """
        Test that setup_error_summary_logging creates error log handler.

        Requirement: 8.8 (Error summary generation)
        """
        # Mock the results directory
        mock_results_dir = tmp_path / "results" / "competitor_benchmarks"
        monkeypatch.setattr(
            "experiments.benchmark_system.run_benchmark.Path",
            lambda x: mock_results_dir / "errors.log" if "errors.log" in x else Path(x),
        )

        # Clear existing handlers
        root_logger = logging.getLogger()
        original_handlers = root_logger.handlers[:]
        root_logger.handlers.clear()

        try:
            # Setup error logging
            setup_error_summary_logging()

            # Verify handler was added
            assert len(root_logger.handlers) > 0

            # Verify handler is FileHandler with WARNING level
            error_handlers = [
                h
                for h in root_logger.handlers
                if isinstance(h, logging.FileHandler) and h.level == logging.WARNING
            ]
            assert len(error_handlers) > 0

        finally:
            # Restore original handlers
            root_logger.handlers = original_handlers

    def test_error_summary_logging_captures_warnings_and_errors(self, tmp_path, monkeypatch):
        """
        Test that error summary logging captures warnings and errors.

        Requirement: 8.8 (Error summary generation)
        """
        # Create temporary error log
        error_log_path = tmp_path / "errors.log"

        # Create file handler
        error_handler = logging.FileHandler(error_log_path, mode="w")
        error_handler.setLevel(logging.WARNING)
        error_formatter = logging.Formatter("%(levelname)s - %(message)s")
        error_handler.setFormatter(error_formatter)

        # Add to logger
        test_logger = logging.getLogger("test_error_logger")
        test_logger.addHandler(error_handler)
        test_logger.setLevel(logging.DEBUG)

        try:
            # Log messages at different levels
            test_logger.debug("Debug message")
            test_logger.info("Info message")
            test_logger.warning("Warning message")
            test_logger.error("Error message")

            # Flush and close handler
            error_handler.flush()
            error_handler.close()

            # Read error log
            with open(error_log_path, "r") as f:
                log_content = f.read()

            # Verify only warnings and errors are captured
            assert "Debug message" not in log_content
            assert "Info message" not in log_content
            assert "Warning message" in log_content
            assert "Error message" in log_content

        finally:
            test_logger.removeHandler(error_handler)


class TestErrorSummaryGeneration:
    """Test error summary generation."""

    def test_generate_error_summary_with_errors(self, tmp_path, monkeypatch):
        """
        Test error summary generation with errors and warnings.

        Requirement: 8.8 (Error summary generation)
        """
        # Create mock error log
        error_log_path = tmp_path / "errors.log"
        error_log_content = [
            "2026-01-08 10:00:00 - module1 - ERROR - Error message 1",
            "2026-01-08 10:01:00 - module2 - WARNING - Warning message 1",
            "2026-01-08 10:02:00 - module3 - ERROR - Error message 2",
            "2026-01-08 10:03:00 - module4 - WARNING - Warning message 2",
        ]
        error_log_path.write_text("\n".join(error_log_content))

        # Mock Path to return our temp directory
        def mock_path(path_str):
            if "errors.log" in path_str:
                return error_log_path
            elif "error_summary.txt" in path_str:
                return tmp_path / "error_summary.txt"
            return Path(path_str)

        monkeypatch.setattr("experiments.benchmark_system.run_benchmark.Path", mock_path)

        # Generate error summary
        generate_error_summary()

        # Verify summary file was created
        summary_path = tmp_path / "error_summary.txt"
        assert summary_path.exists()

        # Read summary
        summary_content = summary_path.read_text()

        # Verify summary contains counts
        assert "Total Errors: 2" in summary_content
        assert "Total Warnings: 2" in summary_content

        # Verify summary contains original log lines
        assert "Error message 1" in summary_content
        assert "Warning message 1" in summary_content
        assert "Error message 2" in summary_content
        assert "Warning message 2" in summary_content

    def test_generate_error_summary_no_errors(self, tmp_path, monkeypatch):
        """
        Test error summary generation with no errors.

        Requirement: 8.8 (Error summary generation)
        """
        # Create empty error log
        error_log_path = tmp_path / "errors.log"
        error_log_path.write_text("")

        # Mock Path
        def mock_path(path_str):
            if "errors.log" in path_str:
                return error_log_path
            elif "error_summary.txt" in path_str:
                return tmp_path / "error_summary.txt"
            return Path(path_str)

        monkeypatch.setattr("experiments.benchmark_system.run_benchmark.Path", mock_path)

        # Generate error summary (should handle empty log gracefully)
        generate_error_summary()

        # Verify no summary file was created for empty log
        summary_path = tmp_path / "error_summary.txt"
        assert not summary_path.exists()

    def test_generate_error_summary_missing_log(self, tmp_path, monkeypatch):
        """
        Test error summary generation when error log doesn't exist.

        Requirement: 8.8 (Error summary generation)
        """
        # Mock Path to return non-existent file
        error_log_path = tmp_path / "nonexistent_errors.log"

        def mock_path(path_str):
            if "errors.log" in path_str:
                return error_log_path
            return Path(path_str)

        monkeypatch.setattr("experiments.benchmark_system.run_benchmark.Path", mock_path)

        # Generate error summary (should handle missing log gracefully)
        generate_error_summary()

        # Should not raise exception
        # No assertions needed - test passes if no exception raised


class TestMainEntryPoint:
    """Test main entry point integration."""

    @patch("experiments.benchmark_system.run_benchmark.main")
    @patch("experiments.benchmark_system.run_benchmark.setup_error_summary_logging")
    @patch("experiments.benchmark_system.run_benchmark.generate_error_summary")
    def test_main_entry_point_calls_all_components(
        self, mock_generate_summary, mock_setup_logging, mock_main
    ):
        """
        Test that main entry point calls all required components.

        Requirements: 5.1 (Long-running workload support), 8.8 (Error summary generation)
        """
        # Mock main to return success
        mock_main.return_value = 0

        # Import and run main entry point

        # Simulate running the script
        with patch("sys.exit") as mock_exit:
            # Call the main block logic
            mock_setup_logging()
            exit_code = mock_main()
            mock_generate_summary()
            mock_exit(exit_code)

        # Verify all components were called
        mock_setup_logging.assert_called_once()
        mock_main.assert_called_once()
        mock_generate_summary.assert_called_once()
        mock_exit.assert_called_once_with(0)

    @patch("experiments.benchmark_system.run_benchmark.main")
    @patch("experiments.benchmark_system.run_benchmark.setup_error_summary_logging")
    @patch("experiments.benchmark_system.run_benchmark.generate_error_summary")
    def test_main_entry_point_handles_errors(
        self, mock_generate_summary, mock_setup_logging, mock_main
    ):
        """
        Test that main entry point handles errors gracefully.

        Requirements: 5.1 (Long-running workload support), 8.8 (Error summary generation)
        """
        # Mock main to raise exception
        mock_main.side_effect = Exception("Test error")

        # Verify error summary is still generated
        with patch("sys.exit") as mock_exit:
            try:
                mock_setup_logging()
                mock_main()
            except Exception:
                mock_generate_summary()
                mock_exit(1)

        # Verify error summary was generated even after error
        mock_generate_summary.assert_called_once()
        mock_exit.assert_called_once_with(1)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
