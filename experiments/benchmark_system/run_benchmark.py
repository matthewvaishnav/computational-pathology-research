#!/usr/bin/env python3
"""
Main entry point for the Competitor Benchmark System.

This script provides the primary interface for running benchmark suites,
wiring together all system components (FrameworkManager, TaskExecutor,
ResourceManager, MetricsCollector, CheckpointManager, ErrorHandler,
ReportGenerator, and BenchmarkOrchestrator).

Usage:
    # Quick mode (3-4 hours)
    python experiments/benchmark_system/run_benchmark.py run --mode quick

    # Full mode (20-40+ hours)
    python experiments/benchmark_system/run_benchmark.py run --mode full

    # Resume from checkpoint
    python experiments/benchmark_system/run_benchmark.py resume checkpoint.json

    # Validate setup
    python experiments/benchmark_system/run_benchmark.py validate

Requirements: 5.1 (Long-running workload support), 8.8 (Error summary generation)
"""

import logging
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from experiments.benchmark_system.cli import main

logger = logging.getLogger(__name__)


def setup_error_summary_logging() -> None:
    """
    Configure error summary logging for the benchmark system.
    
    Creates a dedicated error log file that captures all errors and warnings
    for post-execution analysis and error summary generation.
    
    Requirement: 8.8 (Error summary generation)
    """
    # Create error log handler
    error_log_path = Path("results/competitor_benchmarks/errors.log")
    error_log_path.parent.mkdir(parents=True, exist_ok=True)
    
    error_handler = logging.FileHandler(error_log_path, mode='w')
    error_handler.setLevel(logging.WARNING)  # Capture warnings and errors
    
    # Format for error logs
    error_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s\n"
        "  Location: %(pathname)s:%(lineno)d\n"
        "  Function: %(funcName)s\n",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    error_handler.setFormatter(error_formatter)
    
    # Add to root logger
    logging.getLogger().addHandler(error_handler)
    
    logger.info(f"Error summary logging configured: {error_log_path}")


def generate_error_summary() -> None:
    """
    Generate error summary report from error log.
    
    Parses the error log file and creates a human-readable summary of all
    errors and warnings encountered during benchmark execution.
    
    Requirement: 8.8 (Error summary generation)
    """
    error_log_path = Path("results/competitor_benchmarks/errors.log")
    error_summary_path = Path("results/competitor_benchmarks/error_summary.txt")
    
    if not error_log_path.exists():
        logger.info("No error log found, skipping error summary generation")
        return
    
    try:
        # Read error log
        with open(error_log_path, 'r') as f:
            error_lines = f.readlines()
        
        if not error_lines:
            logger.info("No errors or warnings logged")
            return
        
        # Count errors and warnings
        error_count = sum(1 for line in error_lines if " - ERROR - " in line)
        warning_count = sum(1 for line in error_lines if " - WARNING - " in line)
        
        # Generate summary
        summary_lines = [
            "=" * 80,
            "BENCHMARK SYSTEM ERROR SUMMARY",
            "=" * 80,
            "",
            f"Total Errors: {error_count}",
            f"Total Warnings: {warning_count}",
            "",
            "=" * 80,
            "DETAILED LOG",
            "=" * 80,
            "",
        ]
        
        summary_lines.extend(error_lines)
        
        # Write summary
        error_summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(error_summary_path, 'w') as f:
            f.write('\n'.join(summary_lines))
        
        logger.info(
            f"Error summary generated: {error_summary_path} "
            f"({error_count} errors, {warning_count} warnings)"
        )
        
        # Print summary to console
        print("\n" + "=" * 80)
        print("ERROR SUMMARY")
        print("=" * 80)
        print(f"Errors: {error_count}")
        print(f"Warnings: {warning_count}")
        print(f"Detailed log: {error_log_path}")
        print(f"Summary report: {error_summary_path}")
        print("=" * 80 + "\n")
        
    except Exception as e:
        logger.error(f"Failed to generate error summary: {e}", exc_info=True)


if __name__ == "__main__":
    try:
        # Setup error summary logging (Requirement 8.8)
        setup_error_summary_logging()
        
        # Run CLI main function (handles all command-line arguments and orchestration)
        exit_code = main()
        
        # Generate error summary after execution (Requirement 8.8)
        generate_error_summary()
        
        # Exit with appropriate code
        sys.exit(exit_code)
        
    except KeyboardInterrupt:
        logger.warning("Benchmark interrupted by user (Ctrl+C)")
        print("\nBenchmark interrupted by user", file=sys.stderr)
        generate_error_summary()
        sys.exit(130)  # Standard exit code for SIGINT
        
    except Exception as e:
        logger.critical(f"Unexpected error in main entry point: {e}", exc_info=True)
        print(f"\nCritical error: {e}", file=sys.stderr)
        generate_error_summary()
        sys.exit(1)
