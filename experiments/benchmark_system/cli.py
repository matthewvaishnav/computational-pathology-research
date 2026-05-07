#!/usr/bin/env python3
"""
Command-line interface for the Competitor Benchmark System.

This module provides a comprehensive CLI for running benchmark suites,
managing framework installations, and generating performance comparison reports.

Requirements: 6.5, 6.6, 6.7
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

from experiments.benchmark_system.models import BenchmarkConfig, TaskSpecification
from experiments.benchmark_system.orchestrator import BenchmarkOrchestrator

logger = logging.getLogger(__name__)


def create_parser() -> argparse.ArgumentParser:
    """
    Create command-line argument parser.
    
    Returns:
        ArgumentParser configured with all benchmark system options
        
    Requirements: 6.5 (CLI configuration), 6.6 (Mode selection), 6.7 (Framework selection)
    """
    parser = argparse.ArgumentParser(
        description="Competitor Benchmark System - Fair performance comparisons for computational pathology frameworks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run quick mode benchmark (3-4 hours)
  python -m experiments.benchmark_system.cli run --mode quick

  # Run full mode benchmark (20-40+ hours)
  python -m experiments.benchmark_system.cli run --mode full

  # Benchmark specific frameworks only
  python -m experiments.benchmark_system.cli run --mode quick --frameworks HistoCore PathML

  # Resume from checkpoint
  python -m experiments.benchmark_system.cli run --mode full --resume checkpoints/benchmark_20240101_120000.json

  # Use custom configuration file
  python -m experiments.benchmark_system.cli run --config configs/custom_benchmark.yaml

  # Validate setup before running benchmarks
  python -m experiments.benchmark_system.cli validate

  # List available frameworks
  python -m experiments.benchmark_system.cli list-frameworks
        """,
    )

    # Global options
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging (DEBUG level)"
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true", help="Suppress non-error output (ERROR level only)"
    )
    parser.add_argument("--log-file", type=Path, help="Write logs to file")

    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser(
        "run", help="Run benchmark suite (quick or full mode)"
    )
    run_parser.add_argument(
        "--mode",
        "-m",
        type=str,
        choices=["quick", "full"],
        default="quick",
        help="Benchmark mode: 'quick' (3-4 hours) or 'full' (20-40+ hours) (default: quick)",
    )
    run_parser.add_argument(
        "--frameworks",
        "-f",
        nargs="+",
        type=str,
        choices=["HistoCore", "PathML", "CLAM", "PyTorch"],
        help="Frameworks to benchmark (default: all frameworks)",
    )
    run_parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=Path("results/competitor_benchmarks"),
        help="Output directory for benchmark results (default: results/competitor_benchmarks)",
    )
    run_parser.add_argument(
        "--config",
        "-c",
        type=Path,
        help="Custom configuration file (YAML or JSON)",
    )
    run_parser.add_argument(
        "--resume",
        "-r",
        type=Path,
        help="Resume from checkpoint file",
    )
    run_parser.add_argument(
        "--quick-epochs",
        type=int,
        default=3,
        help="Number of epochs for quick mode (default: 3)",
    )
    run_parser.add_argument(
        "--quick-samples",
        type=int,
        default=1000,
        help="Number of samples for quick mode (default: 1000)",
    )
    run_parser.add_argument(
        "--timeout-hours",
        type=float,
        default=48.0,
        help="Timeout for individual framework execution in hours (default: 48.0)",
    )
    run_parser.add_argument(
        "--checkpoint-interval",
        type=int,
        default=30,
        help="Checkpoint interval in minutes (default: 30)",
    )
    run_parser.add_argument(
        "--random-seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    run_parser.add_argument(
        "--dataset",
        type=str,
        default="PatchCamelyon",
        help="Dataset name to use for benchmarking (default: PatchCamelyon)",
    )
    run_parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/pcam_real"),
        help="Root directory for dataset (default: data/pcam_real)",
    )
    run_parser.add_argument(
        "--model-architecture",
        type=str,
        default="resnet18_transformer",
        help="Model architecture to use (default: resnet18_transformer)",
    )
    run_parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training (default: 32)",
    )
    run_parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4)",
    )
    run_parser.add_argument(
        "--num-epochs",
        type=int,
        default=10,
        help="Number of epochs for full mode (default: 10)",
    )

    # Validate command
    validate_parser = subparsers.add_parser(
        "validate", help="Validate setup and hardware availability"
    )
    validate_parser.add_argument(
        "--output", type=Path, help="Save validation results to JSON file"
    )

    # List frameworks command
    list_parser = subparsers.add_parser(
        "list-frameworks", help="List available frameworks for benchmarking"
    )

    # Resume command
    resume_parser = subparsers.add_parser(
        "resume", help="Resume interrupted benchmark from checkpoint"
    )
    resume_parser.add_argument(
        "checkpoint",
        type=Path,
        help="Path to checkpoint file",
    )

    return parser


def load_config_from_file(config_path: Path) -> BenchmarkConfig:
    """
    Load benchmark configuration from YAML or JSON file.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        BenchmarkConfig loaded from file
        
    Raises:
        FileNotFoundError: If config file doesn't exist
        ValueError: If config file format is unsupported
        
    Requirement: 6.5 (Custom configuration file support)
    """
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")

    if config_path.suffix.lower() in [".yaml", ".yml"]:
        import yaml

        with open(config_path, "r") as f:
            config_dict = yaml.safe_load(f)
    elif config_path.suffix.lower() == ".json":
        with open(config_path, "r") as f:
            config_dict = json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")

    # Convert nested dicts to appropriate types
    if "task_spec" in config_dict and config_dict["task_spec"] is not None:
        task_spec_dict = config_dict["task_spec"]
        # Convert paths
        if "data_root" in task_spec_dict:
            task_spec_dict["data_root"] = Path(task_spec_dict["data_root"])
        config_dict["task_spec"] = TaskSpecification(**task_spec_dict)

    if "output_dir" in config_dict:
        config_dict["output_dir"] = Path(config_dict["output_dir"])

    return BenchmarkConfig(**config_dict)


def create_config_from_args(args) -> BenchmarkConfig:
    """
    Create benchmark configuration from command-line arguments.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        BenchmarkConfig constructed from CLI arguments
        
    Requirements: 6.5 (CLI configuration), 6.6 (Mode selection), 6.7 (Framework selection)
    """
    # Create task specification
    task_spec = TaskSpecification(
        dataset_name=args.dataset,
        data_root=args.data_root,
        model_architecture=args.model_architecture,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        random_seed=args.random_seed,
    )

    # Determine frameworks to benchmark
    frameworks = args.frameworks if args.frameworks else ["HistoCore", "PathML", "CLAM", "PyTorch"]

    # Create benchmark config
    config = BenchmarkConfig(
        mode=args.mode,
        frameworks=frameworks,
        task_spec=task_spec,
        quick_mode_epochs=args.quick_epochs,
        quick_mode_samples=args.quick_samples,
        timeout_hours=args.timeout_hours,
        checkpoint_interval_minutes=args.checkpoint_interval,
        output_dir=args.output_dir,
        random_seed=args.random_seed,
    )

    return config


def run_command(args) -> int:
    """
    Handle run command - execute benchmark suite.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, 1 for failure)
        
    Requirements: 6.5 (CLI execution), 6.6 (Mode selection), 6.7 (Framework selection)
    """
    try:
        # Load or create configuration
        if args.config:
            logger.info(f"Loading configuration from {args.config}")
            config = load_config_from_file(args.config)
            # Override with CLI arguments if provided
            if args.frameworks:
                config.frameworks = args.frameworks
            if args.output_dir != Path("results/competitor_benchmarks"):
                config.output_dir = args.output_dir
        else:
            logger.info("Creating configuration from command-line arguments")
            config = create_config_from_args(args)

        # Create output directory
        config.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Starting benchmark suite: mode={config.mode}, "
            f"frameworks={config.frameworks}, "
            f"output_dir={config.output_dir}"
        )

        # Initialize orchestrator
        orchestrator = BenchmarkOrchestrator(config=config)

        # Resume from checkpoint if requested
        if args.resume:
            logger.info(f"Resuming from checkpoint: {args.resume}")
            try:
                checkpoint_data = orchestrator.checkpoint_manager.load_checkpoint(args.resume)
                orchestrator.checkpoint_manager.resume_from_checkpoint(checkpoint_data)
                logger.info(f"Successfully resumed from checkpoint: {args.resume}")
            except Exception as e:
                logger.error(f"Failed to resume from checkpoint: {e}")
                sys.exit(1)

        # Run benchmark suite
        result = orchestrator.run_benchmark_suite()

        # Print summary
        print("\n" + "=" * 80)
        print("BENCHMARK SUITE COMPLETED")
        print("=" * 80)
        print(f"Duration: {result.total_duration_hours:.2f} hours")
        print(f"Successful frameworks: {len(result.successful_frameworks)}")
        print(f"Failed frameworks: {len(result.failed_frameworks)}")
        print(f"\nReport: {result.report_path}")
        print(f"Visualizations: {result.visualization_dir}")
        print("=" * 80)

        # Return success if at least one framework completed
        return 0 if result.successful_frameworks else 1

    except Exception as e:
        logger.error(f"Benchmark suite failed: {e}", exc_info=True)
        print(f"Error: {e}", file=sys.stderr)
        return 1


def validate_command(args) -> int:
    """
    Handle validate command - check setup and hardware.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, 1 for failure)
    """
    try:
        from experiments.benchmark_system.resource_manager import ResourceManager

        print("Validating benchmark system setup...")
        print("-" * 80)

        # Check GPU availability
        print("\n1. Checking GPU availability...")
        resource_manager = ResourceManager()
        gpu_info = resource_manager.verify_gpu_availability()

        if gpu_info.available:
            print(f"   ✅ GPU detected: {gpu_info.name}")
            print(f"   ✅ Memory: {gpu_info.memory_total_mb:.0f} MB")
            print(f"   ✅ CUDA available: {gpu_info.cuda_available}")
        else:
            print(f"   ❌ GPU not available: {gpu_info.error_message}")
            return 1

        # Check Python version
        print("\n2. Checking Python version...")
        import sys

        python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        print(f"   ✅ Python {python_version}")

        # Check PyTorch installation
        print("\n3. Checking PyTorch installation...")
        try:
            import torch

            print(f"   ✅ PyTorch {torch.__version__}")
            print(f"   ✅ CUDA version: {torch.version.cuda}")
        except ImportError:
            print("   ❌ PyTorch not installed")
            return 1

        # Check disk space
        print("\n4. Checking disk space...")
        import shutil

        stat = shutil.disk_usage(".")
        free_gb = stat.free / (1024**3)
        print(f"   ✅ Free disk space: {free_gb:.1f} GB")

        if free_gb < 50:
            print("   ⚠️  Warning: Less than 50 GB free disk space")

        print("\n" + "-" * 80)
        print("✅ Validation passed! System is ready for benchmarking.")
        print("-" * 80)

        # Save validation results if requested
        if args.output:
            validation_results = {
                "gpu_available": gpu_info.available,
                "gpu_name": gpu_info.name,
                "gpu_memory_mb": gpu_info.memory_total_mb,
                "cuda_available": gpu_info.cuda_available,
                "python_version": python_version,
                "pytorch_version": torch.__version__,
                "cuda_version": torch.version.cuda,
                "free_disk_gb": free_gb,
            }

            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, "w") as f:
                json.dump(validation_results, f, indent=2)
            print(f"\nValidation results saved to {args.output}")

        return 0

    except Exception as e:
        logger.error(f"Validation failed: {e}", exc_info=True)
        print(f"Error: {e}", file=sys.stderr)
        return 1


def list_frameworks_command(args) -> int:
    """
    Handle list-frameworks command - show available frameworks.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success)
        
    Requirement: 6.7 (Framework selection)
    """
    print("Available frameworks for benchmarking:")
    print("-" * 80)

    frameworks = [
        ("HistoCore", "Our computational pathology framework with transformer-based architecture"),
        ("PathML", "Comprehensive pathology ML library with preprocessing and analysis tools"),
        ("CLAM", "Clustering-constrained attention multiple instance learning framework"),
        ("PyTorch", "Baseline PyTorch implementation without framework-specific optimizations"),
    ]

    for name, description in frameworks:
        print(f"\n{name}")
        print(f"  {description}")

    print("\n" + "-" * 80)
    print("Use --frameworks to select specific frameworks for benchmarking.")
    print("Example: --frameworks HistoCore PathML")
    print("-" * 80)

    return 0


def resume_command(args) -> int:
    """
    Handle resume command - resume from checkpoint.
    
    Args:
        args: Parsed command-line arguments
        
    Returns:
        Exit code (0 for success, 1 for failure)
        
    Requirement: 6.5 (Checkpoint resume support)
    """
    try:
        if not args.checkpoint.exists():
            print(f"Error: Checkpoint file not found: {args.checkpoint}", file=sys.stderr)
            return 1

        logger.info(f"Resuming benchmark from checkpoint: {args.checkpoint}")

        # Load checkpoint
        from experiments.benchmark_system.checkpoint_manager import CheckpointManager

        checkpoint_manager = CheckpointManager(checkpoint_dir=args.checkpoint.parent)
        benchmark_state = checkpoint_manager.load_checkpoint(args.checkpoint)

        # Reconstruct configuration
        config_dict = benchmark_state["config"]
        if "output_dir" in config_dict:
            config_dict["output_dir"] = Path(config_dict["output_dir"])
        if "task_spec" in config_dict and config_dict["task_spec"] is not None:
            task_spec_dict = config_dict["task_spec"]
            if "data_root" in task_spec_dict:
                task_spec_dict["data_root"] = Path(task_spec_dict["data_root"])
            config_dict["task_spec"] = TaskSpecification(**task_spec_dict)

        config = BenchmarkConfig(**config_dict)

        # Initialize orchestrator
        orchestrator = BenchmarkOrchestrator(config=config)

        # Restore orchestrator state from checkpoint
        checkpoint_state = benchmark_state.get("state", {})
        if checkpoint_state:
            # Restore completed frameworks
            if "completed_frameworks" in checkpoint_state:
                orchestrator.completed_frameworks = set(checkpoint_state["completed_frameworks"])
                logger.info(f"Restored {len(orchestrator.completed_frameworks)} completed frameworks")
            
            # Restore failed frameworks
            if "failed_frameworks" in checkpoint_state:
                orchestrator.failed_frameworks = set(checkpoint_state["failed_frameworks"])
                logger.info(f"Restored {len(orchestrator.failed_frameworks)} failed frameworks")
            
            # Restore framework results
            if "framework_results" in checkpoint_state:
                orchestrator.framework_results = checkpoint_state["framework_results"]
                logger.info(f"Restored results for {len(orchestrator.framework_results)} frameworks")
            
            logger.info("Successfully restored orchestrator state from checkpoint")
        else:
            logger.warning("No state found in checkpoint, starting fresh")

        # Run benchmark suite
        result = orchestrator.run_benchmark_suite()

        # Print summary
        print("\n" + "=" * 80)
        print("BENCHMARK SUITE COMPLETED")
        print("=" * 80)
        print(f"Duration: {result.total_duration_hours:.2f} hours")
        print(f"Successful frameworks: {len(result.successful_frameworks)}")
        print(f"Failed frameworks: {len(result.failed_frameworks)}")
        print(f"\nReport: {result.report_path}")
        print(f"Visualizations: {result.visualization_dir}")
        print("=" * 80)

        return 0 if result.successful_frameworks else 1

    except Exception as e:
        logger.error(f"Resume failed: {e}", exc_info=True)
        print(f"Error: {e}", file=sys.stderr)
        return 1


def setup_logging(level: int, log_file: Optional[Path] = None) -> None:
    """
    Configure logging for the benchmark system.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR)
        log_file: Optional path to log file
    """
    # Create formatter
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # File handler (if requested)
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)  # Always log DEBUG to file
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)


def main() -> int:
    """
    Main CLI entry point.
    
    Returns:
        Exit code (0 for success, non-zero for failure)
        
    Requirements: 6.5 (CLI entry point), 6.6 (Mode selection), 6.7 (Framework selection)
    """
    parser = create_parser()
    args = parser.parse_args()

    # Setup logging
    if args.quiet:
        log_level = logging.ERROR
    elif args.verbose:
        log_level = logging.DEBUG
    else:
        log_level = logging.INFO

    setup_logging(
        level=log_level,
        log_file=args.log_file if hasattr(args, "log_file") else None,
    )

    # Handle commands
    if args.command == "run":
        return run_command(args)
    elif args.command == "validate":
        return validate_command(args)
    elif args.command == "list-frameworks":
        return list_frameworks_command(args)
    elif args.command == "resume":
        return resume_command(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
