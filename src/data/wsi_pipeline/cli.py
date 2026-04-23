#!/usr/bin/env python3
"""
Command-line interface for WSI processing pipeline.

This module provides a comprehensive CLI for processing WSI files,
running benchmarks, and managing the pipeline configuration.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import List, Optional

from .batch_processor import BatchProcessor
from .benchmarks import run_performance_benchmarks
from .config import ProcessingConfig
from .config_validator import ConfigValidator, get_recommended_config
from .logging_utils import setup_logging
from .validation import run_comprehensive_validation


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="WSI Processing Pipeline - Process whole slide images for computational pathology",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process a single slide
  python -m data.wsi_pipeline.cli process slide.svs --output-dir ./features

  # Process multiple slides with custom config
  python -m data.wsi_pipeline.cli process *.svs --config config.yaml --num-workers 8

  # Run performance benchmarks
  python -m data.wsi_pipeline.cli benchmark --quick

  # Validate pipeline installation
  python -m data.wsi_pipeline.cli validate

  # Generate configuration documentation
  python -m data.wsi_pipeline.cli config --generate-docs
        """
    )
    
    # Global options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Suppress non-error output"
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        help="Write logs to file"
    )
    
    # Subcommands
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Process command
    process_parser = subparsers.add_parser(
        "process",
        help="Process WSI files to extract features"
    )
    process_parser.add_argument(
        "input_files",
        nargs="+",
        type=Path,
        help="WSI files to process (supports wildcards)"
    )
    process_parser.add_argument(
        "--output-dir", "-o",
        type=Path,
        default="./wsi_features",
        help="Output directory for processed features (default: ./wsi_features)"
    )
    process_parser.add_argument(
        "--config", "-c",
        type=Path,
        help="Configuration file (YAML or JSON)"
    )
    process_parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of worker processes (default: 4)"
    )
    process_parser.add_argument(
        "--gpu-ids",
        nargs="*",
        type=int,
        help="GPU IDs to use (default: auto-detect)"
    )
    process_parser.add_argument(
        "--batch-size",
        type=int,
        help="Batch size for feature extraction"
    )
    process_parser.add_argument(
        "--patch-size",
        type=int,
        help="Patch size in pixels"
    )
    process_parser.add_argument(
        "--encoder",
        choices=["resnet50", "densenet121", "efficientnet_b0"],
        help="Feature encoder to use"
    )
    process_parser.add_argument(
        "--tissue-threshold",
        type=float,
        help="Tissue detection threshold (0.0-1.0)"
    )
    process_parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files"
    )
    process_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be processed without actually processing"
    )
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser(
        "benchmark",
        help="Run performance benchmarks"
    )
    benchmark_parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick benchmarks with smaller test sizes"
    )
    benchmark_parser.add_argument(
        "--output",
        type=Path,
        help="Save benchmark results to JSON file"
    )
    benchmark_parser.add_argument(
        "--config",
        type=Path,
        help="Configuration file for benchmarking"
    )
    
    # Validate command
    validate_parser = subparsers.add_parser(
        "validate",
        help="Validate pipeline installation and functionality"
    )
    validate_parser.add_argument(
        "--output",
        type=Path,
        help="Save validation results to JSON file"
    )
    validate_parser.add_argument(
        "--config",
        type=Path,
        help="Configuration file for validation"
    )
    
    # Config command
    config_parser = subparsers.add_parser(
        "config",
        help="Configuration management utilities"
    )
    config_group = config_parser.add_mutually_exclusive_group(required=True)
    config_group.add_argument(
        "--generate-docs",
        action="store_true",
        help="Generate configuration documentation"
    )
    config_group.add_argument(
        "--create-template",
        type=str,
        choices=["general", "high_throughput", "high_quality", "memory_limited"],
        help="Create configuration template for specific use case"
    )
    config_group.add_argument(
        "--validate",
        type=Path,
        help="Validate configuration file"
    )
    config_parser.add_argument(
        "--output",
        type=Path,
        help="Output file for generated content"
    )
    config_parser.add_argument(
        "--hardware",
        choices=["auto", "gpu", "cpu", "high_memory", "low_memory"],
        default="auto",
        help="Hardware type for template generation"
    )
    
    return parser


def load_config_from_file(config_path: Path) -> ProcessingConfig:
    """Load configuration from YAML or JSON file."""
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    if config_path.suffix.lower() in ['.yaml', '.yml']:
        import yaml
        with open(config_path, 'r') as f:
            config_dict = yaml.safe_load(f)
    elif config_path.suffix.lower() == '.json':
        with open(config_path, 'r') as f:
            config_dict = json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {config_path.suffix}")
    
    return ProcessingConfig(**config_dict)


def create_config_from_args(args) -> ProcessingConfig:
    """Create configuration from command-line arguments."""
    config_kwargs = {}
    
    # Map CLI arguments to config parameters
    if hasattr(args, 'batch_size') and args.batch_size:
        config_kwargs['batch_size'] = args.batch_size
    if hasattr(args, 'patch_size') and args.patch_size:
        config_kwargs['patch_size'] = args.patch_size
    if hasattr(args, 'encoder') and args.encoder:
        config_kwargs['encoder_name'] = args.encoder
    if hasattr(args, 'tissue_threshold') and args.tissue_threshold:
        config_kwargs['tissue_threshold'] = args.tissue_threshold
    if hasattr(args, 'output_dir') and args.output_dir:
        config_kwargs['cache_dir'] = str(args.output_dir)
    
    return ProcessingConfig(**config_kwargs)


def process_command(args) -> int:
    """Handle process command."""
    try:
        # Load or create configuration
        if args.config:
            config = load_config_from_file(args.config)
            # Override with CLI arguments
            cli_config = create_config_from_args(args)
            # Merge configurations (CLI args take precedence)
            config_dict = config.to_dict()
            config_dict.update({k: v for k, v in cli_config.to_dict().items() if v is not None})
            config = ProcessingConfig(**config_dict)
        else:
            config = create_config_from_args(args)
        
        # Validate configuration
        validator = ConfigValidator()
        validator.validate_and_raise(config)
        
        # Expand input files (handle wildcards)
        input_files = []
        for pattern in args.input_files:
            if '*' in str(pattern) or '?' in str(pattern):
                # Handle wildcards
                matches = list(Path().glob(str(pattern)))
                input_files.extend(matches)
            else:
                input_files.append(pattern)
        
        # Filter existing files
        existing_files = [f for f in input_files if f.exists()]
        if not existing_files:
            print("Error: No valid input files found", file=sys.stderr)
            return 1
        
        if len(existing_files) != len(input_files):
            missing = set(input_files) - set(existing_files)
            print(f"Warning: {len(missing)} files not found: {missing}", file=sys.stderr)
        
        print(f"Processing {len(existing_files)} files...")
        
        if args.dry_run:
            print("Dry run - files that would be processed:")
            for f in existing_files:
                print(f"  {f}")
            return 0
        
        # Create output directory
        args.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize batch processor
        processor = BatchProcessor(
            config=config,
            num_workers=args.num_workers,
            gpu_ids=args.gpu_ids,
        )
        
        # Process files
        results = processor.process_batch(
            slide_paths=[str(f) for f in existing_files],
            overwrite=args.overwrite,
        )
        
        # Print summary
        successful = sum(1 for r in results if r.success)
        print(f"\nProcessing complete: {successful}/{len(results)} files successful")
        
        if successful < len(results):
            print("Failed files:")
            for result in results:
                if not result.success:
                    print(f"  {result.slide_path}: {result.error}")
            return 1
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def benchmark_command(args) -> int:
    """Handle benchmark command."""
    try:
        # Load configuration if provided
        config = None
        if args.config:
            config = load_config_from_file(args.config)
        
        print("Running performance benchmarks...")
        results = run_performance_benchmarks(
            config=config,
            quick_mode=args.quick,
        )
        
        # Save results if requested
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with open(args.output, 'w') as f:
                json.dump(results, f, indent=2, default=str)
            print(f"Benchmark results saved to {args.output}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def validate_command(args) -> int:
    """Handle validate command."""
    try:
        # Load configuration if provided
        config = None
        if args.config:
            config = load_config_from_file(args.config)
        
        print("Running comprehensive validation...")
        results = run_comprehensive_validation(
            config=config,
            output_path=args.output,
        )
        
        # Check if validation passed
        if results.get("overall_summary", {}).get("validation_passed", False):
            print("✅ Validation passed!")
            return 0
        else:
            print("❌ Validation failed!")
            return 1
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def config_command(args) -> int:
    """Handle config command."""
    try:
        validator = ConfigValidator()
        
        if args.generate_docs:
            # Generate documentation
            docs = validator.generate_documentation()
            
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                with open(args.output, 'w') as f:
                    f.write(docs)
                print(f"Configuration documentation saved to {args.output}")
            else:
                print(docs)
        
        elif args.create_template:
            # Create configuration template
            config = get_recommended_config(
                use_case=args.create_template,
                hardware=args.hardware,
            )
            
            config_dict = config.to_dict()
            
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                
                if args.output.suffix.lower() in ['.yaml', '.yml']:
                    import yaml
                    with open(args.output, 'w') as f:
                        yaml.dump(config_dict, f, default_flow_style=False, indent=2)
                else:
                    with open(args.output, 'w') as f:
                        json.dump(config_dict, f, indent=2)
                
                print(f"Configuration template saved to {args.output}")
            else:
                print(json.dumps(config_dict, indent=2))
        
        elif args.validate:
            # Validate configuration file
            config = load_config_from_file(args.validate)
            is_valid, errors = validator.validate_config(config)
            
            if is_valid:
                print(f"✅ Configuration {args.validate} is valid")
                return 0
            else:
                print(f"❌ Configuration {args.validate} is invalid:")
                for error in errors:
                    print(f"  - {error}")
                return 1
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1


def main() -> int:
    """Main CLI entry point."""
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
        log_file=args.log_file,
        use_colors=True,
        component_filter='data.wsi_pipeline',
    )
    
    # Handle commands
    if args.command == "process":
        return process_command(args)
    elif args.command == "benchmark":
        return benchmark_command(args)
    elif args.command == "validate":
        return validate_command(args)
    elif args.command == "config":
        return config_command(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())