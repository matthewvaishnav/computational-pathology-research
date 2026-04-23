"""Example usage of WSI pipeline data models and configuration.

This script demonstrates how to:
1. Create and validate ProcessingConfig
2. Load configuration from YAML
3. Create SlideMetadata and ProcessingResult objects
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.wsi_pipeline import (
    ProcessingConfig,
    SlideMetadata,
    ProcessingResult,
)


def example_config_from_dict():
    """Example: Create configuration from dictionary."""
    print("=" * 60)
    print("Example 1: Create configuration from dictionary")
    print("=" * 60)
    
    config_dict = {
        "patch_size": 512,
        "stride": 256,
        "tissue_threshold": 0.6,
        "encoder_name": "densenet121",
        "batch_size": 64,
        "num_workers": 8,
    }
    
    config = ProcessingConfig.from_dict(config_dict)
    print(f"Patch size: {config.patch_size}")
    print(f"Stride: {config.stride}")
    print(f"Tissue threshold: {config.tissue_threshold}")
    print(f"Encoder: {config.encoder_name}")
    print(f"Batch size: {config.batch_size}")
    print(f"Num workers: {config.num_workers}")
    print()


def example_config_from_yaml():
    """Example: Load configuration from YAML file."""
    print("=" * 60)
    print("Example 2: Load configuration from YAML")
    print("=" * 60)
    
    yaml_path = Path("examples/wsi_pipeline_config.yaml")
    
    if yaml_path.exists():
        config = ProcessingConfig.from_yaml(yaml_path)
        print(f"Loaded configuration from: {yaml_path}")
        print(f"Patch size: {config.patch_size}")
        print(f"Tissue method: {config.tissue_method}")
        print(f"Encoder: {config.encoder_name}")
        print(f"Cache dir: {config.cache_dir}")
        print(f"Num workers: {config.num_workers}")
    else:
        print(f"YAML file not found: {yaml_path}")
    print()


def example_config_validation():
    """Example: Configuration validation."""
    print("=" * 60)
    print("Example 3: Configuration validation")
    print("=" * 60)
    
    # Valid configuration
    try:
        config = ProcessingConfig(patch_size=256, num_workers=4)
        config.validate()
        print("✓ Valid configuration passed validation")
    except ValueError as e:
        print(f"✗ Validation failed: {e}")
    
    # Invalid configuration
    try:
        config = ProcessingConfig(patch_size=32, num_workers=0)
        config.validate()
        print("✓ Invalid configuration passed validation (unexpected!)")
    except ValueError as e:
        print(f"✓ Invalid configuration caught by validation:")
        print(f"  {e}")
    print()


def example_slide_metadata():
    """Example: Create SlideMetadata."""
    print("=" * 60)
    print("Example 4: Create SlideMetadata")
    print("=" * 60)
    
    metadata = SlideMetadata(
        slide_id="slide_001",
        patient_id="patient_001",
        file_path="/data/slides/slide_001.svs",
        label=1,
        split="train",
        width=100000,
        height=80000,
        magnification=40.0,
        mpp=0.25,
        scanner_model="Aperio ScanScope",
        scan_date="2024-01-15",
    )
    
    print(f"Slide ID: {metadata.slide_id}")
    print(f"Patient ID: {metadata.patient_id}")
    print(f"File path: {metadata.file_path}")
    print(f"Label: {metadata.label}")
    print(f"Split: {metadata.split}")
    print(f"Dimensions: {metadata.width} x {metadata.height}")
    print(f"Magnification: {metadata.magnification}x")
    print(f"MPP: {metadata.mpp}")
    print(f"Scanner: {metadata.scanner_model}")
    print(f"Scan date: {metadata.scan_date}")
    print()


def example_processing_result():
    """Example: Create ProcessingResult."""
    print("=" * 60)
    print("Example 5: Create ProcessingResult")
    print("=" * 60)
    
    # Successful result
    result = ProcessingResult(
        slide_id="slide_001",
        success=True,
        feature_file=Path("/data/features/slide_001.h5"),
        num_patches=1500,
        processing_time=120.5,
        qc_metrics={
            "blur_score": 150.0,
            "tissue_coverage": 0.75,
            "num_blurry_patches": 10,
        },
    )
    
    print(f"Slide ID: {result.slide_id}")
    print(f"Success: {result.success}")
    print(f"Feature file: {result.feature_file}")
    print(f"Num patches: {result.num_patches}")
    print(f"Processing time: {result.processing_time:.2f}s")
    print(f"QC metrics: {result.qc_metrics}")
    print()
    
    # Failed result
    failed_result = ProcessingResult(
        slide_id="slide_002",
        success=False,
        error_message="Failed to open slide: file corrupted",
    )
    
    print(f"Slide ID: {failed_result.slide_id}")
    print(f"Success: {failed_result.success}")
    print(f"Error: {failed_result.error_message}")
    print()


def main():
    """Run all examples."""
    print("\n")
    print("=" * 60)
    print("WSI Pipeline Configuration and Models Examples")
    print("=" * 60)
    print()
    
    example_config_from_dict()
    example_config_from_yaml()
    example_config_validation()
    example_slide_metadata()
    example_processing_result()
    
    print("=" * 60)
    print("All examples completed!")
    print("=" * 60)


if __name__ == "__main__":
    main()
