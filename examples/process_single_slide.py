"""
Example: Process a single WSI slide through the complete pipeline.

This script demonstrates:
1. Loading a WSI file
2. Configuring the processing pipeline
3. Processing the slide to extract features
4. Saving features to HDF5 cache
5. Loading and inspecting the results
"""

import sys
import time
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.wsi_pipeline import (
    BatchProcessor,
    ProcessingConfig,
    FeatureCache,
    WSIReader,
)


def process_single_slide(
    wsi_path: str,
    output_dir: str = "data/processed",
    config_path: str = None
):
    """
    Process a single WSI slide through the complete pipeline.
    
    Args:
        wsi_path: Path to the WSI file (.svs, .tiff, .ndpi, or DICOM)
        output_dir: Directory to save processed features
        config_path: Optional path to YAML configuration file
    """
    print("=" * 80)
    print("WSI Processing Pipeline - Single Slide Example")
    print("=" * 80)
    print()
    
    # Step 1: Load or create configuration
    print("Step 1: Loading configuration...")
    if config_path and Path(config_path).exists():
        config = ProcessingConfig.from_yaml(config_path)
        print(f"  ✓ Loaded configuration from: {config_path}")
    else:
        # Create default configuration
        config = ProcessingConfig(
            patch_size=256,
            stride=256,
            level=0,
            tissue_method="otsu",
            tissue_threshold=0.5,
            encoder_name="resnet50",
            encoder_pretrained=True,
            batch_size=32,
            cache_dir="features",
            compression="gzip",
            num_workers=1,
            max_retries=3,
            blur_threshold=100.0,
            min_tissue_coverage=0.1,
        )
        print("  ✓ Created default configuration")
    
    print(f"  - Patch size: {config.patch_size}x{config.patch_size}")
    print(f"  - Tissue threshold: {config.tissue_threshold}")
    print(f"  - Encoder: {config.encoder_name}")
    print(f"  - Batch size: {config.batch_size}")
    print()
    
    # Step 2: Inspect the WSI file
    print("Step 2: Inspecting WSI file...")
    wsi_path = Path(wsi_path)
    if not wsi_path.exists():
        print(f"  ✗ Error: WSI file not found: {wsi_path}")
        return
    
    try:
        with WSIReader(wsi_path) as reader:
            dimensions = reader.dimensions
            level_count = reader.level_count
            magnification = reader.get_magnification()
            mpp = reader.get_mpp()
            scanner_info = reader.get_scanner_info()
            
            print(f"  ✓ Successfully opened: {wsi_path.name}")
            print(f"  - Dimensions: {dimensions[0]} x {dimensions[1]} pixels")
            print(f"  - Pyramid levels: {level_count}")
            print(f"  - Magnification: {magnification}x" if magnification else "  - Magnification: Not available")
            print(f"  - MPP: {mpp}" if mpp else "  - MPP: Not available")
            print(f"  - Scanner: {scanner_info.get('model', 'Unknown')}")
            print(f"  - Scan date: {scanner_info.get('date', 'Unknown')}")
    except Exception as e:
        print(f"  ✗ Error opening WSI file: {e}")
        return
    print()
    
    # Step 3: Initialize processor
    print("Step 3: Initializing batch processor...")
    try:
        processor = BatchProcessor(
            config=config,
            num_workers=1,  # Single worker for single slide
            gpu_ids=None,   # Auto-detect GPU
        )
        print("  ✓ Batch processor initialized")
    except Exception as e:
        print(f"  ✗ Error initializing processor: {e}")
        return
    print()
    
    # Step 4: Process the slide
    print("Step 4: Processing slide...")
    print("  This may take several minutes depending on slide size...")
    start_time = time.time()
    
    try:
        result = processor.process_slide(
            wsi_path=str(wsi_path),
            output_dir=output_dir
        )
        
        processing_time = time.time() - start_time
        
        if result.success:
            print(f"  ✓ Processing completed successfully!")
            print(f"  - Processing time: {result.processing_time:.2f}s")
            print(f"  - Number of patches: {result.num_patches}")
            print(f"  - Feature file: {result.feature_file}")
            
            if result.qc_metrics:
                print(f"  - Quality metrics:")
                for key, value in result.qc_metrics.items():
                    print(f"    • {key}: {value}")
        else:
            print(f"  ✗ Processing failed: {result.error_message}")
            return
            
    except Exception as e:
        print(f"  ✗ Error during processing: {e}")
        return
    print()
    
    # Step 5: Load and inspect the cached features
    print("Step 5: Loading and inspecting cached features...")
    try:
        cache = FeatureCache(cache_dir=config.cache_dir)
        slide_id = wsi_path.stem
        
        # Check if features exist
        if not cache.exists(slide_id):
            print(f"  ✗ Cached features not found for slide: {slide_id}")
            return
        
        # Validate HDF5 file
        if not cache.validate(slide_id):
            print(f"  ✗ HDF5 file validation failed for slide: {slide_id}")
            return
        
        # Load features
        data = cache.load_features(slide_id)
        features = data["features"]
        coordinates = data["coordinates"]
        metadata = data["metadata"]
        
        print(f"  ✓ Successfully loaded cached features")
        print(f"  - Features shape: {features.shape}")
        print(f"  - Coordinates shape: {coordinates.shape}")
        print(f"  - Feature dimension: {features.shape[1]}")
        print(f"  - Number of patches: {features.shape[0]}")
        print()
        print(f"  Metadata:")
        for key, value in metadata.items():
            print(f"    • {key}: {value}")
            
    except Exception as e:
        print(f"  ✗ Error loading cached features: {e}")
        return
    print()
    
    # Step 6: Summary
    print("=" * 80)
    print("Processing Summary")
    print("=" * 80)
    print(f"Input slide: {wsi_path}")
    print(f"Output directory: {output_dir}")
    print(f"Feature file: {result.feature_file}")
    print(f"Number of patches: {result.num_patches}")
    print(f"Processing time: {result.processing_time:.2f}s")
    print(f"Patches per second: {result.num_patches / result.processing_time:.2f}")
    print()
    print("✓ Single slide processing completed successfully!")
    print("=" * 80)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process a single WSI slide through the complete pipeline"
    )
    parser.add_argument(
        "wsi_path",
        type=str,
        help="Path to WSI file (.svs, .tiff, .ndpi, or DICOM)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/processed",
        help="Directory to save processed features (default: data/processed)"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to YAML configuration file (optional)"
    )
    
    args = parser.parse_args()
    
    process_single_slide(
        wsi_path=args.wsi_path,
        output_dir=args.output_dir,
        config_path=args.config
    )


if __name__ == "__main__":
    main()
