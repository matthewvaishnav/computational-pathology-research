"""
Example: Process multiple WSI slides in batch with parallel execution.

This script demonstrates:
1. Loading multiple WSI files from a directory
2. Configuring batch processing with parallel workers
3. Processing slides with GPU acceleration
4. Generating a slide index for training
5. Analyzing batch processing results
"""

import sys
import time
from pathlib import Path
from typing import List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.wsi_pipeline import (
    BatchProcessor,
    ProcessingConfig,
    FeatureCache,
)


def process_batch_slides(
    input_dir: str,
    output_dir: str = "data/processed",
    config_path: str = None,
    num_workers: int = 4,
    gpu_ids: List[int] = None,
    slide_pattern: str = "*.svs",
):
    """
    Process multiple WSI slides in batch with parallel execution.
    
    Args:
        input_dir: Directory containing WSI files
        output_dir: Directory to save processed features
        config_path: Optional path to YAML configuration file
        num_workers: Number of parallel worker processes
        gpu_ids: List of GPU IDs to use (None = auto-detect all GPUs)
        slide_pattern: Glob pattern for slide files (e.g., "*.svs", "*.tiff")
    """
    print("=" * 80)
    print("WSI Processing Pipeline - Batch Processing Example")
    print("=" * 80)
    print()
    
    # Step 1: Find WSI files
    print("Step 1: Finding WSI files...")
    input_path = Path(input_dir)
    if not input_path.exists():
        print(f"  ✗ Error: Input directory not found: {input_dir}")
        return
    
    slide_paths = list(input_path.glob(slide_pattern))
    
    if not slide_paths:
        print(f"  ✗ No slides found matching pattern: {slide_pattern}")
        print(f"  Try different patterns: *.svs, *.tiff, *.ndpi, *.dcm")
        return
    
    print(f"  ✓ Found {len(slide_paths)} slides")
    for i, path in enumerate(slide_paths[:5], 1):
        print(f"    {i}. {path.name}")
    if len(slide_paths) > 5:
        print(f"    ... and {len(slide_paths) - 5} more")
    print()
    
    # Step 2: Load or create configuration
    print("Step 2: Loading configuration...")
    if config_path and Path(config_path).exists():
        config = ProcessingConfig.from_yaml(config_path)
        print(f"  ✓ Loaded configuration from: {config_path}")
    else:
        # Create default configuration optimized for batch processing
        config = ProcessingConfig(
            patch_size=256,
            stride=256,
            level=0,
            tissue_method="otsu",
            tissue_threshold=0.5,
            encoder_name="resnet50",
            encoder_pretrained=True,
            batch_size=64,  # Larger batch size for GPU efficiency
            cache_dir="features",
            compression="gzip",
            num_workers=num_workers,
            max_retries=3,
            blur_threshold=100.0,
            min_tissue_coverage=0.1,
        )
        print("  ✓ Created default configuration")
    
    print(f"  - Patch size: {config.patch_size}x{config.patch_size}")
    print(f"  - Tissue threshold: {config.tissue_threshold}")
    print(f"  - Encoder: {config.encoder_name}")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - Num workers: {num_workers}")
    print(f"  - GPU IDs: {gpu_ids if gpu_ids else 'Auto-detect'}")
    print()
    
    # Step 3: Initialize batch processor
    print("Step 3: Initializing batch processor...")
    try:
        processor = BatchProcessor(
            config=config,
            num_workers=num_workers,
            gpu_ids=gpu_ids,
        )
        print("  ✓ Batch processor initialized")
    except Exception as e:
        print(f"  ✗ Error initializing processor: {e}")
        return
    print()
    
    # Step 4: Process batch
    print("Step 4: Processing batch...")
    print(f"  Processing {len(slide_paths)} slides with {num_workers} workers...")
    print("  This may take a while depending on slide sizes and hardware...")
    print()
    
    start_time = time.time()
    
    try:
        results = processor.process_batch(
            wsi_paths=[str(p) for p in slide_paths],
            output_dir=output_dir
        )
        
        total_time = time.time() - start_time
        
        print()
        print("  ✓ Batch processing completed!")
        print()
        
    except Exception as e:
        print(f"  ✗ Error during batch processing: {e}")
        return
    
    # Step 5: Analyze results
    print("Step 5: Analyzing results...")
    print()
    print(f"  Summary:")
    print(f"  - Total slides: {results['num_total']}")
    print(f"  - Successfully processed: {results['num_success']}")
    print(f"  - Failed: {results['num_failed']}")
    print(f"  - Total time: {results['total_time']:.2f}s")
    print(f"  - Average time per slide: {results['avg_time_per_slide']:.2f}s")
    print()
    
    # Show successful slides
    if results['successful_slides']:
        print(f"  Successfully processed slides:")
        for slide_result in results['successful_slides'][:5]:
            print(f"    ✓ {slide_result.slide_id}")
            print(f"      - Patches: {slide_result.num_patches}")
            print(f"      - Time: {slide_result.processing_time:.2f}s")
            print(f"      - File: {slide_result.feature_file}")
        if len(results['successful_slides']) > 5:
            print(f"    ... and {len(results['successful_slides']) - 5} more")
        print()
    
    # Show failed slides
    if results['failed_slides']:
        print(f"  Failed slides:")
        for slide_result in results['failed_slides']:
            print(f"    ✗ {slide_result.slide_id}")
            print(f"      - Error: {slide_result.error_message}")
        print()
    
    # Step 6: Generate slide index
    print("Step 6: Generating slide index...")
    try:
        index_path = processor.generate_slide_index(
            output_dir=output_dir,
            split_ratios=(0.7, 0.15, 0.15)  # train, val, test
        )
        print(f"  ✓ Slide index generated: {index_path}")
        print(f"  - Compatible with CAMELYONSlideDataset")
        print(f"  - Split ratios: 70% train, 15% val, 15% test")
    except Exception as e:
        print(f"  ✗ Error generating slide index: {e}")
    print()
    
    # Step 7: Verify cached features
    print("Step 7: Verifying cached features...")
    cache = FeatureCache(cache_dir=config.cache_dir)
    
    verified_count = 0
    for slide_path in slide_paths:
        slide_id = slide_path.stem
        if cache.exists(slide_id) and cache.validate(slide_id):
            verified_count += 1
    
    print(f"  ✓ Verified {verified_count}/{len(slide_paths)} cached feature files")
    print()
    
    # Step 8: Summary
    print("=" * 80)
    print("Batch Processing Summary")
    print("=" * 80)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Slide pattern: {slide_pattern}")
    print(f"Total slides: {results['num_total']}")
    print(f"Successfully processed: {results['num_success']}")
    print(f"Failed: {results['num_failed']}")
    print(f"Total time: {results['total_time']:.2f}s")
    print(f"Average time per slide: {results['avg_time_per_slide']:.2f}s")
    print(f"Throughput: {results['num_success'] / results['total_time']:.2f} slides/second")
    print()
    
    if results['num_success'] > 0:
        print("✓ Batch processing completed successfully!")
        print()
        print("Next steps:")
        print("  1. Load features with CAMELYONSlideDataset")
        print("  2. Train models using train_camelyon.py")
        print("  3. Evaluate models using evaluate_camelyon.py")
    else:
        print("✗ No slides were successfully processed")
        print("  Check error messages above for details")
    
    print("=" * 80)


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Process multiple WSI slides in batch with parallel execution"
    )
    parser.add_argument(
        "input_dir",
        type=str,
        help="Directory containing WSI files"
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
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of parallel worker processes (default: 4)"
    )
    parser.add_argument(
        "--gpu-ids",
        type=int,
        nargs="+",
        default=None,
        help="GPU IDs to use (e.g., --gpu-ids 0 1 2)"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.svs",
        help="Glob pattern for slide files (default: *.svs)"
    )
    
    args = parser.parse_args()
    
    process_batch_slides(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        config_path=args.config,
        num_workers=args.num_workers,
        gpu_ids=args.gpu_ids,
        slide_pattern=args.pattern,
    )


if __name__ == "__main__":
    main()
