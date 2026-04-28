"""
Real-Time WSI Streaming Demo

End-to-end demonstration of the real-time WSI streaming system showing:
- <30 second processing for gigapixel slides
- <2GB memory footprint
- Real-time confidence updates
- Progressive visualization
- Hospital demo-ready workflow

Author: Matthew Vaishnav
Date: 2026-04-28
"""

import asyncio
import sys
from pathlib import Path
import logging
import argparse

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.streaming import (
    RealTimeWSIProcessor,
    StreamingConfig,
    process_wsi_realtime
)


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


async def demo_basic_processing(wsi_path: str):
    """
    Demo 1: Basic real-time processing with default configuration.
    
    Shows the simplest way to process a WSI file in real-time.
    """
    print("\n" + "="*80)
    print("DEMO 1: Basic Real-Time Processing")
    print("="*80)
    
    # Use default configuration
    config = StreamingConfig(
        tile_size=1024,
        batch_size=64,
        memory_budget_gb=2.0,
        target_time=30.0,
        confidence_threshold=0.95
    )
    
    print(f"\nProcessing: {wsi_path}")
    print(f"Configuration:")
    print(f"  - Target time: {config.target_time}s")
    print(f"  - Memory budget: {config.memory_budget_gb}GB")
    print(f"  - Confidence threshold: {config.confidence_threshold}")
    print(f"  - Batch size: {config.batch_size}")
    
    # Create processor
    processor = RealTimeWSIProcessor(config)
    
    # Process WSI
    result = await processor.process_wsi_realtime(wsi_path)
    
    # Display results
    print(f"\n{'='*80}")
    print("RESULTS:")
    print(f"{'='*80}")
    print(f"Prediction: {result.prediction}")
    print(f"Confidence: {result.confidence:.3f}")
    print(f"Processing time: {result.processing_time:.2f}s")
    print(f"Patches processed: {result.patches_processed:,}")
    print(f"Throughput: {result.throughput_patches_per_sec:.1f} patches/sec")
    print(f"Peak memory: {result.peak_memory_gb:.2f}GB")
    print(f"Average memory: {result.avg_memory_gb:.2f}GB")
    print(f"Early stopped: {result.early_stopped}")
    
    # Performance summary
    perf_summary = processor.get_performance_summary(result)
    print(f"\n{'='*80}")
    print("PERFORMANCE REQUIREMENTS:")
    print(f"{'='*80}")
    print(f"✓ Time requirement (<30s): {perf_summary['time_requirement_met']}")
    print(f"✓ Memory requirement (<2GB): {perf_summary['memory_requirement_met']}")
    print(f"✓ Confidence requirement (>80%): {perf_summary['confidence_requirement_met']}")
    print(f"✓ All requirements met: {perf_summary['all_requirements_met']}")
    
    return result


async def demo_convenience_function(wsi_path: str):
    """
    Demo 2: Using the convenience function for quick processing.
    
    Shows the simplest API for one-line processing.
    """
    print("\n" + "="*80)
    print("DEMO 2: Convenience Function")
    print("="*80)
    
    print(f"\nProcessing with default config: {wsi_path}")
    
    # One-line processing
    result = await process_wsi_realtime(wsi_path)
    
    print(f"\nResult: Prediction={result.prediction}, Confidence={result.confidence:.3f}")
    print(f"Time: {result.processing_time:.2f}s, Memory: {result.peak_memory_gb:.2f}GB")
    
    return result


async def demo_custom_configuration(wsi_path: str):
    """
    Demo 3: Custom configuration for specific requirements.
    
    Shows how to customize processing parameters.
    """
    print("\n" + "="*80)
    print("DEMO 3: Custom Configuration")
    print("="*80)
    
    # Custom configuration for faster processing
    config = StreamingConfig(
        tile_size=512,  # Smaller tiles for faster loading
        batch_size=128,  # Larger batches for higher throughput
        memory_budget_gb=4.0,  # More memory available
        target_time=20.0,  # Faster target
        confidence_threshold=0.90,  # Lower threshold for early stopping
        enable_fp16=True,  # FP16 for memory savings
        enable_advanced_memory_optimization=True,
        enable_visualization=True
    )
    
    print(f"\nCustom configuration:")
    print(f"  - Tile size: {config.tile_size}")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - Memory budget: {config.memory_budget_gb}GB")
    print(f"  - Target time: {config.target_time}s")
    print(f"  - FP16 enabled: {config.enable_fp16}")
    
    processor = RealTimeWSIProcessor(config)
    result = await processor.process_wsi_realtime(wsi_path)
    
    print(f"\nResult: {result.processing_time:.2f}s, {result.confidence:.3f} confidence")
    
    return result


async def demo_batch_processing(wsi_paths: list):
    """
    Demo 4: Concurrent batch processing of multiple slides.
    
    Shows how to process multiple slides concurrently.
    """
    print("\n" + "="*80)
    print("DEMO 4: Batch Processing")
    print("="*80)
    
    print(f"\nProcessing {len(wsi_paths)} slides concurrently...")
    
    config = StreamingConfig(
        batch_size=64,
        memory_budget_gb=2.0,
        target_time=30.0
    )
    
    processor = RealTimeWSIProcessor(config)
    
    # Process batch with max 4 concurrent
    results = await processor.process_batch_realtime(
        wsi_paths=wsi_paths,
        max_concurrent=4
    )
    
    print(f"\nProcessed {len(results)} slides:")
    for slide_id, result in results.items():
        print(f"  - {slide_id}: {result.processing_time:.1f}s, "
              f"confidence={result.confidence:.3f}, "
              f"prediction={result.prediction}")
    
    return results


async def demo_memory_constrained(wsi_path: str):
    """
    Demo 5: Processing with strict memory constraints.
    
    Shows how the system adapts to limited memory.
    """
    print("\n" + "="*80)
    print("DEMO 5: Memory-Constrained Processing")
    print("="*80)
    
    # Very strict memory limit
    config = StreamingConfig(
        batch_size=32,  # Smaller batches
        memory_budget_gb=1.0,  # Only 1GB
        target_time=45.0,  # Allow more time
        enable_fp16=True,  # Use FP16 for memory savings
        enable_advanced_memory_optimization=True
    )
    
    print(f"\nStrict memory constraint: {config.memory_budget_gb}GB")
    print(f"Batch size: {config.batch_size}")
    print(f"FP16 enabled: {config.enable_fp16}")
    
    processor = RealTimeWSIProcessor(config)
    result = await processor.process_wsi_realtime(wsi_path)
    
    print(f"\nResult:")
    print(f"  - Peak memory: {result.peak_memory_gb:.2f}GB (limit: {config.memory_budget_gb}GB)")
    print(f"  - Processing time: {result.processing_time:.2f}s")
    print(f"  - Memory requirement met: {result.peak_memory_gb <= config.memory_budget_gb}")
    
    return result


async def demo_confidence_tracking(wsi_path: str):
    """
    Demo 6: Tracking confidence progression over time.
    
    Shows how confidence builds progressively during streaming.
    """
    print("\n" + "="*80)
    print("DEMO 6: Confidence Progression Tracking")
    print("="*80)
    
    config = StreamingConfig(
        confidence_threshold=0.95,
        enable_visualization=True
    )
    
    processor = RealTimeWSIProcessor(config)
    result = await processor.process_wsi_realtime(wsi_path)
    
    print(f"\nConfidence progression:")
    print(f"  - Initial: {result.confidence_history[0]:.3f}")
    print(f"  - Final: {result.confidence_history[-1]:.3f}")
    print(f"  - Early stopped: {result.early_stopped}")
    
    # Show confidence at key milestones
    milestones = [0.25, 0.5, 0.75, 1.0]
    for milestone in milestones:
        idx = int(len(result.confidence_history) * milestone) - 1
        if idx >= 0 and idx < len(result.confidence_history):
            print(f"  - At {milestone*100:.0f}% patches: {result.confidence_history[idx]:.3f}")
    
    return result


def create_synthetic_wsi(output_path: str, size: tuple = (10000, 10000)):
    """
    Create a synthetic WSI file for testing.
    
    Args:
        output_path: Path to save synthetic WSI
        size: Dimensions (width, height)
    """
    import numpy as np
    from PIL import Image
    
    print(f"\nCreating synthetic WSI: {output_path}")
    print(f"Dimensions: {size[0]}x{size[1]}")
    
    # Create random image
    img_array = np.random.randint(0, 256, (size[1], size[0], 3), dtype=np.uint8)
    img = Image.fromarray(img_array)
    
    # Save as TIFF
    img.save(output_path, format='TIFF')
    print(f"Synthetic WSI created: {output_path}")


async def run_all_demos(wsi_path: str):
    """Run all demo scenarios."""
    print("\n" + "="*80)
    print("REAL-TIME WSI STREAMING - COMPLETE DEMO SUITE")
    print("="*80)
    print(f"\nWSI File: {wsi_path}")
    
    # Demo 1: Basic processing
    await demo_basic_processing(wsi_path)
    
    # Demo 2: Convenience function
    await demo_convenience_function(wsi_path)
    
    # Demo 3: Custom configuration
    await demo_custom_configuration(wsi_path)
    
    # Demo 5: Memory constrained
    await demo_memory_constrained(wsi_path)
    
    # Demo 6: Confidence tracking
    await demo_confidence_tracking(wsi_path)
    
    # Demo 4: Batch processing (if multiple files available)
    # Uncomment if you have multiple WSI files:
    # await demo_batch_processing([wsi_path, wsi_path])
    
    print("\n" + "="*80)
    print("ALL DEMOS COMPLETE")
    print("="*80)


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Real-Time WSI Streaming Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run all demos with a WSI file
  python streaming_demo.py --wsi slide.svs
  
  # Run specific demo
  python streaming_demo.py --wsi slide.svs --demo basic
  
  # Create synthetic WSI for testing
  python streaming_demo.py --create-synthetic synthetic.tiff
  
  # Run with synthetic WSI
  python streaming_demo.py --create-synthetic synthetic.tiff --wsi synthetic.tiff
        """
    )
    
    parser.add_argument(
        '--wsi',
        type=str,
        help='Path to WSI file (.svs, .tiff, .ndpi, DICOM)'
    )
    
    parser.add_argument(
        '--demo',
        type=str,
        choices=['all', 'basic', 'convenience', 'custom', 'batch', 'memory', 'confidence'],
        default='all',
        help='Which demo to run (default: all)'
    )
    
    parser.add_argument(
        '--create-synthetic',
        type=str,
        help='Create a synthetic WSI file for testing'
    )
    
    parser.add_argument(
        '--synthetic-size',
        type=int,
        nargs=2,
        default=[10000, 10000],
        metavar=('WIDTH', 'HEIGHT'),
        help='Size of synthetic WSI (default: 10000 10000)'
    )
    
    args = parser.parse_args()
    
    # Create synthetic WSI if requested
    if args.create_synthetic:
        create_synthetic_wsi(
            args.create_synthetic,
            size=tuple(args.synthetic_size)
        )
        if not args.wsi:
            print("\nSynthetic WSI created. Use --wsi to process it.")
            return
    
    # Validate WSI path
    if not args.wsi:
        parser.print_help()
        print("\nError: --wsi is required")
        sys.exit(1)
    
    if not Path(args.wsi).exists():
        print(f"Error: WSI file not found: {args.wsi}")
        sys.exit(1)
    
    # Run selected demo
    if args.demo == 'all':
        asyncio.run(run_all_demos(args.wsi))
    elif args.demo == 'basic':
        asyncio.run(demo_basic_processing(args.wsi))
    elif args.demo == 'convenience':
        asyncio.run(demo_convenience_function(args.wsi))
    elif args.demo == 'custom':
        asyncio.run(demo_custom_configuration(args.wsi))
    elif args.demo == 'batch':
        asyncio.run(demo_batch_processing([args.wsi]))
    elif args.demo == 'memory':
        asyncio.run(demo_memory_constrained(args.wsi))
    elif args.demo == 'confidence':
        asyncio.run(demo_confidence_tracking(args.wsi))


if __name__ == '__main__':
    main()
