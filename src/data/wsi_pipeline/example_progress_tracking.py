"""
Example demonstrating WSI streaming with comprehensive progress tracking and ETA estimation.

This script shows how to use the enhanced WSIStreamReader with real-time progress tracking,
confidence-based early stopping, and progress callbacks for visualization.
"""

import time
import logging
from pathlib import Path
from typing import Dict, Any
import json

from .wsi_stream_reader import WSIStreamReader, ProgressCallback, StreamingProgress
from .tile_buffer_pool import TileBufferConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class ProgressVisualizer:
    """Simple progress visualizer for demonstration."""
    
    def __init__(self):
        self.progress_history = []
        self.last_update_time = 0
    
    def update_progress(self, progress: StreamingProgress) -> None:
        """
        Update progress visualization.
        
        Args:
            progress: Current progress information
        """
        current_time = time.time()
        
        # Store progress history
        self.progress_history.append({
            'timestamp': current_time,
            'progress_ratio': progress.progress_ratio,
            'confidence': progress.current_confidence,
            'eta_seconds': progress.estimated_time_remaining,
            'throughput': progress.throughput_tiles_per_second,
            'memory_gb': progress.memory_usage_gb,
        })
        
        # Print progress update (limit frequency)
        if current_time - self.last_update_time >= 2.0:  # Every 2 seconds
            self._print_progress_update(progress)
            self.last_update_time = current_time
    
    def _print_progress_update(self, progress: StreamingProgress) -> None:
        """Print formatted progress update."""
        print(f"\n{'='*60}")
        print(f"WSI Streaming Progress Update")
        print(f"{'='*60}")
        print(f"Progress: {progress.get_progress_percentage()} "
              f"({progress.tiles_processed}/{progress.total_tiles} tiles)")
        print(f"ETA: {progress.get_eta_string()}")
        print(f"Elapsed: {progress.elapsed_time:.1f}s")
        print(f"Stage: {progress.current_stage} ({progress.stage_progress:.1%})")
        print(f"Throughput: {progress.throughput_tiles_per_second:.1f} tiles/sec")
        print(f"Memory: {progress.memory_usage_gb:.2f}GB "
              f"(peak: {progress.peak_memory_usage_gb:.2f}GB)")
        print(f"Confidence: {progress.current_confidence:.3f} "
              f"(target: {progress.confidence_threshold:.3f})")
        
        if progress.early_stop_recommended:
            print(f"🎯 Early stopping recommended!")
        
        print(f"Quality: {progress.data_quality_score:.2%} "
              f"(failed: {progress.tiles_failed}, skipped: {progress.tiles_skipped})")
        print(f"{'='*60}")
    
    def save_progress_history(self, output_path: str) -> None:
        """Save progress history to JSON file."""
        with open(output_path, 'w') as f:
            json.dump(self.progress_history, f, indent=2)
        print(f"Progress history saved to {output_path}")


def simulate_confidence_updates(reader: WSIStreamReader, visualizer: ProgressVisualizer) -> None:
    """
    Simulate confidence updates during processing.
    
    Args:
        reader: WSI stream reader
        visualizer: Progress visualizer
    """
    # Simulate gradually increasing confidence
    confidence_values = [0.1, 0.3, 0.5, 0.7, 0.85, 0.92, 0.96, 0.98]
    
    for i, confidence in enumerate(confidence_values):
        time.sleep(1.0)  # Simulate processing time
        reader.update_confidence(confidence)
        
        # Get updated progress
        progress = reader.get_progress()
        visualizer.update_progress(progress)
        
        # Check for early stopping
        if progress.early_stop_recommended:
            print(f"\n🛑 Early stopping triggered at confidence {confidence:.3f}")
            break


def demonstrate_basic_progress_tracking():
    """Demonstrate basic progress tracking functionality."""
    print("\n" + "="*80)
    print("DEMONSTRATION: Basic Progress Tracking")
    print("="*80)
    
    # Create progress visualizer
    visualizer = ProgressVisualizer()
    
    # Create progress callback
    progress_callback = ProgressCallback(
        callback_func=visualizer.update_progress,
        update_interval=1.0,  # Update every second
        min_progress_delta=0.05  # Update on 5% progress change
    )
    
    # Configure WSI reader
    config = TileBufferConfig(
        max_memory_gb=2.0,
        tile_size=1024,
        adaptive_sizing_enabled=True,
        initial_buffer_size=16,
        max_buffer_size=64
    )
    
    # Note: This would normally use a real WSI file
    # For demonstration, we'll show the configuration
    print(f"Configuration:")
    print(f"  Memory limit: {config.max_memory_gb}GB")
    print(f"  Tile size: {config.tile_size}px")
    print(f"  Adaptive sizing: {config.adaptive_sizing_enabled}")
    print(f"  Buffer size: {config.initial_buffer_size}-{config.max_buffer_size}")
    
    # This would be the actual usage with a real WSI file:
    """
    try:
        reader = WSIStreamReader(
            wsi_path="path/to/slide.svs",
            config=config,
            progress_callbacks=[progress_callback]
        )
        
        # Initialize streaming
        metadata = reader.initialize_streaming(
            target_processing_time=30.0,
            confidence_threshold=0.95
        )
        
        print(f"\\nSlide metadata:")
        print(f"  Dimensions: {metadata.dimensions}")
        print(f"  Estimated patches: {metadata.estimated_patches}")
        print(f"  Target time: {metadata.target_processing_time}s")
        
        # Process tiles with progress tracking
        batch_count = 0
        for batch in reader.stream_tiles(batch_size=16):
            batch_count += 1
            
            # Simulate some processing time
            time.sleep(0.1)
            
            # Update confidence periodically
            if batch_count % 5 == 0:
                confidence = min(0.95, 0.1 + (batch_count * 0.05))
                reader.update_confidence(confidence)
            
            # Check for early stopping
            progress = reader.get_progress()
            if progress.early_stop_recommended:
                break
        
        # Get final statistics
        final_progress = reader.finish_streaming()
        performance_summary = reader.get_performance_summary()
        
        print(f"\\nFinal Results:")
        print(f"  Processing time: {final_progress.elapsed_time:.1f}s")
        print(f"  Tiles processed: {final_progress.tiles_processed}")
        print(f"  Final confidence: {final_progress.current_confidence:.3f}")
        print(f"  Throughput: {final_progress.throughput_tiles_per_second:.1f} tiles/sec")
        
        # Save progress history
        visualizer.save_progress_history("progress_history.json")
        
    except Exception as e:
        print(f"Error during processing: {e}")
    """
    
    print("\n(Note: This demonstration shows configuration only.")
    print("Replace with actual WSI file path for real processing.)")


def demonstrate_advanced_features():
    """Demonstrate advanced progress tracking features."""
    print("\n" + "="*80)
    print("DEMONSTRATION: Advanced Progress Tracking Features")
    print("="*80)
    
    # Multiple progress callbacks for different purposes
    class DetailedLogger:
        def log_progress(self, progress: StreamingProgress):
            logger.info(
                f"Progress: {progress.progress_ratio:.1%}, "
                f"ETA: {progress.estimated_time_remaining:.1f}s, "
                f"Confidence: {progress.current_confidence:.3f}"
            )
    
    class PerformanceMonitor:
        def monitor_performance(self, progress: StreamingProgress):
            if progress.memory_usage_gb > 1.5:  # Alert on high memory usage
                logger.warning(f"High memory usage: {progress.memory_usage_gb:.2f}GB")
            
            if progress.throughput_tiles_per_second < 10:  # Alert on low throughput
                logger.warning(f"Low throughput: {progress.throughput_tiles_per_second:.1f} tiles/sec")
    
    detailed_logger = DetailedLogger()
    performance_monitor = PerformanceMonitor()
    
    # Create multiple callbacks
    callbacks = [
        ProgressCallback(
            callback_func=detailed_logger.log_progress,
            update_interval=2.0,
            min_progress_delta=0.1
        ),
        ProgressCallback(
            callback_func=performance_monitor.monitor_performance,
            update_interval=5.0,
            min_progress_delta=0.05
        )
    ]
    
    print("Advanced features demonstrated:")
    print("  ✓ Multiple progress callbacks")
    print("  ✓ Performance monitoring")
    print("  ✓ Memory usage alerts")
    print("  ✓ Throughput monitoring")
    print("  ✓ Confidence-based early stopping")
    print("  ✓ Adaptive tile sizing")
    print("  ✓ Detailed progress statistics")
    
    # Configuration for high-performance processing
    config = TileBufferConfig(
        max_memory_gb=4.0,
        tile_size=2048,
        adaptive_sizing_enabled=True,
        memory_pressure_threshold=0.7,
        cleanup_threshold=0.85,
        compression_enabled=True
    )
    
    print(f"\nHigh-performance configuration:")
    print(f"  Memory limit: {config.max_memory_gb}GB")
    print(f"  Tile size: {config.tile_size}px")
    print(f"  Memory pressure threshold: {config.memory_pressure_threshold}")
    print(f"  Compression enabled: {config.compression_enabled}")


def demonstrate_eta_accuracy():
    """Demonstrate ETA calculation accuracy."""
    print("\n" + "="*80)
    print("DEMONSTRATION: ETA Calculation Accuracy")
    print("="*80)
    
    print("ETA calculation features:")
    print("  ✓ Adaptive estimation based on recent processing times")
    print("  ✓ Early stopping consideration in ETA")
    print("  ✓ Confidence-based time adjustment")
    print("  ✓ Multi-stage processing time breakdown")
    
    # Simulate different processing scenarios
    scenarios = [
        {
            "name": "Fast Processing",
            "avg_tile_time": 0.05,
            "total_tiles": 1000,
            "description": "High-end GPU with optimized model"
        },
        {
            "name": "Standard Processing", 
            "avg_tile_time": 0.1,
            "total_tiles": 1000,
            "description": "Standard GPU configuration"
        },
        {
            "name": "Memory-Constrained",
            "avg_tile_time": 0.2,
            "total_tiles": 1000,
            "description": "Limited memory, smaller batches"
        }
    ]
    
    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")
        print(f"  Description: {scenario['description']}")
        print(f"  Average tile time: {scenario['avg_tile_time']}s")
        print(f"  Total tiles: {scenario['total_tiles']}")
        
        estimated_total_time = scenario['total_tiles'] * scenario['avg_tile_time']
        print(f"  Estimated total time: {estimated_total_time:.1f}s ({estimated_total_time/60:.1f} minutes)")
        
        # Calculate with early stopping
        early_stop_tiles = int(scenario['total_tiles'] * 0.7)  # Stop at 70% with high confidence
        early_stop_time = early_stop_tiles * scenario['avg_tile_time']
        print(f"  With early stopping (70%): {early_stop_time:.1f}s ({early_stop_time/60:.1f} minutes)")


def main():
    """Main demonstration function."""
    print("WSI Streaming Progress Tracking Demonstration")
    print("=" * 80)
    
    try:
        # Run demonstrations
        demonstrate_basic_progress_tracking()
        demonstrate_advanced_features()
        demonstrate_eta_accuracy()
        
        print("\n" + "="*80)
        print("DEMONSTRATION COMPLETE")
        print("="*80)
        print("\nKey benefits of enhanced progress tracking:")
        print("  🎯 Accurate ETA estimation with confidence intervals")
        print("  📊 Real-time progress callbacks for visualization")
        print("  🛑 Confidence-based early stopping")
        print("  💾 Comprehensive memory and performance monitoring")
        print("  📈 Multi-stage processing breakdown")
        print("  🔧 Adaptive tile sizing based on system resources")
        print("  📋 Detailed statistics for optimization")
        
    except Exception as e:
        logger.error(f"Demonstration failed: {e}")
        raise


if __name__ == "__main__":
    main()