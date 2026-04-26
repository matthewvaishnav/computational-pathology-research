#!/usr/bin/env python3
"""
Adaptive Tile Sizing Demo for Real-Time WSI Streaming

This script demonstrates the adaptive tile sizing functionality implemented
for the Real-Time WSI Streaming system. It shows how tile sizes automatically
adjust based on available memory and system conditions.

Key Features Demonstrated:
- Adaptive tile sizing based on available memory
- Memory pressure detection and response
- GPU memory awareness
- Dynamic adjustment during streaming
- Performance optimization for different hardware configurations

Usage:
    python examples/adaptive_tile_sizing_demo.py
"""

import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import psutil

import sys
from pathlib import Path

# Add the project root to the Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.wsi_pipeline import (
    TileBufferConfig,
    TileBufferPool,
    WSIStreamReader,
    StreamingMetadata,
    ProcessingError,
    ResourceError
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_synthetic_wsi_data(dimensions: tuple = (10000, 10000)) -> np.ndarray:
    """
    Create synthetic WSI data for demonstration purposes.
    
    Args:
        dimensions: (width, height) of the synthetic slide
        
    Returns:
        Synthetic WSI data as numpy array
    """
    logger.info(f"Creating synthetic WSI data with dimensions {dimensions}")
    
    # Create a simple synthetic slide with some patterns
    width, height = dimensions
    
    # Create base tissue-like pattern
    slide_data = np.random.randint(200, 255, (height, width, 3), dtype=np.uint8)
    
    # Add some "tissue" regions with different characteristics
    for i in range(5):
        center_x = np.random.randint(width // 4, 3 * width // 4)
        center_y = np.random.randint(height // 4, 3 * height // 4)
        radius = np.random.randint(500, 1500)
        
        # Create circular tissue region
        y, x = np.ogrid[:height, :width]
        mask = (x - center_x)**2 + (y - center_y)**2 <= radius**2
        
        # Add tissue-like coloring
        slide_data[mask] = [180 + np.random.randint(-20, 20) for _ in range(3)]
    
    return slide_data


class MockWSIReader:
    """Mock WSI reader for demonstration purposes."""
    
    def __init__(self, slide_data: np.ndarray):
        self.slide_data = slide_data
        self.dimensions = (slide_data.shape[1], slide_data.shape[0])  # (width, height)
        self.level_count = 1
        self.level_dimensions = [self.dimensions]
    
    def get_magnification(self):
        return 20.0
    
    def get_mpp(self):
        return (0.25, 0.25)
    
    def read_region(self, location, level, size):
        """Read a region from the synthetic slide."""
        x, y = location
        w, h = size
        
        # Ensure we don't go out of bounds
        x = max(0, min(x, self.slide_data.shape[1] - 1))
        y = max(0, min(y, self.slide_data.shape[0] - 1))
        w = min(w, self.slide_data.shape[1] - x)
        h = min(h, self.slide_data.shape[0] - y)
        
        # Extract region
        region = self.slide_data[y:y+h, x:x+w]
        
        # Pad if necessary to match requested size
        if region.shape[0] < size[1] or region.shape[1] < size[0]:
            padded = np.zeros((size[1], size[0], 3), dtype=np.uint8)
            padded[:region.shape[0], :region.shape[1]] = region
            return padded
        
        return region
    
    def close(self):
        pass


def demonstrate_adaptive_tile_sizing():
    """Demonstrate adaptive tile sizing functionality."""
    
    logger.info("=== Adaptive Tile Sizing Demonstration ===")
    
    # Create different configurations to test various scenarios
    configs = [
        {
            "name": "High Memory Configuration",
            "config": TileBufferConfig(
                adaptive_sizing_enabled=True,
                min_tile_size=256,
                max_tile_size=2048,
                tile_size=1024,
                max_memory_gb=4.0,
                memory_pressure_threshold=0.7
            ),
            "slide_size": (8000, 8000)
        },
        {
            "name": "Low Memory Configuration", 
            "config": TileBufferConfig(
                adaptive_sizing_enabled=True,
                min_tile_size=128,
                max_tile_size=1024,
                tile_size=512,
                max_memory_gb=1.0,
                memory_pressure_threshold=0.6
            ),
            "slide_size": (12000, 12000)
        },
        {
            "name": "Memory Constrained Configuration",
            "config": TileBufferConfig(
                adaptive_sizing_enabled=True,
                min_tile_size=64,
                max_tile_size=512,
                tile_size=256,
                max_memory_gb=0.5,
                memory_pressure_threshold=0.5
            ),
            "slide_size": (15000, 15000)
        }
    ]
    
    results = []
    
    for scenario in configs:
        logger.info(f"\n--- Testing {scenario['name']} ---")
        
        try:
            # Create tile buffer pool
            pool = TileBufferPool(scenario['config'])
            
            # Test adaptive sizing calculation
            initial_tile_size = pool.get_current_tile_size()
            logger.info(f"Initial tile size: {initial_tile_size}px")
            
            # Calculate adaptive tile size
            adaptive_size = pool.calculate_adaptive_tile_size()
            logger.info(f"Calculated adaptive tile size: {adaptive_size}px")
            
            # Update tile size
            size_changed = pool.update_adaptive_tile_size()
            final_tile_size = pool.get_current_tile_size()
            
            logger.info(f"Final tile size: {final_tile_size}px (changed: {size_changed})")
            
            # Get adaptive sizing statistics
            stats = pool.get_adaptive_sizing_stats()
            logger.info(f"Adaptive sizing enabled: {stats['adaptive_sizing_enabled']}")
            logger.info(f"Recommended tile size: {stats['recommended_tile_size']}px")
            
            # Test memory estimation
            estimated_memory = pool.estimate_tile_memory_usage(final_tile_size)
            logger.info(f"Estimated memory per tile: {estimated_memory / 1024:.1f}KB")
            
            # Test optimal batch size calculation
            optimal_batch = pool.get_optimal_batch_size_for_tile_size(
                final_tile_size, target_memory_gb=0.5
            )
            logger.info(f"Optimal batch size for 0.5GB: {optimal_batch}")
            
            # Store results
            results.append({
                'scenario': scenario['name'],
                'initial_size': initial_tile_size,
                'adaptive_size': adaptive_size,
                'final_size': final_tile_size,
                'size_changed': size_changed,
                'estimated_memory_kb': estimated_memory / 1024,
                'optimal_batch': optimal_batch
            })
            
        except Exception as e:
            logger.error(f"Error in scenario {scenario['name']}: {e}")
            results.append({
                'scenario': scenario['name'],
                'error': str(e)
            })
    
    return results


def demonstrate_streaming_with_adaptive_sizing():
    """Demonstrate streaming with adaptive tile sizing."""
    
    logger.info("\n=== Streaming with Adaptive Sizing Demonstration ===")
    
    # Create synthetic WSI data
    slide_data = create_synthetic_wsi_data((5000, 5000))
    mock_reader = MockWSIReader(slide_data)
    
    # Configure adaptive sizing
    config = TileBufferConfig(
        adaptive_sizing_enabled=True,
        min_tile_size=128,
        max_tile_size=1024,
        tile_size=512,
        max_memory_gb=1.0,
        memory_pressure_threshold=0.7
    )
    
    try:
        # Create a mock WSIStreamReader (we'll simulate it since we need the actual implementation)
        logger.info("Simulating WSI streaming with adaptive tile sizing...")
        
        # Create tile buffer pool
        pool = TileBufferPool(config)
        
        # Simulate processing tiles with different memory conditions
        logger.info("Simulating memory pressure scenarios...")
        
        # Scenario 1: Normal memory conditions
        logger.info("Scenario 1: Normal memory conditions")
        initial_size = pool.get_current_tile_size()
        logger.info(f"Initial tile size: {initial_size}px")
        
        # Simulate some tile processing
        for i in range(10):
            # Create synthetic tile
            tile = np.random.randint(0, 255, (initial_size, initial_size, 3), dtype=np.uint8)
            coord = (i * initial_size, 0)
            
            # Store tile in pool
            success = pool.store_tile(coord, 0, tile)
            if success:
                logger.debug(f"Stored tile at {coord}")
        
        # Check memory usage
        memory_usage = pool.get_memory_usage()
        logger.info(f"Memory usage after storing tiles: {memory_usage:.3f}GB")
        
        # Scenario 2: Simulate memory pressure
        logger.info("\nScenario 2: Simulating memory pressure")
        
        # Force memory optimization
        optimization_result = pool.optimize_memory_usage()
        logger.info(f"Memory optimization result: {optimization_result}")
        
        # Check if tile size adapted
        new_size = pool.get_current_tile_size()
        if new_size != initial_size:
            logger.info(f"Tile size adapted from {initial_size}px to {new_size}px")
        else:
            logger.info("Tile size remained unchanged")
        
        # Get final statistics
        final_stats = pool.get_buffer_stats()
        logger.info(f"Final buffer statistics:")
        logger.info(f"  - Tiles in buffer: {final_stats['tile_count']}")
        logger.info(f"  - Memory usage: {final_stats['memory_usage_gb']:.3f}GB")
        logger.info(f"  - Memory utilization: {final_stats['memory_utilization']:.1%}")
        logger.info(f"  - Hit rate: {final_stats['hit_rate']:.1%}")
        logger.info(f"  - Tile size adjustments: {final_stats['tile_size_adjustments']}")
        
        return final_stats
        
    except Exception as e:
        logger.error(f"Error in streaming demonstration: {e}")
        return None


def print_system_info():
    """Print current system information."""
    
    logger.info("\n=== System Information ===")
    
    # Memory information
    memory = psutil.virtual_memory()
    logger.info(f"Total system memory: {memory.total / 1024**3:.1f}GB")
    logger.info(f"Available memory: {memory.available / 1024**3:.1f}GB")
    logger.info(f"Memory usage: {memory.percent:.1f}%")
    
    # CPU information
    logger.info(f"CPU count: {psutil.cpu_count()}")
    logger.info(f"CPU usage: {psutil.cpu_percent(interval=1):.1f}%")
    
    # GPU information (if available)
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            logger.info(f"GPU count: {gpu_count}")
            
            for i in range(gpu_count):
                props = torch.cuda.get_device_properties(i)
                memory_gb = props.total_memory / 1024**3
                logger.info(f"GPU {i}: {props.name} ({memory_gb:.1f}GB)")
        else:
            logger.info("No CUDA GPUs available")
    except ImportError:
        logger.info("PyTorch not available - cannot check GPU status")


def main():
    """Main demonstration function."""
    
    print("Adaptive Tile Sizing Demonstration")
    print("=" * 50)
    
    # Print system information
    print_system_info()
    
    # Demonstrate adaptive tile sizing
    sizing_results = demonstrate_adaptive_tile_sizing()
    
    # Demonstrate streaming with adaptive sizing
    streaming_results = demonstrate_streaming_with_adaptive_sizing()
    
    # Print summary
    logger.info("\n=== Demonstration Summary ===")
    
    logger.info("Adaptive Tile Sizing Results:")
    for result in sizing_results:
        if 'error' in result:
            logger.info(f"  {result['scenario']}: ERROR - {result['error']}")
        else:
            logger.info(f"  {result['scenario']}:")
            logger.info(f"    Initial size: {result['initial_size']}px")
            logger.info(f"    Adaptive size: {result['adaptive_size']}px")
            logger.info(f"    Final size: {result['final_size']}px")
            logger.info(f"    Size changed: {result['size_changed']}")
            logger.info(f"    Memory per tile: {result['estimated_memory_kb']:.1f}KB")
            logger.info(f"    Optimal batch size: {result['optimal_batch']}")
    
    if streaming_results:
        logger.info(f"\nStreaming Results:")
        logger.info(f"  Final memory usage: {streaming_results['memory_usage_gb']:.3f}GB")
        logger.info(f"  Tiles processed: {streaming_results['tile_count']}")
        logger.info(f"  Hit rate: {streaming_results['hit_rate']:.1%}")
        logger.info(f"  Tile size adjustments: {streaming_results['tile_size_adjustments']}")
    
    logger.info("\nDemonstration completed successfully!")
    logger.info("The adaptive tile sizing system automatically adjusts tile dimensions")
    logger.info("based on available memory, ensuring optimal performance across different")
    logger.info("hardware configurations while maintaining memory usage within limits.")


if __name__ == "__main__":
    main()