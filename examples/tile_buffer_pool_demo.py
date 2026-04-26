#!/usr/bin/env python3
"""
Demonstration of TileBufferPool for Real-Time WSI Streaming.

This example shows how to use the TileBufferPool for efficient tile caching
and memory management during WSI streaming operations.
"""

import numpy as np
import time
from pathlib import Path

# Add src to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data.wsi_pipeline.tile_buffer_pool import TileBufferPool, TileBufferConfig


def create_synthetic_tile(size: int = 256) -> np.ndarray:
    """Create a synthetic WSI tile for demonstration."""
    return np.random.randint(0, 255, (size, size, 3), dtype=np.uint8)


def simulate_wsi_streaming():
    """Simulate WSI streaming with tile buffer pool."""
    print("=== TileBufferPool Demo: Real-Time WSI Streaming ===\n")
    
    # Configure buffer pool for streaming
    config = TileBufferConfig(
        max_memory_gb=1.0,  # 1GB memory limit
        min_memory_gb=0.5,  # 500MB minimum
        initial_buffer_size=16,
        max_buffer_size=64,
        tile_size=256,
        memory_pressure_threshold=0.8,
        cleanup_threshold=0.9,
        preload_enabled=True,
        compression_enabled=False,  # Disable for demo speed
        thread_safe=True
    )
    
    print(f"Buffer Configuration:")
    print(f"  Memory Limit: {config.max_memory_gb:.1f}GB")
    print(f"  Buffer Size: {config.initial_buffer_size}-{config.max_buffer_size} tiles")
    print(f"  Tile Size: {config.tile_size}x{config.tile_size} pixels")
    print(f"  Memory Pressure Threshold: {config.memory_pressure_threshold:.0%}")
    print()
    
    # Initialize buffer pool
    with TileBufferPool(config) as pool:
        print("✓ TileBufferPool initialized")
        print(f"  Initial memory usage: {pool.get_memory_usage():.3f}GB")
        print()
        
        # Simulate streaming WSI tiles
        print("Simulating WSI tile streaming...")
        
        # Phase 1: Load initial tiles
        print("\nPhase 1: Loading initial tiles...")
        for i in range(20):
            coordinate = (i * 256, 0)  # Simulate scanning across slide
            level = 0
            tile = create_synthetic_tile()
            
            success = pool.store_tile(coordinate, level, tile)
            if success:
                print(f"  Stored tile at {coordinate}: {tile.nbytes / 1024:.1f}KB")
            else:
                print(f"  Failed to store tile at {coordinate}")
            
            # Show progress every 5 tiles
            if (i + 1) % 5 == 0:
                stats = pool.get_buffer_stats()
                print(f"    Progress: {i+1}/20 tiles, "
                      f"Memory: {stats['memory_usage_gb']:.3f}GB, "
                      f"Hit Rate: {stats['hit_rate']:.2%}")
        
        # Phase 2: Access patterns (simulate attention computation)
        print("\nPhase 2: Simulating attention computation access patterns...")
        access_coordinates = [(i * 256, 0) for i in [0, 5, 10, 15, 2, 8, 12]]
        
        for coord in access_coordinates:
            tile = pool.get_tile(coord, 0)
            if tile is not None:
                print(f"  ✓ Retrieved tile at {coord} for attention computation")
            else:
                print(f"  ✗ Tile at {coord} not in cache (cache miss)")
        
        # Phase 3: Memory pressure simulation
        print("\nPhase 3: Simulating memory pressure with large tiles...")
        large_tiles_stored = 0
        
        for i in range(50):  # Try to store many tiles to trigger eviction
            coordinate = (i * 256, 256)  # Second row
            level = 0
            tile = create_synthetic_tile(512)  # Larger tiles (4x memory)
            
            success = pool.store_tile(coordinate, level, tile)
            if success:
                large_tiles_stored += 1
            
            # Check memory pressure every 10 tiles
            if (i + 1) % 10 == 0:
                stats = pool.get_buffer_stats()
                print(f"    Large tiles stored: {large_tiles_stored}, "
                      f"Memory: {stats['memory_usage_gb']:.3f}GB, "
                      f"Evictions: {stats['total_evictions']}")
                
                if stats['memory_utilization'] > 0.9:
                    print(f"    ⚠️  High memory utilization: {stats['memory_utilization']:.1%}")
        
        # Phase 4: Adaptive optimization
        print("\nPhase 4: Running adaptive memory optimization...")
        optimization_results = pool.optimize_memory_usage()
        
        print(f"  Memory freed: {optimization_results['memory_freed_gb']:.3f}GB")
        print(f"  Tiles evicted: {optimization_results['tiles_evicted']}")
        print(f"  System memory pressure: {optimization_results['system_memory_pressure']}")
        
        # Phase 5: Preloading simulation
        print("\nPhase 5: Simulating tile preloading...")
        
        def mock_tile_loader(x: int, y: int, level: int) -> np.ndarray:
            """Mock tile loader function."""
            return create_synthetic_tile()
        
        preload_coords = [(i * 256, 512) for i in range(10)]  # Third row
        loaded_count = pool.preload_tiles(preload_coords, 0, mock_tile_loader)
        
        print(f"  Preloaded {loaded_count}/{len(preload_coords)} tiles")
        
        # Final statistics
        print("\n=== Final Statistics ===")
        final_stats = pool.get_buffer_stats()
        
        print(f"Total tiles in buffer: {final_stats['tile_count']}")
        print(f"Memory usage: {final_stats['memory_usage_gb']:.3f}GB / {final_stats['memory_limit_gb']:.1f}GB")
        print(f"Memory utilization: {final_stats['memory_utilization']:.1%}")
        print(f"Cache hit rate: {final_stats['hit_rate']:.1%}")
        print(f"Total hits: {final_stats['total_hits']}")
        print(f"Total misses: {final_stats['total_misses']}")
        print(f"Total evictions: {final_stats['total_evictions']}")
        print(f"Memory cleanups: {final_stats['memory_cleanups']}")
        
        # Adaptive buffer size recommendation
        recommended_size = pool.get_adaptive_buffer_size()
        print(f"Recommended buffer size: {recommended_size} tiles")
        
        print("\n✓ Demo completed successfully!")


def demonstrate_configuration_validation():
    """Demonstrate configuration validation."""
    print("\n=== Configuration Validation Demo ===\n")
    
    # Valid configuration
    try:
        valid_config = TileBufferConfig(
            max_memory_gb=2.0,
            min_memory_gb=1.0,
            tile_size=512
        )
        valid_config.validate()
        print("✓ Valid configuration accepted")
    except ValueError as e:
        print(f"✗ Configuration error: {e}")
    
    # Invalid configurations
    invalid_configs = [
        {"max_memory_gb": 0.1, "description": "Memory limit too low"},
        {"max_memory_gb": 50.0, "description": "Memory limit too high"},
        {"min_memory_gb": 2.0, "max_memory_gb": 1.0, "description": "Min > Max memory"},
        {"tile_size": 32, "description": "Tile size too small"},
        {"memory_pressure_threshold": 1.5, "description": "Invalid threshold"},
    ]
    
    for config_params in invalid_configs:
        description = config_params.pop("description")
        try:
            config = TileBufferConfig(**config_params)
            config.validate()
            print(f"✗ Should have failed: {description}")
        except ValueError as e:
            print(f"✓ Correctly rejected: {description}")


def demonstrate_memory_management():
    """Demonstrate memory management features."""
    print("\n=== Memory Management Demo ===\n")
    
    # Small memory limit to trigger management
    config = TileBufferConfig(
        max_memory_gb=0.5,  # Small limit
        min_memory_gb=0.5,
        initial_buffer_size=8,  # Fix: must be <= max_buffer_size
        max_buffer_size=10,
        memory_pressure_threshold=0.7,
        cleanup_threshold=0.9
    )
    
    with TileBufferPool(config) as pool:
        print(f"Memory limit: {pool.get_memory_limit():.1f}GB")
        
        # Store tiles until memory pressure
        tiles_stored = 0
        for i in range(100):
            tile = create_synthetic_tile(512)  # Large tiles
            coordinate = (i * 512, 0)
            
            success = pool.store_tile(coordinate, 0, tile)
            if success:
                tiles_stored += 1
            
            stats = pool.get_buffer_stats()
            
            if i % 10 == 0:
                print(f"Tiles stored: {tiles_stored}, "
                      f"Memory: {stats['memory_usage_gb']:.3f}GB, "
                      f"Utilization: {stats['memory_utilization']:.1%}, "
                      f"Evictions: {stats['total_evictions']}")
            
            # Stop if we can't store more tiles
            if not success and i > 5:  # Allow some initial failures
                print(f"Stopped at {i} attempts due to memory constraints")
                break
        
        print(f"\nFinal memory utilization: {stats['memory_utilization']:.1%}")
        print(f"Total evictions performed: {stats['total_evictions']}")


if __name__ == "__main__":
    # Run all demonstrations
    simulate_wsi_streaming()
    demonstrate_configuration_validation()
    demonstrate_memory_management()
    
    print("\n🎉 All demonstrations completed!")