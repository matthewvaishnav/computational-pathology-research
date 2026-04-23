"""
Demo script for PatchExtractor usage.

This script demonstrates how to use the PatchExtractor class to:
1. Generate grid coordinates for patch extraction
2. Extract patches at different pyramid levels
3. Use streaming extraction for memory efficiency
4. Filter patches using tissue masks
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.data.wsi_pipeline import PatchExtractor, WSIReader


def demo_basic_extraction():
    """Demonstrate basic patch extraction."""
    print("=" * 60)
    print("Demo 1: Basic Patch Extraction")
    print("=" * 60)

    # Initialize extractor with 256x256 patches, non-overlapping
    extractor = PatchExtractor(patch_size=256, stride=256, level=0)

    # Generate coordinates for a 10000x10000 slide
    slide_dimensions = (10000, 10000)
    coords = extractor.generate_coordinates(slide_dimensions)

    print(f"Slide dimensions: {slide_dimensions}")
    print(f"Patch size: {extractor.patch_size}")
    print(f"Stride: {extractor.stride}")
    print(f"Generated {len(coords)} patch coordinates")
    print(f"First 5 coordinates: {coords[:5]}")
    print(f"Last 5 coordinates: {coords[-5:]}")
    print()


def demo_overlapping_extraction():
    """Demonstrate overlapping patch extraction."""
    print("=" * 60)
    print("Demo 2: Overlapping Patch Extraction (50% overlap)")
    print("=" * 60)

    # Initialize extractor with 50% overlap (stride = patch_size / 2)
    extractor = PatchExtractor(patch_size=256, stride=128, level=0)

    # Generate coordinates for a 2048x2048 slide
    slide_dimensions = (2048, 2048)
    coords = extractor.generate_coordinates(slide_dimensions)

    print(f"Slide dimensions: {slide_dimensions}")
    print(f"Patch size: {extractor.patch_size}")
    print(f"Stride: {extractor.stride} (50% overlap)")
    print(f"Generated {len(coords)} patch coordinates")
    print(f"Overlap increases coverage for better feature extraction")
    print()


def demo_tissue_mask_filtering():
    """Demonstrate coordinate filtering with tissue mask."""
    print("=" * 60)
    print("Demo 3: Tissue Mask Filtering")
    print("=" * 60)

    extractor = PatchExtractor(patch_size=256, stride=256, level=0)

    # Create a synthetic tissue mask (tissue in center region)
    slide_dimensions = (2048, 2048)
    mask_size = (200, 200)
    tissue_mask = np.zeros(mask_size, dtype=bool)
    # Mark center region as tissue
    tissue_mask[50:150, 50:150] = True

    # Generate coordinates without mask
    coords_all = extractor.generate_coordinates(slide_dimensions)

    # Generate coordinates with tissue mask
    coords_tissue = extractor.generate_coordinates(
        slide_dimensions, tissue_mask=tissue_mask
    )

    print(f"Slide dimensions: {slide_dimensions}")
    print(f"Tissue mask size: {mask_size}")
    print(f"Coordinates without mask: {len(coords_all)}")
    print(f"Coordinates with tissue mask: {len(coords_tissue)}")
    print(f"Filtered out {len(coords_all) - len(coords_tissue)} background patches")
    print(f"Efficiency gain: {(1 - len(coords_tissue)/len(coords_all))*100:.1f}% reduction")
    print()


def demo_coordinate_conversion():
    """Demonstrate coordinate conversion between pyramid levels."""
    print("=" * 60)
    print("Demo 4: Coordinate Conversion Between Pyramid Levels")
    print("=" * 60)

    extractor = PatchExtractor(patch_size=256, stride=256, level=0)

    # Define pyramid levels with downsamples
    level_downsamples = [1.0, 2.0, 4.0, 8.0]

    # Generate coordinates at level 0
    coords_level0 = [(0, 0), (256, 0), (512, 0), (1024, 1024)]

    print(f"Original coordinates at level 0: {coords_level0}")
    print()

    # Convert to different levels
    for target_level in [1, 2, 3]:
        coords_converted = extractor.convert_coordinates_to_level(
            coords_level0,
            from_level=0,
            to_level=target_level,
            level_downsamples=level_downsamples,
        )
        downsample = level_downsamples[target_level]
        print(f"Level {target_level} (downsample {downsample}x): {coords_converted}")

    print()


def demo_streaming_extraction():
    """Demonstrate streaming extraction pattern."""
    print("=" * 60)
    print("Demo 5: Streaming Extraction Pattern")
    print("=" * 60)

    extractor = PatchExtractor(patch_size=256, stride=256, level=0)

    # Generate a small set of coordinates
    slide_dimensions = (1024, 1024)
    coords = extractor.generate_coordinates(slide_dimensions)

    print(f"Slide dimensions: {slide_dimensions}")
    print(f"Total patches to extract: {len(coords)}")
    print()
    print("Streaming extraction pattern (memory efficient):")
    print("for patch, coord in extractor.extract_patches_streaming(reader, coords):")
    print("    # Process patch immediately without storing all patches")
    print("    features = model.extract_features(patch)")
    print("    cache.save(features, coord)")
    print()
    print("This approach avoids loading all patches into memory at once,")
    print("enabling processing of gigapixel slides on limited hardware.")
    print()


def demo_multi_resolution():
    """Demonstrate multi-resolution extraction."""
    print("=" * 60)
    print("Demo 6: Multi-Resolution Extraction")
    print("=" * 60)

    slide_dimensions = (40000, 40000)  # Large 40x magnification slide

    print(f"Slide dimensions: {slide_dimensions}")
    print()

    # Extract at different pyramid levels
    for level, downsample in [(0, 1.0), (1, 2.0), (2, 4.0)]:
        extractor = PatchExtractor(patch_size=256, stride=256, level=level)
        coords = extractor.generate_coordinates(slide_dimensions)

        print(f"Level {level} (downsample {downsample}x):")
        print(f"  - Effective resolution: {slide_dimensions[0]/downsample:.0f}x{slide_dimensions[1]/downsample:.0f}")
        print(f"  - Number of patches: {len(coords)}")
        print(f"  - Coverage: {len(coords) * 256 * 256 / (slide_dimensions[0] * slide_dimensions[1]) * 100:.1f}%")
        print()


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║" + " " * 10 + "WSI Patch Extraction Demo" + " " * 23 + "║")
    print("╚" + "=" * 58 + "╝")
    print()

    demo_basic_extraction()
    demo_overlapping_extraction()
    demo_tissue_mask_filtering()
    demo_coordinate_conversion()
    demo_streaming_extraction()
    demo_multi_resolution()

    print("=" * 60)
    print("Demo Complete!")
    print("=" * 60)
    print()
    print("Key Features Demonstrated:")
    print("  ✓ Grid-based coordinate generation")
    print("  ✓ Configurable stride for overlapping patches")
    print("  ✓ Tissue mask filtering for efficiency")
    print("  ✓ Coordinate conversion between pyramid levels")
    print("  ✓ Streaming extraction for memory efficiency")
    print("  ✓ Multi-resolution support")
    print()
