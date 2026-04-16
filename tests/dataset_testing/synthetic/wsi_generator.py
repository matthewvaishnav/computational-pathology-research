"""
WSI synthetic data generator for testing.

This module provides synthetic whole-slide image data generation
for OpenSlide integration testing.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional, List
from pathlib import Path
import json
from dataclasses import dataclass
from tests.dataset_testing.base_interfaces import DatasetGenerator


@dataclass
class WSISyntheticSpec:
    """Specification for synthetic WSI data generation."""

    num_slides: int
    slide_dimensions: Tuple[int, int] = (50000, 50000)  # Width, height in pixels
    patch_size: int = 256
    num_levels: int = 4
    tissue_percentage_range: Tuple[float, float] = (0.3, 0.8)


class WSISyntheticGenerator(DatasetGenerator):
    """Synthetic WSI data generator for OpenSlide testing."""

    def __init__(self, random_seed: int = 42):
        """Initialize WSI synthetic generator.

        Args:
            random_seed: Random seed for reproducible generation
        """
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)

    def generate_samples(self, num_slides: int, **kwargs) -> Dict[str, Any]:
        """Generate synthetic WSI metadata and patch information.

        Args:
            num_slides: Number of slides to generate
            **kwargs: Additional parameters (spec, output_dir, etc.)

        Returns:
            Dictionary containing generated WSI data and metadata
        """
        spec = kwargs.get("spec", WSISyntheticSpec(num_slides=num_slides))
        output_dir = kwargs.get("output_dir", None)

        slides = []

        for i in range(num_slides):
            slide_data = self._generate_slide(slide_id=f"slide_{i:05d}", spec=spec)
            slides.append(slide_data)

        # Create dataset metadata
        metadata = {
            "num_slides": num_slides,
            "slide_dimensions": spec.slide_dimensions,
            "patch_size": spec.patch_size,
            "num_levels": spec.num_levels,
            "tissue_percentage_range": spec.tissue_percentage_range,
            "generator_seed": self.random_seed,
        }

        dataset = {
            "slides": slides,
            "metadata": metadata,
        }

        # Save to files if output directory provided
        if output_dir:
            self._save_dataset(dataset, Path(output_dir))

        return dataset

    def _generate_slide(self, slide_id: str, spec: WSISyntheticSpec) -> Dict[str, Any]:
        """Generate a single synthetic WSI slide.

        Args:
            slide_id: Unique slide identifier
            spec: Generation specification

        Returns:
            Dictionary containing slide data
        """
        width, height = spec.slide_dimensions

        # Generate pyramid level information
        levels = []
        for level in range(spec.num_levels):
            downsample = 2**level
            level_width = width // downsample
            level_height = height // downsample

            levels.append(
                {
                    "level": level,
                    "dimensions": (level_width, level_height),
                    "downsample": downsample,
                }
            )

        # Generate tissue regions
        tissue_percentage = self.rng.uniform(*spec.tissue_percentage_range)
        tissue_regions = self._generate_tissue_regions(width, height, tissue_percentage)

        # Generate patch information
        patches = self._generate_patch_info(width, height, spec.patch_size, tissue_regions)

        slide_data = {
            "slide_id": slide_id,
            "dimensions": (width, height),
            "levels": levels,
            "patch_size": spec.patch_size,
            "tissue_percentage": tissue_percentage,
            "tissue_regions": tissue_regions,
            "patches": patches,
            "num_patches": len(patches),
        }

        return slide_data

    def _generate_tissue_regions(
        self, width: int, height: int, tissue_percentage: float
    ) -> List[Dict[str, Any]]:
        """Generate tissue regions within the slide.

        Args:
            width: Slide width
            height: Slide height
            tissue_percentage: Percentage of slide covered by tissue

        Returns:
            List of tissue region dictionaries
        """
        total_area = width * height
        tissue_area = total_area * tissue_percentage

        # Generate 3-8 tissue regions
        num_regions = self.rng.randint(3, 9)
        regions = []

        remaining_area = tissue_area

        for i in range(num_regions):
            if i == num_regions - 1:
                # Last region gets remaining area
                region_area = remaining_area
            else:
                # Random area for this region (10-40% of remaining)
                region_area = remaining_area * self.rng.uniform(0.1, 0.4)
                remaining_area -= region_area

            # Generate region dimensions (roughly square)
            region_size = int(np.sqrt(region_area))
            region_width = self.rng.randint(int(region_size * 0.7), int(region_size * 1.3))
            region_height = int(region_area / region_width)

            # Random position within slide
            x = self.rng.randint(0, max(1, width - region_width))
            y = self.rng.randint(0, max(1, height - region_height))

            regions.append(
                {
                    "x": x,
                    "y": y,
                    "width": region_width,
                    "height": region_height,
                    "area": region_width * region_height,
                }
            )

        return regions

    def _generate_patch_info(
        self,
        slide_width: int,
        slide_height: int,
        patch_size: int,
        tissue_regions: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Generate patch extraction information.

        Args:
            slide_width: Slide width
            slide_height: Slide height
            patch_size: Size of patches to extract
            tissue_regions: List of tissue regions

        Returns:
            List of patch information dictionaries
        """
        patches = []

        # Grid-based patch extraction
        step_size = patch_size // 2  # 50% overlap

        for y in range(0, slide_height - patch_size, step_size):
            for x in range(0, slide_width - patch_size, step_size):
                # Check if patch overlaps with tissue
                patch_tissue_percentage = self._calculate_patch_tissue_overlap(
                    x, y, patch_size, tissue_regions
                )

                # Only include patches with sufficient tissue (>30%)
                if patch_tissue_percentage > 0.3:
                    patches.append(
                        {
                            "x": x,
                            "y": y,
                            "width": patch_size,
                            "height": patch_size,
                            "tissue_percentage": patch_tissue_percentage,
                            "level": 0,  # Base level
                        }
                    )

        return patches

    def _calculate_patch_tissue_overlap(
        self, patch_x: int, patch_y: int, patch_size: int, tissue_regions: List[Dict[str, Any]]
    ) -> float:
        """Calculate percentage of patch that overlaps with tissue.

        Args:
            patch_x: Patch x coordinate
            patch_y: Patch y coordinate
            patch_size: Patch size
            tissue_regions: List of tissue regions

        Returns:
            Percentage of patch covered by tissue (0.0 to 1.0)
        """
        patch_area = patch_size * patch_size
        overlap_area = 0

        patch_x2 = patch_x + patch_size
        patch_y2 = patch_y + patch_size

        for region in tissue_regions:
            region_x2 = region["x"] + region["width"]
            region_y2 = region["y"] + region["height"]

            # Calculate intersection
            intersect_x1 = max(patch_x, region["x"])
            intersect_y1 = max(patch_y, region["y"])
            intersect_x2 = min(patch_x2, region_x2)
            intersect_y2 = min(patch_y2, region_y2)

            if intersect_x2 > intersect_x1 and intersect_y2 > intersect_y1:
                intersect_area = (intersect_x2 - intersect_x1) * (intersect_y2 - intersect_y1)
                overlap_area += intersect_area

        return min(1.0, overlap_area / patch_area)

    def corrupt_samples(self, samples: Dict[str, Any], corruption_type: str) -> Dict[str, Any]:
        """Introduce controlled corruption for testing error handling.

        Args:
            samples: Original samples to corrupt
            corruption_type: Type of corruption to introduce

        Returns:
            Dictionary containing corrupted samples
        """
        corrupted_samples = samples.copy()
        slides = [slide.copy() for slide in corrupted_samples["slides"]]

        if corruption_type == "invalid_dimensions":
            # Create slides with invalid dimensions
            for slide in slides:
                if self.rng.random() < 0.2:  # Corrupt 20% of slides
                    slide["dimensions"] = (0, 0)  # Invalid dimensions

        elif corruption_type == "missing_levels":
            # Remove pyramid levels
            for slide in slides:
                if self.rng.random() < 0.15:  # Corrupt 15% of slides
                    slide["levels"] = []  # No levels available

        elif corruption_type == "coordinate_overflow":
            # Create patches with coordinates outside slide bounds
            for slide in slides:
                if self.rng.random() < 0.1:  # Corrupt 10% of slides
                    width, height = slide["dimensions"]
                    for patch in slide["patches"]:
                        if self.rng.random() < 0.3:  # Corrupt 30% of patches
                            patch["x"] = width + 1000  # Outside bounds
                            patch["y"] = height + 1000

        elif corruption_type == "negative_coordinates":
            # Create patches with negative coordinates
            for slide in slides:
                if self.rng.random() < 0.1:  # Corrupt 10% of slides
                    for patch in slide["patches"]:
                        if self.rng.random() < 0.2:  # Corrupt 20% of patches
                            patch["x"] = -100
                            patch["y"] = -100

        elif corruption_type == "inconsistent_patch_count":
            # Make patch count inconsistent with actual patches
            for slide in slides:
                if self.rng.random() < 0.1:  # Corrupt 10% of slides
                    slide["num_patches"] = len(slide["patches"]) + 100  # Wrong count

        corrupted_samples["slides"] = slides
        corrupted_samples["metadata"]["corruption_type"] = corruption_type

        return corrupted_samples

    def validate_samples(self, samples: Dict[str, Any]) -> bool:
        """Validate that generated samples meet expected criteria.

        Args:
            samples: Samples to validate

        Returns:
            True if samples are valid, False otherwise
        """
        try:
            slides = samples["slides"]

            # Check basic structure
            if not isinstance(slides, list):
                return False

            # Validate each slide
            for slide in slides:
                # Check required fields
                required_fields = [
                    "slide_id",
                    "dimensions",
                    "levels",
                    "patch_size",
                    "tissue_percentage",
                    "tissue_regions",
                    "patches",
                    "num_patches",
                ]
                if not all(field in slide for field in required_fields):
                    return False

                # Check slide ID is valid
                if not isinstance(slide["slide_id"], str) or not slide["slide_id"]:
                    return False

                # Check dimensions
                dimensions = slide["dimensions"]
                if not isinstance(dimensions, tuple) or len(dimensions) != 2:
                    return False

                width, height = dimensions
                if width <= 0 or height <= 0:
                    return False

                # Check levels
                levels = slide["levels"]
                if not isinstance(levels, list) or not levels:
                    return False

                for level in levels:
                    if not all(key in level for key in ["level", "dimensions", "downsample"]):
                        return False

                    if level["downsample"] <= 0:
                        return False

                # Check tissue percentage
                tissue_percentage = slide["tissue_percentage"]
                if not (0.0 <= tissue_percentage <= 1.0):
                    return False

                # Check patches
                patches = slide["patches"]
                if not isinstance(patches, list):
                    return False

                if len(patches) != slide["num_patches"]:
                    return False

                # Validate patch coordinates are within slide bounds
                for patch in patches:
                    if not all(key in patch for key in ["x", "y", "width", "height"]):
                        return False

                    if patch["x"] < 0 or patch["y"] < 0:
                        return False

                    if patch["x"] + patch["width"] > width:
                        return False

                    if patch["y"] + patch["height"] > height:
                        return False

                    if not (0.0 <= patch["tissue_percentage"] <= 1.0):
                        return False

            return True

        except Exception:
            return False

    def _save_dataset(self, dataset: Dict[str, Any], output_dir: Path):
        """Save generated dataset to files.

        Args:
            dataset: Generated dataset
            output_dir: Output directory
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save slide information
        slides_info = []

        for slide in dataset["slides"]:
            slide_info = {
                "slide_id": slide["slide_id"],
                "dimensions": slide["dimensions"],
                "levels": slide["levels"],
                "patch_size": slide["patch_size"],
                "tissue_percentage": slide["tissue_percentage"],
                "num_patches": slide["num_patches"],
            }
            slides_info.append(slide_info)

            # Save detailed patch information for each slide
            patch_file = output_dir / f"{slide['slide_id']}_patches.json"
            with open(patch_file, "w") as f:
                json.dump(
                    {
                        "slide_id": slide["slide_id"],
                        "tissue_regions": slide["tissue_regions"],
                        "patches": slide["patches"],
                    },
                    f,
                    indent=2,
                )

        # Save slide index
        with open(output_dir / "slide_index.json", "w") as f:
            json.dump(slides_info, f, indent=2)

        # Save dataset metadata
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(dataset["metadata"], f, indent=2)
