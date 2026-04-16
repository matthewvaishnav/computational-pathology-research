"""
PCam synthetic data generator for testing.

This module provides synthetic PatchCamelyon (PCam) data generation
for comprehensive testing without requiring large real datasets.
"""

import numpy as np
import torch
from typing import Dict, Any, Tuple, Optional
from pathlib import Path
import h5py
from dataclasses import dataclass
from tests.dataset_testing.base_interfaces import DatasetGenerator


@dataclass
class PCamSyntheticSpec:
    """Specification for synthetic PCam data generation."""
    
    num_samples: int
    image_shape: Tuple[int, int, int] = (96, 96, 3)
    label_distribution: Dict[int, float] = None
    noise_level: float = 0.1
    corruption_probability: float = 0.0
    
    def __post_init__(self):
        if self.label_distribution is None:
            self.label_distribution = {0: 0.5, 1: 0.5}


class PCamSyntheticGenerator(DatasetGenerator):
    """Synthetic PCam data generator with realistic histopathology characteristics."""
    
    def __init__(self, random_seed: int = 42):
        """Initialize PCam synthetic generator.
        
        Args:
            random_seed: Random seed for reproducible generation
        """
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)
        
        # Realistic PCam statistics (approximated from real data)
        self.color_stats = {
            "mean": np.array([0.485, 0.456, 0.406]),  # ImageNet-like
            "std": np.array([0.229, 0.224, 0.225]),
        }
        
        # Histopathology-specific patterns
        self.tissue_patterns = self._create_tissue_patterns()
    
    def _create_tissue_patterns(self) -> Dict[str, np.ndarray]:
        """Create basic tissue pattern templates."""
        patterns = {}
        
        # Normal tissue pattern (more uniform)
        normal_pattern = self.rng.normal(0.6, 0.1, (96, 96, 3))
        normal_pattern = np.clip(normal_pattern, 0, 1)
        patterns["normal"] = normal_pattern
        
        # Tumor tissue pattern (more heterogeneous)
        tumor_pattern = self.rng.normal(0.4, 0.2, (96, 96, 3))
        # Add some structure/nuclei-like spots
        for _ in range(20):
            x, y = self.rng.randint(10, 86, 2)
            tumor_pattern[x-5:x+5, y-5:y+5] *= 0.3  # Dark nuclei
        tumor_pattern = np.clip(tumor_pattern, 0, 1)
        patterns["tumor"] = tumor_pattern
        
        return patterns
    
    def generate_samples(self, num_samples: int, **kwargs) -> Dict[str, Any]:
        """Generate synthetic PCam samples.
        
        Args:
            num_samples: Number of samples to generate
            **kwargs: Additional parameters (spec, output_dir, etc.)
        
        Returns:
            Dictionary containing generated samples and metadata
        """
        spec = kwargs.get("spec", PCamSyntheticSpec(num_samples=num_samples))
        output_dir = kwargs.get("output_dir", None)
        
        # Generate labels according to distribution
        labels = self._generate_labels(num_samples, spec.label_distribution)
        
        # Generate images
        images = []
        for i in range(num_samples):
            image = self._generate_image(labels[i], spec)
            images.append(image)
        
        images = np.array(images)
        
        # Create sample metadata
        metadata = {
            "num_samples": num_samples,
            "image_shape": spec.image_shape,
            "label_distribution": spec.label_distribution,
            "noise_level": spec.noise_level,
            "corruption_probability": spec.corruption_probability,
            "generator_seed": self.random_seed,
        }
        
        samples = {
            "images": images,
            "labels": labels,
            "metadata": metadata,
        }
        
        # Save to files if output directory provided
        if output_dir:
            self._save_samples(samples, Path(output_dir))
        
        return samples
    
    def _generate_labels(self, num_samples: int, distribution: Dict[int, float]) -> np.ndarray:
        """Generate labels according to specified distribution.
        
        Args:
            num_samples: Number of labels to generate
            distribution: Label distribution {label: probability}
        
        Returns:
            Array of labels
        """
        labels = []
        probs = list(distribution.values())
        label_values = list(distribution.keys())
        
        for _ in range(num_samples):
            label = self.rng.choice(label_values, p=probs)
            labels.append(label)
        
        return np.array(labels)
    
    def _generate_image(self, label: int, spec: PCamSyntheticSpec) -> np.ndarray:
        """Generate a single synthetic histopathology image.
        
        Args:
            label: Image label (0=normal, 1=tumor)
            spec: Generation specification
        
        Returns:
            Generated image as uint8 array
        """
        # Start with appropriate tissue pattern
        if label == 0:
            base_image = self.tissue_patterns["normal"].copy()
        else:
            base_image = self.tissue_patterns["tumor"].copy()
        
        # Add realistic color variation
        base_image = self._add_color_variation(base_image)
        
        # Add noise
        if spec.noise_level > 0:
            noise = self.rng.normal(0, spec.noise_level, spec.image_shape)
            base_image = base_image + noise
        
        # Add stain variation (H&E staining simulation)
        base_image = self._simulate_stain_variation(base_image)
        
        # Convert to uint8
        base_image = np.clip(base_image * 255, 0, 255).astype(np.uint8)
        
        return base_image
    
    def _add_color_variation(self, image: np.ndarray) -> np.ndarray:
        """Add realistic color variation to simulate H&E staining."""
        # Simulate hematoxylin (blue/purple) and eosin (pink) channels
        h_channel = self.rng.beta(2, 5, image.shape[:2])  # More blue in nuclei
        e_channel = self.rng.beta(5, 2, image.shape[:2])  # More pink in cytoplasm
        
        # Apply to RGB channels
        image[:, :, 0] = image[:, :, 0] * (0.7 + 0.3 * e_channel)  # Red (eosin)
        image[:, :, 1] = image[:, :, 1] * (0.8 + 0.2 * (h_channel + e_channel) / 2)  # Green
        image[:, :, 2] = image[:, :, 2] * (0.6 + 0.4 * h_channel)  # Blue (hematoxylin)
        
        return image
    
    def _simulate_stain_variation(self, image: np.ndarray) -> np.ndarray:
        """Simulate stain intensity and color variation."""
        # Random stain intensity variation
        stain_intensity = self.rng.uniform(0.8, 1.2)
        
        # Random color shift (simulate staining variations)
        color_shift = self.rng.uniform(-0.1, 0.1, 3)
        
        image = image * stain_intensity
        image = image + color_shift
        
        return np.clip(image, 0, 1)
    
    def corrupt_samples(self, samples: Dict[str, Any], corruption_type: str) -> Dict[str, Any]:
        """Introduce controlled corruption for testing error handling.
        
        Args:
            samples: Original samples to corrupt
            corruption_type: Type of corruption to introduce
        
        Returns:
            Dictionary containing corrupted samples
        """
        corrupted_samples = samples.copy()
        images = corrupted_samples["images"].copy()
        labels = corrupted_samples["labels"].copy()
        
        if corruption_type == "image_noise":
            # Add excessive noise to images
            noise = self.rng.normal(0, 0.5, images.shape)
            images = np.clip(images.astype(np.float32) + noise * 255, 0, 255).astype(np.uint8)
        
        elif corruption_type == "label_flip":
            # Randomly flip some labels
            flip_mask = self.rng.random(len(labels)) < 0.2
            labels[flip_mask] = 1 - labels[flip_mask]
        
        elif corruption_type == "image_truncation":
            # Truncate some images (simulate incomplete loading)
            for i in range(len(images)):
                if self.rng.random() < 0.1:
                    truncate_point = self.rng.randint(images.shape[1] // 2, images.shape[1])
                    images[i, truncate_point:] = 0
        
        elif corruption_type == "dimension_mismatch":
            # Create images with wrong dimensions
            wrong_shape_indices = self.rng.choice(len(images), size=max(1, len(images) // 10), replace=False)
            for idx in wrong_shape_indices:
                # Create new array with wrong dimensions instead of resizing in place
                wrong_image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
                # Replace the entire images array to avoid shape mismatch
                if idx == 0:
                    # Create new images array with mixed dimensions
                    new_images = []
                    for i in range(len(images)):
                        if i in wrong_shape_indices:
                            new_images.append(wrong_image)
                        else:
                            new_images.append(images[i])
                    # Convert to object array to allow different shapes
                    images = np.array(new_images, dtype=object)
                    break
        
        elif corruption_type == "missing_data":
            # Set some samples to None/invalid
            missing_indices = self.rng.choice(len(images), size=max(1, len(images) // 20), replace=False)
            for idx in missing_indices:
                images[idx] = np.zeros_like(images[idx])
                labels[idx] = -1  # Invalid label
        
        corrupted_samples["images"] = images
        corrupted_samples["labels"] = labels
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
            images = samples["images"]
            labels = samples["labels"]
            
            # Check basic structure
            if not isinstance(images, np.ndarray) or not isinstance(labels, np.ndarray):
                return False
            
            # Check dimensions
            if len(images.shape) != 4 or images.shape[1:] != (96, 96, 3):
                return False
            
            if len(labels.shape) != 1 or len(labels) != len(images):
                return False
            
            # Check data types
            if images.dtype != np.uint8:
                return False
            
            # Check value ranges
            if np.any(images < 0) or np.any(images > 255):
                return False
            
            if not np.all(np.isin(labels, [0, 1])):
                return False
            
            # Check for reasonable image statistics
            mean_intensity = np.mean(images)
            if mean_intensity < 50 or mean_intensity > 200:  # Reasonable range for histopathology
                return False
            
            return True
            
        except Exception:
            return False
    
    def _save_samples(self, samples: Dict[str, Any], output_dir: Path):
        """Save generated samples to files.
        
        Args:
            samples: Generated samples
            output_dir: Output directory
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save as HDF5 (PCam format)
        h5_path = output_dir / "synthetic_pcam.h5"
        with h5py.File(h5_path, "w") as f:
            f.create_dataset("x", data=samples["images"], compression="gzip")
            f.create_dataset("y", data=samples["labels"], compression="gzip")
            
            # Save metadata as attributes
            for key, value in samples["metadata"].items():
                if isinstance(value, dict):
                    # Convert dict to string for HDF5 storage
                    f.attrs[key] = str(value)
                else:
                    f.attrs[key] = value
        
        # Also save as numpy arrays for convenience
        np.save(output_dir / "images.npy", samples["images"])
        np.save(output_dir / "labels.npy", samples["labels"])
        
        # Save metadata as JSON
        import json
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(samples["metadata"], f, indent=2)
    
    def create_dataset_splits(
        self, 
        samples: Dict[str, Any], 
        split_ratios: Dict[str, float] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Create train/val/test splits from generated samples.
        
        Args:
            samples: Generated samples
            split_ratios: Split ratios {"train": 0.7, "val": 0.15, "test": 0.15}
        
        Returns:
            Dictionary with splits
        """
        if split_ratios is None:
            split_ratios = {"train": 0.7, "val": 0.15, "test": 0.15}
        
        num_samples = len(samples["images"])
        indices = np.arange(num_samples)
        self.rng.shuffle(indices)
        
        # Calculate split boundaries
        train_end = int(num_samples * split_ratios["train"])
        val_end = train_end + int(num_samples * split_ratios["val"])
        
        splits = {}
        split_indices = {
            "train": indices[:train_end],
            "val": indices[train_end:val_end],
            "test": indices[val_end:],
        }
        
        for split_name, split_idx in split_indices.items():
            splits[split_name] = {
                "images": samples["images"][split_idx],
                "labels": samples["labels"][split_idx],
                "metadata": samples["metadata"].copy(),
            }
            splits[split_name]["metadata"]["split"] = split_name
            splits[split_name]["metadata"]["num_samples"] = len(split_idx)
        
        return splits