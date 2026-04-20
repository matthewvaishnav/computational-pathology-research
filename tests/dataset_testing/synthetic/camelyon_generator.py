"""
CAMELYON synthetic data generator for testing.

This module provides synthetic CAMELYON slide-level data generation
for comprehensive testing without requiring large real datasets.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Tuple

import h5py
import numpy as np

from tests.dataset_testing.base_interfaces import DatasetGenerator


@dataclass
class CAMELYONSyntheticSpec:
    """Specification for synthetic CAMELYON data generation."""

    num_slides: int
    patches_per_slide_range: Tuple[int, int] = (50, 500)
    feature_dim: int = 2048
    coordinate_range: Tuple[int, int] = (0, 10000)
    patient_slide_distribution: Dict[str, int] = field(default_factory=dict)
    label_distribution: Dict[int, float] = field(default_factory=lambda: {0: 0.6, 1: 0.4})


class CAMELYONSyntheticGenerator(DatasetGenerator):
    """Synthetic CAMELYON data generator for slide-level classification."""

    def __init__(self, random_seed: int = 42):
        """Initialize CAMELYON synthetic generator.

        Args:
            random_seed: Random seed for reproducible generation
        """
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)

        # Realistic feature statistics (approximated from ResNet features)
        self.feature_stats = {
            "mean": 0.0,
            "std": 1.0,
            "feature_dim": 2048,
        }

        # Slide-level patterns
        self.slide_patterns = self._create_slide_patterns()

    def _create_slide_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Create slide-level pattern templates."""
        patterns = {}

        # Normal slide pattern (more uniform features)
        patterns["normal"] = {
            "feature_mean": 0.0,
            "feature_std": 0.8,
            "spatial_clustering": 0.2,  # Low clustering
            "attention_sparsity": 0.1,  # Uniform attention
        }

        # Tumor slide pattern (more heterogeneous features)
        patterns["tumor"] = {
            "feature_mean": 0.2,
            "feature_std": 1.2,
            "spatial_clustering": 0.8,  # High clustering
            "attention_sparsity": 0.6,  # Sparse attention on tumor regions
        }

        return patterns

    def generate_samples(self, num_slides: int, **kwargs) -> Dict[str, Any]:
        """Generate synthetic CAMELYON slide data.

        Args:
            num_slides: Number of slides to generate
            **kwargs: Additional parameters (spec, output_dir, etc.)

        Returns:
            Dictionary containing generated slides and metadata
        """
        spec = kwargs.get("spec", CAMELYONSyntheticSpec(num_slides=num_slides))
        output_dir = kwargs.get("output_dir", None)

        slides = []
        slide_index = []

        # Generate patient distribution if not provided
        if not spec.patient_slide_distribution:
            spec.patient_slide_distribution = self._generate_patient_distribution(num_slides)

        slide_id = 0
        for patient_id, num_patient_slides in spec.patient_slide_distribution.items():
            for slide_idx in range(num_patient_slides):
                slide_data = self._generate_slide(
                    slide_id=f"slide_{slide_id:05d}", patient_id=patient_id, spec=spec
                )
                slides.append(slide_data)

                # Create slide index entry
                slide_index.append(
                    {
                        "slide_id": slide_data["slide_id"],
                        "patient_id": slide_data["patient_id"],
                        "label": int(slide_data["label"]),  # Convert numpy int to Python int
                        "num_patches": int(
                            slide_data["num_patches"]
                        ),  # Convert numpy int to Python int
                        "file_path": f"features/{slide_data['slide_id']}.h5",
                    }
                )

                slide_id += 1

        # Create dataset metadata
        metadata = {
            "num_slides": len(slides),
            "num_patients": len(spec.patient_slide_distribution),
            "feature_dim": spec.feature_dim,
            "patches_per_slide_range": spec.patches_per_slide_range,
            "coordinate_range": spec.coordinate_range,
            "label_distribution": spec.label_distribution,
            "generator_seed": self.random_seed,
        }

        dataset = {
            "slides": slides,
            "slide_index": slide_index,
            "metadata": metadata,
        }

        # Save to files if output directory provided
        if output_dir:
            self._save_dataset(dataset, Path(output_dir))

        return dataset

    def _generate_patient_distribution(self, num_slides: int) -> Dict[str, int]:
        """Generate realistic patient-slide distribution.

        Args:
            num_slides: Total number of slides

        Returns:
            Dictionary mapping patient_id to number of slides
        """
        # Realistic distribution: most patients have 1-3 slides, some have more
        num_patients = max(1, num_slides // 2)  # Roughly 2 slides per patient on average

        distribution = {}
        slides_assigned = 0

        for i in range(num_patients):
            patient_id = f"patient_{i:04d}"

            if i < num_patients - 1:
                # Assign 1-4 slides per patient (weighted towards 1-2)
                slides_for_patient = self.rng.choice([1, 2, 3, 4], p=[0.4, 0.4, 0.15, 0.05])
                slides_for_patient = min(slides_for_patient, num_slides - slides_assigned)
            else:
                # Last patient gets remaining slides
                slides_for_patient = num_slides - slides_assigned

            if slides_for_patient > 0:
                distribution[patient_id] = slides_for_patient
                slides_assigned += slides_for_patient

        return distribution

    def _generate_slide(
        self, slide_id: str, patient_id: str, spec: CAMELYONSyntheticSpec
    ) -> Dict[str, Any]:
        """Generate a single synthetic slide.

        Args:
            slide_id: Unique slide identifier
            patient_id: Patient identifier
            spec: Generation specification

        Returns:
            Dictionary containing slide data
        """
        # Generate slide label
        label_probs = list(spec.label_distribution.values())
        label_values = list(spec.label_distribution.keys())
        label = int(self.rng.choice(label_values, p=label_probs))  # Convert to Python int

        # Generate number of patches
        num_patches = int(self.rng.randint(*spec.patches_per_slide_range))  # Convert to Python int

        # Generate features based on slide pattern
        pattern = self.slide_patterns["tumor" if label == 1 else "normal"]
        features = self._generate_slide_features(num_patches, spec.feature_dim, pattern)

        # Generate coordinates
        coordinates = self._generate_coordinates(num_patches, spec.coordinate_range, pattern)

        # Generate attention weights (for attention-based models)
        attention_weights = self._generate_attention_weights(num_patches, pattern)

        slide_data = {
            "slide_id": slide_id,
            "patient_id": patient_id,
            "label": label,
            "num_patches": num_patches,
            "features": features,
            "coordinates": coordinates,
            "attention_weights": attention_weights,
        }

        return slide_data

    def _generate_slide_features(
        self, num_patches: int, feature_dim: int, pattern: Dict[str, Any]
    ) -> np.ndarray:
        """Generate feature vectors for a slide.

        Args:
            num_patches: Number of patches in slide
            feature_dim: Dimension of feature vectors
            pattern: Slide pattern parameters

        Returns:
            Feature array of shape (num_patches, feature_dim)
        """
        # Base features from normal distribution
        features = self.rng.normal(
            pattern["feature_mean"], pattern["feature_std"], (num_patches, feature_dim)
        )

        # Add spatial clustering for tumor slides
        if pattern["spatial_clustering"] > 0.5:
            # Create clusters of similar features
            num_clusters = max(2, num_patches // 20)
            cluster_centers = self.rng.normal(0, 2, (num_clusters, feature_dim))

            # Assign patches to clusters
            cluster_assignments = self.rng.randint(0, num_clusters, num_patches)

            for i, cluster_id in enumerate(cluster_assignments):
                # Add cluster-specific variation
                cluster_noise = self.rng.normal(0, 0.5, feature_dim)
                features[i] += cluster_centers[cluster_id] + cluster_noise

        # Normalize features (common in deep learning)
        features = features / np.linalg.norm(features, axis=1, keepdims=True)

        return features.astype(np.float32)

    def _generate_coordinates(
        self, num_patches: int, coordinate_range: Tuple[int, int], pattern: Dict[str, Any]
    ) -> np.ndarray:
        """Generate patch coordinates.

        Args:
            num_patches: Number of patches
            coordinate_range: Range of coordinate values
            pattern: Slide pattern parameters

        Returns:
            Coordinate array of shape (num_patches, 2)
        """
        min_coord, max_coord = coordinate_range

        if pattern["spatial_clustering"] > 0.5:
            # Generate clustered coordinates for tumor slides
            num_clusters = max(2, num_patches // 20)
            cluster_centers = self.rng.uniform(
                min_coord + 1000, max_coord - 1000, (num_clusters, 2)
            )

            coordinates = []
            patches_per_cluster = num_patches // num_clusters

            for i in range(num_clusters):
                cluster_size = patches_per_cluster
                if i == num_clusters - 1:  # Last cluster gets remaining patches
                    cluster_size = num_patches - len(coordinates)

                # Generate coordinates around cluster center
                cluster_coords = self.rng.normal(cluster_centers[i], 500, (cluster_size, 2))
                coordinates.extend(cluster_coords)

            coordinates = np.array(coordinates)
        else:
            # Generate uniform random coordinates for normal slides
            coordinates = self.rng.uniform(min_coord, max_coord, (num_patches, 2))

        # Ensure coordinates are within bounds
        coordinates = np.clip(coordinates, min_coord, max_coord)

        return coordinates.astype(np.int32)

    def _generate_attention_weights(self, num_patches: int, pattern: Dict[str, Any]) -> np.ndarray:
        """Generate attention weights for patches.

        Args:
            num_patches: Number of patches
            pattern: Slide pattern parameters

        Returns:
            Attention weights array of shape (num_patches,)
        """
        if pattern["attention_sparsity"] > 0.5:
            # Sparse attention for tumor slides (few important patches)
            attention_weights = self.rng.exponential(0.1, num_patches)

            # Make some patches highly attended
            high_attention_indices = self.rng.choice(
                num_patches, size=max(1, int(num_patches * 0.1)), replace=False
            )
            attention_weights[high_attention_indices] *= 10
        else:
            # Uniform attention for normal slides
            attention_weights = self.rng.uniform(0.5, 1.5, num_patches)

        # Normalize to sum to 1
        attention_weights = attention_weights / np.sum(attention_weights)

        return attention_weights.astype(np.float32)

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

        if corruption_type == "feature_corruption":
            # Corrupt feature vectors
            for slide in slides:
                if self.rng.random() < 0.2:  # Corrupt 20% of slides
                    noise = self.rng.normal(0, 5, slide["features"].shape)
                    slide["features"] = slide["features"] + noise

        elif corruption_type == "coordinate_mismatch":
            # Create coordinate-feature misalignment
            for slide in slides:
                if self.rng.random() < 0.1:  # Corrupt 10% of slides
                    # Shuffle coordinates
                    indices = np.arange(len(slide["coordinates"]))
                    self.rng.shuffle(indices)
                    slide["coordinates"] = slide["coordinates"][indices]

        elif corruption_type == "missing_patches":
            # Remove some patches randomly
            for slide in slides:
                if self.rng.random() < 0.15:  # Corrupt 15% of slides
                    num_to_remove = max(1, slide["num_patches"] // 10)
                    keep_indices = self.rng.choice(
                        slide["num_patches"],
                        size=slide["num_patches"] - num_to_remove,
                        replace=False,
                    )

                    slide["features"] = slide["features"][keep_indices]
                    slide["coordinates"] = slide["coordinates"][keep_indices]
                    slide["attention_weights"] = slide["attention_weights"][keep_indices]
                    slide["num_patches"] = len(keep_indices)

        elif corruption_type == "dimension_mismatch":
            # Create slides with wrong feature dimensions
            for slide in slides:
                if self.rng.random() < 0.05:  # Corrupt 5% of slides
                    wrong_dim = self.rng.choice([1024, 512, 4096])  # Wrong dimensions
                    slide["features"] = self.rng.normal(0, 1, (slide["num_patches"], wrong_dim))

        elif corruption_type == "invalid_labels":
            # Create invalid labels
            for slide in slides:
                if self.rng.random() < 0.1:  # Corrupt 10% of slides
                    slide["label"] = self.rng.choice([-1, 2, 3, 999])  # Invalid labels

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
            slide_index = samples["slide_index"]
            metadata = samples["metadata"]

            # Check basic structure
            if not isinstance(slides, list) or not isinstance(slide_index, list):
                return False

            if len(slides) != len(slide_index):
                return False

            # Validate each slide
            for i, slide in enumerate(slides):
                # Check required fields
                required_fields = [
                    "slide_id",
                    "patient_id",
                    "label",
                    "num_patches",
                    "features",
                    "coordinates",
                    "attention_weights",
                ]
                if not all(field in slide for field in required_fields):
                    return False

                # Check data types and shapes
                features = slide["features"]
                coordinates = slide["coordinates"]
                attention_weights = slide["attention_weights"]

                if not isinstance(features, np.ndarray) or features.dtype != np.float32:
                    return False

                if not isinstance(coordinates, np.ndarray) or coordinates.dtype != np.int32:
                    return False

                if (
                    not isinstance(attention_weights, np.ndarray)
                    or attention_weights.dtype != np.float32
                ):
                    return False

                # Check dimensions
                num_patches = slide["num_patches"]
                if features.shape[0] != num_patches or coordinates.shape[0] != num_patches:
                    return False

                if attention_weights.shape[0] != num_patches:
                    return False

                if features.shape[1] != metadata["feature_dim"]:
                    return False

                if coordinates.shape[1] != 2:
                    return False

                # Check value ranges
                if not np.all(np.isin(slide["label"], [0, 1])):
                    return False

                # Check attention weights sum to approximately 1
                if abs(np.sum(attention_weights) - 1.0) > 0.01:
                    return False

                # Validate slide index entry
                index_entry = slide_index[i]
                if index_entry["slide_id"] != slide["slide_id"]:
                    return False

                if index_entry["num_patches"] != slide["num_patches"]:
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

        # Create features directory
        features_dir = output_dir / "features"
        features_dir.mkdir(exist_ok=True)

        # Save individual slide features as HDF5 files
        for slide in dataset["slides"]:
            slide_path = features_dir / f"{slide['slide_id']}.h5"

            with h5py.File(slide_path, "w") as f:
                f.create_dataset("features", data=slide["features"], compression="gzip")
                f.create_dataset("coordinates", data=slide["coordinates"], compression="gzip")
                f.create_dataset(
                    "attention_weights", data=slide["attention_weights"], compression="gzip"
                )

                # Save slide metadata as attributes
                f.attrs["slide_id"] = slide["slide_id"]
                f.attrs["patient_id"] = slide["patient_id"]
                f.attrs["label"] = slide["label"]
                f.attrs["num_patches"] = slide["num_patches"]

        # Save slide index
        with open(output_dir / "slide_index.json", "w") as f:
            json.dump(dataset["slide_index"], f, indent=2)

        # Save dataset metadata
        with open(output_dir / "metadata.json", "w") as f:
            json.dump(dataset["metadata"], f, indent=2)

    def create_patient_splits(
        self, dataset: Dict[str, Any], split_ratios: Dict[str, float] = None
    ) -> Dict[str, Dict[str, Any]]:
        """Create patient-level train/val/test splits.

        Args:
            dataset: Generated dataset
            split_ratios: Split ratios {"train": 0.7, "val": 0.15, "test": 0.15}

        Returns:
            Dictionary with patient-level splits
        """
        if split_ratios is None:
            split_ratios = {"train": 0.7, "val": 0.15, "test": 0.15}

        # Get unique patients
        patients = list(set(slide["patient_id"] for slide in dataset["slides"]))
        self.rng.shuffle(patients)

        # Calculate split boundaries
        num_patients = len(patients)
        train_end = int(num_patients * split_ratios["train"])
        val_end = train_end + int(num_patients * split_ratios["val"])

        patient_splits = {
            "train": patients[:train_end],
            "val": patients[train_end:val_end],
            "test": patients[val_end:],
        }

        # Create splits
        splits = {}
        for split_name, split_patients in patient_splits.items():
            split_slides = [
                slide for slide in dataset["slides"] if slide["patient_id"] in split_patients
            ]

            split_index = [
                entry for entry in dataset["slide_index"] if entry["patient_id"] in split_patients
            ]

            splits[split_name] = {
                "slides": split_slides,
                "slide_index": split_index,
                "metadata": dataset["metadata"].copy(),
            }
            splits[split_name]["metadata"]["split"] = split_name
            splits[split_name]["metadata"]["num_slides"] = len(split_slides)
            splits[split_name]["metadata"]["num_patients"] = len(split_patients)

        return splits
