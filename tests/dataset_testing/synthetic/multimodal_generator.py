"""
Multimodal synthetic data generator for testing.

This module provides synthetic multimodal data generation combining
WSI features, genomic data, and clinical text for comprehensive testing.
"""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np

from tests.dataset_testing.base_interfaces import DatasetGenerator


@dataclass
class MultimodalSyntheticSpec:
    """Specification for synthetic multimodal data generation."""

    num_patients: int
    wsi_feature_dim: int = 2048
    genomic_feature_dim: int = 1000
    clinical_text_length_range: Tuple[int, int] = (10, 100)
    missing_modality_probability: float = 0.2
    label_distribution: Dict[int, float] = field(
        default_factory=lambda: {0: 0.3, 1: 0.25, 2: 0.25, 3: 0.2}
    )


class MultimodalSyntheticGenerator(DatasetGenerator):
    """Synthetic multimodal data generator for cross-modal testing."""

    def __init__(self, random_seed: int = 42):
        """Initialize multimodal synthetic generator.

        Args:
            random_seed: Random seed for reproducible generation
        """
        self.random_seed = random_seed
        self.rng = np.random.RandomState(random_seed)

        # Modality-specific statistics
        self.modality_stats = {
            "wsi": {
                "feature_mean": 0.0,
                "feature_std": 1.0,
                "patches_per_patient_range": (50, 800),
            },
            "genomic": {
                "feature_mean": 0.0,
                "feature_std": 2.0,
                "sparsity": 0.7,  # 70% of features are zero/low
            },
            "clinical": {
                "vocab_size": 30000,
                "common_tokens": list(range(1, 1000)),  # Most frequent tokens
                "rare_tokens": list(range(1000, 30000)),
            },
        }

        # Cross-modal correlation patterns
        self.correlation_patterns = self._create_correlation_patterns()

    def _create_correlation_patterns(self) -> Dict[str, Dict[str, Any]]:
        """Create cross-modal correlation patterns for different disease types."""
        patterns = {}

        # Pattern for class 0 (e.g., normal/benign)
        patterns[0] = {
            "wsi_genomic_correlation": 0.1,  # Low correlation
            "wsi_clinical_correlation": 0.2,
            "genomic_clinical_correlation": 0.15,
            "modality_availability": {"wsi": 0.95, "genomic": 0.8, "clinical": 0.9},
        }

        # Pattern for class 1 (e.g., low-grade malignancy)
        patterns[1] = {
            "wsi_genomic_correlation": 0.4,
            "wsi_clinical_correlation": 0.5,
            "genomic_clinical_correlation": 0.3,
            "modality_availability": {"wsi": 0.98, "genomic": 0.85, "clinical": 0.95},
        }

        # Pattern for class 2 (e.g., high-grade malignancy)
        patterns[2] = {
            "wsi_genomic_correlation": 0.7,
            "wsi_clinical_correlation": 0.6,
            "genomic_clinical_correlation": 0.5,
            "modality_availability": {"wsi": 0.99, "genomic": 0.9, "clinical": 0.98},
        }

        # Pattern for class 3 (e.g., metastatic)
        patterns[3] = {
            "wsi_genomic_correlation": 0.8,
            "wsi_clinical_correlation": 0.8,
            "genomic_clinical_correlation": 0.7,
            "modality_availability": {"wsi": 1.0, "genomic": 0.95, "clinical": 1.0},
        }

        return patterns

    def generate_samples(self, num_patients: int, **kwargs) -> Dict[str, Any]:
        """Generate synthetic multimodal patient data.

        Args:
            num_patients: Number of patients to generate
            **kwargs: Additional parameters (spec, output_dir, etc.)

        Returns:
            Dictionary containing generated patient data and metadata
        """
        spec = kwargs.get("spec", MultimodalSyntheticSpec(num_patients=num_patients))
        output_dir = kwargs.get("output_dir", None)

        patients = []

        for i in range(num_patients):
            patient_data = self._generate_patient(patient_id=f"patient_{i:05d}", spec=spec)
            patients.append(patient_data)

        # Create dataset metadata
        metadata = {
            "num_patients": num_patients,
            "wsi_feature_dim": spec.wsi_feature_dim,
            "genomic_feature_dim": spec.genomic_feature_dim,
            "clinical_text_length_range": spec.clinical_text_length_range,
            "missing_modality_probability": spec.missing_modality_probability,
            "label_distribution": spec.label_distribution,
            "generator_seed": self.random_seed,
            "modality_stats": self.modality_stats,
        }

        dataset = {
            "patients": patients,
            "metadata": metadata,
        }

        # Save to files if output directory provided
        if output_dir:
            self._save_dataset(dataset, Path(output_dir))

        return dataset

    def _generate_patient(self, patient_id: str, spec: MultimodalSyntheticSpec) -> Dict[str, Any]:
        """Generate a single synthetic patient with multimodal data.

        Args:
            patient_id: Unique patient identifier
            spec: Generation specification

        Returns:
            Dictionary containing patient data
        """
        # Generate patient label
        label_probs = list(spec.label_distribution.values())
        label_values = list(spec.label_distribution.keys())
        label = self.rng.choice(label_values, p=label_probs)

        # Get correlation pattern for this label
        pattern = self.correlation_patterns[label]

        # Determine which modalities are available
        available_modalities = self._determine_available_modalities(pattern, spec)

        patient_data = {
            "patient_id": patient_id,
            "label": label,
            "available_modalities": available_modalities,
        }

        # Generate base latent representation (shared across modalities)
        latent_dim = 50
        base_latent = self.rng.normal(0, 1, latent_dim)

        # Generate each modality
        if "wsi" in available_modalities:
            patient_data["wsi_data"] = self._generate_wsi_features(
                base_latent, label, spec, pattern
            )

        if "genomic" in available_modalities:
            patient_data["genomic_data"] = self._generate_genomic_features(
                base_latent, label, spec, pattern
            )

        if "clinical" in available_modalities:
            patient_data["clinical_data"] = self._generate_clinical_text(
                base_latent, label, spec, pattern
            )

        return patient_data

    def _determine_available_modalities(
        self, pattern: Dict[str, Any], spec: MultimodalSyntheticSpec
    ) -> List[str]:
        """Determine which modalities are available for a patient.

        Args:
            pattern: Correlation pattern for patient's label
            spec: Generation specification

        Returns:
            List of available modality names
        """
        modalities = ["wsi", "genomic", "clinical"]
        available = []

        for modality in modalities:
            availability_prob = pattern["modality_availability"][modality]
            # Apply global missing modality probability
            availability_prob *= 1 - spec.missing_modality_probability

            if self.rng.random() < availability_prob:
                available.append(modality)

        # Ensure at least one modality is available
        if not available:
            available = [self.rng.choice(modalities)]

        return available

    def _generate_wsi_features(
        self,
        base_latent: np.ndarray,
        label: int,
        spec: MultimodalSyntheticSpec,
        pattern: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate WSI features for a patient.

        Args:
            base_latent: Shared latent representation
            label: Patient label
            spec: Generation specification
            pattern: Correlation pattern

        Returns:
            Dictionary with WSI data
        """
        # Number of patches
        min_patches, max_patches = self.modality_stats["wsi"]["patches_per_patient_range"]
        num_patches = self.rng.randint(min_patches, max_patches)

        # Generate patch features
        features = self.rng.normal(
            self.modality_stats["wsi"]["feature_mean"],
            self.modality_stats["wsi"]["feature_std"],
            (num_patches, spec.wsi_feature_dim),
        )

        # Add correlation with base latent representation
        correlation_strength = pattern["wsi_genomic_correlation"]
        if correlation_strength > 0:
            # Project latent to feature space
            projection_matrix = self.rng.normal(0, 1, (len(base_latent), spec.wsi_feature_dim))
            # Create a single correlated feature vector
            correlated_feature_vector = np.dot(base_latent, projection_matrix)

            # Mix with random features
            for i in range(num_patches):
                features[i] = (1 - correlation_strength) * features[
                    i
                ] + correlation_strength * correlated_feature_vector

        # Generate patch coordinates
        coordinates = self.rng.uniform(0, 50000, (num_patches, 2)).astype(np.int32)

        # Generate attention weights (for attention-based models)
        if label > 0:  # Disease cases have more focused attention
            attention_weights = self.rng.exponential(0.2, num_patches)
            # Make some patches highly attended
            high_attention_indices = self.rng.choice(
                num_patches, size=max(1, num_patches // 10), replace=False
            )
            attention_weights[high_attention_indices] *= 5
        else:
            attention_weights = self.rng.uniform(0.5, 1.5, num_patches)

        attention_weights = attention_weights / np.sum(attention_weights)

        return {
            "features": features.astype(np.float32),
            "coordinates": coordinates,
            "attention_weights": attention_weights.astype(np.float32),
            "num_patches": num_patches,
        }

    def _generate_genomic_features(
        self,
        base_latent: np.ndarray,
        label: int,
        spec: MultimodalSyntheticSpec,
        pattern: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate genomic features for a patient.

        Args:
            base_latent: Shared latent representation
            label: Patient label
            spec: Generation specification
            pattern: Correlation pattern

        Returns:
            Dictionary with genomic data
        """
        # Generate base genomic features
        features = self.rng.normal(
            self.modality_stats["genomic"]["feature_mean"],
            self.modality_stats["genomic"]["feature_std"],
            spec.genomic_feature_dim,
        )

        # Add sparsity (many genes have low/zero expression)
        sparsity = self.modality_stats["genomic"]["sparsity"]
        sparse_mask = self.rng.random(spec.genomic_feature_dim) < sparsity
        features[sparse_mask] *= 0.1  # Reduce expression for sparse genes

        # Add correlation with base latent representation
        correlation_strength = pattern["wsi_genomic_correlation"]
        if correlation_strength > 0:
            # Project latent to genomic space
            projection_matrix = self.rng.normal(0, 1, (len(base_latent), spec.genomic_feature_dim))
            correlated_features = base_latent @ projection_matrix

            # Mix with random features
            features = (
                1 - correlation_strength
            ) * features + correlation_strength * correlated_features

        # Add disease-specific mutations/alterations
        if label > 0:
            # Simulate disease-associated gene expression changes
            num_altered_genes = self.rng.randint(10, 100)
            altered_indices = self.rng.choice(
                spec.genomic_feature_dim, size=num_altered_genes, replace=False
            )

            # Some genes up-regulated, some down-regulated
            for idx in altered_indices:
                if self.rng.random() < 0.5:
                    features[idx] += self.rng.exponential(2)  # Up-regulation
                else:
                    features[idx] -= self.rng.exponential(1)  # Down-regulation

        # Generate gene names (for interpretability)
        gene_names = [f"GENE_{i:05d}" for i in range(spec.genomic_feature_dim)]

        return {
            "features": features.astype(np.float32),
            "gene_names": gene_names,
            "num_genes": spec.genomic_feature_dim,
        }

    def _generate_clinical_text(
        self,
        base_latent: np.ndarray,
        label: int,
        spec: MultimodalSyntheticSpec,
        pattern: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate clinical text for a patient.

        Args:
            base_latent: Shared latent representation
            label: Patient label
            spec: Generation specification
            pattern: Correlation pattern

        Returns:
            Dictionary with clinical text data
        """
        # Generate text length
        min_length, max_length = spec.clinical_text_length_range
        text_length = self.rng.randint(min_length, max_length)

        # Generate token IDs
        common_tokens = self.modality_stats["clinical"]["common_tokens"]
        rare_tokens = self.modality_stats["clinical"]["rare_tokens"]

        # Mix of common and rare tokens based on disease severity
        if label == 0:
            # Normal cases use more common medical terms
            token_pool = common_tokens + rare_tokens[:500]
            common_prob = 0.8
        else:
            # Disease cases use more specialized terminology
            token_pool = common_tokens + rare_tokens
            common_prob = 0.6

        tokens = []
        for _ in range(text_length):
            if self.rng.random() < common_prob:
                token = self.rng.choice(common_tokens)
            else:
                token = self.rng.choice(rare_tokens)
            tokens.append(token)

        # Add disease-specific patterns
        if label > 0:
            # Insert disease-related keywords
            disease_keywords = {
                1: [100, 101, 102],  # Low-grade disease keywords
                2: [200, 201, 202, 203],  # High-grade disease keywords
                3: [300, 301, 302, 303, 304],  # Metastatic disease keywords
            }

            if label in disease_keywords:
                # Insert 1-3 disease keywords
                num_keywords = self.rng.randint(1, 4)
                keyword_positions = self.rng.choice(text_length, size=num_keywords, replace=False)

                for pos in keyword_positions:
                    tokens[pos] = self.rng.choice(disease_keywords[label])

        # Add correlation with base latent (through token selection bias)
        correlation_strength = pattern["wsi_clinical_correlation"]
        if correlation_strength > 0:
            # Use latent representation to bias token selection
            latent_influence = np.mean(base_latent)
            if latent_influence > 0:
                # Bias towards higher token IDs
                bias_tokens = [t for t in tokens if t > np.median(token_pool)]
                if bias_tokens:
                    num_biased = int(text_length * correlation_strength * 0.3)
                    bias_positions = self.rng.choice(text_length, size=num_biased, replace=False)
                    for pos in bias_positions:
                        tokens[pos] = self.rng.choice(bias_tokens)

        # Generate attention mask (1 for real tokens, 0 for padding)
        attention_mask = np.ones(text_length, dtype=np.int32)

        return {
            "token_ids": np.array(tokens, dtype=np.int32),
            "attention_mask": attention_mask,
            "text_length": text_length,
        }

    def corrupt_samples(self, samples: Dict[str, Any], corruption_type: str) -> Dict[str, Any]:
        """Introduce controlled corruption for testing error handling.

        Args:
            samples: Original samples to corrupt
            corruption_type: Type of corruption to introduce

        Returns:
            Dictionary containing corrupted samples
        """
        corrupted_samples = samples.copy()
        patients = [patient.copy() for patient in corrupted_samples["patients"]]

        if corruption_type == "modality_mismatch":
            # Create patients with mismatched modality dimensions
            for patient in patients:
                if self.rng.random() < 0.1:  # Corrupt 10% of patients
                    if "wsi_data" in patient:
                        wrong_dim = self.rng.choice([1024, 512, 4096])
                        num_patches = patient["wsi_data"]["num_patches"]
                        patient["wsi_data"]["features"] = self.rng.normal(
                            0, 1, (num_patches, wrong_dim)
                        )

        elif corruption_type == "missing_required_modality":
            # Remove all modalities from some patients
            for patient in patients:
                if self.rng.random() < 0.05:  # Corrupt 5% of patients
                    modalities_to_remove = ["wsi_data", "genomic_data", "clinical_data"]
                    for modality in modalities_to_remove:
                        if modality in patient:
                            del patient[modality]
                    patient["available_modalities"] = []

        elif corruption_type == "patient_id_mismatch":
            # Create duplicate or invalid patient IDs
            for i, patient in enumerate(patients):
                if self.rng.random() < 0.1:  # Corrupt 10% of patients
                    if i > 0 and self.rng.random() < 0.5:
                        # Duplicate previous patient ID
                        patient["patient_id"] = patients[i - 1]["patient_id"]
                    else:
                        # Invalid patient ID
                        patient["patient_id"] = ""

        elif corruption_type == "cross_modal_inconsistency":
            # Create inconsistent information across modalities
            for patient in patients:
                if self.rng.random() < 0.15:  # Corrupt 15% of patients
                    # Change label but keep modality data unchanged (creates inconsistency)
                    original_label = patient["label"]
                    new_label = self.rng.choice(
                        [label for label in [0, 1, 2, 3] if label != original_label]
                    )
                    patient["label"] = new_label

        elif corruption_type == "invalid_tensor_shapes":
            # Create tensors with invalid shapes
            for patient in patients:
                if self.rng.random() < 0.08:  # Corrupt 8% of patients
                    if "genomic_data" in patient:
                        # Wrong genomic feature dimension
                        wrong_shape = (self.rng.randint(500, 2000),)
                        patient["genomic_data"]["features"] = self.rng.normal(0, 1, wrong_shape)

        corrupted_samples["patients"] = patients
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
            patients = samples["patients"]
            metadata = samples["metadata"]

            # Check basic structure
            if not isinstance(patients, list):
                return False

            # Validate each patient
            for patient in patients:
                # Check required fields
                required_fields = ["patient_id", "label", "available_modalities"]
                if not all(field in patient for field in required_fields):
                    return False

                # Check patient ID is valid
                if not isinstance(patient["patient_id"], str) or not patient["patient_id"]:
                    return False

                # Check label is valid
                if patient["label"] not in [0, 1, 2, 3]:
                    return False

                # Check available modalities
                available_modalities = patient["available_modalities"]
                if not isinstance(available_modalities, list):
                    return False

                # Validate each available modality
                for modality in available_modalities:
                    modality_key = f"{modality}_data"
                    if modality_key not in patient:
                        return False

                    modality_data = patient[modality_key]

                    if modality == "wsi":
                        # Validate WSI data
                        required_wsi_fields = [
                            "features",
                            "coordinates",
                            "attention_weights",
                            "num_patches",
                        ]
                        if not all(field in modality_data for field in required_wsi_fields):
                            return False

                        features = modality_data["features"]
                        if not isinstance(features, np.ndarray) or features.dtype != np.float32:
                            return False

                        if features.shape[1] != metadata["wsi_feature_dim"]:
                            return False

                        if features.shape[0] != modality_data["num_patches"]:
                            return False

                    elif modality == "genomic":
                        # Validate genomic data
                        required_genomic_fields = ["features", "gene_names", "num_genes"]
                        if not all(field in modality_data for field in required_genomic_fields):
                            return False

                        features = modality_data["features"]
                        if not isinstance(features, np.ndarray) or features.dtype != np.float32:
                            return False

                        if len(features) != metadata["genomic_feature_dim"]:
                            return False

                    elif modality == "clinical":
                        # Validate clinical data
                        required_clinical_fields = ["token_ids", "attention_mask", "text_length"]
                        if not all(field in modality_data for field in required_clinical_fields):
                            return False

                        token_ids = modality_data["token_ids"]
                        if not isinstance(token_ids, np.ndarray) or token_ids.dtype != np.int32:
                            return False

                        if len(token_ids) != modality_data["text_length"]:
                            return False

                # Ensure at least one modality is available
                if not available_modalities:
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

        # Create modality-specific directories (matching MultimodalDataset expectations)
        wsi_dir = output_dir / "wsi_features"
        genomic_dir = output_dir / "genomic"  # Note: not "genomic_features"
        clinical_dir = output_dir / "clinical_text"

        wsi_dir.mkdir(exist_ok=True)
        genomic_dir.mkdir(exist_ok=True)
        clinical_dir.mkdir(exist_ok=True)

        # Save patient data and create samples list
        samples = []

        for patient in dataset["patients"]:
            patient_id = patient["patient_id"]

            # Initialize sample entry
            sample_entry = {
                "patient_id": patient_id,
                "label": int(patient["label"]),  # Convert numpy int64 to Python int
            }

            # Save WSI data
            if "wsi_data" in patient:
                wsi_filename = f"{patient_id}_wsi.h5"
                wsi_path = wsi_dir / wsi_filename
                with h5py.File(wsi_path, "w") as f:
                    wsi_data = patient["wsi_data"]
                    f.create_dataset("features", data=wsi_data["features"], compression="gzip")
                    f.create_dataset(
                        "coordinates", data=wsi_data["coordinates"], compression="gzip"
                    )
                    f.create_dataset(
                        "attention_weights", data=wsi_data["attention_weights"], compression="gzip"
                    )
                    f.attrs["num_patches"] = wsi_data["num_patches"]
                sample_entry["wsi_file"] = wsi_filename
            else:
                sample_entry["wsi_file"] = None

            # Save genomic data
            if "genomic_data" in patient:
                genomic_filename = f"{patient_id}_genomic.npy"
                genomic_path = genomic_dir / genomic_filename
                np.save(genomic_path, patient["genomic_data"]["features"])
                sample_entry["genomic_file"] = genomic_filename
            else:
                sample_entry["genomic_file"] = None

            # Save clinical text
            if "clinical_data" in patient:
                clinical_filename = f"{patient_id}_clinical.npy"
                clinical_path = clinical_dir / clinical_filename
                clinical_data = patient["clinical_data"]
                # Save just the token_ids as expected by MultimodalDataset
                np.save(clinical_path, clinical_data["token_ids"])
                sample_entry["clinical_file"] = clinical_filename
            else:
                sample_entry["clinical_file"] = None

            samples.append(sample_entry)

        # Create train/val/test splits (for now, put everything in train)
        train_samples = samples
        val_samples = []
        test_samples = []

        # Save split metadata files
        for split_name, split_samples in [
            ("train", train_samples),
            ("val", val_samples),
            ("test", test_samples),
        ]:
            metadata = {
                "samples": split_samples,
                "num_samples": len(split_samples),
                "split": split_name,
            }

            metadata_path = output_dir / f"{split_name}_metadata.json"
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

        # Save overall dataset metadata
        with open(output_dir / "dataset_metadata.json", "w") as f:
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

        patients = dataset["patients"]
        num_patients = len(patients)

        # Shuffle patients
        indices = np.arange(num_patients)
        self.rng.shuffle(indices)

        # Calculate split boundaries
        train_end = int(num_patients * split_ratios["train"])
        val_end = train_end + int(num_patients * split_ratios["val"])

        split_indices = {
            "train": indices[:train_end],
            "val": indices[train_end:val_end],
            "test": indices[val_end:],
        }

        # Create splits
        splits = {}
        for split_name, split_idx in split_indices.items():
            split_patients = [patients[i] for i in split_idx]

            splits[split_name] = {
                "patients": split_patients,
                "metadata": dataset["metadata"].copy(),
            }
            splits[split_name]["metadata"]["split"] = split_name
            splits[split_name]["metadata"]["num_patients"] = len(split_patients)

        return splits
