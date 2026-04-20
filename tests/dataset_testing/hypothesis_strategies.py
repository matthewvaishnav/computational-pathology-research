"""
Custom Hypothesis strategies for pathology data testing.

This module provides custom Hypothesis strategies for generating
valid test data for computational pathology datasets.
"""

from pathlib import Path
from typing import Any, Dict, List, Tuple

import h5py
import numpy as np
import torch
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays


# Basic data type strategies
@st.composite
def pcam_sample_strategy(draw):
    """Generate valid PCam sample data.

    Returns:
        Dictionary with 'image' (96x96x3 uint8) and 'label' (0 or 1)
    """
    # Generate realistic histopathology image
    image = draw(arrays(dtype=np.uint8, shape=(96, 96, 3), elements=st.integers(0, 255)))

    # Binary classification label
    label = draw(st.integers(0, 1))

    return {"image": image, "label": label}


@st.composite
def camelyon_slide_strategy(draw):
    """Generate valid CAMELYON slide metadata.

    Returns:
        Dictionary with slide features, coordinates, and metadata
    """
    # Number of patches per slide
    num_patches = draw(st.integers(10, 1000))

    # Feature vectors (typical ResNet features)
    features = draw(
        arrays(
            dtype=np.float32,
            shape=(num_patches, 2048),
            elements=st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False),
        )
    )

    # Patch coordinates
    coordinates = draw(
        arrays(dtype=np.int32, shape=(num_patches, 2), elements=st.integers(0, 50000))
    )

    # Slide metadata
    slide_id = draw(
        st.text(
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")), min_size=5, max_size=20
        )
    )

    patient_id = draw(
        st.text(
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")), min_size=3, max_size=10
        )
    )

    label = draw(st.integers(0, 1))

    return {
        "slide_id": slide_id,
        "patient_id": patient_id,
        "features": features,
        "coordinates": coordinates,
        "label": label,
        "num_patches": num_patches,
    }


@st.composite
def camelyon_spec_strategy(draw):
    """Generate valid CAMELYONSyntheticSpec for testing.

    Returns:
        CAMELYONSyntheticSpec instance
    """
    from tests.dataset_testing.synthetic.camelyon_generator import CAMELYONSyntheticSpec

    num_slides = draw(st.integers(1, 10))

    # Patches per slide range
    min_patches = draw(st.integers(5, 50))
    max_patches = draw(st.integers(min_patches + 1, min_patches + 100))

    # Feature dimension
    feature_dim = draw(st.sampled_from([512, 1024, 2048]))

    # Coordinate range
    min_coord = draw(st.integers(0, 1000))
    max_coord = draw(st.integers(min_coord + 1000, min_coord + 20000))

    # Label distribution
    label_0_prob = draw(st.floats(0.1, 0.9))
    label_1_prob = 1.0 - label_0_prob

    return CAMELYONSyntheticSpec(
        num_slides=num_slides,
        patches_per_slide_range=(min_patches, max_patches),
        feature_dim=feature_dim,
        coordinate_range=(min_coord, max_coord),
        label_distribution={0: label_0_prob, 1: label_1_prob},
    )


@st.composite
def multimodal_sample_strategy(draw):
    """Generate valid multimodal sample data.

    Returns:
        Dictionary with WSI features, genomic data, and clinical text
    """
    patient_id = draw(
        st.text(
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")), min_size=5, max_size=15
        )
    )

    # WSI features (variable length)
    num_patches = draw(st.integers(10, 500))
    wsi_features = draw(
        arrays(
            dtype=np.float32,
            shape=(num_patches, 2048),
            elements=st.floats(-5.0, 5.0, allow_nan=False, allow_infinity=False),
        )
    )

    # Genomic features (fixed length)
    genomic_features = draw(
        arrays(
            dtype=np.float32,
            shape=(1000,),
            elements=st.floats(-3.0, 3.0, allow_nan=False, allow_infinity=False),
        )
    )

    # Clinical text (tokenized)
    text_length = draw(st.integers(10, 100))
    clinical_text = draw(
        arrays(
            dtype=np.int32, shape=(text_length,), elements=st.integers(1, 30000)  # Vocabulary range
        )
    )

    # Labels
    label = draw(st.integers(0, 3))  # Multi-class

    # Modality availability (some may be missing)
    has_wsi = draw(st.booleans())
    has_genomic = draw(st.booleans())
    has_clinical = draw(st.booleans())

    # Ensure at least one modality is present
    if not (has_wsi or has_genomic or has_clinical):
        has_wsi = True

    sample = {
        "patient_id": patient_id,
        "label": label,
    }

    if has_wsi:
        sample["wsi_features"] = wsi_features
        sample["num_patches"] = num_patches

    if has_genomic:
        sample["genomic_features"] = genomic_features

    if has_clinical:
        sample["clinical_text"] = clinical_text

    return sample


@st.composite
def wsi_patch_strategy(draw):
    """Generate valid WSI patch data.

    Returns:
        Dictionary with patch image, coordinates, and metadata
    """
    # Patch size (common sizes in pathology)
    patch_size = draw(st.sampled_from([224, 256, 512, 1024]))

    # Patch image
    patch = draw(
        arrays(dtype=np.uint8, shape=(patch_size, patch_size, 3), elements=st.integers(0, 255))
    )

    # Coordinates in WSI
    x_coord = draw(st.integers(0, 100000))
    y_coord = draw(st.integers(0, 100000))

    # Magnification level
    level = draw(st.integers(0, 4))

    # Tissue percentage (0-100%)
    tissue_percentage = draw(st.floats(0.0, 100.0))

    return {
        "patch": patch,
        "coordinates": (x_coord, y_coord),
        "level": level,
        "patch_size": patch_size,
        "tissue_percentage": tissue_percentage,
    }


@st.composite
def batch_strategy(draw, sample_strategy, min_batch_size=1, max_batch_size=32):
    """Generate batches of samples using a given sample strategy.

    Args:
        sample_strategy: Hypothesis strategy for individual samples
        min_batch_size: Minimum batch size
        max_batch_size: Maximum batch size

    Returns:
        List of samples forming a batch
    """
    batch_size = draw(st.integers(min_batch_size, max_batch_size))
    batch = []

    for _ in range(batch_size):
        sample = draw(sample_strategy)
        batch.append(sample)

    return batch


@st.composite
def tensor_strategy(draw, shape: Tuple[int, ...], dtype=torch.float32):
    """Generate PyTorch tensors with specified shape and dtype.

    Args:
        shape: Tensor shape
        dtype: PyTorch data type

    Returns:
        PyTorch tensor
    """
    if dtype == torch.float32:
        elements = st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False)
    elif dtype == torch.int64:
        elements = st.integers(-1000, 1000)
    elif dtype == torch.bool:
        elements = st.booleans()
    else:
        elements = st.floats(-10.0, 10.0, allow_nan=False, allow_infinity=False)

    array = draw(
        arrays(
            dtype=dtype.numpy() if hasattr(dtype, "numpy") else np.float32,
            shape=shape,
            elements=elements,
        )
    )

    return torch.from_numpy(array)


@st.composite
def configuration_strategy(draw):
    """Generate valid dataset configuration parameters.

    Returns:
        Dictionary with configuration parameters
    """
    config = {
        "batch_size": draw(st.integers(1, 64)),
        "num_workers": draw(st.integers(0, 8)),
        "shuffle": draw(st.booleans()),
        "pin_memory": draw(st.booleans()),
        "drop_last": draw(st.booleans()),
    }

    # Data augmentation parameters
    if draw(st.booleans()):
        config["augmentation"] = {
            "horizontal_flip": draw(st.booleans()),
            "vertical_flip": draw(st.booleans()),
            "rotation": draw(st.floats(0.0, 360.0)),
            "color_jitter": {
                "brightness": draw(st.floats(0.0, 0.5)),
                "contrast": draw(st.floats(0.0, 0.5)),
                "saturation": draw(st.floats(0.0, 0.5)),
                "hue": draw(st.floats(0.0, 0.1)),
            },
        }

    # Normalization parameters
    config["normalization"] = {
        "mean": [draw(st.floats(0.0, 1.0)), draw(st.floats(0.0, 1.0)), draw(st.floats(0.0, 1.0))],
        "std": [draw(st.floats(0.1, 1.0)), draw(st.floats(0.1, 1.0)), draw(st.floats(0.1, 1.0))],
    }

    return config


@st.composite
def file_path_strategy(draw, temp_dir: Path):
    """Generate valid file paths for testing.

    Args:
        temp_dir: Temporary directory for creating files

    Returns:
        Path object
    """
    filename = draw(
        st.text(
            alphabet=st.characters(whitelist_categories=("Lu", "Ll", "Nd")), min_size=5, max_size=20
        )
    )

    extension = draw(st.sampled_from([".h5", ".npy", ".pt", ".pth", ".json", ".csv"]))

    return temp_dir / f"{filename}{extension}"


# Composite strategies for complex scenarios
@st.composite
def dataset_corruption_scenario(draw):
    """Generate dataset corruption scenarios for error testing.

    Returns:
        Dictionary describing corruption scenario
    """
    corruption_type = draw(
        st.sampled_from(
            [
                "file_truncation",
                "random_bytes",
                "header_corruption",
                "permission_denied",
                "missing_file",
                "network_timeout",
            ]
        )
    )

    corruption_rate = draw(st.floats(0.01, 0.5))  # 1-50% corruption

    affected_files = draw(st.lists(st.text(min_size=5, max_size=20), min_size=1, max_size=10))

    return {
        "corruption_type": corruption_type,
        "corruption_rate": corruption_rate,
        "affected_files": affected_files,
    }


@st.composite
def performance_scenario(draw):
    """Generate performance testing scenarios.

    Returns:
        Dictionary with performance test parameters
    """
    dataset_size = draw(st.integers(100, 10000))
    batch_size = draw(st.integers(1, 128))
    num_workers = draw(st.integers(0, 8))

    # Memory constraints
    memory_limit_mb = draw(st.integers(512, 8192))

    # Expected thresholds
    max_loading_time = draw(st.floats(1.0, 30.0))
    min_throughput = draw(st.floats(1.0, 1000.0))

    return {
        "dataset_size": dataset_size,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "memory_limit_mb": memory_limit_mb,
        "max_loading_time_seconds": max_loading_time,
        "min_throughput_samples_per_second": min_throughput,
    }


# Property test utilities
class PropertyTestBase:
    """Base class for property-based tests with common utilities."""

    @staticmethod
    def create_temp_h5_file(data: Dict[str, np.ndarray], temp_dir: Path) -> Path:
        """Create temporary HDF5 file with given data.

        Args:
            data: Dictionary of arrays to store
            temp_dir: Temporary directory

        Returns:
            Path to created HDF5 file
        """
        h5_path = temp_dir / "temp_data.h5"

        with h5py.File(h5_path, "w") as f:
            for key, array in data.items():
                f.create_dataset(key, data=array, compression="gzip")

        return h5_path

    @staticmethod
    def validate_tensor_properties(
        tensor: torch.Tensor, expected_shape: Tuple[int, ...] = None
    ) -> bool:
        """Validate basic tensor properties.

        Args:
            tensor: Tensor to validate
            expected_shape: Expected tensor shape (optional)

        Returns:
            True if tensor is valid
        """
        if not isinstance(tensor, torch.Tensor):
            return False

        if torch.isnan(tensor).any() or torch.isinf(tensor).any():
            return False

        if expected_shape and tensor.shape != expected_shape:
            return False

        return True

    @staticmethod
    def validate_batch_consistency(batch: List[Dict[str, Any]]) -> bool:
        """Validate that batch samples are consistent.

        Args:
            batch: List of sample dictionaries

        Returns:
            True if batch is consistent
        """
        if not batch:
            return False

        # Check that all samples have same keys
        first_keys = set(batch[0].keys())
        for sample in batch[1:]:
            if set(sample.keys()) != first_keys:
                return False

        # Check tensor shapes are compatible for batching
        for key in first_keys:
            if isinstance(batch[0][key], (torch.Tensor, np.ndarray)):
                first_shape = batch[0][key].shape
                for sample in batch[1:]:
                    if sample[key].shape[1:] != first_shape[1:]:  # Allow different batch dim
                        return False

        return True


# Additional strategies for multimodal testing
@st.composite
def multimodal_config_strategy(draw):
    """Generate valid multimodal dataset configuration."""
    return {
        "wsi_enabled": draw(st.booleans()),
        "genomic_enabled": draw(st.booleans()),
        "clinical_text_enabled": draw(st.booleans()),
        "wsi_feature_dim": draw(st.integers(min_value=512, max_value=4096)),
        "genomic_feature_dim": draw(st.integers(min_value=100, max_value=2000)),
        "max_text_length": draw(st.integers(min_value=50, max_value=200)),
    }


@st.composite
def patient_count_strategy(draw):
    """Generate valid patient count for testing."""
    return draw(st.integers(min_value=2, max_value=10))


@st.composite
def feature_dimension_strategy(draw, min_dim=100, max_dim=2000):
    """Generate valid feature dimensions."""
    return draw(st.integers(min_value=min_dim, max_value=max_dim))


# OpenSlide-specific strategies
@st.composite
def wsi_coordinates_strategy(draw):
    """Generate valid WSI coordinates."""
    x = draw(st.integers(min_value=0, max_value=50000))
    y = draw(st.integers(min_value=0, max_value=50000))
    return (x, y)


@st.composite
def patch_size_strategy(draw):
    """Generate valid patch sizes for WSI extraction."""
    return draw(st.sampled_from([128, 224, 256, 512, 1024]))


@st.composite
def pyramid_level_strategy(draw):
    """Generate valid pyramid levels."""
    return draw(st.integers(min_value=0, max_value=4))
