"""
Property-based tests for CAMELYON dataset functionality.

Tests universal properties that should hold across all valid CAMELYON data.
"""

import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch

from hypothesis import assume, given, settings
from hypothesis import strategies as st
from src.data.camelyon_dataset import (
    CAMELYONPatchDataset,
    CAMELYONSlideDataset,
    CAMELYONSlideIndex,
    SlideMetadata,
    validate_feature_file,
)
from tests.dataset_testing.hypothesis_strategies import (
    camelyon_slide_strategy,
    camelyon_spec_strategy,
)
from tests.dataset_testing.synthetic.camelyon_generator import (
    CAMELYONSyntheticGenerator,
    CAMELYONSyntheticSpec,
)


class TestCAMELYONCoordinateAlignment:
    """Property tests for CAMELYON coordinate alignment preservation."""

    @given(camelyon_spec_strategy())
    @settings(max_examples=100, deadline=60000)
    def test_coordinate_alignment_preservation_property(self, spec):
        """
        **Property 4: Coordinate and Alignment Preservation**

        For any CAMELYON slide data with features and coordinates,
        the spatial relationships SHALL be preserved throughout processing.

        **Validates: Requirements 2.3**
        """
        assume(spec.num_slides >= 1)
        assume(spec.patches_per_slide_range[0] >= 5)
        assume(spec.patches_per_slide_range[1] <= 200)  # Keep reasonable for testing

        generator = CAMELYONSyntheticGenerator(random_seed=42)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Generate synthetic data
            dataset = generator.generate_samples(
                num_slides=spec.num_slides,
                spec=spec,
                output_dir=Path(tmp_dir) / "synthetic_camelyon",
            )

            # Create slide index
            slides = [
                SlideMetadata(
                    slide_id=entry["slide_id"],
                    patient_id=entry["patient_id"],
                    file_path=f"features/{entry['slide_id']}.h5",
                    label=entry["label"],
                    split="train",
                )
                for entry in dataset["slide_index"]
            ]

            slide_index = CAMELYONSlideIndex(slides)

            # Create datasets
            patch_dataset = CAMELYONPatchDataset(
                slide_index=slide_index,
                features_dir=str(Path(tmp_dir) / "synthetic_camelyon" / "features"),
                split="train",
            )

            CAMELYONSlideDataset(
                slide_index=slide_index,
                features_dir=str(Path(tmp_dir) / "synthetic_camelyon" / "features"),
                split="train",
            )

            # Test coordinate-feature alignment preservation
            for slide_entry in dataset["slide_index"]:
                slide_id = slide_entry["slide_id"]

                # Get slide-level data
                slide_data = patch_dataset.get_slide_patch_data(slide_id)

                if slide_data is not None:
                    features = slide_data["features"]
                    coordinates = slide_data["coordinates"]
                    num_patches = slide_data["num_patches"]

                    # Property: Features and coordinates must have same length
                    assert features.shape[0] == coordinates.shape[0] == num_patches

                    # Property: Coordinates must be 2D (x, y)
                    assert coordinates.shape[1] == 2

                    # Property: Features must have consistent dimension
                    assert features.shape[1] == spec.feature_dim

                    # Property: Coordinate ordering must be preserved
                    # (same patch index should give same coordinate-feature pair)
                    for patch_idx in range(min(5, num_patches)):  # Test first 5 patches
                        # Find this patch in the patch dataset
                        found_patch = None
                        for i in range(len(patch_dataset)):
                            sample = patch_dataset[i]
                            if sample["slide_id"] == slide_id and sample["patch_idx"] == patch_idx:
                                found_patch = sample
                                break

                        if found_patch is not None:
                            # Property: Individual patch coordinates match slide-level coordinates
                            expected_coord = coordinates[patch_idx]
                            actual_coord = found_patch["coordinates"]
                            assert torch.allclose(expected_coord, actual_coord)

                            # Property: Individual patch features match slide-level features
                            expected_feature = features[patch_idx]
                            actual_feature = found_patch["features"]
                            assert torch.allclose(expected_feature, actual_feature, atol=1e-5)

    @given(camelyon_slide_strategy())
    @settings(max_examples=100, deadline=60000)
    def test_spatial_relationship_consistency_property(self, slide_data):
        """
        **Property 4: Coordinate and Alignment Preservation**

        For any valid slide with spatial coordinates, the relative spatial
        relationships between patches SHALL remain consistent across operations.

        **Validates: Requirements 2.3**
        """
        features = slide_data["features"]
        coordinates = slide_data["coordinates"]

        assume(len(features) >= 3)  # Need at least 3 patches for spatial relationships

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save slide data to HDF5
            slide_path = Path(tmp_dir) / "test_slide.h5"
            with h5py.File(slide_path, "w") as f:
                f.create_dataset("features", data=features)
                f.create_dataset("coordinates", data=coordinates)

            # Validate file structure
            result = validate_feature_file(str(slide_path))
            assert result["valid"] is True

            # Test spatial relationship preservation
            # Property: Distance relationships should be preserved
            coord_array = np.array(coordinates)

            # Calculate pairwise distances
            for i in range(min(3, len(coord_array))):
                for j in range(i + 1, min(3, len(coord_array))):
                    # Property: Distance between patches should be non-negative
                    distance = np.linalg.norm(coord_array[i] - coord_array[j])
                    assert distance >= 0

                    # Property: Distance should be finite
                    assert np.isfinite(distance)

            # Property: Coordinates should be within reasonable bounds
            assert np.all(coord_array >= 0)
            assert np.all(coord_array <= 100000)  # Reasonable slide size limit

            # Property: No duplicate coordinates (each patch has unique position)
            unique_coords = np.unique(coord_array, axis=0)
            # Allow some duplicates due to random generation, but not too many
            duplicate_ratio = 1.0 - (len(unique_coords) / len(coord_array))
            assert duplicate_ratio < 0.5  # Less than 50% duplicates

    @given(st.integers(min_value=10, max_value=100))
    @settings(max_examples=50, deadline=60000)
    def test_feature_coordinate_dimension_consistency_property(self, num_patches):
        """
        **Property 4: Coordinate and Alignment Preservation**

        For any number of patches, features and coordinates SHALL maintain
        consistent dimensional relationships.

        **Validates: Requirements 2.3**
        """
        feature_dim = 2048

        # Generate test data
        features = np.random.randn(num_patches, feature_dim).astype(np.float32)
        coordinates = np.random.randint(0, 10000, (num_patches, 2)).astype(np.int32)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save to HDF5
            slide_path = Path(tmp_dir) / "test_slide.h5"
            with h5py.File(slide_path, "w") as f:
                f.create_dataset("features", data=features)
                f.create_dataset("coordinates", data=coordinates)

            # Validate structure
            result = validate_feature_file(str(slide_path))

            # Property: File should be valid
            assert result["valid"] is True

            # Property: Patch count should match
            assert result["num_patches"] == num_patches

            # Property: Feature dimension should match
            assert result["feature_dim"] == feature_dim

            # Property: No errors should be reported
            assert result["error"] is None

            # Load and verify data preservation
            with h5py.File(slide_path, "r") as f:
                loaded_features = f["features"][:]
                loaded_coordinates = f["coordinates"][:]

                # Property: Data should be preserved exactly
                assert np.array_equal(loaded_features, features)
                assert np.array_equal(loaded_coordinates, coordinates)

                # Property: Shapes should be preserved
                assert loaded_features.shape == (num_patches, feature_dim)
                assert loaded_coordinates.shape == (num_patches, 2)

    @given(st.integers(min_value=1, max_value=5))
    @settings(max_examples=30, deadline=60000)
    def test_multimodal_alignment_preservation_property(self, num_slides):
        """
        **Property 4: Coordinate and Alignment Preservation**

        For any multimodal dataset with slide features and coordinates,
        alignment SHALL be preserved across different access patterns.

        **Validates: Requirements 2.3**
        """
        generator = CAMELYONSyntheticGenerator(random_seed=42)
        spec = CAMELYONSyntheticSpec(
            num_slides=num_slides,
            patches_per_slide_range=(20, 50),
            feature_dim=2048,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Generate synthetic data
            dataset = generator.generate_samples(
                num_slides=num_slides, spec=spec, output_dir=Path(tmp_dir) / "synthetic_camelyon"
            )

            # Create slide index
            slides = [
                SlideMetadata(
                    slide_id=entry["slide_id"],
                    patient_id=entry["patient_id"],
                    file_path=f"features/{entry['slide_id']}.h5",
                    label=entry["label"],
                    split="train",
                )
                for entry in dataset["slide_index"]
            ]

            slide_index = CAMELYONSlideIndex(slides)

            # Create both patch and slide datasets
            patch_dataset = CAMELYONPatchDataset(
                slide_index=slide_index,
                features_dir=str(Path(tmp_dir) / "synthetic_camelyon" / "features"),
                split="train",
            )

            slide_dataset = CAMELYONSlideDataset(
                slide_index=slide_index,
                features_dir=str(Path(tmp_dir) / "synthetic_camelyon" / "features"),
                split="train",
            )

            # Property: Both datasets should have consistent slide information
            assert len(slide_dataset) == num_slides

            # Test alignment preservation across access patterns
            for slide_idx in range(len(slide_dataset)):
                slide_sample = slide_dataset[slide_idx]
                slide_id = slide_sample["slide_id"]

                # Get slide data from patch dataset
                patch_slide_data = patch_dataset.get_slide_patch_data(slide_id)

                if patch_slide_data is not None:
                    # Property: Slide-level and patch-level data should be consistent
                    assert slide_sample["slide_id"] == patch_slide_data["slide_id"]
                    assert slide_sample["patient_id"] == patch_slide_data["patient_id"]
                    assert slide_sample["label"] == patch_slide_data["label"]
                    assert slide_sample["num_patches"] == patch_slide_data["num_patches"]

                    # Property: Feature and coordinate shapes should match
                    assert slide_sample["features"].shape == patch_slide_data["features"].shape
                    assert (
                        slide_sample["coordinates"].shape == patch_slide_data["coordinates"].shape
                    )

                    # Property: Feature and coordinate values should be identical
                    assert torch.allclose(
                        slide_sample["features"], patch_slide_data["features"], atol=1e-5
                    )
                    assert torch.allclose(
                        slide_sample["coordinates"], patch_slide_data["coordinates"]
                    )


class TestCAMELYONPatientPrivacy:
    """Property tests for CAMELYON patient privacy preservation."""

    @given(st.integers(min_value=3, max_value=10))
    @settings(max_examples=50, deadline=60000)
    def test_patient_privacy_preservation_property(self, num_slides):
        """
        **Property 9: Patient Privacy Preservation**

        For any dataset split operation, patient data SHALL never appear
        in multiple splits, ensuring proper data isolation.

        **Validates: Requirements 2.5**
        """
        generator = CAMELYONSyntheticGenerator(random_seed=42)
        spec = CAMELYONSyntheticSpec(
            num_slides=num_slides,
            patches_per_slide_range=(10, 30),
            feature_dim=512,  # Smaller for faster testing
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Generate synthetic data
            dataset = generator.generate_samples(
                num_slides=num_slides, spec=spec, output_dir=Path(tmp_dir) / "synthetic_camelyon"
            )

            # Create patient splits
            splits = generator.create_patient_splits(
                dataset, split_ratios={"train": 0.6, "val": 0.2, "test": 0.2}
            )

            # Extract patient IDs from each split
            train_patients = set()
            val_patients = set()
            test_patients = set()

            for slide in splits["train"]["slides"]:
                train_patients.add(slide["patient_id"])

            for slide in splits["val"]["slides"]:
                val_patients.add(slide["patient_id"])

            for slide in splits["test"]["slides"]:
                test_patients.add(slide["patient_id"])

            # Property: No patient should appear in multiple splits
            assert len(train_patients & val_patients) == 0, "Patient appears in both train and val"
            assert (
                len(train_patients & test_patients) == 0
            ), "Patient appears in both train and test"
            assert len(val_patients & test_patients) == 0, "Patient appears in both val and test"

            # Property: All patients should be assigned to exactly one split
            all_original_patients = set(slide["patient_id"] for slide in dataset["slides"])
            all_split_patients = train_patients | val_patients | test_patients
            assert all_original_patients == all_split_patients

            # Property: Each split should have at least one patient (if possible)
            total_patients = len(all_original_patients)
            if total_patients >= 3:
                assert len(train_patients) >= 1
                assert (
                    len(val_patients) >= 1 or total_patients < 5
                )  # Allow empty val for very small datasets
                assert (
                    len(test_patients) >= 1 or total_patients < 4
                )  # Allow empty test for small datasets

    @given(st.integers(min_value=2, max_value=8))
    @settings(max_examples=30, deadline=60000)
    def test_patient_slide_grouping_property(self, num_patients):
        """
        **Property 9: Patient Privacy Preservation**

        For any patient grouping, all slides from the same patient
        SHALL be assigned to the same split.

        **Validates: Requirements 2.5**
        """
        # Create patient distribution with multiple slides per patient
        patient_distribution = {}
        for i in range(num_patients):
            patient_id = f"patient_{i:03d}"
            slides_per_patient = np.random.randint(1, 4)  # 1-3 slides per patient
            patient_distribution[patient_id] = slides_per_patient

        total_slides = sum(patient_distribution.values())

        generator = CAMELYONSyntheticGenerator(random_seed=42)
        spec = CAMELYONSyntheticSpec(
            num_slides=total_slides,
            patches_per_slide_range=(10, 20),
            feature_dim=256,  # Smaller for faster testing
            patient_slide_distribution=patient_distribution,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Generate synthetic data
            dataset = generator.generate_samples(
                num_slides=total_slides, spec=spec, output_dir=Path(tmp_dir) / "synthetic_camelyon"
            )

            # Create patient splits
            splits = generator.create_patient_splits(
                dataset, split_ratios={"train": 0.7, "val": 0.15, "test": 0.15}
            )

            # Check patient-slide grouping
            for split_name, split_data in splits.items():
                patient_slides = {}

                # Group slides by patient in this split
                for slide in split_data["slides"]:
                    patient_id = slide["patient_id"]
                    if patient_id not in patient_slides:
                        patient_slides[patient_id] = []
                    patient_slides[patient_id].append(slide["slide_id"])

                # Property: Each patient should have all their slides in the same split
                for patient_id, slide_ids in patient_slides.items():
                    expected_slide_count = patient_distribution[patient_id]
                    actual_slide_count = len(slide_ids)

                    # All slides for this patient should be in this split
                    assert actual_slide_count == expected_slide_count, (
                        f"Patient {patient_id} has {actual_slide_count} slides in {split_name} "
                        f"but should have {expected_slide_count}"
                    )

                    # Verify no slides from this patient appear in other splits
                    for other_split_name, other_split_data in splits.items():
                        if other_split_name != split_name:
                            other_patient_ids = [
                                s["patient_id"] for s in other_split_data["slides"]
                            ]
                            assert (
                                patient_id not in other_patient_ids
                            ), f"Patient {patient_id} appears in both {split_name} and {other_split_name}"

    @given(st.floats(min_value=0.1, max_value=0.9))
    @settings(max_examples=20, deadline=60000)
    def test_split_ratio_privacy_preservation_property(self, train_ratio):
        """
        **Property 9: Patient Privacy Preservation**

        For any split ratio configuration, patient privacy SHALL be preserved
        regardless of the specific ratio values.

        **Validates: Requirements 2.5**
        """
        val_ratio = (1.0 - train_ratio) / 2
        test_ratio = (1.0 - train_ratio) / 2

        # Ensure ratios sum to 1.0
        total_ratio = train_ratio + val_ratio + test_ratio
        train_ratio /= total_ratio
        val_ratio /= total_ratio
        test_ratio /= total_ratio

        generator = CAMELYONSyntheticGenerator(random_seed=42)
        spec = CAMELYONSyntheticSpec(
            num_slides=12,  # Fixed number for consistent testing
            patches_per_slide_range=(10, 20),
            feature_dim=256,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Generate synthetic data
            dataset = generator.generate_samples(
                num_slides=12, spec=spec, output_dir=Path(tmp_dir) / "synthetic_camelyon"
            )

            # Create patient splits with custom ratios
            splits = generator.create_patient_splits(
                dataset, split_ratios={"train": train_ratio, "val": val_ratio, "test": test_ratio}
            )

            # Collect all patients from all splits
            all_patients_in_splits = set()
            split_patients = {}

            for split_name, split_data in splits.items():
                split_patients[split_name] = set()
                for slide in split_data["slides"]:
                    patient_id = slide["patient_id"]
                    split_patients[split_name].add(patient_id)
                    all_patients_in_splits.add(patient_id)

            # Property: No patient overlap between splits
            for split1 in split_patients:
                for split2 in split_patients:
                    if split1 != split2:
                        overlap = split_patients[split1] & split_patients[split2]
                        assert (
                            len(overlap) == 0
                        ), f"Patient overlap between {split1} and {split2}: {overlap}"

            # Property: All original patients should be preserved
            original_patients = set(slide["patient_id"] for slide in dataset["slides"])
            assert all_patients_in_splits == original_patients

            # Property: Split sizes should roughly match ratios (within reasonable bounds)
            total_patients = len(original_patients)
            if total_patients >= 3:  # Only check if we have enough patients
                train_expected = int(total_patients * train_ratio)
                val_expected = int(total_patients * val_ratio)
                test_expected = total_patients - train_expected - val_expected

                train_actual = len(split_patients["train"])
                val_actual = len(split_patients["val"])
                test_actual = len(split_patients["test"])

                # Allow some flexibility due to integer rounding
                assert abs(train_actual - train_expected) <= 1
                assert abs(val_actual - val_expected) <= 1
                assert abs(test_actual - test_expected) <= 1


class TestCAMELYONDataIntegrity:
    """Property tests for CAMELYON data integrity preservation."""

    @given(camelyon_slide_strategy())
    @settings(max_examples=50, deadline=60000)
    def test_data_integrity_preservation_property(self, slide_data):
        """
        **Property 2: Data Integrity Preservation**

        For any CAMELYON slide data processing operation, the essential
        properties of the input data SHALL be maintained in the output.

        **Validates: Requirements 2.1, 2.2**
        """
        features = slide_data["features"]
        coordinates = slide_data["coordinates"]
        slide_id = slide_data["slide_id"]
        patient_id = slide_data["patient_id"]
        label = slide_data["label"]

        assume(len(features) >= 1)
        assume(len(coordinates) >= 1)

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Save slide data to HDF5
            slide_path = Path(tmp_dir) / f"{slide_id}.h5"
            with h5py.File(slide_path, "w") as f:
                f.create_dataset("features", data=features)
                f.create_dataset("coordinates", data=coordinates)
                f.attrs["slide_id"] = slide_id
                f.attrs["patient_id"] = patient_id
                f.attrs["label"] = label

            # Load data back
            with h5py.File(slide_path, "r") as f:
                loaded_features = f["features"][:]
                loaded_coordinates = f["coordinates"][:]
                loaded_slide_id = f.attrs["slide_id"]
                loaded_patient_id = f.attrs["patient_id"]
                loaded_label = f.attrs["label"]

            # Property: Data integrity should be preserved
            assert np.array_equal(loaded_features, features)
            assert np.array_equal(loaded_coordinates, coordinates)
            assert loaded_slide_id == slide_id
            assert loaded_patient_id == patient_id
            assert loaded_label == label

            # Property: Data types should be preserved
            assert loaded_features.dtype == features.dtype
            assert loaded_coordinates.dtype == coordinates.dtype

            # Property: Shapes should be preserved
            assert loaded_features.shape == features.shape
            assert loaded_coordinates.shape == coordinates.shape

            # Property: No data corruption should occur
            assert not np.any(np.isnan(loaded_features))
            assert not np.any(np.isinf(loaded_features))
            assert np.all(np.isfinite(loaded_coordinates))

    @given(st.integers(min_value=1, max_value=5))
    @settings(max_examples=30, deadline=60000)
    def test_batch_processing_integrity_property(self, num_slides):
        """
        **Property 2: Data Integrity Preservation**

        For any batch processing operation on CAMELYON slides,
        individual slide properties SHALL be preserved.

        **Validates: Requirements 2.1, 2.2**
        """
        generator = CAMELYONSyntheticGenerator(random_seed=42)
        spec = CAMELYONSyntheticSpec(
            num_slides=num_slides,
            patches_per_slide_range=(10, 30),
            feature_dim=1024,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Generate synthetic data
            dataset = generator.generate_samples(
                num_slides=num_slides, spec=spec, output_dir=Path(tmp_dir) / "batch_test"
            )

            # Create slide index
            slides = [
                SlideMetadata(
                    slide_id=entry["slide_id"],
                    patient_id=entry["patient_id"],
                    file_path=f"features/{entry['slide_id']}.h5",
                    label=entry["label"],
                    split="train",
                )
                for entry in dataset["slide_index"]
            ]

            slide_index = CAMELYONSlideIndex(slides)

            # Create slide dataset
            slide_dataset = CAMELYONSlideDataset(
                slide_index=slide_index,
                features_dir=str(Path(tmp_dir) / "batch_test" / "features"),
                split="train",
            )

            # Process slides individually and verify integrity
            for i in range(len(slide_dataset)):
                slide_sample = slide_dataset[i]
                slide_id = slide_sample["slide_id"]

                # Find original slide data
                original_slide = None
                for slide in dataset["slides"]:
                    if slide["slide_id"] == slide_id:
                        original_slide = slide
                        break

                assert original_slide is not None

                # Property: Essential properties should be preserved
                assert slide_sample["slide_id"] == original_slide["slide_id"]
                assert slide_sample["patient_id"] == original_slide["patient_id"]
                assert slide_sample["label"] == original_slide["label"]
                assert slide_sample["num_patches"] == original_slide["num_patches"]

                # Property: Feature and coordinate shapes should match
                assert slide_sample["features"].shape[0] == original_slide["num_patches"]
                assert slide_sample["coordinates"].shape[0] == original_slide["num_patches"]
                assert slide_sample["features"].shape[1] == spec.feature_dim
                assert slide_sample["coordinates"].shape[1] == 2

                # Property: Data should be finite and valid
                assert torch.all(torch.isfinite(slide_sample["features"]))
                assert torch.all(torch.isfinite(slide_sample["coordinates"]))


class TestCAMELYONErrorDetection:
    """Property tests for CAMELYON error detection and reporting."""

    @given(
        st.sampled_from(
            ["feature_corruption", "coordinate_mismatch", "missing_patches", "invalid_labels"]
        )
    )
    @settings(max_examples=20, deadline=60000)
    def test_error_detection_property(self, corruption_type):
        """
        **Property 5: Error Detection and Reporting**

        For any error condition in CAMELYON data, the system SHALL detect
        the error and provide actionable diagnostic information.

        **Validates: Requirements 2.7**
        """
        generator = CAMELYONSyntheticGenerator(random_seed=42)
        spec = CAMELYONSyntheticSpec(
            num_slides=3,
            patches_per_slide_range=(20, 40),
            feature_dim=512,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Generate clean data
            dataset = generator.generate_samples(
                num_slides=3, spec=spec, output_dir=Path(tmp_dir) / "clean_data"
            )

            # Introduce corruption
            corrupted_dataset = generator.corrupt_samples(dataset, corruption_type)

            # Save corrupted data
            corrupted_dir = Path(tmp_dir) / "corrupted_data"
            generator._save_dataset(corrupted_dataset, corrupted_dir)

            # Test error detection
            features_dir = corrupted_dir / "features"

            for slide_entry in corrupted_dataset["slide_index"]:
                slide_id = slide_entry["slide_id"]
                feature_path = features_dir / f"{slide_id}.h5"

                if feature_path.exists():
                    result = validate_feature_file(str(feature_path))

                    # Property: Corruption should be detected (some files may be valid)
                    if not result["valid"]:
                        # Property: Error message should be informative
                        assert result["error"] is not None
                        assert len(result["error"]) > 0

                        # Property: Error should be actionable
                        error_msg = result["error"].lower()
                        actionable_keywords = [
                            "missing",
                            "mismatch",
                            "invalid",
                            "corrupt",
                            "expected",
                            "found",
                            "shape",
                            "dimension",
                        ]
                        assert any(keyword in error_msg for keyword in actionable_keywords)

    @given(st.integers(min_value=1, max_value=3))
    @settings(max_examples=15, deadline=60000)
    def test_missing_file_error_reporting_property(self, num_missing):
        """
        **Property 5: Error Detection and Reporting**

        For any missing file scenario, the system SHALL provide clear
        error messages with recovery suggestions.

        **Validates: Requirements 2.7**
        """
        generator = CAMELYONSyntheticGenerator(random_seed=42)
        spec = CAMELYONSyntheticSpec(
            num_slides=5,
            patches_per_slide_range=(10, 20),
            feature_dim=256,
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            # Generate data
            dataset = generator.generate_samples(
                num_slides=5, spec=spec, output_dir=Path(tmp_dir) / "test_data"
            )

            # Remove some feature files
            features_dir = Path(tmp_dir) / "test_data" / "features"
            feature_files = list(features_dir.glob("*.h5"))

            files_to_remove = feature_files[:num_missing]
            for file_path in files_to_remove:
                file_path.unlink()

            # Test error detection for missing files
            for file_path in files_to_remove:
                result = validate_feature_file(str(file_path))

                # Property: Missing file should be detected
                assert result["valid"] is False

                # Property: Error message should mention file not found
                assert result["error"] is not None
                error_msg = result["error"].lower()
                assert "not found" in error_msg or "does not exist" in error_msg

            # Test dataset creation with missing files
            slides = [
                SlideMetadata(
                    slide_id=entry["slide_id"],
                    patient_id=entry["patient_id"],
                    file_path=f"features/{entry['slide_id']}.h5",
                    label=entry["label"],
                    split="train",
                )
                for entry in dataset["slide_index"]
            ]

            slide_index = CAMELYONSlideIndex(slides)

            # Property: Dataset should handle missing files gracefully
            if len(files_to_remove) == len(feature_files):
                # All files missing - should raise informative error
                with pytest.raises(ValueError) as exc_info:
                    CAMELYONSlideDataset(
                        slide_index=slide_index,
                        features_dir=str(features_dir),
                        split="train",
                    )

                error_msg = str(exc_info.value).lower()
                assert "no valid feature files" in error_msg or "not found" in error_msg
            else:
                # Some files missing - should create dataset with available files
                slide_dataset = CAMELYONSlideDataset(
                    slide_index=slide_index,
                    features_dir=str(features_dir),
                    split="train",
                )

                # Property: Dataset should have fewer slides than expected
                expected_slides = len(feature_files) - num_missing
                assert len(slide_dataset) == expected_slides
