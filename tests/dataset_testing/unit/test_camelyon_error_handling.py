"""
CAMELYON error handling unit tests.

Tests missing slide files, XML parsing errors, and mask generation failures.
"""

import json
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any

import h5py
import numpy as np
import pytest
import torch

from src.data.camelyon_dataset import (
    CAMELYONPatchDataset,
    CAMELYONSlideDataset,
    CAMELYONSlideIndex,
    SlideMetadata,
    validate_feature_file,
)
from tests.dataset_testing.synthetic.camelyon_generator import (
    CAMELYONSyntheticGenerator,
    CAMELYONSyntheticSpec,
)


class TestCAMELYONMissingSlideFiles:
    """Test handling of missing slide files with proper error messages."""

    @pytest.fixture
    def synthetic_generator(self):
        """Create synthetic data generator."""
        return CAMELYONSyntheticGenerator(random_seed=42)

    @pytest.fixture
    def sample_dataset(self, synthetic_generator, tmp_path):
        """Generate sample dataset for testing."""
        spec = CAMELYONSyntheticSpec(
            num_slides=5,
            patches_per_slide_range=(10, 30),
            feature_dim=1024,
        )
        
        dataset = synthetic_generator.generate_samples(
            num_slides=5,
            spec=spec,
            output_dir=tmp_path / "sample_camelyon"
        )
        
        return dataset, tmp_path / "sample_camelyon"

    def test_missing_single_slide_file(self, sample_dataset):
        """Test handling when a single slide file is missing."""
        dataset, data_dir = sample_dataset
        
        # Remove one feature file
        features_dir = data_dir / "features"
        first_slide_id = dataset["slide_index"][0]["slide_id"]
        missing_file = features_dir / f"{first_slide_id}.h5"
        missing_file.unlink()
        
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
        
        # Test slide dataset creation
        slide_dataset = CAMELYONSlideDataset(
            slide_index=slide_index,
            features_dir=str(features_dir),
            split="train",
        )
        
        # Should have one fewer slide
        assert len(slide_dataset) == len(dataset["slide_index"]) - 1
        
        # Verify missing slide is not accessible
        slide_ids = [slide_dataset[i]["slide_id"] for i in range(len(slide_dataset))]
        assert first_slide_id not in slide_ids

    def test_missing_multiple_slide_files(self, sample_dataset):
        """Test handling when multiple slide files are missing."""
        dataset, data_dir = sample_dataset
        
        # Remove multiple feature files
        features_dir = data_dir / "features"
        missing_slide_ids = [
            dataset["slide_index"][0]["slide_id"],
            dataset["slide_index"][2]["slide_id"],
        ]
        
        for slide_id in missing_slide_ids:
            missing_file = features_dir / f"{slide_id}.h5"
            missing_file.unlink()
        
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
        
        # Test slide dataset creation
        slide_dataset = CAMELYONSlideDataset(
            slide_index=slide_index,
            features_dir=str(features_dir),
            split="train",
        )
        
        # Should have fewer slides
        expected_count = len(dataset["slide_index"]) - len(missing_slide_ids)
        assert len(slide_dataset) == expected_count
        
        # Verify missing slides are not accessible
        slide_ids = [slide_dataset[i]["slide_id"] for i in range(len(slide_dataset))]
        for missing_id in missing_slide_ids:
            assert missing_id not in slide_ids

    def test_all_slide_files_missing(self, sample_dataset):
        """Test error when all slide files are missing."""
        dataset, data_dir = sample_dataset
        
        # Remove all feature files
        features_dir = data_dir / "features"
        for h5_file in features_dir.glob("*.h5"):
            h5_file.unlink()
        
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
        
        # Should raise informative error
        with pytest.raises(ValueError, match="No valid feature files found"):
            CAMELYONSlideDataset(
                slide_index=slide_index,
                features_dir=str(features_dir),
                split="train",
            )

    def test_nonexistent_features_directory(self, sample_dataset):
        """Test error when features directory doesn't exist."""
        dataset, data_dir = sample_dataset
        
        # Create slide index with nonexistent directory
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
        
        # Should raise informative error
        with pytest.raises(ValueError, match="No valid feature files found"):
            CAMELYONSlideDataset(
                slide_index=slide_index,
                features_dir=str(data_dir / "nonexistent_features"),
                split="train",
            )

    def test_missing_file_error_messages(self, tmp_path):
        """Test that missing file errors provide clear messages."""
        nonexistent_file = tmp_path / "nonexistent.h5"
        
        result = validate_feature_file(str(nonexistent_file))
        
        assert result["valid"] is False
        assert result["error"] is not None
        assert "not found" in result["error"].lower()
        assert str(nonexistent_file) in result["error"]

    def test_patch_dataset_missing_files(self, sample_dataset):
        """Test patch dataset handling of missing files."""
        dataset, data_dir = sample_dataset
        
        # Remove one feature file
        features_dir = data_dir / "features"
        first_slide_id = dataset["slide_index"][0]["slide_id"]
        missing_file = features_dir / f"{first_slide_id}.h5"
        missing_file.unlink()
        
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
        
        # Test patch dataset creation
        patch_dataset = CAMELYONPatchDataset(
            slide_index=slide_index,
            features_dir=str(features_dir),
            split="train",
        )
        
        # Should have fewer patches (missing slide's patches excluded)
        original_slide = None
        for slide in dataset["slides"]:
            if slide["slide_id"] == first_slide_id:
                original_slide = slide
                break
        
        expected_total_patches = sum(
            slide["num_patches"] for slide in dataset["slides"]
            if slide["slide_id"] != first_slide_id
        )
        
        assert len(patch_dataset) == expected_total_patches


class TestCAMELYONXMLParsingErrors:
    """Test XML annotation parsing error handling."""

    def test_malformed_xml_annotation(self, tmp_path):
        """Test handling of malformed XML annotation files."""
        # Create malformed XML file
        malformed_xml = tmp_path / "malformed.xml"
        with open(malformed_xml, "w") as f:
            f.write("<?xml version='1.0'?>\n<root><unclosed_tag></root>")
        
        # Test XML parsing (this would be in annotation processing code)
        with pytest.raises(ET.ParseError):
            ET.parse(str(malformed_xml))

    def test_missing_xml_annotation_file(self, tmp_path):
        """Test handling of missing XML annotation files."""
        nonexistent_xml = tmp_path / "nonexistent.xml"
        
        # Test file access
        with pytest.raises(FileNotFoundError):
            ET.parse(str(nonexistent_xml))

    def test_xml_with_invalid_structure(self, tmp_path):
        """Test handling of XML files with invalid structure."""
        # Create XML with valid syntax but invalid structure
        invalid_xml = tmp_path / "invalid_structure.xml"
        with open(invalid_xml, "w") as f:
            f.write("""<?xml version='1.0'?>
            <root>
                <unexpected_element>
                    <data>Some data</data>
                </unexpected_element>
            </root>""")
        
        # Should parse successfully but may fail validation
        tree = ET.parse(str(invalid_xml))
        root = tree.getroot()
        
        # Test that expected elements are missing
        assert root.find("Annotations") is None
        assert root.find("Annotation") is None

    def test_empty_xml_annotation_file(self, tmp_path):
        """Test handling of empty XML annotation files."""
        empty_xml = tmp_path / "empty.xml"
        empty_xml.touch()
        
        # Should raise parse error for empty file
        with pytest.raises(ET.ParseError):
            ET.parse(str(empty_xml))

    def test_xml_with_encoding_issues(self, tmp_path):
        """Test handling of XML files with encoding issues."""
        # Create XML with special characters
        special_xml = tmp_path / "special_chars.xml"
        with open(special_xml, "w", encoding="utf-8") as f:
            f.write("""<?xml version='1.0' encoding='UTF-8'?>
            <root>
                <text>Special chars: àáâãäåæçèéêë</text>
            </root>""")
        
        # Should parse successfully with proper encoding
        tree = ET.parse(str(special_xml))
        root = tree.getroot()
        text_elem = root.find("text")
        assert text_elem is not None
        assert "àáâãäåæçèéêë" in text_elem.text


class TestCAMELYONHDF5StructureErrors:
    """Test HDF5 file structure error handling."""

    def test_corrupted_hdf5_file(self, tmp_path):
        """Test handling of corrupted HDF5 files."""
        corrupted_file = tmp_path / "corrupted.h5"
        
        # Create file with invalid HDF5 content
        with open(corrupted_file, "w") as f:
            f.write("This is not a valid HDF5 file content")
        
        result = validate_feature_file(str(corrupted_file))
        
        assert result["valid"] is False
        assert result["error"] is not None
        assert "unable to open" in result["error"].lower() or "not a valid" in result["error"].lower() or "file signature" in result["error"].lower()

    def test_hdf5_missing_required_datasets(self, tmp_path):
        """Test HDF5 files missing required datasets."""
        # Test missing features dataset
        missing_features = tmp_path / "missing_features.h5"
        with h5py.File(missing_features, "w") as f:
            coordinates = np.random.randint(0, 1000, (50, 2)).astype(np.int32)
            f.create_dataset("coordinates", data=coordinates)
        
        result = validate_feature_file(str(missing_features))
        assert result["valid"] is False
        assert "missing 'features'" in result["error"].lower()
        
        # Test missing coordinates dataset
        missing_coords = tmp_path / "missing_coords.h5"
        with h5py.File(missing_coords, "w") as f:
            features = np.random.randn(50, 1024).astype(np.float32)
            f.create_dataset("features", data=features)
        
        result = validate_feature_file(str(missing_coords))
        assert result["valid"] is False
        assert "missing 'coordinates'" in result["error"].lower()

    def test_hdf5_dimension_mismatches(self, tmp_path):
        """Test HDF5 files with dimension mismatches."""
        # Test mismatched patch counts
        mismatch_file = tmp_path / "mismatch.h5"
        with h5py.File(mismatch_file, "w") as f:
            features = np.random.randn(50, 1024).astype(np.float32)
            coordinates = np.random.randint(0, 1000, (40, 2)).astype(np.int32)  # Different count
            
            f.create_dataset("features", data=features)
            f.create_dataset("coordinates", data=coordinates)
        
        result = validate_feature_file(str(mismatch_file))
        assert result["valid"] is False
        assert "mismatched" in result["error"].lower()

    def test_hdf5_invalid_coordinate_dimensions(self, tmp_path):
        """Test HDF5 files with invalid coordinate dimensions."""
        invalid_coords = tmp_path / "invalid_coords.h5"
        with h5py.File(invalid_coords, "w") as f:
            features = np.random.randn(50, 1024).astype(np.float32)
            coordinates = np.random.randint(0, 1000, (50, 3)).astype(np.int32)  # Wrong shape
            
            f.create_dataset("features", data=features)
            f.create_dataset("coordinates", data=coordinates)
        
        result = validate_feature_file(str(invalid_coords))
        assert result["valid"] is False
        assert "[n, 2]" in result["error"].lower()

    def test_hdf5_invalid_data_types(self, tmp_path):
        """Test HDF5 files with invalid data types."""
        invalid_types = tmp_path / "invalid_types.h5"
        with h5py.File(invalid_types, "w") as f:
            # Use wrong data types
            features = np.random.randint(0, 255, (50, 1024)).astype(np.uint8)  # Should be float32
            coordinates = np.random.randn(50, 2).astype(np.float64)  # Should be int32
            
            f.create_dataset("features", data=features)
            f.create_dataset("coordinates", data=coordinates)
        
        # File structure is valid, but data types might cause issues in processing
        result = validate_feature_file(str(invalid_types))
        # Basic validation should pass, but processing might fail
        assert result["valid"] is True  # Structure is valid
        assert result["num_patches"] == 50
        assert result["feature_dim"] == 1024

    def test_hdf5_empty_datasets(self, tmp_path):
        """Test HDF5 files with empty datasets."""
        empty_datasets = tmp_path / "empty.h5"
        with h5py.File(empty_datasets, "w") as f:
            # Create empty datasets
            f.create_dataset("features", shape=(0, 1024), dtype=np.float32)
            f.create_dataset("coordinates", shape=(0, 2), dtype=np.int32)
        
        result = validate_feature_file(str(empty_datasets))
        assert result["valid"] is True  # Technically valid structure
        assert result["num_patches"] == 0
        assert result["feature_dim"] == 1024

    def test_hdf5_extremely_large_datasets(self, tmp_path):
        """Test handling of extremely large datasets."""
        # This test simulates what would happen with very large datasets
        # without actually creating them (to avoid memory issues)
        
        large_file = tmp_path / "large.h5"
        with h5py.File(large_file, "w") as f:
            # Create datasets with large shapes but don't fill them
            f.create_dataset("features", shape=(1000000, 2048), dtype=np.float32)
            f.create_dataset("coordinates", shape=(1000000, 2), dtype=np.int32)
        
        result = validate_feature_file(str(large_file))
        assert result["valid"] is True
        assert result["num_patches"] == 1000000
        assert result["feature_dim"] == 2048


class TestCAMELYONDatasetRecoveryStrategies:
    """Test dataset recovery strategies for various error conditions."""

    def test_partial_dataset_recovery(self, tmp_path):
        """Test recovery when some files are valid and others are corrupted."""
        # Create mix of valid and invalid files
        features_dir = tmp_path / "mixed_features"
        features_dir.mkdir()
        
        # Create valid files
        for i in range(3):
            valid_file = features_dir / f"valid_{i}.h5"
            with h5py.File(valid_file, "w") as f:
                features = np.random.randn(20, 512).astype(np.float32)
                coordinates = np.random.randint(0, 1000, (20, 2)).astype(np.int32)
                f.create_dataset("features", data=features)
                f.create_dataset("coordinates", data=coordinates)
        
        # Create invalid files
        for i in range(2):
            invalid_file = features_dir / f"invalid_{i}.h5"
            with open(invalid_file, "w") as f:
                f.write("Not a valid HDF5 file")
        
        # Test validation of all files
        valid_count = 0
        invalid_count = 0
        
        for h5_file in features_dir.glob("*.h5"):
            result = validate_feature_file(str(h5_file))
            if result["valid"]:
                valid_count += 1
            else:
                invalid_count += 1
        
        assert valid_count == 3
        assert invalid_count == 2

    def test_error_message_informativeness(self, tmp_path):
        """Test that error messages provide actionable information."""
        # Test various error conditions and verify message quality
        
        # Missing file
        result = validate_feature_file(str(tmp_path / "missing.h5"))
        assert "not found" in result["error"].lower()
        assert tmp_path.name in result["error"]
        
        # Corrupted file
        corrupted = tmp_path / "corrupted.h5"
        with open(corrupted, "w") as f:
            f.write("corrupted content")
        
        result = validate_feature_file(str(corrupted))
        assert result["error"] is not None
        assert len(result["error"]) > 10  # Should be descriptive
        
        # Missing dataset
        missing_dataset = tmp_path / "missing_dataset.h5"
        with h5py.File(missing_dataset, "w") as f:
            f.create_dataset("features", data=np.random.randn(10, 512))
            # Missing coordinates dataset
        
        result = validate_feature_file(str(missing_dataset))
        assert "coordinates" in result["error"].lower()
        assert "missing" in result["error"].lower()

    def test_graceful_degradation_strategies(self, tmp_path):
        """Test graceful degradation when encountering errors."""
        # Create dataset with some problematic files
        features_dir = tmp_path / "degradation_test"
        features_dir.mkdir()
        
        # Create mostly valid files with one problematic file
        valid_files = []
        for i in range(4):
            valid_file = features_dir / f"slide_{i:03d}.h5"
            with h5py.File(valid_file, "w") as f:
                features = np.random.randn(15, 256).astype(np.float32)
                coordinates = np.random.randint(0, 1000, (15, 2)).astype(np.int32)
                f.create_dataset("features", data=features)
                f.create_dataset("coordinates", data=coordinates)
            valid_files.append(f"slide_{i:03d}")
        
        # Create one problematic file
        problem_file = features_dir / "slide_004.h5"
        with open(problem_file, "w") as f:
            f.write("problematic content")
        
        # Create slide index including all files
        slides = []
        for i in range(5):
            slides.append(
                SlideMetadata(
                    slide_id=f"slide_{i:03d}",
                    patient_id=f"patient_{i//2:03d}",
                    file_path=f"features/slide_{i:03d}.h5",
                    label=i % 2,
                    split="train",
                )
            )
        
        slide_index = CAMELYONSlideIndex(slides)
        
        # Dataset should gracefully handle the problematic file
        slide_dataset = CAMELYONSlideDataset(
            slide_index=slide_index,
            features_dir=str(features_dir),
            split="train",
        )
        
        # Should have 5 slides (including the problematic one, but handled gracefully)
        assert len(slide_dataset) == 5
        
        # Verify slides are accessible, but problematic one may raise exception
        accessible_slide_ids = []
        for i in range(len(slide_dataset)):
            try:
                slide_data = slide_dataset[i]
                accessible_slide_ids.append(slide_data["slide_id"])
            except (OSError, ValueError) as e:
                # Problematic file should raise exception
                print(f"Expected error for slide {i}: {e}")
        
        # Should have at least 4 valid slides accessible
        assert len(accessible_slide_ids) >= 4
        for valid_id in valid_files:
            assert valid_id in accessible_slide_ids