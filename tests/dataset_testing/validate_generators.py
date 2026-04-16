"""
Validation script for synthetic data generators.

This script tests all synthetic data generators to ensure they
produce valid samples before proceeding with the full test suite.
"""

import sys
import tempfile
import numpy as np
from pathlib import Path
from typing import Dict, Any

# Add the tests directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

from dataset_testing.synthetic.pcam_generator import PCamSyntheticGenerator, PCamSyntheticSpec
from dataset_testing.synthetic.camelyon_generator import (
    CAMELYONSyntheticGenerator,
    CAMELYONSyntheticSpec,
)
from dataset_testing.synthetic.multimodal_generator import (
    MultimodalSyntheticGenerator,
    MultimodalSyntheticSpec,
)
from dataset_testing.synthetic.wsi_generator import WSISyntheticGenerator, WSISyntheticSpec


def validate_pcam_generator() -> bool:
    """Validate PCam synthetic data generator."""
    print("Validating PCam synthetic data generator...")

    try:
        generator = PCamSyntheticGenerator(random_seed=42)
        spec = PCamSyntheticSpec(num_samples=50)

        # Generate samples
        samples = generator.generate_samples(50, spec=spec)

        # Validate samples
        if not generator.validate_samples(samples):
            print("❌ PCam generator validation failed")
            return False

        # Test corruption
        corrupted = generator.corrupt_samples(samples, "image_noise")
        if corrupted["metadata"]["corruption_type"] != "image_noise":
            print("❌ PCam corruption simulation failed")
            return False

        # Test splits
        splits = generator.create_dataset_splits(samples)
        if not all(split in splits for split in ["train", "val", "test"]):
            print("❌ PCam split creation failed")
            return False

        print("✅ PCam generator validation passed")
        return True

    except Exception as e:
        print(f"❌ PCam generator validation failed with error: {e}")
        return False


def validate_camelyon_generator() -> bool:
    """Validate CAMELYON synthetic data generator."""
    print("Validating CAMELYON synthetic data generator...")

    try:
        generator = CAMELYONSyntheticGenerator(random_seed=42)
        spec = CAMELYONSyntheticSpec(num_slides=10)

        # Generate samples
        dataset = generator.generate_samples(10, spec=spec)

        # Validate samples
        if not generator.validate_samples(dataset):
            print("❌ CAMELYON generator validation failed")
            return False

        # Test corruption
        corrupted = generator.corrupt_samples(dataset, "feature_corruption")
        if corrupted["metadata"]["corruption_type"] != "feature_corruption":
            print("❌ CAMELYON corruption simulation failed")
            return False

        # Test patient splits
        splits = generator.create_patient_splits(dataset)
        if not all(split in splits for split in ["train", "val", "test"]):
            print("❌ CAMELYON split creation failed")
            return False

        print("✅ CAMELYON generator validation passed")
        return True

    except Exception as e:
        print(f"❌ CAMELYON generator validation failed with error: {e}")
        return False


def validate_multimodal_generator() -> bool:
    """Validate multimodal synthetic data generator."""
    print("Validating multimodal synthetic data generator...")

    try:
        generator = MultimodalSyntheticGenerator(random_seed=42)
        spec = MultimodalSyntheticSpec(num_patients=20)

        # Generate samples
        dataset = generator.generate_samples(20, spec=spec)

        # Validate samples
        if not generator.validate_samples(dataset):
            print("❌ Multimodal generator validation failed")
            return False

        # Check that patients have different modality combinations
        modality_combinations = set()
        for patient in dataset["patients"]:
            modalities = tuple(sorted(patient["available_modalities"]))
            modality_combinations.add(modalities)

        if len(modality_combinations) < 2:
            print("❌ Multimodal generator should create varied modality combinations")
            return False

        # Test corruption
        corrupted = generator.corrupt_samples(dataset, "modality_mismatch")
        if corrupted["metadata"]["corruption_type"] != "modality_mismatch":
            print("❌ Multimodal corruption simulation failed")
            return False

        # Test patient splits
        splits = generator.create_patient_splits(dataset)
        if not all(split in splits for split in ["train", "val", "test"]):
            print("❌ Multimodal split creation failed")
            return False

        print("✅ Multimodal generator validation passed")
        return True

    except Exception as e:
        print(f"❌ Multimodal generator validation failed with error: {e}")
        return False


def validate_wsi_generator() -> bool:
    """Validate WSI synthetic data generator."""
    print("Validating WSI synthetic data generator...")

    try:
        generator = WSISyntheticGenerator(random_seed=42)
        spec = WSISyntheticSpec(num_slides=5)

        # Generate samples
        dataset = generator.generate_samples(5, spec=spec)

        # Validate samples
        if not generator.validate_samples(dataset):
            print("❌ WSI generator validation failed")
            return False

        # Check that slides have patches
        for slide in dataset["slides"]:
            if slide["num_patches"] == 0:
                print("❌ WSI generator should create slides with patches")
                return False

        # Test corruption
        corrupted = generator.corrupt_samples(dataset, "invalid_dimensions")
        if corrupted["metadata"]["corruption_type"] != "invalid_dimensions":
            print("❌ WSI corruption simulation failed")
            return False

        print("✅ WSI generator validation passed")
        return True

    except Exception as e:
        print(f"❌ WSI generator validation failed with error: {e}")
        return False


def validate_file_saving() -> bool:
    """Validate that generators can save files correctly."""
    print("Validating file saving functionality...")

    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # Test PCam file saving
            pcam_generator = PCamSyntheticGenerator(random_seed=42)
            pcam_generator.generate_samples(10, output_dir=temp_path / "pcam")

            pcam_files = list((temp_path / "pcam").glob("*"))
            if len(pcam_files) == 0:
                print("❌ PCam generator did not save files")
                return False

            # Test CAMELYON file saving
            camelyon_generator = CAMELYONSyntheticGenerator(random_seed=42)
            camelyon_generator.generate_samples(5, output_dir=temp_path / "camelyon")

            camelyon_files = list((temp_path / "camelyon").glob("**/*"))
            if len(camelyon_files) == 0:
                print("❌ CAMELYON generator did not save files")
                return False

            # Test multimodal file saving
            multimodal_generator = MultimodalSyntheticGenerator(random_seed=42)
            multimodal_generator.generate_samples(5, output_dir=temp_path / "multimodal")

            multimodal_files = list((temp_path / "multimodal").glob("**/*"))
            if len(multimodal_files) == 0:
                print("❌ Multimodal generator did not save files")
                return False

            print("✅ File saving validation passed")
            return True

    except Exception as e:
        print(f"❌ File saving validation failed with error: {e}")
        return False


def main():
    """Run all generator validations."""
    print("🔍 Validating synthetic data generators...")
    print("=" * 60)

    validations = [
        validate_pcam_generator,
        validate_camelyon_generator,
        validate_multimodal_generator,
        validate_wsi_generator,
        validate_file_saving,
    ]

    results = []
    for validation_func in validations:
        result = validation_func()
        results.append(result)
        print()

    print("=" * 60)
    print("📊 Validation Summary:")
    print(f"✅ Passed: {sum(results)}/{len(results)}")
    print(f"❌ Failed: {len(results) - sum(results)}/{len(results)}")

    if all(results):
        print("\n🎉 All synthetic data generators are working correctly!")
        print("Ready to proceed with comprehensive dataset testing.")
        return True
    else:
        print("\n⚠️  Some generators failed validation.")
        print("Please fix the issues before proceeding.")
        return False


if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
