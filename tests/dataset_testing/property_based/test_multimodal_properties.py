"""
Property-based tests for multimodal dataset functionality.

Tests universal properties that should hold across all valid multimodal data inputs
using Hypothesis for comprehensive input coverage.

**Feature: comprehensive-dataset-testing**
"""

import tempfile
from pathlib import Path
from typing import Dict, Any, List

import torch
from hypothesis import given, settings, strategies as st, assume
from omegaconf import DictConfig

from src.data.loaders import MultimodalDataset, collate_multimodal
from tests.dataset_testing.synthetic.multimodal_generator import (
    MultimodalSyntheticGenerator,
    MultimodalSyntheticSpec,
)
from tests.dataset_testing.hypothesis_strategies import (
    multimodal_config_strategy,
    patient_count_strategy,
    feature_dimension_strategy,
)


class TestMultimodalDataIntegrityProperties:
    """Property-based tests for multimodal data integrity preservation."""

    @given(
        num_patients=patient_count_strategy(),
        wsi_dim=feature_dimension_strategy(min_dim=512, max_dim=4096),
        genomic_dim=feature_dimension_strategy(min_dim=100, max_dim=2000),
        missing_prob=st.floats(min_value=0.0, max_value=0.5),
    )
    @settings(max_examples=100, deadline=60000)
    def test_multimodal_data_fusion_integrity_property(
        self, num_patients: int, wsi_dim: int, genomic_dim: int, missing_prob: float
    ):
        """
        **Property 2: Data Integrity Preservation**

        For any multimodal dataset operation that should preserve data characteristics
        (loading, preprocessing, multimodal fusion), the essential properties of the
        input data SHALL be maintained in the output.

        **Validates: Requirements 3.4**
        """
        assume(num_patients >= 2)  # Need at least 2 patients for meaningful testing

        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            generator = MultimodalSyntheticGenerator(random_seed=42)

            # Generate synthetic multimodal dataset
            spec = MultimodalSyntheticSpec(
                num_patients=num_patients,
                wsi_feature_dim=wsi_dim,
                genomic_feature_dim=genomic_dim,
                clinical_text_length_range=(10, 50),
                missing_modality_probability=missing_prob,
            )

            dataset_samples = generator.generate_samples(
                num_patients=num_patients, spec=spec, output_dir=data_dir
            )

            # Save dataset to disk
            generator._save_dataset(dataset_samples, data_dir)

            # Create dataset and config
            config = DictConfig(
                {
                    "wsi_enabled": True,
                    "genomic_enabled": True,
                    "clinical_text_enabled": True,
                    "wsi_feature_dim": wsi_dim,
                    "genomic_feature_dim": genomic_dim,
                    "max_text_length": 100,
                }
            )

            dataset = MultimodalDataset(data_dir, "train", config)

            # Property: Dataset length preservation
            assert (
                len(dataset) == num_patients
            ), f"Dataset length {len(dataset)} should equal number of patients {num_patients}"

            # Property: Patient ID uniqueness and consistency
            patient_ids = []
            original_data = {}

            for i in range(len(dataset)):
                sample = dataset[i]
                patient_id = sample["patient_id"]

                # Patient IDs should be unique
                assert patient_id not in patient_ids, f"Duplicate patient ID found: {patient_id}"
                patient_ids.append(patient_id)

                # Store original data characteristics
                original_data[patient_id] = {
                    "has_wsi": sample["wsi_features"] is not None,
                    "has_genomic": sample["genomic"] is not None,
                    "has_clinical": sample["clinical_text"] is not None,
                    "label": (
                        sample["label"].item()
                        if isinstance(sample["label"], torch.Tensor)
                        else sample["label"]
                    ),
                }

                # Property: Data type consistency
                if sample["wsi_features"] is not None:
                    assert isinstance(
                        sample["wsi_features"], torch.Tensor
                    ), "WSI features should be torch.Tensor"
                    assert (
                        sample["wsi_features"].dtype == torch.float32
                    ), "WSI features should be float32"
                    assert (
                        sample["wsi_features"].shape[1] == wsi_dim
                    ), f"WSI feature dimension should be {wsi_dim}"

                if sample["genomic"] is not None:
                    assert isinstance(
                        sample["genomic"], torch.Tensor
                    ), "Genomic features should be torch.Tensor"
                    assert (
                        sample["genomic"].dtype == torch.float32
                    ), "Genomic features should be float32"
                    assert (
                        sample["genomic"].shape[0] == genomic_dim
                    ), f"Genomic feature dimension should be {genomic_dim}"

                if sample["clinical_text"] is not None:
                    assert isinstance(
                        sample["clinical_text"], torch.Tensor
                    ), "Clinical text should be torch.Tensor"
                    assert (
                        sample["clinical_text"].dtype == torch.long
                    ), "Clinical text should be long (token IDs)"

            # Property: Batch fusion preserves individual sample properties
            if len(dataset) >= 2:
                batch_size = min(4, len(dataset))
                batch = [dataset[i] for i in range(batch_size)]
                collated_batch = collate_multimodal(batch)

                # Batch should preserve patient count
                assert (
                    len(collated_batch["patient_ids"]) == batch_size
                ), f"Collated batch should have {batch_size} patients"

                # Batch should preserve label information
                assert (
                    collated_batch["label"].shape[0] == batch_size
                ), f"Collated batch labels should have {batch_size} entries"

                # Individual sample properties should be preserved in batch
                for i, patient_id in enumerate(collated_batch["patient_ids"]):
                    original_sample = original_data[patient_id]

                    # Check modality presence consistency
                    if (
                        "wsi_features" in collated_batch
                        and collated_batch["wsi_features"] is not None
                    ):
                        if original_sample["has_wsi"]:
                            # If original had WSI, batch should have non-zero features
                            wsi_sample = collated_batch["wsi_features"][i]
                            assert not torch.all(
                                wsi_sample == 0
                            ), f"Patient {patient_id} should have non-zero WSI features"

                    if "genomic" in collated_batch and collated_batch["genomic"] is not None:
                        if original_sample["has_genomic"]:
                            # If original had genomic, batch should have non-zero features
                            genomic_sample = collated_batch["genomic"][i]
                            assert not torch.all(
                                genomic_sample == 0
                            ), f"Patient {patient_id} should have non-zero genomic features"

    @given(
        num_patients=patient_count_strategy(),
        batch_sizes=st.lists(st.integers(min_value=1, max_value=8), min_size=2, max_size=5),
    )
    @settings(max_examples=100, deadline=60000)
    def test_multimodal_alignment_preservation_property(
        self, num_patients: int, batch_sizes: List[int]
    ):
        """
        **Property 4: Coordinate and Alignment Preservation**

        For any operation involving spatial coordinates or multimodal alignment
        (patch extraction, feature alignment, batch creation), the spatial and
        cross-modal relationships SHALL be preserved throughout processing.

        **Validates: Requirements 3.1, 3.6**
        """
        assume(num_patients >= max(batch_sizes))  # Ensure we have enough patients

        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            generator = MultimodalSyntheticGenerator(random_seed=42)

            # Generate synthetic multimodal dataset
            spec = MultimodalSyntheticSpec(
                num_patients=num_patients,
                wsi_feature_dim=2048,
                genomic_feature_dim=1000,
                clinical_text_length_range=(20, 80),
                missing_modality_probability=0.1,
            )

            dataset_samples = generator.generate_samples(
                num_patients=num_patients, spec=spec, output_dir=data_dir
            )

            # Save dataset to disk
            generator._save_dataset(dataset_samples, data_dir)

            # Create dataset
            config = DictConfig(
                {
                    "wsi_enabled": True,
                    "genomic_enabled": True,
                    "clinical_text_enabled": True,
                    "wsi_feature_dim": 2048,
                    "genomic_feature_dim": 1000,
                    "max_text_length": 100,
                }
            )

            dataset = MultimodalDataset(data_dir, "train", config)

            # Property: Alignment preservation across different batch sizes
            for batch_size in batch_sizes:
                if batch_size > len(dataset):
                    continue

                batch = [dataset[i] for i in range(batch_size)]
                collated_batch = collate_multimodal(batch)

                # Property: Patient ID order preservation
                for i, expected_patient_id in enumerate([sample["patient_id"] for sample in batch]):
                    actual_patient_id = collated_batch["patient_ids"][i]
                    assert (
                        actual_patient_id == expected_patient_id
                    ), f"Patient ID order not preserved: expected {expected_patient_id}, got {actual_patient_id}"

                # Property: Label alignment preservation
                for i, expected_label in enumerate([sample["label"] for sample in batch]):
                    actual_label = collated_batch["label"][i]
                    if isinstance(expected_label, torch.Tensor):
                        expected_label = expected_label.item()
                    if isinstance(actual_label, torch.Tensor):
                        actual_label = actual_label.item()
                    assert (
                        actual_label == expected_label
                    ), f"Label alignment not preserved for patient {collated_batch['patient_ids'][i]}"

                # Property: Cross-modal feature alignment
                # Features for the same patient should be aligned across modalities
                for i in range(batch_size):
                    patient_id = collated_batch["patient_ids"][i]
                    original_sample = batch[i]

                    # If multiple modalities are present, they should be aligned
                    modalities_present = []
                    if collated_batch.get("wsi_features") is not None and not torch.all(
                        collated_batch["wsi_features"][i] == 0
                    ):
                        modalities_present.append("wsi")
                    if collated_batch.get("genomic") is not None and not torch.all(
                        collated_batch["genomic"][i] == 0
                    ):
                        modalities_present.append("genomic")
                    if collated_batch.get("clinical_text") is not None and not torch.all(
                        collated_batch["clinical_text"][i] == 0
                    ):
                        modalities_present.append("clinical")

                    # If multiple modalities are present, verify they correspond to the same patient
                    if len(modalities_present) > 1:
                        # This is implicitly verified by the patient_id alignment,
                        # but we can add additional checks here if needed
                        assert (
                            patient_id == original_sample["patient_id"]
                        ), f"Cross-modal alignment broken for patient {patient_id}"

    @given(config=multimodal_config_strategy(), num_patients=st.integers(min_value=3, max_value=10))
    @settings(max_examples=100, deadline=60000)
    def test_multimodal_missing_data_handling_property(
        self, config: Dict[str, Any], num_patients: int
    ):
        """
        Property test for missing data handling in multimodal datasets.

        Verifies that missing modalities are handled gracefully and consistently
        across different configurations and batch sizes.

        **Validates: Requirements 3.5, 3.7**
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            data_dir = Path(temp_dir)
            generator = MultimodalSyntheticGenerator(random_seed=42)

            # Generate dataset with high missing probability to test edge cases
            spec = MultimodalSyntheticSpec(
                num_patients=num_patients,
                wsi_feature_dim=config.get("wsi_feature_dim", 2048),
                genomic_feature_dim=config.get("genomic_feature_dim", 1000),
                clinical_text_length_range=(10, 50),
                missing_modality_probability=0.4,  # High missing rate for testing
            )

            dataset_samples = generator.generate_samples(
                num_patients=num_patients, spec=spec, output_dir=data_dir
            )

            # Save dataset to disk
            generator._save_dataset(dataset_samples, data_dir)

            # Create dataset with the given config
            dataset_config = DictConfig(config)
            dataset = MultimodalDataset(data_dir, "train", dataset_config)

            # Property: Dataset should handle missing modalities gracefully
            missing_counts = {"wsi": 0, "genomic": 0, "clinical": 0}
            total_samples = len(dataset)

            for i in range(total_samples):
                sample = dataset[i]

                # Property: Sample should always have required fields
                assert "patient_id" in sample, "Sample should have patient_id"
                assert "label" in sample, "Sample should have label"

                # Count missing modalities
                if sample["wsi_features"] is None:
                    missing_counts["wsi"] += 1
                if sample["genomic"] is None:
                    missing_counts["genomic"] += 1
                if sample["clinical_text"] is None:
                    missing_counts["clinical"] += 1

                # Property: Disabled modalities should be None
                if not config.get("wsi_enabled", True):
                    assert sample["wsi_features"] is None, "Disabled WSI modality should be None"
                if not config.get("genomic_enabled", True):
                    assert sample["genomic"] is None, "Disabled genomic modality should be None"
                if not config.get("clinical_text_enabled", True):
                    assert (
                        sample["clinical_text"] is None
                    ), "Disabled clinical text modality should be None"

            # Property: Batch collation should handle mixed missing/present modalities
            if total_samples >= 2:
                # Create a mixed batch with different missing patterns
                batch = [dataset[i] for i in range(min(4, total_samples))]
                collated_batch = collate_multimodal(batch)

                # Property: Batch structure should be preserved
                assert len(collated_batch["patient_ids"]) == len(
                    batch
                ), "Batch size should be preserved in collation"
                assert collated_batch["label"].shape[0] == len(
                    batch
                ), "Label batch size should match sample count"

                # Property: Missing modalities should not cause errors
                for modality in ["wsi_features", "genomic", "clinical_text"]:
                    if modality in collated_batch:
                        if collated_batch[modality] is not None:
                            assert collated_batch[modality].shape[0] == len(
                                batch
                            ), f"{modality} batch dimension should match sample count"


if __name__ == "__main__":
    # Run property tests
    import pytest

    pytest.main([__file__, "-v"])
