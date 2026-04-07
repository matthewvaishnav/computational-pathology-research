"""
Unit tests for data loading utilities.

Tests MultimodalDataset, TemporalDataset, and collation functions.
"""

import json
import tempfile
from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from omegaconf import DictConfig

from src.data.loaders import (
    MultimodalDataset,
    TemporalDataset,
    collate_multimodal,
    collate_temporal,
)
from src.models import MultimodalFusionModel


@pytest.fixture
def temp_data_dir():
    """Create temporary directory with mock data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)

        # Create subdirectories
        (data_dir / "wsi_features").mkdir()
        (data_dir / "genomic").mkdir()
        (data_dir / "clinical_text").mkdir()

        # Create mock WSI features (HDF5)
        wsi_file = data_dir / "wsi_features" / "patient_001_wsi.h5"
        with h5py.File(wsi_file, "w") as f:
            f.create_dataset("features", data=np.random.randn(100, 512).astype(np.float32))

        # Create mock genomic features
        genomic_file = data_dir / "genomic" / "patient_001_genomic.npy"
        np.save(genomic_file, np.random.randn(1000).astype(np.float32))

        # Create mock clinical text
        clinical_file = data_dir / "clinical_text" / "patient_001_clinical.npy"
        np.save(clinical_file, np.random.randint(0, 1000, size=50).astype(np.int64))

        # Create metadata for train split
        metadata = {
            "samples": [
                {
                    "patient_id": "patient_001",
                    "wsi_file": "patient_001_wsi.h5",
                    "genomic_file": "patient_001_genomic.npy",
                    "clinical_file": "patient_001_clinical.npy",
                    "label": 1,
                    "timestamp": 1234567890.0,
                },
                {
                    "patient_id": "patient_002",
                    "wsi_file": None,  # Missing WSI
                    "genomic_file": None,  # Missing genomic
                    "clinical_file": None,  # Missing clinical
                    "label": 0,
                    "timestamp": None,
                },
            ]
        }

        with open(data_dir / "train_metadata.json", "w") as f:
            json.dump(metadata, f)

        yield data_dir


@pytest.fixture
def temp_temporal_data_dir():
    """Create temporary directory with mock temporal data."""
    with tempfile.TemporaryDirectory() as tmpdir:
        data_dir = Path(tmpdir)

        # Create subdirectories
        (data_dir / "wsi_features").mkdir()
        (data_dir / "genomic").mkdir()
        (data_dir / "clinical_text").mkdir()

        # Create mock data for multiple slides
        for i in range(3):
            wsi_file = data_dir / "wsi_features" / f"patient_001_slide_{i}_wsi.h5"
            with h5py.File(wsi_file, "w") as f:
                f.create_dataset("features", data=np.random.randn(100, 512).astype(np.float32))

        # Create temporal metadata
        temporal_metadata = {
            "slides": [
                {
                    "patient_id": "patient_001",
                    "slide_id": "slide_0",
                    "wsi_file": "patient_001_slide_0_wsi.h5",
                    "genomic_file": None,
                    "clinical_file": None,
                    "label": 0,
                    "timestamp": 1000.0,
                },
                {
                    "patient_id": "patient_001",
                    "slide_id": "slide_1",
                    "wsi_file": "patient_001_slide_1_wsi.h5",
                    "genomic_file": None,
                    "clinical_file": None,
                    "label": 0,
                    "timestamp": 2000.0,
                },
                {
                    "patient_id": "patient_001",
                    "slide_id": "slide_2",
                    "wsi_file": "patient_001_slide_2_wsi.h5",
                    "genomic_file": None,
                    "clinical_file": None,
                    "label": 1,
                    "timestamp": 3000.0,
                },
            ]
        }

        with open(data_dir / "train_temporal_metadata.json", "w") as f:
            json.dump(temporal_metadata, f)

        yield data_dir


def test_multimodal_dataset_initialization(temp_data_dir):
    """Test MultimodalDataset can be initialized."""
    config = DictConfig(
        {
            "wsi_enabled": True,
            "genomic_enabled": True,
            "clinical_text_enabled": True,
            "max_text_length": 100,
        }
    )

    dataset = MultimodalDataset(temp_data_dir, "train", config)
    assert len(dataset) == 2


def test_multimodal_dataset_complete_sample(temp_data_dir):
    """Test loading sample with all modalities present."""
    config = DictConfig(
        {
            "wsi_enabled": True,
            "genomic_enabled": True,
            "clinical_text_enabled": True,
            "max_text_length": 100,
        }
    )

    dataset = MultimodalDataset(temp_data_dir, "train", config)
    sample = dataset[0]

    assert sample["patient_id"] == "patient_001"
    assert sample["wsi_features"] is not None
    assert sample["wsi_features"].shape == (100, 512)
    assert sample["genomic"] is not None
    assert sample["genomic"].shape == (1000,)
    assert sample["clinical_text"] is not None
    assert sample["clinical_text"].shape == (100,)  # Padded to max_text_length
    assert sample["label"].item() == 1
    assert sample["timestamp"] == 1234567890.0


def test_multimodal_dataset_missing_modalities(temp_data_dir):
    """Test handling of missing modalities."""
    config = DictConfig(
        {
            "wsi_enabled": True,
            "genomic_enabled": True,
            "clinical_text_enabled": True,
            "max_text_length": 100,
        }
    )

    dataset = MultimodalDataset(temp_data_dir, "train", config)
    sample = dataset[1]  # Sample with all missing modalities

    assert sample["patient_id"] == "patient_002"
    assert sample["wsi_features"] is None
    assert sample["genomic"] is None
    assert sample["clinical_text"] is None
    assert sample["label"].item() == 0
    assert sample["timestamp"] is None


def test_multimodal_dataset_disabled_modalities(temp_data_dir):
    """Test disabling modalities via config."""
    config = DictConfig(
        {
            "wsi_enabled": False,
            "genomic_enabled": True,
            "clinical_text_enabled": False,
            "max_text_length": 100,
        }
    )

    dataset = MultimodalDataset(temp_data_dir, "train", config)
    sample = dataset[0]

    assert sample["wsi_features"] is None  # Disabled
    assert sample["genomic"] is not None  # Enabled
    assert sample["clinical_text"] is None  # Disabled


def test_temporal_dataset_initialization(temp_temporal_data_dir):
    """Test TemporalDataset can be initialized."""
    config = DictConfig(
        {
            "min_slides_per_patient": 2,
            "max_slides_per_patient": 5,
            "wsi_enabled": True,
            "genomic_enabled": False,
            "clinical_text_enabled": False,
        }
    )

    dataset = TemporalDataset(temp_temporal_data_dir, "train", config)
    assert len(dataset) == 1  # One patient with 3 slides


def test_temporal_dataset_slide_sequence(temp_temporal_data_dir):
    """Test temporal ordering of slides."""
    config = DictConfig(
        {
            "min_slides_per_patient": 2,
            "wsi_enabled": True,
            "genomic_enabled": False,
            "clinical_text_enabled": False,
        }
    )

    dataset = TemporalDataset(temp_temporal_data_dir, "train", config)
    sample = dataset[0]

    assert sample["patient_id"] == "patient_001"
    assert len(sample["slide_sequence"]) == 3
    assert sample["timestamps"].shape == (3,)
    assert torch.allclose(sample["timestamps"], torch.tensor([1000.0, 2000.0, 3000.0]))
    assert sample["label"].item() == 1  # Label from most recent slide

    # Check each slide has WSI features
    for slide_data in sample["slide_sequence"]:
        assert slide_data["wsi_features"] is not None
        assert slide_data["wsi_features"].shape == (100, 512)


def test_collate_multimodal(temp_data_dir):
    """Test collation function for multimodal data."""
    config = DictConfig(
        {
            "wsi_enabled": True,
            "genomic_enabled": True,
            "clinical_text_enabled": True,
            "max_text_length": 100,
        }
    )

    dataset = MultimodalDataset(temp_data_dir, "train", config)
    batch = [dataset[0], dataset[1]]

    collated = collate_multimodal(batch)

    assert collated["wsi_features"] is not None
    assert collated["wsi_features"].shape == (2, 100, 512)
    assert collated["wsi_mask"].shape == (2, 100)
    assert collated["wsi_mask"][0].all()
    assert not collated["wsi_mask"][1].any()  # Second sample has no WSI

    assert collated["genomic"] is not None
    assert collated["genomic"].shape == (2, 1000)
    assert collated["genomic_mask"].sum().item() == 1  # Only first sample has genomic

    assert collated["clinical_text"] is not None
    assert collated["clinical_text"].shape == (2, 100)
    assert collated["clinical_mask"].shape == (2, 100)
    assert collated["clinical_mask"][0].any()
    assert not collated["clinical_mask"][1].any()  # Only first sample has clinical

    assert collated["label"].shape == (2,)
    assert len(collated["patient_ids"]) == 2


def test_collate_multimodal_contract_with_model(temp_data_dir):
    """Regression test: collated batch should work with model/trainer contract."""
    config = DictConfig(
        {
            "wsi_enabled": True,
            "genomic_enabled": True,
            "clinical_text_enabled": True,
            "max_text_length": 100,
        }
    )

    dataset = MultimodalDataset(temp_data_dir, "train", config)
    collated = collate_multimodal([dataset[0], dataset[1]])

    # Trainer expects singular key.
    assert "label" in collated
    assert "labels" not in collated

    model_batch = {k: v for k, v in collated.items() if k != "label"}
    model = MultimodalFusionModel(
        wsi_config={"input_dim": 512, "hidden_dim": 64, "output_dim": 32, "num_heads": 4, "num_layers": 1},
        genomic_config={"input_dim": 1000, "hidden_dims": [128], "output_dim": 32, "dropout": 0.1, "use_batch_norm": True},
        clinical_config={
            "vocab_size": 1000,
            "embed_dim": 32,
            "hidden_dim": 64,
            "output_dim": 32,
            "num_heads": 4,
            "num_layers": 1,
            "max_seq_length": 100,
            "dropout": 0.1,
            "pooling": "mean",
        },
        fusion_config={"embed_dim": 32, "num_heads": 4, "dropout": 0.1, "modalities": ["wsi", "genomic", "clinical"]},
        embed_dim=32,
    )
    model.eval()

    with torch.no_grad():
        fused = model(model_batch)

    assert fused.shape == (2, 32)


def test_collate_temporal(temp_temporal_data_dir):
    """Test collation function for temporal data."""
    config = DictConfig(
        {
            "min_slides_per_patient": 2,
            "wsi_enabled": True,
            "genomic_enabled": False,
            "clinical_text_enabled": False,
        }
    )

    dataset = TemporalDataset(temp_temporal_data_dir, "train", config)
    batch = [dataset[0]]

    collated = collate_temporal(batch)

    assert len(collated["slide_sequences"]) == 1
    assert len(collated["slide_sequences"][0]) == 3
    assert collated["sequence_lengths"].shape == (1,)
    assert collated["sequence_lengths"][0].item() == 3
    assert len(collated["timestamps"]) == 1
    assert collated["timestamps"][0].shape == (3,)
    assert collated["label"].shape == (1,)
    assert len(collated["patient_ids"]) == 1
