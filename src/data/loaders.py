"""
Data loading utilities for multimodal pathology data.

This module provides dataset classes for loading WSI features, genomic data,
and clinical text with support for missing modalities and temporal sequences.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import h5py
import numpy as np
import torch
from omegaconf import DictConfig
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


class MultimodalDataset(Dataset):
    """
    Loads WSI features, genomic data, and clinical text for a patient cohort.

    Handles missing modalities by returning None for unavailable data.
    Supports flexible data formats (HDF5 for WSI, JSON for metadata, CSV-like for genomics).
    """

    def __init__(self, data_dir: Path, split: str, config: DictConfig):
        """
        Initialize multimodal dataset.

        Args:
            data_dir: Root directory containing processed data
            split: One of 'train', 'val', 'test'
            config: Configuration dict with modality settings
                Expected keys:
                - wsi_enabled: bool
                - genomic_enabled: bool
                - clinical_text_enabled: bool
                - wsi_feature_dim: int
                - genomic_feature_dim: int
                - max_text_length: int
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.config = config

        # Load metadata for this split
        metadata_path = self.data_dir / f"{split}_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        self.samples = self.metadata["samples"]

        # Paths to modality data
        self.wsi_dir = self.data_dir / "wsi_features"
        self.genomic_dir = self.data_dir / "genomic"
        self.clinical_dir = self.data_dir / "clinical_text"

    def __len__(self) -> int:
        """Returns number of samples in split."""
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Optional[torch.Tensor]]:
        """
        Load multimodal data for a single sample.

        Returns:
            Dictionary containing:
                - 'wsi_features': Tensor [num_patches, feature_dim] or None
                - 'genomic': Tensor [num_genes] or None
                - 'clinical_text': Tensor [seq_len] or None
                - 'label': Tensor (scalar or multi-label)
                - 'patient_id': str
                - 'timestamp': Optional[float]
        """
        sample_info = self.samples[idx]
        patient_id = sample_info["patient_id"]

        result = {
            "patient_id": patient_id,
            "label": torch.tensor(sample_info["label"], dtype=torch.long),
            "timestamp": sample_info.get("timestamp", None),
        }

        # Load WSI features
        if self.config.get("wsi_enabled", True):
            wsi_features = self._load_wsi_features(patient_id, sample_info)
            result["wsi_features"] = wsi_features
        else:
            result["wsi_features"] = None

        # Load genomic data
        if self.config.get("genomic_enabled", True):
            genomic_features = self._load_genomic_features(patient_id, sample_info)
            result["genomic"] = genomic_features
        else:
            result["genomic"] = None

        # Load clinical text
        if self.config.get("clinical_text_enabled", True):
            clinical_text = self._load_clinical_text(patient_id, sample_info)
            result["clinical_text"] = clinical_text
        else:
            result["clinical_text"] = None

        return result

    def _load_wsi_features(self, patient_id: str, sample_info: Dict) -> Optional[torch.Tensor]:
        """
        Load WSI features from HDF5 file.

        Args:
            patient_id: Patient identifier
            sample_info: Sample metadata dictionary

        Returns:
            Tensor [num_patches, feature_dim] or None if not available
        """
        wsi_file = sample_info.get("wsi_file", None)
        if wsi_file is None:
            return None

        wsi_path = self.wsi_dir / wsi_file
        if not wsi_path.exists():
            return None

        try:
            with h5py.File(wsi_path, "r") as f:
                features = f["features"][:]
                features = torch.from_numpy(features).float()
            return features
        except Exception as e:
            logger.warning(f"Failed to load WSI features for {patient_id}: {e}", exc_info=True)
            return None

    def _load_genomic_features(self, patient_id: str, sample_info: Dict) -> Optional[torch.Tensor]:
        """
        Load genomic features from numpy file.

        Args:
            patient_id: Patient identifier
            sample_info: Sample metadata dictionary

        Returns:
            Tensor [num_genes] or None if not available
        """
        genomic_file = sample_info.get("genomic_file", None)
        if genomic_file is None:
            return None

        genomic_path = self.genomic_dir / genomic_file
        if not genomic_path.exists():
            return None

        try:
            features = np.load(genomic_path)
            features = torch.from_numpy(features).float()
            return features
        except Exception as e:
            logger.warning(f"Failed to load genomic features for {patient_id}: {e}", exc_info=True)
            return None

    def _load_clinical_text(self, patient_id: str, sample_info: Dict) -> Optional[torch.Tensor]:
        """
        Load clinical text token IDs from numpy file.

        Args:
            patient_id: Patient identifier
            sample_info: Sample metadata dictionary

        Returns:
            Tensor [seq_len] or None if not available
        """
        clinical_file = sample_info.get("clinical_file", None)
        if clinical_file is None:
            return None

        clinical_path = self.clinical_dir / clinical_file
        if not clinical_path.exists():
            return None

        try:
            token_ids = np.load(clinical_path)
            token_ids = torch.from_numpy(token_ids).long()

            # Truncate or pad to max_text_length if specified
            max_len = self.config.get("max_text_length", None)
            if max_len is not None:
                if len(token_ids) > max_len:
                    token_ids = token_ids[:max_len]
                elif len(token_ids) < max_len:
                    # Pad with zeros
                    padding = torch.zeros(max_len - len(token_ids), dtype=torch.long)
                    token_ids = torch.cat([token_ids, padding])

            return token_ids
        except Exception as e:
            logger.warning(f"Failed to load clinical text for {patient_id}: {e}", exc_info=True)
            return None


class TemporalDataset(Dataset):
    """
    Groups multiple slides from same patient for temporal reasoning.

    Returns sequences of slides ordered by timestamp for analyzing
    disease progression across time.
    """

    def __init__(self, data_dir: Path, split: str, config: DictConfig):
        """
        Initialize temporal dataset.

        Args:
            data_dir: Root directory containing processed data
            split: One of 'train', 'val', 'test'
            config: Configuration dict with temporal settings
                Expected keys:
                - max_slides_per_patient: int (optional)
                - min_slides_per_patient: int (default: 2)
                - wsi_enabled: bool
                - genomic_enabled: bool
                - clinical_text_enabled: bool
        """
        self.data_dir = Path(data_dir)
        self.split = split
        self.config = config

        # Load temporal metadata
        metadata_path = self.data_dir / f"{split}_temporal_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Temporal metadata file not found: {metadata_path}")

        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        # Group slides by patient
        self.patient_sequences = self._build_patient_sequences()

        # Paths to modality data
        self.wsi_dir = self.data_dir / "wsi_features"
        self.genomic_dir = self.data_dir / "genomic"
        self.clinical_dir = self.data_dir / "clinical_text"

    def _build_patient_sequences(self) -> List[Dict[str, Any]]:
        """
        Build patient sequences from metadata.

        Groups slides by patient and sorts by timestamp.
        Filters patients with insufficient slides.

        Returns:
            List of patient sequence dictionaries
        """
        # Group by patient
        patient_slides = {}
        for slide in self.metadata["slides"]:
            patient_id = slide["patient_id"]
            if patient_id not in patient_slides:
                patient_slides[patient_id] = []
            patient_slides[patient_id].append(slide)

        # Sort each patient's slides by timestamp and filter
        min_slides = self.config.get("min_slides_per_patient", 2)
        max_slides = self.config.get("max_slides_per_patient", None)

        sequences = []
        for patient_id, slides in patient_slides.items():
            # Sort by timestamp
            slides_sorted = sorted(slides, key=lambda x: x.get("timestamp", 0))

            # Filter by minimum slides
            if len(slides_sorted) < min_slides:
                continue

            # Limit to max slides if specified
            if max_slides is not None and len(slides_sorted) > max_slides:
                slides_sorted = slides_sorted[:max_slides]

            sequences.append(
                {
                    "patient_id": patient_id,
                    "slides": slides_sorted,
                    "label": slides_sorted[-1]["label"],  # Use label from most recent slide
                }
            )

        return sequences

    def __len__(self) -> int:
        """Returns number of patients in split."""
        return len(self.patient_sequences)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Load temporal sequence for a patient.

        Returns:
            Dictionary containing:
                - 'slide_sequence': List[Dict] - ordered list of slide data
                - 'timestamps': Tensor [num_slides]
                - 'patient_id': str
                - 'label': Tensor
        """
        patient_seq = self.patient_sequences[idx]
        patient_id = patient_seq["patient_id"]
        slides = patient_seq["slides"]

        # Load data for each slide in sequence
        slide_sequence = []
        timestamps = []

        for slide_info in slides:
            slide_data = {}

            # Load WSI features
            if self.config.get("wsi_enabled", True):
                wsi_features = self._load_wsi_features(slide_info)
                slide_data["wsi_features"] = wsi_features
            else:
                slide_data["wsi_features"] = None

            # Load genomic data
            if self.config.get("genomic_enabled", True):
                genomic_features = self._load_genomic_features(slide_info)
                slide_data["genomic"] = genomic_features
            else:
                slide_data["genomic"] = None

            # Load clinical text
            if self.config.get("clinical_text_enabled", True):
                clinical_text = self._load_clinical_text(slide_info)
                slide_data["clinical_text"] = clinical_text
            else:
                slide_data["clinical_text"] = None

            slide_data["slide_id"] = slide_info.get("slide_id", "")
            slide_sequence.append(slide_data)
            timestamps.append(slide_info.get("timestamp", 0.0))

        return {
            "slide_sequence": slide_sequence,
            "timestamps": torch.tensor(timestamps, dtype=torch.float32),
            "patient_id": patient_id,
            "label": torch.tensor(patient_seq["label"], dtype=torch.long),
        }

    def _load_wsi_features(self, slide_info: Dict) -> Optional[torch.Tensor]:
        """Load WSI features for a single slide."""
        wsi_file = slide_info.get("wsi_file", None)
        if wsi_file is None:
            return None

        wsi_path = self.wsi_dir / wsi_file
        if not wsi_path.exists():
            return None

        try:
            with h5py.File(wsi_path, "r") as f:
                features = f["features"][:]
                features = torch.from_numpy(features).float()
            return features
        except Exception:
            return None

    def _load_genomic_features(self, slide_info: Dict) -> Optional[torch.Tensor]:
        """Load genomic features for a single slide."""
        genomic_file = slide_info.get("genomic_file", None)
        if genomic_file is None:
            return None

        genomic_path = self.genomic_dir / genomic_file
        if not genomic_path.exists():
            return None

        try:
            features = np.load(genomic_path)
            features = torch.from_numpy(features).float()
            return features
        except Exception:
            return None

    def _load_clinical_text(self, slide_info: Dict) -> Optional[torch.Tensor]:
        """Load clinical text for a single slide."""
        clinical_file = slide_info.get("clinical_file", None)
        if clinical_file is None:
            return None

        clinical_path = self.clinical_dir / clinical_file
        if not clinical_path.exists():
            return None

        try:
            token_ids = np.load(clinical_path)
            token_ids = torch.from_numpy(token_ids).long()

            # Truncate or pad to max_text_length if specified
            max_len = self.config.get("max_text_length", None)
            if max_len is not None:
                if len(token_ids) > max_len:
                    token_ids = token_ids[:max_len]
                elif len(token_ids) < max_len:
                    padding = torch.zeros(max_len - len(token_ids), dtype=torch.long)
                    token_ids = torch.cat([token_ids, padding])

            return token_ids
        except Exception:
            return None


def collate_multimodal(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for MultimodalDataset with variable-length sequences.

    Handles missing modalities and variable numbers of patches/tokens.

    Args:
        batch: List of samples from MultimodalDataset

    Returns:
        Batched dictionary with:
            - 'wsi_features': Tensor [batch_size, max_num_patches, feature_dim] or None
            - 'wsi_mask': Tensor [batch_size, max_num_patches] where True indicates valid patches
            - 'genomic': Tensor [batch_size, num_genes] or None
            - 'genomic_mask': Tensor [batch_size] indicating availability
            - 'clinical_text': Tensor [batch_size, max_seq_len] or None
            - 'clinical_mask': Tensor [batch_size] indicating availability
            - 'label': Tensor [batch_size]
            - 'patient_ids': List[str]
            - 'timestamps': List[Optional[float]]
    """
    batch_size = len(batch)

    # Handle empty batch
    if batch_size == 0:
        return {
            "wsi_features": None,
            "wsi_mask": None,
            "genomic": None,
            "genomic_mask": torch.zeros(0, dtype=torch.bool),
            "clinical_text": None,
            "clinical_mask": None,
            "label": torch.zeros(0, dtype=torch.long),
            "patient_ids": [],
            "timestamps": [],
        }

    # Collect patient IDs and labels
    patient_ids = [sample["patient_id"] for sample in batch]
    labels = torch.stack([sample["label"] for sample in batch])
    timestamps = [sample["timestamp"] for sample in batch]

    # Collate WSI features (pad to max patches in batch)
    wsi_mask = torch.zeros(batch_size, dtype=torch.bool)
    wsi_list = []
    max_patches = 0
    feature_dim = None

    for i, sample in enumerate(batch):
        if sample["wsi_features"] is not None:
            wsi_list.append(sample["wsi_features"])
            wsi_mask[i] = True
            max_patches = max(max_patches, sample["wsi_features"].shape[0])
            if feature_dim is None:
                feature_dim = sample["wsi_features"].shape[1]
        else:
            wsi_list.append(None)

    if wsi_mask.any() and feature_dim is not None:
        # Create padded tensor and per-patch mask
        wsi_features = torch.zeros(batch_size, max_patches, feature_dim, dtype=torch.float32)
        wsi_patch_mask = torch.zeros(batch_size, max_patches, dtype=torch.bool)
        for i, wsi in enumerate(wsi_list):
            if wsi is not None:
                num_patches = wsi.shape[0]
                wsi_features[i, :num_patches] = wsi
                wsi_patch_mask[i, :num_patches] = True
    else:
        wsi_features = None
        wsi_patch_mask = None

    # Collate genomic features
    genomic_mask = torch.zeros(batch_size, dtype=torch.bool)
    for i, sample in enumerate(batch):
        if sample["genomic"] is not None:
            genomic_mask[i] = True

    if genomic_mask.any():
        genomic_dim = batch[genomic_mask.nonzero()[0].item()]["genomic"].shape[0]
        genomic = torch.zeros(batch_size, genomic_dim, dtype=torch.float32)
        for i, sample in enumerate(batch):
            if sample["genomic"] is not None:
                genomic[i] = sample["genomic"]
    else:
        genomic = None

    # Collate clinical text
    clinical_list = [
        sample["clinical_text"] for sample in batch if sample["clinical_text"] is not None
    ]
    clinical_mask = None
    if len(clinical_list) > 0:
        # Pad to max length in batch
        max_len = max(text.shape[0] for text in clinical_list)
        clinical_padded = []
        for text in clinical_list:
            if text.shape[0] < max_len:
                padding = torch.zeros(max_len - text.shape[0], dtype=torch.long)
                text = torch.cat([text, padding])
            clinical_padded.append(text)

        torch.stack(clinical_padded)
        # Create full batch tensor with zeros for missing
        clinical_batch = torch.zeros(batch_size, max_len, dtype=torch.long)
        clinical_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        clinical_idx = 0
        for i, sample in enumerate(batch):
            if sample["clinical_text"] is not None:
                clinical_batch[i] = clinical_padded[clinical_idx]
                clinical_mask[i] = clinical_padded[clinical_idx] != 0
                clinical_idx += 1
        clinical_text = clinical_batch
    else:
        clinical_text = None

    return {
        "wsi_features": wsi_features,
        "wsi_mask": wsi_patch_mask,
        "genomic": genomic,
        "genomic_mask": genomic_mask,
        "clinical_text": clinical_text,
        "clinical_mask": clinical_mask,
        "label": labels,
        "patient_ids": patient_ids,
        "timestamps": timestamps,
    }


def collate_temporal(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Collate function for TemporalDataset with variable-length slide sequences.

    Handles variable numbers of slides per patient and missing modalities.

    Args:
        batch: List of samples from TemporalDataset

    Returns:
        Batched dictionary with:
            - 'slide_sequences': List[List[Dict]] - nested list structure
            - 'sequence_lengths': Tensor [batch_size] - number of slides per patient
            - 'timestamps': List[Tensor] - timestamps for each patient's slides
            - 'label': Tensor [batch_size]
            - 'patient_ids': List[str]
    """
    len(batch)

    # Collect patient IDs and labels
    patient_ids = [sample["patient_id"] for sample in batch]
    labels = torch.stack([sample["label"] for sample in batch])

    # Collect slide sequences (keep nested structure)
    slide_sequences = [sample["slide_sequence"] for sample in batch]
    sequence_lengths = torch.tensor([len(seq) for seq in slide_sequences], dtype=torch.long)

    # Collect timestamps
    timestamps = [sample["timestamps"] for sample in batch]

    return {
        "slide_sequences": slide_sequences,
        "sequence_lengths": sequence_lengths,
        "timestamps": timestamps,
        "label": labels,
        "patient_ids": patient_ids,
    }
