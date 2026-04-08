"""
CAMELYON Whole-Slide Image Dataset Integration.

This module provides dataset classes for the CAMELYON16 and CAMELYON17 challenges,
which focus on metastasis detection in histopathology whole-slide images of
lymph node sections.

Dataset: https://camelyon17.grand-challenge.org/
Paper: Bandi et al. (2019), IEEE TMI 38(2), 550-560

The implementation provides:
- Slide-level indexing and metadata management
- Patch/tile sampling from WSI with spatial coordinates
- Annotation/mask loading for ground truth regions
- HDF5 feature caching for pre-extracted patch features
- Slide-level aggregation helpers for MIL training
"""

import json
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


@dataclass
class SlideMetadata:
    """Metadata for a single WSI slide.

    Attributes:
        slide_id: Unique identifier for the slide
        patient_id: Patient identifier (may have multiple slides per patient)
        file_path: Path to the WSI file (typically .tif or .svs)
        label: Slide-level label (0=normal, 1=metastasis for CAMELYON)
        split: Data split ('train', 'val', 'test')
        annotation_path: Optional path to annotation XML/mask file
        width: Slide width in pixels at base level
        height: Slide height in pixels at base level
        magnification: Objective magnification (e.g., 40 for 40x)
        mpp: Microns per pixel at base level
    """

    slide_id: str
    patient_id: str
    file_path: str
    label: int
    split: str
    annotation_path: Optional[str] = None
    width: Optional[int] = None
    height: Optional[int] = None
    magnification: Optional[float] = None
    mpp: Optional[float] = None


class CAMELYONSlideIndex:
    """Index for managing CAMELYON slide metadata and train/val/test splits.

    This class provides slide-level indexing without loading the actual
    WSI files, enabling efficient dataset discovery and split management.

    Example:
        >>> index = CAMELYONSlideIndex.from_directory(
        ...     root_dir='data/camelyon16',
        ...     annotation_dir='data/camelyon16/annotations'
        ... )
        >>> index.save('data/camelyon16/slide_index.json')
        >>>
        >>> # Load existing index
        >>> index = CAMELYONSlideIndex.load('data/camelyon16/slide_index.json')
        >>> train_slides = index.get_slides_by_split('train')
    """

    def __init__(self, slides: List[SlideMetadata]):
        """Initialize with a list of slide metadata.

        Args:
            slides: List of SlideMetadata objects
        """
        self.slides = slides
        self._slide_by_id: Dict[str, SlideMetadata] = {s.slide_id: s for s in slides}

    def __len__(self) -> int:
        """Return total number of slides."""
        return len(self.slides)

    def __getitem__(self, slide_id: str) -> SlideMetadata:
        """Get slide metadata by ID."""
        return self._slide_by_id[slide_id]

    def get_slides_by_split(self, split: str) -> List[SlideMetadata]:
        """Get all slides for a specific split.

        Args:
            split: One of 'train', 'val', 'test'

        Returns:
            List of SlideMetadata for the split
        """
        return [s for s in self.slides if s.split == split]

    def get_slides_by_patient(self, patient_id: str) -> List[SlideMetadata]:
        """Get all slides for a specific patient.

        Args:
            patient_id: Patient identifier

        Returns:
            List of SlideMetadata for the patient
        """
        return [s for s in self.slides if s.patient_id == patient_id]

    def get_annotated_slides(self) -> List[SlideMetadata]:
        """Get slides with annotation files.

        Returns:
            List of SlideMetadata with non-null annotation_path
        """
        return [s for s in self.slides if s.annotation_path is not None]

    def save(self, output_path: Union[str, Path]) -> None:
        """Save slide index to JSON file.

        Args:
            output_path: Path to save the index JSON
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "dataset": "CAMELYON",
            "num_slides": len(self.slides),
            "slides": [asdict(s) for s in self.slides],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved slide index to {output_path}")

    @classmethod
    def load(cls, index_path: Union[str, Path]) -> "CAMELYONSlideIndex":
        """Load slide index from JSON file.

        Args:
            index_path: Path to the index JSON file

        Returns:
            CAMELYONSlideIndex instance
        """
        with open(index_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        slides = [SlideMetadata(**s) for s in data["slides"]]
        logger.info(f"Loaded {len(slides)} slides from {index_path}")
        return cls(slides)

    @classmethod
    def from_directory(
        cls,
        root_dir: Union[str, Path],
        slide_pattern: str = "*.tif",
        annotation_dir: Optional[Union[str, Path]] = None,
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        seed: int = 42,
    ) -> "CAMELYONSlideIndex":
        """Create slide index by scanning a directory.

        This is a factory method that discovers slides and creates a basic
        index with random train/val/test splits. For CAMELYON16/17, you
        typically want to use the official splits instead.

        Args:
            root_dir: Directory containing WSI files
            slide_pattern: Glob pattern to match slide files
            annotation_dir: Optional directory with annotation files
            split_ratios: (train, val, test) ratios
            seed: Random seed for splitting

        Returns:
            CAMELYONSlideIndex with discovered slides
        """
        root_dir = Path(root_dir)
        annotation_dir = Path(annotation_dir) if annotation_dir else None

        slide_files = sorted(root_dir.glob(slide_pattern))

        # Create metadata for each slide
        np.random.seed(seed)
        splits = np.random.choice(["train", "val", "test"], size=len(slide_files), p=split_ratios)

        slides = []
        for i, slide_path in enumerate(slide_files):
            slide_id = slide_path.stem

            # Look for annotation file
            annotation_path = None
            if annotation_dir:
                ann_file = annotation_dir / f"{slide_id}.xml"
                if ann_file.exists():
                    annotation_path = str(ann_file)

            # For CAMELYON, patient_id is typically in the filename
            # e.g., "patient_001_node_0.tif" -> "patient_001"
            patient_id = slide_id.split("_node_")[0] if "_node_" in slide_id else slide_id

            slides.append(
                SlideMetadata(
                    slide_id=slide_id,
                    patient_id=patient_id,
                    file_path=str(slide_path),
                    label=-1,  # Unknown - will be set from annotations or metadata
                    split=splits[i],
                    annotation_path=annotation_path,
                )
            )

        logger.info(f"Indexed {len(slides)} slides from {root_dir}")
        return cls(slides)


class SlideAggregator:
    """Helper for aggregating patch-level predictions/features to slide-level.

    Useful for Multiple Instance Learning (MIL) where you need to:
    - Group patch predictions by slide
    - Aggregate patch features (attention, transformer pooling)
    - Compute slide-level metrics from patch outputs

    Example:
        >>> aggregator = SlideAggregator()
        >>> for batch in dataloader:
        ...     slide_ids = batch['slide_id']
        ...     patch_preds = model(batch['features'])
        ...     aggregator.add_predictions(slide_ids, patch_preds)
        >>> slide_preds = aggregator.get_slide_predictions('attention')  # or 'max', 'mean'
    """

    def __init__(self):
        """Initialize empty aggregator."""
        self.slide_data: Dict[str, Dict[str, List]] = {}

    def add_predictions(
        self,
        slide_ids: List[str],
        predictions: torch.Tensor,
        attention_weights: Optional[torch.Tensor] = None,
    ) -> None:
        """Add patch predictions for a batch.

        Args:
            slide_ids: List of slide IDs (one per patch)
            predictions: Tensor [num_patches, ...] of patch predictions
            attention_weights: Optional attention weights [num_patches] or [num_patches, num_patches]
        """
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.detach().cpu().numpy()
        if attention_weights is not None and isinstance(attention_weights, torch.Tensor):
            attention_weights = attention_weights.detach().cpu().numpy()

        for i, slide_id in enumerate(slide_ids):
            if slide_id not in self.slide_data:
                self.slide_data[slide_id] = {
                    "predictions": [],
                    "attention_weights": [] if attention_weights is not None else None,
                }

            self.slide_data[slide_id]["predictions"].append(predictions[i])
            if attention_weights is not None:
                self.slide_data[slide_id]["attention_weights"].append(attention_weights[i])

    def add_features(
        self,
        slide_ids: List[str],
        features: torch.Tensor,
        coordinates: Optional[torch.Tensor] = None,
    ) -> None:
        """Add patch features for a batch.

        Args:
            slide_ids: List of slide IDs (one per patch)
            features: Tensor [num_patches, feature_dim] of patch features
            coordinates: Optional tensor [num_patches, 2] of (x, y) coordinates
        """
        if isinstance(features, torch.Tensor):
            features = features.detach().cpu().numpy()
        if coordinates is not None and isinstance(coordinates, torch.Tensor):
            coordinates = coordinates.detach().cpu().numpy()

        for i, slide_id in enumerate(slide_ids):
            if slide_id not in self.slide_data:
                self.slide_data[slide_id] = {
                    "features": [],
                    "coordinates": [] if coordinates is not None else None,
                }

            self.slide_data[slide_id]["features"].append(features[i])
            if coordinates is not None:
                self.slide_data[slide_id]["coordinates"].append(coordinates[i])

    def get_slide_predictions(
        self,
        aggregation: str = "attention",
    ) -> Dict[str, Union[float, np.ndarray]]:
        """Get aggregated slide-level predictions.

        Args:
            aggregation: Method to aggregate patches ('attention', 'max', 'mean', 'sum')

        Returns:
            Dictionary mapping slide_id to aggregated prediction
        """
        slide_preds = {}

        for slide_id, data in self.slide_data.items():
            if "predictions" not in data:
                continue

            preds = np.stack(data["predictions"])  # [num_patches, ...]

            if aggregation == "attention" and data.get("attention_weights") is not None:
                # Attention-weighted aggregation
                attn = np.stack(data["attention_weights"])  # [num_patches]
                attn = attn / attn.sum()  # Normalize

                if preds.ndim == 1:
                    slide_preds[slide_id] = np.dot(attn, preds)
                else:
                    slide_preds[slide_id] = np.dot(attn, preds.reshape(preds.shape[0], -1)).reshape(
                        preds.shape[1:]
                    )

            elif aggregation == "max":
                slide_preds[slide_id] = np.max(preds, axis=0)

            elif aggregation == "mean":
                slide_preds[slide_id] = np.mean(preds, axis=0)

            elif aggregation == "sum":
                slide_preds[slide_id] = np.sum(preds, axis=0)

            else:
                raise ValueError(f"Unknown aggregation: {aggregation}")

        return slide_preds

    def get_slide_features(self, slide_id: str) -> Optional[Dict[str, np.ndarray]]:
        """Get all aggregated data for a specific slide.

        Args:
            slide_id: Slide identifier

        Returns:
            Dictionary with 'features', 'predictions', 'coordinates' arrays,
            or None if slide not found
        """
        if slide_id not in self.slide_data:
            return None

        data = self.slide_data[slide_id]
        result = {}

        for key in ["features", "predictions", "coordinates", "attention_weights"]:
            if key in data and data[key] is not None:
                result[key] = np.stack(data[key])

        return result

    def get_slide_ids(self) -> List[str]:
        """Get list of all slide IDs in aggregator."""
        return list(self.slide_data.keys())

    def clear(self) -> None:
        """Clear all aggregated data."""
        self.slide_data.clear()


class CAMELYONSlideDataset(Dataset):
    """Dataset that returns complete slides with all patch features.
    
    This dataset loads all patches for a slide from HDF5 feature files,
    enabling true slide-level training where each sample represents a
    complete whole-slide image with all its patches.
    
    Args:
        slide_index: CAMELYONSlideIndex with slide metadata
        features_dir: Directory containing HDF5 feature files
        split: Which split to load ('train', 'val', 'test')
        transform: Optional transform for features
        
    Returns:
        Dictionary containing:
            - 'slide_id': str
            - 'patient_id': str
            - 'label': int (slide-level label)
            - 'features': Tensor [num_patches, feature_dim]
            - 'coordinates': Tensor [num_patches, 2]
            - 'num_patches': int
            
    Example:
        >>> index = CAMELYONSlideIndex.load('data/camelyon16/slide_index.json')
        >>> dataset = CAMELYONSlideDataset(
        ...     slide_index=index,
        ...     features_dir='data/camelyon16/features',
        ...     split='train'
        ... )
        >>> sample = dataset[0]
        >>> print(sample['features'].shape)  # [num_patches, feature_dim]
        >>> print(sample['num_patches'])  # Number of patches in this slide
    """
    
    def __init__(
        self,
        slide_index: CAMELYONSlideIndex,
        features_dir: Union[str, Path],
        split: str = "train",
        transform: Optional[Callable] = None,
    ):
        self.slide_index = slide_index
        self.features_dir = Path(features_dir)
        self.split = split
        self.transform = transform
        
        # Get slides for this split
        self.slides = slide_index.get_slides_by_split(split)
        
        # Validate feature files exist
        self.valid_slides = []
        for slide in self.slides:
            feature_file = self.features_dir / f"{slide.slide_id}.h5"
            if feature_file.exists():
                self.valid_slides.append(slide)
            else:
                logger.warning(f"Feature file not found: {feature_file}")
        
        if len(self.valid_slides) == 0:
            raise ValueError(
                f"No valid feature files found in {self.features_dir} for {split} split. "
                f"Please ensure HDF5 feature files exist for the slides in the index."
            )
        
        logger.info(
            f"Loaded {len(self.valid_slides)} slides for {split} split"
        )
    
    def __len__(self) -> int:
        """Return number of slides."""
        return len(self.valid_slides)
    
    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        """Get all patches for a single slide."""
        slide = self.valid_slides[idx]
        feature_file = self.features_dir / f"{slide.slide_id}.h5"
        
        try:
            with h5py.File(feature_file, "r") as f:
                if "features" not in f or "coordinates" not in f:
                    raise KeyError(f"HDF5 file missing required datasets: {feature_file}")
                
                features = torch.tensor(f["features"][:], dtype=torch.float32)
                coordinates = torch.tensor(f["coordinates"][:], dtype=torch.int32)
                
                if features.shape[0] != coordinates.shape[0]:
                    raise ValueError(
                        f"Mismatched patch counts in {feature_file}: "
                        f"features={features.shape[0]}, coordinates={coordinates.shape[0]}"
                    )
        except Exception as e:
            logger.error(f"Error loading slide {slide.slide_id}: {e}")
            raise
        
        if self.transform:
            features = self.transform(features)
        
        return {
            "slide_id": slide.slide_id,
            "patient_id": slide.patient_id,
            "label": slide.label,
            "features": features,  # [num_patches, feature_dim]
            "coordinates": coordinates,  # [num_patches, 2]
            "num_patches": int(features.shape[0]),
        }


class CAMELYONPatchDataset(Dataset):
    """Dataset for patch-level sampling from CAMELYON slides.

    This dataset loads pre-extracted patch features from HDF5 files
    (following the format specified in data/README.md) rather than
    extracting patches on-the-fly from raw WSIs.

    Args:
        slide_index: CAMELYONSlideIndex with slide metadata
        features_dir: Directory containing HDF5 feature files
        split: Which split to load ('train', 'val', 'test')
        transform: Optional transform for features

    Example:
        >>> index = CAMELYONSlideIndex.load('data/camelyon16/slide_index.json')
        >>> dataset = CAMELYONPatchDataset(
        ...     slide_index=index,
        ...     features_dir='data/camelyon16/features',
        ...     split='train'
        ... )
        >>> sample = dataset[0]
        >>> print(sample['features'].shape)  # [num_patches, feature_dim]
        >>> print(sample['label'])  # Slide-level label
    """

    def __init__(
        self,
        slide_index: CAMELYONSlideIndex,
        features_dir: Union[str, Path],
        split: str = "train",
        transform: Optional[Callable] = None,
    ):
        self.slide_index = slide_index
        self.features_dir = Path(features_dir)
        self.split = split
        self.transform = transform

        # Get slides for this split
        self.slides = slide_index.get_slides_by_split(split)

        # Build patch-level index
        self.patch_index: List[Tuple[str, int]] = []  # (slide_id, patch_idx)
        self.slide_patch_counts: Dict[str, int] = {}

        for slide in self.slides:
            feature_file = self.features_dir / f"{slide.slide_id}.h5"
            if feature_file.exists():
                with h5py.File(feature_file, "r") as f:
                    num_patches = f["features"].shape[0]
                    self.slide_patch_counts[slide.slide_id] = num_patches
                    for i in range(num_patches):
                        self.patch_index.append((slide.slide_id, i))
            else:
                logger.warning(f"Feature file not found: {feature_file}")

        logger.info(
            f"Loaded {len(self.slides)} slides with "
            f"{len(self.patch_index)} total patches for {split} split"
        )

    def __len__(self) -> int:
        """Return total number of patches across all slides."""
        return len(self.patch_index)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        """Get a single patch sample.

        Returns:
            Dictionary containing:
                - 'features': Tensor [feature_dim] - patch feature vector
                - 'coordinates': Tensor [2] - (x, y) coordinates in slide
                - 'slide_id': str - identifier for the source slide
                - 'patient_id': str - patient identifier
                - 'label': int - slide-level label
                - 'patch_idx': int - index within the slide
        """
        slide_id, patch_idx = self.patch_index[idx]
        slide = self.slide_index[slide_id]

        feature_file = self.features_dir / f"{slide_id}.h5"

        with h5py.File(feature_file, "r") as f:
            features = f["features"][patch_idx]  # [feature_dim]
            coordinates = f["coordinates"][patch_idx]  # [2]

        features = torch.tensor(features, dtype=torch.float32)
        coordinates = torch.tensor(coordinates, dtype=torch.int32)

        if self.transform:
            features = self.transform(features)

        return {
            "features": features,
            "coordinates": coordinates,
            "slide_id": slide_id,
            "patient_id": slide.patient_id,
            "label": slide.label,
            "patch_idx": patch_idx,
        }

    def get_slide_features(self, slide_id: str) -> Optional[torch.Tensor]:
        """Get all patches for a specific slide.

        Args:
            slide_id: Slide identifier

        Returns:
            Tensor [num_patches, feature_dim] or None if not found
        """
        feature_file = self.features_dir / f"{slide_id}.h5"
        if not feature_file.exists():
            return None

        with h5py.File(feature_file, "r") as f:
            features = f["features"][:]  # [num_patches, feature_dim]

        return torch.tensor(features, dtype=torch.float32)

    def get_slide_patch_data(
        self, slide_id: str
    ) -> Optional[Dict[str, Union[torch.Tensor, str, int]]]:
        """Get all patch features and coordinates for one slide.

        Args:
            slide_id: Slide identifier

        Returns:
            Dictionary with slide-level patch data or None if features are missing.
        """
        feature_file = self.features_dir / f"{slide_id}.h5"
        if not feature_file.exists():
            return None

        slide = self.slide_index[slide_id]
        with h5py.File(feature_file, "r") as f:
            features = torch.tensor(f["features"][:], dtype=torch.float32)
            coordinates = torch.tensor(f["coordinates"][:], dtype=torch.int32)

        return {
            "slide_id": slide_id,
            "patient_id": slide.patient_id,
            "label": slide.label,
            "features": features,
            "coordinates": coordinates,
            "num_patches": int(features.shape[0]),
        }

    def aggregate_slide_features(
        self, slide_id: str, method: str = "mean"
    ) -> Optional[torch.Tensor]:
        """Aggregate all patch features for one slide into a single vector.

        Args:
            slide_id: Slide identifier
            method: Aggregation method ('mean' or 'max')

        Returns:
            Aggregated feature tensor [feature_dim] or None if slide is missing.
        """
        slide_data = self.get_slide_patch_data(slide_id)
        if slide_data is None:
            return None

        features = slide_data["features"]
        if method == "mean":
            return features.mean(dim=0)
        if method == "max":
            return features.max(dim=0).values
        raise ValueError(f"Unknown aggregation method: {method}")


def collate_slide_bags(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List]]:
    """Collate function for variable-length slide bags.
    
    Pads all slides to the maximum number of patches in the batch.
    
    Args:
        batch: List of samples from CAMELYONSlideDataset
        
    Returns:
        Dictionary containing:
            - 'features': Tensor [batch_size, max_patches, feature_dim]
            - 'coordinates': Tensor [batch_size, max_patches, 2]
            - 'labels': Tensor [batch_size]
            - 'num_patches': Tensor [batch_size] - actual patch counts
            - 'slide_ids': List[str]
            - 'patient_ids': List[str]
    """
    # Extract components
    features_list = [item["features"] for item in batch]
    coordinates_list = [item["coordinates"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    num_patches = torch.tensor([item["num_patches"] for item in batch], dtype=torch.long)
    slide_ids = [item["slide_id"] for item in batch]
    patient_ids = [item["patient_id"] for item in batch]
    
    # Pad to max length
    max_patches = max(f.shape[0] for f in features_list)
    feature_dim = features_list[0].shape[1]
    batch_size = len(batch)
    
    padded_features = torch.zeros(batch_size, max_patches, feature_dim)
    padded_coordinates = torch.zeros(batch_size, max_patches, 2, dtype=torch.int32)
    
    for i, (features, coordinates) in enumerate(zip(features_list, coordinates_list)):
        n_patches = features.shape[0]
        padded_features[i, :n_patches, :] = features
        padded_coordinates[i, :n_patches, :] = coordinates
    
    return {
        "features": padded_features,
        "coordinates": padded_coordinates,
        "labels": labels,
        "num_patches": num_patches,
        "slide_ids": slide_ids,
        "patient_ids": patient_ids,
    }


def create_patch_index(
    slide_index: CAMELYONSlideIndex,
    features_dir: Union[str, Path],
    output_path: Union[str, Path],
    splits: List[str] = ["train", "val", "test"],
) -> Dict[str, int]:
    """Create a patch-level index for efficient sampling.

    This creates a JSON file mapping (slide_id, patch_idx) to a global
    patch index, useful for deterministic sampling and debugging.

    Args:
        slide_index: Slide index
        features_dir: Directory with HDF5 feature files
        output_path: Where to save the patch index JSON
        splits: Which splits to include

    Returns:
        Dictionary with patch counts per split
    """
    features_dir = Path(features_dir)
    output_path = Path(output_path)

    patch_index = {split: [] for split in splits}

    for split in splits:
        slides = slide_index.get_slides_by_split(split)
        global_idx = 0

        for slide in slides:
            feature_file = features_dir / f"{slide.slide_id}.h5"
            if not feature_file.exists():
                continue

            with h5py.File(feature_file, "r") as f:
                num_patches = f["features"].shape[0]

            for patch_idx in range(num_patches):
                patch_index[split].append(
                    {
                        "global_idx": global_idx,
                        "slide_id": slide.slide_id,
                        "patch_idx": patch_idx,
                        "patient_id": slide.patient_id,
                        "label": slide.label,
                    }
                )
                global_idx += 1

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(patch_index, f, indent=2)

    counts = {split: len(patch_index[split]) for split in splits}
    logger.info(f"Saved patch index to {output_path}: {counts}")
    return counts


def validate_feature_file(feature_path: Union[str, Path]) -> Dict[str, Union[bool, str, int]]:
    """Validate an HDF5 feature file has correct structure.

    Checks for:
    - File exists and is readable
    - Required datasets: 'features' and 'coordinates'
    - Compatible shapes between features and coordinates
    - Reasonable data types

    Args:
        feature_path: Path to HDF5 feature file

    Returns:
        Dictionary with validation results:
            - 'valid': bool - whether file is valid
            - 'error': str - error message if invalid
            - 'num_patches': int - number of patches if valid
            - 'feature_dim': int - feature dimension if valid
    """
    feature_path = Path(feature_path)
    result = {"valid": False, "error": None, "num_patches": 0, "feature_dim": 0}

    if not feature_path.exists():
        result["error"] = f"File not found: {feature_path}"
        return result

    try:
        with h5py.File(feature_path, "r") as f:
            # Check required datasets exist
            if "features" not in f:
                result["error"] = "Missing 'features' dataset"
                return result

            if "coordinates" not in f:
                result["error"] = "Missing 'coordinates' dataset"
                return result

            features = f["features"]
            coordinates = f["coordinates"]

            # Check shapes
            if features.ndim != 2:
                result["error"] = f"Features should be 2D, got {features.ndim}D"
                return result

            if coordinates.ndim != 2 or coordinates.shape[1] != 2:
                result["error"] = f"Coordinates should be [N, 2], got {coordinates.shape}"
                return result

            num_patches = features.shape[0]
            if coordinates.shape[0] != num_patches:
                result["error"] = (
                    f"Mismatched patch counts: features={num_patches}, coordinates={coordinates.shape[0]}"
                )
                return result

            result["valid"] = True
            result["num_patches"] = num_patches
            result["feature_dim"] = features.shape[1]

    except Exception as e:
        result["error"] = f"Error reading file: {e}"

    return result
