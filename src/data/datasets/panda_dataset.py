"""
PANDA (Prostate cANcer graDe Assessment) Dataset Implementation.

Dataset: Kaggle PANDA Challenge
Task: Gleason grading (ISUP grades 0-5) from prostate biopsy WSIs
Paper: Bulten et al. (2022), Lancet Oncology 23(2), 252-261

The PANDA dataset contains:
- 10,616 whole-slide images of prostate biopsies
- ISUP grades: 0 (benign) to 5 (most aggressive)
- Multiple data providers (Radboud, Karolinska)
- Gleason patterns annotated by pathologists

This implementation provides:
- Slide-level dataset with ISUP grade labels
- Patch extraction from WSI with tissue detection
- Feature caching for pre-extracted patches
- Ordinal regression support for grade prediction
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
class PANDASlideMetadata:
    """Metadata for a single PANDA slide.

    Attributes:
        slide_id: Unique identifier (image_id from Kaggle)
        file_path: Path to the WSI file (.tiff)
        isup_grade: ISUP grade (0-5)
        gleason_score: Primary + secondary Gleason pattern (e.g., "3+4")
        data_provider: Source institution (radboud or karolinska)
        split: Data split ('train', 'val', 'test')
        mask_path: Optional path to tissue mask
    """

    slide_id: str
    file_path: str
    isup_grade: int
    gleason_score: str
    data_provider: str
    split: str
    mask_path: Optional[str] = None


class PANDASlideIndex:
    """Index for managing PANDA slide metadata and splits.

    Example:
        >>> index = PANDASlideIndex.from_csv(
        ...     csv_path='data/panda/train.csv',
        ...     image_dir='data/panda/train_images'
        ... )
        >>> index.save('data/panda/slide_index.json')
    """

    def __init__(self, slides: List[PANDASlideMetadata]):
        self.slides = slides
        self._slide_by_id: Dict[str, PANDASlideMetadata] = {s.slide_id: s for s in slides}

    def __len__(self) -> int:
        return len(self.slides)

    def __getitem__(self, slide_id: str) -> PANDASlideMetadata:
        return self._slide_by_id[slide_id]

    def get_slides_by_split(self, split: str) -> List[PANDASlideMetadata]:
        return [s for s in self.slides if s.split == split]

    def get_slides_by_grade(self, isup_grade: int) -> List[PANDASlideMetadata]:
        return [s for s in self.slides if s.isup_grade == isup_grade]

    def get_slides_by_provider(self, provider: str) -> List[PANDASlideMetadata]:
        return [s for s in self.slides if s.data_provider == provider]

    def get_grade_distribution(self, split: Optional[str] = None) -> Dict[int, int]:
        """Get distribution of ISUP grades.

        Args:
            split: Optional split to filter by

        Returns:
            Dictionary mapping ISUP grade to count
        """
        slides = self.get_slides_by_split(split) if split else self.slides
        dist = {i: 0 for i in range(6)}
        for slide in slides:
            dist[slide.isup_grade] += 1
        return dist

    def save(self, output_path: Union[str, Path]) -> None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "dataset": "PANDA",
            "num_slides": len(self.slides),
            "grade_distribution": self.get_grade_distribution(),
            "slides": [asdict(s) for s in self.slides],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

        logger.info(f"Saved PANDA index to {output_path}")

    @classmethod
    def load(cls, index_path: Union[str, Path]) -> "PANDASlideIndex":
        with open(index_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        slides = [PANDASlideMetadata(**s) for s in data["slides"]]
        logger.info(f"Loaded {len(slides)} PANDA slides from {index_path}")
        return cls(slides)

    @classmethod
    def from_csv(
        cls,
        csv_path: Union[str, Path],
        image_dir: Union[str, Path],
        mask_dir: Optional[Union[str, Path]] = None,
        split_ratios: Tuple[float, float, float] = (0.7, 0.15, 0.15),
        stratify: bool = True,
        seed: int = 42,
    ) -> "PANDASlideIndex":
        """Create index from PANDA CSV file.

        Args:
            csv_path: Path to train.csv from Kaggle
            image_dir: Directory containing .tiff files
            mask_dir: Optional directory with tissue masks
            split_ratios: (train, val, test) ratios
            stratify: Whether to stratify splits by ISUP grade
            seed: Random seed

        Returns:
            PANDASlideIndex
        """
        import pandas as pd

        csv_path = Path(csv_path)
        image_dir = Path(image_dir)
        mask_dir = Path(mask_dir) if mask_dir else None

        # Load CSV
        df = pd.read_csv(csv_path)

        # Create splits
        np.random.seed(seed)

        if stratify:
            # Stratified split by ISUP grade
            splits = []
            for grade in range(6):
                grade_mask = df["isup_grade"] == grade
                grade_indices = df[grade_mask].index.tolist()
                n = len(grade_indices)

                n_train = int(n * split_ratios[0])
                n_val = int(n * split_ratios[1])

                np.random.shuffle(grade_indices)
                train_idx = grade_indices[:n_train]
                val_idx = grade_indices[n_train : n_train + n_val]
                test_idx = grade_indices[n_train + n_val :]

                splits.extend([(i, "train") for i in train_idx])
                splits.extend([(i, "val") for i in val_idx])
                splits.extend([(i, "test") for i in test_idx])

            split_dict = dict(splits)
        else:
            # Random split
            n = len(df)
            indices = np.random.permutation(n)
            n_train = int(n * split_ratios[0])
            n_val = int(n * split_ratios[1])

            split_dict = {}
            for i in indices[:n_train]:
                split_dict[i] = "train"
            for i in indices[n_train : n_train + n_val]:
                split_dict[i] = "val"
            for i in indices[n_train + n_val :]:
                split_dict[i] = "test"

        # Create metadata
        slides = []
        for idx, row in df.iterrows():
            slide_id = row["image_id"]
            image_path = image_dir / f"{slide_id}.tiff"

            if not image_path.exists():
                logger.warning(f"Image not found: {image_path}")
                continue

            mask_path = None
            if mask_dir:
                mask_file = mask_dir / f"{slide_id}_mask.tiff"
                if mask_file.exists():
                    mask_path = str(mask_file)

            slides.append(
                PANDASlideMetadata(
                    slide_id=slide_id,
                    file_path=str(image_path),
                    isup_grade=int(row["isup_grade"]),
                    gleason_score=row["gleason_score"],
                    data_provider=row["data_provider"],
                    split=split_dict[idx],
                    mask_path=mask_path,
                )
            )

        logger.info(f"Indexed {len(slides)} PANDA slides from {csv_path}")
        return cls(slides)


class PANDASlideDataset(Dataset):
    """PANDA dataset returning complete slides with patch features.

    Args:
        slide_index: PANDASlideIndex with slide metadata
        features_dir: Directory containing HDF5 feature files
        split: Which split to load ('train', 'val', 'test')
        transform: Optional transform for features
        ordinal: Whether to use ordinal encoding for grades

    Returns:
        Dictionary containing:
            - 'slide_id': str
            - 'isup_grade': int (0-5)
            - 'gleason_score': str
            - 'data_provider': str
            - 'features': Tensor [num_patches, feature_dim]
            - 'coordinates': Tensor [num_patches, 2]
            - 'num_patches': int
    """

    def __init__(
        self,
        slide_index: PANDASlideIndex,
        features_dir: Union[str, Path],
        split: str = "train",
        transform: Optional[Callable] = None,
        ordinal: bool = False,
    ):
        self.slide_index = slide_index
        self.features_dir = Path(features_dir)
        self.split = split
        self.transform = transform
        self.ordinal = ordinal

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
                f"Please extract features first using a foundation model."
            )

        logger.info(
            f"Loaded {len(self.valid_slides)} PANDA slides for {split} split "
            f"(grade distribution: {self._get_grade_dist()})"
        )

    def _get_grade_dist(self) -> Dict[int, int]:
        """Get grade distribution for loaded slides."""
        dist = {i: 0 for i in range(6)}
        for slide in self.valid_slides:
            dist[slide.isup_grade] += 1
        return dist

    def __len__(self) -> int:
        return len(self.valid_slides)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
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

        # Prepare label
        if self.ordinal:
            # Ordinal encoding: [1,1,1,0,0,0] for grade 3
            label = torch.zeros(6, dtype=torch.float32)
            label[: slide.isup_grade] = 1.0
        else:
            label = slide.isup_grade

        return {
            "slide_id": slide.slide_id,
            "isup_grade": slide.isup_grade,
            "gleason_score": slide.gleason_score,
            "data_provider": slide.data_provider,
            "label": label,
            "features": features,
            "coordinates": coordinates,
            "num_patches": int(features.shape[0]),
        }


class PANDAPatchDataset(Dataset):
    """PANDA dataset for patch-level sampling.

    Args:
        slide_index: PANDASlideIndex with slide metadata
        features_dir: Directory containing HDF5 feature files
        split: Which split to load ('train', 'val', 'test')
        transform: Optional transform for features
        ordinal: Whether to use ordinal encoding for grades
    """

    def __init__(
        self,
        slide_index: PANDASlideIndex,
        features_dir: Union[str, Path],
        split: str = "train",
        transform: Optional[Callable] = None,
        ordinal: bool = False,
    ):
        self.slide_index = slide_index
        self.features_dir = Path(features_dir)
        self.split = split
        self.transform = transform
        self.ordinal = ordinal

        # Get slides for this split
        self.slides = slide_index.get_slides_by_split(split)

        # Build patch-level index
        self.patch_index: List[Tuple[str, int]] = []
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
            f"Loaded {len(self.slides)} PANDA slides with "
            f"{len(self.patch_index)} total patches for {split} split"
        )

    def __len__(self) -> int:
        return len(self.patch_index)

    def __getitem__(self, idx: int) -> Dict[str, Union[torch.Tensor, str, int]]:
        slide_id, patch_idx = self.patch_index[idx]
        slide = self.slide_index[slide_id]

        feature_file = self.features_dir / f"{slide_id}.h5"

        with h5py.File(feature_file, "r") as f:
            features = f["features"][patch_idx]
            coordinates = f["coordinates"][patch_idx]

        features = torch.tensor(features, dtype=torch.float32)
        coordinates = torch.tensor(coordinates, dtype=torch.int32)

        if self.transform:
            features = self.transform(features)

        # Prepare label
        if self.ordinal:
            label = torch.zeros(6, dtype=torch.float32)
            label[: slide.isup_grade] = 1.0
        else:
            label = slide.isup_grade

        return {
            "features": features,
            "coordinates": coordinates,
            "slide_id": slide_id,
            "isup_grade": slide.isup_grade,
            "gleason_score": slide.gleason_score,
            "data_provider": slide.data_provider,
            "label": label,
            "patch_idx": patch_idx,
        }


def collate_panda_bags(batch: List[Dict]) -> Dict[str, Union[torch.Tensor, List]]:
    """Collate function for variable-length PANDA slide bags.

    Args:
        batch: List of samples from PANDASlideDataset

    Returns:
        Dictionary with padded features and metadata
    """
    features_list = [item["features"] for item in batch]
    coordinates_list = [item["coordinates"] for item in batch]
    labels = torch.stack([torch.tensor(item["label"]) for item in batch])
    num_patches = torch.tensor([item["num_patches"] for item in batch], dtype=torch.long)
    slide_ids = [item["slide_id"] for item in batch]
    isup_grades = [item["isup_grade"] for item in batch]
    gleason_scores = [item["gleason_score"] for item in batch]
    data_providers = [item["data_provider"] for item in batch]

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
        "isup_grades": isup_grades,
        "gleason_scores": gleason_scores,
        "data_providers": data_providers,
    }


def compute_quadratic_weighted_kappa(
    y_true: np.ndarray, y_pred: np.ndarray, num_classes: int = 6
) -> float:
    """Compute quadratic weighted kappa for PANDA evaluation.

    This is the official metric for the PANDA challenge.

    Args:
        y_true: Ground truth grades [N]
        y_pred: Predicted grades [N]
        num_classes: Number of classes (6 for ISUP grades 0-5)

    Returns:
        Quadratic weighted kappa score
    """
    from sklearn.metrics import cohen_kappa_score

    return cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=list(range(num_classes)))


def validate_panda_dataset(dataset: PANDASlideDataset) -> Dict[str, Union[int, float, Dict]]:
    """Validate PANDA dataset integrity.

    Args:
        dataset: PANDASlideDataset instance

    Returns:
        Dictionary with validation statistics
    """
    logger.info("Validating PANDA dataset...")

    if len(dataset) == 0:
        raise ValueError("Dataset is empty")

    # Sample validation
    num_samples = min(10, len(dataset))
    grade_counts = {i: 0 for i in range(6)}
    patch_counts = []

    for i in range(num_samples):
        try:
            sample = dataset[i]
        except Exception as e:
            logger.error(f"Failed to load sample {i}: {e}")
            raise

        # Validate shapes
        assert sample["features"].ndim == 2, f"Expected 2D features, got {sample['features'].ndim}D"
        assert (
            sample["coordinates"].shape[1] == 2
        ), f"Expected coordinates [N, 2], got {sample['coordinates'].shape}"

        # Validate labels
        grade = sample["isup_grade"]
        assert 0 <= grade <= 5, f"Invalid ISUP grade: {grade}"
        grade_counts[grade] += 1
        patch_counts.append(sample["num_patches"])

    stats = {
        "num_slides": len(dataset),
        "grade_distribution": dataset._get_grade_dist(),
        "avg_patches_per_slide": np.mean(patch_counts) if patch_counts else 0,
        "min_patches": min(patch_counts) if patch_counts else 0,
        "max_patches": max(patch_counts) if patch_counts else 0,
    }

    logger.info(f"PANDA dataset validation passed: {stats}")
    return stats
