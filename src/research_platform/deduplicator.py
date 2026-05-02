"""
Deduplication System

Remove duplicate/near-duplicate samples from datasets.
"""

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Set, Tuple

import numpy as np


@dataclass
class DuplicateGroup:
    """Group of duplicate samples"""

    representative: Path
    duplicates: List[Path]
    similarity_scores: List[float]


class Deduplicator:
    """
    Image deduplication

    Methods:
    - Exact: MD5 hash
    - Perceptual: pHash
    - Feature: Deep feature similarity
    """

    def __init__(self, method: str = "perceptual", threshold: float = 0.95):
        self.method = method
        self.threshold = threshold
        self.logger = logging.getLogger(__name__)

    def deduplicate(self, image_paths: List[Path]) -> Tuple[List[Path], List[DuplicateGroup]]:
        """
        Deduplicate images

        Args:
            image_paths: List of image paths

        Returns:
            (unique_paths, duplicate_groups)
        """

        if self.method == "exact":
            return self._deduplicate_exact(image_paths)
        elif self.method == "perceptual":
            return self._deduplicate_perceptual(image_paths)
        elif self.method == "feature":
            return self._deduplicate_feature(image_paths)
        else:
            raise ValueError(f"Unknown method: {self.method}")

    def _deduplicate_exact(
        self, image_paths: List[Path]
    ) -> Tuple[List[Path], List[DuplicateGroup]]:
        """Exact deduplication using MD5"""

        hash_map: Dict[str, Path] = {}
        duplicates: Dict[str, List[Path]] = {}

        for path in image_paths:
            # Compute hash
            file_hash = self._compute_md5(path)

            if file_hash in hash_map:
                # Duplicate
                if file_hash not in duplicates:
                    duplicates[file_hash] = []
                duplicates[file_hash].append(path)
            else:
                # Unique
                hash_map[file_hash] = path

        # Build groups
        unique = list(hash_map.values())
        groups = [
            DuplicateGroup(
                representative=hash_map[h], duplicates=dups, similarity_scores=[1.0] * len(dups)
            )
            for h, dups in duplicates.items()
        ]

        self.logger.info(f"Exact dedup: {len(unique)} unique, {len(groups)} duplicate groups")

        return unique, groups

    def _deduplicate_perceptual(
        self, image_paths: List[Path]
    ) -> Tuple[List[Path], List[DuplicateGroup]]:
        """Perceptual deduplication using pHash"""

        # Compute hashes
        hashes = {}
        for path in image_paths:
            try:
                phash = self._compute_phash(path)
                hashes[path] = phash
            except Exception as e:
                self.logger.error(f"Error hashing {path}: {e}")

        # Find duplicates
        unique = []
        groups = []
        processed = set()

        paths = list(hashes.keys())

        for i, path1 in enumerate(paths):
            if path1 in processed:
                continue

            # Find similar
            similar = []
            scores = []

            for j, path2 in enumerate(paths[i + 1 :], start=i + 1):
                if path2 in processed:
                    continue

                # Hamming distance
                similarity = self._hamming_similarity(hashes[path1], hashes[path2])

                if similarity >= self.threshold:
                    similar.append(path2)
                    scores.append(similarity)
                    processed.add(path2)

            # Add to results
            unique.append(path1)
            processed.add(path1)

            if similar:
                groups.append(
                    DuplicateGroup(
                        representative=path1, duplicates=similar, similarity_scores=scores
                    )
                )

        self.logger.info(f"Perceptual dedup: {len(unique)} unique, {len(groups)} duplicate groups")

        return unique, groups

    def _deduplicate_feature(
        self, image_paths: List[Path]
    ) -> Tuple[List[Path], List[DuplicateGroup]]:
        """Feature-based deduplication using deep features"""

        # Extract features
        features = {}
        for path in image_paths:
            try:
                feat = self._extract_features(path)
                features[path] = feat
            except Exception as e:
                self.logger.error(f"Error extracting features {path}: {e}")

        # Find duplicates using cosine similarity
        unique = []
        groups = []
        processed = set()

        paths = list(features.keys())

        for i, path1 in enumerate(paths):
            if path1 in processed:
                continue

            # Find similar
            similar = []
            scores = []

            for j, path2 in enumerate(paths[i + 1 :], start=i + 1):
                if path2 in processed:
                    continue

                # Cosine similarity
                similarity = self._cosine_similarity(features[path1], features[path2])

                if similarity >= self.threshold:
                    similar.append(path2)
                    scores.append(similarity)
                    processed.add(path2)

            # Add to results
            unique.append(path1)
            processed.add(path1)

            if similar:
                groups.append(
                    DuplicateGroup(
                        representative=path1, duplicates=similar, similarity_scores=scores
                    )
                )

        self.logger.info(f"Feature dedup: {len(unique)} unique, {len(groups)} duplicate groups")

        return unique, groups

    def _compute_md5(self, path: Path) -> str:
        """Compute MD5 hash"""

        hash_md5 = hashlib.sha256()

        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)

        return hash_md5.hexdigest()

    def _compute_phash(self, path: Path) -> int:
        """Compute perceptual hash"""

        from PIL import Image

        # Load and resize
        img = Image.open(path).convert("L")
        img = img.resize((32, 32), Image.Resampling.LANCZOS)

        # Convert to array
        pixels = np.array(img).flatten()

        # DCT
        from scipy.fftpack import dct

        dct_coeffs = dct(pixels)

        # Take low frequencies
        low_freq = dct_coeffs[:64]

        # Median
        median = np.median(low_freq)

        # Hash
        hash_bits = (low_freq > median).astype(int)

        # Convert to int
        phash = int("".join(map(str, hash_bits)), 2)

        return phash

    def _hamming_similarity(self, hash1: int, hash2: int) -> float:
        """Compute Hamming similarity"""

        # XOR
        xor = hash1 ^ hash2

        # Count bits
        distance = bin(xor).count("1")

        # Similarity (64 bits)
        similarity = 1.0 - (distance / 64.0)

        return similarity

    def _extract_features(self, path: Path) -> np.ndarray:
        """Extract deep features"""

        import torch
        import torchvision.models as models
        import torchvision.transforms as transforms
        from PIL import Image

        # Load model (ResNet50)
        model = models.resnet50(pretrained=True)
        model.eval()

        # Remove classifier
        model = torch.nn.Sequential(*list(model.children())[:-1])

        # Load image
        img = Image.open(path).convert("RGB")

        # Transform
        transform = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

        img_tensor = transform(img).unsqueeze(0)

        # Extract features
        with torch.no_grad():
            features = model(img_tensor)

        return features.squeeze().numpy()

    def _cosine_similarity(self, feat1: np.ndarray, feat2: np.ndarray) -> float:
        """Compute cosine similarity"""

        dot = np.dot(feat1, feat2)
        norm1 = np.linalg.norm(feat1)
        norm2 = np.linalg.norm(feat2)

        similarity = dot / (norm1 * norm2)

        return similarity


class DatasetDeduplicator:
    """Deduplicate entire datasets"""

    def __init__(self, method: str = "perceptual", threshold: float = 0.95):
        self.dedup = Deduplicator(method, threshold)
        self.logger = logging.getLogger(__name__)

    def deduplicate_dataset(
        self, image_paths: List[Path], output_dir: Path = None, remove_duplicates: bool = False
    ) -> Dict:
        """
        Deduplicate dataset

        Args:
            image_paths: List of image paths
            output_dir: Output directory for unique images
            remove_duplicates: Delete duplicate files

        Returns:
            Deduplication report
        """

        # Deduplicate
        unique, groups = self.dedup.deduplicate(image_paths)

        # Copy unique to output
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            for path in unique:
                self._copy_image(path, output_dir)

        # Remove duplicates
        if remove_duplicates:
            for group in groups:
                for dup in group.duplicates:
                    dup.unlink()
                    self.logger.debug(f"Removed: {dup}")

        # Report
        total_duplicates = sum(len(g.duplicates) for g in groups)

        report = {
            "total_samples": len(image_paths),
            "unique_samples": len(unique),
            "duplicate_groups": len(groups),
            "total_duplicates": total_duplicates,
            "deduplication_rate": total_duplicates / len(image_paths) if image_paths else 0.0,
            "groups": [
                {
                    "representative": str(g.representative),
                    "num_duplicates": len(g.duplicates),
                    "avg_similarity": np.mean(g.similarity_scores),
                }
                for g in groups
            ],
        }

        self.logger.info(
            f"Dedup complete: {len(unique)} unique, {total_duplicates} duplicates removed"
        )

        return report

    def _copy_image(self, src: Path, dst_dir: Path):
        """Copy image"""
        import shutil

        shutil.copy2(src, dst_dir / src.name)


# Convenience functions


def deduplicate_images(
    image_paths: List[Path], method: str = "perceptual", threshold: float = 0.95
) -> Tuple[List[Path], List[DuplicateGroup]]:
    """Deduplicate images"""

    dedup = Deduplicator(method, threshold)
    return dedup.deduplicate(image_paths)


def deduplicate_dataset(
    image_paths: List[Path],
    output_dir: Path = None,
    method: str = "perceptual",
    threshold: float = 0.95,
) -> Dict:
    """Deduplicate dataset"""

    dedup = DatasetDeduplicator(method, threshold)
    return dedup.deduplicate_dataset(image_paths, output_dir)
