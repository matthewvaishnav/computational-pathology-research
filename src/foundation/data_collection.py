"""
Data Collection and Curation System for Foundation Model Pre-Training
Handles 100K+ unlabeled WSI slides with quality filtering and deduplication
"""

import hashlib
import json
import logging
import os
import sqlite3
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import cv2
import numpy as np
import openslide
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm


@dataclass
class SlideMetadata:
    """Metadata for a single WSI slide"""

    slide_id: str
    file_path: str
    file_size: int
    dimensions: Tuple[int, int]
    mpp: Optional[float]  # microns per pixel
    vendor: str
    scanner_model: Optional[str]
    magnification: Optional[int]
    tissue_type: Optional[str]
    staining: str
    quality_score: float
    tissue_percentage: float
    blur_score: float
    color_variance: float
    artifact_score: float
    sha256_hash: str
    created_at: str
    processed_at: Optional[str] = None


@dataclass
class QualityMetrics:
    """Quality assessment metrics for WSI"""

    tissue_percentage: float
    blur_score: float
    color_variance: float
    artifact_score: float
    overall_score: float
    passed_filters: bool


class WSIQualityAssessment:
    """Quality assessment for whole slide images"""

    def __init__(
        self,
        min_tissue_percentage: float = 0.1,
        max_blur_threshold: float = 100.0,
        min_color_variance: float = 0.01,
        max_artifact_score: float = 0.3,
    ):
        self.min_tissue_percentage = min_tissue_percentage
        self.max_blur_threshold = max_blur_threshold
        self.min_color_variance = min_color_variance
        self.max_artifact_score = max_artifact_score

        self.logger = logging.getLogger(__name__)

    def assess_slide_quality(self, slide_path: str) -> QualityMetrics:
        """Assess quality of a single WSI slide"""
        try:
            slide = openslide.OpenSlide(slide_path)

            # Get thumbnail for quality assessment
            thumbnail = slide.get_thumbnail((1024, 1024))
            thumbnail_array = np.array(thumbnail)

            # Calculate quality metrics
            tissue_percentage = self._calculate_tissue_percentage(thumbnail_array)
            blur_score = self._calculate_blur_score(thumbnail_array)
            color_variance = self._calculate_color_variance(thumbnail_array)
            artifact_score = self._calculate_artifact_score(thumbnail_array)

            # Overall quality score (weighted combination)
            overall_score = (
                0.3 * min(tissue_percentage / 0.5, 1.0)
                + 0.25 * max(0, 1.0 - blur_score / self.max_blur_threshold)
                + 0.25 * min(color_variance / 0.1, 1.0)
                + 0.2 * max(0, 1.0 - artifact_score)
            )

            # Check if slide passes quality filters
            passed_filters = (
                tissue_percentage >= self.min_tissue_percentage
                and blur_score <= self.max_blur_threshold
                and color_variance >= self.min_color_variance
                and artifact_score <= self.max_artifact_score
            )

            slide.close()

            return QualityMetrics(
                tissue_percentage=tissue_percentage,
                blur_score=blur_score,
                color_variance=color_variance,
                artifact_score=artifact_score,
                overall_score=overall_score,
                passed_filters=passed_filters,
            )

        except Exception as e:
            self.logger.error(f"Error assessing slide quality for {slide_path}: {e}")
            return QualityMetrics(0.0, 1000.0, 0.0, 1.0, 0.0, False)

    def _calculate_tissue_percentage(self, image: np.ndarray) -> float:
        """Calculate percentage of tissue vs background"""
        # Convert to HSV for better tissue detection
        hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)

        # Create tissue mask (exclude white/light backgrounds)
        # Tissue typically has lower value (brightness) and higher saturation
        tissue_mask = (hsv[:, :, 1] > 20) & (  # Saturation > threshold
            hsv[:, :, 2] < 240
        )  # Value < threshold (not too bright)

        tissue_percentage = np.sum(tissue_mask) / tissue_mask.size
        return tissue_percentage

    def _calculate_blur_score(self, image: np.ndarray) -> float:
        """Calculate blur score using Laplacian variance"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        return laplacian_var

    def _calculate_color_variance(self, image: np.ndarray) -> float:
        """Calculate color variance across channels"""
        # Calculate variance for each channel
        r_var = np.var(image[:, :, 0])
        g_var = np.var(image[:, :, 1])
        b_var = np.var(image[:, :, 2])

        # Average variance across channels
        color_variance = (r_var + g_var + b_var) / 3.0 / (255.0**2)
        return color_variance

    def _calculate_artifact_score(self, image: np.ndarray) -> float:
        """Calculate artifact score (bubbles, folds, etc.)"""
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

        # Detect circular artifacts (bubbles)
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            dp=1,
            minDist=30,
            param1=50,
            param2=30,
            minRadius=5,
            maxRadius=50,
        )

        bubble_score = 0.0
        if circles is not None:
            bubble_score = len(circles[0]) / 100.0  # Normalize by expected max

        # Detect linear artifacts (folds, tears)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180, threshold=100, minLineLength=50, maxLineGap=10
        )

        line_score = 0.0
        if lines is not None:
            line_score = len(lines) / 50.0  # Normalize by expected max

        # Combined artifact score
        artifact_score = min(bubble_score + line_score, 1.0)
        return artifact_score


class SlideDatabase:
    """SQLite database for slide metadata and indexing"""

    def __init__(self, db_path: str):
        self.db_path = db_path
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()

    def _create_tables(self):
        """Create database tables"""
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS slides (
                slide_id TEXT PRIMARY KEY,
                file_path TEXT UNIQUE NOT NULL,
                file_size INTEGER,
                width INTEGER,
                height INTEGER,
                mpp REAL,
                vendor TEXT,
                scanner_model TEXT,
                magnification INTEGER,
                tissue_type TEXT,
                staining TEXT,
                quality_score REAL,
                tissue_percentage REAL,
                blur_score REAL,
                color_variance REAL,
                artifact_score REAL,
                sha256_hash TEXT UNIQUE,
                created_at TEXT,
                processed_at TEXT
            )
        """)

        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_quality_score ON slides(quality_score);
        """)

        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_tissue_type ON slides(tissue_type);
        """)

        self.conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_staining ON slides(staining);
        """)

        self.conn.commit()

    def insert_slide(self, metadata: SlideMetadata) -> bool:
        """Insert slide metadata into database"""
        try:
            self.conn.execute(
                """
                INSERT OR REPLACE INTO slides VALUES (
                    ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?
                )
            """,
                (
                    metadata.slide_id,
                    metadata.file_path,
                    metadata.file_size,
                    metadata.dimensions[0],
                    metadata.dimensions[1],
                    metadata.mpp,
                    metadata.vendor,
                    metadata.scanner_model,
                    metadata.magnification,
                    metadata.tissue_type,
                    metadata.staining,
                    metadata.quality_score,
                    metadata.tissue_percentage,
                    metadata.blur_score,
                    metadata.color_variance,
                    metadata.artifact_score,
                    metadata.sha256_hash,
                    metadata.created_at,
                    metadata.processed_at,
                ),
            )
            self.conn.commit()
            return True
        except sqlite3.IntegrityError as e:
            logging.warning(f"Duplicate slide detected: {metadata.slide_id} - {e}")
            return False

    def get_slide_by_hash(self, sha256_hash: str) -> Optional[SlideMetadata]:
        """Get slide by SHA256 hash (for deduplication)"""
        cursor = self.conn.execute("SELECT * FROM slides WHERE sha256_hash = ?", (sha256_hash,))
        row = cursor.fetchone()

        if row:
            return SlideMetadata(
                slide_id=row[0],
                file_path=row[1],
                file_size=row[2],
                dimensions=(row[3], row[4]),
                mpp=row[5],
                vendor=row[6],
                scanner_model=row[7],
                magnification=row[8],
                tissue_type=row[9],
                staining=row[10],
                quality_score=row[11],
                tissue_percentage=row[12],
                blur_score=row[13],
                color_variance=row[14],
                artifact_score=row[15],
                sha256_hash=row[16],
                created_at=row[17],
                processed_at=row[18],
            )
        return None

    def get_high_quality_slides(
        self, min_quality_score: float = 0.7, limit: Optional[int] = None
    ) -> List[SlideMetadata]:
        """Get high-quality slides for training"""
        query = """
            SELECT * FROM slides 
            WHERE quality_score >= ? 
            ORDER BY quality_score DESC
        """

        if limit:
            query += f" LIMIT {limit}"

        cursor = self.conn.execute(query, (min_quality_score,))
        rows = cursor.fetchall()

        slides = []
        for row in rows:
            slides.append(
                SlideMetadata(
                    slide_id=row[0],
                    file_path=row[1],
                    file_size=row[2],
                    dimensions=(row[3], row[4]),
                    mpp=row[5],
                    vendor=row[6],
                    scanner_model=row[7],
                    magnification=row[8],
                    tissue_type=row[9],
                    staining=row[10],
                    quality_score=row[11],
                    tissue_percentage=row[12],
                    blur_score=row[13],
                    color_variance=row[14],
                    artifact_score=row[15],
                    sha256_hash=row[16],
                    created_at=row[17],
                    processed_at=row[18],
                )
            )

        return slides

    def get_statistics(self) -> Dict[str, Any]:
        """Get database statistics"""
        cursor = self.conn.execute("SELECT COUNT(*) FROM slides")
        total_slides = cursor.fetchone()[0]

        cursor = self.conn.execute("SELECT AVG(quality_score) FROM slides")
        avg_quality = cursor.fetchone()[0] or 0.0

        cursor = self.conn.execute("SELECT COUNT(*) FROM slides WHERE quality_score >= 0.7")
        high_quality_slides = cursor.fetchone()[0]

        cursor = self.conn.execute("SELECT vendor, COUNT(*) FROM slides GROUP BY vendor")
        vendor_counts = dict(cursor.fetchall())

        cursor = self.conn.execute("SELECT staining, COUNT(*) FROM slides GROUP BY staining")
        staining_counts = dict(cursor.fetchall())

        return {
            "total_slides": total_slides,
            "average_quality": avg_quality,
            "high_quality_slides": high_quality_slides,
            "vendor_distribution": vendor_counts,
            "staining_distribution": staining_counts,
        }


class WSIDataCollector:
    """Main data collection and curation system"""

    def __init__(
        self, database_path: str = "slide_database.db", quality_config: Optional[Dict] = None
    ):
        self.database = SlideDatabase(database_path)
        self.quality_assessor = WSIQualityAssessment(**(quality_config or {}))
        self.logger = logging.getLogger(__name__)

    def collect_slides_from_directory(
        self, directory: str, recursive: bool = True, max_workers: int = 4
    ) -> Dict[str, Any]:
        """Collect and process slides from directory"""
        slide_extensions = {".svs", ".tif", ".tiff", ".ndpi", ".vms", ".vmu", ".scn"}

        # Find all slide files
        slide_files = []
        directory_path = Path(directory)

        if recursive:
            for ext in slide_extensions:
                slide_files.extend(directory_path.rglob(f"*{ext}"))
        else:
            for ext in slide_extensions:
                slide_files.extend(directory_path.glob(f"*{ext}"))

        self.logger.info(f"Found {len(slide_files)} slide files in {directory}")

        # Process slides in parallel
        processed_count = 0
        duplicate_count = 0
        failed_count = 0

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_file = {
                executor.submit(self._process_single_slide, str(slide_file)): slide_file
                for slide_file in slide_files
            }

            # Process results
            for future in tqdm(as_completed(future_to_file), total=len(slide_files)):
                slide_file = future_to_file[future]
                try:
                    result = future.result()
                    if result == "processed":
                        processed_count += 1
                    elif result == "duplicate":
                        duplicate_count += 1
                    else:
                        failed_count += 1
                except Exception as e:
                    self.logger.error(f"Error processing {slide_file}: {e}")
                    failed_count += 1

        # Get final statistics
        stats = self.database.get_statistics()

        return {
            "processed_slides": processed_count,
            "duplicate_slides": duplicate_count,
            "failed_slides": failed_count,
            "database_stats": stats,
        }

    def _process_single_slide(self, slide_path: str) -> str:
        """Process a single slide file"""
        try:
            # Calculate file hash for deduplication
            sha256_hash = self._calculate_file_hash(slide_path)

            # Check if slide already exists
            existing_slide = self.database.get_slide_by_hash(sha256_hash)
            if existing_slide:
                return "duplicate"

            # Extract metadata
            metadata = self._extract_slide_metadata(slide_path, sha256_hash)
            if not metadata:
                return "failed"

            # Insert into database
            success = self.database.insert_slide(metadata)
            return "processed" if success else "failed"

        except Exception as e:
            self.logger.error(f"Error processing slide {slide_path}: {e}")
            return "failed"

    def _calculate_file_hash(self, file_path: str) -> str:
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            # Read file in chunks to handle large files
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def _extract_slide_metadata(self, slide_path: str, sha256_hash: str) -> Optional[SlideMetadata]:
        """Extract metadata from slide file"""
        try:
            slide = openslide.OpenSlide(slide_path)

            # Basic file info
            file_size = os.path.getsize(slide_path)
            dimensions = slide.dimensions

            # Extract properties
            properties = slide.properties
            mpp = self._extract_mpp(properties)
            vendor = properties.get(openslide.PROPERTY_NAME_VENDOR, "unknown")
            scanner_model = properties.get(openslide.PROPERTY_NAME_COMMENT, None)
            magnification = self._extract_magnification(properties)

            # Assess quality
            quality_metrics = self.quality_assessor.assess_slide_quality(slide_path)

            slide.close()

            # Create slide ID from filename
            slide_id = Path(slide_path).stem

            # Determine tissue type and staining (simplified heuristics)
            tissue_type = self._infer_tissue_type(slide_path)
            staining = self._infer_staining(slide_path, properties)

            metadata = SlideMetadata(
                slide_id=slide_id,
                file_path=slide_path,
                file_size=file_size,
                dimensions=dimensions,
                mpp=mpp,
                vendor=vendor,
                scanner_model=scanner_model,
                magnification=magnification,
                tissue_type=tissue_type,
                staining=staining,
                quality_score=quality_metrics.overall_score,
                tissue_percentage=quality_metrics.tissue_percentage,
                blur_score=quality_metrics.blur_score,
                color_variance=quality_metrics.color_variance,
                artifact_score=quality_metrics.artifact_score,
                sha256_hash=sha256_hash,
                created_at=str(Path(slide_path).stat().st_ctime),
                processed_at=None,
            )

            return metadata

        except Exception as e:
            self.logger.error(f"Error extracting metadata from {slide_path}: {e}")
            return None

    def _extract_mpp(self, properties: Dict[str, str]) -> Optional[float]:
        """Extract microns per pixel from slide properties"""
        # Try different property keys
        mpp_keys = [
            openslide.PROPERTY_NAME_MPP_X,
            openslide.PROPERTY_NAME_MPP_Y,
            "aperio.MPP",
            "hamamatsu.SourceLens",
        ]

        for key in mpp_keys:
            if key in properties:
                try:
                    return float(properties[key])
                except ValueError:
                    continue

        return None

    def _extract_magnification(self, properties: Dict[str, str]) -> Optional[int]:
        """Extract magnification from slide properties"""
        mag_keys = [
            openslide.PROPERTY_NAME_OBJECTIVE_POWER,
            "aperio.AppMag",
            "hamamatsu.SourceLens",
        ]

        for key in mag_keys:
            if key in properties:
                try:
                    return int(float(properties[key]))
                except ValueError:
                    continue

        return None

    def _infer_tissue_type(self, slide_path: str) -> Optional[str]:
        """Infer tissue type from filename (simplified)"""
        filename = Path(slide_path).name.lower()

        tissue_keywords = {
            "breast": ["breast", "mammary"],
            "lung": ["lung", "pulmonary"],
            "prostate": ["prostate"],
            "colon": ["colon", "colorectal"],
            "melanoma": ["melanoma", "skin"],
            "liver": ["liver", "hepatic"],
            "kidney": ["kidney", "renal"],
            "brain": ["brain", "neural"],
        }

        for tissue_type, keywords in tissue_keywords.items():
            if any(keyword in filename for keyword in keywords):
                return tissue_type

        return None

    def _infer_staining(self, slide_path: str, properties: Dict[str, str]) -> str:
        """Infer staining type from filename and properties"""
        filename = Path(slide_path).name.lower()

        # Check filename for staining indicators
        if "he" in filename or "h&e" in filename:
            return "H&E"
        elif "ihc" in filename:
            return "IHC"
        elif "pas" in filename:
            return "PAS"
        elif "trichrome" in filename:
            return "Trichrome"

        # Default to H&E (most common)
        return "H&E"

    def get_training_dataset(
        self,
        min_quality_score: float = 0.7,
        max_slides: Optional[int] = None,
        tissue_types: Optional[List[str]] = None,
    ) -> List[SlideMetadata]:
        """Get curated dataset for training"""
        slides = self.database.get_high_quality_slides(min_quality_score, max_slides)

        if tissue_types:
            slides = [s for s in slides if s.tissue_type in tissue_types]

        return slides


class UnlabeledWSIDataset(Dataset):
    """PyTorch dataset for unlabeled WSI data"""

    def __init__(
        self,
        slide_metadata: List[SlideMetadata],
        patch_size: int = 224,
        patches_per_slide: int = 100,
        level: int = 0,
        transform=None,
    ):
        self.slide_metadata = slide_metadata
        self.patch_size = patch_size
        self.patches_per_slide = patches_per_slide
        self.level = level
        self.transform = transform

        # Pre-compute patch coordinates for each slide
        self.patch_coordinates = []
        self._precompute_coordinates()

    def _precompute_coordinates(self):
        """Pre-compute random patch coordinates for each slide"""
        for metadata in self.slide_metadata:
            try:
                slide = openslide.OpenSlide(metadata.file_path)
                level_dimensions = slide.level_dimensions[self.level]

                # Generate random coordinates
                coords = []
                for _ in range(self.patches_per_slide):
                    x = np.random.randint(0, max(1, level_dimensions[0] - self.patch_size))
                    y = np.random.randint(0, max(1, level_dimensions[1] - self.patch_size))
                    coords.append((x, y))

                self.patch_coordinates.append(coords)
                slide.close()

            except Exception as e:
                logging.warning(f"Error processing slide {metadata.slide_id}: {e}")
                self.patch_coordinates.append([])

    def __len__(self) -> int:
        return sum(len(coords) for coords in self.patch_coordinates)

    def __getitem__(self, idx: int) -> torch.Tensor:
        # Find which slide and patch this index corresponds to
        slide_idx = 0
        patch_idx = idx

        for i, coords in enumerate(self.patch_coordinates):
            if patch_idx < len(coords):
                slide_idx = i
                break
            patch_idx -= len(coords)

        # Load patch
        metadata = self.slide_metadata[slide_idx]
        x, y = self.patch_coordinates[slide_idx][patch_idx]

        try:
            slide = openslide.OpenSlide(metadata.file_path)
            patch = slide.read_region((x, y), self.level, (self.patch_size, self.patch_size))
            patch = patch.convert("RGB")
            slide.close()

            # Convert to tensor
            patch_array = np.array(patch)
            patch_tensor = torch.from_numpy(patch_array).permute(2, 0, 1).float() / 255.0

            if self.transform:
                patch_tensor = self.transform(patch_tensor)

            return patch_tensor

        except Exception as e:
            logging.warning(f"Error loading patch from {metadata.slide_id}: {e}")
            # Return random patch as fallback
            return torch.randn(3, self.patch_size, self.patch_size)


# Example usage
if __name__ == "__main__":
    # Initialize data collector
    collector = WSIDataCollector("foundation_slides.db")

    # Collect slides from directory
    results = collector.collect_slides_from_directory(
        "/path/to/wsi/slides", recursive=True, max_workers=8
    )

    print(f"Collection results: {results}")

    # Get training dataset
    training_slides = collector.get_training_dataset(min_quality_score=0.7, max_slides=100000)

    print(f"Training dataset: {len(training_slides)} slides")

    # Create PyTorch dataset
    dataset = UnlabeledWSIDataset(training_slides[:1000])  # First 1000 slides
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True, num_workers=4)

    print(f"Dataset size: {len(dataset)} patches")

    # Test loading a batch
    for batch in dataloader:
        print(f"Batch shape: {batch.shape}")
        break
