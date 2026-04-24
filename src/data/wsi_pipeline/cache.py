"""
Feature caching for WSI processing pipeline.

This module provides HDF5-based caching for processed WSI features,
enabling efficient storage and retrieval of feature embeddings with
associated metadata and coordinates.

The HDF5 structure is compatible with existing CAMELYONSlideDataset
and CAMELYONPatchDataset classes.
"""

import hashlib
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Union

import h5py
import numpy as np

from .exceptions import ProcessingError, ResourceError

logger = logging.getLogger(__name__)


class FeatureCache:
    """Cache for storing and retrieving WSI features in HDF5 format.

    This class manages HDF5 files containing:
    - Feature embeddings [num_patches, feature_dim] (float32)
    - Patch coordinates [num_patches, 2] (int32)
    - Slide metadata (HDF5 attributes)

    The HDF5 structure is compatible with existing CAMELYONSlideDataset
    and CAMELYONPatchDataset classes.

    Args:
        cache_dir: Directory to store HDF5 cache files
        compression: Compression algorithm ('gzip', 'lzf', or None)
        compression_level: Compression level for gzip (0-9, default 4)

    Example:
        >>> cache = FeatureCache(cache_dir='features', compression='gzip')
        >>> cache.save_features(
        ...     slide_id='patient_001_node_0',
        ...     features=features_array,
        ...     coordinates=coords_array,
        ...     metadata={'magnification': 40.0, 'mpp': 0.25}
        ... )
        >>> data = cache.load_features('patient_001_node_0')
        >>> print(data['features'].shape)  # [num_patches, feature_dim]
    """

    def __init__(
        self,
        cache_dir: Union[str, Path],
        compression: str = "gzip",
        compression_level: int = 4,
        use_chunking: bool = True,
        chunk_size: Optional[int] = None,
    ):
        """Initialize feature cache with storage optimizations.

        Args:
            cache_dir: Directory to store HDF5 cache files
            compression: Compression algorithm ('gzip', 'lzf', or None)
            compression_level: Compression level for gzip (0-9)
            use_chunking: Enable HDF5 chunking for efficient partial reads
            chunk_size: Chunk size for datasets (auto-calculated if None)
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Validate compression settings
        if compression not in ["gzip", "lzf", None]:
            raise ValueError(f"Unsupported compression: {compression}. Use 'gzip', 'lzf', or None")

        if compression == "gzip" and not (0 <= compression_level <= 9):
            raise ValueError(f"Invalid gzip compression level: {compression_level}. Must be 0-9")

        self.compression = compression
        self.compression_level = compression_level if compression == "gzip" else None
        self.use_chunking = use_chunking
        self.chunk_size = chunk_size

        logger.info(
            f"Initialized FeatureCache at {self.cache_dir} "
            f"with compression={compression}, level={compression_level}, "
            f"chunking={use_chunking}"
        )

    def _get_cache_path(self, slide_id: str) -> Path:
        """Get HDF5 file path for a slide.

        Args:
            slide_id: Slide identifier

        Returns:
            Path to HDF5 file
        """
        return self.cache_dir / f"{slide_id}.h5"

    def _anonymize_metadata(
        self,
        metadata: Dict[str, Any],
        anonymize_phi: bool = False,
    ) -> Dict[str, Any]:
        """Anonymize PHI in metadata if requested.

        Args:
            metadata: Original metadata dictionary
            anonymize_phi: Whether to anonymize patient_id and scan_date

        Returns:
            Metadata with PHI anonymized if requested
        """
        if not anonymize_phi:
            return metadata

        anonymized = metadata.copy()

        # Anonymize patient_id with hash
        if "patient_id" in anonymized and anonymized["patient_id"]:
            patient_id = str(anonymized["patient_id"])
            hash_obj = hashlib.sha256(patient_id.encode())
            anonymized["patient_id"] = f"anon_{hash_obj.hexdigest()[:16]}"

        # Remove scan_date
        if "scan_date" in anonymized:
            anonymized["scan_date"] = "REDACTED"

        logger.debug(f"Anonymized PHI in metadata")
        return anonymized

    def _calculate_optimal_chunk_size(
        self,
        data_shape: tuple,
        dtype_size: int,
        target_chunk_mb: float = 1.0,
    ) -> tuple:
        """
        Calculate optimal chunk size for HDF5 dataset.

        Args:
            data_shape: Shape of the dataset
            dtype_size: Size of data type in bytes
            target_chunk_mb: Target chunk size in MB

        Returns:
            Optimal chunk shape
        """
        if not self.use_chunking or len(data_shape) < 2:
            return None

        target_chunk_bytes = target_chunk_mb * 1024 * 1024

        # For 2D arrays (features, coordinates)
        if len(data_shape) == 2:
            rows, cols = data_shape

            # Calculate chunk rows to achieve target size
            bytes_per_row = cols * dtype_size
            chunk_rows = max(1, min(rows, int(target_chunk_bytes // bytes_per_row)))

            return (chunk_rows, cols)

        return None

    def _create_optimized_dataset(
        self,
        h5_file: h5py.File,
        name: str,
        data: np.ndarray,
        dtype: np.dtype,
    ) -> h5py.Dataset:
        """
        Create HDF5 dataset with storage optimizations.

        Args:
            h5_file: HDF5 file object
            name: Dataset name
            data: Data to store
            dtype: Target data type

        Returns:
            Created HDF5 dataset
        """
        # Convert data to target dtype
        data = data.astype(dtype)

        # Calculate optimal chunk size
        chunk_shape = None
        if self.use_chunking:
            if self.chunk_size:
                # Use specified chunk size
                chunk_shape = (min(self.chunk_size, data.shape[0]),) + data.shape[1:]
            else:
                # Calculate optimal chunk size - get itemsize properly
                try:
                    dtype_size = np.dtype(dtype).itemsize
                except TypeError:
                    dtype_size = data.dtype.itemsize

                chunk_shape = self._calculate_optimal_chunk_size(data.shape, dtype_size)

        # Create dataset with optimizations
        kwargs = {
            "data": data,
            "chunks": chunk_shape,
            "shuffle": True,  # Reorder bytes for better compression
            "fletcher32": True,  # Add checksum for data integrity
        }

        # Add compression if enabled
        if self.compression:
            kwargs["compression"] = self.compression
            if self.compression_level is not None:
                kwargs["compression_opts"] = self.compression_level

        dataset = h5_file.create_dataset(name, **kwargs)

        logger.debug(
            f"Created dataset '{name}': shape={data.shape}, dtype={dtype}, "
            f"chunks={chunk_shape}, compression={self.compression}"
        )

        return dataset

    def _check_disk_space(self, required_bytes: int) -> None:
        """
        Check if sufficient disk space is available.

        Args:
            required_bytes: Required space in bytes

        Raises:
            ResourceError: If insufficient disk space
        """
        try:
            import shutil

            free_bytes = shutil.disk_usage(self.cache_dir).free

            if free_bytes < required_bytes * 1.1:  # 10% safety margin
                raise ResourceError(
                    f"Insufficient disk space: need {required_bytes / 1024**3:.2f}GB, "
                    f"available {free_bytes / 1024**3:.2f}GB"
                )
        except ImportError:
            # shutil.disk_usage not available, skip check
            logger.debug("Disk space check skipped (shutil.disk_usage not available)")
        except Exception as e:
            logger.warning(f"Disk space check failed: {e}")

    def get_compression_stats(self, slide_id: str) -> Dict[str, Any]:
        """
        Get compression statistics for a cached slide.

        Args:
            slide_id: Slide identifier

        Returns:
            Dictionary with compression statistics
        """
        cache_path = self._get_cache_path(slide_id)

        if not cache_path.exists():
            raise FileNotFoundError(f"Cache file not found: {cache_path}")

        try:
            with h5py.File(cache_path, "r") as f:
                features_ds = f["features"]
                coords_ds = f["coordinates"]

                # Calculate uncompressed sizes
                features_uncompressed = features_ds.size * features_ds.dtype.itemsize
                coords_uncompressed = coords_ds.size * coords_ds.dtype.itemsize
                total_uncompressed = features_uncompressed + coords_uncompressed

                # Get actual file size
                file_size = cache_path.stat().st_size

                # Calculate compression ratio
                compression_ratio = total_uncompressed / file_size if file_size > 0 else 1.0

                return {
                    "file_size_mb": file_size / 1024**2,
                    "uncompressed_size_mb": total_uncompressed / 1024**2,
                    "compression_ratio": compression_ratio,
                    "space_saved_mb": (total_uncompressed - file_size) / 1024**2,
                    "compression_method": self.compression,
                    "features_shape": features_ds.shape,
                    "coordinates_shape": coords_ds.shape,
                }

        except Exception as e:
            raise ProcessingError(f"Failed to get compression stats for {slide_id}: {e}")

    def save_features(
        self,
        slide_id: str,
        features: np.ndarray,
        coordinates: np.ndarray,
        metadata: Dict[str, Any],
        anonymize_phi: bool = False,
    ) -> Path:
        """Save features and metadata to HDF5 file.

        Args:
            slide_id: Unique slide identifier
            features: Feature embeddings [num_patches, feature_dim] (float32)
            coordinates: Patch coordinates [num_patches, 2] (int32)
            metadata: Slide metadata dictionary with keys:
                - patient_id: str
                - scan_date: str (optional)
                - scanner_model: str (optional)
                - magnification: float (optional)
                - mpp: float (optional)
                - patch_size: int
                - stride: int
                - level: int
                - encoder_name: str
                - processing_timestamp: str (auto-generated if not provided)
                - num_patches: int (auto-generated from features)
            anonymize_phi: Whether to anonymize patient_id and scan_date

        Returns:
            Path to saved HDF5 file

        Raises:
            ProcessingError: If HDF5 write fails
            ResourceError: If disk space is insufficient
            ValueError: If features and coordinates have mismatched shapes
        """
        # Validate inputs
        if features.ndim != 2:
            raise ValueError(f"Features must be 2D array, got shape {features.shape}")

        if coordinates.ndim != 2 or coordinates.shape[1] != 2:
            raise ValueError(f"Coordinates must be [N, 2] array, got shape {coordinates.shape}")

        num_patches = features.shape[0]
        if coordinates.shape[0] != num_patches:
            raise ValueError(
                f"Mismatched patch counts: features={num_patches}, "
                f"coordinates={coordinates.shape[0]}"
            )

        # Prepare metadata
        metadata = self._anonymize_metadata(metadata, anonymize_phi)
        metadata = metadata.copy()
        metadata["slide_id"] = slide_id
        metadata["num_patches"] = num_patches

        if "processing_timestamp" not in metadata:
            metadata["processing_timestamp"] = datetime.now().isoformat()

        # Get cache file path
        cache_path = self._get_cache_path(slide_id)

        try:
            # Check disk space (estimate with compression)
            features_bytes = features.nbytes
            coords_bytes = coordinates.nbytes
            # Estimate compressed size (conservative estimate: 50% compression)
            estimated_compressed = (features_bytes + coords_bytes) * 0.5
            required_space = estimated_compressed + 10 * 1024 * 1024  # 10MB overhead

            self._check_disk_space(int(required_space))

            # Write HDF5 file with optimizations
            with h5py.File(cache_path, "w") as f:
                # Store features dataset (float32 for space efficiency)
                self._create_optimized_dataset(f, "features", features, np.float32)

                # Store coordinates dataset (int32 for space efficiency)
                self._create_optimized_dataset(f, "coordinates", coordinates, np.int32)

                # Store metadata as attributes
                for key, value in metadata.items():
                    if value is not None:
                        # Convert to appropriate type for HDF5 attributes
                        if isinstance(value, (int, float, str, bool)):
                            f.attrs[key] = value
                        else:
                            f.attrs[key] = str(value)

            # Get compression statistics
            file_size = cache_path.stat().st_size
            original_size = features.nbytes + coordinates.nbytes
            compression_ratio = original_size / file_size if file_size > 0 else 1.0

            logger.info(
                f"Saved features for {slide_id}: {num_patches} patches, "
                f"{features.shape[1]} dims, "
                f"size={file_size / 1024 / 1024:.2f}MB "
                f"(compression: {compression_ratio:.1f}x)"
            )

            return cache_path

        except OSError as e:
            if "No space left on device" in str(e):
                raise ResourceError(f"Insufficient disk space to save features: {e}")
            raise ProcessingError(f"Failed to write HDF5 file {cache_path}: {e}")

        except Exception as e:
            raise ProcessingError(f"Failed to save features for {slide_id}: {e}")

    def load_features(
        self,
        slide_id: str,
    ) -> Dict[str, Union[np.ndarray, Dict[str, Any]]]:
        """Load features and metadata from HDF5 file.

        Args:
            slide_id: Slide identifier

        Returns:
            Dictionary containing:
                - 'features': np.ndarray [num_patches, feature_dim] (float32)
                - 'coordinates': np.ndarray [num_patches, 2] (int32)
                - 'metadata': Dict with all HDF5 attributes

        Raises:
            FileNotFoundError: If cache file doesn't exist
            ProcessingError: If HDF5 read fails or file is corrupted
        """
        cache_path = self._get_cache_path(slide_id)

        if not cache_path.exists():
            raise FileNotFoundError(f"Cache file not found: {cache_path}")

        try:
            with h5py.File(cache_path, "r") as f:
                # Validate structure
                if "features" not in f:
                    raise ProcessingError(f"Missing 'features' dataset in {cache_path}")

                if "coordinates" not in f:
                    raise ProcessingError(f"Missing 'coordinates' dataset in {cache_path}")

                # Load datasets
                features = f["features"][:]
                coordinates = f["coordinates"][:]

                # Load metadata from attributes
                metadata = dict(f.attrs)

            logger.debug(
                f"Loaded features for {slide_id}: {features.shape[0]} patches, "
                f"{features.shape[1]} dims"
            )

            return {
                "features": features,
                "coordinates": coordinates,
                "metadata": metadata,
            }

        except Exception as e:
            if isinstance(e, (FileNotFoundError, ProcessingError)):
                raise
            raise ProcessingError(f"Failed to load features for {slide_id}: {e}")

    def exists(self, slide_id: str) -> bool:
        """Check if cached features exist for a slide.

        Args:
            slide_id: Slide identifier

        Returns:
            True if cache file exists, False otherwise
        """
        cache_path = self._get_cache_path(slide_id)
        return cache_path.exists()

    def validate(self, slide_id: str) -> Dict[str, Union[bool, str, int]]:
        """Validate HDF5 file structure and integrity.

        Checks for:
        - File exists and is readable
        - Required datasets: 'features' and 'coordinates'
        - Compatible shapes between features and coordinates
        - Reasonable data types

        Args:
            slide_id: Slide identifier

        Returns:
            Dictionary with validation results:
                - 'valid': bool - whether file is valid
                - 'error': str - error message if invalid (None if valid)
                - 'num_patches': int - number of patches if valid
                - 'feature_dim': int - feature dimension if valid
        """
        result = {
            "valid": False,
            "error": None,
            "num_patches": 0,
            "feature_dim": 0,
        }

        cache_path = self._get_cache_path(slide_id)

        if not cache_path.exists():
            result["error"] = f"File not found: {cache_path}"
            return result

        try:
            with h5py.File(cache_path, "r") as f:
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
                        f"Mismatched patch counts: features={num_patches}, "
                        f"coordinates={coordinates.shape[0]}"
                    )
                    return result

                # Check data types
                if features.dtype not in [np.float32, np.float64]:
                    result["error"] = f"Features should be float32/float64, got {features.dtype}"
                    return result

                if coordinates.dtype not in [np.int32, np.int64]:
                    result["error"] = f"Coordinates should be int32/int64, got {coordinates.dtype}"
                    return result

                # All checks passed
                result["valid"] = True
                result["num_patches"] = num_patches
                result["feature_dim"] = features.shape[1]

        except Exception as e:
            result["error"] = f"Error reading file: {e}"

        return result

    def delete(self, slide_id: str) -> bool:
        """Delete cached features for a slide.

        Args:
            slide_id: Slide identifier

        Returns:
            True if file was deleted, False if file didn't exist
        """
        cache_path = self._get_cache_path(slide_id)

        if cache_path.exists():
            cache_path.unlink()
            logger.info(f"Deleted cache file: {cache_path}")
            return True

        return False

    def list_cached_slides(self) -> list[str]:
        """List all slide IDs with cached features.

        Returns:
            List of slide IDs (without .h5 extension)
        """
        cache_files = self.cache_dir.glob("*.h5")
        return [f.stem for f in cache_files]

    def get_cache_size(self, slide_id: Optional[str] = None) -> int:
        """Get cache size in bytes.

        Args:
            slide_id: Optional slide ID to get size for specific file.
                     If None, returns total cache directory size.

        Returns:
            Size in bytes
        """
        if slide_id is not None:
            cache_path = self._get_cache_path(slide_id)
            if cache_path.exists():
                return cache_path.stat().st_size
            return 0

        # Calculate total cache size
        total_size = 0
        for cache_file in self.cache_dir.glob("*.h5"):
            total_size += cache_file.stat().st_size

        return total_size
