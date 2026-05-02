"""
Storage optimization for HistoCore Real-Time WSI Streaming.

Provides compressed feature storage, automatic cleanup of temporary files,
and cloud storage integration (S3, Azure Blob, Google Cloud Storage).
"""

import gzip
import json
import logging
import os
import shutil
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import h5py
import numpy as np
import torch

from .metrics import storage_operations_duration, storage_size_bytes

logger = logging.getLogger(__name__)


@dataclass
class StorageConfig:
    """Configuration for storage system."""

    # Local storage
    temp_dir: str = tempfile.mkdtemp(prefix="histocore_")
    cache_dir: str = "./cache"
    results_dir: str = "./results"

    # Compression
    enable_compression: bool = True
    compression_level: int = 6  # 1-9

    # Cleanup
    auto_cleanup: bool = True
    cleanup_age_hours: int = 24
    max_temp_size_gb: float = 10.0

    # Cloud storage
    cloud_enabled: bool = False
    cloud_provider: str = "s3"  # s3, azure, gcs
    cloud_bucket: str = ""
    cloud_prefix: str = "histocore"

    # Feature storage
    feature_format: str = "hdf5"  # hdf5, npz, pt
    feature_compression: str = "gzip"  # gzip, lzf, none


class LocalStorage:
    """Local file storage with compression and cleanup."""

    def __init__(self, config: StorageConfig):
        """Initialize local storage."""
        self.config = config
        self._ensure_directories()

    def _ensure_directories(self) -> None:
        """Ensure storage directories exist."""
        for directory in [self.config.temp_dir, self.config.cache_dir, self.config.results_dir]:
            Path(directory).mkdir(parents=True, exist_ok=True)

        logger.info(
            "Storage directories initialized: temp=%s cache=%s results=%s",
            self.config.temp_dir,
            self.config.cache_dir,
            self.config.results_dir,
        )

    @storage_operations_duration.labels(operation="write").time()
    def write_features(
        self,
        slide_id: str,
        features: Union[np.ndarray, torch.Tensor],
        coordinates: Optional[np.ndarray] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Write features to storage."""
        if isinstance(features, torch.Tensor):
            features = features.cpu().numpy()

        filepath = self._get_feature_path(slide_id)

        if self.config.feature_format == "hdf5":
            self._write_hdf5(filepath, features, coordinates, metadata)
        elif self.config.feature_format == "npz":
            self._write_npz(filepath, features, coordinates, metadata)
        elif self.config.feature_format == "pt":
            self._write_pytorch(filepath, features, coordinates, metadata)
        else:
            raise ValueError(f"Unsupported feature format: {self.config.feature_format}")

        # Update metrics
        file_size = os.path.getsize(filepath)
        storage_size_bytes.labels(storage_type="features").set(file_size)

        logger.debug(
            "Features written for slide %s: format=%s size_mb=%.2f",
            slide_id,
            self.config.feature_format,
            file_size / (1024 * 1024),
        )

        return filepath

    def _write_hdf5(
        self,
        filepath: str,
        features: np.ndarray,
        coordinates: Optional[np.ndarray],
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        """Write features to HDF5 file."""
        with h5py.File(filepath, "w") as f:
            # Write features with compression
            f.create_dataset(
                "features",
                data=features,
                compression=(
                    self.config.feature_compression if self.config.enable_compression else None
                ),
                compression_opts=(
                    self.config.compression_level if self.config.enable_compression else None
                ),
            )

            # Write coordinates if provided
            if coordinates is not None:
                f.create_dataset("coordinates", data=coordinates)

            # Write metadata if provided
            if metadata is not None:
                for key, value in metadata.items():
                    f.attrs[key] = value

    def _write_npz(
        self,
        filepath: str,
        features: np.ndarray,
        coordinates: Optional[np.ndarray],
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        """Write features to NPZ file."""
        data = {"features": features}

        if coordinates is not None:
            data["coordinates"] = coordinates

        if metadata is not None:
            data["metadata"] = np.array([json.dumps(metadata)])

        if self.config.enable_compression:
            np.savez_compressed(filepath, **data)
        else:
            np.savez(filepath, **data)

    def _write_pytorch(
        self,
        filepath: str,
        features: np.ndarray,
        coordinates: Optional[np.ndarray],
        metadata: Optional[Dict[str, Any]],
    ) -> None:
        """Write features to PyTorch file."""
        data = {
            "features": torch.from_numpy(features),
        }

        if coordinates is not None:
            data["coordinates"] = torch.from_numpy(coordinates)

        if metadata is not None:
            data["metadata"] = metadata

        torch.save(data, filepath)

    @storage_operations_duration.labels(operation="read").time()
    def read_features(self, slide_id: str) -> Optional[Dict[str, Any]]:
        """Read features from storage."""
        filepath = self._get_feature_path(slide_id)

        if not os.path.exists(filepath):
            return None

        if self.config.feature_format == "hdf5":
            return self._read_hdf5(filepath)
        elif self.config.feature_format == "npz":
            return self._read_npz(filepath)
        elif self.config.feature_format == "pt":
            return self._read_pytorch(filepath)
        else:
            raise ValueError(f"Unsupported feature format: {self.config.feature_format}")

    def _read_hdf5(self, filepath: str) -> Dict[str, Any]:
        """Read features from HDF5 file."""
        with h5py.File(filepath, "r") as f:
            data = {"features": f["features"][:]}

            if "coordinates" in f:
                data["coordinates"] = f["coordinates"][:]

            # Read metadata
            metadata = dict(f.attrs)
            if metadata:
                data["metadata"] = metadata

            return data

    def _read_npz(self, filepath: str) -> Dict[str, Any]:
        """Read features from NPZ file."""
        loaded = np.load(filepath, allow_pickle=True)

        data = {"features": loaded["features"]}

        if "coordinates" in loaded:
            data["coordinates"] = loaded["coordinates"]

        if "metadata" in loaded:
            data["metadata"] = json.loads(loaded["metadata"][0])

        return data

    def _read_pytorch(self, filepath: str) -> Dict[str, Any]:
        """Read features from PyTorch file."""
        loaded = torch.load(filepath)

        data = {"features": loaded["features"].numpy()}

        if "coordinates" in loaded:
            data["coordinates"] = loaded["coordinates"].numpy()

        if "metadata" in loaded:
            data["metadata"] = loaded["metadata"]

        return data

    def _get_feature_path(self, slide_id: str) -> str:
        """Get filepath for slide features."""
        format_extensions = {
            "hdf5": ".h5",
            "npz": ".npz",
            "pt": ".pt",
        }

        if self.config.feature_format not in format_extensions:
            raise ValueError(f"Unsupported feature format: {self.config.feature_format}")

        ext = format_extensions[self.config.feature_format]
        return os.path.join(self.config.cache_dir, f"{slide_id}_features{ext}")

    def write_result(self, slide_id: str, result: Dict[str, Any]) -> str:
        """Write processing result to storage."""
        filepath = os.path.join(self.config.results_dir, f"{slide_id}_result.json")

        if self.config.enable_compression:
            with gzip.open(filepath + ".gz", "wt", encoding="utf-8") as f:
                json.dump(result, f, indent=2)
            filepath = filepath + ".gz"
        else:
            with open(filepath, "w") as f:
                json.dump(result, f, indent=2)

        logger.debug("Result written for slide %s", slide_id)

        return filepath

    def read_result(self, slide_id: str) -> Optional[Dict[str, Any]]:
        """Read processing result from storage."""
        filepath = os.path.join(self.config.results_dir, f"{slide_id}_result.json")

        # Try compressed first
        if os.path.exists(filepath + ".gz"):
            with gzip.open(filepath + ".gz", "rt", encoding="utf-8") as f:
                return json.load(f)

        # Try uncompressed
        if os.path.exists(filepath):
            with open(filepath) as f:
                return json.load(f)

        return None

    def create_temp_file(self, suffix: str = "") -> str:
        """Create temporary file."""
        fd, filepath = tempfile.mkstemp(suffix=suffix, dir=self.config.temp_dir)
        os.close(fd)
        return filepath

    def create_temp_dir(self) -> str:
        """Create temporary directory."""
        return tempfile.mkdtemp(dir=self.config.temp_dir)

    def cleanup_temp_files(self, max_age_hours: Optional[int] = None) -> int:
        """Clean up old temporary files."""
        if max_age_hours is None:
            max_age_hours = self.config.cleanup_age_hours

        max_age_seconds = max_age_hours * 3600
        current_time = time.time()

        cleaned_count = 0
        cleaned_size = 0

        for root, dirs, files in os.walk(self.config.temp_dir):
            for filename in files:
                filepath = os.path.join(root, filename)

                try:
                    file_age = current_time - os.path.getmtime(filepath)

                    if file_age > max_age_seconds:
                        file_size = os.path.getsize(filepath)
                        # Safe deletion with archival
                        from src.utils.safe_operations import safe_delete
                        archive_dir = self.cache_dir / "archive"
                        safe_delete(Path(filepath), archive_dir=archive_dir)
                        cleaned_count += 1
                        cleaned_size += file_size

                except Exception as e:
                    logger.warning("Failed to clean up %s: %s", filepath, e)

        if cleaned_count > 0:
            logger.info(
                "Cleaned up %d temporary files (%.2f MB)",
                cleaned_count,
                cleaned_size / (1024 * 1024),
            )

        return cleaned_count

    def get_temp_size(self) -> float:
        """Get total size of temporary files in GB."""
        total_size = 0

        for root, dirs, files in os.walk(self.config.temp_dir):
            for filename in files:
                filepath = os.path.join(root, filename)
                try:
                    total_size += os.path.getsize(filepath)
                except Exception:
                    pass

        return total_size / (1024**3)

    def check_temp_size_limit(self) -> bool:
        """Check if temporary storage is within limits."""
        current_size = self.get_temp_size()

        if current_size > self.config.max_temp_size_gb:
            logger.warning(
                "Temporary storage exceeds limit: %.2fGB > %.2fGB",
                current_size,
                self.config.max_temp_size_gb,
            )
            return False

        return True

    def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics."""

        def get_dir_size(directory: str) -> float:
            total = 0
            try:
                for root, dirs, files in os.walk(directory):
                    for filename in files:
                        filepath = os.path.join(root, filename)
                        total += os.path.getsize(filepath)
            except Exception:
                pass
            return total / (1024**3)

        return {
            "temp_size_gb": get_dir_size(self.config.temp_dir),
            "cache_size_gb": get_dir_size(self.config.cache_dir),
            "results_size_gb": get_dir_size(self.config.results_dir),
            "temp_limit_gb": self.config.max_temp_size_gb,
        }


class CloudStorage:
    """Cloud storage integration (S3, Azure Blob, GCS)."""

    def __init__(self, config: StorageConfig):
        """Initialize cloud storage."""
        self.config = config
        self.client = None

        if config.cloud_enabled:
            self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize cloud storage client."""
        if self.config.cloud_provider == "s3":
            self._initialize_s3()
        elif self.config.cloud_provider == "azure":
            self._initialize_azure()
        elif self.config.cloud_provider == "gcs":
            self._initialize_gcs()
        else:
            raise ValueError(f"Unsupported cloud provider: {self.config.cloud_provider}")

    def _initialize_s3(self) -> None:
        """Initialize AWS S3 client."""
        try:
            import boto3

            self.client = boto3.client("s3")

            logger.info("S3 client initialized: bucket=%s", self.config.cloud_bucket)
        except ImportError:
            logger.error("boto3 not installed. Install with: pip install boto3")
            self.client = None

    def _initialize_azure(self) -> None:
        """Initialize Azure Blob Storage client."""
        try:
            from azure.storage.blob import BlobServiceClient

            # Requires AZURE_STORAGE_CONNECTION_STRING environment variable
            connection_string = os.environ.get("AZURE_STORAGE_CONNECTION_STRING")
            if not connection_string:
                raise ValueError("AZURE_STORAGE_CONNECTION_STRING not set")

            self.client = BlobServiceClient.from_connection_string(connection_string)

            logger.info(
                "Azure Blob Storage client initialized: container=%s", self.config.cloud_bucket
            )
        except ImportError:
            logger.error(
                "azure-storage-blob not installed. Install with: pip install azure-storage-blob"
            )
            self.client = None

    def _initialize_gcs(self) -> None:
        """Initialize Google Cloud Storage client."""
        try:
            from google.cloud import storage

            self.client = storage.Client()

            logger.info("GCS client initialized: bucket=%s", self.config.cloud_bucket)
        except ImportError:
            logger.error(
                "google-cloud-storage not installed. Install with: pip install google-cloud-storage"
            )
            self.client = None

    def upload_file(self, local_path: str, remote_key: str) -> bool:
        """Upload file to cloud storage."""
        if self.client is None:
            return False

        try:
            full_key = f"{self.config.cloud_prefix}/{remote_key}"

            if self.config.cloud_provider == "s3":
                self.client.upload_file(local_path, self.config.cloud_bucket, full_key)
            elif self.config.cloud_provider == "azure":
                container_client = self.client.get_container_client(self.config.cloud_bucket)
                with open(local_path, "rb") as data:
                    container_client.upload_blob(name=full_key, data=data, overwrite=True)
            elif self.config.cloud_provider == "gcs":
                bucket = self.client.bucket(self.config.cloud_bucket)
                blob = bucket.blob(full_key)
                blob.upload_from_filename(local_path)

            logger.debug("Uploaded %s to %s", local_path, full_key)
            return True

        except Exception as e:
            logger.error("Failed to upload %s: %s", local_path, e)
            return False

    def download_file(self, remote_key: str, local_path: str) -> bool:
        """Download file from cloud storage."""
        if self.client is None:
            return False

        try:
            full_key = f"{self.config.cloud_prefix}/{remote_key}"

            if self.config.cloud_provider == "s3":
                self.client.download_file(self.config.cloud_bucket, full_key, local_path)
            elif self.config.cloud_provider == "azure":
                container_client = self.client.get_container_client(self.config.cloud_bucket)
                with open(local_path, "wb") as data:
                    data.write(container_client.download_blob(full_key).readall())
            elif self.config.cloud_provider == "gcs":
                bucket = self.client.bucket(self.config.cloud_bucket)
                blob = bucket.blob(full_key)
                blob.download_to_filename(local_path)

            logger.debug("Downloaded %s to %s", full_key, local_path)
            return True

        except Exception as e:
            logger.error("Failed to download %s: %s", remote_key, e)
            return False


# Global storage instances
_local_storage: Optional[LocalStorage] = None
_cloud_storage: Optional[CloudStorage] = None


def initialize_storage(config: StorageConfig) -> None:
    """Initialize global storage instances."""
    global _local_storage, _cloud_storage

    _local_storage = LocalStorage(config)
    _cloud_storage = CloudStorage(config) if config.cloud_enabled else None

    logger.info("Global storage initialized")


def get_local_storage() -> LocalStorage:
    """Get global local storage instance."""
    if _local_storage is None:
        raise RuntimeError("Local storage not initialized. Call initialize_storage() first.")
    return _local_storage


def get_cloud_storage() -> Optional[CloudStorage]:
    """Get global cloud storage instance."""
    return _cloud_storage
