"""
Tests for storage optimization system.
"""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import h5py
import numpy as np
import pytest
import torch

from src.streaming.storage import (
    CloudStorage,
    LocalStorage,
    StorageConfig,
)


@pytest.fixture
def temp_storage_dir():
    """Create temporary storage directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir


@pytest.fixture
def storage_config(temp_storage_dir):
    """Create test storage configuration."""
    return StorageConfig(
        temp_dir=os.path.join(temp_storage_dir, "temp"),
        cache_dir=os.path.join(temp_storage_dir, "cache"),
        results_dir=os.path.join(temp_storage_dir, "results"),
        enable_compression=True,
        compression_level=6,
        auto_cleanup=True,
        cleanup_age_hours=1,
        max_temp_size_gb=1.0,
        feature_format="hdf5",
        feature_compression="gzip",
    )


class TestLocalStorage:
    """Test local storage."""
    
    def test_initialization(self, storage_config):
        """Test storage initialization."""
        storage = LocalStorage(storage_config)
        
        # Check directories were created
        assert os.path.exists(storage_config.temp_dir)
        assert os.path.exists(storage_config.cache_dir)
        assert os.path.exists(storage_config.results_dir)
    
    def test_write_read_features_hdf5(self, storage_config):
        """Test writing and reading features in HDF5 format."""
        storage = LocalStorage(storage_config)
        
        slide_id = "test_slide_001"
        features = np.random.randn(100, 2048).astype(np.float32)
        coordinates = np.array([[i, i * 10] for i in range(100)])
        metadata = {
            "slide_id": slide_id,
            "patch_size": 256,
            "magnification": 40,
        }
        
        # Write features
        filepath = storage.write_features(slide_id, features, coordinates, metadata)
        
        assert os.path.exists(filepath)
        assert filepath.endswith(".h5")
        
        # Read features
        data = storage.read_features(slide_id)
        
        assert data is not None
        assert "features" in data
        assert "coordinates" in data
        assert "metadata" in data
        
        np.testing.assert_array_almost_equal(data["features"], features)
        np.testing.assert_array_equal(data["coordinates"], coordinates)
        assert data["metadata"]["slide_id"] == slide_id
    
    def test_write_read_features_npz(self, storage_config):
        """Test writing and reading features in NPZ format."""
        storage_config.feature_format = "npz"
        storage = LocalStorage(storage_config)
        
        slide_id = "test_slide_002"
        features = np.random.randn(50, 1024).astype(np.float32)
        coordinates = np.array([[i, i * 5] for i in range(50)])
        metadata = {"test": "data"}
        
        # Write features
        filepath = storage.write_features(slide_id, features, coordinates, metadata)
        
        assert os.path.exists(filepath)
        assert filepath.endswith(".npz")
        
        # Read features
        data = storage.read_features(slide_id)
        
        assert data is not None
        np.testing.assert_array_almost_equal(data["features"], features)
        np.testing.assert_array_equal(data["coordinates"], coordinates)
        assert data["metadata"]["test"] == "data"
    
    def test_write_read_features_pytorch(self, storage_config):
        """Test writing and reading features in PyTorch format."""
        storage_config.feature_format = "pt"
        storage = LocalStorage(storage_config)
        
        slide_id = "test_slide_003"
        features = torch.randn(75, 512)
        coordinates = np.array([[i, i * 3] for i in range(75)])
        metadata = {"model": "resnet50"}
        
        # Write features
        filepath = storage.write_features(slide_id, features, coordinates, metadata)
        
        assert os.path.exists(filepath)
        assert filepath.endswith(".pt")
        
        # Read features
        data = storage.read_features(slide_id)
        
        assert data is not None
        np.testing.assert_array_almost_equal(data["features"], features.numpy())
        np.testing.assert_array_equal(data["coordinates"], coordinates)
        assert data["metadata"]["model"] == "resnet50"
    
    def test_write_features_torch_tensor(self, storage_config):
        """Test writing torch tensor features."""
        storage = LocalStorage(storage_config)
        
        slide_id = "test_slide_004"
        features = torch.randn(100, 2048)
        
        # Write features (should convert to numpy)
        filepath = storage.write_features(slide_id, features)
        
        assert os.path.exists(filepath)
        
        # Read features
        data = storage.read_features(slide_id)
        
        assert data is not None
        np.testing.assert_array_almost_equal(data["features"], features.numpy())
    
    def test_write_features_compression(self, storage_config):
        """Test feature compression."""
        storage_config.enable_compression = True
        storage = LocalStorage(storage_config)
        
        slide_id = "test_slide_005"
        # Create compressible data (lots of zeros)
        features = np.zeros((1000, 2048), dtype=np.float32)
        features[:10, :10] = 1.0
        
        # Write with compression
        filepath = storage.write_features(slide_id, features)
        compressed_size = os.path.getsize(filepath)
        
        # Write without compression
        storage_config.enable_compression = False
        storage_no_comp = LocalStorage(storage_config)
        slide_id_no_comp = "test_slide_006"
        filepath_no_comp = storage_no_comp.write_features(slide_id_no_comp, features)
        uncompressed_size = os.path.getsize(filepath_no_comp)
        
        # Compressed should be smaller
        assert compressed_size < uncompressed_size
    
    def test_read_nonexistent_features(self, storage_config):
        """Test reading non-existent features."""
        storage = LocalStorage(storage_config)
        
        data = storage.read_features("nonexistent_slide")
        
        assert data is None
    
    def test_write_read_result(self, storage_config):
        """Test writing and reading results."""
        storage = LocalStorage(storage_config)
        
        slide_id = "test_slide_007"
        result = {
            "prediction": 1,
            "confidence": 0.95,
            "processing_time": 25.5,
            "attention_weights": [0.1, 0.2, 0.3, 0.4],
        }
        
        # Write result
        filepath = storage.write_result(slide_id, result)
        
        assert os.path.exists(filepath)
        
        # Read result
        loaded = storage.read_result(slide_id)
        
        assert loaded is not None
        assert loaded["prediction"] == result["prediction"]
        assert loaded["confidence"] == result["confidence"]
        assert loaded["processing_time"] == result["processing_time"]
        assert loaded["attention_weights"] == result["attention_weights"]
    
    def test_write_result_compression(self, storage_config):
        """Test result compression."""
        storage_config.enable_compression = True
        storage = LocalStorage(storage_config)
        
        slide_id = "test_slide_008"
        result = {"data": "x" * 10000}  # Large compressible data
        
        # Write with compression
        filepath = storage.write_result(slide_id, result)
        
        assert filepath.endswith(".gz")
        assert os.path.exists(filepath)
        
        # Read compressed result
        loaded = storage.read_result(slide_id)
        
        assert loaded is not None
        assert loaded["data"] == result["data"]
    
    def test_read_nonexistent_result(self, storage_config):
        """Test reading non-existent result."""
        storage = LocalStorage(storage_config)
        
        result = storage.read_result("nonexistent_slide")
        
        assert result is None
    
    def test_create_temp_file(self, storage_config):
        """Test temporary file creation."""
        storage = LocalStorage(storage_config)
        
        filepath = storage.create_temp_file(suffix=".txt")
        
        assert os.path.exists(filepath)
        assert filepath.endswith(".txt")
        assert filepath.startswith(storage_config.temp_dir)
        
        # Cleanup
        os.remove(filepath)
    
    def test_create_temp_dir(self, storage_config):
        """Test temporary directory creation."""
        storage = LocalStorage(storage_config)
        
        dirpath = storage.create_temp_dir()
        
        assert os.path.exists(dirpath)
        assert os.path.isdir(dirpath)
        assert dirpath.startswith(storage_config.temp_dir)
        
        # Cleanup
        os.rmdir(dirpath)
    
    def test_cleanup_temp_files(self, storage_config):
        """Test temporary file cleanup."""
        storage = LocalStorage(storage_config)
        
        # Create some temporary files
        old_file = storage.create_temp_file(suffix=".old")
        new_file = storage.create_temp_file(suffix=".new")
        
        # Make old file appear old
        old_time = os.path.getmtime(old_file) - 7200  # 2 hours ago
        os.utime(old_file, (old_time, old_time))
        
        # Cleanup files older than 1 hour
        cleaned = storage.cleanup_temp_files(max_age_hours=1)
        
        assert cleaned >= 1
        assert not os.path.exists(old_file)
        assert os.path.exists(new_file)
        
        # Cleanup
        os.remove(new_file)
    
    def test_get_temp_size(self, storage_config):
        """Test getting temporary storage size."""
        storage = LocalStorage(storage_config)
        
        # Create some temporary files
        file1 = storage.create_temp_file()
        file2 = storage.create_temp_file()
        
        # Write some data
        with open(file1, "wb") as f:
            f.write(b"x" * 1024 * 1024)  # 1MB
        
        with open(file2, "wb") as f:
            f.write(b"x" * 1024 * 1024)  # 1MB
        
        # Get size
        size_gb = storage.get_temp_size()
        
        assert size_gb > 0
        assert size_gb < 1.0  # Should be less than 1GB
        
        # Cleanup
        os.remove(file1)
        os.remove(file2)
    
    def test_check_temp_size_limit(self, storage_config):
        """Test temporary size limit checking."""
        storage_config.max_temp_size_gb = 0.001  # 1MB limit
        storage = LocalStorage(storage_config)
        
        # Should be within limit initially
        assert storage.check_temp_size_limit() is True
        
        # Create large file
        large_file = storage.create_temp_file()
        with open(large_file, "wb") as f:
            f.write(b"x" * 2 * 1024 * 1024)  # 2MB
        
        # Should exceed limit
        assert storage.check_temp_size_limit() is False
        
        # Cleanup
        os.remove(large_file)
    
    def test_get_storage_stats(self, storage_config):
        """Test storage statistics."""
        storage = LocalStorage(storage_config)
        
        # Create some files
        slide_id = "test_slide_009"
        features = np.random.randn(100, 2048).astype(np.float32)
        storage.write_features(slide_id, features)
        
        result = {"test": "data"}
        storage.write_result(slide_id, result)
        
        # Get stats
        stats = storage.get_storage_stats()
        
        assert "temp_size_gb" in stats
        assert "cache_size_gb" in stats
        assert "results_size_gb" in stats
        assert "temp_limit_gb" in stats
        
        assert stats["cache_size_gb"] > 0
        assert stats["results_size_gb"] > 0
        assert stats["temp_limit_gb"] == storage_config.max_temp_size_gb
    
    def test_unsupported_format(self, storage_config):
        """Test unsupported feature format."""
        storage_config.feature_format = "unsupported"
        storage = LocalStorage(storage_config)
        
        slide_id = "test_slide_010"
        features = np.random.randn(10, 10)
        
        with pytest.raises(ValueError, match="Unsupported feature format"):
            storage.write_features(slide_id, features)


class TestCloudStorage:
    """Test cloud storage."""
    
    def test_initialization_disabled(self, storage_config):
        """Test cloud storage initialization when disabled."""
        storage_config.cloud_enabled = False
        cloud = CloudStorage(storage_config)
        
        assert cloud.client is None
    
    def test_initialization_s3(self, storage_config):
        """Test S3 initialization."""
        storage_config.cloud_enabled = True
        storage_config.cloud_provider = "s3"
        storage_config.cloud_bucket = "test-bucket"
        
        with patch("src.streaming.storage.boto3") as mock_boto3:
            mock_client = MagicMock()
            mock_boto3.client.return_value = mock_client
            
            cloud = CloudStorage(storage_config)
            
            assert cloud.client is not None
            mock_boto3.client.assert_called_once_with("s3")
    
    def test_initialization_s3_missing_boto3(self, storage_config):
        """Test S3 initialization with missing boto3."""
        storage_config.cloud_enabled = True
        storage_config.cloud_provider = "s3"
        
        with patch("src.streaming.storage.boto3", side_effect=ImportError):
            cloud = CloudStorage(storage_config)
            
            assert cloud.client is None
    
    def test_initialization_azure(self, storage_config):
        """Test Azure initialization."""
        storage_config.cloud_enabled = True
        storage_config.cloud_provider = "azure"
        storage_config.cloud_bucket = "test-container"
        
        with patch.dict(os.environ, {"AZURE_STORAGE_CONNECTION_STRING": "test_connection"}):
            with patch("src.streaming.storage.BlobServiceClient") as mock_blob:
                mock_client = MagicMock()
                mock_blob.from_connection_string.return_value = mock_client
                
                cloud = CloudStorage(storage_config)
                
                assert cloud.client is not None
                mock_blob.from_connection_string.assert_called_once_with("test_connection")
    
    def test_initialization_azure_missing_connection(self, storage_config):
        """Test Azure initialization with missing connection string."""
        storage_config.cloud_enabled = True
        storage_config.cloud_provider = "azure"
        
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ValueError, match="AZURE_STORAGE_CONNECTION_STRING not set"):
                CloudStorage(storage_config)
    
    def test_initialization_gcs(self, storage_config):
        """Test GCS initialization."""
        storage_config.cloud_enabled = True
        storage_config.cloud_provider = "gcs"
        storage_config.cloud_bucket = "test-bucket"
        
        with patch("src.streaming.storage.storage") as mock_storage:
            mock_client = MagicMock()
            mock_storage.Client.return_value = mock_client
            
            cloud = CloudStorage(storage_config)
            
            assert cloud.client is not None
            mock_storage.Client.assert_called_once()
    
    def test_initialization_unsupported_provider(self, storage_config):
        """Test unsupported cloud provider."""
        storage_config.cloud_enabled = True
        storage_config.cloud_provider = "unsupported"
        
        with pytest.raises(ValueError, match="Unsupported cloud provider"):
            CloudStorage(storage_config)
    
    def test_upload_file_s3(self, storage_config, temp_storage_dir):
        """Test S3 file upload."""
        storage_config.cloud_enabled = True
        storage_config.cloud_provider = "s3"
        storage_config.cloud_bucket = "test-bucket"
        storage_config.cloud_prefix = "test-prefix"
        
        with patch("src.streaming.storage.boto3") as mock_boto3:
            mock_client = MagicMock()
            mock_boto3.client.return_value = mock_client
            
            cloud = CloudStorage(storage_config)
            
            # Create test file
            test_file = os.path.join(temp_storage_dir, "test.txt")
            with open(test_file, "w") as f:
                f.write("test data")
            
            # Upload
            result = cloud.upload_file(test_file, "test.txt")
            
            assert result is True
            mock_client.upload_file.assert_called_once_with(
                test_file,
                "test-bucket",
                "test-prefix/test.txt"
            )
    
    def test_upload_file_disabled(self, storage_config, temp_storage_dir):
        """Test upload when cloud storage is disabled."""
        storage_config.cloud_enabled = False
        cloud = CloudStorage(storage_config)
        
        test_file = os.path.join(temp_storage_dir, "test.txt")
        with open(test_file, "w") as f:
            f.write("test data")
        
        result = cloud.upload_file(test_file, "test.txt")
        
        assert result is False
    
    def test_download_file_s3(self, storage_config, temp_storage_dir):
        """Test S3 file download."""
        storage_config.cloud_enabled = True
        storage_config.cloud_provider = "s3"
        storage_config.cloud_bucket = "test-bucket"
        storage_config.cloud_prefix = "test-prefix"
        
        with patch("src.streaming.storage.boto3") as mock_boto3:
            mock_client = MagicMock()
            mock_boto3.client.return_value = mock_client
            
            cloud = CloudStorage(storage_config)
            
            # Download
            local_path = os.path.join(temp_storage_dir, "downloaded.txt")
            result = cloud.download_file("test.txt", local_path)
            
            assert result is True
            mock_client.download_file.assert_called_once_with(
                "test-bucket",
                "test-prefix/test.txt",
                local_path
            )
    
    def test_download_file_disabled(self, storage_config, temp_storage_dir):
        """Test download when cloud storage is disabled."""
        storage_config.cloud_enabled = False
        cloud = CloudStorage(storage_config)
        
        local_path = os.path.join(temp_storage_dir, "downloaded.txt")
        result = cloud.download_file("test.txt", local_path)
        
        assert result is False
    
    def test_upload_file_error(self, storage_config, temp_storage_dir):
        """Test upload error handling."""
        storage_config.cloud_enabled = True
        storage_config.cloud_provider = "s3"
        
        with patch("src.streaming.storage.boto3") as mock_boto3:
            mock_client = MagicMock()
            mock_client.upload_file.side_effect = Exception("Upload failed")
            mock_boto3.client.return_value = mock_client
            
            cloud = CloudStorage(storage_config)
            
            test_file = os.path.join(temp_storage_dir, "test.txt")
            with open(test_file, "w") as f:
                f.write("test data")
            
            result = cloud.upload_file(test_file, "test.txt")
            
            assert result is False
    
    def test_download_file_error(self, storage_config, temp_storage_dir):
        """Test download error handling."""
        storage_config.cloud_enabled = True
        storage_config.cloud_provider = "s3"
        
        with patch("src.streaming.storage.boto3") as mock_boto3:
            mock_client = MagicMock()
            mock_client.download_file.side_effect = Exception("Download failed")
            mock_boto3.client.return_value = mock_client
            
            cloud = CloudStorage(storage_config)
            
            local_path = os.path.join(temp_storage_dir, "downloaded.txt")
            result = cloud.download_file("test.txt", local_path)
            
            assert result is False


@pytest.mark.integration
class TestStorageIntegration:
    """Integration tests for storage system."""
    
    def test_full_workflow(self, storage_config):
        """Test complete storage workflow."""
        storage = LocalStorage(storage_config)
        
        # Write features
        slide_id = "integration_test_slide"
        features = np.random.randn(200, 2048).astype(np.float32)
        coordinates = np.array([[i, i * 10] for i in range(200)])
        metadata = {
            "slide_id": slide_id,
            "model": "resnet50",
            "timestamp": "2026-04-26T12:00:00",
        }
        
        feature_path = storage.write_features(slide_id, features, coordinates, metadata)
        assert os.path.exists(feature_path)
        
        # Write result
        result = {
            "prediction": 1,
            "confidence": 0.92,
            "processing_time": 28.3,
            "feature_path": feature_path,
        }
        
        result_path = storage.write_result(slide_id, result)
        assert os.path.exists(result_path)
        
        # Read features
        loaded_features = storage.read_features(slide_id)
        assert loaded_features is not None
        np.testing.assert_array_almost_equal(loaded_features["features"], features)
        np.testing.assert_array_equal(loaded_features["coordinates"], coordinates)
        assert loaded_features["metadata"]["model"] == "resnet50"
        
        # Read result
        loaded_result = storage.read_result(slide_id)
        assert loaded_result is not None
        assert loaded_result["prediction"] == 1
        assert loaded_result["confidence"] == 0.92
        
        # Get stats
        stats = storage.get_storage_stats()
        assert stats["cache_size_gb"] > 0
        assert stats["results_size_gb"] > 0
    
    def test_multiple_formats(self, storage_config):
        """Test storage with multiple formats."""
        formats = ["hdf5", "npz", "pt"]
        
        for fmt in formats:
            storage_config.feature_format = fmt
            storage = LocalStorage(storage_config)
            
            slide_id = f"test_slide_{fmt}"
            features = np.random.randn(50, 512).astype(np.float32)
            
            # Write
            filepath = storage.write_features(slide_id, features)
            assert os.path.exists(filepath)
            
            # Read
            data = storage.read_features(slide_id)
            assert data is not None
            np.testing.assert_array_almost_equal(data["features"], features, decimal=5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
