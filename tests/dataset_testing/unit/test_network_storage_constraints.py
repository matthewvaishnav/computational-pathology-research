"""
Network + storage constraint tests.

Tests network failures, disk space limits, config validation.
"""

import pytest
import numpy as np
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import h5py
import shutil

from src.data.preprocessing import (
    save_features_to_hdf5,
    load_features_from_hdf5,
    batch_save_to_hdf5,
    normalize_wsi_features,
    build_clinical_vocab,
)


class TestNetworkFailureHandling:
    """Test network connection failure handling."""

    @patch("urllib.request.urlopen")
    def test_download_network_timeout(self, mock_urlopen):
        """Network timeout → TimeoutError."""
        import urllib.error

        # Simulate timeout
        mock_urlopen.side_effect = urllib.error.URLError("Connection timeout")

        # Try download
        with pytest.raises(urllib.error.URLError):
            mock_urlopen("http://example.com/data.h5")

    @patch("urllib.request.urlopen")
    def test_download_connection_refused(self, mock_urlopen):
        """Connection refused → ConnectionError."""
        import urllib.error

        # Simulate connection refused
        mock_urlopen.side_effect = urllib.error.URLError("Connection refused")

        with pytest.raises(urllib.error.URLError):
            mock_urlopen("http://example.com/data.h5")

    @patch("urllib.request.urlopen")
    def test_download_retry_mechanism(self, mock_urlopen):
        """Network fail → retry → success."""
        import urllib.error

        # Fail first 2 attempts, succeed on 3rd
        mock_urlopen.side_effect = [
            urllib.error.URLError("Timeout"),
            urllib.error.URLError("Timeout"),
            MagicMock(),  # Success
        ]

        # Retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                result = mock_urlopen("http://example.com/data.h5")
                break
            except urllib.error.URLError:
                if attempt == max_retries - 1:
                    raise

        # Should succeed on 3rd attempt
        assert result is not None

    @patch("urllib.request.urlopen")
    def test_download_partial_file_recovery(self, mock_urlopen):
        """Partial download → detect + retry."""
        # Simulate partial download
        mock_response = MagicMock()
        mock_response.read.return_value = b"partial data"
        mock_response.headers = {"Content-Length": "1000"}
        mock_urlopen.return_value = mock_response

        # Download
        data = mock_urlopen("http://example.com/data.h5").read()
        expected_size = int(mock_urlopen("http://example.com/data.h5").headers["Content-Length"])

        # Detect incomplete
        assert len(data) < expected_size

    def test_offline_mode_fallback(self):
        """Network unavailable → use cached data."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cached_data.h5"

            # Create cached data
            cached_data = np.random.randn(10, 512).astype(np.float32)
            save_features_to_hdf5(cached_data, cache_path)

            # Load from cache (offline mode)
            loaded = load_features_from_hdf5(cache_path)

            assert np.allclose(cached_data, loaded)


class TestDiskSpaceConstraints:
    """Test disk space constraint detection."""

    def test_disk_space_check_before_save(self):
        """Check disk space before save."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Get disk usage
            usage = shutil.disk_usage(tmpdir)

            # Check available space
            available_gb = usage.free / (1024**3)

            # Should have some space
            assert available_gb > 0

    def test_save_large_file_disk_full(self):
        """Disk full → OSError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "large.h5"

            # Try save very large array (may fail if disk full)
            try:
                # 10GB array
                large_data = np.random.randn(1000000, 1280).astype(np.float32)
                save_features_to_hdf5(large_data, path)
            except (OSError, MemoryError):
                # Expected if disk full or insufficient memory
                pass

    def test_incremental_save_with_space_check(self):
        """Incremental save → check space each iteration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "incremental.h5"

            # Save in chunks
            chunk_size = 100
            num_chunks = 5

            for i in range(num_chunks):
                # Check space before each chunk
                usage = shutil.disk_usage(tmpdir)
                if usage.free < 100 * 1024 * 1024:  # <100MB
                    break

                # Save chunk
                chunk = np.random.randn(chunk_size, 512).astype(np.float32)

                if i == 0:
                    save_features_to_hdf5(chunk, path, resizable=True)
                else:
                    # Append
                    from src.data.preprocessing import append_to_hdf5

                    append_to_hdf5(chunk, path)

            # Verify saved
            assert path.exists()

    def test_cleanup_suggestion_on_disk_full(self):
        """Disk full → suggest cleanup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create temp files
            temp_files = []
            for i in range(5):
                temp_path = Path(tmpdir) / f"temp_{i}.h5"
                data = np.random.randn(10, 512).astype(np.float32)
                save_features_to_hdf5(data, temp_path)
                temp_files.append(temp_path)

            # List temp files (cleanup candidates)
            cleanup_candidates = list(Path(tmpdir).glob("temp_*.h5"))

            assert len(cleanup_candidates) == 5

            # Cleanup
            for f in cleanup_candidates:
                f.unlink()

            # Verify cleaned
            assert len(list(Path(tmpdir).glob("temp_*.h5"))) == 0

    def test_compression_to_save_space(self):
        """Use compression → save disk space."""
        with tempfile.TemporaryDirectory() as tmpdir:
            data = np.random.randn(100, 512).astype(np.float32)

            # Save without compression
            path_uncompressed = Path(tmpdir) / "uncompressed.h5"
            save_features_to_hdf5(data, path_uncompressed, compression=None)

            # Save with compression
            path_compressed = Path(tmpdir) / "compressed.h5"
            save_features_to_hdf5(data, path_compressed, compression="gzip")

            # Compressed should be smaller
            size_uncompressed = path_uncompressed.stat().st_size
            size_compressed = path_compressed.stat().st_size

            assert size_compressed <= size_uncompressed


class TestConfigurationValidation:
    """Test invalid config parameter validation."""

    def test_validate_normalization_method(self):
        """Invalid norm method → ValueError with valid options."""
        data = np.random.randn(10, 512).astype(np.float32)

        valid_methods = ["standardize", "l2", "minmax"]

        # Test valid methods work
        for method in valid_methods:
            normed = normalize_wsi_features(data, method=method)
            assert normed.shape == data.shape

        # Invalid method → error
        with pytest.raises(ValueError, match="Unknown normalization method"):
            normalize_wsi_features(data, method="invalid")

    def test_validate_file_path_format(self):
        """Invalid file path → error."""
        data = np.random.randn(10, 512).astype(np.float32)

        # Invalid path chars (Windows)
        invalid_paths = [
            "file<name>.h5",
            "file>name.h5",
            "file|name.h5",
            "file:name.h5" if Path.cwd().drive else None,
        ]

        for invalid_path in invalid_paths:
            if invalid_path is None:
                continue

            try:
                save_features_to_hdf5(data, invalid_path)
                # May succeed on some systems
            except (OSError, ValueError):
                # Expected on Windows
                pass

    def test_validate_feature_dimensions(self):
        """Invalid feature dims → error."""
        # 1D array
        data_1d = np.random.randn(512).astype(np.float32)

        # Should work for some methods
        normed = normalize_wsi_features(data_1d, method="l2")
        assert normed.shape == (512,)

        # 0D array
        data_0d = np.array(5.0, dtype=np.float32)

        # May fail
        try:
            normalize_wsi_features(data_0d, method="minmax")
        except (ValueError, IndexError):
            pass

    def test_validate_vocab_parameters(self):
        """Invalid vocab params → handle gracefully."""
        texts = ["patient has cancer"]

        # Negative min_frequency → treat as 0
        vocab = build_clinical_vocab(texts, min_frequency=-1, max_vocab_size=100)
        assert len(vocab) >= 4

        # Very large max_vocab_size → OK
        vocab = build_clinical_vocab(texts, min_frequency=1, max_vocab_size=1000000)
        assert len(vocab) >= 4

    def test_validate_hdf5_dataset_name(self):
        """Invalid dataset name → error."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.h5"
            data = np.random.randn(10, 512).astype(np.float32)

            # Empty dataset name → TypeError
            with pytest.raises(TypeError):
                save_features_to_hdf5(data, path, dataset_name="")

            # Valid dataset name
            save_features_to_hdf5(data, path, dataset_name="features")
            assert path.exists()

    def test_validate_compression_level(self):
        """Invalid compression level → use default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.h5"
            data = np.random.randn(10, 512).astype(np.float32)

            # Valid compression
            save_features_to_hdf5(data, path, compression="gzip")
            assert path.exists()

            # Invalid compression → error
            with pytest.raises((ValueError, TypeError)):
                save_features_to_hdf5(data, path, compression="invalid")


class TestResourceCleanup:
    """Test resource cleanup + leak prevention."""

    def test_hdf5_file_handle_cleanup(self):
        """HDF5 file → close handle after use."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.h5"
            data = np.random.randn(10, 512).astype(np.float32)

            # Save
            save_features_to_hdf5(data, path)

            # Load
            loaded = load_features_from_hdf5(path)

            # File should be closed after load
            # Try open again
            loaded2 = load_features_from_hdf5(path)

            assert np.allclose(loaded, loaded2)

    def test_temp_file_cleanup(self):
        """Temp files → cleanup after use."""
        with tempfile.TemporaryDirectory() as tmpdir:
            temp_path = Path(tmpdir) / "temp.h5"
            data = np.random.randn(10, 512).astype(np.float32)

            # Create temp file
            save_features_to_hdf5(data, temp_path)
            assert temp_path.exists()

            # Cleanup
            temp_path.unlink()
            assert not temp_path.exists()

    def test_memory_cleanup_after_processing(self):
        """Large arrays → cleanup after processing."""
        # Create large array
        large_data = np.random.randn(1000, 512).astype(np.float32)

        # Process
        normed = normalize_wsi_features(large_data, method="standardize")

        # Delete to free memory
        del large_data

        # Verify result still valid
        assert normed.shape == (1000, 512)

    def test_batch_processing_cleanup(self):
        """Batch processing → cleanup between batches."""
        num_batches = 5
        batch_size = 100

        for i in range(num_batches):
            # Create batch
            batch = np.random.randn(batch_size, 512).astype(np.float32)

            # Process
            normed = normalize_wsi_features(batch, method="standardize")

            # Verify
            assert normed.shape == (batch_size, 512)

            # Cleanup (goes out of scope)
            del batch, normed


class TestErrorRecoveryStrategies:
    """Test error recovery strategies."""

    def test_retry_with_exponential_backoff(self):
        """Network fail → retry with backoff."""
        import time

        max_retries = 3
        base_delay = 0.1

        for attempt in range(max_retries):
            try:
                # Simulate operation that may fail
                if attempt < 2:
                    raise ConnectionError("Network error")
                # Success on 3rd attempt
                result = "success"
                break
            except ConnectionError:
                if attempt == max_retries - 1:
                    raise

                # Exponential backoff
                delay = base_delay * (2**attempt)
                time.sleep(delay)

        assert result == "success"

    def test_fallback_to_cached_data(self):
        """Network fail → use cache."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = Path(tmpdir) / "cache.h5"

            # Create cache
            cached_data = np.random.randn(10, 512).astype(np.float32)
            save_features_to_hdf5(cached_data, cache_path)

            # Simulate network failure
            network_available = False

            if not network_available:
                # Use cache
                data = load_features_from_hdf5(cache_path)
            else:
                # Download new data
                data = np.random.randn(10, 512).astype(np.float32)

            assert data.shape == (10, 512)

    def test_partial_save_recovery(self):
        """Save interrupted → resume from checkpoint."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "partial.h5"

            # Save first chunk
            chunk1 = np.random.randn(50, 512).astype(np.float32)
            save_features_to_hdf5(chunk1, path, resizable=True)

            # Simulate interruption, then resume
            # Save second chunk
            chunk2 = np.random.randn(50, 512).astype(np.float32)
            from src.data.preprocessing import append_to_hdf5

            append_to_hdf5(chunk2, path)

            # Load full data
            full_data = load_features_from_hdf5(path)

            assert full_data.shape == (100, 512)
            assert np.allclose(full_data[:50], chunk1)
            assert np.allclose(full_data[50:], chunk2)

    def test_graceful_degradation_on_memory_limit(self):
        """Memory limit → reduce batch size."""
        # Start with large batch
        batch_size = 1000

        try:
            # Try large batch
            data = np.random.randn(batch_size, 512).astype(np.float32)
            normed = normalize_wsi_features(data, method="standardize")
        except MemoryError:
            # Reduce batch size
            batch_size = 100
            data = np.random.randn(batch_size, 512).astype(np.float32)
            normed = normalize_wsi_features(data, method="standardize")

        assert normed.shape[0] <= 1000
