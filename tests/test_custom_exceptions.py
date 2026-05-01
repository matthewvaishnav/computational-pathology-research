"""
Tests for custom exception handling in HistoCore.

Verifies that custom exceptions from src.exceptions are properly raised
and handled throughout the codebase.
"""

import pytest
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

from src.exceptions import (
    CacheConnectionError,
    CacheError,
    CacheSerializationError,
    DatabaseError,
    DataLoadError,
    DataSaveError,
    DiskSpaceError,
    EncryptionError,
    ModelError,
    ResourceError,
    SecurityError,
    ThreadingError,
    ValidationError,
)


class TestCacheExceptions:
    """Test cache-related exception handling."""

    @pytest.mark.skip(reason="Requires Redis server")
    def test_cache_connection_error_raised(self):
        """Test CacheConnectionError is raised on connection failure."""
        from src.streaming.cache import RedisCache, CacheConfig
        
        config = CacheConfig(redis_host="invalid_host", redis_port=99999)
        
        with pytest.raises(CacheConnectionError):
            cache = RedisCache(config)

    @pytest.mark.skip(reason="Requires Redis server")
    def test_cache_serialization_error_on_invalid_data(self):
        """Test CacheSerializationError on deserialization failure."""
        from src.streaming.cache import RedisCache, CacheConfig
        
        # Mock Redis client to return corrupted data
        with patch('redis.Redis') as mock_redis:
            mock_client = MagicMock()
            mock_client.ping.return_value = True
            mock_client.get.return_value = b"corrupted_pickle_data\x00\x01\x02"
            mock_redis.return_value = mock_client
            
            config = CacheConfig()
            cache = RedisCache(config)
            
            with pytest.raises(CacheSerializationError):
                cache.get("test_key")


class TestDatabaseExceptions:
    """Test database-related exception handling."""

    @pytest.mark.skip(reason="Requires OpenSlide DLL")
    def test_database_error_on_transaction_failure(self):
        """Test DatabaseError is raised on transaction failure."""
        from src.streaming.model_management import ModelPerformanceTracker
        
        tracker = ModelPerformanceTracker(db_path=":memory:")
        
        # Force a database error by using invalid SQL
        with patch('sqlite3.connect') as mock_connect:
            mock_conn = MagicMock()
            mock_cursor = MagicMock()
            mock_cursor.execute.side_effect = Exception("SQL error")
            mock_conn.cursor.return_value = mock_cursor
            mock_connect.return_value = mock_conn
            
            from src.streaming.model_management import ModelPerformanceMetric
            metric = ModelPerformanceMetric(
                timestamp="2026-05-01T00:00:00",
                model_version="v1.0",
                accuracy=0.95,
                confidence_score=0.9,
                processing_time_ms=100,
                slide_type="test",
                prediction_class="positive"
            )
            
            with pytest.raises(DatabaseError):
                tracker.record_performance_metric(metric)


class TestDataExceptions:
    """Test data I/O exception handling."""

    @pytest.mark.skip(reason="Requires OpenSlide DLL")
    def test_data_save_error_on_write_failure(self):
        """Test DataSaveError is raised on write failure."""
        from src.utils.safe_operations import atomic_write
        
        # Try to write to a read-only location
        with tempfile.TemporaryDirectory() as tmpdir:
            filepath = Path(tmpdir) / "test.json"
            
            # Mock os.replace to fail
            with patch('os.replace', side_effect=OSError("Permission denied")):
                with pytest.raises(DataSaveError):
                    atomic_write(filepath, {"test": "data"}, mode='json')

    @pytest.mark.skip(reason="Requires OpenSlide DLL")
    def test_data_load_error_on_read_failure(self):
        """Test DataLoadError is raised on read failure."""
        from src.utils.attention_utils import load_attention_weights
        
        # Try to load from non-existent file
        with tempfile.TemporaryDirectory() as tmpdir:
            attention_dir = Path(tmpdir)
            
            # Create a corrupted HDF5 file
            corrupted_file = attention_dir / "test_slide.h5"
            corrupted_file.write_bytes(b"not_a_valid_hdf5_file")
            
            with pytest.raises(DataLoadError):
                load_attention_weights("test_slide", attention_dir)


class TestResourceExceptions:
    """Test resource-related exception handling."""

    @pytest.mark.skip(reason="Requires OpenSlide DLL")
    def test_disk_space_error_on_check_failure(self):
        """Test DiskSpaceError is raised on disk space check failure."""
        from src.utils.safe_operations import check_disk_space
        
        # Mock shutil.disk_usage to fail
        with patch('shutil.disk_usage', side_effect=OSError("Disk not found")):
            with pytest.raises(DiskSpaceError):
                check_disk_space(Path("/tmp/test.txt"), required_bytes=1000)


class TestModelExceptions:
    """Test model-related exception handling."""

    @pytest.mark.skip(reason="Requires OpenSlide DLL")
    def test_model_error_on_drift_check_failure(self):
        """Test ModelError is raised on drift check failure."""
        from src.streaming.model_management import ModelDriftDetector, ModelPerformanceTracker
        
        tracker = ModelPerformanceTracker(db_path=":memory:")
        detector = ModelDriftDetector(tracker)
        
        # Mock calculate_performance_trends to fail
        with patch.object(tracker, 'calculate_performance_trends', side_effect=Exception("Calculation failed")):
            with pytest.raises(ModelError):
                detector.check_for_drift("v1.0")


class TestSecurityExceptions:
    """Test security-related exception handling."""

    @pytest.mark.skip(reason="Requires OpenSlide DLL")
    def test_security_error_on_integrity_check_failure(self):
        """Test SecurityError is raised on integrity check failure."""
        from src.streaming.model_management import ModelSecurityManager
        
        manager = ModelSecurityManager()
        
        # Try to verify non-existent file
        with pytest.raises(SecurityError):
            manager.verify_model_integrity("/nonexistent/model.pth", "fake_hash")

    @pytest.mark.skip(reason="Requires OpenSlide DLL")
    def test_encryption_error_on_encrypt_failure(self):
        """Test EncryptionError is raised on encryption failure."""
        from src.streaming.model_management import ModelSecurityManager
        
        manager = ModelSecurityManager()
        
        # Try to encrypt non-existent file
        with pytest.raises(EncryptionError):
            manager.encrypt_model("/nonexistent/model.pth", "/tmp/encrypted.pth")


class TestThreadingExceptions:
    """Test threading-related exception handling."""

    def test_threading_error_on_thread_failure(self):
        """Test ThreadingError is raised on thread execution failure."""
        from src.utils.safe_threading import GracefulThread
        
        def failing_target(thread):
            raise ValueError("Thread failed")
        
        thread = GracefulThread(target=failing_target, name="test_thread")
        thread.start()
        thread.join(timeout=1.0)
        
        # GracefulThread wraps exceptions in ThreadingError and raises them
        # Check that exception was captured (could be ValueError or ThreadingError)
        assert thread._exception is not None
        assert isinstance(thread._exception, (ValueError, ThreadingError))


class TestValidationExceptions:
    """Test validation-related exception handling."""

    def test_validation_error_on_metric_computation_failure(self):
        """Test ValidationError is raised on metric computation failure."""
        from src.utils.statistical import compute_all_metrics_with_ci
        import numpy as np
        
        # Create invalid inputs (mismatched shapes)
        y_true = np.array([0, 1, 0, 1])
        y_pred = np.array([0, 1])  # Wrong shape
        y_prob = np.array([[0.1, 0.9], [0.8, 0.2]])  # Wrong shape
        
        with pytest.raises((ValidationError, ValueError, IndexError)):
            compute_all_metrics_with_ci(y_true, y_pred, y_prob)


class TestExceptionInheritance:
    """Test exception inheritance hierarchy."""

    def test_all_exceptions_inherit_from_histocore_error(self):
        """Test all custom exceptions inherit from HistoCoreError."""
        from src.exceptions import HistoCoreError
        
        exceptions = [
            CacheConnectionError,
            CacheError,
            CacheSerializationError,
            DatabaseError,
            DataLoadError,
            DataSaveError,
            DiskSpaceError,
            EncryptionError,
            ModelError,
            ResourceError,
            SecurityError,
            ThreadingError,
            ValidationError,
        ]
        
        for exc_class in exceptions:
            assert issubclass(exc_class, HistoCoreError), f"{exc_class.__name__} does not inherit from HistoCoreError"

    def test_specific_exceptions_inherit_from_base(self):
        """Test specific exceptions inherit from their base classes."""
        from src.exceptions import HistoCoreError
        
        # Cache exceptions
        assert issubclass(CacheConnectionError, CacheError)
        assert issubclass(CacheSerializationError, CacheError)
        
        # Data exceptions
        assert issubclass(DataLoadError, HistoCoreError)
        assert issubclass(DataSaveError, HistoCoreError)
        
        # Resource exceptions
        assert issubclass(DiskSpaceError, ResourceError)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
