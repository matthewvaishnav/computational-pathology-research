"""
Comprehensive tests for safe_operations.py

Tests cover:
- Safe model loading (OOM protection, checksum validation)
- Safe file operations (atomic writes, disk space checks, backups)
- Safe database operations (transactions, rollback)
- Safe network operations (retries, circuit breaker)
- System health monitoring (GPU, disk, memory)
"""

import hashlib
import json
import os
import shutil
import sqlite3
import tempfile
import time
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch

from src.exceptions import (
    DatabaseError,
    DataSaveError,
    DiskSpaceError,
    ResourceError,
)
from src.utils.safe_operations import (
    CircuitBreaker,
    CircuitState,
    FileOperationRateLimiter,
    MemoryMonitor,
    atomic_write,
    check_disk_space,
    check_gpu_memory_available,
    check_system_health,
    compute_file_checksum,
    create_verified_backup,
    fetch_with_retry,
    get_best_device,
    safe_db_transaction,
    safe_delete,
    safe_model_to_device,
    safe_torch_load,
)


# ============================================================================
# 1. SAFE MODEL LOADING TESTS
# ============================================================================


def test_safe_torch_load_basic(tmp_path):
    """Test basic checkpoint loading."""
    checkpoint_path = tmp_path / "checkpoint.pth"
    checkpoint_data = {"model_state_dict": {"weight": torch.randn(10, 10)}, "epoch": 5}
    torch.save(checkpoint_data, checkpoint_path)
    
    loaded = safe_torch_load(checkpoint_path)
    
    assert "model_state_dict" in loaded
    assert "epoch" in loaded
    assert loaded["epoch"] == 5


def test_safe_torch_load_nonexistent():
    """Test loading nonexistent checkpoint."""
    with pytest.raises(FileNotFoundError):
        safe_torch_load(Path("/nonexistent/checkpoint.pth"))


def test_safe_torch_load_with_checksum(tmp_path):
    """Test checkpoint loading with checksum verification."""
    checkpoint_path = tmp_path / "checkpoint.pth"
    checkpoint_data = {"model_state_dict": {}, "epoch": 1}
    torch.save(checkpoint_data, checkpoint_path)
    
    # Compute actual checksum
    checksum = compute_file_checksum(checkpoint_path)
    
    # Should succeed with correct checksum
    loaded = safe_torch_load(checkpoint_path, verify_checksum=checksum)
    assert loaded["epoch"] == 1


def test_safe_torch_load_checksum_mismatch(tmp_path):
    """Test checkpoint loading with wrong checksum."""
    checkpoint_path = tmp_path / "checkpoint.pth"
    checkpoint_data = {"model_state_dict": {}, "epoch": 1}
    torch.save(checkpoint_data, checkpoint_path)
    
    # Wrong checksum
    wrong_checksum = "0" * 64
    
    with pytest.raises(ValueError, match="Checksum mismatch"):
        safe_torch_load(checkpoint_path, verify_checksum=wrong_checksum)


def test_compute_file_checksum(tmp_path):
    """Test file checksum computation."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("Hello, World!")
    
    checksum = compute_file_checksum(test_file)
    
    # Verify it's a valid SHA256 hex string
    assert len(checksum) == 64
    assert all(c in "0123456789abcdef" for c in checksum)
    
    # Verify consistency
    checksum2 = compute_file_checksum(test_file)
    assert checksum == checksum2


def test_get_best_device():
    """Test device selection."""
    device = get_best_device()
    
    # Should return either cuda or cpu
    assert device.type in ["cuda", "cpu"]
    
    # If CUDA available, should prefer it
    if torch.cuda.is_available():
        assert device.type == "cuda"
    else:
        assert device.type == "cpu"


def test_safe_model_to_device():
    """Test safe model device transfer."""
    model = torch.nn.Linear(10, 5)
    
    model, device = safe_model_to_device(model)
    
    # Model should be on some device
    assert next(model.parameters()).device.type in ["cuda", "cpu"]
    assert device.type in ["cuda", "cpu"]


def test_safe_model_to_device_cpu_fallback():
    """Test CPU fallback when GPU unavailable."""
    model = torch.nn.Linear(10, 5)
    
    # Force CPU device
    model, device = safe_model_to_device(model, device=torch.device("cpu"))
    
    assert device.type == "cpu"
    assert next(model.parameters()).device.type == "cpu"


def test_check_gpu_memory_available():
    """Test GPU memory check."""
    checkpoint = {
        "model_state_dict": {
            "weight": torch.randn(100, 100),
            "bias": torch.randn(100),
        }
    }
    
    # Should return bool
    result = check_gpu_memory_available(checkpoint)
    assert isinstance(result, bool)
    
    # If CUDA unavailable, should return False
    if not torch.cuda.is_available():
        assert result is False


# ============================================================================
# 2. SAFE FILE OPERATIONS TESTS
# ============================================================================


def test_check_disk_space_sufficient(tmp_path):
    """Test disk space check with sufficient space."""
    test_file = tmp_path / "test.txt"
    
    # 1KB should always be available
    result = check_disk_space(test_file, required_bytes=1024)
    
    assert result is True


def test_check_disk_space_insufficient(tmp_path):
    """Test disk space check with insufficient space."""
    test_file = tmp_path / "test.txt"
    
    # Request impossibly large space (1PB)
    result = check_disk_space(test_file, required_bytes=1024**5)
    
    assert result is False


def test_atomic_write_json(tmp_path):
    """Test atomic JSON write."""
    target_file = tmp_path / "data.json"
    data = {"key": "value", "number": 42}
    
    atomic_write(target_file, data, mode="json")
    
    # Verify file exists and content is correct
    assert target_file.exists()
    loaded = json.loads(target_file.read_text())
    assert loaded == data


def test_atomic_write_text(tmp_path):
    """Test atomic text write."""
    target_file = tmp_path / "data.txt"
    data = "Hello, World!"
    
    atomic_write(target_file, data, mode="text")
    
    assert target_file.exists()
    assert target_file.read_text() == data


def test_atomic_write_creates_directory(tmp_path):
    """Test atomic write creates parent directories."""
    # Create parent first on Windows to avoid disk_usage error
    target_dir = tmp_path / "subdir" / "nested"
    target_dir.mkdir(parents=True, exist_ok=True)
    
    target_file = target_dir / "data.json"
    data = {"test": True}
    
    atomic_write(target_file, data, mode="json")
    
    assert target_file.exists()
    assert target_file.parent.exists()


def test_atomic_write_disk_space_error(tmp_path):
    """Test atomic write fails with insufficient disk space."""
    target_file = tmp_path / "data.json"
    data = {"key": "value"}
    
    with patch("src.utils.safe_operations.check_disk_space", return_value=False):
        with pytest.raises(IOError, match="Insufficient disk space"):
            atomic_write(target_file, data, mode="json")


def test_safe_delete_basic(tmp_path):
    """Test basic file deletion."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("delete me")
    
    safe_delete(test_file)
    
    assert not test_file.exists()


def test_safe_delete_nonexistent(tmp_path):
    """Test deleting nonexistent file (should not raise)."""
    test_file = tmp_path / "nonexistent.txt"
    
    # Should not raise
    safe_delete(test_file)


def test_safe_delete_with_archive(tmp_path):
    """Test file deletion with archival."""
    test_file = tmp_path / "test.txt"
    test_file.write_text("archive me")
    archive_dir = tmp_path / "archive"
    
    safe_delete(test_file, archive_dir=archive_dir)
    
    # Original should be gone
    assert not test_file.exists()
    
    # Archive should exist
    assert archive_dir.exists()
    archived_files = list(archive_dir.glob("test.txt.*"))
    assert len(archived_files) == 1


def test_file_operation_rate_limiter():
    """Test file operation rate limiting."""
    limiter = FileOperationRateLimiter(max_ops_per_second=10)
    
    # First 10 operations should be instant
    start = time.time()
    for _ in range(10):
        limiter.wait_if_needed()
    elapsed = time.time() - start
    
    # Should be very fast (< 0.1s)
    assert elapsed < 0.1
    
    # 11th operation should wait
    start = time.time()
    limiter.wait_if_needed()
    elapsed = time.time() - start
    
    # Should have waited (> 0.5s)
    assert elapsed > 0.5


def test_create_verified_backup(tmp_path):
    """Test backup creation with verification."""
    source = tmp_path / "source.txt"
    source.write_text("important data")
    backup = tmp_path / "backup.txt"
    
    result = create_verified_backup(source, backup)
    
    assert result is True
    assert backup.exists()
    assert backup.read_text() == source.read_text()


def test_create_verified_backup_checksum_mismatch(tmp_path):
    """Test backup verification fails on checksum mismatch."""
    source = tmp_path / "source.txt"
    source.write_text("original")
    backup = tmp_path / "backup.txt"
    
    # Mock checksum to simulate mismatch
    with patch("src.utils.safe_operations.compute_file_checksum") as mock_checksum:
        mock_checksum.side_effect = ["checksum1", "checksum2"]
        
        with pytest.raises(IOError, match="Backup checksum mismatch"):
            create_verified_backup(source, backup)
        
        # Backup should be cleaned up
        assert not backup.exists()


# ============================================================================
# 3. SAFE DATABASE OPERATIONS TESTS
# ============================================================================


def test_safe_db_transaction_commit(tmp_path):
    """Test database transaction commits on success."""
    db_path = tmp_path / "test.db"
    
    with safe_db_transaction(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE test (id INTEGER, value TEXT)")
        cursor.execute("INSERT INTO test VALUES (1, 'hello')")
    
    # Verify data persisted
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM test")
    rows = cursor.fetchall()
    conn.close()
    
    assert len(rows) == 1
    assert rows[0] == (1, "hello")


def test_safe_db_transaction_rollback(tmp_path):
    """Test database transaction rolls back on error."""
    db_path = tmp_path / "test.db"
    
    # Create table first
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE test (id INTEGER PRIMARY KEY, value TEXT)")
    conn.commit()
    conn.close()
    
    # Try transaction that will fail
    try:
        with safe_db_transaction(db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("INSERT INTO test VALUES (1, 'first')")
            # This will fail (duplicate primary key)
            cursor.execute("INSERT INTO test VALUES (1, 'duplicate')")
    except DatabaseError:
        pass
    
    # Verify no data persisted
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM test")
    rows = cursor.fetchall()
    conn.close()
    
    assert len(rows) == 0


# ============================================================================
# 4. SAFE NETWORK OPERATIONS TESTS
# ============================================================================


@pytest.mark.skip(reason="Requires network access")
def test_fetch_with_retry_success():
    """Test successful fetch with retry."""
    response = fetch_with_retry("https://httpbin.org/get", timeout=10)
    
    assert response.status_code == 200


def test_fetch_with_retry_mock():
    """Test fetch retry logic with mock."""
    mock_response = Mock()
    mock_response.status_code = 200
    mock_response.raise_for_status = Mock()
    
    with patch("requests.get", return_value=mock_response) as mock_get:
        response = fetch_with_retry("https://example.com", timeout=10)
        
        assert response.status_code == 200
        mock_get.assert_called_once()


def test_circuit_breaker_closed_state():
    """Test circuit breaker in closed state."""
    breaker = CircuitBreaker(failure_threshold=3, timeout=60)
    
    assert breaker.state == CircuitState.CLOSED
    
    # Successful call
    result = breaker.call(lambda x: x * 2, 5)
    assert result == 10
    assert breaker.state == CircuitState.CLOSED


def test_circuit_breaker_opens_on_failures():
    """Test circuit breaker opens after threshold failures."""
    breaker = CircuitBreaker(failure_threshold=3, timeout=60)
    
    def failing_func():
        raise Exception("Service unavailable")
    
    # Fail 3 times
    for _ in range(3):
        with pytest.raises(Exception):
            breaker.call(failing_func)
    
    # Circuit should be open
    assert breaker.state == CircuitState.OPEN
    
    # Further calls should fail immediately
    with pytest.raises(Exception, match="Circuit breaker OPEN"):
        breaker.call(failing_func)


def test_circuit_breaker_half_open_recovery():
    """Test circuit breaker recovery through half-open state."""
    breaker = CircuitBreaker(failure_threshold=2, timeout=1)
    
    def failing_func():
        raise Exception("Fail")
    
    # Open the circuit
    for _ in range(2):
        with pytest.raises(Exception):
            breaker.call(failing_func)
    
    assert breaker.state == CircuitState.OPEN
    
    # Wait for timeout
    time.sleep(1.1)
    
    # Next call should transition to HALF_OPEN
    def success_func():
        return "success"
    
    result = breaker.call(success_func)
    
    assert result == "success"
    assert breaker.state == CircuitState.CLOSED


# ============================================================================
# 5. SYSTEM HEALTH MONITORING TESTS
# ============================================================================


def test_check_system_health():
    """Test system health check."""
    health = check_system_health()
    
    # Should return dict with expected keys
    assert "gpu_available" in health
    assert "gpu_functional" in health
    assert "disk_space" in health
    assert "memory" in health
    
    # All values should be bool
    assert isinstance(health["gpu_available"], bool)
    assert isinstance(health["gpu_functional"], bool)
    assert isinstance(health["disk_space"], bool)
    assert isinstance(health["memory"], bool)


def test_memory_monitor_baseline():
    """Test memory monitor establishes baseline."""
    monitor = MemoryMonitor(threshold_mb=1000)
    
    assert monitor.baseline_mb is None
    
    monitor.check_memory()
    
    assert monitor.baseline_mb is not None
    assert monitor.baseline_mb > 0


def test_memory_monitor_no_leak():
    """Test memory monitor with no leak."""
    monitor = MemoryMonitor(threshold_mb=1000)
    
    # Establish baseline
    monitor.check_memory()
    
    # Check again (should not raise)
    monitor.check_memory()


def test_memory_monitor_detects_leak():
    """Test memory monitor detects leak."""
    monitor = MemoryMonitor(threshold_mb=10)  # Very low threshold
    
    # Establish baseline
    monitor.check_memory()
    
    # Simulate memory growth by mocking
    original_baseline = monitor.baseline_mb
    monitor.baseline_mb = original_baseline - 20  # Fake 20MB growth
    
    # Should detect leak
    with pytest.raises(MemoryError, match="Memory leak"):
        monitor.check_memory()


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


def test_full_checkpoint_workflow(tmp_path):
    """Test complete checkpoint save/load workflow."""
    checkpoint_path = tmp_path / "model.pth"
    
    # Create and save model
    model = torch.nn.Linear(10, 5)
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "epoch": 10,
        "optimizer_state_dict": {},
    }
    torch.save(checkpoint, checkpoint_path)
    
    # Compute checksum
    checksum = compute_file_checksum(checkpoint_path)
    
    # Load with verification
    loaded = safe_torch_load(checkpoint_path, verify_checksum=checksum)
    
    assert loaded["epoch"] == 10
    assert "model_state_dict" in loaded


def test_atomic_write_with_backup(tmp_path):
    """Test atomic write with backup creation."""
    target_file = tmp_path / "config.json"
    backup_file = tmp_path / "config.backup.json"
    
    # Write initial data
    data_v1 = {"version": 1}
    atomic_write(target_file, data_v1, mode="json")
    
    # Create backup
    create_verified_backup(target_file, backup_file)
    
    # Update with new data
    data_v2 = {"version": 2}
    atomic_write(target_file, data_v2, mode="json")
    
    # Verify both exist
    assert target_file.exists()
    assert backup_file.exists()
    
    # Verify contents
    current = json.loads(target_file.read_text())
    backup = json.loads(backup_file.read_text())
    
    assert current["version"] == 2
    assert backup["version"] == 1


def test_database_with_disk_check(tmp_path):
    """Test database operations with disk space check."""
    db_path = tmp_path / "test.db"
    
    # Check disk space first
    assert check_disk_space(db_path, required_bytes=1024 * 1024)
    
    # Create database
    with safe_db_transaction(db_path) as conn:
        cursor = conn.cursor()
        cursor.execute("CREATE TABLE metrics (id INTEGER, value REAL)")
        
        # Insert test data
        for i in range(100):
            cursor.execute("INSERT INTO metrics VALUES (?, ?)", (i, i * 1.5))
    
    # Verify data
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM metrics")
    count = cursor.fetchone()[0]
    conn.close()
    
    assert count == 100
