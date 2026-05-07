"""Tests for OptimizerConfig class."""

import pytest
import torch

from src.streaming.memory.config import OptimizerConfig


def test_config_defaults():
    """Test default configuration values."""
    config = OptimizerConfig()

    assert config.device == torch.device("cpu")
    assert config.memory_limit_gb == 8.0
    assert config.cache_size_mb == 1000.0
    assert config.enable_monitoring is True
    assert config.enable_profiling is True
    assert config.sampling_interval_ms == 100.0
    assert config.alert_threshold_percent == 90.0
    assert config.batch_size_range == (1, 64)
    assert config.tile_size_range == (224, 512)


def test_config_custom_values():
    """Test custom configuration values."""
    config = OptimizerConfig(
        device=torch.device("cuda"),
        memory_limit_gb=16.0,
        cache_size_mb=2000.0,
        enable_monitoring=False,
        sampling_interval_ms=50.0,
    )

    assert config.device == torch.device("cuda")
    assert config.memory_limit_gb == 16.0
    assert config.cache_size_mb == 2000.0
    assert config.enable_monitoring is False
    assert config.sampling_interval_ms == 50.0


def test_config_validation_memory_limit():
    """Test validation of memory_limit_gb."""
    with pytest.raises(ValueError, match="memory_limit_gb must be positive"):
        OptimizerConfig(memory_limit_gb=0)

    with pytest.raises(ValueError, match="memory_limit_gb must be positive"):
        OptimizerConfig(memory_limit_gb=-1.0)


def test_config_validation_cache_size():
    """Test validation of cache_size_mb."""
    with pytest.raises(ValueError, match="cache_size_mb must be positive"):
        OptimizerConfig(cache_size_mb=0)

    with pytest.raises(ValueError, match="cache_size_mb must be positive"):
        OptimizerConfig(cache_size_mb=-100.0)


def test_config_validation_alert_threshold():
    """Test validation of alert_threshold_percent."""
    with pytest.raises(ValueError, match="alert_threshold_percent must be between 0 and 100"):
        OptimizerConfig(alert_threshold_percent=0)

    with pytest.raises(ValueError, match="alert_threshold_percent must be between 0 and 100"):
        OptimizerConfig(alert_threshold_percent=101)


def test_config_validation_sampling_interval():
    """Test validation of sampling_interval_ms."""
    with pytest.raises(ValueError, match="sampling_interval_ms must be positive"):
        OptimizerConfig(sampling_interval_ms=0)

    with pytest.raises(ValueError, match="sampling_interval_ms must be positive"):
        OptimizerConfig(sampling_interval_ms=-10.0)


def test_config_validation_batch_size_range():
    """Test validation of batch_size_range."""
    with pytest.raises(ValueError, match="Invalid batch_size_range"):
        OptimizerConfig(batch_size_range=(0, 64))

    with pytest.raises(ValueError, match="Invalid batch_size_range"):
        OptimizerConfig(batch_size_range=(64, 32))


def test_config_validation_tile_size_range():
    """Test validation of tile_size_range."""
    with pytest.raises(ValueError, match="Invalid tile_size_range"):
        OptimizerConfig(tile_size_range=(0, 512))

    with pytest.raises(ValueError, match="Invalid tile_size_range"):
        OptimizerConfig(tile_size_range=(512, 224))


def test_config_from_dict():
    """Test creating config from dictionary."""
    config_dict = {
        "device": "cuda",
        "memory_limit_gb": 12.0,
        "cache_size_mb": 1500.0,
        "enable_monitoring": False,
    }

    config = OptimizerConfig.from_dict(config_dict)

    assert config.device == torch.device("cuda")
    assert config.memory_limit_gb == 12.0
    assert config.cache_size_mb == 1500.0
    assert config.enable_monitoring is False


def test_config_to_dict():
    """Test converting config to dictionary."""
    config = OptimizerConfig(
        device=torch.device("cuda"),
        memory_limit_gb=10.0,
        cache_size_mb=800.0,
    )

    config_dict = config.to_dict()

    assert config_dict["device"] == "cuda"
    assert config_dict["memory_limit_gb"] == 10.0
    assert config_dict["cache_size_mb"] == 800.0
    assert "enable_monitoring" in config_dict
    assert "batch_size_range" in config_dict


def test_config_roundtrip():
    """Test config serialization roundtrip."""
    original = OptimizerConfig(
        device=torch.device("cuda"),
        memory_limit_gb=16.0,
        cache_size_mb=2000.0,
        enable_monitoring=False,
        sampling_interval_ms=75.0,
    )

    # Convert to dict and back
    config_dict = original.to_dict()
    restored = OptimizerConfig.from_dict(config_dict)

    assert restored.device == original.device
    assert restored.memory_limit_gb == original.memory_limit_gb
    assert restored.cache_size_mb == original.cache_size_mb
    assert restored.enable_monitoring == original.enable_monitoring
    assert restored.sampling_interval_ms == original.sampling_interval_ms
