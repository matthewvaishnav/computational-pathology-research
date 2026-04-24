"""Unit tests for WSI pipeline configuration."""

import pytest
import tempfile
from pathlib import Path
from src.data.wsi_pipeline.config import ProcessingConfig


class TestProcessingConfigDefaults:
    """Tests for ProcessingConfig default values."""

    def test_default_config(self):
        """Test creating ProcessingConfig with default values."""
        config = ProcessingConfig()

        # Patch extraction defaults
        assert config.patch_size == 256
        assert config.stride is None
        assert config.level == 0
        assert config.target_mpp is None

        # Tissue detection defaults
        assert config.tissue_method == "otsu"
        assert config.tissue_threshold == 0.5

        # Feature extraction defaults
        assert config.encoder_name == "resnet50"
        assert config.encoder_pretrained is True
        assert config.batch_size == 32

        # Caching defaults
        assert config.cache_dir == "features"
        assert config.compression == "gzip"

        # Batch processing defaults
        assert config.num_workers == 4
        assert config.gpu_ids is None
        assert config.max_retries == 3

        # Quality control defaults
        assert config.blur_threshold == 100.0
        assert config.min_tissue_coverage == 0.1

    def test_default_config_validates(self):
        """Test that default configuration passes validation."""
        config = ProcessingConfig()
        config.validate()  # Should not raise


class TestProcessingConfigValidation:
    """Tests for ProcessingConfig validation."""

    def test_valid_patch_size(self):
        """Test valid patch_size values."""
        for size in [64, 128, 256, 512, 1024, 2048]:
            config = ProcessingConfig(patch_size=size)
            config.validate()  # Should not raise

    def test_invalid_patch_size_too_small(self):
        """Test patch_size below minimum."""
        config = ProcessingConfig(patch_size=32)
        with pytest.raises(ValueError, match="patch_size must be between 64 and 2048"):
            config.validate()

    def test_invalid_patch_size_too_large(self):
        """Test patch_size above maximum."""
        config = ProcessingConfig(patch_size=4096)
        with pytest.raises(ValueError, match="patch_size must be between 64 and 2048"):
            config.validate()

    def test_valid_tissue_threshold(self):
        """Test valid tissue_threshold values."""
        for threshold in [0.0, 0.25, 0.5, 0.75, 1.0]:
            config = ProcessingConfig(tissue_threshold=threshold)
            config.validate()  # Should not raise

    def test_invalid_tissue_threshold_negative(self):
        """Test negative tissue_threshold."""
        config = ProcessingConfig(tissue_threshold=-0.1)
        with pytest.raises(ValueError, match="tissue_threshold must be between 0.0 and 1.0"):
            config.validate()

    def test_invalid_tissue_threshold_too_large(self):
        """Test tissue_threshold above 1.0."""
        config = ProcessingConfig(tissue_threshold=1.5)
        with pytest.raises(ValueError, match="tissue_threshold must be between 0.0 and 1.0"):
            config.validate()

    def test_valid_num_workers(self):
        """Test valid num_workers values."""
        for workers in [1, 4, 8, 16]:
            config = ProcessingConfig(num_workers=workers)
            config.validate()  # Should not raise

    def test_invalid_num_workers_zero(self):
        """Test num_workers of zero."""
        config = ProcessingConfig(num_workers=0)
        with pytest.raises(ValueError, match="num_workers must be between 1 and 16"):
            config.validate()

    def test_invalid_num_workers_too_large(self):
        """Test num_workers above maximum."""
        config = ProcessingConfig(num_workers=32)
        with pytest.raises(ValueError, match="num_workers must be between 1 and 16"):
            config.validate()

    def test_valid_max_retries(self):
        """Test valid max_retries values."""
        for retries in [0, 1, 3, 5]:
            config = ProcessingConfig(max_retries=retries)
            config.validate()  # Should not raise

    def test_invalid_max_retries_negative(self):
        """Test negative max_retries."""
        config = ProcessingConfig(max_retries=-1)
        with pytest.raises(ValueError, match="max_retries must be between 0 and 5"):
            config.validate()

    def test_invalid_max_retries_too_large(self):
        """Test max_retries above maximum."""
        config = ProcessingConfig(max_retries=10)
        with pytest.raises(ValueError, match="max_retries must be between 0 and 5"):
            config.validate()

    def test_valid_batch_size(self):
        """Test valid batch_size values."""
        for size in [1, 16, 32, 64, 128, 256, 512, 1024]:
            config = ProcessingConfig(batch_size=size)
            config.validate()  # Should not raise

    def test_invalid_batch_size_zero(self):
        """Test batch_size of zero."""
        config = ProcessingConfig(batch_size=0)
        with pytest.raises(ValueError, match="batch_size must be between 1 and 1024"):
            config.validate()

    def test_invalid_batch_size_too_large(self):
        """Test batch_size above maximum."""
        config = ProcessingConfig(batch_size=2048)
        with pytest.raises(ValueError, match="batch_size must be between 1 and 1024"):
            config.validate()

    def test_valid_stride(self):
        """Test valid stride values."""
        for stride in [1, 64, 128, 256]:
            config = ProcessingConfig(stride=stride)
            config.validate()  # Should not raise

    def test_invalid_stride_zero(self):
        """Test stride of zero."""
        config = ProcessingConfig(stride=0)
        with pytest.raises(ValueError, match="stride must be at least 1"):
            config.validate()

    def test_invalid_stride_negative(self):
        """Test negative stride."""
        config = ProcessingConfig(stride=-10)
        with pytest.raises(ValueError, match="stride must be at least 1"):
            config.validate()

    def test_valid_level(self):
        """Test valid level values."""
        for level in [0, 1, 2, 3]:
            config = ProcessingConfig(level=level)
            config.validate()  # Should not raise

    def test_invalid_level_negative(self):
        """Test negative level."""
        config = ProcessingConfig(level=-1)
        with pytest.raises(ValueError, match="level must be non-negative"):
            config.validate()

    def test_valid_target_mpp(self):
        """Test valid target_mpp values."""
        for mpp in [0.25, 0.5, 1.0, 2.0]:
            config = ProcessingConfig(target_mpp=mpp)
            config.validate()  # Should not raise

    def test_invalid_target_mpp_zero(self):
        """Test target_mpp of zero."""
        config = ProcessingConfig(target_mpp=0.0)
        with pytest.raises(ValueError, match="target_mpp must be positive"):
            config.validate()

    def test_invalid_target_mpp_negative(self):
        """Test negative target_mpp."""
        config = ProcessingConfig(target_mpp=-0.5)
        with pytest.raises(ValueError, match="target_mpp must be positive"):
            config.validate()

    def test_valid_tissue_method(self):
        """Test valid tissue_method values."""
        for method in ["otsu", "deep_learning", "hybrid"]:
            config = ProcessingConfig(tissue_method=method)
            config.validate()  # Should not raise

    def test_invalid_tissue_method(self):
        """Test invalid tissue_method."""
        config = ProcessingConfig(tissue_method="invalid_method")
        with pytest.raises(ValueError, match="tissue_method must be one of"):
            config.validate()

    def test_valid_blur_threshold(self):
        """Test valid blur_threshold values."""
        for threshold in [0.0, 50.0, 100.0, 200.0]:
            config = ProcessingConfig(blur_threshold=threshold)
            config.validate()  # Should not raise

    def test_invalid_blur_threshold_negative(self):
        """Test negative blur_threshold."""
        config = ProcessingConfig(blur_threshold=-10.0)
        with pytest.raises(ValueError, match="blur_threshold must be non-negative"):
            config.validate()

    def test_valid_min_tissue_coverage(self):
        """Test valid min_tissue_coverage values."""
        for coverage in [0.0, 0.1, 0.5, 1.0]:
            config = ProcessingConfig(min_tissue_coverage=coverage)
            config.validate()  # Should not raise

    def test_invalid_min_tissue_coverage_negative(self):
        """Test negative min_tissue_coverage."""
        config = ProcessingConfig(min_tissue_coverage=-0.1)
        with pytest.raises(ValueError, match="min_tissue_coverage must be between 0.0 and 1.0"):
            config.validate()

    def test_invalid_min_tissue_coverage_too_large(self):
        """Test min_tissue_coverage above 1.0."""
        config = ProcessingConfig(min_tissue_coverage=1.5)
        with pytest.raises(ValueError, match="min_tissue_coverage must be between 0.0 and 1.0"):
            config.validate()

    def test_valid_compression(self):
        """Test valid compression values."""
        for compression in ["gzip", "lzf", None, "none"]:
            config = ProcessingConfig(compression=compression)
            config.validate()  # Should not raise

    def test_invalid_compression(self):
        """Test invalid compression."""
        config = ProcessingConfig(compression="invalid")
        with pytest.raises(ValueError, match="compression must be one of"):
            config.validate()

    def test_multiple_validation_errors(self):
        """Test that multiple validation errors are reported together."""
        config = ProcessingConfig(
            patch_size=32,  # Too small
            tissue_threshold=1.5,  # Too large
            num_workers=0,  # Too small
        )

        with pytest.raises(ValueError) as exc_info:
            config.validate()

        error_message = str(exc_info.value)
        assert "patch_size" in error_message
        assert "tissue_threshold" in error_message
        assert "num_workers" in error_message


class TestProcessingConfigFromDict:
    """Tests for creating ProcessingConfig from dictionary."""

    def test_from_dict_minimal(self):
        """Test creating config from minimal dictionary."""
        config_dict = {"patch_size": 512, "encoder_name": "densenet121"}

        config = ProcessingConfig.from_dict(config_dict)

        assert config.patch_size == 512
        assert config.encoder_name == "densenet121"
        # Other fields should have defaults
        assert config.tissue_threshold == 0.5
        assert config.num_workers == 4

    def test_from_dict_complete(self):
        """Test creating config from complete dictionary."""
        config_dict = {
            "patch_size": 256,
            "stride": 128,
            "level": 1,
            "target_mpp": 0.5,
            "tissue_method": "hybrid",
            "tissue_threshold": 0.6,
            "encoder_name": "efficientnet_b0",
            "encoder_pretrained": True,
            "batch_size": 64,
            "cache_dir": "output/features",
            "compression": "lzf",
            "num_workers": 8,
            "gpu_ids": [0, 1],
            "max_retries": 5,
            "blur_threshold": 150.0,
            "min_tissue_coverage": 0.15,
        }

        config = ProcessingConfig.from_dict(config_dict)

        assert config.patch_size == 256
        assert config.stride == 128
        assert config.level == 1
        assert config.target_mpp == 0.5
        assert config.tissue_method == "hybrid"
        assert config.tissue_threshold == 0.6
        assert config.encoder_name == "efficientnet_b0"
        assert config.encoder_pretrained is True
        assert config.batch_size == 64
        assert config.cache_dir == "output/features"
        assert config.compression == "lzf"
        assert config.num_workers == 8
        assert config.gpu_ids == [0, 1]
        assert config.max_retries == 5
        assert config.blur_threshold == 150.0
        assert config.min_tissue_coverage == 0.15

    def test_from_dict_invalid_config(self):
        """Test that from_dict validates configuration."""
        config_dict = {"patch_size": 32, "num_workers": 0}  # Invalid  # Invalid

        with pytest.raises(ValueError):
            ProcessingConfig.from_dict(config_dict)


class TestProcessingConfigFromYAML:
    """Tests for loading ProcessingConfig from YAML file."""

    def test_from_yaml_flat_structure(self):
        """Test loading config from YAML with flat structure."""
        yaml_content = """
patch_size: 512
stride: 256
level: 1
tissue_threshold: 0.6
encoder_name: densenet121
batch_size: 64
num_workers: 8
max_retries: 5
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name

        try:
            config = ProcessingConfig.from_yaml(yaml_path)

            assert config.patch_size == 512
            assert config.stride == 256
            assert config.level == 1
            assert config.tissue_threshold == 0.6
            assert config.encoder_name == "densenet121"
            assert config.batch_size == 64
            assert config.num_workers == 8
            assert config.max_retries == 5
        finally:
            Path(yaml_path).unlink()

    def test_from_yaml_nested_structure(self):
        """Test loading config from YAML with nested structure."""
        yaml_content = """
patch_extraction:
  patch_size: 256
  stride: 128
  level: 0

tissue_detection:
  tissue_method: otsu
  tissue_threshold: 0.5

feature_extraction:
  encoder_name: resnet50
  batch_size: 32

batch_processing:
  num_workers: 4
  max_retries: 3
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name

        try:
            config = ProcessingConfig.from_yaml(yaml_path)

            assert config.patch_size == 256
            assert config.stride == 128
            assert config.level == 0
            assert config.tissue_method == "otsu"
            assert config.tissue_threshold == 0.5
            assert config.encoder_name == "resnet50"
            assert config.batch_size == 32
            assert config.num_workers == 4
            assert config.max_retries == 3
        finally:
            Path(yaml_path).unlink()

    def test_from_yaml_file_not_found(self):
        """Test loading config from non-existent YAML file."""
        with pytest.raises(FileNotFoundError):
            ProcessingConfig.from_yaml("nonexistent.yaml")

    def test_from_yaml_invalid_config(self):
        """Test loading invalid config from YAML."""
        yaml_content = """
patch_size: 32
num_workers: 0
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name

        try:
            with pytest.raises(ValueError):
                ProcessingConfig.from_yaml(yaml_path)
        finally:
            Path(yaml_path).unlink()

    def test_from_yaml_with_gpu_ids(self):
        """Test loading config with GPU IDs from YAML."""
        yaml_content = """
patch_size: 256
num_workers: 4
gpu_ids: [0, 1, 2]
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name

        try:
            config = ProcessingConfig.from_yaml(yaml_path)

            assert config.gpu_ids == [0, 1, 2]
        finally:
            Path(yaml_path).unlink()

    def test_from_yaml_with_null_values(self):
        """Test loading config with null values from YAML."""
        yaml_content = """
patch_size: 256
stride: null
target_mpp: null
gpu_ids: null
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write(yaml_content)
            yaml_path = f.name

        try:
            config = ProcessingConfig.from_yaml(yaml_path)

            assert config.patch_size == 256
            assert config.stride is None
            assert config.target_mpp is None
            assert config.gpu_ids is None
        finally:
            Path(yaml_path).unlink()
