"""Config management + validation for HistoCore streaming."""

import json
import logging
import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from pydantic import BaseModel, Field, ValidationError, validator

logger = logging.getLogger(__name__)


class LogLevel(str, Enum):
    """Log level enum."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class StreamingConfig(BaseModel):
    """Streaming pipeline config."""

    # Processing
    batch_size: int = Field(default=32, ge=1, le=256)
    tile_size: int = Field(default=224, ge=64, le=1024)
    overlap: int = Field(default=0, ge=0, le=512)
    max_memory_gb: float = Field(default=2.0, ge=0.1, le=64.0)

    # GPU
    gpu_ids: List[int] = Field(default_factory=lambda: [0])
    fp16_mode: bool = Field(default=True)
    cudnn_benchmark: bool = Field(default=True)

    # Attention
    attention_threshold: float = Field(default=0.95, ge=0.0, le=1.0)
    early_stopping: bool = Field(default=True)
    confidence_threshold: float = Field(default=0.9, ge=0.0, le=1.0)

    # Performance
    num_workers: int = Field(default=4, ge=0, le=32)
    prefetch_factor: int = Field(default=2, ge=1, le=10)
    pin_memory: bool = Field(default=True)

    @validator("gpu_ids")
    def validate_gpu_ids(cls, v):
        if not v:
            raise ValueError("At least one GPU ID required")
        return v


class PACSConfig(BaseModel):
    """PACS integration config."""

    host: str = Field(default="localhost")
    port: int = Field(default=11112, ge=1, le=65535)
    ae_title: str = Field(default="HISTOCORE")
    called_ae_title: str = Field(default="PACS")
    timeout: int = Field(default=30, ge=1, le=300)
    max_retries: int = Field(default=3, ge=0, le=10)
    tls_enabled: bool = Field(default=True)


class MonitoringConfig(BaseModel):
    """Monitoring config."""

    metrics_port: int = Field(default=9090, ge=1024, le=65535)
    health_port: int = Field(default=8080, ge=1024, le=65535)

    prometheus_enabled: bool = Field(default=True)
    jaeger_endpoint: Optional[str] = Field(default=None)
    otlp_endpoint: Optional[str] = Field(default=None)

    log_level: LogLevel = Field(default=LogLevel.INFO)
    log_format: str = Field(default="json")
    log_file: Optional[str] = Field(default=None)


class CacheConfig(BaseModel):
    """Cache config."""

    enabled: bool = Field(default=True)
    redis_url: Optional[str] = Field(default=None)
    ttl_seconds: int = Field(default=3600, ge=60, le=86400)
    max_memory_mb: int = Field(default=1024, ge=64, le=16384)


class HistoCoreConfig(BaseModel):
    """Complete HistoCore config."""

    streaming: StreamingConfig = Field(default_factory=StreamingConfig)
    pacs: PACSConfig = Field(default_factory=PACSConfig)
    monitoring: MonitoringConfig = Field(default_factory=MonitoringConfig)
    cache: CacheConfig = Field(default_factory=CacheConfig)

    # Model
    model_path: str = Field(default="models/histocore.pth")
    model_type: str = Field(default="resnet50")

    # Paths
    data_dir: str = Field(default="/data")
    output_dir: str = Field(default="/output")
    temp_dir: str = Field(default="/tmp/histocore")

    class Config:
        use_enum_values = True


class ConfigManager:
    """Config manager with validation + hot reload."""

    def __init__(self, config_path: Optional[str] = None):
        self.config_path = config_path or os.getenv("HISTOCORE_CONFIG", "config/histocore.yml")
        self.config: Optional[HistoCoreConfig] = None
        self._watchers = []

    def load(self) -> HistoCoreConfig:
        """Load + validate config."""
        try:
            config_file = Path(self.config_path)

            if not config_file.exists():
                logger.warning(f"Config not found: {self.config_path}, using defaults")
                self.config = HistoCoreConfig()
                return self.config

            with open(config_file) as f:
                if config_file.suffix in [".yml", ".yaml"]:
                    data = yaml.safe_load(f)
                elif config_file.suffix == ".json":
                    data = json.load(f)
                else:
                    raise ValueError(f"Unsupported config format: {config_file.suffix}")

            self.config = HistoCoreConfig(**data)
            logger.info(f"Config loaded: {self.config_path}")

            # Notify watchers
            for watcher in self._watchers:
                watcher(self.config)

            return self.config

        except ValidationError as e:
            logger.error(f"Config validation failed: {e}")
            raise
        except Exception as e:
            logger.error(f"Config load failed: {e}")
            raise

    def save(self, config: Optional[HistoCoreConfig] = None):
        """Save config to file."""
        if config:
            self.config = config

        if not self.config:
            raise ValueError("No config to save")

        config_file = Path(self.config_path)
        config_file.parent.mkdir(parents=True, exist_ok=True)

        data = self.config.dict()

        with open(config_file, "w") as f:
            if config_file.suffix in [".yml", ".yaml"]:
                yaml.dump(data, f, default_flow_style=False)
            elif config_file.suffix == ".json":
                json.dump(data, f, indent=2)
            else:
                raise ValueError(f"Unsupported config format: {config_file.suffix}")

        logger.info(f"Config saved: {self.config_path}")

    def get(self) -> HistoCoreConfig:
        """Get current config."""
        if not self.config:
            self.load()
        return self.config

    def update(self, updates: Dict[str, Any]):
        """Update config values."""
        if not self.config:
            self.load()

        # Apply updates
        current_data = self.config.dict()
        self._deep_update(current_data, updates)

        # Validate
        self.config = HistoCoreConfig(**current_data)

        logger.info(f"Config updated: {list(updates.keys())}")

        # Notify watchers
        for watcher in self._watchers:
            watcher(self.config)

    def _deep_update(self, base: Dict, updates: Dict):
        """Deep update dict."""
        for key, value in updates.items():
            if isinstance(value, dict) and key in base and isinstance(base[key], dict):
                self._deep_update(base[key], value)
            else:
                base[key] = value

    def watch(self, callback):
        """Register config change watcher."""
        self._watchers.append(callback)

    def validate_file(self, config_path: str) -> tuple[bool, Optional[str]]:
        """Validate config file."""
        try:
            with open(config_path) as f:
                if config_path.endswith((".yml", ".yaml")):
                    data = yaml.safe_load(f)
                elif config_path.endswith(".json"):
                    data = json.load(f)
                else:
                    return False, f"Unsupported format: {config_path}"

            HistoCoreConfig(**data)
            return True, None

        except ValidationError as e:
            return False, str(e)
        except Exception as e:
            return False, str(e)

    def export_schema(self, output_path: str):
        """Export JSON schema."""
        schema = HistoCoreConfig.schema()

        with open(output_path, "w") as f:
            json.dump(schema, f, indent=2)

        logger.info(f"Schema exported: {output_path}")

    def export_defaults(self, output_path: str):
        """Export default config."""
        default_config = HistoCoreConfig()

        with open(output_path, "w") as f:
            if output_path.endswith((".yml", ".yaml")):
                yaml.dump(default_config.dict(), f, default_flow_style=False)
            elif output_path.endswith(".json"):
                json.dump(default_config.dict(), f, indent=2)
            else:
                raise ValueError(f"Unsupported format: {output_path}")

        logger.info(f"Defaults exported: {output_path}")


# Global instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager(config_path: Optional[str] = None) -> ConfigManager:
    """Get global config manager."""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager(config_path)
    return _config_manager


def get_config() -> HistoCoreConfig:
    """Get current config."""
    return get_config_manager().get()


def reload_config():
    """Reload config from file."""
    return get_config_manager().load()


def update_config(updates: Dict[str, Any]):
    """Update config values."""
    get_config_manager().update(updates)


def watch_config(callback):
    """Watch config changes."""
    get_config_manager().watch(callback)
