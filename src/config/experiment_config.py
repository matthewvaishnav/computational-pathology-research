"""
Configuration Dataclasses

Type-safe configuration using dataclasses instead of dictionaries.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LoggingConfig:
    """Logging configuration."""

    log_interval: int = 10
    tensorboard: bool = True
    log_dir: str = "logs"


@dataclass
class CheckpointConfig:
    """Checkpoint configuration."""

    checkpoint_dir: str = "checkpoints"
    save_frequency: int = 1
    keep_last_n: int = 5
    stability_frequency: int = 50
    rolling_window: int = 5


@dataclass
class TrainingConfig:
    """Training configuration."""

    epochs: int = 20
    batch_size: int = 128
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    channels_last: bool = False
    mixed_precision: bool = True
    gradient_accumulation_steps: int = 1


@dataclass
class DataConfig:
    """Data configuration."""

    data_root: str = "data"
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True


@dataclass
class ModelConfig:
    """Model configuration."""

    feature_extractor: str = "resnet18"
    feature_dim: int = 512
    hidden_dim: int = 128
    num_classes: int = 2
    dropout: float = 0.3


@dataclass
class ExperimentConfig:
    """Complete experiment configuration."""

    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    seed: int = 42
    device: str = "cuda"

    @classmethod
    def from_dict(cls, config_dict: dict) -> "ExperimentConfig":
        """Create config from dictionary."""
        return cls(
            training=TrainingConfig(**config_dict.get("training", {})),
            data=DataConfig(**config_dict.get("data", {})),
            model=ModelConfig(**config_dict.get("model", {})),
            logging=LoggingConfig(**config_dict.get("logging", {})),
            checkpoint=CheckpointConfig(**config_dict.get("checkpoint", {})),
            seed=config_dict.get("seed", 42),
            device=config_dict.get("device", "cuda"),
        )
