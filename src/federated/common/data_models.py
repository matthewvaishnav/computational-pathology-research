"""Data models for federated learning system."""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

import torch


@dataclass
class TrainingRound:
    """Metadata for a single federated training round."""

    round_id: int
    global_model_version: int
    start_time: datetime
    end_time: Optional[datetime] = None
    participants: List[str] = field(default_factory=list)  # Client IDs
    aggregation_algorithm: str = "fedavg"  # "fedavg", "fedprox", "fedadam"
    convergence_metrics: Dict[str, float] = field(default_factory=dict)  # loss, accuracy, grad_norm
    status: str = "in_progress"  # "in_progress", "completed", "failed"


@dataclass
class ClientUpdate:
    """Update from a single client after local training."""

    client_id: str
    round_id: int
    model_version: int
    gradients: Dict[str, torch.Tensor]  # param_name -> gradient
    dataset_size: int
    training_time_seconds: float
    privacy_epsilon: float = 0.0
    is_encrypted: bool = False
    compression_method: Optional[str] = None  # "quantize_8bit", "sparsify_10pct"


@dataclass
class ModelCheckpoint:
    """Versioned global model checkpoint."""

    version: int
    round_id: int
    timestamp: datetime
    model_state_dict: Dict[str, torch.Tensor]
    optimizer_state_dict: Dict[str, Any]
    contributors: List[str] = field(default_factory=list)  # Client IDs
    metrics: Dict[str, float] = field(default_factory=dict)  # validation loss, accuracy
    provenance: Dict[str, Any] = field(default_factory=dict)  # training config, hyperparams


@dataclass
class PrivacyBudget:
    """Privacy budget tracker for differential privacy."""

    client_id: str
    total_epsilon: float
    total_delta: float
    epsilon_per_round: List[float] = field(default_factory=list)
    budget_limit: float = 1.0  # Maximum allowed epsilon
    is_exhausted: bool = False


@dataclass
class AuditLogEntry:
    """Tamper-evident audit log entry."""

    timestamp: datetime
    event_type: str  # "round_start", "update_received", "aggregation_complete"
    client_id: Optional[str]
    round_id: int
    details: Dict[str, Any] = field(default_factory=dict)
    hash: str = ""  # SHA-256 for tamper detection
