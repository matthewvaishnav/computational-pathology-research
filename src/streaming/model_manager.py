"""
Model Management System for HistoCore Real-Time WSI Streaming.

Provides model versioning, hot-swapping, A/B testing, and rollback capabilities
for zero-downtime model updates in production.

Task 9.1.1: Model versioning and hot-swapping
"""

import hashlib
import json
import logging
import os
import shutil
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from .metrics import record_processing_time, timed_operation
from src.utils.safe_threading import TimeoutLock

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model deployment status."""

    PENDING = "pending"
    ACTIVE = "active"
    TESTING = "testing"
    DEPRECATED = "deprecated"
    FAILED = "failed"


@dataclass
class ModelMetadata:
    """Metadata for a model version."""

    model_id: str
    version: str
    name: str
    description: str
    created_at: str
    model_path: str
    config_path: Optional[str] = None

    # Model architecture info
    architecture: str = "unknown"
    input_shape: Optional[Tuple[int, ...]] = None
    output_shape: Optional[Tuple[int, ...]] = None
    num_parameters: int = 0

    # Performance metrics
    accuracy: Optional[float] = None
    inference_time_ms: Optional[float] = None
    memory_usage_mb: Optional[float] = None

    # Deployment info
    status: str = ModelStatus.PENDING.value
    deployed_at: Optional[str] = None
    deprecated_at: Optional[str] = None

    # Compatibility
    min_framework_version: str = "2.0.0"
    required_features: List[str] = None

    # Checksum for integrity
    checksum: Optional[str] = None

    def __post_init__(self):
        if self.required_features is None:
            self.required_features = []

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ModelMetadata":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ModelCompatibilityCheck:
    """Result of model compatibility validation."""

    compatible: bool
    issues: List[str]
    warnings: List[str]

    def is_safe_to_deploy(self) -> bool:
        """Check if model is safe to deploy."""
        return self.compatible and len(self.issues) == 0


class ModelRegistry:
    """Registry for managing model versions."""

    def __init__(self, registry_dir: str = "./models/registry"):
        """Initialize model registry."""
        self.registry_dir = Path(registry_dir)
        self.registry_dir.mkdir(parents=True, exist_ok=True)

        self.registry_file = self.registry_dir / "registry.json"
        self.models: Dict[str, ModelMetadata] = {}

        self._load_registry()

        logger.info(f"Model registry initialized: {len(self.models)} models registered")

    def _load_registry(self):
        """Load registry from disk."""
        if self.registry_file.exists():
            try:
                with open(self.registry_file, "r") as f:
                    data = json.load(f)

                for model_id, model_data in data.items():
                    self.models[model_id] = ModelMetadata.from_dict(model_data)

                logger.info(f"Loaded {len(self.models)} models from registry")

            except Exception as e:
                logger.error(f"Failed to load registry: {e}")
                self.models = {}
        else:
            logger.info("No existing registry found, starting fresh")

    def _save_registry(self):
        """Save registry to disk."""
        try:
            data = {model_id: model.to_dict() for model_id, model in self.models.items()}

            with open(self.registry_file, "w") as f:
                json.dump(data, f, indent=2)

            logger.debug("Registry saved to disk")

        except Exception as e:
            logger.error(f"Failed to save registry: {e}")

    def register_model(
        self,
        model: nn.Module,
        model_path: str,
        version: str,
        name: str,
        description: str = "",
        config: Optional[Dict[str, Any]] = None,
    ) -> ModelMetadata:
        """Register a new model version."""
        # Generate model ID
        model_id = f"{name}_{version}_{int(time.time())}"

        # Calculate checksum
        checksum = self._calculate_checksum(model_path)

        # Get model info
        num_params = sum(p.numel() for p in model.parameters())

        # Create metadata
        metadata = ModelMetadata(
            model_id=model_id,
            version=version,
            name=name,
            description=description,
            created_at=datetime.now().isoformat(),
            model_path=model_path,
            architecture=type(model).__name__,
            num_parameters=num_params,
            checksum=checksum,
            status=ModelStatus.PENDING.value,
        )

        # Save config if provided
        if config:
            config_path = str(self.registry_dir / f"{model_id}_config.json")
            with open(config_path, "w") as f:
                json.dump(config, f, indent=2)
            metadata.config_path = config_path

        # Register
        self.models[model_id] = metadata
        self._save_registry()

        logger.info(f"Registered model: {model_id} (version={version}, params={num_params:,})")

        return metadata

    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum of model file."""
        sha256 = hashlib.sha256()

        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)

        return sha256.hexdigest()

    def get_model(self, model_id: str) -> Optional[ModelMetadata]:
        """Get model metadata by ID."""
        return self.models.get(model_id)

    def list_models(
        self, status: Optional[ModelStatus] = None, name: Optional[str] = None
    ) -> List[ModelMetadata]:
        """List models with optional filtering."""
        models = list(self.models.values())

        if status:
            models = [m for m in models if m.status == status.value]

        if name:
            models = [m for m in models if m.name == name]

        return sorted(models, key=lambda m: m.created_at, reverse=True)

    def update_status(self, model_id: str, status: ModelStatus):
        """Update model status."""
        if model_id in self.models:
            self.models[model_id].status = status.value

            if status == ModelStatus.ACTIVE:
                self.models[model_id].deployed_at = datetime.now().isoformat()
            elif status == ModelStatus.DEPRECATED:
                self.models[model_id].deprecated_at = datetime.now().isoformat()

            self._save_registry()
            logger.info(f"Updated model {model_id} status to {status.value}")

    def get_active_model(self, name: str) -> Optional[ModelMetadata]:
        """Get currently active model for a given name."""
        active_models = [
            m
            for m in self.models.values()
            if m.name == name and m.status == ModelStatus.ACTIVE.value
        ]

        if active_models:
            # Return most recently deployed
            return max(active_models, key=lambda m: m.deployed_at or "")

        return None


class ModelValidator:
    """Validator for model compatibility and safety checks."""

    def __init__(self):
        """Initialize model validator."""
        self.required_methods = ["forward"]
        self.required_torch_version = "2.0.0"

    def validate_model(
        self, model: nn.Module, metadata: ModelMetadata, test_input: Optional[torch.Tensor] = None
    ) -> ModelCompatibilityCheck:
        """Validate model compatibility."""
        issues = []
        warnings = []

        # Check model has required methods
        for method in self.required_methods:
            if not hasattr(model, method):
                issues.append(f"Model missing required method: {method}")

        # Check PyTorch version
        import torch

        current_version = torch.__version__.split("+")[0]
        if self._version_less_than(current_version, self.required_torch_version):
            issues.append(
                f"PyTorch version {current_version} < required {self.required_torch_version}"
            )

        # Verify checksum
        if metadata.checksum:
            actual_checksum = self._calculate_checksum(metadata.model_path)
            if actual_checksum != metadata.checksum:
                issues.append(f"Checksum mismatch: model file may be corrupted")

        # Test inference if test input provided
        if test_input is not None:
            try:
                model.eval()
                with torch.no_grad():
                    output = model(test_input)

                # Check output shape
                if metadata.output_shape and output.shape != metadata.output_shape:
                    warnings.append(
                        f"Output shape {output.shape} != expected {metadata.output_shape}"
                    )

            except Exception as e:
                issues.append(f"Test inference failed: {e}")

        # Check model size
        model_size_mb = os.path.getsize(metadata.model_path) / (1024 * 1024)
        if model_size_mb > 1000:  # 1GB
            warnings.append(f"Large model size: {model_size_mb:.1f}MB")

        compatible = len(issues) == 0

        return ModelCompatibilityCheck(compatible=compatible, issues=issues, warnings=warnings)

    def _version_less_than(self, version1: str, version2: str) -> bool:
        """Compare version strings."""
        v1_parts = [int(x) for x in version1.split(".")]
        v2_parts = [int(x) for x in version2.split(".")]

        for v1, v2 in zip(v1_parts, v2_parts):
            if v1 < v2:
                return True
            elif v1 > v2:
                return False

        return False

    def _calculate_checksum(self, file_path: str) -> str:
        """Calculate SHA256 checksum."""
        sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256.update(chunk)
        return sha256.hexdigest()


class ModelHotSwapper:
    """Hot-swapping manager for zero-downtime model updates."""

    def __init__(
        self, registry: ModelRegistry, validator: ModelValidator, warmup_batches: int = 10
    ):
        """Initialize hot swapper."""
        self.registry = registry
        self.validator = validator
        self.warmup_batches = warmup_batches

        self.current_model: Optional[nn.Module] = None
        self.current_metadata: Optional[ModelMetadata] = None
        self.model_lock = TimeoutLock(timeout=30.0, name='model_swap_lock')

        logger.info("Model hot swapper initialized")

    @timed_operation("model_hot_swap")
    def swap_model(
        self,
        new_model: nn.Module,
        new_metadata: ModelMetadata,
        test_input: Optional[torch.Tensor] = None,
        force: bool = False,
    ) -> Tuple[bool, str]:
        """
        Swap to new model with validation.

        Returns:
            (success, message)
        """
        # Validate new model
        if not force:
            validation = self.validator.validate_model(new_model, new_metadata, test_input)

            if not validation.is_safe_to_deploy():
                error_msg = f"Model validation failed: {validation.issues}"
                logger.error(error_msg)
                return False, error_msg

            if validation.warnings:
                logger.warning(f"Model validation warnings: {validation.warnings}")

        # Warmup new model
        if test_input is not None:
            try:
                self._warmup_model(new_model, test_input)
            except Exception as e:
                error_msg = f"Model warmup failed: {e}"
                logger.error(error_msg)
                return False, error_msg

        # Perform swap with lock
        try:
            with self.model_lock:
                old_model = self.current_model
                old_metadata = self.current_metadata

                # Swap
                self.current_model = new_model
                self.current_metadata = new_metadata

                # Update registry
                self.registry.update_status(new_metadata.model_id, ModelStatus.ACTIVE)

                if old_metadata:
                    self.registry.update_status(old_metadata.model_id, ModelStatus.DEPRECATED)

                logger.info(
                    f"Model swapped: {old_metadata.model_id if old_metadata else 'None'} -> {new_metadata.model_id}"
                )
        except TimeoutError as e:
            error_msg = f"Failed to acquire model swap lock: {e}"
            logger.error(error_msg)
            return False, error_msg

        return True, f"Successfully swapped to model {new_metadata.model_id}"

    def _warmup_model(self, model: nn.Module, test_input: torch.Tensor):
        """Warmup model with test batches."""
        model.eval()

        logger.info(f"Warming up model with {self.warmup_batches} batches")

        with torch.no_grad():
            for i in range(self.warmup_batches):
                _ = model(test_input)

        logger.info("Model warmup complete")

    def get_current_model(self) -> Tuple[Optional[nn.Module], Optional[ModelMetadata]]:
        """Get current active model."""
        try:
            with self.model_lock:
                return self.current_model, self.current_metadata
        except TimeoutError as e:
            logger.error(f"Failed to acquire model lock for get_current_model: {e}")
            return None, None

    def rollback(self, target_model_id: str) -> Tuple[bool, str]:
        """Rollback to a previous model version."""
        # Get target model metadata
        target_metadata = self.registry.get_model(target_model_id)

        if not target_metadata:
            return False, f"Model {target_model_id} not found in registry"

        # Load model
        try:
            target_model = torch.load(target_metadata.model_path)

            # Swap
            success, message = self.swap_model(target_model, target_metadata, force=True)

            if success:
                logger.info(f"Rolled back to model {target_model_id}")

            return success, message

        except Exception as e:
            error_msg = f"Rollback failed: {e}"
            logger.error(error_msg)
            return False, error_msg


class ABTestingManager:
    """A/B testing manager for model comparison."""

    def __init__(self, registry: ModelRegistry, traffic_split: float = 0.5):
        """
        Initialize A/B testing manager.

        Args:
            traffic_split: Fraction of traffic for model B (0.0 to 1.0)
        """
        self.registry = registry
        self.traffic_split = traffic_split

        self.model_a: Optional[nn.Module] = None
        self.model_b: Optional[nn.Module] = None
        self.metadata_a: Optional[ModelMetadata] = None
        self.metadata_b: Optional[ModelMetadata] = None

        self.stats_a = {"requests": 0, "total_time": 0.0, "errors": 0}
        self.stats_b = {"requests": 0, "total_time": 0.0, "errors": 0}

        self.lock = TimeoutLock(timeout=30.0, name='ab_testing_lock')

        logger.info(f"A/B testing manager initialized: traffic_split={traffic_split}")

    def setup_ab_test(
        self,
        model_a: nn.Module,
        metadata_a: ModelMetadata,
        model_b: nn.Module,
        metadata_b: ModelMetadata,
    ):
        """Setup A/B test with two models."""
        try:
            with self.lock:
                self.model_a = model_a
                self.model_b = model_b
                self.metadata_a = metadata_a
                self.metadata_b = metadata_b

                # Reset stats
                self.stats_a = {"requests": 0, "total_time": 0.0, "errors": 0}
                self.stats_b = {"requests": 0, "total_time": 0.0, "errors": 0}

                # Update registry
                self.registry.update_status(metadata_a.model_id, ModelStatus.TESTING)
                self.registry.update_status(metadata_b.model_id, ModelStatus.TESTING)

            logger.info(f"A/B test setup: A={metadata_a.model_id} B={metadata_b.model_id}")
        except TimeoutError as e:
            logger.error(f"Failed to acquire A/B testing lock for setup: {e}")
            raise

    def get_model_for_request(self) -> Tuple[nn.Module, ModelMetadata, str]:
        """
        Get model for current request based on traffic split.

        Returns:
            (model, metadata, variant) where variant is 'A' or 'B'
        """
        try:
            with self.lock:
                if self.model_a is None or self.model_b is None:
                    raise RuntimeError("A/B test not setup")

                # Random assignment based on traffic split
                if np.random.random() < self.traffic_split:
                    return self.model_b, self.metadata_b, "B"
                else:
                    return self.model_a, self.metadata_a, "A"
        except TimeoutError as e:
            logger.error(f"Failed to acquire A/B testing lock for get_model_for_request: {e}")
            raise

    def record_request(self, variant: str, processing_time: float, error: bool = False):
        """Record request statistics."""
        try:
            with self.lock:
                if variant == "A":
                    stats = self.stats_a
                elif variant == "B":
                    stats = self.stats_b
                else:
                    return

                stats["requests"] += 1
                stats["total_time"] += processing_time

                if error:
                    stats["errors"] += 1
        except TimeoutError as e:
            logger.error(f"Failed to acquire A/B testing lock for record_request: {e}")

    def get_results(self) -> Dict[str, Any]:
        """Get A/B test results."""
        try:
            with self.lock:
                results = {
                    "model_a": {
                        "model_id": self.metadata_a.model_id if self.metadata_a else None,
                        "requests": self.stats_a["requests"],
                        "avg_time_ms": (
                            (self.stats_a["total_time"] / self.stats_a["requests"] * 1000)
                            if self.stats_a["requests"] > 0
                            else 0
                        ),
                        "error_rate": (
                            (self.stats_a["errors"] / self.stats_a["requests"])
                            if self.stats_a["requests"] > 0
                            else 0
                        ),
                    },
                    "model_b": {
                        "model_id": self.metadata_b.model_id if self.metadata_b else None,
                        "requests": self.stats_b["requests"],
                        "avg_time_ms": (
                            (self.stats_b["total_time"] / self.stats_b["requests"] * 1000)
                            if self.stats_b["requests"] > 0
                            else 0
                        ),
                        "error_rate": (
                            (self.stats_b["errors"] / self.stats_b["requests"])
                            if self.stats_b["requests"] > 0
                            else 0
                        ),
                    },
                }

                # Calculate winner
                if results["model_a"]["requests"] > 0 and results["model_b"]["requests"] > 0:
                    # Compare by avg time and error rate
                    a_score = results["model_a"]["avg_time_ms"] * (1 + results["model_a"]["error_rate"])
                    b_score = results["model_b"]["avg_time_ms"] * (1 + results["model_b"]["error_rate"])

                    results["winner"] = "A" if a_score < b_score else "B"
                    results["improvement_pct"] = abs((b_score - a_score) / a_score * 100)
                else:
                    results["winner"] = None
                    results["improvement_pct"] = 0

                return results
        except TimeoutError as e:
            logger.error(f"Failed to acquire A/B testing lock for get_results: {e}")
            # Return empty results on timeout
            return {
                "model_a": {"model_id": None, "requests": 0, "avg_time_ms": 0, "error_rate": 0},
                "model_b": {"model_id": None, "requests": 0, "avg_time_ms": 0, "error_rate": 0},
                "winner": None,
                "improvement_pct": 0
            }

    def finalize_test(self, promote_winner: bool = True) -> str:
        """Finalize A/B test and optionally promote winner."""
        results = self.get_results()

        winner = results.get("winner")

        if winner and promote_winner:
            if winner == "A":
                self.registry.update_status(self.metadata_a.model_id, ModelStatus.ACTIVE)
                self.registry.update_status(self.metadata_b.model_id, ModelStatus.DEPRECATED)
                promoted_model = self.metadata_a.model_id
            else:
                self.registry.update_status(self.metadata_b.model_id, ModelStatus.ACTIVE)
                self.registry.update_status(self.metadata_a.model_id, ModelStatus.DEPRECATED)
                promoted_model = self.metadata_b.model_id

            logger.info(f"A/B test finalized: Winner={winner}, Promoted={promoted_model}")
            return promoted_model
        else:
            logger.info("A/B test finalized: No winner promoted")
            return None


class ModelManager:
    """Comprehensive model management system."""

    def __init__(self, registry_dir: str = "./models/registry"):
        """Initialize model manager."""
        self.registry = ModelRegistry(registry_dir)
        self.validator = ModelValidator()
        self.hot_swapper = ModelHotSwapper(self.registry, self.validator)
        self.ab_tester = ABTestingManager(self.registry)

        logger.info("Model manager initialized")

    def register_and_deploy_model(
        self,
        model: nn.Module,
        model_path: str,
        version: str,
        name: str,
        description: str = "",
        config: Optional[Dict[str, Any]] = None,
        test_input: Optional[torch.Tensor] = None,
        auto_deploy: bool = True,
    ) -> Tuple[bool, str, Optional[ModelMetadata]]:
        """Register and optionally deploy a new model."""
        # Register model
        metadata = self.registry.register_model(
            model, model_path, version, name, description, config
        )

        # Deploy if requested
        if auto_deploy:
            success, message = self.hot_swapper.swap_model(model, metadata, test_input)
            return success, message, metadata

        return True, f"Model registered: {metadata.model_id}", metadata

    def get_current_model(self) -> Tuple[Optional[nn.Module], Optional[ModelMetadata]]:
        """Get currently active model."""
        return self.hot_swapper.get_current_model()

    def list_models(self, status: Optional[ModelStatus] = None) -> List[ModelMetadata]:
        """List all models."""
        return self.registry.list_models(status=status)

    def rollback_to_version(self, model_id: str) -> Tuple[bool, str]:
        """Rollback to a specific model version."""
        return self.hot_swapper.rollback(model_id)

    def start_ab_test(
        self, model_a_id: str, model_b_id: str, traffic_split: float = 0.5
    ) -> Tuple[bool, str]:
        """Start A/B test between two models."""
        # Load models
        metadata_a = self.registry.get_model(model_a_id)
        metadata_b = self.registry.get_model(model_b_id)

        if not metadata_a or not metadata_b:
            return False, "One or both models not found"

        try:
            model_a = torch.load(metadata_a.model_path)
            model_b = torch.load(metadata_b.model_path)

            self.ab_tester.traffic_split = traffic_split
            self.ab_tester.setup_ab_test(model_a, metadata_a, model_b, metadata_b)

            return True, f"A/B test started: {model_a_id} vs {model_b_id}"

        except Exception as e:
            return False, f"Failed to start A/B test: {e}"

    def get_ab_test_results(self) -> Dict[str, Any]:
        """Get current A/B test results."""
        return self.ab_tester.get_results()

    def finalize_ab_test(self, promote_winner: bool = True) -> str:
        """Finalize A/B test."""
        return self.ab_tester.finalize_test(promote_winner)


# Convenience functions
def create_model_manager(registry_dir: str = "./models/registry") -> ModelManager:
    """Create model manager instance."""
    return ModelManager(registry_dir)


def deploy_model(
    manager: ModelManager,
    model: nn.Module,
    model_path: str,
    version: str,
    name: str = "attention_mil",
    test_input: Optional[torch.Tensor] = None,
) -> Tuple[bool, str]:
    """Deploy a model with validation."""
    success, message, metadata = manager.register_and_deploy_model(
        model=model,
        model_path=model_path,
        version=version,
        name=name,
        test_input=test_input,
        auto_deploy=True,
    )

    return success, message
