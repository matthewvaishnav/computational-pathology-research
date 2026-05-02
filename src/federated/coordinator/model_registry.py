"""Model registry for federated learning with versioning and provenance tracking."""

import hashlib
import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn

from src.federated.common.data_models import ModelCheckpoint

logger = logging.getLogger(__name__)


class ModelRegistry:
    """
    Centralized model checkpoint management with versioning and provenance.

    Responsibilities:
        - Save model checkpoints with metadata
        - Load model checkpoints by version
        - Maintain version index for discovery
        - Track provenance (contributors, algorithms, hyperparameters)
        - Support rollback to previous versions
        - Enforce retention policies

    Correctness Properties:
        - Invariant: Model version numbers are strictly increasing
        - Invariant: Each version has associated metadata (timestamp, participants, metrics)
        - Round-trip: Save model → load model → verify produces identical parameters
        - Metamorphic: Rollback to version N then train produces different version N+1 than original
    """

    def __init__(
        self,
        storage_path: str = "./fl_model_registry",
        retention_policy: int = 10,
        enable_compression: bool = False,
        enable_integrity_checks: bool = True,
    ):
        """
        Initialize model registry.

        Args:
            storage_path: Directory for storing model checkpoints
            retention_policy: Maximum number of versions to keep (0 = unlimited)
            enable_compression: Enable checkpoint compression to save disk space
            enable_integrity_checks: Enable SHA-256 checksums for integrity verification
        """
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        self.retention_policy = retention_policy
        self.enable_compression = enable_compression
        self.enable_integrity_checks = enable_integrity_checks
        
        # Version tracking
        self.version_index_path = self.storage_path / "version_index.json"
        self.version_index = self._load_version_index()
        
        # Provenance tracking
        self.provenance_path = self.storage_path / "provenance.json"
        self.provenance_db = self._load_provenance_db()
        
        logger.info(
            f"ModelRegistry initialized at {self.storage_path} "
            f"(retention={retention_policy}, compression={enable_compression})"
        )

    def save_checkpoint(
        self,
        checkpoint: ModelCheckpoint,
        tags: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Save model checkpoint with metadata and provenance.

        Implements:
            - 9.1 Checkpoint saving
            - 9.3 Version indexing
            - 9.4 Provenance tracking

        Args:
            checkpoint: Model checkpoint to save
            tags: Optional tags for categorization (e.g., {"type": "best", "dataset": "pcam"})

        Returns:
            checkpoint_path: Path to saved checkpoint

        Raises:
            ValueError: If checkpoint version already exists
        """
        # Validate version uniqueness
        if self._version_exists(checkpoint.version):
            raise ValueError(f"Checkpoint version {checkpoint.version} already exists")
        
        # Construct checkpoint path
        checkpoint_path = self._get_checkpoint_path(checkpoint.version)
        
        # Prepare checkpoint data
        checkpoint_data = {
            "version": checkpoint.version,
            "round_id": checkpoint.round_id,
            "timestamp": checkpoint.timestamp.isoformat(),
            "model_state_dict": checkpoint.model_state_dict,
            "optimizer_state_dict": checkpoint.optimizer_state_dict,
            "contributors": checkpoint.contributors,
            "metrics": checkpoint.metrics,
            "provenance": checkpoint.provenance,
            "tags": tags or {},
        }
        
        # Save checkpoint
        torch.save(checkpoint_data, checkpoint_path)
        
        # Compute integrity checksum if enabled
        checksum = None
        if self.enable_integrity_checks:
            checksum = self._compute_checksum(checkpoint_path)
        
        # Update version index
        self._add_to_version_index(checkpoint, checkpoint_path, checksum, tags)
        
        # Update provenance database
        self._add_to_provenance_db(checkpoint)
        
        # Apply retention policy
        if self.retention_policy > 0:
            self._apply_retention_policy()
        
        logger.info(
            f"Checkpoint saved: v{checkpoint.version} at {checkpoint_path} "
            f"(contributors={len(checkpoint.contributors)}, "
            f"metrics={list(checkpoint.metrics.keys())})"
        )
        
        return str(checkpoint_path)

    def load_checkpoint(
        self,
        version: int,
        verify_integrity: bool = True,
    ) -> ModelCheckpoint:
        """
        Load model checkpoint by version.

        Implements:
            - 9.2 Checkpoint loading

        Args:
            version: Model version to load
            verify_integrity: Whether to verify checksum integrity

        Returns:
            checkpoint: Loaded model checkpoint

        Raises:
            FileNotFoundError: If checkpoint version not found
            ValueError: If integrity check fails
        """
        # Check version exists
        if not self._version_exists(version):
            raise FileNotFoundError(f"Checkpoint version {version} not found")
        
        # Get checkpoint path
        checkpoint_path = self._get_checkpoint_path(version)
        
        # Verify integrity if enabled
        if verify_integrity and self.enable_integrity_checks:
            if not self._verify_integrity(version):
                raise ValueError(f"Integrity check failed for version {version}")
        
        # Load checkpoint
        checkpoint_data = torch.load(checkpoint_path)
        
        # Reconstruct ModelCheckpoint
        checkpoint = ModelCheckpoint(
            version=checkpoint_data["version"],
            round_id=checkpoint_data["round_id"],
            timestamp=datetime.fromisoformat(checkpoint_data["timestamp"]),
            model_state_dict=checkpoint_data["model_state_dict"],
            optimizer_state_dict=checkpoint_data.get("optimizer_state_dict", {}),
            contributors=checkpoint_data["contributors"],
            metrics=checkpoint_data["metrics"],
            provenance=checkpoint_data["provenance"],
        )
        
        logger.info(f"Checkpoint loaded: v{version} from {checkpoint_path}")
        
        return checkpoint

    def load_model_state(
        self,
        version: int,
        model: nn.Module,
        strict: bool = True,
    ) -> nn.Module:
        """
        Load model state dict into a model instance.

        Args:
            version: Model version to load
            model: Model instance to load state into
            strict: Whether to strictly enforce state dict keys match

        Returns:
            model: Model with loaded state
        """
        checkpoint = self.load_checkpoint(version)
        model.load_state_dict(checkpoint.model_state_dict, strict=strict)
        
        logger.info(f"Model state loaded: v{version}")
        
        return model

    def get_latest_version(self) -> Optional[int]:
        """
        Get the latest model version number.

        Returns:
            version: Latest version number, or None if no checkpoints exist
        """
        if not self.version_index["versions"]:
            return None
        
        return max(v["version"] for v in self.version_index["versions"])

    def get_version_metadata(self, version: int) -> Dict:
        """
        Get metadata for a specific version.

        Implements:
            - 9.3 Version indexing

        Args:
            version: Model version

        Returns:
            metadata: Version metadata including timestamp, contributors, metrics

        Raises:
            ValueError: If version not found
        """
        for v in self.version_index["versions"]:
            if v["version"] == version:
                return v
        
        raise ValueError(f"Version {version} not found in index")

    def list_versions(
        self,
        start_version: Optional[int] = None,
        end_version: Optional[int] = None,
        tags: Optional[Dict[str, str]] = None,
    ) -> List[Dict]:
        """
        List available model versions with optional filtering.

        Implements:
            - 9.3 Version indexing

        Args:
            start_version: Minimum version (inclusive)
            end_version: Maximum version (inclusive)
            tags: Filter by tags (e.g., {"type": "best"})

        Returns:
            versions: List of version metadata dicts
        """
        versions = self.version_index["versions"]
        
        # Filter by version range
        if start_version is not None:
            versions = [v for v in versions if v["version"] >= start_version]
        
        if end_version is not None:
            versions = [v for v in versions if v["version"] <= end_version]
        
        # Filter by tags
        if tags:
            versions = [
                v for v in versions
                if all(v.get("tags", {}).get(k) == val for k, val in tags.items())
            ]
        
        # Sort by version
        versions.sort(key=lambda v: v["version"])
        
        return versions

    def get_provenance(self, version: int) -> Dict:
        """
        Get provenance information for a model version.

        Implements:
            - 9.4 Provenance tracking

        Args:
            version: Model version

        Returns:
            provenance: Provenance information including:
                - contributors: List of client IDs that contributed
                - aggregation_algorithm: Algorithm used (FedAvg, FedProx, etc.)
                - hyperparameters: Training hyperparameters
                - training_rounds: Number of rounds
                - timestamp: Creation timestamp
                - metrics: Performance metrics

        Raises:
            ValueError: If version not found
        """
        version_str = str(version)
        
        if version_str not in self.provenance_db:
            raise ValueError(f"Provenance not found for version {version}")
        
        return self.provenance_db[version_str]

    def get_contributors(self, version: int) -> List[str]:
        """
        Get list of clients that contributed to a model version.

        Implements:
            - 9.4 Provenance tracking

        Args:
            version: Model version

        Returns:
            contributors: List of client IDs
        """
        provenance = self.get_provenance(version)
        return provenance.get("contributors", [])

    def rollback(
        self,
        target_version: int,
        model: nn.Module,
        reason: str = "manual_rollback",
    ) -> nn.Module:
        """
        Rollback to a previous model version.

        Implements:
            - 9.5 Rollback support

        Args:
            target_version: Version to rollback to
            model: Model instance to load state into
            reason: Reason for rollback (for audit logging)

        Returns:
            model: Model with rolled back state

        Raises:
            ValueError: If target version not found
        """
        logger.info(f"Rolling back to version {target_version} (reason: {reason})")
        
        # Load target version
        checkpoint = self.load_checkpoint(target_version)
        model.load_state_dict(checkpoint.model_state_dict)
        
        # Log rollback event
        self._log_rollback_event(target_version, reason)
        
        logger.info(f"Rollback complete: now at v{target_version}")
        
        return model

    def compare_versions(
        self,
        version_a: int,
        version_b: int,
    ) -> Dict[str, any]:
        """
        Compare two model versions.

        Args:
            version_a: First version
            version_b: Second version

        Returns:
            comparison: Comparison results including:
                - metric_diff: Difference in metrics
                - contributor_diff: Difference in contributors
                - provenance_diff: Difference in provenance
        """
        # Get metadata
        metadata_a = self.get_version_metadata(version_a)
        metadata_b = self.get_version_metadata(version_b)
        
        # Get provenance
        provenance_a = self.get_provenance(version_a)
        provenance_b = self.get_provenance(version_b)
        
        # Compare metrics
        metrics_a = metadata_a.get("metrics", {})
        metrics_b = metadata_b.get("metrics", {})
        metric_diff = {
            k: metrics_b.get(k, 0) - metrics_a.get(k, 0)
            for k in set(metrics_a.keys()) | set(metrics_b.keys())
        }
        
        # Compare contributors
        contributors_a = set(provenance_a.get("contributors", []))
        contributors_b = set(provenance_b.get("contributors", []))
        contributor_diff = {
            "added": list(contributors_b - contributors_a),
            "removed": list(contributors_a - contributors_b),
            "common": list(contributors_a & contributors_b),
        }
        
        # Compare provenance
        provenance_diff = {
            "aggregation_algorithm": {
                "a": provenance_a.get("aggregation_algorithm"),
                "b": provenance_b.get("aggregation_algorithm"),
            },
            "training_rounds": {
                "a": provenance_a.get("training_rounds"),
                "b": provenance_b.get("training_rounds"),
            },
        }
        
        return {
            "version_a": version_a,
            "version_b": version_b,
            "metric_diff": metric_diff,
            "contributor_diff": contributor_diff,
            "provenance_diff": provenance_diff,
        }

    def delete_version(self, version: int, force: bool = False):
        """
        Delete a specific model version.

        Args:
            version: Version to delete
            force: Force deletion even if it's the latest version

        Raises:
            ValueError: If trying to delete latest version without force
        """
        latest = self.get_latest_version()
        
        if not force and version == latest:
            raise ValueError(
                f"Cannot delete latest version {version} without force=True"
            )
        
        # Remove checkpoint file
        checkpoint_path = self._get_checkpoint_path(version)
        if checkpoint_path.exists():
            checkpoint_path.unlink()
        
        # Remove from version index
        self.version_index["versions"] = [
            v for v in self.version_index["versions"]
            if v["version"] != version
        ]
        self._save_version_index()
        
        # Remove from provenance database
        version_str = str(version)
        if version_str in self.provenance_db:
            del self.provenance_db[version_str]
            self._save_provenance_db()
        
        logger.info(f"Version {version} deleted")

    def export_checkpoint(
        self,
        version: int,
        export_path: str,
        include_optimizer: bool = False,
    ):
        """
        Export a checkpoint to a different location.

        Args:
            version: Version to export
            export_path: Destination path
            include_optimizer: Whether to include optimizer state
        """
        checkpoint = self.load_checkpoint(version)
        
        export_data = {
            "version": checkpoint.version,
            "round_id": checkpoint.round_id,
            "timestamp": checkpoint.timestamp.isoformat(),
            "model_state_dict": checkpoint.model_state_dict,
            "metrics": checkpoint.metrics,
            "provenance": checkpoint.provenance,
        }
        
        if include_optimizer:
            export_data["optimizer_state_dict"] = checkpoint.optimizer_state_dict
        
        torch.save(export_data, export_path)
        
        logger.info(f"Checkpoint v{version} exported to {export_path}")

    # Private helper methods

    def _get_checkpoint_path(self, version: int) -> Path:
        """Get checkpoint file path for a version."""
        return self.storage_path / f"model_v{version}.pt"

    def _version_exists(self, version: int) -> bool:
        """Check if a version exists."""
        return any(v["version"] == version for v in self.version_index["versions"])

    def _load_version_index(self) -> Dict:
        """Load version index from disk."""
        if self.version_index_path.exists():
            with open(self.version_index_path, "r") as f:
                return json.load(f)
        else:
            return {"versions": []}

    def _save_version_index(self):
        """Save version index to disk."""
        with open(self.version_index_path, "w") as f:
            json.dump(self.version_index, f, indent=2)

    def _add_to_version_index(
        self,
        checkpoint: ModelCheckpoint,
        checkpoint_path: Path,
        checksum: Optional[str],
        tags: Optional[Dict[str, str]],
    ):
        """Add a new version to the index."""
        version_entry = {
            "version": checkpoint.version,
            "round_id": checkpoint.round_id,
            "timestamp": checkpoint.timestamp.isoformat(),
            "path": str(checkpoint_path),
            "contributors": checkpoint.contributors,
            "metrics": checkpoint.metrics,
            "provenance": checkpoint.provenance,
            "checksum": checksum,
            "tags": tags or {},
        }
        
        self.version_index["versions"].append(version_entry)
        self._save_version_index()

    def _load_provenance_db(self) -> Dict:
        """Load provenance database from disk."""
        if self.provenance_path.exists():
            with open(self.provenance_path, "r") as f:
                return json.load(f)
        else:
            return {}

    def _save_provenance_db(self):
        """Save provenance database to disk."""
        with open(self.provenance_path, "w") as f:
            json.dump(self.provenance_db, f, indent=2)

    def _add_to_provenance_db(self, checkpoint: ModelCheckpoint):
        """Add provenance information for a checkpoint."""
        version_str = str(checkpoint.version)
        
        self.provenance_db[version_str] = {
            "version": checkpoint.version,
            "round_id": checkpoint.round_id,
            "timestamp": checkpoint.timestamp.isoformat(),
            "contributors": checkpoint.contributors,
            "aggregation_algorithm": checkpoint.provenance.get("aggregation_algorithm"),
            "hyperparameters": {
                "local_epochs": checkpoint.provenance.get("local_epochs"),
                "learning_rate": checkpoint.provenance.get("learning_rate"),
            },
            "training_rounds": checkpoint.provenance.get("total_rounds"),
            "byzantine_detection": checkpoint.provenance.get("byzantine_detection"),
            "metrics": checkpoint.metrics,
        }
        
        self._save_provenance_db()

    def _compute_checksum(self, file_path: Path) -> str:
        """Compute SHA-256 checksum of a file."""
        sha256 = hashlib.sha256()
        
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        
        return sha256.hexdigest()

    def _verify_integrity(self, version: int) -> bool:
        """Verify checkpoint integrity using stored checksum."""
        metadata = self.get_version_metadata(version)
        stored_checksum = metadata.get("checksum")
        
        if not stored_checksum:
            logger.warning(f"No checksum found for version {version}")
            return True  # Skip verification if no checksum
        
        checkpoint_path = self._get_checkpoint_path(version)
        current_checksum = self._compute_checksum(checkpoint_path)
        
        if current_checksum != stored_checksum:
            logger.error(
                f"Integrity check failed for version {version}: "
                f"expected {stored_checksum}, got {current_checksum}"
            )
            return False
        
        return True

    def _apply_retention_policy(self):
        """Apply retention policy by deleting old versions."""
        versions = sorted(
            self.version_index["versions"],
            key=lambda v: v["version"],
            reverse=True
        )
        
        if len(versions) <= self.retention_policy:
            return
        
        # Keep only the most recent versions
        versions_to_keep = versions[:self.retention_policy]
        versions_to_delete = versions[self.retention_policy:]
        
        for version_entry in versions_to_delete:
            version = version_entry["version"]
            
            # Delete checkpoint file
            checkpoint_path = Path(version_entry["path"])
            if checkpoint_path.exists():
                checkpoint_path.unlink()
            
            # Remove from provenance database
            version_str = str(version)
            if version_str in self.provenance_db:
                del self.provenance_db[version_str]
            
            logger.info(f"Retention policy: deleted version {version}")
        
        # Update version index
        self.version_index["versions"] = versions_to_keep
        self._save_version_index()
        self._save_provenance_db()

    def _log_rollback_event(self, target_version: int, reason: str):
        """Log a rollback event."""
        rollback_log_path = self.storage_path / "rollback_log.jsonl"
        
        event = {
            "timestamp": datetime.now().isoformat(),
            "target_version": target_version,
            "reason": reason,
        }
        
        with open(rollback_log_path, "a") as f:
            f.write(json.dumps(event) + "\n")
