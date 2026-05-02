"""Training orchestrator for federated learning."""

import hashlib
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set

import torch
import torch.nn as nn

from src.federated.aggregator.base import BaseAggregator
from src.federated.aggregator.byzantine_robust import ByzantineDetector
from src.federated.aggregator.fedavg import FedAvgAggregator
from src.federated.common.data_models import (
    AuditLogEntry,
    ClientUpdate,
    ModelCheckpoint,
    TrainingRound,
)

logger = logging.getLogger(__name__)


class TrainingOrchestrator:
    """
    Orchestrates federated training rounds.

    Responsibilities:
        - Initialize training rounds with client selection
        - Broadcast global model to clients
        - Collect client updates
        - Trigger aggregation with Byzantine detection
        - Update global model with versioning
        - Track training progress and metrics
        - Generate audit logs for HIPAA compliance

    Correctness Properties:
        - Invariant: Model version increments by 1 per round
        - Invariant: Number of aggregated updates ≤ number of active clients
        - Invariant: Audit log generated for every round event
    """

    def __init__(
        self,
        model: nn.Module,
        aggregator: Optional[BaseAggregator] = None,
        byzantine_detector: Optional[ByzantineDetector] = None,
        checkpoint_dir: str = "./fl_checkpoints",
        audit_log_dir: str = "./fl_audit_logs",
        local_epochs: int = 5,
        learning_rate: float = 0.01,
        min_clients_per_round: int = 3,
        client_selection_fraction: float = 1.0,
        enable_byzantine_detection: bool = True,
        enable_audit_logging: bool = True,
    ):
        """
        Initialize training orchestrator.

        Args:
            model: Global model to train
            aggregator: Aggregation algorithm (default: FedAvg)
            byzantine_detector: Byzantine detection component
            checkpoint_dir: Directory for saving checkpoints
            audit_log_dir: Directory for audit logs
            local_epochs: Number of local training epochs per round
            learning_rate: Learning rate for client training
            min_clients_per_round: Minimum clients required per round
            client_selection_fraction: Fraction of clients to select per round
            enable_byzantine_detection: Enable Byzantine robustness
            enable_audit_logging: Enable HIPAA-compliant audit logging
        """
        self.global_model = model
        self.aggregator = aggregator or FedAvgAggregator()
        self.byzantine_detector = byzantine_detector or ByzantineDetector()
        
        # Directories
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.audit_log_dir = Path(audit_log_dir)
        self.audit_log_dir.mkdir(parents=True, exist_ok=True)
        
        # Training hyperparameters
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.min_clients_per_round = min_clients_per_round
        self.client_selection_fraction = client_selection_fraction
        
        # Feature flags
        self.enable_byzantine_detection = enable_byzantine_detection
        self.enable_audit_logging = enable_audit_logging
        
        # Initialize optimizer for global model
        self.optimizer = torch.optim.SGD(self.global_model.parameters(), lr=learning_rate)

        # State tracking
        self.current_version = 0
        self.current_round = 0
        self.training_history: List[TrainingRound] = []
        self.round_contributors: Dict[int, List[str]] = {}  # round_id -> client_ids
        self.available_clients: Set[str] = set()  # All registered clients
        self.active_clients: Set[str] = set()  # Clients participating in current round
        
        # Metrics tracking
        self.round_metrics: Dict[int, Dict[str, float]] = {}  # round_id -> metrics
        self.client_metrics: Dict[str, List[Dict[str, float]]] = {}  # client_id -> metrics history
        
        # Audit logging
        self.audit_log: List[AuditLogEntry] = []
        self.last_audit_hash = ""
        
        logger.info(f"TrainingOrchestrator initialized with {self.aggregator.algorithm_name} aggregation")
        self._log_audit_event("orchestrator_initialized", None, 0, {
            "aggregation_algorithm": self.aggregator.algorithm_name,
            "byzantine_detection": enable_byzantine_detection,
            "min_clients": min_clients_per_round,
        })

    def register_client(self, client_id: str):
        """
        Register a client as available for training.
        
        Args:
            client_id: Unique client identifier
        """
        self.available_clients.add(client_id)
        logger.info(f"Client {client_id} registered. Total clients: {len(self.available_clients)}")
        
        self._log_audit_event("client_registered", client_id, self.current_round, {
            "total_clients": len(self.available_clients)
        })
    
    def unregister_client(self, client_id: str):
        """
        Unregister a client from training.
        
        Args:
            client_id: Unique client identifier
        """
        self.available_clients.discard(client_id)
        self.active_clients.discard(client_id)
        logger.info(f"Client {client_id} unregistered. Total clients: {len(self.available_clients)}")
        
        self._log_audit_event("client_unregistered", client_id, self.current_round, {
            "total_clients": len(self.available_clients)
        })
    
    def select_clients(self, available_clients: Optional[List[str]] = None) -> List[str]:
        """
        Select clients for the next training round.
        
        Implements client selection strategy based on:
        - Client availability
        - Selection fraction
        - Minimum client threshold
        
        Args:
            available_clients: Optional list of available clients (uses registered if None)
        
        Returns:
            selected_clients: List of selected client IDs
        
        Raises:
            ValueError: If insufficient clients available
        """
        if available_clients is None:
            available_clients = list(self.available_clients)
        
        if len(available_clients) < self.min_clients_per_round:
            raise ValueError(
                f"Insufficient clients: {len(available_clients)} < {self.min_clients_per_round}"
            )
        
        # Calculate number of clients to select
        num_to_select = max(
            self.min_clients_per_round,
            int(len(available_clients) * self.client_selection_fraction)
        )
        num_to_select = min(num_to_select, len(available_clients))
        
        # Random selection (can be enhanced with more sophisticated strategies)
        import random
        selected = random.sample(available_clients, num_to_select)
        
        logger.info(f"Selected {len(selected)}/{len(available_clients)} clients for round {self.current_round + 1}")
        
        return selected

    def start_round(self, client_ids: Optional[List[str]] = None) -> TrainingRound:
        """
        Start a new training round with client selection.

        Args:
            client_ids: Optional list of participating client IDs (auto-selects if None)

        Returns:
            training_round: Metadata for the new round
        
        Raises:
            ValueError: If insufficient clients available
        """
        # Select clients if not provided
        if client_ids is None:
            client_ids = self.select_clients()
        
        # Validate minimum clients
        if len(client_ids) < self.min_clients_per_round:
            raise ValueError(
                f"Insufficient clients for round: {len(client_ids)} < {self.min_clients_per_round}"
            )
        
        self.current_round += 1
        self.active_clients = set(client_ids)

        round_metadata = TrainingRound(
            round_id=self.current_round,
            global_model_version=self.current_version,
            start_time=datetime.now(),
            participants=client_ids,
            aggregation_algorithm=self.aggregator.algorithm_name,
            status="in_progress",
        )

        self.training_history.append(round_metadata)
        
        # Audit log
        self._log_audit_event("round_started", None, self.current_round, {
            "participants": client_ids,
            "model_version": self.current_version,
            "aggregation_algorithm": self.aggregator.algorithm_name,
        })
        
        logger.info(
            f"Round {self.current_round} started with {len(client_ids)} clients "
            f"(model v{self.current_version})"
        )
        
        return round_metadata

    def broadcast_model(self) -> Dict[str, torch.Tensor]:
        """
        Broadcast current global model to clients.
        
        This method is called by the communication layer (gRPC server)
        to serve the global model to clients.
        
        Returns:
            model_state: Global model state dict
        """
        model_state = self.get_global_model()
        
        logger.info(
            f"Broadcasting model v{self.current_version} "
            f"(round {self.current_round}, {len(model_state)} parameters)"
        )
        
        return model_state

    def get_global_model(self) -> Dict[str, torch.Tensor]:
        """
        Get current global model state dict.

        Returns:
            model_state: Global model parameters
        """
        return self.global_model.state_dict()

    def collect_updates(
        self, 
        client_updates: List[ClientUpdate],
        validate: bool = True
    ) -> List[ClientUpdate]:
        """
        Collect and validate client updates.
        
        Args:
            client_updates: List of updates from clients
            validate: Whether to validate updates
        
        Returns:
            valid_updates: List of validated updates
        """
        if validate:
            valid_updates = []
            for update in client_updates:
                if self._validate_update(update):
                    valid_updates.append(update)
                else:
                    logger.warning(f"Invalid update from client {update.client_id}")
                    self._log_audit_event("invalid_update", update.client_id, self.current_round, {
                        "reason": "validation_failed"
                    })
        else:
            valid_updates = client_updates
        
        logger.info(
            f"Collected {len(valid_updates)}/{len(client_updates)} valid updates "
            f"for round {self.current_round}"
        )
        
        # Log each update
        for update in valid_updates:
            self._log_audit_event("update_received", update.client_id, self.current_round, {
                "dataset_size": update.dataset_size,
                "training_time": update.training_time_seconds,
                "privacy_epsilon": update.privacy_epsilon,
            })
        
        return valid_updates
    
    def _validate_update(self, update: ClientUpdate) -> bool:
        """
        Validate a client update.
        
        Args:
            update: Client update to validate
        
        Returns:
            is_valid: True if update is valid
        """
        # Check round ID matches
        if update.round_id != self.current_round:
            logger.warning(
                f"Round ID mismatch: update={update.round_id}, current={self.current_round}"
            )
            return False
        
        # Check client is in active set
        if update.client_id not in self.active_clients:
            logger.warning(f"Client {update.client_id} not in active set")
            return False
        
        # Check gradients are not empty
        if not update.gradients:
            logger.warning(f"Empty gradients from client {update.client_id}")
            return False
        
        # Check gradient shapes match model
        model_state = self.global_model.state_dict()
        for param_name, gradient in update.gradients.items():
            if param_name not in model_state:
                logger.warning(f"Unknown parameter: {param_name}")
                return False
            if gradient.shape != model_state[param_name].shape:
                logger.warning(
                    f"Shape mismatch for {param_name}: "
                    f"{gradient.shape} != {model_state[param_name].shape}"
                )
                return False
        
        return True

    def aggregate_updates(
        self, 
        client_updates: List[ClientUpdate],
        apply_byzantine_detection: bool = True
    ) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates with optional Byzantine detection.

        Args:
            client_updates: List of updates from clients
            apply_byzantine_detection: Whether to apply Byzantine detection

        Returns:
            aggregated_update: Aggregated gradients
        """
        if not client_updates:
            raise ValueError("No client updates to aggregate")
        
        # Track contributors for this round
        contributor_ids = [update.client_id for update in client_updates]
        self.round_contributors[self.current_round] = contributor_ids
        
        # Apply Byzantine detection if enabled
        if apply_byzantine_detection and self.enable_byzantine_detection:
            logger.info("Applying Byzantine detection...")
            
            # Detect Byzantine updates
            byzantine_indices, clean_indices = self.byzantine_detector.detect_byzantine_clients(client_updates)
            
            # Filter out Byzantine updates
            clean_updates = [client_updates[i] for i in clean_indices]
            
            num_excluded = len(byzantine_indices)
            if num_excluded > 0:
                excluded_clients = [client_updates[i].client_id for i in byzantine_indices]
                logger.warning(f"Excluded {num_excluded} Byzantine updates: {excluded_clients}")
                
                # Audit log
                for client_id in excluded_clients:
                    self._log_audit_event("byzantine_detected", client_id, self.current_round, {
                        "reason": "outlier_detection"
                    })
            
            client_updates = clean_updates
        
        if not client_updates:
            raise ValueError("All updates flagged as Byzantine")
        
        # Aggregate using configured algorithm
        logger.info(f"Aggregating {len(client_updates)} updates using {self.aggregator.algorithm_name}")
        aggregated_update = self.aggregator.aggregate(client_updates)
        
        # Audit log
        self._log_audit_event("aggregation_complete", None, self.current_round, {
            "num_updates": len(client_updates),
            "algorithm": self.aggregator.algorithm_name,
        })
        
        return aggregated_update

    def update_global_model(
        self, 
        aggregated_update: Dict[str, torch.Tensor],
        increment_version: bool = True
    ):
        """
        Update global model with aggregated gradients and increment version.

        Args:
            aggregated_update: Aggregated gradients from clients
            increment_version: Whether to increment model version
        """
        # Apply aggregated update to global model
        with torch.no_grad():
            for param_name, param in self.global_model.named_parameters():
                if param_name in aggregated_update:
                    param.add_(aggregated_update[param_name])

        # Increment version
        if increment_version:
            old_version = self.current_version
            self.current_version += 1
            
            logger.info(f"Global model updated: v{old_version} -> v{self.current_version}")
            
            # Audit log
            self._log_audit_event("model_updated", None, self.current_round, {
                "old_version": old_version,
                "new_version": self.current_version,
            })

    def complete_round(
        self, 
        convergence_metrics: Optional[Dict[str, float]] = None,
        save_checkpoint: bool = True
    ):
        """
        Mark current round as complete and optionally save checkpoint.

        Args:
            convergence_metrics: Metrics for this round (loss, accuracy, etc.)
            save_checkpoint: Whether to save model checkpoint
        """
        if not self.training_history:
            raise ValueError("No active training round")

        current_round = self.training_history[-1]
        current_round.end_time = datetime.now()
        current_round.status = "completed"

        if convergence_metrics:
            current_round.convergence_metrics = convergence_metrics
            self.round_metrics[self.current_round] = convergence_metrics
            
            logger.info(
                f"Round {self.current_round} metrics: "
                f"{', '.join(f'{k}={v:.4f}' for k, v in convergence_metrics.items())}"
            )
        
        # Calculate round duration
        duration = (current_round.end_time - current_round.start_time).total_seconds()
        
        # Audit log
        self._log_audit_event("round_completed", None, self.current_round, {
            "duration_seconds": duration,
            "metrics": convergence_metrics or {},
            "contributors": current_round.participants,
        })
        
        # Save checkpoint if requested
        if save_checkpoint:
            self.save_checkpoint(convergence_metrics)
        
        logger.info(f"Round {self.current_round} completed in {duration:.1f}s")

    def save_checkpoint(
        self, 
        metrics: Optional[Dict[str, float]] = None,
        include_optimizer: bool = True
    ):
        """
        Save global model checkpoint with versioning and provenance.

        Args:
            metrics: Validation metrics for this checkpoint
            include_optimizer: Whether to include optimizer state
        """
        checkpoint = ModelCheckpoint(
            version=self.current_version,
            round_id=self.current_round,
            timestamp=datetime.now(),
            model_state_dict=self.global_model.state_dict(),
            optimizer_state_dict=self.optimizer.state_dict() if include_optimizer else {},
            contributors=self.round_contributors.get(self.current_round, []),
            metrics=metrics or {},
            provenance={
                "aggregation_algorithm": self.aggregator.algorithm_name,
                "local_epochs": self.local_epochs,
                "learning_rate": self.learning_rate,
                "byzantine_detection": self.enable_byzantine_detection,
                "total_rounds": self.current_round,
            },
        )

        checkpoint_path = self.checkpoint_dir / f"model_v{self.current_version}.pt"
        torch.save(
            {
                "version": checkpoint.version,
                "round_id": checkpoint.round_id,
                "timestamp": checkpoint.timestamp.isoformat(),
                "model_state_dict": checkpoint.model_state_dict,
                "optimizer_state_dict": checkpoint.optimizer_state_dict,
                "contributors": checkpoint.contributors,
                "metrics": checkpoint.metrics,
                "provenance": checkpoint.provenance,
            },
            checkpoint_path,
        )
        
        # Update version index
        self._update_version_index(checkpoint)
        
        # Audit log
        self._log_audit_event("checkpoint_saved", None, self.current_round, {
            "version": self.current_version,
            "path": str(checkpoint_path),
            "metrics": metrics or {},
        })
        
        logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def _update_version_index(self, checkpoint: ModelCheckpoint):
        """
        Update version index file with checkpoint metadata.
        
        Args:
            checkpoint: Model checkpoint to index
        """
        index_path = self.checkpoint_dir / "version_index.json"
        
        # Load existing index
        if index_path.exists():
            with open(index_path, "r") as f:
                index = json.load(f)
        else:
            index = {"versions": []}
        
        # Add new version
        index["versions"].append({
            "version": checkpoint.version,
            "round_id": checkpoint.round_id,
            "timestamp": checkpoint.timestamp.isoformat(),
            "contributors": checkpoint.contributors,
            "metrics": checkpoint.metrics,
            "provenance": checkpoint.provenance,
        })
        
        # Save index
        with open(index_path, "w") as f:
            json.dump(index, f, indent=2)

    def load_checkpoint(self, version: int, load_optimizer: bool = True):
        """
        Load a specific model checkpoint for rollback or recovery.

        Args:
            version: Model version to load
            load_optimizer: Whether to load optimizer state
        """
        checkpoint_path = self.checkpoint_dir / f"model_v{version}.pt"

        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint version {version} not found")

        checkpoint = torch.load(checkpoint_path)
        self.global_model.load_state_dict(checkpoint["model_state_dict"])
        
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        self.current_version = checkpoint["version"]
        
        # Audit log
        self._log_audit_event("checkpoint_loaded", None, self.current_round, {
            "version": version,
            "path": str(checkpoint_path),
        })
        
        logger.info(f"Checkpoint loaded: v{version} from {checkpoint_path}")
    
    def get_training_history(self) -> List[TrainingRound]:
        """
        Get complete training history.
        
        Returns:
            training_history: List of all training rounds
        """
        return self.training_history.copy()
    
    def get_round_metrics(self, round_id: Optional[int] = None) -> Dict[str, float]:
        """
        Get metrics for a specific round or current round.
        
        Args:
            round_id: Round ID (uses current round if None)
        
        Returns:
            metrics: Round metrics
        """
        if round_id is None:
            round_id = self.current_round
        
        return self.round_metrics.get(round_id, {})
    
    def _log_audit_event(
        self,
        event_type: str,
        client_id: Optional[str],
        round_id: int,
        details: Dict[str, any]
    ):
        """
        Log an audit event with tamper-evident hashing.
        
        Args:
            event_type: Type of event
            client_id: Client ID (if applicable)
            round_id: Round ID
            details: Event details
        """
        if not self.enable_audit_logging:
            return
        
        timestamp = datetime.now()
        
        # Create audit entry
        entry = AuditLogEntry(
            timestamp=timestamp,
            event_type=event_type,
            client_id=client_id,
            round_id=round_id,
            details=details,
            hash="",  # Will be computed
        )
        
        # Compute tamper-evident hash
        entry_data = {
            "timestamp": timestamp.isoformat(),
            "event_type": event_type,
            "client_id": client_id,
            "round_id": round_id,
            "details": details,
            "previous_hash": self.last_audit_hash,
        }
        entry_json = json.dumps(entry_data, sort_keys=True)
        entry_hash = hashlib.sha256(entry_json.encode()).hexdigest()
        entry.hash = entry_hash
        
        # Update last hash
        self.last_audit_hash = entry_hash
        
        # Store entry
        self.audit_log.append(entry)
        
        # Persist to disk
        self._persist_audit_log(entry)
    
    def _persist_audit_log(self, entry: AuditLogEntry):
        """
        Persist audit log entry to disk.
        
        Args:
            entry: Audit log entry
        """
        log_file = self.audit_log_dir / f"audit_log_{datetime.now().strftime('%Y%m%d')}.jsonl"
        
        entry_data = {
            "timestamp": entry.timestamp.isoformat(),
            "event_type": entry.event_type,
            "client_id": entry.client_id,
            "round_id": entry.round_id,
            "details": entry.details,
            "hash": entry.hash,
        }
        
        with open(log_file, "a") as f:
            f.write(json.dumps(entry_data) + "\n")
    
    def get_audit_log(
        self, 
        start_round: Optional[int] = None,
        end_round: Optional[int] = None,
        event_type: Optional[str] = None
    ) -> List[AuditLogEntry]:
        """
        Get audit log entries with optional filtering.
        
        Args:
            start_round: Start round ID (inclusive)
            end_round: End round ID (inclusive)
            event_type: Filter by event type
        
        Returns:
            filtered_log: Filtered audit log entries
        """
        filtered = self.audit_log
        
        if start_round is not None:
            filtered = [e for e in filtered if e.round_id >= start_round]
        
        if end_round is not None:
            filtered = [e for e in filtered if e.round_id <= end_round]
        
        if event_type is not None:
            filtered = [e for e in filtered if e.event_type == event_type]
        
        return filtered
    
    def verify_audit_log(self) -> bool:
        """
        Verify audit log integrity using hash chain.
        
        Returns:
            is_valid: True if audit log is untampered
        """
        if not self.audit_log:
            return True
        
        previous_hash = ""
        for entry in self.audit_log:
            # Recompute hash
            entry_data = {
                "timestamp": entry.timestamp.isoformat(),
                "event_type": entry.event_type,
                "client_id": entry.client_id,
                "round_id": entry.round_id,
                "details": entry.details,
                "previous_hash": previous_hash,
            }
            entry_json = json.dumps(entry_data, sort_keys=True)
            expected_hash = hashlib.sha256(entry_json.encode()).hexdigest()
            
            if entry.hash != expected_hash:
                logger.error(f"Audit log tampered: entry {entry.timestamp}")
                return False
            
            previous_hash = entry.hash
        
        return True
