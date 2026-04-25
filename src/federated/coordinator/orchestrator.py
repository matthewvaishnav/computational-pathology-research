"""Training orchestrator for federated learning."""

from typing import List, Dict, Optional, Callable
from datetime import datetime
import torch
import torch.nn as nn
from pathlib import Path

from src.federated.common.data_models import TrainingRound, ClientUpdate, ModelCheckpoint
from src.federated.aggregator.fedavg import FedAvgAggregator


class TrainingOrchestrator:
    """
    Orchestrates federated training rounds.
    
    Responsibilities:
        - Initialize training rounds
        - Broadcast global model to clients
        - Collect client updates
        - Trigger aggregation
        - Update global model
        - Track training progress
    
    Correctness Properties:
        - Invariant: Model version increments by 1 per round
        - Invariant: Number of aggregated updates ≤ number of active clients
    """
    
    def __init__(
        self,
        model: nn.Module,
        aggregator: Optional[FedAvgAggregator] = None,
        checkpoint_dir: str = "./fl_checkpoints"
    ):
        """
        Initialize training orchestrator.
        
        Args:
            model: Global model to train
            aggregator: Aggregation algorithm (default: FedAvg)
            checkpoint_dir: Directory for saving checkpoints
        """
        self.global_model = model
        self.aggregator = aggregator or FedAvgAggregator()
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.current_version = 0
        self.current_round = 0
        self.training_history: List[TrainingRound] = []
    
    def start_round(self, client_ids: List[str]) -> TrainingRound:
        """
        Start a new training round.
        
        Args:
            client_ids: List of participating client IDs
        
        Returns:
            training_round: Metadata for the new round
        """
        self.current_round += 1
        
        round_metadata = TrainingRound(
            round_id=self.current_round,
            global_model_version=self.current_version,
            start_time=datetime.now(),
            participants=client_ids,
            aggregation_algorithm=self.aggregator.name,
            status="in_progress"
        )
        
        self.training_history.append(round_metadata)
        return round_metadata
    
    def get_global_model(self) -> Dict[str, torch.Tensor]:
        """
        Get current global model state dict.
        
        Returns:
            model_state: Global model parameters
        """
        return self.global_model.state_dict()
    
    def aggregate_updates(self, client_updates: List[ClientUpdate]) -> Dict[str, torch.Tensor]:
        """
        Aggregate client updates.
        
        Args:
            client_updates: List of updates from clients
        
        Returns:
            aggregated_update: Aggregated gradients
        """
        return self.aggregator.aggregate(client_updates)
    
    def update_global_model(self, aggregated_update: Dict[str, torch.Tensor]):
        """
        Update global model with aggregated gradients.
        
        Args:
            aggregated_update: Aggregated gradients from clients
        """
        # Apply aggregated update to global model
        with torch.no_grad():
            for param_name, param in self.global_model.named_parameters():
                if param_name in aggregated_update:
                    param.add_(aggregated_update[param_name])
        
        # Increment version
        self.current_version += 1
    
    def complete_round(
        self, 
        convergence_metrics: Optional[Dict[str, float]] = None
    ):
        """
        Mark current round as complete.
        
        Args:
            convergence_metrics: Metrics for this round (loss, accuracy, etc.)
        """
        if not self.training_history:
            raise ValueError("No active training round")
        
        current_round = self.training_history[-1]
        current_round.end_time = datetime.now()
        current_round.status = "completed"
        
        if convergence_metrics:
            current_round.convergence_metrics = convergence_metrics
    
    def save_checkpoint(self, metrics: Optional[Dict[str, float]] = None):
        """
        Save global model checkpoint.
        
        Args:
            metrics: Validation metrics for this checkpoint
        """
        checkpoint = ModelCheckpoint(
            version=self.current_version,
            round_id=self.current_round,
            timestamp=datetime.now(),
            model_state_dict=self.global_model.state_dict(),
            optimizer_state_dict={},  # TODO: Add optimizer state
            contributors=[],  # TODO: Track contributors
            metrics=metrics or {},
            provenance={"aggregation_algorithm": self.aggregator.name}
        )
        
        checkpoint_path = self.checkpoint_dir / f"model_v{self.current_version}.pt"
        torch.save({
            'version': checkpoint.version,
            'round_id': checkpoint.round_id,
            'timestamp': checkpoint.timestamp,
            'model_state_dict': checkpoint.model_state_dict,
            'metrics': checkpoint.metrics,
            'provenance': checkpoint.provenance
        }, checkpoint_path)
    
    def load_checkpoint(self, version: int):
        """
        Load a specific model checkpoint.
        
        Args:
            version: Model version to load
        """
        checkpoint_path = self.checkpoint_dir / f"model_v{version}.pt"
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint version {version} not found")
        
        checkpoint = torch.load(checkpoint_path)
        self.global_model.load_state_dict(checkpoint['model_state_dict'])
        self.current_version = checkpoint['version']
