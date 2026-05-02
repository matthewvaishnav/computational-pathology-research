"""
Checkpoint management for fault-tolerant federated learning.

Provides checkpoint save/load, crash recovery, and training state persistence.
"""

import json
import logging
import shutil
import torch
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Any, List

logger = logging.getLogger(__name__)


@dataclass
class CheckpointMetadata:
    """Metadata for training checkpoint."""
    
    client_id: str
    round_id: int
    epoch: int
    timestamp: str
    model_version: int
    training_loss: float
    samples_processed: int
    privacy_epsilon: float
    checkpoint_path: str
    is_complete: bool = True
    error_message: Optional[str] = None


class CheckpointManager:
    """
    Manages training checkpoints for crash recovery.
    
    **Validates: Requirements 9.2, 9.4**
    
    Features:
    - Automatic checkpoint saving during training
    - Crash recovery from last valid checkpoint
    - Checkpoint versioning and cleanup
    - Training state persistence
    """
    
    def __init__(
        self,
        checkpoint_dir: Path,
        client_id: str,
        max_checkpoints: int = 5,
        save_interval: int = 1,  # Save every N rounds
    ):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for checkpoint storage
            client_id: Unique client identifier
            max_checkpoints: Maximum checkpoints to retain
            save_interval: Save checkpoint every N rounds
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.client_id = client_id
        self.max_checkpoints = max_checkpoints
        self.save_interval = save_interval
        
        # Create checkpoint directory
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Metadata tracking
        self.checkpoints: List[CheckpointMetadata] = []
        self.load_checkpoint_index()
        
        logger.info(f"Checkpoint manager initialized: {self.checkpoint_dir}")
    
    def save_checkpoint(
        self,
        model_state: Dict[str, torch.Tensor],
        optimizer_state: Dict[str, Any],
        round_id: int,
        epoch: int,
        training_loss: float,
        samples_processed: int,
        privacy_epsilon: float,
        model_version: int,
        additional_state: Optional[Dict] = None,
    ) -> CheckpointMetadata:
        """
        Save training checkpoint.
        
        Args:
            model_state: Model state_dict
            optimizer_state: Optimizer state_dict
            round_id: Current round ID
            epoch: Current epoch
            training_loss: Training loss
            samples_processed: Number of samples processed
            privacy_epsilon: Current privacy epsilon
            model_version: Global model version
            additional_state: Additional state to save
        
        Returns:
            CheckpointMetadata for saved checkpoint
        
        **Validates: Requirements 9.2**
        """
        # Check if should save based on interval
        if round_id % self.save_interval != 0:
            return None
        
        timestamp = datetime.now().isoformat()
        checkpoint_name = f"checkpoint_r{round_id}_e{epoch}_{timestamp.replace(':', '-')}.pt"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        
        try:
            # Prepare checkpoint data
            checkpoint_data = {
                'model_state_dict': model_state,
                'optimizer_state_dict': optimizer_state,
                'round_id': round_id,
                'epoch': epoch,
                'training_loss': training_loss,
                'samples_processed': samples_processed,
                'privacy_epsilon': privacy_epsilon,
                'model_version': model_version,
                'timestamp': timestamp,
                'client_id': self.client_id,
            }
            
            if additional_state:
                checkpoint_data['additional_state'] = additional_state
            
            # Save checkpoint
            torch.save(checkpoint_data, checkpoint_path)
            
            # Create metadata
            metadata = CheckpointMetadata(
                client_id=self.client_id,
                round_id=round_id,
                epoch=epoch,
                timestamp=timestamp,
                model_version=model_version,
                training_loss=training_loss,
                samples_processed=samples_processed,
                privacy_epsilon=privacy_epsilon,
                checkpoint_path=str(checkpoint_path),
                is_complete=True,
            )
            
            # Add to index
            self.checkpoints.append(metadata)
            self.save_checkpoint_index()
            
            # Cleanup old checkpoints
            self._cleanup_old_checkpoints()
            
            logger.info(f"Saved checkpoint: {checkpoint_name}")
            return metadata
            
        except Exception as e:
            logger.error(f"Failed to save checkpoint: {e}")
            return CheckpointMetadata(
                client_id=self.client_id,
                round_id=round_id,
                epoch=epoch,
                timestamp=timestamp,
                model_version=model_version,
                training_loss=training_loss,
                samples_processed=samples_processed,
                privacy_epsilon=privacy_epsilon,
                checkpoint_path=str(checkpoint_path),
                is_complete=False,
                error_message=str(e),
            )
    
    def load_checkpoint(
        self,
        round_id: Optional[int] = None,
    ) -> Optional[Dict[str, Any]]:
        """
        Load checkpoint for recovery.
        
        Args:
            round_id: Specific round to load (None = latest)
        
        Returns:
            Checkpoint data or None if not found
        
        **Validates: Requirements 9.4**
        """
        if not self.checkpoints:
            logger.warning("No checkpoints available for recovery")
            return None
        
        # Find checkpoint
        if round_id is not None:
            # Load specific round
            checkpoint_meta = next(
                (c for c in reversed(self.checkpoints) if c.round_id == round_id and c.is_complete),
                None
            )
        else:
            # Load latest complete checkpoint
            checkpoint_meta = next(
                (c for c in reversed(self.checkpoints) if c.is_complete),
                None
            )
        
        if not checkpoint_meta:
            logger.warning(f"No valid checkpoint found for round {round_id}")
            return None
        
        try:
            checkpoint_path = Path(checkpoint_meta.checkpoint_path)
            if not checkpoint_path.exists():
                logger.error(f"Checkpoint file not found: {checkpoint_path}")
                return None
            
            # Load checkpoint
            checkpoint_data = torch.load(checkpoint_path)
            
            logger.info(
                f"Loaded checkpoint from round {checkpoint_meta.round_id}, "
                f"epoch {checkpoint_meta.epoch}"
            )
            
            return checkpoint_data
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint: {e}")
            return None
    
    def get_latest_checkpoint_metadata(self) -> Optional[CheckpointMetadata]:
        """Get metadata for latest checkpoint."""
        if not self.checkpoints:
            return None
        
        return next(
            (c for c in reversed(self.checkpoints) if c.is_complete),
            None
        )
    
    def has_checkpoint(self) -> bool:
        """Check if any valid checkpoint exists."""
        return any(c.is_complete for c in self.checkpoints)
    
    def recover_from_crash(self) -> Optional[Dict[str, Any]]:
        """
        Recover training state after crash.
        
        Returns:
            Latest checkpoint data or None
        
        **Validates: Requirements 9.2, 9.4**
        """
        logger.info("Attempting crash recovery...")
        
        checkpoint_data = self.load_checkpoint()
        
        if checkpoint_data:
            logger.info(
                f"Recovered from round {checkpoint_data['round_id']}, "
                f"epoch {checkpoint_data['epoch']}"
            )
        else:
            logger.warning("No checkpoint available for recovery")
        
        return checkpoint_data
    
    def save_checkpoint_index(self) -> None:
        """Save checkpoint index to disk."""
        index_path = self.checkpoint_dir / "checkpoint_index.json"
        
        try:
            index_data = {
                'client_id': self.client_id,
                'checkpoints': [asdict(c) for c in self.checkpoints],
            }
            
            with open(index_path, 'w') as f:
                json.dump(index_data, f, indent=2)
                
        except Exception as e:
            logger.error(f"Failed to save checkpoint index: {e}")
    
    def load_checkpoint_index(self) -> None:
        """Load checkpoint index from disk."""
        index_path = self.checkpoint_dir / "checkpoint_index.json"
        
        if not index_path.exists():
            return
        
        try:
            with open(index_path, 'r') as f:
                index_data = json.load(f)
            
            self.checkpoints = [
                CheckpointMetadata(**c) for c in index_data.get('checkpoints', [])
            ]
            
            logger.info(f"Loaded {len(self.checkpoints)} checkpoints from index")
            
        except Exception as e:
            logger.error(f"Failed to load checkpoint index: {e}")
    
    def _cleanup_old_checkpoints(self) -> None:
        """Remove old checkpoints beyond max_checkpoints limit."""
        if len(self.checkpoints) <= self.max_checkpoints:
            return
        
        # Sort by round_id
        self.checkpoints.sort(key=lambda c: c.round_id)
        
        # Remove oldest checkpoints
        to_remove = self.checkpoints[:-self.max_checkpoints]
        
        for checkpoint_meta in to_remove:
            try:
                checkpoint_path = Path(checkpoint_meta.checkpoint_path)
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
                    logger.debug(f"Removed old checkpoint: {checkpoint_path.name}")
            except Exception as e:
                logger.error(f"Failed to remove checkpoint: {e}")
        
        # Update index
        self.checkpoints = self.checkpoints[-self.max_checkpoints:]
        self.save_checkpoint_index()
    
    def clear_all_checkpoints(self) -> None:
        """Remove all checkpoints (for testing/cleanup)."""
        for checkpoint_meta in self.checkpoints:
            try:
                checkpoint_path = Path(checkpoint_meta.checkpoint_path)
                if checkpoint_path.exists():
                    checkpoint_path.unlink()
            except Exception as e:
                logger.error(f"Failed to remove checkpoint: {e}")
        
        self.checkpoints.clear()
        self.save_checkpoint_index()
        
        logger.info("Cleared all checkpoints")
    
    def get_checkpoint_statistics(self) -> Dict[str, Any]:
        """Get checkpoint statistics."""
        if not self.checkpoints:
            return {
                'total_checkpoints': 0,
                'complete_checkpoints': 0,
                'failed_checkpoints': 0,
            }
        
        complete = sum(1 for c in self.checkpoints if c.is_complete)
        failed = len(self.checkpoints) - complete
        
        latest = self.get_latest_checkpoint_metadata()
        
        return {
            'total_checkpoints': len(self.checkpoints),
            'complete_checkpoints': complete,
            'failed_checkpoints': failed,
            'latest_round': latest.round_id if latest else None,
            'latest_timestamp': latest.timestamp if latest else None,
            'disk_usage_mb': self._calculate_disk_usage(),
        }
    
    def _calculate_disk_usage(self) -> float:
        """Calculate total disk usage of checkpoints in MB."""
        total_size = 0
        
        for checkpoint_meta in self.checkpoints:
            try:
                checkpoint_path = Path(checkpoint_meta.checkpoint_path)
                if checkpoint_path.exists():
                    total_size += checkpoint_path.stat().st_size
            except Exception:
                pass
        
        return total_size / (1024 * 1024)  # Convert to MB
