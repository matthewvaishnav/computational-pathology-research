"""
Training infrastructure for nnMIL Multiple Instance Learning.

This module implements the nnMILTrainer class with large-batch optimization,
gradient accumulation, learning rate scaling, and comprehensive monitoring.
"""

import logging
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from ..config.nnmil_config import nnMILConfig
from ..data.batch_samplers import BalancedBatchSampler, RegressionBatchSampler, SurvivalBatchSampler
from ..data.data_models import TrainingBatch, InferenceOutput


class nnMILTrainer:
    """
    Trainer for nnMIL with large-batch optimization and monitoring.
    
    Supports batch sizes 1-64 with gradient accumulation for memory-constrained GPUs.
    Implements learning rate scaling, task-aware batch sampling, and comprehensive
    logging with TensorBoard integration.
    
    Args:
        model: nnMIL model to train
        config: Training configuration
        device: Device for training (default: 'cuda' if available)
        logger: Logger instance (creates new if None)
    
    Example:
        >>> config = nnMILConfig(
        ...     batch_size=32,
        ...     learning_rate=3e-4,
        ...     num_epochs=100
        ... )
        >>> trainer = nnMILTrainer(model, config)
        >>> trainer.train(train_loader, val_loader)
    """
    
    def __init__(
        self,
        model: nn.Module,
        config: nnMILConfig,
        device: Optional[str] = None,
        logger: Optional[logging.Logger] = None
    ):
        self.model = model
        self.config = config
        self.device = device if device is not None else (
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.logger = logger if logger is not None else logging.getLogger(__name__)
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Initialize training state
        self.current_epoch = 0
        self.best_metric = float('-inf')
        self.patience_counter = 0
        self.training_history = {
            'train_loss': [],
            'val_loss': [],
            'val_metric': [],
            'learning_rate': [],
            'epoch_time': []
        }
        
        # Mixed precision training (AMP)
        self.use_amp = config.use_amp if hasattr(config, 'use_amp') else True
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp and self.device == 'cuda' else None
        
        # Setup optimizer and scheduler
        self._setup_optimizer()
        self._setup_scheduler()
        
        # Setup loss function
        self._setup_loss_function()
        
        # Setup logging
        self._setup_logging()
        
        # Calculate effective batch size and accumulation steps
        self._calculate_accumulation_steps()
        
        self.logger.info(f"nnMILTrainer initialized on {self.device}")
        self.logger.info(f"Effective batch size: {self.effective_batch_size}")
        self.logger.info(f"Gradient accumulation steps: {self.accumulation_steps}")
        if self.use_amp:
            self.logger.info("Mixed precision training (AMP) enabled")
    
    def _setup_optimizer(self):
        """Setup optimizer with learning rate scaling."""
        # Scale learning rate based on batch size
        base_lr = self.config.learning_rate
        batch_size = self.config.batch_size
        
        # Learning rate scaling: lr_scaled = lr_base * sqrt(batch_size)
        # This maintains training dynamics across different batch sizes
        lr_scale_factor = (batch_size / 32) ** 0.5  # 32 is reference batch size
        scaled_lr = base_lr * lr_scale_factor
        
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=scaled_lr,
            weight_decay=self.config.weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        self.logger.info(f"Optimizer: AdamW with scaled LR {scaled_lr:.2e} (base: {base_lr:.2e})")
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler."""
        # Cosine annealing with warm restarts
        self.scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=10,  # Initial restart period
            T_mult=2,  # Period multiplier
            eta_min=self.config.learning_rate * 0.01  # Minimum LR
        )
    
    def _setup_loss_function(self):
        """Setup loss function based on task type."""
        if self.config.task_type == 'classification':
            if self.config.class_weights is not None:
                class_weights = torch.tensor(
                    self.config.class_weights, dtype=torch.float32, device=self.device
                )
                self.criterion = nn.CrossEntropyLoss(weight=class_weights)
            else:
                self.criterion = nn.CrossEntropyLoss()
        
        elif self.config.task_type == 'regression':
            self.criterion = nn.MSELoss()
        
        elif self.config.task_type == 'survival':
            # Cox proportional hazards loss (simplified implementation)
            self.criterion = self._cox_loss
        
        else:
            raise ValueError(f"Unknown task_type: {self.config.task_type}")
    
    def _cox_loss(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Simplified Cox proportional hazards loss.
        
        Args:
            logits: Model predictions [B]
            targets: Survival targets [B, 2] with (time, event)
        
        Returns:
            Cox loss value
        """
        # Extract survival times and events
        times = targets[:, 0]
        events = targets[:, 1]
        
        # Sort by survival time (descending)
        sorted_indices = torch.argsort(times, descending=True)
        sorted_logits = logits[sorted_indices]
        sorted_events = events[sorted_indices]
        
        # Compute risk scores
        risk_scores = torch.exp(sorted_logits)
        
        # Compute cumulative hazard
        cumulative_hazard = torch.cumsum(risk_scores, dim=0)
        
        # Compute log partial likelihood
        log_likelihood = sorted_logits - torch.log(cumulative_hazard + 1e-8)
        
        # Only include observed events
        observed_log_likelihood = log_likelihood * sorted_events
        
        # Return negative log likelihood
        return -observed_log_likelihood.sum() / (sorted_events.sum() + 1e-8)
    
    def _setup_logging(self):
        """Setup TensorBoard logging."""
        log_dir = Path("logs") / f"nnmil_{int(time.time())}"
        log_dir.mkdir(parents=True, exist_ok=True)
        
        self.writer = SummaryWriter(log_dir=log_dir)
        self.logger.info(f"TensorBoard logging to: {log_dir}")
    
    def _calculate_accumulation_steps(self):
        """Calculate gradient accumulation steps for memory management."""
        # Estimate GPU memory requirements
        # This is a simplified heuristic - would need actual profiling
        
        # Assume each sample uses ~100MB (rough estimate for large bags)
        estimated_memory_per_sample = 100  # MB
        available_memory = 8000  # MB (8GB GPU, conservative estimate)
        
        max_batch_size = available_memory // estimated_memory_per_sample
        max_batch_size = max(1, min(max_batch_size, 16))  # Clamp to reasonable range
        
        # Calculate accumulation steps
        desired_batch_size = self.config.batch_size
        if desired_batch_size <= max_batch_size:
            self.accumulation_steps = 1
            self.effective_batch_size = desired_batch_size
        else:
            self.accumulation_steps = (desired_batch_size + max_batch_size - 1) // max_batch_size
            self.effective_batch_size = self.accumulation_steps * max_batch_size
        
        # Update config batch size to actual batch size per step
        self.actual_batch_size = min(desired_batch_size, max_batch_size)
    
    def train(
        self,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        checkpoint_dir: Optional[Union[str, Path]] = None
    ) -> Dict[str, List[float]]:
        """
        Train the nnMIL model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            checkpoint_dir: Directory to save checkpoints
        
        Returns:
            Training history dictionary
        """
        if checkpoint_dir is not None:
            checkpoint_dir = Path(checkpoint_dir)
            checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger.info("Starting nnMIL training")
        self.logger.info(f"Training for {self.config.num_epochs} epochs")
        
        for epoch in range(self.config.num_epochs):
            epoch_start_time = time.time()
            
            # Training phase
            train_loss = self._train_epoch(train_loader, epoch)
            
            # Validation phase
            val_loss = None
            val_metric = None
            if val_loader is not None:
                val_loss, val_metric = self._validate_epoch(val_loader, epoch)
            
            # Update learning rate
            self.scheduler.step()
            current_lr = self.optimizer.param_groups[0]['lr']
            
            # Record epoch time
            epoch_time = time.time() - epoch_start_time
            
            # Update history
            self.training_history['train_loss'].append(train_loss)
            if val_loss is not None:
                self.training_history['val_loss'].append(val_loss)
            if val_metric is not None:
                self.training_history['val_metric'].append(val_metric)
            self.training_history['learning_rate'].append(current_lr)
            self.training_history['epoch_time'].append(epoch_time)
            
            # Log epoch results
            self._log_epoch_results(epoch, train_loss, val_loss, val_metric, current_lr, epoch_time)
            
            # Check for improvement and save checkpoint
            improved = False
            if val_metric is not None and val_metric > self.best_metric:
                self.best_metric = val_metric
                self.patience_counter = 0
                improved = True
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            if checkpoint_dir is not None:
                self._save_checkpoint(checkpoint_dir, epoch, improved)
            
            # Early stopping
            if self.patience_counter >= self.config.patience:
                self.logger.info(f"Early stopping at epoch {epoch + 1}")
                break
            
            self.current_epoch = epoch + 1
        
        self.logger.info("Training completed")
        self.writer.close()
        
        return self.training_history
    
    def _train_epoch(self, train_loader: DataLoader, epoch: int) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        self.optimizer.zero_grad()
        
        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Forward pass with autocast
            with torch.cuda.amp.autocast(enabled=self.use_amp):
                if hasattr(self.model, 'forward_with_attention'):
                    logits, attention_weights = self.model.forward_with_attention(
                        batch.features, batch.masks
                    )
                else:
                    logits = self.model(batch.features, batch.masks)
                
                # Compute loss
                loss = self.criterion(logits, batch.labels)
                
                # Scale loss for gradient accumulation
                loss = loss / self.accumulation_steps
            
            # Backward pass with scaler
            if self.scaler is not None:
                self.scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % self.accumulation_steps == 0:
                # Gradient clipping
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                
                # Optimizer step
                if self.scaler is not None:
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    self.optimizer.step()
                
                self.optimizer.zero_grad()
                
                # Log batch metrics
                if (batch_idx + 1) % (self.config.log_interval * self.accumulation_steps) == 0:
                    self._log_batch_metrics(epoch, batch_idx, loss.item() * self.accumulation_steps)
            
            total_loss += loss.item() * self.accumulation_steps
            num_batches += 1
        
        # Handle remaining gradients
        if num_batches % self.accumulation_steps != 0:
            if self.scaler is not None:
                self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            if self.scaler is not None:
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                self.optimizer.step()
            self.optimizer.zero_grad()
        
        return total_loss / num_batches
    
    def _validate_epoch(self, val_loader: DataLoader, epoch: int) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch in val_loader:
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Forward pass
                if hasattr(self.model, 'forward_with_attention'):
                    logits, attention_weights = self.model.forward_with_attention(
                        batch.features, batch.masks
                    )
                else:
                    logits = self.model(batch.features, batch.masks)
                
                # Compute loss
                loss = self.criterion(logits, batch.labels)
                total_loss += loss.item()
                
                # Collect predictions and targets
                all_predictions.append(logits.cpu())
                all_targets.append(batch.labels.cpu())
        
        # Compute validation metric
        all_predictions = torch.cat(all_predictions, dim=0)
        all_targets = torch.cat(all_targets, dim=0)
        
        val_metric = self._compute_validation_metric(all_predictions, all_targets)
        
        return total_loss / len(val_loader), val_metric
    
    def _compute_validation_metric(self, predictions: torch.Tensor, targets: torch.Tensor) -> float:
        """Compute validation metric based on task type."""
        if self.config.task_type == 'classification':
            # Compute accuracy
            pred_classes = torch.argmax(predictions, dim=1)
            accuracy = (pred_classes == targets).float().mean().item()
            return accuracy
        
        elif self.config.task_type == 'regression':
            # Compute R²
            ss_res = ((targets - predictions.squeeze()) ** 2).sum()
            ss_tot = ((targets - targets.mean()) ** 2).sum()
            r2 = 1 - (ss_res / (ss_tot + 1e-8))
            return r2.item()
        
        elif self.config.task_type == 'survival':
            # Compute concordance index (simplified)
            # This is a placeholder - would need proper C-index implementation
            return 0.5  # Placeholder
        
        else:
            return 0.0
    
    def _move_batch_to_device(self, batch: TrainingBatch) -> TrainingBatch:
        """Move batch to training device."""
        return TrainingBatch(
            features=batch.features.to(self.device),
            labels=batch.labels.to(self.device),
            masks=batch.masks.to(self.device),
            num_patches=batch.num_patches.to(self.device),
            slide_ids=batch.slide_ids
        )
    
    def _log_batch_metrics(self, epoch: int, batch_idx: int, loss: float):
        """Log batch-level metrics."""
        global_step = epoch * 1000 + batch_idx  # Approximate global step
        
        # Log to TensorBoard
        self.writer.add_scalar('Train/BatchLoss', loss, global_step)
        self.writer.add_scalar('Train/LearningRate', self.optimizer.param_groups[0]['lr'], global_step)
        
        # Log gradient norm
        total_norm = 0
        for p in self.model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        self.writer.add_scalar('Train/GradientNorm', total_norm, global_step)
        
        # Log GPU memory usage
        if torch.cuda.is_available():
            memory_used = torch.cuda.memory_allocated() / 1024**3  # GB
            self.writer.add_scalar('System/GPUMemoryGB', memory_used, global_step)
    
    def _log_epoch_results(
        self,
        epoch: int,
        train_loss: float,
        val_loss: Optional[float],
        val_metric: Optional[float],
        learning_rate: float,
        epoch_time: float
    ):
        """Log epoch-level results."""
        # Console logging
        log_msg = f"Epoch {epoch + 1}/{self.config.num_epochs} - "
        log_msg += f"Train Loss: {train_loss:.4f}, "
        if val_loss is not None:
            log_msg += f"Val Loss: {val_loss:.4f}, "
        if val_metric is not None:
            log_msg += f"Val Metric: {val_metric:.4f}, "
        log_msg += f"LR: {learning_rate:.2e}, "
        log_msg += f"Time: {epoch_time:.1f}s"
        
        self.logger.info(log_msg)
        
        # TensorBoard logging
        self.writer.add_scalar('Train/EpochLoss', train_loss, epoch)
        if val_loss is not None:
            self.writer.add_scalar('Validation/Loss', val_loss, epoch)
        if val_metric is not None:
            self.writer.add_scalar('Validation/Metric', val_metric, epoch)
        self.writer.add_scalar('Train/LearningRate', learning_rate, epoch)
        self.writer.add_scalar('System/EpochTime', epoch_time, epoch)
    
    def _save_checkpoint(self, checkpoint_dir: Path, epoch: int, is_best: bool):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_metric': self.best_metric,
            'config': self.config.to_dict(),
            'training_history': self.training_history
        }
        
        # Save regular checkpoint
        if (epoch + 1) % self.config.checkpoint_interval == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pth"
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"Checkpoint saved: {checkpoint_path}")
        
        # Save best model
        if is_best and self.config.save_best_only:
            best_path = checkpoint_dir / "best_model.pth"
            torch.save(checkpoint, best_path)
            self.logger.info(f"Best model saved: {best_path}")
    
    def load_checkpoint(self, checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
        """
        Load model checkpoint.
        
        Args:
            checkpoint_path: Path to checkpoint file
        
        Returns:
            Checkpoint dictionary
        """
        checkpoint_path = Path(checkpoint_path)
        
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load optimizer state
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        # Load scheduler state
        if 'scheduler_state_dict' in checkpoint:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        # Load training state
        self.current_epoch = checkpoint.get('epoch', 0) + 1
        self.best_metric = checkpoint.get('best_metric', float('-inf'))
        self.training_history = checkpoint.get('training_history', {
            'train_loss': [], 'val_loss': [], 'val_metric': [],
            'learning_rate': [], 'epoch_time': []
        })
        
        self.logger.info(f"Checkpoint loaded from: {checkpoint_path}")
        self.logger.info(f"Resuming from epoch {self.current_epoch}")
        
        return checkpoint
    
    def get_training_stats(self) -> Dict[str, Any]:
        """Get training statistics."""
        if not self.training_history['train_loss']:
            return {}
        
        final_val_metric = (
            self.training_history['val_metric'][-1]
            if self.training_history['val_metric']
            else None
        )
        
        return {
            'total_epochs': len(self.training_history['train_loss']),
            'best_metric': self.best_metric,
            'final_train_loss': self.training_history['train_loss'][-1],
            'final_val_loss': self.training_history['val_loss'][-1] if self.training_history['val_loss'] else None,
            'final_val_metric': final_val_metric,
            'total_training_time': sum(self.training_history['epoch_time']),
            'avg_epoch_time': sum(self.training_history['epoch_time']) / len(self.training_history['epoch_time'])
        }