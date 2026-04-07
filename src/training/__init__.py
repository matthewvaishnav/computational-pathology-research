"""
Supervised training module for multimodal and baseline models.

This module provides a comprehensive trainer for:
- Multimodal fusion models
- Baseline models (single modality, late fusion)
- Classification and survival prediction tasks

Features:
- Automatic mixed precision (AMP) support
- Learning rate scheduling
- Early stopping
- Checkpoint management
- TensorBoard logging
- Cross-validation support
"""

import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from src.utils.monitoring import get_logger

logger = get_logger(__name__)


class MetricsComputer:
    """
    Compute and track classification metrics.

    Supports binary and multi-class classification with
    metrics like accuracy, AUC, F1, precision, and recall.
    """

    def __init__(self, num_classes: int, task_type: str = "classification"):
        """
        Initialize metrics computer.

        Args:
            num_classes: Number of classes (2 for binary, >2 for multi-class)
            task_type: Type of task ('classification' or 'survival')
        """
        self.num_classes = num_classes
        self.task_type = task_type
        self.reset()

    def reset(self):
        """Reset all accumulated metrics."""
        self.all_preds = []
        self.all_probs = []
        self.all_labels = []
        self.total_loss = 0.0
        self.num_samples = 0

    def update(self, logits: torch.Tensor, labels: torch.Tensor, loss: float):
        """
        Update metrics with a batch of predictions.

        Args:
            logits: Model output logits [batch_size, num_classes]
            labels: Ground truth labels [batch_size]
            loss: Batch loss value
        """
        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(logits, dim=1)

        self.all_preds.extend(preds.cpu().numpy())
        self.all_probs.extend(probs.cpu().numpy())
        self.all_labels.extend(labels.cpu().numpy())

        batch_size = labels.size(0)
        self.total_loss += loss * batch_size
        self.num_samples += batch_size

    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics from accumulated predictions.

        Returns:
            Dictionary of metric names to values
        """
        if len(self.all_labels) == 0:
            return {}

        labels = np.array(self.all_labels)
        preds = np.array(self.all_preds)
        probs = np.array(self.all_probs)

        metrics = {
            "loss": self.total_loss / max(self.num_samples, 1),
            "accuracy": accuracy_score(labels, preds),
        }

        # Binary classification metrics
        if self.num_classes == 2:
            try:
                metrics["auc"] = roc_auc_score(labels, probs[:, 1])
                metrics["average_precision"] = average_precision_score(labels, probs[:, 1])
            except ValueError:
                # Handle case where only one class present
                metrics["auc"] = 0.5
                metrics["average_precision"] = 0.5

            metrics["f1"] = f1_score(labels, preds, zero_division=0)
            metrics["precision"] = precision_score(labels, preds, zero_division=0)
            metrics["recall"] = recall_score(labels, preds, zero_division=0)

        # Multi-class metrics
        else:
            try:
                # One-vs-rest AUC
                metrics["auc"] = roc_auc_score(labels, probs, multi_class="ovr", average="macro")
            except ValueError:
                metrics["auc"] = 0.5

            metrics["f1_macro"] = f1_score(labels, preds, average="macro", zero_division=0)
            metrics["f1_weighted"] = f1_score(labels, preds, average="weighted", zero_division=0)

            # Per-class F1
            per_class_f1 = f1_score(labels, preds, average=None, zero_division=0)
            for i, f1 in enumerate(per_class_f1):
                metrics[f"f1_class_{i}"] = f1

        return metrics


class SupervisedTrainer:
    """
    Supervised trainer for multimodal and baseline pathology models.

    Supports training with automatic mixed precision, gradient clipping,
    learning rate scheduling, and comprehensive logging.

    Example:
        >>> model = MultimodalFusionModel(embed_dim=256)
        >>> classifier = ClassificationHead(input_dim=256, num_classes=4)
        >>> trainer = SupervisedTrainer(
        ...     model=model,
        ...     task_head=classifier,
        ...     num_classes=4,
        ...     device="cuda"
        ... )
        >>> history = trainer.fit(train_loader, num_epochs=100, val_loader=val_loader)
        >>> history_no_val = trainer.fit(train_loader, num_epochs=10)
    """

    def __init__(
        self,
        model: nn.Module,
        task_head: nn.Module,
        num_classes: int,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
        use_amp: bool = True,
        grad_clip_norm: Optional[float] = 1.0,
        scheduler_type: str = "cosine",
        warmup_epochs: int = 5,
        early_stopping_patience: int = 15,
        checkpoint_dir: Optional[str] = None,
        log_dir: Optional[str] = None,
    ):
        """
        Initialize supervised trainer.

        Args:
            model: Backbone model (MultimodalFusionModel or baseline)
            task_head: Task-specific head (ClassificationHead, etc.)
            num_classes: Number of output classes
            device: Device to train on ('cuda' or 'cpu')
            learning_rate: Initial learning rate
            weight_decay: Weight decay for optimizer
            use_amp: Whether to use automatic mixed precision
            grad_clip_norm: Gradient clipping norm (None to disable)
            scheduler_type: LR scheduler ('cosine', 'step', 'plateau')
            warmup_epochs: Number of warmup epochs
            early_stopping_patience: Epochs to wait before early stopping
            checkpoint_dir: Directory to save checkpoints
            log_dir: Directory for TensorBoard logs
        """
        self.model = model.to(device)
        self.task_head = task_head.to(device)
        self.num_classes = num_classes
        self.device = device
        self.use_amp = use_amp and torch.cuda.is_available()
        self.grad_clip_norm = grad_clip_norm

        # Optimizer
        self.optimizer = optim.AdamW(
            list(model.parameters()) + list(task_head.parameters()),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Loss function
        self.criterion = nn.CrossEntropyLoss()

        # Metrics
        self.train_metrics = MetricsComputer(num_classes)
        self.val_metrics = MetricsComputer(num_classes)

        # AMP scaler
        self.scaler = GradScaler() if self.use_amp else None

        # Checkpointing
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
        if self.checkpoint_dir:
            self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard logging
        self.writer = SummaryWriter(log_dir) if log_dir else None

        # Early stopping
        self.early_stopping_patience = early_stopping_patience
        self.best_val_metric = 0.0
        self.epochs_without_improvement = 0

        # Training history
        self.history = {
            "train_loss": [],
            "train_acc": [],
            "val_loss": [],
            "val_acc": [],
            "val_auc": [],
            "learning_rate": [],
        }

        self.current_epoch = 0
        self.global_step = 0

    def _train_epoch(self, train_loader: DataLoader) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.task_head.train()
        self.train_metrics.reset()

        epoch_start = time.time()

        for batch_idx, batch in enumerate(train_loader):
            # Move batch to device
            batch = self._move_to_device(batch)
            labels = batch.pop("label")

            # Forward pass with AMP
            self.optimizer.zero_grad()

            with autocast(enabled=self.use_amp):
                embeddings = self.model(batch)
                logits = self.task_head(embeddings)
                loss = self.criterion(logits, labels)

            # Backward pass
            if self.use_amp:
                self.scaler.scale(loss).backward()

                if self.grad_clip_norm:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.parameters()) + list(self.task_head.parameters()),
                        self.grad_clip_norm,
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()

                if self.grad_clip_norm:
                    torch.nn.utils.clip_grad_norm_(
                        list(self.model.parameters()) + list(self.task_head.parameters()),
                        self.grad_clip_norm,
                    )

                self.optimizer.step()

            # Update metrics
            self.train_metrics.update(logits.detach(), labels, loss.item())
            self.global_step += 1

            # Log to TensorBoard every 10 batches
            if self.writer and batch_idx % 10 == 0:
                self.writer.add_scalar("train/batch_loss", loss.item(), self.global_step)

        metrics = self.train_metrics.compute()
        metrics["epoch_time"] = time.time() - epoch_start

        return metrics

    @torch.no_grad()
    def _validate(self, val_loader: DataLoader) -> Dict[str, float]:
        """Validate on validation set."""
        self.model.eval()
        self.task_head.eval()
        self.val_metrics.reset()

        for batch in val_loader:
            batch = self._move_to_device(batch)
            labels = batch.pop("label")

            with autocast(enabled=self.use_amp):
                embeddings = self.model(batch)
                logits = self.task_head(embeddings)
                loss = self.criterion(logits, labels)

            self.val_metrics.update(logits, labels, loss.item())

        return self.val_metrics.compute()

    def _move_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch tensors to device."""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            else:
                device_batch[key] = value
        return device_batch

    def _log_epoch(self, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Log epoch results."""
        logger.info(
            f"Epoch {self.current_epoch}: "
            f"train_loss={train_metrics['loss']:.4f}, "
            f"train_acc={train_metrics['accuracy']:.4f}, "
            f"val_loss={val_metrics['loss']:.4f}, "
            f"val_acc={val_metrics['accuracy']:.4f}, "
            f"val_auc={val_metrics.get('auc', 0):.4f}"
        )

        if self.writer:
            self.writer.add_scalar("train/loss", train_metrics["loss"], self.current_epoch)
            self.writer.add_scalar("train/accuracy", train_metrics["accuracy"], self.current_epoch)
            self.writer.add_scalar("val/loss", val_metrics["loss"], self.current_epoch)
            self.writer.add_scalar("val/accuracy", val_metrics["accuracy"], self.current_epoch)

            if "auc" in val_metrics:
                self.writer.add_scalar("val/auc", val_metrics["auc"], self.current_epoch)
            if "f1" in val_metrics:
                self.writer.add_scalar("val/f1", val_metrics["f1"], self.current_epoch)

    def _save_checkpoint(self, is_best: bool = False):
        """Save model checkpoint."""
        if not self.checkpoint_dir:
            return

        checkpoint = {
            "epoch": self.current_epoch,
            "model_state_dict": self.model.state_dict(),
            "task_head_state_dict": self.task_head.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict() if self.scaler else None,
            "best_val_metric": self.best_val_metric,
            "history": self.history,
        }

        # Save latest checkpoint
        latest_path = self.checkpoint_dir / "checkpoint_latest.pth"
        torch.save(checkpoint, latest_path)

        # Save best checkpoint
        if is_best:
            best_path = self.checkpoint_dir / "checkpoint_best.pth"
            torch.save(checkpoint, best_path)
            logger.info(f"Saved best checkpoint to {best_path}")

    def fit(
        self,
        train_loader: DataLoader,
        num_epochs: int,
        val_loader: Optional[DataLoader] = None,
    ) -> Dict[str, List[float]]:
        """
        Train model for specified number of epochs.

        Args:
            train_loader: Training data loader
            num_epochs: Number of epochs to train
            val_loader: Validation data loader (optional, defaults to None)

        Returns:
            Training history dictionary
        """
        logger.info(f"Starting training for {num_epochs} epochs on {self.device}")

        if val_loader is None:
            logger.warning(
                "Training without validation - early stopping disabled, saving only latest checkpoint"
            )

        for epoch in range(num_epochs):
            self.current_epoch = epoch + 1

            # Train
            train_metrics = self._train_epoch(train_loader)

            # Validate (if validation data available)
            if val_loader is not None:
                val_metrics = self._validate(val_loader)

                # Log with validation metrics
                self._log_epoch(train_metrics, val_metrics)

                # Update history with validation
                self.history["val_loss"].append(val_metrics["loss"])
                self.history["val_acc"].append(val_metrics["accuracy"])
                self.history["val_auc"].append(val_metrics.get("auc", 0))

                # Check for improvement
                val_auc = val_metrics.get("auc", val_metrics["accuracy"])
                is_best = val_auc > self.best_val_metric

                if is_best:
                    self.best_val_metric = val_auc
                    self.epochs_without_improvement = 0
                else:
                    self.epochs_without_improvement += 1

                # Save checkpoint (best and latest)
                self._save_checkpoint(is_best=is_best)

                # Early stopping
                if self.epochs_without_improvement >= self.early_stopping_patience:
                    logger.info(
                        f"Early stopping triggered after {self.early_stopping_patience} "
                        f"epochs without improvement"
                    )
                    break
            else:
                # Log without validation metrics
                empty_val_metrics = {"loss": 0.0, "accuracy": 0.0}
                self._log_epoch(train_metrics, empty_val_metrics)

                # Update history (no validation)
                self.history["val_loss"].append(0.0)
                self.history["val_acc"].append(0.0)
                self.history["val_auc"].append(0.0)

                # Save only latest checkpoint (no best checkpoint without validation)
                self._save_checkpoint(is_best=False)

            # Update common history
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])
            self.history["learning_rate"].append(self.optimizer.param_groups[0]["lr"])

        logger.info(f"Training completed. Best val metric: {self.best_val_metric:.4f}")
        return self.history

    def load_checkpoint(self, checkpoint_path: str):
        """Load model from checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.task_head.load_state_dict(checkpoint["task_head_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if self.scaler and checkpoint.get("scaler_state_dict"):
            self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

        self.current_epoch = checkpoint["epoch"]
        self.best_val_metric = checkpoint["best_val_metric"]
        self.history = checkpoint["history"]

        logger.info(f"Loaded checkpoint from epoch {self.current_epoch}")

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate model on test set.

        Args:
            test_loader: Test data loader

        Returns:
            Dictionary of test metrics
        """
        self.model.eval()
        self.task_head.eval()

        test_metrics = MetricsComputer(self.num_classes)

        for batch in test_loader:
            batch = self._move_to_device(batch)
            labels = batch.pop("label")

            with autocast(enabled=self.use_amp):
                embeddings = self.model(batch)
                logits = self.task_head(embeddings)
                loss = self.criterion(logits, labels)

            test_metrics.update(logits, labels, loss.item())

        metrics = test_metrics.compute()

        logger.info("Test Results:")
        for metric_name, value in metrics.items():
            logger.info(f"  {metric_name}: {value:.4f}")

        return metrics
