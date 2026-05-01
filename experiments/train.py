"""
Main training script for multimodal fusion model.

This script provides a complete training pipeline for the multimodal pathology
fusion model with support for:
- Multiple modalities (WSI, genomic, clinical text)
- Missing modality handling
- Configurable hyperparameters via Hydra
- Checkpointing and early stopping
- TensorBoard logging
- Multiple task heads (classification, survival)
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.data import MultimodalDataset
from src.data.loaders import collate_multimodal
from src.data.prefetch import DataPrefetcher
from src.models import ClassificationHead, MultimodalFusionModel, SurvivalPredictionHead

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class MultimodalTrainer:
    """
    Trainer for multimodal fusion model.

    Handles the complete training loop including:
    - Forward/backward passes
    - Gradient clipping
    - Learning rate scheduling
    - Checkpointing
    - Validation
    - Logging

    Args:
        model: Multimodal fusion model
        task_head: Task-specific prediction head
        train_loader: Training data loader
        val_loader: Validation data loader
        config: Training configuration dictionary
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        log_dir: Directory for TensorBoard logs
    """

    def __init__(
        self,
        model: nn.Module,
        task_head: nn.Module,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: Dict,
        device: str = "cuda",
        checkpoint_dir: Optional[Path] = None,
        log_dir: Optional[Path] = None,
    ):
        self.model = model.to(device)
        self.task_head = task_head.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.device = device

        # Setup directories
        self.checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else Path("checkpoints")
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self.log_dir = Path(log_dir) if log_dir else Path("logs")
        self.log_dir.mkdir(parents=True, exist_ok=True)

        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.log_dir)

        # Optimizer
        self.optimizer = self._create_optimizer()

        # Learning rate scheduler
        self.scheduler = self._create_scheduler()

        # Loss function
        self.criterion = self._create_criterion()

        # Mixed precision training
        self.use_amp = config.get("use_amp", False) and device == "cuda"
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        if self.use_amp:
            logger.info("Mixed precision training (AMP) enabled")

        # Gradient accumulation
        self.accumulation_steps = config.get("accumulation_steps", 1)
        if self.accumulation_steps > 1:
            logger.info(f"Gradient accumulation enabled: {self.accumulation_steps} steps")
            logger.info(
                f"Effective batch size: {config.get('batch_size', 16) * self.accumulation_steps}"
            )

        # Data prefetching
        self.use_prefetch = config.get("use_prefetch", True) and device == "cuda"
        if self.use_prefetch:
            logger.info("Data prefetching enabled for improved I/O performance")

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_metric = (
            float("-inf") if config.get("maximize_metric", True) else float("inf")
        )
        self.patience_counter = 0

        # Log model info
        total_params = sum(p.numel() for p in model.parameters()) + sum(
            p.numel() for p in task_head.parameters()
        )
        logger.info(f"Total parameters: {total_params:,}")

    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config."""
        optimizer_name = self.config.get("optimizer", "adamw")
        lr = self.config.get("learning_rate", 1e-4)
        weight_decay = self.config.get("weight_decay", 0.01)

        params = list(self.model.parameters()) + list(self.task_head.parameters())

        if optimizer_name.lower() == "adamw":
            return optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == "adam":
            return optim.Adam(params, lr=lr, weight_decay=weight_decay)
        elif optimizer_name.lower() == "sgd":
            momentum = self.config.get("momentum", 0.9)
            return optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
        else:
            raise ValueError(f"Unknown optimizer: {optimizer_name}")

    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler based on config."""
        scheduler_name = self.config.get("scheduler", "cosine")

        if scheduler_name is None or scheduler_name.lower() == "none":
            return None

        if scheduler_name.lower() == "cosine":
            T_max = self.config.get("num_epochs", 100) * len(self.train_loader)
            eta_min = self.config.get("min_lr", 1e-6)
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=T_max, eta_min=eta_min
            )
        elif scheduler_name.lower() == "step":
            step_size = self.config.get("step_size", 30)
            gamma = self.config.get("gamma", 0.1)
            return optim.lr_scheduler.StepLR(self.optimizer, step_size=step_size, gamma=gamma)
        elif scheduler_name.lower() == "plateau":
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, mode="max", factor=0.5, patience=5, verbose=True
            )
        else:
            raise ValueError(f"Unknown scheduler: {scheduler_name}")

    def _create_criterion(self) -> nn.Module:
        """Create loss function based on task type."""
        task_type = self.config.get("task_type", "classification")

        if task_type == "classification":
            num_classes = self.config.get("num_classes", 2)
            if num_classes == 1:
                return nn.BCEWithLogitsLoss()
            else:
                return nn.CrossEntropyLoss()
        elif task_type == "survival":
            return nn.MSELoss()  # Placeholder - use proper survival loss
        else:
            raise ValueError(f"Unknown task type: {task_type}")

    def _compute_loss(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Compute task loss for the configured classification mode."""
        if self.config.get("task_type", "classification") == "classification":
            if self.config.get("num_classes", 2) == 1:
                return self.criterion(logits.view(-1), labels.float())
        return self.criterion(logits, labels)

    def _get_classification_outputs(
        self, logits: torch.Tensor
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """Return class predictions and positive-class probabilities when available."""
        num_classes = self.config.get("num_classes", 2)

        if num_classes == 1:
            positive_probs = torch.sigmoid(logits.view(-1)).detach().cpu().numpy()
            preds = (positive_probs > 0.5).astype(int)
            return preds, positive_probs

        probs = torch.softmax(logits, dim=1).detach().cpu().numpy()
        preds = torch.argmax(logits, dim=1).detach().cpu().numpy()
        positive_probs = probs[:, 1] if num_classes == 2 else None
        return preds, positive_probs

    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.task_head.train()

        total_loss = 0.0
        all_preds = []
        all_labels = []

        # Use prefetcher if enabled
        if self.use_prefetch:
            data_iter = DataPrefetcher(self.train_loader, self.device)
        else:
            data_iter = self.train_loader

        pbar = tqdm(data_iter, desc=f"Epoch {self.current_epoch}")

        for batch_idx, batch in enumerate(pbar):
            # Move batch to device (if not using prefetcher)
            if not self.use_prefetch:
                batch = self._batch_to_device(batch)
            labels = batch.pop("label")

            # Forward pass with AMP
            if self.use_amp:
                with torch.cuda.amp.autocast():
                    embeddings = self.model(batch)
                    logits = self.task_head(embeddings)
                    loss = self._compute_loss(logits, labels)
                
                # Scale loss for gradient accumulation
                loss = loss / self.accumulation_steps
                
                # Backward pass with gradient scaling
                self.scaler.scale(loss).backward()
                
                # Optimizer step with gradient accumulation
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    # Gradient clipping
                    max_grad_norm = self.config.get("max_grad_norm", 1.0)
                    if max_grad_norm > 0:
                        self.scaler.unscale_(self.optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            list(self.model.parameters()) + list(self.task_head.parameters()),
                            max_norm=max_grad_norm,
                        )
                    
                    # Optimizer step with scaler
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    self.optimizer.zero_grad()
            else:
                # Standard training without AMP
                embeddings = self.model(batch)
                logits = self.task_head(embeddings)
                loss = self._compute_loss(logits, labels)
                
                # Scale loss for gradient accumulation
                loss = loss / self.accumulation_steps
                
                # Backward pass
                loss.backward()
                
                # Optimizer step with gradient accumulation
                if (batch_idx + 1) % self.accumulation_steps == 0:
                    # Gradient clipping
                    max_grad_norm = self.config.get("max_grad_norm", 1.0)
                    if max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(
                            list(self.model.parameters()) + list(self.task_head.parameters()),
                            max_norm=max_grad_norm,
                        )
                    
                    self.optimizer.step()
                    self.optimizer.zero_grad()

            # Update learning rate (only on actual optimizer steps)
            if (batch_idx + 1) % self.accumulation_steps == 0:
                if self.scheduler is not None and not isinstance(
                    self.scheduler, optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step()

            # Accumulate metrics (use unscaled loss for logging)
            total_loss += loss.item() * self.accumulation_steps

            # Get predictions
            if self.config.get("task_type") == "classification":
                preds, _ = self._get_classification_outputs(logits)
            else:
                preds = logits.detach().cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())

            # Update progress bar
            pbar.set_postfix({"loss": loss.item() * self.accumulation_steps})

            # Log to TensorBoard (only on actual optimizer steps)
            if (batch_idx + 1) % self.accumulation_steps == 0:
                if batch_idx % self.config.get("log_interval", 10) == 0:
                    self.writer.add_scalar(
                        "train/loss", loss.item() * self.accumulation_steps, self.global_step
                    )
                    self.writer.add_scalar(
                        "train/lr", self.optimizer.param_groups[0]["lr"], self.global_step
                    )
                self.global_step += 1

        # Compute epoch metrics
        avg_loss = total_loss / len(self.train_loader)
        metrics = {"loss": avg_loss}

        if self.config.get("task_type") == "classification":
            metrics["accuracy"] = accuracy_score(all_labels, all_preds)
            metrics["f1"] = f1_score(all_labels, all_preds, average="weighted")

        return metrics

    def validate(self) -> Dict[str, float]:
        """Validate the model."""
        self.model.eval()
        self.task_head.eval()

        total_loss = 0.0
        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validation"):
                batch = self._batch_to_device(batch)
                labels = batch.pop("label")

                # Forward pass
                embeddings = self.model(batch)
                logits = self.task_head(embeddings)

                # Compute loss
                loss = self._compute_loss(logits, labels)
                total_loss += loss.item()

                # Get predictions
                if self.config.get("task_type") == "classification":
                    preds, positive_probs = self._get_classification_outputs(logits)
                    if positive_probs is not None:
                        all_probs.extend(positive_probs)
                else:
                    preds = logits.cpu().numpy()

                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())

        # Compute metrics
        avg_loss = total_loss / len(self.val_loader)
        metrics = {"val_loss": avg_loss}

        if self.config.get("task_type") == "classification":
            metrics["val_accuracy"] = accuracy_score(all_labels, all_preds)
            metrics["val_f1"] = f1_score(all_labels, all_preds, average="weighted")

            # Compute AUC if binary classification
            if self.config.get("num_classes", 2) in {1, 2} and all_probs:
                metrics["val_auc"] = roc_auc_score(all_labels, all_probs)

        return metrics

    def train(self):
        """Main training loop."""
        num_epochs = self.config.get("num_epochs", 100)
        patience = self.config.get("patience", 10)
        save_interval = self.config.get("save_interval", 5)

        logger.info(f"Starting training for {num_epochs} epochs")
        logger.info(f"Training samples: {len(self.train_loader.dataset)}")
        logger.info(f"Validation samples: {len(self.val_loader.dataset)}")

        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch

            logger.info(f"\n{'='*60}")
            logger.info(f"Epoch {epoch}/{num_epochs}")
            logger.info(f"{'='*60}")

            # Train
            train_metrics = self.train_epoch()
            logger.info(f"Training metrics: {train_metrics}")

            # Log to TensorBoard
            for key, value in train_metrics.items():
                self.writer.add_scalar(f"epoch/{key}", value, epoch)

            # Validate
            val_metrics = self.validate()
            logger.info(f"Validation metrics: {val_metrics}")

            # Log to TensorBoard
            for key, value in val_metrics.items():
                self.writer.add_scalar(f"epoch/{key}", value, epoch)

            # Update scheduler if using ReduceLROnPlateau
            if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                metric_name = self.config.get("monitor_metric", "val_accuracy")
                self.scheduler.step(val_metrics.get(metric_name, val_metrics["val_loss"]))

            # Save checkpoint
            if epoch % save_interval == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pth", val_metrics)

            # Check for improvement
            metric_name = self.config.get("monitor_metric", "val_accuracy")
            current_metric = val_metrics.get(metric_name, val_metrics["val_loss"])

            maximize = self.config.get("maximize_metric", True)
            improved = (
                (current_metric > self.best_val_metric)
                if maximize
                else (current_metric < self.best_val_metric)
            )

            if improved:
                self.best_val_metric = current_metric
                self.patience_counter = 0
                self.save_checkpoint("best_model.pth", val_metrics)
                logger.info(f"✓ New best {metric_name}: {current_metric:.4f}")
            else:
                self.patience_counter += 1
                logger.info(
                    f"No improvement for {self.patience_counter} epochs "
                    f"(best {metric_name}: {self.best_val_metric:.4f})"
                )

            # Early stopping
            if self.patience_counter >= patience:
                logger.info(f"Early stopping triggered after {epoch} epochs")
                break

        logger.info("\nTraining complete!")
        logger.info(f"Best {metric_name}: {self.best_val_metric:.4f}")
        self.writer.close()

    def save_checkpoint(self, filename: str, metrics: Dict[str, float]):
        """Save model checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename

        torch.save(
            {
                "epoch": self.current_epoch,
                "global_step": self.global_step,
                "model_state_dict": self.model.state_dict(),
                "task_head_state_dict": self.task_head.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "scheduler_state_dict": self.scheduler.state_dict() if self.scheduler else None,
                "metrics": metrics,
                "config": self.config,
            },
            checkpoint_path,
        )

        logger.info(f"Saved checkpoint to {checkpoint_path}")

    def load_checkpoint(self, checkpoint_path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.task_head.load_state_dict(checkpoint["task_head_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if checkpoint.get("scheduler_state_dict") and self.scheduler:
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.current_epoch = checkpoint.get("epoch", 0)
        self.global_step = checkpoint.get("global_step", 0)

        logger.info(f"Loaded checkpoint from {checkpoint_path}")
        logger.info(f"Resuming from epoch {self.current_epoch}")

    def _batch_to_device(self, batch: Dict) -> Dict:
        """Move batch tensors to device."""
        return {
            k: v.to(self.device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
        }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train multimodal fusion model")

    # Data arguments
    parser.add_argument("--data-dir", type=str, default="./data", help="Path to data directory")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size for training")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loading workers")

    # Model arguments
    parser.add_argument("--embed-dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument(
        "--num-classes", type=int, default=4, help="Number of classes for classification"
    )
    parser.add_argument(
        "--task-type",
        type=str,
        default="classification",
        choices=["classification", "survival"],
        help="Task type",
    )

    # Training arguments
    parser.add_argument("--num-epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--learning-rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument(
        "--optimizer", type=str, default="adamw", choices=["adamw", "adam", "sgd"], help="Optimizer"
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="cosine",
        choices=["cosine", "step", "plateau", "none"],
        help="Learning rate scheduler",
    )
    parser.add_argument(
        "--max-grad-norm", type=float, default=1.0, help="Maximum gradient norm for clipping"
    )
    parser.add_argument(
        "--accumulation-steps",
        type=int,
        default=1,
        help="Gradient accumulation steps (effective batch size = batch_size * accumulation_steps)",
    )

    # Checkpoint arguments
    parser.add_argument(
        "--checkpoint-dir", type=str, default="./checkpoints", help="Directory to save checkpoints"
    )
    parser.add_argument(
        "--log-dir", type=str, default="./logs", help="Directory for TensorBoard logs"
    )
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )
    parser.add_argument(
        "--save-interval", type=int, default=5, help="Save checkpoint every N epochs"
    )

    # Early stopping
    parser.add_argument("--patience", type=int, default=10, help="Patience for early stopping")
    parser.add_argument(
        "--monitor-metric",
        type=str,
        default="val_accuracy",
        help="Metric to monitor for early stopping",
    )

    # Other
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device to train on")
    parser.add_argument(
        "--use-amp",
        action="store_true",
        help="Use mixed precision training (AMP) for faster training and lower memory usage",
    )
    parser.add_argument(
        "--use-gradient-checkpointing",
        action="store_true",
        help="Use gradient checkpointing to save memory (trades compute for memory)",
    )

    return parser.parse_args()


def main():
    """Main training function."""
    args = parse_args()

    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Create config dict
    config = vars(args)

    # Initialize model
    logger.info("Initializing model...")
    model = MultimodalFusionModel(embed_dim=args.embed_dim)
    
    # Enable gradient checkpointing if requested
    if args.use_gradient_checkpointing:
        logger.info("Enabling gradient checkpointing...")
        model.enable_gradient_checkpointing()

    # Initialize task head
    if args.task_type == "classification":
        task_head = ClassificationHead(input_dim=args.embed_dim, num_classes=args.num_classes)
    else:
        task_head = SurvivalPredictionHead(input_dim=args.embed_dim)

    # Create data loaders
    logger.info("Loading data...")
    try:
        train_dataset = MultimodalDataset(data_dir=args.data_dir, split="train", config=config)
        val_dataset = MultimodalDataset(data_dir=args.data_dir, split="val", config=config)

        train_loader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True if device.type == "cuda" else False,
            prefetch_factor=2 if args.num_workers > 0 else None,
            persistent_workers=True if args.num_workers > 0 else False,
            collate_fn=collate_multimodal,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            pin_memory=True if device.type == "cuda" else False,
            prefetch_factor=2 if args.num_workers > 0 else None,
            persistent_workers=True if args.num_workers > 0 else False,
            collate_fn=collate_multimodal,
        )
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        logger.info("Please ensure data is available in the specified directory")
        logger.info("See data/README.md for dataset preparation instructions")
        return

    # Initialize trainer
    trainer = MultimodalTrainer(
        model=model,
        task_head=task_head,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        checkpoint_dir=args.checkpoint_dir,
        log_dir=args.log_dir,
    )

    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)

    # Train
    trainer.train()


if __name__ == "__main__":
    main()
