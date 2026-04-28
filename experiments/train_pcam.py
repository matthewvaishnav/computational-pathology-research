"""
Training script for PatchCamelyon (PCam) binary classification experiment.

This script implements the complete training pipeline for the PCam dataset
with the following architecture:
- Raw images [3, 96, 96] returned by PCamDataset
- ResNetFeatureExtractor applied at BATCH TIME to extract features
- WSIEncoder for encoding the extracted features
- ClassificationHead for binary classification

Supports:
- Mixed precision training (AMP)
- TensorBoard logging
- Checkpointing with resume support
- Early stopping
- GPU OOM handling with batch size reduction
"""

import argparse
import json
import logging
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.pcam_dataset import PCamDataset, get_pcam_transforms
from src.models.encoders import WSIEncoder
from src.models.feature_extractors import ResNetFeatureExtractor
from src.models.foundation import load_foundation_model
from src.models.foundation.projector import FeatureProjector
from src.models.foundation.cache import cache_features, CachedFeatureDataset, get_cache_path
from src.models.heads import ClassificationHead

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class RecoveryStatistics:
    """Track recovery statistics for monitoring and analysis."""

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all statistics."""
        self.total_recovery_attempts = 0
        self.successful_recoveries = 0
        self.failed_recoveries = 0
        self.recovery_events = []
        self.cascading_nan_events = 0
        self.total_recovery_time = 0.0
        self.max_consecutive_nans = 0
        self.recovery_triggers = {
            "parameter_corruption": 0,
            "gradient_corruption": 0,
            "loss_corruption": 0,
            "prediction_corruption": 0,
        }

    def record_cascading_nan_event(self, consecutive_count: int, trigger_type: str):
        """Record a cascading NaN event."""
        self.cascading_nan_events += 1
        self.max_consecutive_nans = max(self.max_consecutive_nans, consecutive_count)
        if trigger_type in self.recovery_triggers:
            self.recovery_triggers[trigger_type] += 1

    def record_recovery_attempt(self, epoch: int, batch_idx: int, trigger_type: str):
        """Record the start of a recovery attempt."""
        self.total_recovery_attempts += 1
        recovery_event = {
            "attempt_id": self.total_recovery_attempts,
            "epoch": epoch,
            "batch_idx": batch_idx,
            "trigger_type": trigger_type,
            "start_time": time.time(),
            "end_time": None,
            "duration": None,
            "success": None,
            "error": None,
        }
        self.recovery_events.append(recovery_event)
        return len(self.recovery_events) - 1  # Return index for updating

    def record_recovery_outcome(self, event_idx: int, success: bool, error: str = None):
        """Record the outcome of a recovery attempt."""
        if event_idx < len(self.recovery_events):
            event = self.recovery_events[event_idx]
            event["end_time"] = time.time()
            event["duration"] = event["end_time"] - event["start_time"]
            event["success"] = success
            event["error"] = error

            self.total_recovery_time += event["duration"]

            if success:
                self.successful_recoveries += 1
            else:
                self.failed_recoveries += 1

    def get_summary(self) -> Dict[str, Any]:
        """Get a summary of recovery statistics."""
        success_rate = (
            (self.successful_recoveries / self.total_recovery_attempts * 100)
            if self.total_recovery_attempts > 0
            else 0
        )
        avg_recovery_time = (
            (self.total_recovery_time / self.total_recovery_attempts)
            if self.total_recovery_attempts > 0
            else 0
        )

        return {
            "total_recovery_attempts": self.total_recovery_attempts,
            "successful_recoveries": self.successful_recoveries,
            "failed_recoveries": self.failed_recoveries,
            "success_rate_percent": success_rate,
            "cascading_nan_events": self.cascading_nan_events,
            "max_consecutive_nans": self.max_consecutive_nans,
            "total_recovery_time_seconds": self.total_recovery_time,
            "average_recovery_time_seconds": avg_recovery_time,
            "recovery_triggers": self.recovery_triggers.copy(),
            "recent_events": self.recovery_events[-5:] if self.recovery_events else [],
        }

    def log_summary(self, logger, epoch: int = None):
        """Log a comprehensive summary of recovery statistics."""
        summary = self.get_summary()

        epoch_str = f" for Epoch {epoch}" if epoch is not None else ""

        logger.info(
            f"=== RECOVERY STATISTICS SUMMARY{epoch_str} ===\n"
            f"  Total Recovery Attempts: {summary['total_recovery_attempts']}\n"
            f"  Successful Recoveries: {summary['successful_recoveries']}\n"
            f"  Failed Recoveries: {summary['failed_recoveries']}\n"
            f"  Success Rate: {summary['success_rate_percent']:.1f}%\n"
            f"  Cascading NaN Events: {summary['cascading_nan_events']}\n"
            f"  Max Consecutive NaNs: {summary['max_consecutive_nans']}\n"
            f"  Total Recovery Time: {summary['total_recovery_time_seconds']:.2f}s\n"
            f"  Average Recovery Time: {summary['average_recovery_time_seconds']:.2f}s\n"
            f"  Recovery Triggers: {summary['recovery_triggers']}"
        )


# Global recovery statistics instance
recovery_stats = RecoveryStatistics()


def write_training_status(status_path: Path, payload: Dict[str, Any]) -> None:
    """Write current training status atomically for external monitoring."""
    status_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = status_path.with_suffix(".tmp")
    with open(tmp_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, status_path)


def coerce_numeric_training_config(config: Dict[str, Any], run_id: str) -> None:
    """Normalize numeric training fields that may be loaded as strings."""
    training_cfg = config.get("training", {})
    scheduler_cfg = training_cfg.get("scheduler", {})
    numeric_fields = [
        ("training.learning_rate", training_cfg, "learning_rate", float),
        ("training.weight_decay", training_cfg, "weight_decay", float),
        ("training.batch_size", training_cfg, "batch_size", int),
        ("training.scheduler.min_lr", scheduler_cfg, "min_lr", float),
    ]
    for field_name, parent, key, cast_type in numeric_fields:
        if key not in parent:
            continue
        original = parent[key]
        if isinstance(original, str):
            parent[key] = cast_type(original)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility.

    Args:
        seed: Random seed value.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    logger.info(f"Random seed set to {seed}")


def create_pcam_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, val, test dataloaders using PCamDataset.

    Args:
        config: Configuration dictionary containing data and training settings.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    root_dir = config["data"]["root_dir"]
    num_workers = config["data"].get("num_workers", 4)
    pin_memory = config["data"].get("pin_memory", True)
    batch_size = config["training"]["batch_size"]
    persistent_workers = config["data"].get("persistent_workers", False)
    prefetch_factor = config["data"].get("prefetch_factor", 2)

    # Get transforms
    train_transform = get_pcam_transforms(split="train", augmentation=True)
    val_transform = get_pcam_transforms(split="val", augmentation=False)
    test_transform = get_pcam_transforms(split="test", augmentation=False)

    # Create datasets
    train_dataset = PCamDataset(
        root_dir=root_dir,
        split="train",
        transform=train_transform,
        download=config["data"].get("download", True),
    )

    val_dataset = PCamDataset(
        root_dir=root_dir,
        split="val",
        transform=val_transform,
        download=config["data"].get("download", True),
    )

    test_dataset = PCamDataset(
        root_dir=root_dir,
        split="test",
        transform=test_transform,
        download=config["data"].get("download", True),
    )

    # DataLoader kwargs
    loader_kwargs = {
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
    }
    
    # Add persistent_workers only if num_workers > 0
    if num_workers > 0 and persistent_workers:
        loader_kwargs["persistent_workers"] = True
        loader_kwargs["prefetch_factor"] = prefetch_factor

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        shuffle=True,
        drop_last=True,
        **loader_kwargs
    )

    val_loader = DataLoader(
        val_dataset,
        shuffle=False,
        **loader_kwargs
    )

    test_loader = DataLoader(
        test_dataset,
        shuffle=False,
        **loader_kwargs
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Num workers: {num_workers}")
    if num_workers > 0 and persistent_workers:
        logger.info(f"Persistent workers: enabled")
        logger.info(f"Prefetch factor: {prefetch_factor}")

    return train_loader, val_loader, test_loader


def create_single_modality_model(config: Dict) -> Tuple[nn.Module, nn.Module, nn.Module]:
    """Create feature extractor, WSI encoder, and classification head.

    Supports both ResNet baseline and foundation models (Phikon/UNI/CONCH).

    Args:
        config: Configuration dictionary containing model settings.

    Returns:
        Tuple of (feature_extractor, encoder, head).
    """
    # Check if using foundation model
    use_foundation = config["model"].get("use_foundation", False)

    if use_foundation:
        # Foundation model path
        foundation_config = config["model"]["foundation"]
        model_name = foundation_config["model_name"]
        freeze = foundation_config.get("freeze", True)

        logger.info(f"Using foundation model: {model_name} (freeze={freeze})")

        # Load foundation encoder
        foundation_encoder = load_foundation_model(
            model_name=model_name,
            freeze=freeze,
        )

        # Create projector to map foundation features to standard dimension
        projector_config = foundation_config.get("projector", {})
        projector = FeatureProjector(
            input_dim=foundation_encoder.feature_dim,
            output_dim=projector_config.get("output_dim", 256),
            dropout=projector_config.get("dropout", 0.1),
        )

        # Combine foundation encoder + projector into a single feature extractor
        class FoundationFeatureExtractor(nn.Module):
            """Wrapper combining foundation encoder and projector."""

            def __init__(self, encoder, projector):
                super().__init__()
                self.encoder = encoder
                self.projector = projector

            def forward(self, x):
                features = self.encoder(x)
                return self.projector(features)

            def parameters(self, recurse=True):
                # Return parameters from both encoder and projector
                return list(self.encoder.parameters(recurse)) + list(
                    self.projector.parameters(recurse)
                )

        feature_extractor = FoundationFeatureExtractor(foundation_encoder, projector)

        logger.info(
            f"Foundation encoder: {foundation_encoder.feature_dim}-dim → "
            f"Projector: {projector_config.get('output_dim', 256)}-dim"
        )
        logger.info(f"Projector parameters: {projector.get_num_params():,}")

    else:
        # ResNet baseline path
        feature_extractor_config = config["model"]["feature_extractor"]
        feature_extractor = ResNetFeatureExtractor(
            model_name=feature_extractor_config["model"],
            pretrained=feature_extractor_config.get("pretrained", True),
            feature_dim=feature_extractor_config.get("feature_dim", 512),
        )

    # WSI Encoder for single patch encoding
    wsi_config = config["model"]["wsi"]
    encoder = WSIEncoder(
        input_dim=wsi_config["input_dim"],
        hidden_dim=wsi_config["hidden_dim"],
        output_dim=config["model"]["embed_dim"],
        num_heads=wsi_config["num_heads"],
        num_layers=wsi_config["num_layers"],
        pooling=wsi_config.get("pooling", "mean"),
        dropout=config["training"].get("dropout", 0.1),
    )

    # Classification head - binary classification uses single output logit with BCE loss
    classification_config = config["task"]["classification"]
    hidden_dims = classification_config.get("hidden_dims", [128])
    use_hidden_layer = len(hidden_dims) > 0
    hidden_dim = hidden_dims[0] if use_hidden_layer else 128  # Default if no hidden layer

    head = ClassificationHead(
        input_dim=config["model"]["embed_dim"],
        hidden_dim=hidden_dim,
        num_classes=1,  # Binary classification: single logit for BCEWithLogitsLoss
        dropout=classification_config["dropout"],
        use_hidden_layer=use_hidden_layer,
    )

    total_params_fe = sum(p.numel() for p in feature_extractor.parameters())
    total_params_enc = sum(p.numel() for p in encoder.parameters())
    total_params_head = sum(p.numel() for p in head.parameters())

    logger.info(f"Feature extractor parameters: {total_params_fe:,}")
    logger.info(f"Encoder parameters: {total_params_enc:,}")
    logger.info(f"Classification head parameters: {total_params_head:,}")
    logger.info(f"Total parameters: {total_params_fe + total_params_enc + total_params_head:,}")

    return feature_extractor, encoder, head


def check_model_parameters_for_nan(
    feature_extractor: nn.Module, encoder: nn.Module, head: nn.Module
) -> bool:
    """Check if any model parameters contain NaN values.

    Args:
        feature_extractor: Feature extractor model.
        encoder: Encoder model.
        head: Classification head model.

    Returns:
        True if any parameters contain NaN, False otherwise.
    """
    for model in [feature_extractor, encoder, head]:
        for param in model.parameters():
            if torch.isnan(param).any():
                return True
    return False


def validate_model_parameters_after_batch(
    feature_extractor: nn.Module,
    encoder: nn.Module,
    head: nn.Module,
    batch_idx: int,
    epoch: int,
    logger_instance: Optional[Any] = None,
) -> bool:
    """Validate model parameters immediately after optimizer step.

    This function checks for NaN in model parameters after each batch
    to detect parameter corruption immediately after optimizer updates.

    Args:
        feature_extractor: Feature extractor model.
        encoder: Encoder model.
        head: Classification head model.
        batch_idx: Current batch index for logging.
        epoch: Current epoch for logging.
        logger_instance: Logger instance for debugging output.

    Returns:
        True if parameters are valid (no NaN), False if corruption detected.
    """
    model_names = ["feature_extractor", "encoder", "head"]
    models = [feature_extractor, encoder, head]

    for model_name, model in zip(model_names, models):
        for param_name, param in model.named_parameters():
            if torch.isnan(param).any():
                if logger_instance:
                    logger_instance.error(
                        f"Parameter corruption detected in {model_name}.{param_name} "
                        f"after optimizer step at batch {batch_idx}, epoch {epoch}. "
                        f"Parameter shape: {param.shape}, "
                        f"NaN count: {torch.isnan(param).sum().item()}"
                    )
                else:
                    logger.error(
                        f"Parameter corruption detected in {model_name}.{param_name} "
                        f"after optimizer step at batch {batch_idx}, epoch {epoch}. "
                        f"Parameter shape: {param.shape}, "
                        f"NaN count: {torch.isnan(param).sum().item()}"
                    )
                return False

    # Log successful validation for debugging (only occasionally to avoid spam)
    if batch_idx % 100 == 0 and logger_instance:
        logger_instance.debug(
            f"Model parameter validation passed at batch {batch_idx}, epoch {epoch}. "
            f"All parameters in feature_extractor, encoder, and head are valid."
        )

    return True


def train_epoch(
    feature_extractor: nn.Module,
    encoder: nn.Module,
    head: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: str,
    config: Dict,
    epoch: int,
    scheduler: Any,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
    writer: Optional[SummaryWriter] = None,
    run_id: str = "run1",
    status_path: Optional[Path] = None,
) -> Dict[str, float]:
    """Train for one epoch.

    Args:
        feature_extractor: ResNet feature extractor model.
        encoder: WSI encoder model.
        head: Classification head model.
        dataloader: Training data loader.
        optimizer: Optimizer.
        criterion: Loss function.
        device: Device to train on.
        config: Configuration dictionary.
        epoch: Current epoch number.
        scheduler: Learning rate scheduler.
        scaler: Optional mixed precision gradient scaler.
        writer: Optional TensorBoard writer for logging.

    Returns:
        Dictionary of training metrics.
    """
    feature_extractor.train()
    encoder.train()
    head.train()

    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    num_valid_batches = 0
    num_skipped_batches = 0

    # Cascading NaN detection mechanism
    consecutive_nan_count = 0
    cascading_nan_threshold = 3

    # Recovery state management
    recovery_attempts = 0
    max_recovery_attempts = 3
    recovery_attempted = False

    # Enhanced checkpoint strategy for stability
    checkpoint_dir = Path(config.get("checkpoint", {}).get("checkpoint_dir", "checkpoints"))
    stability_checkpoint_frequency = config.get("checkpoint", {}).get(
        "stability_frequency", 50
    )  # Every N batches during instability
    rolling_checkpoint_count = config.get("checkpoint", {}).get(
        "rolling_window", 5
    )  # Keep last N checkpoints
    nan_detected_in_epoch = False  # Track if any NaN detected this epoch
    last_stability_checkpoint_batch = -1  # Track last stability checkpoint batch

    log_interval = config["logging"]["log_interval"]

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    status_interval = max(1, len(dataloader) // 5)

    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device)
        labels = batch["label"].to(device).float().unsqueeze(1)

        # OPTIMIZATION: Convert to channels_last if enabled
        if config.get("training", {}).get("channels_last", False) and device.type == "cuda":
            images = images.to(memory_format=torch.channels_last)

        # Mixed precision training
        if scaler is not None:
            with torch.cuda.amp.autocast():
                # Extract features at batch time [batch, 3, 96, 96] -> [batch, feature_dim]
                features = feature_extractor(images)

                # Add sequence dimension: [batch, feature_dim] -> [batch, 1, feature_dim]
                features = features.unsqueeze(1)

                # Encode via WSI encoder
                encoded = encoder(features)

                # Classify
                logits = head(encoded)

                # Compute loss
                loss = criterion(logits, labels)

            # Check for NaN loss or extremely small loss (numerical instability)
            # Note: Avoid .item() here to prevent GPU sync - check on GPU first
            if torch.isnan(loss) or loss < 1e-7:
                loss_val = loss.item()  # Single GPU→CPU transfer for logging
                if torch.isnan(loss):
                    logger.warning(f"NaN loss detected at batch {batch_idx}. Skipping batch.")
                else:
                    logger.warning(
                        f"Extremely small loss ({loss_val:.2e}) at batch {batch_idx}. Skipping to prevent numerical instability."
                    )
                num_skipped_batches += 1
                consecutive_nan_count += 1
                nan_detected_in_epoch = True

                # Save stability checkpoint during unstable periods
                if (
                    consecutive_nan_count >= 2
                    and batch_idx - last_stability_checkpoint_batch
                    >= stability_checkpoint_frequency
                ):
                    stability_checkpoint_path = (
                        checkpoint_dir / f"{run_id}_stability_epoch_{epoch}_batch_{batch_idx}.pth"
                    )
                    if save_checkpoint(
                        epoch,
                        feature_extractor,
                        encoder,
                        head,
                        optimizer,
                        scheduler,
                        {"loss": float("nan"), "batch_idx": batch_idx},
                        config,
                        str(stability_checkpoint_path),
                        batch_idx,
                        is_stability_checkpoint=True,
                    ):
                        last_stability_checkpoint_batch = batch_idx
                        logger.info(
                            f"Stability checkpoint saved due to instability at batch {batch_idx}"
                        )

                # Check for cascading NaN condition
                if consecutive_nan_count >= cascading_nan_threshold:
                    # Trigger recovery regardless of parameter state
                    # Cascading NaN indicates fundamental training instability
                    params_have_nan = check_model_parameters_for_nan(
                        feature_extractor, encoder, head
                    )
                    logger.error(
                        f"Cascading NaN detected: {consecutive_nan_count} consecutive NaN batches "
                        f"at batch {batch_idx}. Model parameters {'contain' if params_have_nan else 'do not contain'} NaN. "
                        f"Attempting recovery from checkpoint."
                    )

                    # Attempt recovery from checkpoint
                    recovery_successful, recovery_attempts = perform_recovery(
                        feature_extractor,
                        encoder,
                        head,
                        optimizer,
                        scheduler,
                        scaler,
                        config,
                        run_id,
                        epoch,
                        batch_idx,
                        recovery_attempts,
                        max_recovery_attempts,
                    )

                    if recovery_successful:
                        logger.info(
                            "Recovery successful. Resetting consecutive NaN count and continuing training."
                        )
                        consecutive_nan_count = 0
                        recovery_attempted = True
                        continue
                    else:
                        logger.error("Recovery failed. Training cannot continue.")
                        raise RuntimeError(
                            f"Cascading NaN losses detected. "
                            f"Recovery failed after {recovery_attempts} attempts. "
                            f"Consecutive NaN count: {consecutive_nan_count}, "
                            f"Batch: {batch_idx}, Epoch: {epoch}"
                        )

                continue

            scaler.scale(loss).backward()

            # Gradient clipping
            if config["training"].get("max_grad_norm", 1.0) > 0:
                scaler.unscale_(optimizer)

                # Check for NaN gradients before clipping
                has_nan_grad = False
                for param in (
                    list(feature_extractor.parameters())
                    + list(encoder.parameters())
                    + list(head.parameters())
                ):
                    if param.grad is not None and torch.isnan(param.grad).any():
                        has_nan_grad = True
                        break

                if has_nan_grad:
                    logger.warning(f"NaN gradients detected at batch {batch_idx}. Skipping batch.")
                    num_skipped_batches += 1
                    consecutive_nan_count += 1

                    # Check for cascading NaN condition
                    if consecutive_nan_count >= cascading_nan_threshold:
                        # Trigger recovery regardless of parameter state
                        # Cascading NaN indicates fundamental training instability
                        params_have_nan = check_model_parameters_for_nan(
                            feature_extractor, encoder, head
                        )
                        logger.error(
                            f"Cascading NaN detected: {consecutive_nan_count} consecutive NaN batches "
                            f"at batch {batch_idx}. Model parameters {'contain' if params_have_nan else 'do not contain'} NaN. "
                            f"Attempting recovery from checkpoint."
                        )

                        # Attempt recovery from checkpoint
                        recovery_successful, recovery_attempts = perform_recovery(
                            feature_extractor,
                            encoder,
                            head,
                            optimizer,
                            scheduler,
                            scaler,
                            config,
                            run_id,
                            epoch,
                            batch_idx,
                            recovery_attempts,
                            max_recovery_attempts,
                        )

                        if recovery_successful:
                            logger.info(
                                "Recovery successful. Resetting consecutive NaN count and continuing training."
                            )
                            consecutive_nan_count = 0
                            recovery_attempted = True
                            continue
                        else:
                            logger.error("Recovery failed. Training cannot continue.")
                            raise RuntimeError(
                                f"Cascading NaN losses detected. "
                                f"Recovery failed after {recovery_attempts} attempts. "
                                f"Consecutive NaN count: {consecutive_nan_count}, "
                                f"Batch: {batch_idx}, Epoch: {epoch}"
                            )

                    # Reset gradients and update scaler to maintain proper state
                    optimizer.zero_grad()
                    scaler.update()
                    continue

                torch.nn.utils.clip_grad_norm_(
                    list(feature_extractor.parameters())
                    + list(encoder.parameters())
                    + list(head.parameters()),
                    max_norm=config["training"]["max_grad_norm"],
                )

            scaler.step(optimizer)
            scaler.update()

            # Validate model parameters immediately after optimizer step
            if not validate_model_parameters_after_batch(
                feature_extractor, encoder, head, batch_idx, epoch, logger
            ):
                logger.error(
                    f"Model parameter corruption detected immediately after optimizer step "
                    f"at batch {batch_idx}, epoch {epoch}. Parameter corruption occurred during update."
                )
                consecutive_nan_count += 1

                # Check for cascading NaN condition due to parameter corruption
                if consecutive_nan_count >= cascading_nan_threshold:
                    logger.error(
                        f"Cascading NaN detected: {consecutive_nan_count} consecutive NaN batches "
                        f"with corrupted model parameters at batch {batch_idx}. "
                        f"Model parameter corruption detected after optimizer step - attempting recovery."
                    )

                    # Attempt recovery from checkpoint
                    recovery_successful, recovery_attempts = perform_recovery(
                        feature_extractor,
                        encoder,
                        head,
                        optimizer,
                        scheduler,
                        scaler,
                        config,
                        run_id,
                        epoch,
                        batch_idx,
                        recovery_attempts,
                        max_recovery_attempts,
                    )

                    if recovery_successful:
                        logger.info(
                            "Recovery successful. Resetting consecutive NaN count and continuing training."
                        )
                        consecutive_nan_count = 0
                        recovery_attempted = True
                        continue
                    else:
                        logger.error("Recovery failed. Training cannot continue.")
                        raise RuntimeError(
                            f"Cascading NaN losses detected with model parameter corruption after optimizer step. "
                            f"Recovery failed after {recovery_attempts} attempts. "
                            f"Consecutive NaN count: {consecutive_nan_count}, "
                            f"Batch: {batch_idx}, Epoch: {epoch}"
                        )

                # Skip this batch due to parameter corruption
                continue
        else:
            # Standard training without mixed precision
            features = feature_extractor(images)
            features = features.unsqueeze(1)
            encoded = encoder(features)
            logits = head(encoded)
            loss = criterion(logits, labels)

            if torch.isnan(loss) or loss.item() < 1e-7:
                if torch.isnan(loss):
                    logger.warning(f"NaN loss detected at batch {batch_idx}. Skipping batch.")
                else:
                    logger.warning(
                        f"Extremely small loss ({loss.item():.2e}) at batch {batch_idx}. Skipping to prevent numerical instability."
                    )
                num_skipped_batches += 1
                consecutive_nan_count += 1

                # Check for cascading NaN condition
                if consecutive_nan_count >= cascading_nan_threshold:
                    # Trigger recovery regardless of parameter state
                    # Cascading NaN indicates fundamental training instability
                    params_have_nan = check_model_parameters_for_nan(
                        feature_extractor, encoder, head
                    )
                    logger.error(
                        f"Cascading NaN detected: {consecutive_nan_count} consecutive NaN batches "
                        f"at batch {batch_idx}. Model parameters {'contain' if params_have_nan else 'do not contain'} NaN. "
                        f"Attempting recovery from checkpoint."
                    )

                    # Attempt recovery from checkpoint
                    recovery_successful, recovery_attempts = perform_recovery(
                        feature_extractor,
                        encoder,
                        head,
                        optimizer,
                        scheduler,
                        scaler,
                        config,
                        run_id,
                        epoch,
                        batch_idx,
                        recovery_attempts,
                        max_recovery_attempts,
                    )

                    if recovery_successful:
                        logger.info(
                            "Recovery successful. Resetting consecutive NaN count and continuing training."
                        )
                        consecutive_nan_count = 0
                        recovery_attempted = True
                        continue
                    else:
                        logger.error("Recovery failed. Training cannot continue.")
                        raise RuntimeError(
                            f"Cascading NaN losses detected. "
                            f"Recovery failed after {recovery_attempts} attempts. "
                            f"Consecutive NaN count: {consecutive_nan_count}, "
                            f"Batch: {batch_idx}, Epoch: {epoch}"
                        )

                continue

            loss.backward()

            if config["training"].get("max_grad_norm", 1.0) > 0:
                # Check for NaN gradients before clipping
                has_nan_grad = False
                for param in (
                    list(feature_extractor.parameters())
                    + list(encoder.parameters())
                    + list(head.parameters())
                ):
                    if param.grad is not None and torch.isnan(param.grad).any():
                        has_nan_grad = True
                        break

                if has_nan_grad:
                    logger.warning(f"NaN gradients detected at batch {batch_idx}. Skipping batch.")
                    num_skipped_batches += 1
                    consecutive_nan_count += 1

                    # Check for cascading NaN condition
                    if consecutive_nan_count >= cascading_nan_threshold:
                        # Trigger recovery regardless of parameter state
                        # Cascading NaN indicates fundamental training instability
                        params_have_nan = check_model_parameters_for_nan(
                            feature_extractor, encoder, head
                        )
                        logger.error(
                            f"Cascading NaN detected: {consecutive_nan_count} consecutive NaN batches "
                            f"at batch {batch_idx}. Model parameters {'contain' if params_have_nan else 'do not contain'} NaN. "
                            f"Attempting recovery from checkpoint."
                        )

                        # Attempt recovery from checkpoint
                        recovery_successful, recovery_attempts = perform_recovery(
                            feature_extractor,
                            encoder,
                            head,
                            optimizer,
                            scheduler,
                            scaler,
                            config,
                            run_id,
                            epoch,
                            batch_idx,
                            recovery_attempts,
                            max_recovery_attempts,
                        )

                        if recovery_successful:
                            logger.info(
                                "Recovery successful. Resetting consecutive NaN count and continuing training."
                            )
                            consecutive_nan_count = 0
                            recovery_attempted = True
                            continue
                        else:
                            logger.error("Recovery failed. Training cannot continue.")
                            raise RuntimeError(
                                f"Cascading NaN losses detected. "
                                f"Recovery failed after {recovery_attempts} attempts. "
                                f"Consecutive NaN count: {consecutive_nan_count}, "
                                f"Batch: {batch_idx}, Epoch: {epoch}"
                            )

                    optimizer.zero_grad()
                    continue

                torch.nn.utils.clip_grad_norm_(
                    list(feature_extractor.parameters())
                    + list(encoder.parameters())
                    + list(head.parameters()),
                    max_norm=config["training"]["max_grad_norm"],
                )

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)  # Faster than default zero_grad()

            # Validate model parameters immediately after optimizer step
            if not validate_model_parameters_after_batch(
                feature_extractor, encoder, head, batch_idx, epoch, logger
            ):
                logger.error(
                    f"Model parameter corruption detected immediately after optimizer step "
                    f"at batch {batch_idx}, epoch {epoch}. Parameter corruption occurred during update."
                )
                consecutive_nan_count += 1

                # Check for cascading NaN condition due to parameter corruption
                if consecutive_nan_count >= cascading_nan_threshold:
                    logger.error(
                        f"Cascading NaN detected: {consecutive_nan_count} consecutive NaN batches "
                        f"with corrupted model parameters at batch {batch_idx}. "
                        f"Model parameter corruption detected after optimizer step - attempting recovery."
                    )

                    # Attempt recovery from checkpoint
                    recovery_successful, recovery_attempts = perform_recovery(
                        feature_extractor,
                        encoder,
                        head,
                        optimizer,
                        scheduler,
                        scaler,
                        config,
                        run_id,
                        epoch,
                        batch_idx,
                        recovery_attempts,
                        max_recovery_attempts,
                    )

                    if recovery_successful:
                        logger.info(
                            "Recovery successful. Resetting consecutive NaN count and continuing training."
                        )
                        consecutive_nan_count = 0
                        recovery_attempted = True
                        continue
                    else:
                        logger.error("Recovery failed. Training cannot continue.")
                        raise RuntimeError(
                            f"Cascading NaN losses detected with model parameter corruption after optimizer step. "
                            f"Recovery failed after {recovery_attempts} attempts. "
                            f"Consecutive NaN count: {consecutive_nan_count}, "
                            f"Batch: {batch_idx}, Epoch: {epoch}"
                        )

                # Skip this batch due to parameter corruption
                continue

        # Get predictions
        probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
        preds = (probs > 0.5).astype(int)

        # Check for NaN in predictions
        if np.isnan(probs).any():
            logger.warning(f"NaN predictions detected at batch {batch_idx}. Skipping batch.")
            num_skipped_batches += 1
            consecutive_nan_count += 1

            # Check for cascading NaN condition
            if consecutive_nan_count >= cascading_nan_threshold:
                # Trigger recovery regardless of parameter state
                # Cascading NaN indicates fundamental training instability
                params_have_nan = check_model_parameters_for_nan(feature_extractor, encoder, head)
                logger.error(
                    f"Cascading NaN detected: {consecutive_nan_count} consecutive NaN batches "
                    f"at batch {batch_idx}. Model parameters {'contain' if params_have_nan else 'do not contain'} NaN. "
                    f"Attempting recovery from checkpoint."
                )

                # Attempt recovery from checkpoint
                recovery_successful, recovery_attempts = perform_recovery(
                    feature_extractor,
                    encoder,
                    head,
                    optimizer,
                    scheduler,
                    scaler,
                    config,
                    run_id,
                    epoch,
                    batch_idx,
                    recovery_attempts,
                    max_recovery_attempts,
                )

                if recovery_successful:
                    logger.info(
                        "Recovery successful. Resetting consecutive NaN count and continuing training."
                    )
                    consecutive_nan_count = 0
                    recovery_attempted = True
                    continue
                else:
                    logger.error("Recovery failed. Training cannot continue.")
                    raise RuntimeError(
                        f"Cascading NaN losses detected. "
                        f"Recovery failed after {recovery_attempts} attempts. "
                        f"Consecutive NaN count: {consecutive_nan_count}, "
                        f"Batch: {batch_idx}, Epoch: {epoch}"
                    )

            continue

        # Only accumulate metrics if batch is valid
        total_loss += loss.item()
        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.squeeze(1).cpu().numpy())
        num_valid_batches += 1

        # Reset consecutive NaN count on successful batch
        consecutive_nan_count = 0

        # Update progress bar
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if status_path is not None and (
            batch_idx % status_interval == 0 or batch_idx == len(dataloader) - 1
        ):
            status_payload = {
                "state": "training",
                "epoch": epoch,
                "batch_idx": batch_idx,
                "total_batches": len(dataloader),
                "loss": float(loss.item()),
                "timestamp": int(time.time()),
            }
            write_training_status(status_path, status_payload)

        # Log to TensorBoard at interval
        if batch_idx % log_interval == 0 and writer is not None:
            global_step = epoch * len(dataloader) + batch_idx
            writer.add_scalar("train/loss", loss.item(), global_step)

    # Check if we have any valid batches
    if num_valid_batches == 0:
        logger.error("All training batches had NaN values. Cannot compute metrics.")
        raise RuntimeError("Training failed: all batches contained NaN values")

    if num_skipped_batches > 0:
        logger.warning(
            f"Skipped {num_skipped_batches}/{len(dataloader)} training batches due to NaN"
        )
        if consecutive_nan_count > 0:
            logger.info(f"Epoch {epoch} ended with {consecutive_nan_count} consecutive NaN batches")

    # Cleanup old checkpoints at end of epoch to manage disk space
    if nan_detected_in_epoch:
        cleanup_old_checkpoints(checkpoint_dir, run_id, rolling_checkpoint_count)
        logger.info(
            f"Checkpoint cleanup completed - maintaining {rolling_checkpoint_count} recent checkpoints"
        )

    # Log enhanced checkpoint strategy summary
    logger.info(f"Epoch {epoch} checkpoint strategy summary:")
    logger.info(f"  - Cascading NaN threshold: {cascading_nan_threshold}")
    logger.info(f"  - Final consecutive NaN count: {consecutive_nan_count}")
    logger.info(f"  - Stability checkpoint frequency: {stability_checkpoint_frequency} batches")
    logger.info(f"  - Rolling checkpoint window: {rolling_checkpoint_count} checkpoints")
    logger.info(f"  - NaN detected in epoch: {nan_detected_in_epoch}")

    # Log cascading NaN detection summary
    logger.info(
        f"Epoch {epoch} completed - Cascading NaN threshold: {cascading_nan_threshold}, "
        f"Final consecutive NaN count: {consecutive_nan_count}"
    )

    # Log recovery statistics summary
    if recovery_attempted:
        logger.info(
            f"Epoch {epoch} recovery summary - Recovery attempts: {recovery_attempts}/{max_recovery_attempts}"
        )
        recovery_stats.log_summary(logger, epoch)
    else:
        logger.info(f"Epoch {epoch} completed without requiring recovery")

    # Log comprehensive recovery statistics if any events occurred
    if recovery_stats.total_recovery_attempts > 0 or recovery_stats.cascading_nan_events > 0:
        recovery_stats.log_summary(logger, epoch)

    # Compute epoch metrics
    avg_loss = total_loss / num_valid_batches
    metrics = {
        "loss": avg_loss,
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, average="binary"),
        "auc": roc_auc_score(all_labels, all_probs),
    }

    return metrics


def validate(
    feature_extractor: nn.Module,
    encoder: nn.Module,
    head: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Dict[str, float]:
    """Validate model.

    Args:
        feature_extractor: ResNet feature extractor model.
        encoder: WSI encoder model.
        head: Classification head model.
        dataloader: Validation data loader.
        criterion: Loss function.
        device: Device to validate on.
        scaler: Optional mixed precision gradient scaler.

    Returns:
        Dictionary of validation metrics.
    """
    feature_extractor.eval()
    encoder.eval()
    head.eval()

    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    num_valid_batches = 0
    num_skipped_batches = 0

    # Check if dataloader is empty
    if len(dataloader) == 0:
        error_msg = (
            "VALIDATION SET IS EMPTY\n\n"
            "TROUBLESHOOTING STEPS:\n"
            "1. CHECK DATASET SPLIT:\n"
            "   - Verify validation split exists and is not empty\n"
            "   - Check data/pcam/val/ directory\n"
            "   - Ensure val split was downloaded correctly\n\n"
            "2. CHECK BATCH SIZE:\n"
            "   - Validation set size must be >= batch_size\n"
            "   - Reduce batch_size if validation set is small\n\n"
            "3. RE-DOWNLOAD DATASET:\n"
            "   - Delete existing dataset: rm -rf data/pcam\n"
            "   - Enable download: data.download: true\n"
            "   - Restart training\n\n"
            "4. SKIP VALIDATION (NOT RECOMMENDED):\n"
            "   - Set validation.interval: 0 in config\n"
            "   - Only use for debugging purposes\n"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            images = batch["image"].to(device)
            labels = batch["label"].to(device).float().unsqueeze(1)

            # Mixed precision inference
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    features = feature_extractor(images)
                    features = features.unsqueeze(1)
                    encoded = encoder(features)
                    logits = head(encoded)
                    loss = criterion(logits, labels)
            else:
                features = feature_extractor(images)
                features = features.unsqueeze(1)
                encoded = encoder(features)
                logits = head(encoded)
                loss = criterion(logits, labels)

            # Check for NaN loss
            if torch.isnan(loss):
                logger.warning("NaN loss detected during validation. Skipping batch.")
                num_skipped_batches += 1
                continue

            # Get predictions
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)

            # Check for NaN in predictions
            if np.isnan(probs).any():
                logger.warning("NaN predictions detected during validation. Skipping batch.")
                num_skipped_batches += 1
                continue

            # Only accumulate if batch is valid
            total_loss += loss.item()
            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.squeeze(1).cpu().numpy())
            num_valid_batches += 1

    # Check if we have any valid batches
    if num_valid_batches == 0:
        error_msg = (
            "ALL VALIDATION BATCHES CONTAINED NaN VALUES\n\n"
            "TROUBLESHOOTING STEPS:\n"
            "1. CHECK MODEL STATE:\n"
            "   - Model may have corrupted parameters\n"
            "   - Try loading from earlier checkpoint\n"
            "   - Restart training with lower learning rate\n\n"
            "2. CHECK VALIDATION DATA:\n"
            "   - Verify validation images are not corrupted\n"
            "   - Run validate_dataset() on validation set\n"
            "   - Check for NaN/Inf values in validation data\n\n"
            "3. REDUCE LEARNING RATE:\n"
            "   - Current learning rate may be too high\n"
            "   - Try reducing by 10x: learning_rate: 1e-4\n\n"
            "4. ENABLE GRADIENT CLIPPING:\n"
            "   - Set training.max_grad_norm: 1.0\n"
            "   - This prevents gradient explosion\n"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg)

    if num_skipped_batches > 0:
        logger.warning(
            f"Skipped {num_skipped_batches}/{len(dataloader)} validation batches due to NaN"
        )

    # Compute metrics
    avg_loss = total_loss / num_valid_batches
    metrics = {
        "val_loss": avg_loss,
        "val_accuracy": accuracy_score(all_labels, all_preds),
        "val_f1": f1_score(all_labels, all_preds, average="binary"),
        "val_auc": roc_auc_score(all_labels, all_probs),
    }

    return metrics


def validate_checkpoint_integrity(checkpoint_path: str) -> bool:
    """Validate checkpoint integrity before saving or loading.

    Args:
        checkpoint_path: Path to checkpoint file.

    Returns:
        True if checkpoint is valid, False otherwise.
    """
    try:
        # Check if file exists and is readable
        if not Path(checkpoint_path).exists():
            logger.warning(f"Checkpoint file does not exist: {checkpoint_path}")
            return False

        # Try to load the checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu")

        # Validate required keys
        required_keys = [
            "epoch",
            "feature_extractor_state_dict",
            "encoder_state_dict",
            "head_state_dict",
            "optimizer_state_dict",
            "metrics",
            "config",
        ]

        for key in required_keys:
            if key not in checkpoint:
                logger.warning(f"Missing required key in checkpoint: {key}")
                return False

        # Validate state dicts are not empty
        for model_key in ["feature_extractor_state_dict", "encoder_state_dict", "head_state_dict"]:
            if not checkpoint[model_key]:
                logger.warning(f"Empty state dict in checkpoint: {model_key}")
                return False

        # Check for NaN values in model parameters
        for model_key in ["feature_extractor_state_dict", "encoder_state_dict", "head_state_dict"]:
            state_dict = checkpoint[model_key]
            for param_name, param_tensor in state_dict.items():
                if torch.isnan(param_tensor).any():
                    logger.warning(
                        f"NaN values found in checkpoint parameter: {model_key}.{param_name}"
                    )
                    return False

        logger.debug(f"Checkpoint integrity validation passed: {checkpoint_path}")
        return True

    except Exception as e:
        logger.warning(f"Checkpoint integrity validation failed: {e}")
        return False


def cleanup_old_checkpoints(checkpoint_dir: Path, run_id: str, keep_count: int = 5) -> None:
    """Clean up old checkpoints to manage disk space.

    Args:
        checkpoint_dir: Directory containing checkpoints.
        run_id: Current run ID for filtering checkpoints.
        keep_count: Number of recent checkpoints to keep.
    """
    try:
        # Find all checkpoints for this run (excluding best_model.pth)
        checkpoint_pattern = f"{run_id}_epoch_*.pth"
        checkpoint_files = list(checkpoint_dir.glob(checkpoint_pattern))

        if len(checkpoint_files) <= keep_count:
            return  # Nothing to clean up

        # Sort by epoch number (oldest first)
        checkpoint_files.sort(key=lambda x: int(x.stem.split("_")[-1]))

        # Remove oldest checkpoints, keeping only the most recent ones
        files_to_remove = checkpoint_files[:-keep_count]

        for checkpoint_file in files_to_remove:
            try:
                checkpoint_file.unlink()
                logger.debug(f"Cleaned up old checkpoint: {checkpoint_file}")
            except Exception as e:
                logger.warning(f"Failed to remove old checkpoint {checkpoint_file}: {e}")

        if files_to_remove:
            logger.info(
                f"Cleaned up {len(files_to_remove)} old checkpoints, keeping {keep_count} most recent"
            )

    except Exception as e:
        logger.warning(f"Checkpoint cleanup failed: {e}")


def save_checkpoint(
    epoch: int,
    feature_extractor: nn.Module,
    encoder: nn.Module,
    head: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    metrics: Dict,
    config: Dict,
    path: str,
    batch_idx: Optional[int] = None,
    is_stability_checkpoint: bool = False,
) -> bool:
    """Save checkpoint with all state dicts and integrity validation.

    Args:
        epoch: Current epoch number.
        feature_extractor: Feature extractor model.
        encoder: Encoder model.
        head: Classification head model.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        metrics: Current metrics dictionary.
        config: Configuration dictionary.
        path: Path to save checkpoint.
        batch_idx: Optional batch index for stability checkpoints.
        is_stability_checkpoint: Whether this is a stability checkpoint during unstable periods.

    Returns:
        True if checkpoint was saved successfully, False otherwise.
    """
    try:
        # Validate model parameters before saving
        if not validate_model_parameters_after_batch(
            feature_extractor, encoder, head, batch_idx or 0, epoch, logger
        ):
            logger.warning(
                f"Skipping checkpoint save due to corrupted model parameters at epoch {epoch}"
            )
            return False

        checkpoint = {
            "epoch": epoch,
            "batch_idx": batch_idx,
            "feature_extractor_state_dict": feature_extractor.state_dict(),
            "encoder_state_dict": encoder.state_dict(),
            "head_state_dict": head.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
            "metrics": metrics,
            "config": config,
            "timestamp": time.time(),
            "is_stability_checkpoint": is_stability_checkpoint,
        }

        # Save checkpoint
        torch.save(checkpoint, path)

        # Validate checkpoint integrity after saving
        if not validate_checkpoint_integrity(path):
            logger.error(f"Checkpoint integrity validation failed after saving: {path}")
            # Try to remove corrupted checkpoint
            try:
                Path(path).unlink()
                logger.info(f"Removed corrupted checkpoint: {path}")
            except Exception as e:
                logger.warning(f"Failed to remove corrupted checkpoint: {e}")
            return False

        checkpoint_type = "stability" if is_stability_checkpoint else "regular"
        batch_info = f", batch {batch_idx}" if batch_idx is not None else ""
        logger.info(
            f"✓ {checkpoint_type.capitalize()} checkpoint saved to {path} (epoch {epoch}{batch_info})"
        )
        return True

    except Exception as e:
        logger.error(f"Failed to save checkpoint to {path}: {e}")
        return False


def load_checkpoint(
    path: str,
    feature_extractor: nn.Module,
    encoder: nn.Module,
    head: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
) -> Tuple[int, Dict]:
    """Load checkpoint.

    Args:
        path: Path to checkpoint file.
        feature_extractor: Feature extractor model (will have state loaded).
        encoder: Encoder model (will have state loaded).
        head: Classification head model (will have state loaded).
        optimizer: Optimizer (will have state loaded).
        scheduler: Learning rate scheduler (will have state loaded).

    Returns:
        Tuple of (start_epoch, metrics).

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
        RuntimeError: If checkpoint is corrupted.
    """
    if not os.path.exists(path):
        error_msg = (
            f"CHECKPOINT FILE NOT FOUND\n\n"
            f"Path: {path}\n\n"
            "TROUBLESHOOTING STEPS:\n"
            "1. VERIFY CHECKPOINT PATH:\n"
            "   - Check if path is correct\n"
            "   - Use absolute path or path relative to current directory\n"
            "   - Example: --resume checkpoints/pcam/best_model.pth\n\n"
            "2. CHECK CHECKPOINT DIRECTORY:\n"
            "   - List checkpoints: ls -la checkpoints/pcam/\n"
            "   - Verify checkpoint files exist (.pth extension)\n\n"
            "3. CHECK TRAINING COMPLETED:\n"
            "   - Ensure training ran long enough to save checkpoints\n"
            "   - Check logs for checkpoint save messages\n\n"
            "4. USE CORRECT CHECKPOINT:\n"
            "   - For best model: checkpoints/pcam/best_model.pth\n"
            "   - For specific epoch: checkpoints/pcam/pcam-{timestamp}_epoch_{N}.pth\n"
        )
        logger.error(error_msg)
        raise FileNotFoundError(error_msg)

    try:
        checkpoint = torch.load(path, map_location="cpu")

        feature_extractor.load_state_dict(checkpoint["feature_extractor_state_dict"])
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
        head.load_state_dict(checkpoint["head_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

        if scheduler is not None and checkpoint.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        epoch = checkpoint.get("epoch", 0)
        metrics = checkpoint.get("metrics", {})

        logger.info(f"Loaded checkpoint from {path}")
        logger.info(f"Resuming from epoch {epoch}")

        return epoch, metrics

    except KeyError as e:
        error_msg = (
            f"CORRUPTED CHECKPOINT FILE\n\n"
            f"Missing key: {str(e)}\n"
            f"Path: {path}\n\n"
            "TROUBLESHOOTING STEPS:\n"
            "1. CHECKPOINT STRUCTURE MISMATCH:\n"
            "   - This checkpoint may be from an incompatible version\n"
            "   - Required keys: feature_extractor_state_dict, encoder_state_dict, head_state_dict\n\n"
            "2. USE DIFFERENT CHECKPOINT:\n"
            "   - Try another checkpoint from the same training run\n"
            "   - Check for best_model.pth or recent epoch checkpoints\n\n"
            "3. RETRAIN MODEL:\n"
            "   - If all checkpoints are corrupted, retrain from scratch\n"
            "   - Ensure sufficient disk space during training\n"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e
    except Exception as e:
        error_type = type(e).__name__
        error_msg = (
            f"FAILED TO LOAD CHECKPOINT\n\n"
            f"Error: {error_type}: {str(e)}\n"
            f"Path: {path}\n\n"
            "TROUBLESHOOTING STEPS:\n"
            "1. CHECK FILE INTEGRITY:\n"
            "   - File may be corrupted or incomplete\n"
            "   - Check file size: ls -lh {path}\n"
            "   - Compare with other checkpoints\n\n"
            "2. CHECK PYTORCH VERSION:\n"
            "   - Checkpoint may be from different PyTorch version\n"
            "   - Current version: {torch.__version__}\n"
            "   - Try loading with map_location='cpu'\n\n"
            "3. CHECK DISK SPACE:\n"
            "   - Ensure checkpoint was saved completely\n"
            "   - Verify no disk full errors during training\n\n"
            "4. USE DIFFERENT CHECKPOINT:\n"
            "   - Try loading a different checkpoint\n"
            "   - Use best_model.pth or recent epoch checkpoint\n"
        )
        logger.error(error_msg)
        raise RuntimeError(error_msg) from e


def handle_cascading_nan_recovery(
    consecutive_nan_count: int,
    cascading_nan_threshold: int,
    feature_extractor: nn.Module,
    encoder: nn.Module,
    head: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    scaler: Optional[torch.cuda.amp.GradScaler],
    config: Dict,
    run_id: str,
    epoch: int,
    batch_idx: int,
    recovery_attempts: int,
    max_recovery_attempts: int,
) -> Tuple[bool, int, int]:
    """Handle cascading NaN detection and recovery.

    Returns:
        Tuple of (should_continue, updated_consecutive_nan_count, updated_recovery_attempts).
        If should_continue is False, training should stop with an error.
    """
    if consecutive_nan_count >= cascading_nan_threshold:
        # Check if model parameters contain NaN
        if check_model_parameters_for_nan(feature_extractor, encoder, head):
            logger.error(
                f"Cascading NaN detected: {consecutive_nan_count} consecutive NaN batches "
                f"with corrupted model parameters at batch {batch_idx}. "
                f"Model parameter corruption detected - attempting recovery."
            )

            # Attempt recovery from checkpoint
            recovery_successful, updated_recovery_attempts = perform_recovery(
                feature_extractor,
                encoder,
                head,
                optimizer,
                scheduler,
                scaler,
                config,
                run_id,
                epoch,
                batch_idx,
                recovery_attempts,
                max_recovery_attempts,
            )

            if recovery_successful:
                logger.info(
                    "Recovery successful. Resetting consecutive NaN count and continuing training."
                )
                return True, 0, updated_recovery_attempts  # Reset consecutive_nan_count to 0
            else:
                logger.error("Recovery failed. Training cannot continue.")
                raise RuntimeError(
                    f"Cascading NaN losses detected with model parameter corruption. "
                    f"Recovery failed after {updated_recovery_attempts} attempts. "
                    f"Consecutive NaN count: {consecutive_nan_count}, "
                    f"Batch: {batch_idx}, Epoch: {epoch}"
                )
        else:
            logger.warning(
                f"Cascading NaN detected: {consecutive_nan_count} consecutive NaN batches "
                f"at batch {batch_idx}, Model parameters do not contain NaN. Attempting recovery anyway."
            )

    return False, consecutive_nan_count, recovery_attempts


def perform_recovery(
    feature_extractor: nn.Module,
    encoder: nn.Module,
    head: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    scaler: Optional[torch.cuda.amp.GradScaler],
    config: Dict,
    run_id: str,
    epoch: int,
    batch_idx: int,
    recovery_attempts: int,
    max_recovery_attempts: int,
) -> Tuple[bool, int]:
    """Perform checkpoint-based recovery from cascading NaN losses.

    Args:
        feature_extractor: Feature extractor model.
        encoder: Encoder model.
        head: Classification head model.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.
        scaler: Optional gradient scaler.
        config: Configuration dictionary.
        epoch: Current epoch number.
        batch_idx: Current batch index.
        recovery_attempts: Current number of recovery attempts.
        max_recovery_attempts: Maximum allowed recovery attempts.

    Returns:
        Tuple of (recovery_successful, updated_recovery_attempts).
    """
    # Record recovery attempt in statistics
    event_idx = recovery_stats.record_recovery_attempt(epoch, batch_idx, "parameter_corruption")

    # Comprehensive logging for recovery initiation
    logger.info(
        f"=== RECOVERY ATTEMPT #{recovery_attempts + 1} INITIATED ===\n"
        f"  Trigger: Cascading NaN losses detected\n"
        f"  Location: Epoch {epoch}, Batch {batch_idx}\n"
        f"  Attempts: {recovery_attempts + 1}/{max_recovery_attempts}\n"
        f"  Run ID: {run_id}\n"
        f"  Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}"
    )

    if recovery_attempts >= max_recovery_attempts:
        error_msg = f"Maximum recovery attempts ({max_recovery_attempts}) reached"
        logger.error(
            f"=== RECOVERY FAILED - MAXIMUM ATTEMPTS REACHED ===\n"
            f"  Maximum recovery attempts ({max_recovery_attempts}) reached.\n"
            f"  Cannot recover from cascading NaN at epoch {epoch}, batch {batch_idx}.\n"
            f"  Total attempts made: {recovery_attempts}\n"
            f"  Training will be terminated."
        )
        recovery_stats.record_recovery_outcome(event_idx, False, error_msg)
        return False, recovery_attempts

    recovery_attempts += 1

    try:
        # Determine checkpoint path - look for most recent checkpoint
        checkpoint_dir = Path(config.get("checkpoint_dir", "checkpoints"))

        logger.info(f"Searching for recovery checkpoints in: {checkpoint_dir}")

        # Look for the most recent checkpoint file
        checkpoint_pattern = f"{run_id}_epoch_*.pth"
        checkpoint_files = list(checkpoint_dir.glob(checkpoint_pattern))

        # Also look for stability checkpoints
        stability_pattern = f"{run_id}_stability_epoch_*.pth"
        stability_files = list(checkpoint_dir.glob(stability_pattern))

        # Combine all checkpoint files
        all_checkpoint_files = checkpoint_files + stability_files

        if not all_checkpoint_files:
            error_msg = "No checkpoint files found for recovery"
            logger.error(
                f"=== RECOVERY FAILED - NO CHECKPOINTS FOUND ===\n"
                f"  No checkpoint files found for recovery\n"
                f"  Search patterns: {checkpoint_pattern}, {stability_pattern}\n"
                f"  Search directory: {checkpoint_dir}\n"
                f"  Recovery attempt #{recovery_attempts} failed"
            )
            recovery_stats.record_recovery_outcome(event_idx, False, error_msg)
            return False, recovery_attempts

        # Sort by modification time and get the most recent
        all_checkpoint_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)
        latest_checkpoint = all_checkpoint_files[0]

        logger.info(
            f"Recovery checkpoint selected:\n"
            f"  File: {latest_checkpoint}\n"
            f"  Available checkpoints: {len(all_checkpoint_files)}\n"
            f"  Regular checkpoints: {len(checkpoint_files)}\n"
            f"  Stability checkpoints: {len(stability_files)}\n"
            f"  Checkpoint age: {(time.time() - latest_checkpoint.stat().st_mtime):.1f} seconds"
        )

        # Restore from checkpoint
        restored_epoch, metrics = restore_from_checkpoint(
            str(latest_checkpoint),
            feature_extractor,
            encoder,
            head,
            optimizer,
            scheduler,
            scaler,
        )

        logger.info(
            f"=== RECOVERY ATTEMPT #{recovery_attempts} SUCCESSFUL ===\n"
            f"  Restored from epoch: {restored_epoch}\n"
            f"  Restored metrics: {metrics}\n"
            f"  Model state: Successfully restored\n"
            f"  Optimizer state: Successfully restored\n"
            f"  Scheduler state: Successfully restored\n"
            f"  Scaler state: {'Successfully restored' if scaler else 'N/A (no mixed precision)'}\n"
            f"  Training will continue from batch {batch_idx}"
        )

        recovery_stats.record_recovery_outcome(event_idx, True)
        return True, recovery_attempts

    except Exception as e:
        error_msg = f"{type(e).__name__}: {str(e)}"
        logger.error(
            f"=== RECOVERY ATTEMPT #{recovery_attempts} FAILED ===\n"
            f"  Error: {str(e)}\n"
            f"  Error type: {type(e).__name__}\n"
            f"  Remaining attempts: {max_recovery_attempts - recovery_attempts}\n"
            f"  Will {'retry' if recovery_attempts < max_recovery_attempts else 'terminate training'}"
        )
        recovery_stats.record_recovery_outcome(event_idx, False, error_msg)
        return False, recovery_attempts


def restore_from_checkpoint(
    checkpoint_path: str,
    feature_extractor: nn.Module,
    encoder: nn.Module,
    head: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
    scaler: Optional[torch.cuda.amp.GradScaler] = None,
) -> Tuple[int, Dict]:
    """Restore model state from last valid checkpoint for recovery.

    This function implements checkpoint-based recovery mechanism to restore
    model state when cascading NaN losses are detected with parameter corruption.

    Args:
        checkpoint_path: Path to the checkpoint file to restore from.
        feature_extractor: Feature extractor model (will have state restored).
        encoder: Encoder model (will have state restored).
        head: Classification head model (will have state restored).
        optimizer: Optimizer (will have state restored).
        scheduler: Learning rate scheduler (will have state restored).
        scaler: Optional gradient scaler (will be reset to clean state).

    Returns:
        Tuple of (restored_epoch, metrics) from the checkpoint.

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
        RuntimeError: If checkpoint restoration fails.
    """
    logger.info(f"Attempting recovery from checkpoint: {checkpoint_path}")

    try:
        # Load model states from checkpoint
        epoch, metrics = load_checkpoint(
            checkpoint_path, feature_extractor, encoder, head, optimizer, scheduler
        )

        # Reset gradient scaler to clean state if using mixed precision
        if scaler is not None:
            logger.info("Resetting gradient scaler to clean state during recovery")
            # Recreate scaler with default settings instead of accessing internals
            device = next(feature_extractor.parameters()).device
            if device.type == "cuda":
                scaler.__init__()  # Reinitialize with defaults

        logger.info(f"Successfully restored model state from epoch {epoch}")
        logger.info(f"Recovery metrics: {metrics}")

        return epoch, metrics

    except Exception as e:
        logger.error(f"Failed to restore from checkpoint {checkpoint_path}: {e}")
        raise RuntimeError(f"Checkpoint recovery failed: {e}")


def validate_dataset(dataset: PCamDataset) -> None:
    """Validate dataset integrity before training.

    Args:
        dataset: PCam dataset to validate.

    Raises:
        RuntimeError: If dataset has integrity issues.
    """
    logger.info("Validating dataset integrity...")

    # Check dataset length
    if len(dataset) == 0:
        raise RuntimeError("Dataset is empty")

    # Check a few samples
    num_samples_to_check = min(10, len(dataset))
    errors = []

    for i in range(num_samples_to_check):
        try:
            sample = dataset[i]

            # Check image shape
            if sample["image"].shape != torch.Size([3, 96, 96]):
                errors.append(
                    f"Sample {i}: Expected image shape [3, 96, 96], got {sample['image'].shape}"
                )

            # Check label is valid (0 or 1)
            label = sample["label"].item()
            if label not in (0, 1):
                errors.append(f"Sample {i}: Invalid label {label}, expected 0 or 1")

            # Check image_id exists
            if "image_id" not in sample:
                errors.append(f"Sample {i}: Missing image_id")

        except Exception as e:
            errors.append(f"Sample {i}: Error loading sample: {e}")

    if errors:
        error_msg = "Dataset validation failed:\n" + "\n".join(errors)
        raise RuntimeError(error_msg)

    logger.info("Dataset validation passed!")
    logger.info(f"Dataset size: {len(dataset)} samples")
    logger.info("Sample shape: [3, 96, 96]")


def reduce_batch_size_on_oom(
    config: Dict,
    feature_extractor: nn.Module,
    encoder: nn.Module,
    head: nn.Module,
    optimizer: optim.Optimizer,
    scheduler: Any,
) -> Dict:
    """Reduce batch size and attempt to recover from GPU OOM error.

    Args:
        config: Configuration dictionary.
        feature_extractor: Feature extractor model.
        encoder: Encoder model.
        head: Classification head model.
        optimizer: Optimizer.
        scheduler: Learning rate scheduler.

    Returns:
        Updated configuration with reduced batch size.
    """
    current_batch_size = config["training"]["batch_size"]
    new_batch_size = max(8, current_batch_size // 2)

    error_msg = (
        f"GPU OUT OF MEMORY ERROR\n\n"
        f"Current batch size: {current_batch_size}\n"
        f"Attempting automatic reduction to: {new_batch_size}\n\n"
        "TROUBLESHOOTING STEPS:\n"
        "1. REDUCE BATCH SIZE FURTHER:\n"
        f"   - Current: {new_batch_size}\n"
        "   - Try: 64, 32, 16, or 8\n"
        "   - Edit config: training.batch_size: 32\n\n"
        "2. REDUCE MODEL SIZE:\n"
        "   - Decrease embed_dim (current: check config)\n"
        "   - Decrease hidden_dim in WSI encoder\n"
        "   - Use smaller feature extractor (resnet18 instead of resnet50)\n\n"
        "3. DISABLE MIXED PRECISION:\n"
        "   - Set training.use_amp: false in config\n"
        "   - This may increase memory usage but can help with stability\n\n"
        "4. USE GRADIENT ACCUMULATION:\n"
        "   - Simulate larger batch size with smaller batches\n"
        "   - Add training.gradient_accumulation_steps: 4\n\n"
        "5. CLEAR GPU CACHE:\n"
        "   - Run: torch.cuda.empty_cache()\n"
        "   - Restart training script\n\n"
        "6. CHECK GPU MEMORY:\n"
        "   - Run: nvidia-smi\n"
        "   - Ensure no other processes are using GPU\n"
        "   - Close other applications using GPU\n\n"
        "7. USE CPU (SLOWER):\n"
        "   - Set device: cpu in config\n"
        "   - Training will be much slower but won't have memory limits\n"
    )
    logger.warning(error_msg)

    config["training"]["batch_size"] = new_batch_size

    return config


def main():
    """Main training loop with full error handling."""
    parser = argparse.ArgumentParser(description="Train PCam model")
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )

    args = parser.parse_args()

    # Load configuration
    import yaml

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    run_id = f"pcam-{int(time.time())}"
    coerce_numeric_training_config(config, run_id)

    # Override config with command line args if needed
    if args.resume:
        config["resume"] = args.resume

    # Set seed
    seed = config.get("seed", 42)
    set_seed(seed)

    # Setup device
    device_str = config.get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device_str = "cpu"

    device = torch.device(device_str)

    # Log hardware/software info
    logger.info("=" * 60)
    logger.info("PCam Training Configuration")
    logger.info("=" * 60)
    logger.info(f"Config: {args.config}")
    logger.info(f"Device: {device}")
    logger.info(f"PyTorch version: {torch.__version__}")

    if torch.cuda.is_available():
        logger.info(f"CUDA version: {torch.version.cuda}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    logger.info(f"Seed: {seed}")
    logger.info("=" * 60)

    # Create directories
    checkpoint_dir = Path(config["checkpoint"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log_dir = Path(config["logging"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    status_path = log_dir / "training_status.json"
    write_training_status(
        status_path,
        {
            "state": "initializing",
            "epoch": 0,
            "timestamp": int(time.time()),
            "run_id": run_id,
        },
    )

    # Create dataloaders
    logger.info("Creating dataloaders...")
    try:
        train_loader, val_loader, test_loader = create_pcam_dataloaders(config)
    except Exception as e:
        error_type = type(e).__name__
        error_msg = (
            f"FAILED TO CREATE DATALOADERS\n\n"
            f"Error: {error_type}: {str(e)}\n\n"
            "TROUBLESHOOTING STEPS:\n"
            "1. CHECK DATASET AVAILABILITY:\n"
            "   - Verify dataset files exist in data.root_dir\n"
            "   - Enable download: data.download: true in config\n\n"
            "2. CHECK CONFIGURATION:\n"
            "   - Verify config file is valid YAML\n"
            "   - Check required fields: data.root_dir, training.batch_size\n"
            "   - Validate split names: 'train', 'val', 'test'\n\n"
            "3. CHECK SYSTEM RESOURCES:\n"
            "   - Ensure sufficient RAM (need ~4GB)\n"
            "   - Check disk space for dataset (~2GB)\n\n"
            "4. CHECK DEPENDENCIES:\n"
            "   - Verify PyTorch is installed: python -c 'import torch'\n"
            "   - Verify torchvision: python -c 'import torchvision'\n"
            "   - Check versions: pip list | grep torch\n\n"
            "5. REDUCE NUM_WORKERS:\n"
            "   - Set data.num_workers: 0 in config\n"
            "   - This disables multiprocessing which can cause issues\n"
        )
        logger.error(error_msg)
        return

    # Validate dataset
    logger.info("Validating training dataset...")
    try:
        validate_dataset(train_loader.dataset)
    except RuntimeError as e:
        error_msg = (
            f"DATASET VALIDATION FAILED\n\n"
            f"Error: {str(e)}\n\n"
            "TROUBLESHOOTING STEPS:\n"
            "1. CHECK DATASET INTEGRITY:\n"
            "   - Verify all dataset files are present and not corrupted\n"
            "   - Check file sizes match expected values\n\n"
            "2. RE-DOWNLOAD DATASET:\n"
            "   - Delete existing dataset: rm -rf data/pcam\n"
            "   - Enable download: data.download: true in config\n"
            "   - Restart training\n\n"
            "3. CHECK VALIDATION SET:\n"
            "   - Ensure validation split is not empty\n"
            "   - Verify split names are correct: 'train', 'val', 'test'\n\n"
            "4. SKIP VALIDATION (NOT RECOMMENDED):\n"
            "   - Comment out validate_dataset() call\n"
            "   - Only use if you're certain dataset is correct\n"
        )
        logger.error(error_msg)
        return

    # Create model
    logger.info("Creating model...")
    feature_extractor, encoder, head = create_single_modality_model(config)

    # Move to device
    feature_extractor = feature_extractor.to(device)
    encoder = encoder.to(device)
    head = head.to(device)

    # OPTIMIZATION: Enable channels_last memory format for better GPU performance
    if config.get("training", {}).get("channels_last", False) and device.type == "cuda":
        logger.info("Enabling channels_last memory format for better GPU performance")
        feature_extractor = feature_extractor.to(memory_format=torch.channels_last)
        logger.info("✓ Channels last enabled")

    # OPTIMIZATION: Enable torch.compile for 20-40% speedup (PyTorch 2.0+)
    if config.get("training", {}).get("use_torch_compile", False):
        if hasattr(torch, 'compile'):
            compile_mode = config.get("training", {}).get("torch_compile_mode", "default")
            logger.info(f"Compiling models with torch.compile (mode={compile_mode})...")
            try:
                feature_extractor = torch.compile(feature_extractor, mode=compile_mode)
                encoder = torch.compile(encoder, mode=compile_mode)
                head = torch.compile(head, mode=compile_mode)
                logger.info("✓ Models compiled successfully")
            except Exception as e:
                logger.warning(f"torch.compile failed: {e}. Continuing without compilation.")
        else:
            logger.warning("torch.compile not available (requires PyTorch 2.0+)")

    # OPTIMIZATION: Enable cuDNN benchmark for faster convolutions
    if config.get("cudnn_benchmark", False) and device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        logger.info("✓ cuDNN benchmark enabled")

    # Feature caching for foundation models (4x speedup)
    use_cache = config.get("foundation_model", {}).get("use_cache", True)
    if "foundation_model" in config and use_cache:
        foundation_config = config["foundation_model"]
        model_name = foundation_config["model"]
        cache_root = Path(foundation_config.get("cache_dir", "cache/features"))
        
        logger.info(f"Feature caching enabled for {model_name}")
        
        # Check if cache exists
        train_cache_dir = get_cache_path(cache_root, model_name, "train")
        val_cache_dir = get_cache_path(cache_root, model_name, "val")
        
        if not train_cache_dir.exists() or not val_cache_dir.exists():
            logger.info("Cache not found. Pre-extracting features...")
            logger.info("This is a one-time operation (~2 minutes on GPU)")
            
            # Extract foundation encoder from wrapper
            if hasattr(feature_extractor, 'encoder'):
                foundation_encoder = feature_extractor.encoder
            else:
                foundation_encoder = feature_extractor
            
            # Cache training features
            if not train_cache_dir.exists():
                logger.info("Caching training features...")
                cache_features(
                    foundation_encoder,
                    train_loader,
                    train_cache_dir,
                    device=str(device),
                    desc="Caching train features"
                )
            
            # Cache validation features
            if not val_cache_dir.exists():
                logger.info("Caching validation features...")
                cache_features(
                    foundation_encoder,
                    val_loader,
                    val_cache_dir,
                    device=str(device),
                    desc="Caching val features"
                )
            
            logger.info("✓ Feature caching complete!")
        else:
            logger.info("✓ Using cached features")
        
        # Replace dataloaders with cached versions
        train_cached = CachedFeatureDataset(train_cache_dir)
        val_cached = CachedFeatureDataset(val_cache_dir)
        
        train_loader = DataLoader(
            train_cached,
            batch_size=config["training"]["batch_size"],
            shuffle=True,
            num_workers=config["data"].get("num_workers", 0),
            pin_memory=config["data"].get("pin_memory", False)
        )
        
        val_loader = DataLoader(
            val_cached,
            batch_size=config["training"]["batch_size"],
            shuffle=False,
            num_workers=config["data"].get("num_workers", 0),
            pin_memory=config["data"].get("pin_memory", False)
        )
        
        # Replace feature extractor with projector only (features already extracted)
        if hasattr(feature_extractor, 'projector'):
            feature_extractor = feature_extractor.projector.to(device)
            logger.info("Using cached features → training projector only")
        
        logger.info(f"Training speedup: ~4x (40min → 10min per epoch)")
    

    # Setup optimizer
    optimizer = optim.AdamW(
        list(feature_extractor.parameters()) + list(encoder.parameters()) + list(head.parameters()),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
        betas=config["training"]["optimizer"].get("betas", [0.9, 0.999]),
    )

    # Setup scheduler
    num_epochs = config["training"]["num_epochs"]
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=num_epochs,
        eta_min=config["training"]["scheduler"].get("min_lr", 1e-6),
    )

    # Setup loss function
    criterion = nn.BCEWithLogitsLoss()

    # Setup mixed precision training
    use_amp = config["training"].get("use_amp", True)
    scaler = None
    if use_amp and device.type == "cuda":
        scaler = torch.cuda.amp.GradScaler()
        logger.info("Mixed precision training enabled")

    # Setup TensorBoard
    writer = SummaryWriter(log_dir=log_dir)

    # Resume from checkpoint if specified
    start_epoch = 1
    best_val_auc = 0.0
    patience_counter = 0

    if config.get("resume"):
        try:
            start_epoch, metrics = load_checkpoint(
                config["resume"],
                feature_extractor,
                encoder,
                head,
                optimizer,
                scheduler,
            )
            start_epoch += 1  # Start from next epoch
            best_val_auc = metrics.get("val_auc", 0.0)
            logger.info(f"Resuming from epoch {start_epoch}, best val AUC: {best_val_auc:.4f}")
        except FileNotFoundError as e:
            logger.error(str(e))
            logger.error("Cannot resume training without valid checkpoint. Exiting.")
            return
        except RuntimeError as e:
            logger.error(str(e))
            logger.error("Cannot resume training with corrupted checkpoint. Exiting.")
            return
        except Exception as e:
            error_msg = (
                f"UNEXPECTED ERROR LOADING CHECKPOINT\n\n"
                f"Error: {type(e).__name__}: {str(e)}\n\n"
                "TROUBLESHOOTING STEPS:\n"
                "1. VERIFY CHECKPOINT FILE:\n"
                f"   - Path: {config.get('resume')}\n"
                "   - Check file exists and is readable\n\n"
                "2. START FRESH TRAINING:\n"
                "   - Remove --resume flag to train from scratch\n"
                "   - Previous checkpoints may be incompatible\n\n"
                "3. CHECK LOGS:\n"
                "   - Review error message above for specific issue\n"
                "   - Check training logs for checkpoint save errors\n"
            )
            logger.error(error_msg)
            return

    # Training loop
    logger.info(f"Starting training for {num_epochs} epochs")

    # Initialize recovery statistics for this training run
    recovery_stats.reset()
    logger.info("Recovery statistics initialized for training run")

    for epoch in range(start_epoch, num_epochs + 1):
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Epoch {epoch}/{num_epochs}")
        logger.info(f"{'=' * 60}")

        try:
            # Train
            train_metrics = train_epoch(
                feature_extractor,
                encoder,
                head,
                train_loader,
                optimizer,
                criterion,
                device,
                config,
                epoch,
                scheduler,
                scaler,
                writer,
                run_id=run_id,
                status_path=status_path,
            )

            logger.info(f"Training metrics: {train_metrics}")

            # Log to TensorBoard
            for key, value in train_metrics.items():
                writer.add_scalar(f"train/{key}", value, epoch)

            # Validate
            val_interval = config["validation"]["interval"]
            if epoch % val_interval == 0 or epoch == num_epochs:
                val_metrics = validate(
                    feature_extractor,
                    encoder,
                    head,
                    val_loader,
                    criterion,
                    device,
                    scaler,
                )

                logger.info(f"Validation metrics: {val_metrics}")

                for key, value in val_metrics.items():
                    writer.add_scalar(f"epoch/{key}", value, epoch)

                # Update scheduler
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
                writer.add_scalar("train/lr", current_lr, epoch)

                # Check for improvement
                metric_name = config["validation"]["metric"]
                current_metric = val_metrics.get(metric_name, 0.0)
                maximize = config["validation"].get("maximize", True)

                improved = (
                    (current_metric > best_val_auc) if maximize else (current_metric < best_val_auc)
                )

                if improved:
                    best_val_auc = current_metric
                    patience_counter = 0

                    # Save best model
                    if config["checkpoint"].get("save_best", True):
                        if save_checkpoint(
                            epoch,
                            feature_extractor,
                            encoder,
                            head,
                            optimizer,
                            scheduler,
                            val_metrics,
                            config,
                            str(checkpoint_dir / "best_model.pth"),
                        ):
                            logger.info(f"✓ New best {metric_name}: {current_metric:.4f}")
                        else:
                            logger.warning(
                                f"Failed to save best model checkpoint for {metric_name}: {current_metric:.4f}"
                            )
                else:
                    patience_counter += 1

                # Early stopping
                if config["early_stopping"]["enabled"]:
                    patience = config["early_stopping"]["patience"]
                    config["early_stopping"]["min_delta"]

                    if patience_counter >= patience:
                        logger.info(
                            f"Early stopping triggered after {epoch} epochs "
                            f"(no improvement for {patience} epochs)"
                        )
                        break

                # Save checkpoint at interval
                save_interval = config["checkpoint"]["save_interval"]
                if epoch % save_interval == 0:
                    checkpoint_path = str(checkpoint_dir / f"{run_id}_epoch_{epoch}.pth")
                    if save_checkpoint(
                        epoch,
                        feature_extractor,
                        encoder,
                        head,
                        optimizer,
                        scheduler,
                        val_metrics,
                        config,
                        checkpoint_path,
                    ):
                        logger.info(f"✓ Interval checkpoint saved at epoch {epoch}")
                    else:
                        logger.warning(f"Failed to save interval checkpoint at epoch {epoch}")

                    # Cleanup old checkpoints to maintain rolling window
                    cleanup_old_checkpoints(
                        checkpoint_dir,
                        run_id,
                        config.get("checkpoint", {}).get("rolling_window", 5),
                    )

            # Log learning rate
            current_lr = scheduler.get_last_lr()[0]
            writer.add_scalar("train/lr", current_lr, epoch)
            write_training_status(
                status_path,
                {
                    "state": "epoch_complete",
                    "epoch": epoch,
                    "num_epochs": num_epochs,
                    "train_metrics": train_metrics,
                    "timestamp": int(time.time()),
                    "run_id": run_id,
                },
            )

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                logger.error("GPU out of memory!")
                config = reduce_batch_size_on_oom(
                    config,
                    feature_extractor,
                    encoder,
                    head,
                    optimizer,
                    scheduler,
                )
                # Recreate dataloaders with new batch size
                train_loader, val_loader, test_loader = create_pcam_dataloaders(config)
            else:
                raise e

    # Final evaluation on test set
    logger.info("\n" + "=" * 60)
    logger.info("Final evaluation on test set")
    logger.info("=" * 60)

    test_metrics = validate(
        feature_extractor,
        encoder,
        head,
        test_loader,
        criterion,
        device,
        scaler,
    )

    logger.info(f"Test metrics: {test_metrics}")

    for key, value in test_metrics.items():
        writer.add_scalar(f"test/{key}", value, num_epochs)

    writer.close()

    # Log final recovery statistics summary
    if recovery_stats.total_recovery_attempts > 0 or recovery_stats.cascading_nan_events > 0:
        logger.info("\n" + "=" * 60)
        logger.info("FINAL RECOVERY STATISTICS SUMMARY")
        logger.info("=" * 60)
        recovery_stats.log_summary(logger)

    write_training_status(
        status_path,
        {
            "state": "completed",
            "epoch": num_epochs,
            "best_val_auc": float(best_val_auc),
            "test_metrics": test_metrics,
            "recovery_statistics": recovery_stats.get_summary(),
            "timestamp": int(time.time()),
            "run_id": run_id,
        },
    )

    logger.info("\nTraining complete!")
    logger.info(f"Best validation AUC: {best_val_auc:.4f}")


if __name__ == "__main__":
    main()
