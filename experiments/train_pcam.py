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
import shutil
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
from src.models.heads import ClassificationHead

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


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

    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    logger.info(f"Train samples: {len(train_dataset)}")
    logger.info(f"Val samples: {len(val_dataset)}")
    logger.info(f"Test samples: {len(test_dataset)}")

    return train_loader, val_loader, test_loader


def create_single_modality_model(config: Dict) -> Tuple[nn.Module, nn.Module, nn.Module]:
    """Create feature extractor, WSI encoder, and classification head.

    Args:
        config: Configuration dictionary containing model settings.

    Returns:
        Tuple of (feature_extractor, encoder, head).
    """
    # Feature extractor - ResNet applied at batch time
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

    log_interval = config["logging"]["log_interval"]

    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    status_interval = max(1, len(dataloader) // 5)

    for batch_idx, batch in enumerate(pbar):
        images = batch["image"].to(device)
        labels = batch["label"].to(device).float().unsqueeze(1)

        optimizer.zero_grad()

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

            # Check for NaN loss
            if torch.isnan(loss):
                logger.warning(f"NaN loss detected at batch {batch_idx}. Skipping batch.")
                continue

            scaler.scale(loss).backward()

            # Gradient clipping
            if config["training"].get("max_grad_norm", 1.0) > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(feature_extractor.parameters())
                    + list(encoder.parameters())
                    + list(head.parameters()),
                    max_norm=config["training"]["max_grad_norm"],
                )

            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training without mixed precision
            features = feature_extractor(images)
            features = features.unsqueeze(1)
            encoded = encoder(features)
            logits = head(encoded)
            loss = criterion(logits, labels)

            if torch.isnan(loss):
                logger.warning(f"NaN loss detected at batch {batch_idx}. Skipping batch.")
                continue

            loss.backward()

            if config["training"].get("max_grad_norm", 1.0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    list(feature_extractor.parameters())
                    + list(encoder.parameters())
                    + list(head.parameters()),
                    max_norm=config["training"]["max_grad_norm"],
                )

            optimizer.step()

        # Accumulate metrics
        total_loss += loss.item()

        # Get predictions
        probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
        preds = (probs > 0.5).astype(int)

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(labels.squeeze(1).cpu().numpy())

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

    # Compute epoch metrics
    avg_loss = total_loss / len(dataloader)
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
                continue

            total_loss += loss.item()

            # Get predictions
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            preds = (probs > 0.5).astype(int)

            # Check for NaN in predictions
            if np.isnan(probs).any():
                logger.warning("NaN predictions detected during validation. Skipping batch.")
                continue

            all_probs.extend(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.squeeze(1).cpu().numpy())

    # Compute metrics
    avg_loss = total_loss / len(dataloader)
    metrics = {
        "val_loss": avg_loss,
        "val_accuracy": accuracy_score(all_labels, all_preds),
        "val_f1": f1_score(all_labels, all_preds, average="binary"),
        "val_auc": roc_auc_score(all_labels, all_probs),
    }

    return metrics


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
) -> None:
    """Save checkpoint with all state dicts.

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
    """
    checkpoint = {
        "epoch": epoch,
        "feature_extractor_state_dict": feature_extractor.state_dict(),
        "encoder_state_dict": encoder.state_dict(),
        "head_state_dict": head.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict() if scheduler else None,
        "metrics": metrics,
        "config": config,
    }

    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved to {path}")


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
        raise FileNotFoundError(f"Checkpoint not found at {path}")

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

    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint from {path}: {e}")


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
    logger.info(f"Sample shape: [3, 96, 96]")


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

    logger.warning(
        f"GPU out of memory. Reducing batch size from {current_batch_size} to {new_batch_size}"
    )

    config["training"]["batch_size"] = new_batch_size

    # Log warning about potential OOM recovery
    logger.warning(
        "Please consider reducing batch size further if OOM persists. "
        f"Current batch size: {new_batch_size}"
    )

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
        logger.error(f"Failed to create dataloaders: {e}")
        return

    # Validate dataset
    logger.info("Validating training dataset...")
    try:
        validate_dataset(train_loader.dataset)
    except RuntimeError as e:
        logger.error(f"Dataset validation failed: {e}")
        return

    # Create model
    logger.info("Creating model...")
    feature_extractor, encoder, head = create_single_modality_model(config)

    # Move to device
    feature_extractor = feature_extractor.to(device)
    encoder = encoder.to(device)
    head = head.to(device)

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
        except Exception as e:
            logger.error(f"Failed to resume from checkpoint: {e}")

    # Training loop
    logger.info(f"Starting training for {num_epochs} epochs")

    for epoch in range(start_epoch, num_epochs + 1):
        logger.info(f"\n{'='*60}")
        logger.info(f"Epoch {epoch}/{num_epochs}")
        logger.info(f"{'='*60}")

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
                        save_checkpoint(
                            epoch,
                            feature_extractor,
                            encoder,
                            head,
                            optimizer,
                            scheduler,
                            val_metrics,
                            config,
                            str(checkpoint_dir / "best_model.pth"),
                        )
                        logger.info(f"✓ New best {metric_name}: {current_metric:.4f}")
                else:
                    patience_counter += 1

                # Early stopping
                if config["early_stopping"]["enabled"]:
                    patience = config["early_stopping"]["patience"]
                    min_delta = config["early_stopping"]["min_delta"]

                    if patience_counter >= patience:
                        logger.info(
                            f"Early stopping triggered after {epoch} epochs "
                            f"(no improvement for {patience} epochs)"
                        )
                        break

                # Save checkpoint at interval
                save_interval = config["checkpoint"]["save_interval"]
                if epoch % save_interval == 0:
                    save_checkpoint(
                        epoch,
                        feature_extractor,
                        encoder,
                        head,
                        optimizer,
                        scheduler,
                        val_metrics,
                        config,
                        str(checkpoint_dir / f"checkpoint_epoch_{epoch}.pth"),
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
    write_training_status(
        status_path,
        {
            "state": "completed",
            "epoch": num_epochs,
            "best_val_auc": float(best_val_auc),
            "test_metrics": test_metrics,
            "timestamp": int(time.time()),
            "run_id": run_id,
        },
    )

    logger.info("\nTraining complete!")
    logger.info(f"Best validation AUC: {best_val_auc:.4f}")


if __name__ == "__main__":
    main()
