"""
Training script for CAMELYON16 slide-level classification.

This script implements slide-level training using pre-extracted patch features
from HDF5 files. Each training sample represents a complete slide with all its
patches, ensuring consistency with the evaluation pipeline.

IMPORTANT: This is a feature-cache baseline that uses pre-extracted HDF5 features,
not raw WSI files. It does not perform on-the-fly patch extraction from OpenSlide.

Architecture:
    - Slide-level dataset: CAMELYONSlideDataset
    - Aggregation: Mean or max pooling of patch features
    - Classifier: Simple MLP on aggregated features

Usage:
    python experiments/train_camelyon.py --config experiments/configs/camelyon.yaml

Requirements:
    - Slide index JSON at data/camelyon/slide_index.json
    - Pre-extracted HDF5 features at data/camelyon/features/
    - Each HDF5 file contains 'features' [num_patches, feature_dim] and 'coordinates' [num_patches, 2]

Configuration:
    - model.wsi.aggregation: "mean" or "max" pooling method
    - training.batch_size: Number of slides per batch
    - data.root_dir: Root directory containing slide_index.json and features/
"""

import argparse
import inspect
import logging
import random
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.camelyon_dataset import (
    CAMELYONSlideDataset,
    CAMELYONSlideIndex,
    collate_slide_bags,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    logger.info(f"Random seed set to {seed}")


def load_config(config_path: str) -> Dict:
    """Load YAML configuration file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def validate_config(config: Dict) -> None:
    """Validate configuration has required fields.

    Args:
        config: Configuration dictionary

    Raises:
        ValueError: If required fields are missing or invalid
    """
    required_fields = [
        ("data", "root_dir"),
        ("training", "batch_size"),
        ("training", "num_epochs"),
        ("model", "wsi", "hidden_dim"),
        ("task", "num_classes"),
    ]

    for *path, field in required_fields:
        obj = config
        for key in path:
            if key not in obj:
                raise ValueError(f"Missing config field: {'.'.join(path + [field])}")
            obj = obj[key]
        if field not in obj:
            raise ValueError(f"Missing config field: {'.'.join(path + [field])}")

    # Validate aggregation method
    aggregation = config.get("model", {}).get("wsi", {}).get("aggregation", "mean")
    if aggregation not in ["mean", "max"]:
        raise ValueError(f"Invalid aggregation method: {aggregation}. Must be 'mean' or 'max'")

    # Validate data paths exist
    root_dir = Path(config["data"]["root_dir"])
    if not root_dir.exists():
        raise ValueError(f"Data root directory does not exist: {root_dir}")

    logger.info("Configuration validation passed")


def validate_model_config(config: Dict) -> None:
    """Validate model-specific configuration parameters.

    Args:
        config: Configuration dictionary

    Raises:
        ValueError: If model configuration is invalid
    """
    model_config = config.get("model", {}).get("wsi", {})
    model_type = model_config.get("model_type", "mean")

    # Validate model_type
    valid_model_types = ["attention_mil", "clam", "transmil", "mean", "max"]
    if model_type not in valid_model_types:
        raise ValueError(f"Invalid model_type: {model_type}. Must be one of {valid_model_types}")

    # Validate AttentionMIL-specific config
    if model_type == "attention_mil":
        attention_config = config.get("model", {}).get("attention_mil", {})
        attention_mode = attention_config.get("attention_mode", "instance")

        if attention_mode not in ["instance", "bag"]:
            raise ValueError(
                f"Invalid attention_mode for AttentionMIL: {attention_mode}. "
                f"Must be 'instance' or 'bag'"
            )

        logger.info(f"AttentionMIL config validated: attention_mode={attention_mode}")

    # Validate CLAM-specific config
    elif model_type == "clam":
        clam_config = config.get("model", {}).get("clam", {})
        num_clusters = clam_config.get("num_clusters", 10)

        if num_clusters < 2:
            raise ValueError(f"Invalid num_clusters for CLAM: {num_clusters}. Must be >= 2")

        logger.info(f"CLAM config validated: num_clusters={num_clusters}")

    # Validate TransMIL-specific config
    elif model_type == "transmil":
        transmil_config = config.get("model", {}).get("transmil", {})
        hidden_dim = model_config.get("hidden_dim", 256)
        num_heads = transmil_config.get("num_heads", 8)

        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"Invalid TransMIL config: hidden_dim ({hidden_dim}) must be "
                f"divisible by num_heads ({num_heads})"
            )

        logger.info(f"TransMIL config validated: hidden_dim={hidden_dim}, num_heads={num_heads}")

    logger.info(f"Model configuration validated for model_type={model_type}")


class SimpleSlideClassifier(nn.Module):
    """Simple slide-level classifier with patch aggregation.

    This is a minimal baseline that:
    1. Takes patch features [num_patches, feature_dim]
    2. Aggregates via mean/max pooling
    3. Passes through a simple MLP classifier
    """

    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int = 256,
        num_classes: int = 2,
        pooling: str = "mean",
        dropout: float = 0.3,
    ):
        super().__init__()
        self.pooling = pooling

        # Simple MLP classifier
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_classes if num_classes > 2 else 1),
        )

    def forward(
        self, patch_features: torch.Tensor, num_patches: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass with optional masking for padded patches.

        Args:
            patch_features: [batch_size, max_patches, feature_dim]
            num_patches: [batch_size] - actual patch counts for masking

        Returns:
            logits: [batch_size, num_classes] or [batch_size, 1] for binary
        """
        # Aggregate patches
        if self.pooling == "mean":
            if num_patches is not None:
                # Masked mean: only average over actual patches
                mask = (
                    torch.arange(patch_features.size(1), device=patch_features.device)[None, :]
                    < num_patches[:, None]
                )
                mask = mask.unsqueeze(-1).float()  # [batch_size, max_patches, 1]
                slide_features = (patch_features * mask).sum(dim=1) / num_patches.unsqueeze(
                    -1
                ).float()
            else:
                slide_features = patch_features.mean(dim=1)
        elif self.pooling == "max":
            slide_features = patch_features.max(dim=1)[0]  # [batch_size, feature_dim]
        else:
            raise ValueError(f"Unknown pooling: {self.pooling}")

        # Classify
        logits = self.classifier(slide_features)
        return logits


def collate_slide_features(batch):
    """Custom collate function for slide-level batching.

    Since slides have different numbers of patches, we need to handle
    variable-length sequences. For simplicity, we'll pad to max length.
    """
    # Extract features and labels
    features_list = [item["features"] for item in batch]
    labels = torch.tensor([item["label"] for item in batch], dtype=torch.long)
    slide_ids = [item["slide_id"] for item in batch]

    # Pad to max length in batch
    max_patches = max(f.shape[0] for f in features_list)
    feature_dim = features_list[0].shape[1]

    padded_features = torch.zeros(len(batch), max_patches, feature_dim)
    for i, features in enumerate(features_list):
        padded_features[i, : features.shape[0], :] = features

    return {
        "features": padded_features,
        "labels": labels,
        "slide_ids": slide_ids,
    }


def create_slide_dataloaders(config: Dict) -> Tuple[DataLoader, DataLoader]:
    """Create train and val dataloaders for slide-level data.

    Args:
        config: Configuration dictionary

    Returns:
        Tuple of (train_loader, val_loader)
    """
    root_dir = Path(config["data"]["root_dir"])
    features_dir = root_dir / "features"
    index_path = root_dir / "slide_index.json"

    # Check if required files exist
    if not index_path.exists():
        raise FileNotFoundError(
            f"Slide index not found: {index_path}\n"
            f"Please create a slide index using CAMELYONSlideIndex.from_directory() "
            f"or download the pre-built index."
        )

    if not features_dir.exists():
        raise FileNotFoundError(
            f"Features directory not found: {features_dir}\n"
            f"Please extract patch features to HDF5 files first."
        )

    # Load slide index
    slide_index = CAMELYONSlideIndex.load(index_path)
    logger.info(f"Loaded slide index with {len(slide_index)} slides")

    # Create slide-level datasets
    train_dataset = CAMELYONSlideDataset(
        slide_index=slide_index,
        features_dir=features_dir,
        split="train",
    )

    val_dataset = CAMELYONSlideDataset(
        slide_index=slide_index,
        features_dir=features_dir,
        split="val",
    )

    # Create dataloaders with custom collate
    batch_size = config["training"]["batch_size"]
    num_workers = config["data"].get("num_workers", 4)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_slide_bags,
        pin_memory=config["data"].get("pin_memory", True),
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_slide_bags,
        pin_memory=config["data"].get("pin_memory", True),
    )

    logger.info(f"Train: {len(train_dataset)} slides, Val: {len(val_dataset)} slides")

    return train_loader, val_loader


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    save_attention: bool = False,
    attention_dir: Optional[Path] = None,
) -> Dict[str, float]:
    """Train for one epoch.

    Args:
        model: Model to train
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        epoch: Current epoch number
        save_attention: Whether to save attention weights
        attention_dir: Directory to save attention weights (required if save_attention=True)

    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    # Check if model supports return_attention parameter
    forward_signature = inspect.signature(model.forward)
    supports_attention = "return_attention" in forward_signature.parameters

    pbar = tqdm(train_loader, desc=f"Epoch {epoch} [Train]")
    for batch in pbar:
        # Extract slide-level batch data
        features = batch["features"].to(device)  # [batch_size, max_patches, feature_dim]
        labels = batch["labels"].to(device)  # [batch_size]
        num_patches = batch["num_patches"].to(device)  # [batch_size]
        slide_ids = batch.get("slide_ids", [])

        # Forward pass
        optimizer.zero_grad()

        # Call model with or without return_attention based on support
        if supports_attention and save_attention:
            logits, attention_weights = model(features, num_patches, return_attention=True)

            # Save attention weights to HDF5 if requested
            if attention_dir is not None and len(slide_ids) > 0:
                attention_dir.mkdir(parents=True, exist_ok=True)
                for i, slide_id in enumerate(slide_ids):
                    # Only save valid patches (use num_patches to slice)
                    valid_patches = num_patches[i].item()
                    attn = attention_weights[i, :valid_patches].detach().cpu().numpy()

                    # Save to HDF5
                    h5_path = attention_dir / f"{slide_id}_epoch{epoch}.h5"
                    with h5py.File(h5_path, "w") as f:
                        f.create_dataset("attention_weights", data=attn)
                        f.attrs["slide_id"] = slide_id
                        f.attrs["epoch"] = epoch
                        f.attrs["num_patches"] = valid_patches

                # Log attention statistics
                attn_mean = attention_weights.mean().item()
                attn_std = attention_weights.std().item()
                attn_max = attention_weights.max().item()
                attn_min = attention_weights.min().item()
                logger.debug(
                    f"Attention stats - mean: {attn_mean:.4f}, std: {attn_std:.4f}, "
                    f"max: {attn_max:.4f}, min: {attn_min:.4f}"
                )
        else:
            logits = model(features, num_patches)  # [batch_size, 1] or [batch_size, num_classes]

        if logits.ndim > 1 and logits.size(-1) == 1:
            logits = logits.squeeze(-1)  # [batch_size]

        # Compute loss (BCE for binary classification)
        loss = criterion(logits, labels.float())

        # Backward pass
        loss.backward()
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        preds = (torch.sigmoid(logits) > 0.5).long()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        pbar.set_postfix({"loss": f"{loss.item():.4f}"})

    # Compute epoch metrics
    avg_loss = total_loss / len(train_loader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
    }


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    save_attention: bool = False,
    attention_dir: Optional[Path] = None,
    epoch: Optional[int] = None,
) -> Dict[str, float]:
    """Validate the model.

    Args:
        model: Model to validate
        val_loader: Validation data loader
        criterion: Loss function
        device: Device to validate on
        save_attention: Whether to save attention weights
        attention_dir: Directory to save attention weights (required if save_attention=True)
        epoch: Current epoch number (for attention weight filenames)

    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []

    # Check if model supports return_attention parameter
    forward_signature = inspect.signature(model.forward)
    supports_attention = "return_attention" in forward_signature.parameters

    with torch.no_grad():
        pbar = tqdm(val_loader, desc="Validation")
        for batch in pbar:
            # Extract slide-level batch data
            features = batch["features"].to(device)  # [batch_size, max_patches, feature_dim]
            labels = batch["labels"].to(device)  # [batch_size]
            num_patches = batch["num_patches"].to(device)  # [batch_size]
            slide_ids = batch.get("slide_ids", [])
            patient_ids = batch.get("patient_ids", [])

            # Forward pass
            if supports_attention and save_attention:
                logits, attention_weights = model(features, num_patches, return_attention=True)

                # Save attention weights to HDF5 if requested
                if attention_dir is not None and len(slide_ids) > 0:
                    attention_dir.mkdir(parents=True, exist_ok=True)
                    for i, slide_id in enumerate(slide_ids):
                        # Only save valid patches (use num_patches to slice)
                        valid_patches = num_patches[i].item()
                        attn = attention_weights[i, :valid_patches].cpu().numpy()

                        # Get prediction and label
                        prob = torch.sigmoid(logits[i]).item()
                        pred = int(prob > 0.5)
                        label = labels[i].item()

                        # Save to HDF5 with metadata
                        epoch_str = f"_epoch{epoch}" if epoch is not None else ""
                        h5_path = attention_dir / f"{slide_id}{epoch_str}_val.h5"
                        with h5py.File(h5_path, "w") as f:
                            f.create_dataset("attention_weights", data=attn)
                            f.attrs["slide_id"] = slide_id
                            if len(patient_ids) > i:
                                f.attrs["patient_id"] = patient_ids[i]
                            f.attrs["label"] = label
                            f.attrs["prediction"] = pred
                            f.attrs["probability"] = prob
                            f.attrs["num_patches"] = valid_patches
                            if epoch is not None:
                                f.attrs["epoch"] = epoch
            else:
                logits = model(
                    features, num_patches
                )  # [batch_size, 1] or [batch_size, num_classes]

            if logits.ndim > 1 and logits.size(-1) == 1:
                logits = logits.squeeze(-1)  # [batch_size]
            loss = criterion(logits, labels.float())

            # Track metrics
            total_loss += loss.item()
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).long()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    # Compute metrics
    avg_loss = total_loss / len(val_loader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))

    # Compute AUC if we have both classes
    try:
        from sklearn.metrics import roc_auc_score

        auc = roc_auc_score(all_labels, all_probs)
    except (ValueError, RuntimeError):
        auc = 0.0

    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "auc": auc,
    }


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(
        description="Train CAMELYON16 slide-level classification model"
    )
    parser.add_argument("--config", type=str, required=True, help="Path to configuration file")
    parser.add_argument(
        "--resume", type=str, default=None, help="Path to checkpoint to resume from"
    )

    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Validate config
    validate_config(config)

    # Set seed
    set_seed(config.get("seed", 42))

    # Set device
    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Check if data exists
    root_dir = Path(config["data"]["root_dir"])
    index_path = root_dir / "slide_index.json"
    features_dir = root_dir / "features"

    if not index_path.exists() or not features_dir.exists():
        logger.error("=" * 80)
        logger.error("CAMELYON16 Training - Data Not Found")
        logger.error("=" * 80)
        logger.error("")
        logger.error("Required files not found:")
        logger.error(f"  - Slide index: {index_path} {'✓' if index_path.exists() else '✗'}")
        logger.error(f"  - Features dir: {features_dir} {'✓' if features_dir.exists() else '✗'}")
        logger.error("")
        logger.error("This training script requires pre-extracted patch features.")
        logger.error("")
        logger.error("Next steps:")
        logger.error("  1. Download CAMELYON16 dataset from grand-challenge.org")
        logger.error("  2. Create slide index using CAMELYONSlideIndex.from_directory()")
        logger.error("  3. Extract patch features to HDF5 files")
        logger.error("")
        logger.error("=" * 80)
        sys.exit(1)

    # Create dataloaders
    logger.info("Creating dataloaders...")
    train_loader, val_loader = create_slide_dataloaders(config)

    # Get feature dimension from first batch
    sample_batch = next(iter(train_loader))
    feature_dim = sample_batch["features"].shape[-1]
    logger.info(f"Feature dimension: {feature_dim}")

    # Create model
    logger.info("Creating model...")
    aggregation = config.get("model", {}).get("wsi", {}).get("aggregation", "mean")
    model = SimpleSlideClassifier(
        feature_dim=feature_dim,
        hidden_dim=config["model"]["wsi"]["hidden_dim"],
        num_classes=config["task"]["num_classes"],
        pooling=aggregation,
        dropout=config["task"]["classification"]["dropout"],
    ).to(device)

    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    logger.info(f"Aggregation method: {aggregation}")

    # Create optimizer and loss
    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config["training"]["learning_rate"]),
        weight_decay=float(config["training"]["weight_decay"]),
    )

    criterion = nn.BCEWithLogitsLoss()

    # Training loop
    num_epochs = config["training"]["num_epochs"]
    best_val_auc = 0.0

    logger.info("=" * 80)
    logger.info("Starting training...")
    logger.info("=" * 80)

    for epoch in range(1, num_epochs + 1):
        # Train
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, epoch)
        logger.info(
            f"Epoch {epoch}/{num_epochs} - "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Train Acc: {train_metrics['accuracy']:.4f}"
        )

        # Validate
        val_metrics = validate(model, val_loader, criterion, device)
        logger.info(
            f"Epoch {epoch}/{num_epochs} - "
            f"Val Loss: {val_metrics['loss']:.4f}, "
            f"Val Acc: {val_metrics['accuracy']:.4f}, "
            f"Val AUC: {val_metrics['auc']:.4f}"
        )

        # Save best model
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            checkpoint_dir = Path(config["checkpoint"]["checkpoint_dir"])
            checkpoint_dir.mkdir(parents=True, exist_ok=True)

            checkpoint_path = checkpoint_dir / "best_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_auc": best_val_auc,
                    "config": config,
                },
                checkpoint_path,
            )
            logger.info(f"Saved best model to {checkpoint_path}")

    logger.info("=" * 80)
    logger.info(f"Training complete! Best Val AUC: {best_val_auc:.4f}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
