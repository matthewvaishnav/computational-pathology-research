"""
PANDA (Prostate cANcer graDe Assessment) Training Script.

Train MIL models for ISUP grade prediction from prostate biopsy WSIs.
Supports ordinal regression and quadratic weighted kappa evaluation.

Usage:
    python experiments/train_panda.py --config configs/panda_phikon.yaml
    python experiments/train_panda.py --model nnmil --epochs 40 --batch_size 32
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.datasets.panda_dataset import (
    PANDASlideDataset,
    PANDASlideIndex,
    collate_panda_bags,
    compute_quadratic_weighted_kappa,
    validate_panda_dataset,
)
from src.models.mil.nnmil import nnMIL
from src.models.transnnmil.transnnmil import TransnnMIL

logger = logging.getLogger(__name__)


class OrdinalCrossEntropyLoss(nn.Module):
    """Ordinal cross-entropy loss for ordered classification.
    
    Treats ordinal labels as cumulative probabilities.
    For ISUP grade 3: [1,1,1,0,0,0]
    """

    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: [batch_size, num_classes] raw scores
            targets: [batch_size, num_classes] ordinal encoding
        """
        return self.bce(logits, targets)


def create_model(config: dict) -> nn.Module:
    """Create MIL model for PANDA.
    
    Args:
        config: Model configuration
        
    Returns:
        MIL model
    """
    model_type = config.get("model_type", "nnmil")
    feature_dim = config.get("input_dim", 256)  # input_dim from args maps to feature_dim in model
    hidden_dim = config.get("hidden_dim", 256)
    num_classes = config.get("num_classes", 6)
    ordinal = config.get("ordinal", False)
    
    if model_type == "nnmil":
        model = nnMIL(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            dropout=config.get("dropout", 0.2),
        )
    elif model_type == "transnnmil":
        model = TransnnMIL(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            num_layers=config.get("num_layers", 2),
            num_heads=config.get("num_heads", 8),
            dropout=config.get("dropout", 0.1),
            use_pos_encoding=config.get("use_pos_encoding", False),
        )
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    ordinal: bool = False,
) -> dict:
    """Train for one epoch.
    
    Args:
        model: MIL model
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to train on
        ordinal: Whether using ordinal regression
        
    Returns:
        Dictionary with training metrics
    """
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc="Training")
    for batch in pbar:
        features = batch["features"].to(device)  # [B, N, D]
        labels = batch["labels"].to(device)
        num_patches = batch["num_patches"].to(device)
        
        # Forward pass
        optimizer.zero_grad()
        logits = model(features, num_patches)  # [B, num_classes]
        
        if ordinal:
            # Ordinal loss
            loss = criterion(logits, labels)
            # Convert to class predictions
            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).sum(dim=1)
            true_labels = labels.sum(dim=1).long()
        else:
            # Standard cross-entropy
            loss = criterion(logits, labels)
            preds = logits.argmax(dim=1)
            true_labels = labels
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Track metrics
        total_loss += loss.item()
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(true_labels.cpu().numpy())
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    # Compute metrics
    avg_loss = total_loss / len(dataloader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = (all_preds == all_labels).mean()
    kappa = compute_quadratic_weighted_kappa(all_labels, all_preds)
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "kappa": kappa,
    }


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    ordinal: bool = False,
) -> dict:
    """Validate model.
    
    Args:
        model: MIL model
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device
        ordinal: Whether using ordinal regression
        
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []
    all_slide_ids = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            features = batch["features"].to(device)
            labels = batch["labels"].to(device)
            num_patches = batch["num_patches"].to(device)
            
            # Forward pass
            logits = model(features, num_patches)
            
            if ordinal:
                loss = criterion(logits, labels)
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).sum(dim=1)
                true_labels = labels.sum(dim=1).long()
            else:
                loss = criterion(logits, labels)
                preds = logits.argmax(dim=1)
                true_labels = labels
            
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(true_labels.cpu().numpy())
            all_slide_ids.extend(batch["slide_ids"])
    
    # Compute metrics
    avg_loss = total_loss / len(dataloader)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    accuracy = (all_preds == all_labels).mean()
    kappa = compute_quadratic_weighted_kappa(all_labels, all_preds)
    
    # Per-grade accuracy
    grade_acc = {}
    for grade in range(6):
        mask = all_labels == grade
        if mask.sum() > 0:
            grade_acc[f"grade_{grade}_acc"] = (all_preds[mask] == all_labels[mask]).mean()
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "kappa": kappa,
        **grade_acc,
        "predictions": all_preds,
        "labels": all_labels,
        "slide_ids": all_slide_ids,
    }


def main():
    parser = argparse.ArgumentParser(description="Train PANDA model")
    parser.add_argument("--data_dir", type=str, default="data/panda", help="PANDA data directory")
    parser.add_argument("--features_dir", type=str, default="data/panda/features", help="Features directory")
    parser.add_argument("--index_path", type=str, default="data/panda/slide_index.json", help="Slide index JSON")
    parser.add_argument("--model", type=str, default="transnnmil", choices=["nnmil", "transnnmil"], help="Model type")
    parser.add_argument("--input_dim", type=int, default=2048, help="Feature dimension")
    parser.add_argument("--hidden_dim", type=int, default=256, help="Hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of transformer layers (TransNNMIL only)")
    parser.add_argument("--num_heads", type=int, default=8, help="Number of attention heads (TransNNMIL only)")
    parser.add_argument("--ordinal", action="store_true", help="Use ordinal regression")
    parser.add_argument("--epochs", type=int, default=40, help="Number of epochs")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-4, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints/panda", help="Checkpoint directory")
    parser.add_argument("--log_dir", type=str, default="logs/panda", help="Log directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    # Setup
    log_dir = Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "train.log"),
            logging.StreamHandler(),
        ],
    )
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device(args.device)
    
    logger.info(f"Training PANDA model on {device}")
    logger.info(f"Arguments: {args}")
    
    # Load slide index
    logger.info(f"Loading slide index from {args.index_path}")
    if not Path(args.index_path).exists():
        logger.info("Index not found, creating from CSV...")
        csv_path = Path(args.data_dir) / "train.csv"
        image_dir = Path(args.data_dir) / "train_images"
        
        if not csv_path.exists():
            raise FileNotFoundError(
                f"CSV not found: {csv_path}\n"
                f"Please download PANDA dataset from Kaggle:\n"
                f"https://www.kaggle.com/c/prostate-cancer-grade-assessment"
            )
        
        slide_index = PANDASlideIndex.from_csv(
            csv_path=csv_path,
            image_dir=image_dir,
            stratify=True,
            seed=args.seed,
        )
        slide_index.save(args.index_path)
    else:
        slide_index = PANDASlideIndex.load(args.index_path)
    
    logger.info(f"Loaded {len(slide_index)} slides")
    logger.info(f"Grade distribution: {slide_index.get_grade_distribution()}")
    
    # Create datasets
    logger.info("Creating datasets...")
    train_dataset = PANDASlideDataset(
        slide_index=slide_index,
        features_dir=args.features_dir,
        split="train",
        ordinal=args.ordinal,
    )
    
    val_dataset = PANDASlideDataset(
        slide_index=slide_index,
        features_dir=args.features_dir,
        split="val",
        ordinal=args.ordinal,
    )
    
    # Validate datasets
    logger.info("Validating datasets...")
    train_stats = validate_panda_dataset(train_dataset)
    val_stats = validate_panda_dataset(val_dataset)
    logger.info(f"Train stats: {train_stats}")
    logger.info(f"Val stats: {val_stats}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_panda_bags,
        pin_memory=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_panda_bags,
        pin_memory=True,
    )
    
    # Create model
    logger.info("Creating model...")
    config = {
        "model_type": args.model,
        "input_dim": args.input_dim,
        "hidden_dim": args.hidden_dim,
        "num_classes": 6,
        "dropout": args.dropout,
        "ordinal": args.ordinal,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "use_pos_encoding": False,
    }
    model = create_model(config).to(device)
    logger.info(f"Model: {model}")
    
    # Loss and optimizer
    if args.ordinal:
        criterion = OrdinalCrossEntropyLoss()
    else:
        criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.epochs,
        eta_min=1e-6,
    )
    
    # Training loop
    best_kappa = -1.0
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("Starting training...")
    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")
        
        # Train
        train_metrics = train_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            ordinal=args.ordinal,
        )
        
        logger.info(
            f"Train - Loss: {train_metrics['loss']:.4f}, "
            f"Acc: {train_metrics['accuracy']:.4f}, "
            f"Kappa: {train_metrics['kappa']:.4f}"
        )
        
        # Validate
        val_metrics = validate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            ordinal=args.ordinal,
        )
        
        logger.info(
            f"Val - Loss: {val_metrics['loss']:.4f}, "
            f"Acc: {val_metrics['accuracy']:.4f}, "
            f"Kappa: {val_metrics['kappa']:.4f}"
        )
        
        # Log per-grade accuracy
        for grade in range(6):
            key = f"grade_{grade}_acc"
            if key in val_metrics:
                logger.info(f"  Grade {grade} Acc: {val_metrics[key]:.4f}")
        
        # Save best model
        if val_metrics["kappa"] > best_kappa:
            best_kappa = val_metrics["kappa"]
            checkpoint_path = checkpoint_dir / "best_model.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_kappa": best_kappa,
                    "val_accuracy": val_metrics["accuracy"],
                    "config": config,
                },
                checkpoint_path,
            )
            logger.info(f"Saved best model (kappa={best_kappa:.4f}) to {checkpoint_path}")
        
        # Save checkpoint every 5 epochs
        if (epoch + 1) % 5 == 0:
            checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pth"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_kappa": val_metrics["kappa"],
                    "config": config,
                },
                checkpoint_path,
            )
            logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Step scheduler
        scheduler.step()
    
    logger.info(f"\nTraining complete! Best validation kappa: {best_kappa:.4f}")


if __name__ == "__main__":
    main()
