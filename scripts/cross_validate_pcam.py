"""
K-Fold Cross-Validation for PatchCamelyon (PCam) Dataset

This script implements k-fold cross-validation to assess model robustness
and variance across different train/test splits.

Features:
- Stratified k-fold splitting to maintain class balance
- Bootstrap confidence intervals for each fold
- Aggregated statistics across all folds
- Comprehensive reporting with variance analysis
- Checkpoint saving for each fold
- Mixed precision training support
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.pcam_dataset import PCamDataset, get_pcam_transforms
from src.models.encoders import WSIEncoder
from src.models.feature_extractors import ResNetFeatureExtractor
from src.models.heads import ClassificationHead

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    import random
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_model(device: torch.device) -> nn.Module:
    """Create the model architecture."""
    feature_extractor = ResNetFeatureExtractor(
        model_name="resnet18",
        pretrained=True,
        freeze_backbone=False
    ).to(device)
    
    encoder = WSIEncoder(
        input_dim=512,  # ResNet18 feature dim
        hidden_dim=256,
        num_layers=2,
        dropout=0.1
    ).to(device)
    
    head = ClassificationHead(
        input_dim=256,
        num_classes=2,
        dropout=0.5
    ).to(device)
    
    class PCamModel(nn.Module):
        def __init__(self, feature_extractor, encoder, head):
            super().__init__()
            self.feature_extractor = feature_extractor
            self.encoder = encoder
            self.head = head
        
        def forward(self, x):
            features = self.feature_extractor(x)
            encoded = self.encoder(features.unsqueeze(1))
            logits = self.head(encoded.squeeze(1))
            return logits
    
    return PCamModel(feature_extractor, encoder, head)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    use_amp: bool = True
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    for batch_idx, (images, labels) in enumerate(tqdm(dataloader, desc="Training")):
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device
) -> Dict[str, float]:
    """Evaluate the model."""
    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating"):
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)
    
    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "auc": roc_auc_score(all_labels, all_probs),
        "f1": f1_score(all_labels, all_preds, average="macro"),
        "precision": precision_score(all_labels, all_preds, average="macro"),
        "recall": recall_score(all_labels, all_preds, average="macro"),
    }
    
    return metrics, all_preds, all_labels, all_probs


def bootstrap_ci(
    labels: np.ndarray,
    preds: np.ndarray,
    probs: np.ndarray,
    n_bootstrap: int = 1000,
    confidence_level: float = 0.95
) -> Dict[str, Dict[str, float]]:
    """Compute bootstrap confidence intervals."""
    np.random.seed(42)
    n_samples = len(labels)
    
    metrics_bootstrap = {
        "accuracy": [],
        "auc": [],
        "f1": [],
        "precision": [],
        "recall": []
    }
    
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        boot_labels = labels[indices]
        boot_preds = preds[indices]
        boot_probs = probs[indices]
        
        metrics_bootstrap["accuracy"].append(accuracy_score(boot_labels, boot_preds))
        metrics_bootstrap["auc"].append(roc_auc_score(boot_labels, boot_probs))
        metrics_bootstrap["f1"].append(f1_score(boot_labels, boot_preds, average="macro"))
        metrics_bootstrap["precision"].append(precision_score(boot_labels, boot_preds, average="macro"))
        metrics_bootstrap["recall"].append(recall_score(boot_labels, boot_preds, average="macro"))
    
    alpha = (1 - confidence_level) / 2
    ci_results = {}
    
    for metric_name, values in metrics_bootstrap.items():
        values = np.array(values)
        ci_results[metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "ci_lower": float(np.percentile(values, alpha * 100)),
            "ci_upper": float(np.percentile(values, (1 - alpha) * 100))
        }
    
    return ci_results


def train_fold(
    fold_idx: int,
    train_indices: np.ndarray,
    val_indices: np.ndarray,
    full_dataset: PCamDataset,
    args: argparse.Namespace,
    device: torch.device
) -> Dict:
    """Train and evaluate a single fold."""
    logger.info(f"\n{'='*80}")
    logger.info(f"Training Fold {fold_idx + 1}/{args.n_folds}")
    logger.info(f"{'='*80}")
    
    # Create data subsets
    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    # Create model
    model = create_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Training loop
    best_val_auc = 0.0
    fold_results = {
        "fold": fold_idx,
        "train_size": len(train_indices),
        "val_size": len(val_indices),
        "epochs_trained": 0,
        "best_epoch": 0,
        "train_losses": [],
        "val_metrics": []
    }
    
    for epoch in range(args.num_epochs):
        logger.info(f"\nFold {fold_idx + 1}, Epoch {epoch + 1}/{args.num_epochs}")
        
        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, args.use_amp
        )
        fold_results["train_losses"].append(train_loss)
        
        # Validate
        val_metrics, val_preds, val_labels, val_probs = evaluate(model, val_loader, device)
        fold_results["val_metrics"].append(val_metrics)
        
        logger.info(f"Train Loss: {train_loss:.4f}")
        logger.info(f"Val Accuracy: {val_metrics['accuracy']:.4f}")
        logger.info(f"Val AUC: {val_metrics['auc']:.4f}")
        logger.info(f"Val F1: {val_metrics['f1']:.4f}")
        
        # Save best model
        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            fold_results["best_epoch"] = epoch
            fold_results["best_val_metrics"] = val_metrics
            
            # Save checkpoint
            checkpoint_path = os.path.join(
                args.output_dir,
                f"fold_{fold_idx}_best_model.pth"
            )
            torch.save({
                "fold": fold_idx,
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_metrics": val_metrics,
            }, checkpoint_path)
            logger.info(f"Saved best model for fold {fold_idx} (AUC: {best_val_auc:.4f})")
        
        fold_results["epochs_trained"] = epoch + 1
    
    # Final evaluation with bootstrap CI
    logger.info(f"\nComputing bootstrap confidence intervals for fold {fold_idx}...")
    model.load_state_dict(
        torch.load(os.path.join(args.output_dir, f"fold_{fold_idx}_best_model.pth"))["model_state_dict"]
    )
    final_metrics, final_preds, final_labels, final_probs = evaluate(model, val_loader, device)
    
    ci_results = bootstrap_ci(
        final_labels, final_preds, final_probs,
        n_bootstrap=args.bootstrap_samples,
        confidence_level=0.95
    )
    
    fold_results["final_metrics"] = final_metrics
    fold_results["bootstrap_ci"] = ci_results
    
    return fold_results


def aggregate_results(fold_results: List[Dict]) -> Dict:
    """Aggregate results across all folds."""
    metrics_names = ["accuracy", "auc", "f1", "precision", "recall"]
    
    aggregated = {
        "n_folds": len(fold_results),
        "metrics": {}
    }
    
    for metric_name in metrics_names:
        values = [fold["final_metrics"][metric_name] for fold in fold_results]
        
        aggregated["metrics"][metric_name] = {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(np.min(values)),
            "max": float(np.max(values)),
            "values": [float(v) for v in values]
        }
    
    # Aggregate bootstrap CIs
    aggregated["bootstrap_ci_aggregated"] = {}
    for metric_name in metrics_names:
        ci_means = [fold["bootstrap_ci"][metric_name]["mean"] for fold in fold_results]
        ci_stds = [fold["bootstrap_ci"][metric_name]["std"] for fold in fold_results]
        ci_lowers = [fold["bootstrap_ci"][metric_name]["ci_lower"] for fold in fold_results]
        ci_uppers = [fold["bootstrap_ci"][metric_name]["ci_upper"] for fold in fold_results]
        
        aggregated["bootstrap_ci_aggregated"][metric_name] = {
            "mean_across_folds": float(np.mean(ci_means)),
            "std_across_folds": float(np.std(ci_means)),
            "ci_lower_mean": float(np.mean(ci_lowers)),
            "ci_upper_mean": float(np.mean(ci_uppers)),
            "ci_width_mean": float(np.mean(np.array(ci_uppers) - np.array(ci_lowers)))
        }
    
    return aggregated


def main():
    parser = argparse.ArgumentParser(description="K-Fold Cross-Validation for PCam")
    parser.add_argument("--data-root", type=str, required=True, help="Path to PCam data")
    parser.add_argument("--output-dir", type=str, default="results/pcam_cv", help="Output directory")
    parser.add_argument("--n-folds", type=int, default=5, help="Number of folds")
    parser.add_argument("--num-epochs", type=int, default=10, help="Epochs per fold")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--learning-rate", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--num-workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--bootstrap-samples", type=int, default=1000, help="Bootstrap samples")
    parser.add_argument("--use-amp", action="store_true", help="Use mixed precision")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--subset-size", type=int, default=None, help="Use subset for quick test")
    
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load full dataset (train + val combined for cross-validation)
    logger.info("Loading PCam dataset...")
    train_dataset = PCamDataset(
        root_dir=args.data_root,
        split="train",
        transform=get_pcam_transforms(split="train", augmentation=True)
    )
    
    val_dataset = PCamDataset(
        root_dir=args.data_root,
        split="val",
        transform=get_pcam_transforms(split="val", augmentation=False)
    )
    
    # Combine train and val for cross-validation
    from torch.utils.data import ConcatDataset
    full_dataset = ConcatDataset([train_dataset, val_dataset])
    
    # Get all labels for stratification
    all_labels = []
    for i in range(len(train_dataset)):
        _, label = train_dataset[i]
        all_labels.append(label)
    for i in range(len(val_dataset)):
        _, label = val_dataset[i]
        all_labels.append(label)
    all_labels = np.array(all_labels)
    
    # Use subset if specified (for quick testing)
    if args.subset_size:
        logger.info(f"Using subset of {args.subset_size} samples for quick test")
        indices = np.random.choice(len(all_labels), args.subset_size, replace=False)
        full_dataset = Subset(full_dataset, indices)
        all_labels = all_labels[indices]
    
    logger.info(f"Total samples: {len(full_dataset)}")
    logger.info(f"Class distribution: {np.bincount(all_labels)}")
    
    # Create stratified k-fold splits
    skf = StratifiedKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    
    # Train each fold
    fold_results = []
    for fold_idx, (train_indices, val_indices) in enumerate(skf.split(np.arange(len(all_labels)), all_labels)):
        fold_result = train_fold(
            fold_idx, train_indices, val_indices, full_dataset, args, device
        )
        fold_results.append(fold_result)
        
        # Save fold results
        fold_output_path = os.path.join(args.output_dir, f"fold_{fold_idx}_results.json")
        with open(fold_output_path, "w") as f:
            json.dump(fold_result, f, indent=2)
        logger.info(f"Saved fold {fold_idx} results to {fold_output_path}")
    
    # Aggregate results
    logger.info("\n" + "="*80)
    logger.info("Aggregating results across all folds...")
    logger.info("="*80)
    
    aggregated_results = aggregate_results(fold_results)
    aggregated_results["fold_results"] = fold_results
    
    # Save aggregated results
    output_path = os.path.join(args.output_dir, "cross_validation_results.json")
    with open(output_path, "w") as f:
        json.dump(aggregated_results, f, indent=2)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("CROSS-VALIDATION SUMMARY")
    logger.info("="*80)
    logger.info(f"Number of folds: {args.n_folds}")
    logger.info(f"Epochs per fold: {args.num_epochs}")
    logger.info(f"Total samples: {len(full_dataset)}")
    logger.info("")
    
    for metric_name, stats in aggregated_results["metrics"].items():
        logger.info(f"{metric_name.upper()}:")
        logger.info(f"  Mean: {stats['mean']:.4f} ± {stats['std']:.4f}")
        logger.info(f"  Range: [{stats['min']:.4f}, {stats['max']:.4f}]")
        logger.info(f"  Values: {[f'{v:.4f}' for v in stats['values']]}")
        logger.info("")
    
    logger.info("Bootstrap CI Aggregated:")
    for metric_name, stats in aggregated_results["bootstrap_ci_aggregated"].items():
        logger.info(f"{metric_name.upper()}:")
        logger.info(f"  Mean: {stats['mean_across_folds']:.4f} ± {stats['std_across_folds']:.4f}")
        logger.info(f"  CI Range: [{stats['ci_lower_mean']:.4f}, {stats['ci_upper_mean']:.4f}]")
        logger.info(f"  CI Width: {stats['ci_width_mean']:.4f}")
        logger.info("")
    
    logger.info(f"Results saved to {output_path}")
    logger.info("Cross-validation complete!")


if __name__ == "__main__":
    main()
