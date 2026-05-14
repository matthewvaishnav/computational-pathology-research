"""
TransnnMIL v2.0 Training Script: End-to-End Hierarchical Pipeline

Trains TransnnMIL with hierarchical pooling on MIL datasets.

Features:
- Hierarchical spatial clustering (learnable/kmeans/grid)
- Intra-region aggregation (attention/mean/max)
- Inter-region transformer
- Dual-branch fusion (TransMIL + nnMIL)
- Feature-level cross-attention fusion

Usage:
    # Train with hierarchical pooling (16 regions, learnable clustering, attention pooling)
    python scripts/train_v2_0.py --dataset tcga-brca --num_regions 16 --clustering learnable --pooling attention
    
    # Train with k-means clustering
    python scripts/train_v2_0.py --dataset tcga-brca --num_regions 32 --clustering kmeans
    
    # Train with grid clustering + mean pooling
    python scripts/train_v2_0.py --dataset tcga-brca --num_regions 16 --clustering grid --pooling mean
    
    # Train without hierarchical pooling (baseline)
    python scripts/train_v2_0.py --dataset tcga-brca --no_hierarchical

Reference:
- TransnnMIL v2.0: Hierarchical + Topology (2027)
"""

import argparse
import json
import logging
import random
import sys
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.transnnmil import TransnnMIL
from src.utils.monitoring import get_logger

logger = get_logger(__name__)


def set_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class DummyMILDataset(torch.utils.data.Dataset):
    """
    Dummy MIL dataset for testing hierarchical pipeline.
    
    Generates random bags with features + coordinates.
    Replace with real dataset (TCGA, PANDA, etc.) for actual training.
    """
    
    def __init__(
        self,
        num_samples: int = 100,
        num_patches: int = 500,
        feature_dim: int = 1024,
        num_classes: int = 2,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.num_patches = num_patches
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        
        # Set seed for reproducible data
        rng = np.random.RandomState(seed)
        
        # Generate data
        self.features = []
        self.coords = []
        self.labels = []
        
        for _ in range(num_samples):
            # Random features
            feat = torch.randn(num_patches, feature_dim)
            self.features.append(feat)
            
            # Random coordinates [0, 1]
            coord = torch.rand(num_patches, 2)
            self.coords.append(coord)
            
            # Random label
            label = rng.randint(0, num_classes)
            self.labels.append(label)
    
    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        return {
            'features': self.features[idx],
            'coordinates': self.coords[idx],
            'label': self.labels[idx],
            'num_patches': self.num_patches,
        }


def collate_fn(batch):
    """Collate function for MIL batches."""
    features = torch.stack([item['features'] for item in batch])
    coords = torch.stack([item['coordinates'] for item in batch])
    labels = torch.tensor([item['label'] for item in batch])
    num_patches = torch.tensor([item['num_patches'] for item in batch])
    
    return {
        'features': features,
        'coordinates': coords,
        'labels': labels,
        'num_patches': num_patches,
    }


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    epoch: int,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    
    for batch in pbar:
        # Move to device
        features = batch['features'].to(device)
        coords = batch['coordinates'].to(device)
        labels = batch['labels'].to(device)
        num_patches = batch['num_patches'].to(device)
        
        # Forward
        optimizer.zero_grad()
        logits = model(features, num_patches, coordinates=coords)
        
        # Loss
        loss = criterion(logits, labels)
        
        # Backward
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # Update progress
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.0 * correct / total:.2f}%',
        })
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': 100.0 * correct / total,
    }


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    split: str = 'Val',
) -> Dict[str, float]:
    """Evaluate model."""
    model.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    pbar = tqdm(dataloader, desc=f'{split}')
    
    for batch in pbar:
        # Move to device
        features = batch['features'].to(device)
        coords = batch['coordinates'].to(device)
        labels = batch['labels'].to(device)
        num_patches = batch['num_patches'].to(device)
        
        # Forward
        logits = model(features, num_patches, coordinates=coords)
        
        # Loss
        loss = criterion(logits, labels)
        
        # Metrics
        total_loss += loss.item()
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # Update progress
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.0 * correct / total:.2f}%',
        })
    
    return {
        'loss': total_loss / len(dataloader),
        'accuracy': 100.0 * correct / total,
    }


def main():
    parser = argparse.ArgumentParser(description='Train TransnnMIL v2.0')
    
    # Model args
    parser.add_argument('--feature_dim', type=int, default=1024,
                       help='Feature dimension (default: 1024)')
    parser.add_argument('--hidden_dim', type=int, default=256,
                       help='Hidden dimension (default: 256)')
    parser.add_argument('--num_classes', type=int, default=2,
                       help='Number of classes (default: 2)')
    parser.add_argument('--num_layers', type=int, default=2,
                       help='Transformer layers (default: 2)')
    parser.add_argument('--num_heads', type=int, default=8,
                       help='Attention heads (default: 8)')
    parser.add_argument('--dropout', type=float, default=0.1,
                       help='Dropout rate (default: 0.1)')
    
    # Hierarchical args
    parser.add_argument('--no_hierarchical', action='store_true',
                       help='Disable hierarchical pooling (baseline)')
    parser.add_argument('--num_regions', type=int, default=16,
                       help='Number of spatial regions (default: 16)')
    parser.add_argument('--region_hidden_dim', type=int, default=512,
                       help='Region feature dimension (default: 512)')
    parser.add_argument('--clustering', type=str, default='learnable',
                       choices=['learnable', 'kmeans', 'grid'],
                       help='Clustering method (default: learnable)')
    parser.add_argument('--pooling', type=str, default='attention',
                       choices=['attention', 'mean', 'max'],
                       help='Intra-region pooling (default: attention)')
    parser.add_argument('--temperature', type=float, default=1.0,
                       help='Clustering temperature (default: 1.0)')
    
    # Training args
    parser.add_argument('--dataset', type=str, default='dummy',
                       help='Dataset name (default: dummy)')
    parser.add_argument('--batch_size', type=int, default=4,
                       help='Batch size (default: 4)')
    parser.add_argument('--num_epochs', type=int, default=10,
                       help='Number of epochs (default: 10)')
    parser.add_argument('--lr', type=float, default=1e-4,
                       help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-5,
                       help='Weight decay (default: 1e-5)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device (default: cuda)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    # Output args
    parser.add_argument('--output_dir', type=str, default='outputs/v2_0',
                       help='Output directory (default: outputs/v2_0)')
    parser.add_argument('--save_every', type=int, default=5,
                       help='Save checkpoint every N epochs (default: 5)')
    
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    logger.info(f'Device: {device}')
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    logger.info(f'Saved config to {config_path}')
    
    # Create datasets (dummy for now)
    logger.info('Creating datasets...')
    train_dataset = DummyMILDataset(
        num_samples=100,
        num_patches=500,
        feature_dim=args.feature_dim,
        num_classes=args.num_classes,
        seed=args.seed,
    )
    val_dataset = DummyMILDataset(
        num_samples=20,
        num_patches=500,
        feature_dim=args.feature_dim,
        num_classes=args.num_classes,
        seed=args.seed + 1,
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_fn,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn,
    )
    
    logger.info(f'Train samples: {len(train_dataset)}')
    logger.info(f'Val samples: {len(val_dataset)}')
    
    # Create model
    logger.info('Creating model...')
    model = TransnnMIL(
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        use_pos_encoding=False,
        enable_hierarchical=not args.no_hierarchical,
        num_regions=args.num_regions,
        region_hidden_dim=args.region_hidden_dim,
        clustering_method=args.clustering,
        pooling_method=args.pooling,
        temperature=args.temperature,
    )
    model = model.to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Total parameters: {total_params:,}')
    logger.info(f'Trainable parameters: {trainable_params:,}')
    
    # Print model config
    logger.info(f'Hierarchical pooling: {not args.no_hierarchical}')
    if not args.no_hierarchical:
        logger.info(f'  Num regions: {args.num_regions}')
        logger.info(f'  Clustering: {args.clustering}')
        logger.info(f'  Pooling: {args.pooling}')
        logger.info(f'  Temperature: {args.temperature}')
    
    # Optimizer & criterion
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    logger.info('Starting training...')
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
    }
    
    best_val_acc = 0.0
    
    for epoch in range(1, args.num_epochs + 1):
        # Train
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        
        # Validate
        val_metrics = evaluate(
            model, val_loader, criterion, device, split='Val'
        )
        
        # Log
        logger.info(
            f'Epoch {epoch}/{args.num_epochs} | '
            f'Train Loss: {train_metrics["loss"]:.4f} | '
            f'Train Acc: {train_metrics["accuracy"]:.2f}% | '
            f'Val Loss: {val_metrics["loss"]:.4f} | '
            f'Val Acc: {val_metrics["accuracy"]:.2f}%'
        )
        
        # Save history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        
        # Save best checkpoint
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            checkpoint_path = output_dir / 'checkpoint_best.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': best_val_acc,
                'config': vars(args),
            }, checkpoint_path)
            logger.info(f'Saved best checkpoint (val_acc={best_val_acc:.2f}%)')
        
        # Save periodic checkpoint
        if epoch % args.save_every == 0:
            checkpoint_path = output_dir / f'checkpoint_epoch_{epoch}.pth'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'config': vars(args),
            }, checkpoint_path)
            logger.info(f'Saved checkpoint at epoch {epoch}')
    
    # Save final history
    history_path = output_dir / 'history.json'
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    logger.info(f'Saved training history to {history_path}')
    
    logger.info(f'Training complete! Best val acc: {best_val_acc:.2f}%')


if __name__ == '__main__':
    main()
