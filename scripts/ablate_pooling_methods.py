"""
Ablation study: Compare intra-region pooling methods.

Compares:
- Attention pooling (learned)
- Mean pooling (baseline)
- Max pooling (baseline)

Metrics:
- AUC, Accuracy, F1
- Training time
- Memory usage

Usage:
    python scripts/ablate_pooling_methods.py --dataset tcga_brca --num_regions 16
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.hierarchical_pooling import (
    HierarchicalPooling,
    RegionAttentionPooling,
    RegionMeanPooling,
    RegionMaxPooling,
)


class SimpleClassifier(nn.Module):
    """Simple classifier for ablation study."""
    
    def __init__(
        self,
        feature_dim: int,
        num_regions: int,
        num_classes: int,
        pooling_type: str,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.num_regions = num_regions
        self.num_classes = num_classes
        self.pooling_type = pooling_type
        
        # Hierarchical clustering
        self.hierarchical = HierarchicalPooling(
            num_clusters=num_regions,
            temperature=1.0,
            init_method='uniform',
        )
        
        # Region pooling
        if pooling_type == 'attention':
            self.region_pooling = RegionAttentionPooling(
                feature_dim=feature_dim,
                hidden_dim=128,
                dropout=0.1,
            )
        elif pooling_type == 'mean':
            self.region_pooling = RegionMeanPooling()
        elif pooling_type == 'max':
            self.region_pooling = RegionMaxPooling()
        else:
            raise ValueError(f"Unknown pooling type: {pooling_type}")
        
        # Classifier head
        self.classifier = nn.Sequential(
            nn.Linear(feature_dim * num_regions, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes),
        )
    
    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: Patch features [B, N, D]
            coords: Patch coordinates [B, N, 2]
        
        Returns:
            logits: Class logits [B, num_classes]
        """
        # Get region assignments
        assignments = self.hierarchical(coords)  # [B, N, K]
        
        # Pool within regions
        region_features = self.region_pooling(features, assignments)  # [B, K, D]
        
        # Flatten
        region_features = region_features.flatten(1)  # [B, K*D]
        
        # Classify
        logits = self.classifier(region_features)  # [B, num_classes]
        
        return logits


def create_synthetic_data(
    num_samples: int,
    num_patches: int,
    feature_dim: int,
    num_classes: int,
    seed: int = 42,
) -> Tuple[TensorDataset, TensorDataset]:
    """
    Create synthetic MIL dataset for ablation.
    
    Args:
        num_samples: Number of bags
        num_patches: Patches per bag
        feature_dim: Feature dimension
        num_classes: Number of classes
        seed: Random seed
    
    Returns:
        train_dataset, val_dataset
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Generate features
    features = torch.randn(num_samples, num_patches, feature_dim)
    
    # Generate coordinates (normalized to [0, 1])
    coords = torch.rand(num_samples, num_patches, 2)
    
    # Generate labels (balanced)
    labels = torch.randint(0, num_classes, (num_samples,))
    
    # Add class-specific signal to features
    for i in range(num_samples):
        label = labels[i].item()
        # Add signal to random patches
        signal_patches = np.random.choice(num_patches, size=10, replace=False)
        features[i, signal_patches] += label * 0.5
    
    # Split train/val (80/20)
    split_idx = int(0.8 * num_samples)
    
    train_dataset = TensorDataset(
        features[:split_idx],
        coords[:split_idx],
        labels[:split_idx],
    )
    
    val_dataset = TensorDataset(
        features[split_idx:],
        coords[split_idx:],
        labels[split_idx:],
    )
    
    return train_dataset, val_dataset


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: str,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    
    total_loss = 0.0
    all_preds = []
    all_labels = []
    
    for features, coords, labels in dataloader:
        features = features.to(device)
        coords = coords.to(device)
        labels = labels.to(device)
        
        # Forward
        logits = model(features, coords)
        loss = criterion(logits, labels)
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Metrics
        total_loss += loss.item()
        preds = logits.argmax(dim=1).cpu().numpy()
        all_preds.extend(preds)
        all_labels.extend(labels.cpu().numpy())
    
    # Compute metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
    }


def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
    num_classes: int,
) -> Dict[str, float]:
    """Evaluate model."""
    model.eval()
    
    total_loss = 0.0
    all_preds = []
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for features, coords, labels in dataloader:
            features = features.to(device)
            coords = coords.to(device)
            labels = labels.to(device)
            
            # Forward
            logits = model(features, coords)
            loss = criterion(logits, labels)
            
            # Metrics
            total_loss += loss.item()
            probs = torch.softmax(logits, dim=1).cpu().numpy()
            preds = logits.argmax(dim=1).cpu().numpy()
            
            all_probs.append(probs)
            all_preds.extend(preds)
            all_labels.extend(labels.cpu().numpy())
    
    # Concatenate
    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Compute metrics
    avg_loss = total_loss / len(dataloader)
    accuracy = accuracy_score(all_labels, all_preds)
    f1 = f1_score(all_labels, all_preds, average='macro')
    
    # AUC (one-vs-rest for multiclass)
    if num_classes == 2:
        auc = roc_auc_score(all_labels, all_probs[:, 1])
    else:
        auc = roc_auc_score(
            all_labels,
            all_probs,
            multi_class='ovr',
            average='macro',
        )
    
    return {
        'loss': avg_loss,
        'accuracy': accuracy,
        'f1': f1,
        'auc': auc,
    }


def run_ablation(
    pooling_type: str,
    train_dataset: TensorDataset,
    val_dataset: TensorDataset,
    feature_dim: int,
    num_regions: int,
    num_classes: int,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    device: str,
) -> Dict[str, float]:
    """
    Run ablation for one pooling method.
    
    Args:
        pooling_type: 'attention', 'mean', or 'max'
        train_dataset: Training dataset
        val_dataset: Validation dataset
        feature_dim: Feature dimension
        num_regions: Number of regions
        num_classes: Number of classes
        num_epochs: Training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        device: Device
    
    Returns:
        Results dictionary
    """
    print(f"\n{'='*60}")
    print(f"Pooling Method: {pooling_type.upper()}")
    print(f"{'='*60}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
    )
    
    # Create model
    model = SimpleClassifier(
        feature_dim=feature_dim,
        num_regions=num_regions,
        num_classes=num_classes,
        pooling_type=pooling_type,
    ).to(device)
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {num_params:,}")
    
    # Optimizer & loss
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_val_auc = 0.0
    train_times = []
    
    for epoch in range(num_epochs):
        # Train
        start_time = time.time()
        train_metrics = train_epoch(model, train_loader, optimizer, criterion, device)
        train_time = time.time() - start_time
        train_times.append(train_time)
        
        # Validate
        val_metrics = evaluate(model, val_loader, criterion, device, num_classes)
        
        # Track best
        if val_metrics['auc'] > best_val_auc:
            best_val_auc = val_metrics['auc']
        
        # Log
        if (epoch + 1) % 5 == 0 or epoch == 0:
            print(
                f"Epoch {epoch+1}/{num_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Train Acc: {train_metrics['accuracy']:.4f} | "
                f"Val AUC: {val_metrics['auc']:.4f} | "
                f"Val Acc: {val_metrics['accuracy']:.4f} | "
                f"Time: {train_time:.2f}s"
            )
    
    # Final evaluation
    final_metrics = evaluate(model, val_loader, criterion, device, num_classes)
    
    # Compute memory usage (approximate)
    if device == 'cuda':
        memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
        torch.cuda.reset_peak_memory_stats()
    else:
        memory_mb = 0.0
    
    # Results
    results = {
        'pooling_type': pooling_type,
        'num_params': num_params,
        'best_val_auc': best_val_auc,
        'final_val_auc': final_metrics['auc'],
        'final_val_acc': final_metrics['accuracy'],
        'final_val_f1': final_metrics['f1'],
        'avg_train_time': np.mean(train_times),
        'total_train_time': np.sum(train_times),
        'memory_mb': memory_mb,
    }
    
    print(f"\nResults:")
    print(f"  Best Val AUC: {best_val_auc:.4f}")
    print(f"  Final Val AUC: {final_metrics['auc']:.4f}")
    print(f"  Final Val Acc: {final_metrics['accuracy']:.4f}")
    print(f"  Final Val F1: {final_metrics['f1']:.4f}")
    print(f"  Avg Train Time: {np.mean(train_times):.2f}s/epoch")
    print(f"  Memory: {memory_mb:.1f} MB")
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Ablate pooling methods')
    parser.add_argument('--num_samples', type=int, default=500, help='Number of samples')
    parser.add_argument('--num_patches', type=int, default=200, help='Patches per bag')
    parser.add_argument('--feature_dim', type=int, default=512, help='Feature dimension')
    parser.add_argument('--num_regions', type=int, default=16, help='Number of regions')
    parser.add_argument('--num_classes', type=int, default=2, help='Number of classes')
    parser.add_argument('--num_epochs', type=int, default=20, help='Training epochs')
    parser.add_argument('--batch_size', type=int, default=8, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--output_dir', type=str, default='experiments/results/pooling_ablation')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    args = parser.parse_args()
    
    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*60)
    print("POOLING METHOD ABLATION STUDY")
    print("="*60)
    print(f"Samples: {args.num_samples}")
    print(f"Patches per bag: {args.num_patches}")
    print(f"Feature dim: {args.feature_dim}")
    print(f"Regions: {args.num_regions}")
    print(f"Classes: {args.num_classes}")
    print(f"Epochs: {args.num_epochs}")
    print(f"Device: {args.device}")
    
    # Create synthetic data
    print("\nCreating synthetic dataset...")
    train_dataset, val_dataset = create_synthetic_data(
        num_samples=args.num_samples,
        num_patches=args.num_patches,
        feature_dim=args.feature_dim,
        num_classes=args.num_classes,
        seed=args.seed,
    )
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Run ablations
    pooling_methods = ['attention', 'mean', 'max']
    all_results = []
    
    for pooling_type in pooling_methods:
        results = run_ablation(
            pooling_type=pooling_type,
            train_dataset=train_dataset,
            val_dataset=val_dataset,
            feature_dim=args.feature_dim,
            num_regions=args.num_regions,
            num_classes=args.num_classes,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=args.device,
        )
        all_results.append(results)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"{'Method':<12} {'Val AUC':<10} {'Val Acc':<10} {'Val F1':<10} {'Time (s)':<12} {'Memory (MB)':<12}")
    print("-"*60)
    
    for res in all_results:
        print(
            f"{res['pooling_type']:<12} "
            f"{res['final_val_auc']:<10.4f} "
            f"{res['final_val_acc']:<10.4f} "
            f"{res['final_val_f1']:<10.4f} "
            f"{res['avg_train_time']:<12.2f} "
            f"{res['memory_mb']:<12.1f}"
        )
    
    # Find best
    best_result = max(all_results, key=lambda x: x['final_val_auc'])
    print(f"\nBest method: {best_result['pooling_type'].upper()} (AUC: {best_result['final_val_auc']:.4f})")
    
    # Save results
    results_path = output_dir / 'pooling_ablation_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
