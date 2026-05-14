"""
Ablation Study: Clustering Methods (Task 4.3)

Ablates clustering methods: learnable vs k-means vs grid
Compares performance, training dynamics, and learned representations.

Usage:
    python scripts/ablate_clustering_methods.py --dataset tcga-brca --num_epochs 20
    python scripts/ablate_clustering_methods.py --dataset dummy --num_epochs 5 --quick

Output:
    experiments/v2_0/ablations/clustering/
    ├── results.json
    ├── plots/
    │   ├── accuracy_comparison.png
    │   ├── loss_curves.png
    │   └── cluster_centers_*.png
    └── checkpoints/
        ├── learnable/
        ├── kmeans/
        └── grid/
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from train_v2_0 import (
    DummyMILDataset,
    collate_fn,
    train_epoch,
    evaluate,
    set_seed,
)
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.transnnmil import TransnnMIL
from src.utils.monitoring import get_logger

logger = get_logger(__name__)


def run_ablation(
    clustering_method: str,
    args: argparse.Namespace,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
):
    """Run single ablation experiment."""
    logger.info(f'\n{"="*60}')
    logger.info(f'Ablation: clustering_method={clustering_method}')
    logger.info(f'{"="*60}')
    
    # Create model
    model = TransnnMIL(
        feature_dim=args.feature_dim,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        use_pos_encoding=False,
        enable_hierarchical=True,
        num_regions=args.num_regions,
        region_hidden_dim=args.region_hidden_dim,
        clustering_method=clustering_method,
        pooling_method=args.pooling,
        temperature=args.temperature,
    )
    model = model.to(device)
    
    # K-means requires fitting on first batch
    if clustering_method == 'kmeans':
        logger.info('Fitting k-means on first batch...')
        first_batch = next(iter(train_loader))
        coords = first_batch['coordinates'].to(device)
        model.hierarchical_pooling.clusterer.fit(coords)
        logger.info('K-means fitted')
    
    # Count params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f'Total parameters: {total_params:,}')
    logger.info(f'Trainable parameters: {trainable_params:,}')
    
    # Optimizer & criterion
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    criterion = nn.CrossEntropyLoss()
    
    # Training
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': [],
        'epoch_time': [],
    }
    
    best_val_acc = 0.0
    best_epoch = 0
    
    for epoch in range(1, args.num_epochs + 1):
        # Train
        start_time = time.time()
        train_metrics = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )
        epoch_time = time.time() - start_time
        
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
            f'Val Acc: {val_metrics["accuracy"]:.2f}% | '
            f'Time: {epoch_time:.2f}s'
        )
        
        # Save history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['epoch_time'].append(epoch_time)
        
        # Track best
        if val_metrics['accuracy'] > best_val_acc:
            best_val_acc = val_metrics['accuracy']
            best_epoch = epoch
    
    # Save checkpoint
    checkpoint_dir = output_dir / 'checkpoints' / clustering_method
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / 'checkpoint_best.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'clustering_method': clustering_method,
        'val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'history': history,
    }, checkpoint_path)
    
    # Extract cluster centers for visualization
    if hasattr(model.hierarchical_pooling, 'clusterer'):
        cluster_centers = model.hierarchical_pooling.clusterer.get_centers().cpu().numpy()
    else:
        cluster_centers = None
    
    logger.info(f'Best val acc: {best_val_acc:.2f}% (epoch {best_epoch})')
    logger.info(f'Avg epoch time: {np.mean(history["epoch_time"]):.2f}s')
    
    return {
        'clustering_method': clustering_method,
        'best_val_acc': best_val_acc,
        'best_epoch': best_epoch,
        'final_train_acc': history['train_acc'][-1],
        'final_val_acc': history['val_acc'][-1],
        'avg_epoch_time': np.mean(history['epoch_time']),
        'total_params': total_params,
        'trainable_params': trainable_params,
        'history': history,
        'cluster_centers': cluster_centers.tolist() if cluster_centers is not None else None,
    }


def plot_cluster_centers(results: list, output_dir: Path):
    """Visualize learned cluster centers."""
    plot_dir = output_dir / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    for idx, result in enumerate(results):
        method = result['clustering_method']
        centers = result.get('cluster_centers')
        
        if centers is None:
            continue
        
        centers = np.array(centers)
        ax = axes[idx]
        
        # Plot centers
        ax.scatter(centers[:, 0], centers[:, 1], s=200, c='red', 
                  marker='x', linewidths=3, label='Cluster Centers')
        
        # Add labels
        for i, (x, y) in enumerate(centers):
            ax.annotate(f'{i}', (x, y), fontsize=8, ha='center', va='bottom')
        
        ax.set_xlim(-0.1, 1.1)
        ax.set_ylim(-0.1, 1.1)
        ax.set_xlabel('X Coordinate', fontsize=12)
        ax.set_ylabel('Y Coordinate', fontsize=12)
        ax.set_title(f'{method.upper()}\n(Val Acc: {result["best_val_acc"]:.2f}%)', 
                    fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(plot_dir / 'cluster_centers_comparison.png', dpi=300)
    plt.close()
    
    logger.info(f'Cluster centers plot saved to {plot_dir}')


def plot_training_curves(results: list, output_dir: Path):
    """Plot training curves for all methods."""
    plot_dir = output_dir / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Loss curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for result in results:
        method = result['clustering_method']
        history = result['history']
        epochs = range(1, len(history['train_loss']) + 1)
        
        # Train loss
        ax1.plot(epochs, history['train_loss'], label=method, linewidth=2)
        
        # Val loss
        ax2.plot(epochs, history['val_loss'], label=method, linewidth=2)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Validation Loss', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plot_dir / 'loss_curves.png', dpi=300)
    plt.close()
    
    # Accuracy curves
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    for result in results:
        method = result['clustering_method']
        history = result['history']
        epochs = range(1, len(history['train_acc']) + 1)
        
        # Train acc
        ax1.plot(epochs, history['train_acc'], label=method, linewidth=2)
        
        # Val acc
        ax2.plot(epochs, history['val_acc'], label=method, linewidth=2)
    
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Training Accuracy', fontsize=14, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(plot_dir / 'accuracy_curves.png', dpi=300)
    plt.close()
    
    logger.info(f'Training curves saved to {plot_dir}')


def plot_comparison_bar(results: list, output_dir: Path):
    """Bar chart comparing methods."""
    plot_dir = output_dir / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    methods = [r['clustering_method'] for r in results]
    best_val_acc = [r['best_val_acc'] for r in results]
    avg_time = [r['avg_epoch_time'] for r in results]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy comparison
    bars1 = ax1.bar(methods, best_val_acc, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax1.set_ylabel('Best Val Accuracy (%)', fontsize=12)
    ax1.set_title('Accuracy Comparison', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}%', ha='center', va='bottom', fontsize=10)
    
    # Time comparison
    bars2 = ax2.bar(methods, avg_time, color=['#1f77b4', '#ff7f0e', '#2ca02c'])
    ax2.set_ylabel('Avg Epoch Time (s)', fontsize=12)
    ax2.set_title('Training Time Comparison', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar in bars2:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}s', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(plot_dir / 'comparison_bar.png', dpi=300)
    plt.close()
    
    logger.info(f'Comparison bar chart saved to {plot_dir}')


def main():
    parser = argparse.ArgumentParser(description='Ablate clustering methods')
    
    # Model args
    parser.add_argument('--feature_dim', type=int, default=1024)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Hierarchical args (fixed for ablation)
    parser.add_argument('--num_regions', type=int, default=16,
                       help='Number of regions (fixed for this ablation)')
    parser.add_argument('--region_hidden_dim', type=int, default=512)
    parser.add_argument('--pooling', type=str, default='attention',
                       choices=['attention', 'mean', 'max'])
    parser.add_argument('--temperature', type=float, default=1.0)
    
    # Ablation args
    parser.add_argument('--clustering_methods', type=str, nargs='+', 
                       default=['learnable', 'kmeans', 'grid'],
                       help='List of clustering methods to ablate')
    
    # Training args
    parser.add_argument('--dataset', type=str, default='dummy')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    
    # Output args
    parser.add_argument('--output_dir', type=str, 
                       default='experiments/v2_0/ablations/clustering')
    
    # Quick mode (fewer epochs for testing)
    parser.add_argument('--quick', action='store_true',
                       help='Quick mode: 5 epochs, small dataset')
    
    args = parser.parse_args()
    
    # Quick mode overrides
    if args.quick:
        args.num_epochs = 5
        logger.info('Quick mode: 5 epochs')
    
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
    
    # Create datasets
    logger.info('Creating datasets...')
    train_dataset = DummyMILDataset(
        num_samples=100 if not args.quick else 50,
        num_patches=500,
        feature_dim=args.feature_dim,
        num_classes=args.num_classes,
        seed=args.seed,
    )
    val_dataset = DummyMILDataset(
        num_samples=20 if not args.quick else 10,
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
    
    # Run ablations
    logger.info(f'\nRunning ablations for clustering methods: {args.clustering_methods}')
    results = []
    
    for clustering_method in args.clustering_methods:
        result = run_ablation(
            clustering_method=clustering_method,
            args=args,
            train_loader=train_loader,
            val_loader=val_loader,
            device=device,
            output_dir=output_dir,
        )
        results.append(result)
    
    # Save results
    results_path = output_dir / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f'\nResults saved to {results_path}')
    
    # Plot results
    plot_cluster_centers(results, output_dir)
    plot_training_curves(results, output_dir)
    plot_comparison_bar(results, output_dir)
    
    # Print summary
    logger.info('\n' + '='*60)
    logger.info('ABLATION SUMMARY')
    logger.info('='*60)
    logger.info(f'{"Method":<12} {"Best Val Acc":<15} {"Best Epoch":<12} {"Avg Time":<15}')
    logger.info('-'*60)
    for r in results:
        logger.info(
            f'{r["clustering_method"]:<12} '
            f'{r["best_val_acc"]:<15.2f} '
            f'{r["best_epoch"]:<12} '
            f'{r["avg_epoch_time"]:<15.2f}'
        )
    logger.info('='*60)
    
    # Find best
    best_result = max(results, key=lambda x: x['best_val_acc'])
    logger.info(f'\nBest method: {best_result["clustering_method"]} '
               f'(val_acc={best_result["best_val_acc"]:.2f}%)')
    
    # Analysis
    logger.info('\nKey Findings:')
    logger.info(f'- Learnable clustering: {"trainable" if results[0]["trainable_params"] > 0 else "fixed"} centers')
    logger.info(f'- K-means: fixed centers after initial fit')
    logger.info(f'- Grid: deterministic uniform layout')


if __name__ == '__main__':
    main()
