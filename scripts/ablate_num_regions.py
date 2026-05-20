"""
Ablation Study: Number of Regions (Task 4.2)

Ablates num_regions parameter: 8, 16, 32, 64
Compares performance vs computational cost.

Usage:
    python scripts/ablate_num_regions.py --dataset tcga-brca --num_epochs 20
    python scripts/ablate_num_regions.py --dataset dummy --num_epochs 5 --quick

Output:
    experiments/v2_0/ablations/num_regions/
    ├── results.json
    ├── plots/
    │   ├── accuracy_vs_regions.png
    │   ├── loss_vs_regions.png
    │   └── time_vs_regions.png
    └── checkpoints/
        ├── regions_8/
        ├── regions_16/
        ├── regions_32/
        └── regions_64/
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

from src.models.transnnmil.transnnmil import TransnnMIL
from src.core.utils.monitoring import get_logger

logger = get_logger(__name__)


def run_ablation(
    num_regions: int,
    args: argparse.Namespace,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    output_dir: Path,
):
    """Run single ablation experiment."""
    logger.info(f'\n{"="*60}')
    logger.info(f'Ablation: num_regions={num_regions}')
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
        num_regions=num_regions,
        region_hidden_dim=args.region_hidden_dim,
        clustering_method=args.clustering,
        pooling_method=args.pooling,
        temperature=args.temperature,
    )
    model = model.to(device)
    
    # Count params
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f'Parameters: {total_params:,}')
    
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
    
    # Save checkpoint
    checkpoint_dir = output_dir / 'checkpoints' / f'regions_{num_regions}'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    checkpoint_path = checkpoint_dir / 'checkpoint_best.pth'
    torch.save({
        'model_state_dict': model.state_dict(),
        'num_regions': num_regions,
        'val_acc': best_val_acc,
        'history': history,
    }, checkpoint_path)
    
    logger.info(f'Best val acc: {best_val_acc:.2f}%')
    logger.info(f'Avg epoch time: {np.mean(history["epoch_time"]):.2f}s')
    
    return {
        'num_regions': num_regions,
        'best_val_acc': best_val_acc,
        'final_train_acc': history['train_acc'][-1],
        'final_val_acc': history['val_acc'][-1],
        'avg_epoch_time': np.mean(history['epoch_time']),
        'total_params': total_params,
        'history': history,
    }


def plot_results(results: list, output_dir: Path):
    """Plot ablation results."""
    plot_dir = output_dir / 'plots'
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract data
    num_regions_list = [r['num_regions'] for r in results]
    best_val_acc = [r['best_val_acc'] for r in results]
    final_val_acc = [r['final_val_acc'] for r in results]
    avg_time = [r['avg_epoch_time'] for r in results]
    total_params = [r['total_params'] for r in results]
    
    # Plot 1: Accuracy vs Regions
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(num_regions_list, best_val_acc, 'o-', label='Best Val Acc', linewidth=2)
    ax.plot(num_regions_list, final_val_acc, 's--', label='Final Val Acc', linewidth=2)
    ax.set_xlabel('Number of Regions', fontsize=12)
    ax.set_ylabel('Accuracy (%)', fontsize=12)
    ax.set_title('Validation Accuracy vs Number of Regions', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_xticks(num_regions_list)
    ax.set_xticklabels(num_regions_list)
    plt.tight_layout()
    plt.savefig(plot_dir / 'accuracy_vs_regions.png', dpi=300)
    plt.close()
    
    # Plot 2: Time vs Regions
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(num_regions_list, avg_time, 'o-', linewidth=2, color='orange')
    ax.set_xlabel('Number of Regions', fontsize=12)
    ax.set_ylabel('Avg Epoch Time (s)', fontsize=12)
    ax.set_title('Training Time vs Number of Regions', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_xticks(num_regions_list)
    ax.set_xticklabels(num_regions_list)
    plt.tight_layout()
    plt.savefig(plot_dir / 'time_vs_regions.png', dpi=300)
    plt.close()
    
    # Plot 3: Parameters vs Regions
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(num_regions_list, [p / 1e6 for p in total_params], 'o-', linewidth=2, color='green')
    ax.set_xlabel('Number of Regions', fontsize=12)
    ax.set_ylabel('Parameters (M)', fontsize=12)
    ax.set_title('Model Size vs Number of Regions', fontsize=14)
    ax.grid(True, alpha=0.3)
    ax.set_xscale('log', base=2)
    ax.set_xticks(num_regions_list)
    ax.set_xticklabels(num_regions_list)
    plt.tight_layout()
    plt.savefig(plot_dir / 'params_vs_regions.png', dpi=300)
    plt.close()
    
    # Plot 4: Accuracy vs Time (efficiency)
    fig, ax = plt.subplots(figsize=(8, 6))
    scatter = ax.scatter(avg_time, best_val_acc, s=200, c=num_regions_list, 
                        cmap='viridis', alpha=0.7, edgecolors='black')
    for i, num_r in enumerate(num_regions_list):
        ax.annotate(f'{num_r}', (avg_time[i], best_val_acc[i]), 
                   ha='center', va='center', fontsize=10, fontweight='bold')
    ax.set_xlabel('Avg Epoch Time (s)', fontsize=12)
    ax.set_ylabel('Best Val Accuracy (%)', fontsize=12)
    ax.set_title('Accuracy vs Time Tradeoff', fontsize=14)
    ax.grid(True, alpha=0.3)
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Num Regions', fontsize=10)
    plt.tight_layout()
    plt.savefig(plot_dir / 'accuracy_vs_time.png', dpi=300)
    plt.close()
    
    logger.info(f'Plots saved to {plot_dir}')


def main():
    parser = argparse.ArgumentParser(description='Ablate num_regions')
    
    # Model args
    parser.add_argument('--feature_dim', type=int, default=1024)
    parser.add_argument('--hidden_dim', type=int, default=256)
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Hierarchical args (fixed for ablation)
    parser.add_argument('--region_hidden_dim', type=int, default=512)
    parser.add_argument('--clustering', type=str, default='learnable',
                       choices=['learnable', 'kmeans', 'grid'])
    parser.add_argument('--pooling', type=str, default='attention',
                       choices=['attention', 'mean', 'max'])
    parser.add_argument('--temperature', type=float, default=1.0)
    
    # Ablation args
    parser.add_argument('--regions_list', type=int, nargs='+', 
                       default=[8, 16, 32, 64],
                       help='List of num_regions to ablate')
    
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
                       default='experiments/v2_0/ablations/num_regions')
    
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
    logger.info(f'\nRunning ablations for num_regions: {args.regions_list}')
    results = []
    
    for num_regions in args.regions_list:
        result = run_ablation(
            num_regions=num_regions,
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
    plot_results(results, output_dir)
    
    # Print summary
    logger.info('\n' + '='*60)
    logger.info('ABLATION SUMMARY')
    logger.info('='*60)
    logger.info(f'{"Regions":<10} {"Best Val Acc":<15} {"Avg Time":<15} {"Params (M)":<15}')
    logger.info('-'*60)
    for r in results:
        logger.info(
            f'{r["num_regions"]:<10} '
            f'{r["best_val_acc"]:<15.2f} '
            f'{r["avg_epoch_time"]:<15.2f} '
            f'{r["total_params"]/1e6:<15.2f}'
        )
    logger.info('='*60)
    
    # Find best
    best_result = max(results, key=lambda x: x['best_val_acc'])
    logger.info(f'\nBest configuration: num_regions={best_result["num_regions"]} '
               f'(val_acc={best_result["best_val_acc"]:.2f}%)')


if __name__ == '__main__':
    main()
