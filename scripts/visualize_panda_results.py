"""
Visualize PANDA training and evaluation results.

Usage:
    python scripts/visualize_panda_results.py --log_dir logs/panda --results_dir results/panda --output_dir visualizations/panda
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

def parse_training_log(log_file):
    """Parse training log file to extract metrics."""
    epochs = []
    train_loss = []
    train_acc = []
    train_kappa = []
    val_loss = []
    val_acc = []
    val_kappa = []
    
    with open(log_file, 'r') as f:
        for line in f:
            if "Epoch" in line and "/" in line:
                try:
                    epoch_num = int(line.split("Epoch")[1].split("/")[0].strip())
                    epochs.append(epoch_num)
                except:
                    pass
            
            if "Train - Loss:" in line:
                parts = line.split("Loss:")[1].split(",")
                loss = float(parts[0].strip())
                acc = float(parts[1].split(":")[1].strip().replace(",", ""))
                kappa = float(parts[2].split(":")[1].strip())
                train_loss.append(loss)
                train_acc.append(acc)
                train_kappa.append(kappa)
            
            if "Val - Loss:" in line:
                parts = line.split("Loss:")[1].split(",")
                loss = float(parts[0].strip())
                acc = float(parts[1].split(":")[1].strip().replace(",", ""))
                try:
                    kappa_str = parts[2].split(":")[1].strip()
                    kappa = float(kappa_str) if kappa_str != "nan" else np.nan
                except:
                    kappa = np.nan
                val_loss.append(loss)
                val_acc.append(acc)
                val_kappa.append(kappa)
    
    return {
        'epochs': epochs[:len(train_loss)],
        'train_loss': train_loss,
        'train_acc': train_acc,
        'train_kappa': train_kappa,
        'val_loss': val_loss,
        'val_acc': val_acc,
        'val_kappa': val_kappa,
    }


def plot_training_curves(metrics, output_path):
    """Plot training curves."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    epochs = metrics['epochs']
    
    # Loss
    axes[0].plot(epochs, metrics['train_loss'], label='Train', marker='o')
    axes[0].plot(epochs, metrics['val_loss'], label='Validation', marker='s')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    # Accuracy
    axes[1].plot(epochs, metrics['train_acc'], label='Train', marker='o')
    axes[1].plot(epochs, metrics['val_acc'], label='Validation', marker='s')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Training and Validation Accuracy')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    # Kappa
    axes[2].plot(epochs, metrics['train_kappa'], label='Train', marker='o')
    # Filter out NaN values for validation kappa
    val_kappa_clean = [k if not np.isnan(k) else None for k in metrics['val_kappa']]
    axes[2].plot(epochs, val_kappa_clean, label='Validation', marker='s')
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('Quadratic Weighted Kappa')
    axes[2].set_title('Training and Validation Kappa')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved training curves to {output_path}")


def plot_final_results_summary(results_file, output_path):
    """Plot summary of final results."""
    with open(results_file, 'r') as f:
        results = json.load(f)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Overall metrics
    metrics_text = f"""
    Final Results Summary
    
    Accuracy: {results['accuracy']:.4f}
    Quadratic Weighted Kappa: {results['quadratic_weighted_kappa']:.4f}
    Number of Samples: {results['num_samples']}
    """
    
    axes[0, 0].text(0.1, 0.5, metrics_text, fontsize=14, verticalalignment='center')
    axes[0, 0].axis('off')
    axes[0, 0].set_title('Overall Metrics', fontsize=16, fontweight='bold')
    
    # Prediction distribution
    preds = np.array(results['predictions'])
    labels = np.array(results['labels'])
    
    grades = range(6)
    pred_counts = [np.sum(preds == g) for g in grades]
    label_counts = [np.sum(labels == g) for g in grades]
    
    x = np.arange(len(grades))
    width = 0.35
    
    axes[0, 1].bar(x - width/2, label_counts, width, label='True', alpha=0.8)
    axes[0, 1].bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.8)
    axes[0, 1].set_xlabel('ISUP Grade')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('Grade Distribution')
    axes[0, 1].set_xticks(x)
    axes[0, 1].legend()
    axes[0, 1].grid(axis='y', alpha=0.3)
    
    # Per-grade accuracy
    accuracies = []
    for g in grades:
        mask = labels == g
        if mask.sum() > 0:
            acc = (preds[mask] == labels[mask]).mean()
            accuracies.append(acc)
        else:
            accuracies.append(0)
    
    axes[1, 0].bar(grades, accuracies, alpha=0.8, color='green')
    axes[1, 0].set_xlabel('ISUP Grade')
    axes[1, 0].set_ylabel('Accuracy')
    axes[1, 0].set_title('Per-Grade Accuracy')
    axes[1, 0].set_ylim([0, 1])
    axes[1, 0].grid(axis='y', alpha=0.3)
    
    # Error analysis
    errors = preds != labels
    error_by_grade = []
    for g in grades:
        mask = labels == g
        if mask.sum() > 0:
            error_rate = errors[mask].mean()
            error_by_grade.append(error_rate)
        else:
            error_by_grade.append(0)
    
    axes[1, 1].bar(grades, error_by_grade, alpha=0.8, color='red')
    axes[1, 1].set_xlabel('ISUP Grade')
    axes[1, 1].set_ylabel('Error Rate')
    axes[1, 1].set_title('Per-Grade Error Rate')
    axes[1, 1].set_ylim([0, 1])
    axes[1, 1].grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Saved results summary to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize PANDA results")
    parser.add_argument("--log_dir", type=str, default="logs/panda", help="Training log directory")
    parser.add_argument("--results_dir", type=str, default="results/panda", help="Results directory")
    parser.add_argument("--output_dir", type=str, default="visualizations/panda", help="Output directory")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 70)
    print("PANDA Results Visualization")
    print("=" * 70)
    
    # Parse training log
    log_file = Path(args.log_dir) / "train.log"
    if log_file.exists():
        print(f"\nParsing training log: {log_file}")
        metrics = parse_training_log(log_file)
        
        if metrics['epochs']:
            plot_training_curves(metrics, output_dir / "training_curves.png")
        else:
            print("  No training metrics found in log")
    else:
        print(f"  Log file not found: {log_file}")
    
    # Plot results
    results_file = Path(args.results_dir) / "results_test.json"
    if results_file.exists():
        print(f"\nPlotting results: {results_file}")
        plot_final_results_summary(results_file, output_dir / "results_summary.png")
    else:
        print(f"  Results file not found: {results_file}")
    
    print("\n" + "=" * 70)
    print(f"Visualizations saved to: {output_dir}")
    print("=" * 70)


if __name__ == "__main__":
    main()
