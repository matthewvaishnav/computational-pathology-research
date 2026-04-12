"""
Test script to verify visualization notebook functionality.
This script tests the key functions from the notebook.
"""

import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import torch
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path('src/data').resolve()))

# Set style
sns.set_style('whitegrid')
plt.rcParams['figure.dpi'] = 100
plt.rcParams['savefig.dpi'] = 300

# Output directory
OUTPUT_DIR = Path('results/pcam')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print('='*60)
print('TESTING PCAM VISUALIZATION NOTEBOOK FUNCTIONALITY')
print('='*60)

# Test 1: Load evaluation results
print('\n[Test 1] Loading evaluation results...')
def load_evaluation_results(json_path=None):
    """Load evaluation metrics from JSON."""
    default_results = {
        'confusion_matrix': np.array([[4500, 500], [300, 4700]]),
        'y_true': np.concatenate([np.zeros(4800), np.ones(5000)]),
        'y_pred': np.concatenate([np.zeros(4500) + np.random.normal(0, 0.3, 4500), 
                                  np.ones(4700) + np.random.normal(0, 0.3, 4700)]),
        'y_pred_proba': np.concatenate([np.random.beta(2, 5, 4500), np.random.beta(5, 2, 4700)])
    }
    
    # Try multiple possible paths
    possible_paths = [
        json_path,
        Path('results/pcam_eval_test/metrics.json'),
        Path('results/pcam/metrics.json')
    ]
    
    for path in possible_paths:
        if path and Path(path).exists():
            with open(path, 'r') as f:
                data = json.load(f)
                print(f'✓ Loaded evaluation results from {path}')
                
                # Convert to expected format
                results = {
                    'confusion_matrix': np.array(data['confusion_matrix']),
                    'y_true': np.array(data['labels']),
                    'y_pred': np.array(data['predictions']),
                    'y_pred_proba': np.array(data['probabilities'])
                }
                return results
    
    print('⚠ Using sample evaluation data for demonstration.')
    return default_results

eval_results = load_evaluation_results()
cm = eval_results['confusion_matrix']
print(f'  Confusion matrix shape: {cm.shape}')
print(f'  Total samples: {len(eval_results["y_true"])}')

# Test 2: Load training metrics
print('\n[Test 2] Loading training metrics...')
def load_metrics(checkpoint_path=None, json_path=None):
    """Load training metrics from checkpoint or JSON file."""
    metrics = {'train_loss': [], 'val_loss': [], 'train_acc': [], 'val_acc': []}
    
    # Try multiple possible paths
    possible_checkpoint_paths = [
        checkpoint_path,
        Path('checkpoints/pcam/best_model.pth'),
        Path('checkpoints/pcam/checkpoint_epoch_5.pth')
    ]
    
    for path in possible_checkpoint_paths:
        if path and Path(path).exists():
            try:
                checkpoint = torch.load(path, map_location='cpu')
                if 'history' in checkpoint:
                    metrics = checkpoint['history']
                    print(f'✓ Loaded metrics from checkpoint: {path}')
                    return metrics
                elif 'metrics' in checkpoint:
                    # Single epoch metrics - create history
                    metrics = {
                        'train_loss': [checkpoint['metrics'].get('train_loss', 0)],
                        'val_loss': [checkpoint['metrics'].get('val_loss', 0)],
                        'train_acc': [checkpoint['metrics'].get('train_accuracy', 0)],
                        'val_acc': [checkpoint['metrics'].get('val_accuracy', 0)]
                    }
                    print(f'✓ Loaded single epoch metrics from checkpoint: {path}')
                    return metrics
            except Exception as e:
                print(f'  Could not load checkpoint {path}: {e}')
                continue
    
    print('⚠ Using sample metrics data for demonstration.')
    # Generate sample data for demonstration
    epochs = np.arange(1, 21)
    metrics = {
        'train_loss': list(0.7 * np.exp(-0.1 * epochs) + np.random.normal(0, 0.02, 20)),
        'val_loss': list(0.75 * np.exp(-0.1 * epochs) + np.random.normal(0, 0.03, 20)),
        'train_acc': list(1 - 0.3 * np.exp(-0.15 * epochs) + np.random.normal(0, 0.01, 20)),
        'val_acc': list(1 - 0.35 * np.exp(-0.15 * epochs) + np.random.normal(0, 0.015, 20))
    }
    return metrics

metrics = load_metrics()
print(f'  Metrics loaded: {list(metrics.keys())}')
print(f'  Number of epochs: {len(metrics["train_loss"])}')

# Test 3: Generate confusion matrix plot
print('\n[Test 3] Generating confusion matrix plot...')
fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
            xticklabels=['Normal (0)', 'Metastatic (1)'],
            yticklabels=['Normal (0)', 'Metastatic (1)'],
            annot_kws={'size': 14})
ax.set_xlabel('Predicted Label', fontsize=12)
ax.set_ylabel('True Label', fontsize=12)
ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print(f'✓ Saved confusion matrix to {OUTPUT_DIR / "confusion_matrix.png"}')

# Test 4: Generate loss curves
print('\n[Test 4] Generating loss curves...')
fig, ax = plt.subplots(figsize=(12, 6))
epochs = range(1, len(metrics['train_loss']) + 1)
ax.plot(epochs, metrics['train_loss'], 'b-', label='Train Loss', linewidth=2, marker='o', markersize=4)
ax.plot(epochs, metrics['val_loss'], 'r-', label='Val Loss', linewidth=2, marker='s', markersize=4)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Loss', fontsize=12)
ax.set_title('Training and Validation Loss Curves', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'loss_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print(f'✓ Saved loss curves to {OUTPUT_DIR / "loss_curves.png"}')

# Test 5: Generate accuracy curves
print('\n[Test 5] Generating accuracy curves...')
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(epochs, metrics['train_acc'], 'b-', label='Train Accuracy', linewidth=2, marker='o', markersize=4)
ax.plot(epochs, metrics['val_acc'], 'r-', label='Val Accuracy', linewidth=2, marker='s', markersize=4)
ax.set_xlabel('Epoch', fontsize=12)
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_title('Training and Validation Accuracy Curves', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_ylim([0, 1.05])
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'accuracy_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print(f'✓ Saved accuracy curves to {OUTPUT_DIR / "accuracy_curves.png"}')

# Test 6: Generate ROC curve
print('\n[Test 6] Generating ROC curve...')
from sklearn.metrics import roc_curve, auc
y_true = eval_results['y_true']
y_pred_proba = eval_results['y_pred_proba']
fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
roc_auc = auc(fpr, tpr)

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(fpr, tpr, 'b-', linewidth=2, label=f'ROC Curve (AUC = {roc_auc:.4f})')
ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
ax.set_xlabel('False Positive Rate', fontsize=12)
ax.set_ylabel('True Positive Rate', fontsize=12)
ax.set_title('Receiver Operating Characteristic (ROC) Curve', fontsize=14, fontweight='bold')
ax.legend(fontsize=11, loc='lower right')
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.05])
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print(f'✓ Saved ROC curve to {OUTPUT_DIR / "roc_curve.png"}')

# Test 7: Generate precision-recall curve
print('\n[Test 7] Generating precision-recall curve...')
from sklearn.metrics import precision_recall_curve, average_precision_score
precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
avg_precision = average_precision_score(y_true, y_pred_proba)

fig, ax = plt.subplots(figsize=(10, 8))
ax.plot(recall, precision, 'g-', linewidth=2, label=f'PR Curve (AP = {avg_precision:.4f})')
ax.set_xlabel('Recall', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xlim([0, 1])
ax.set_ylim([0, 1.05])
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print(f'✓ Saved precision-recall curve to {OUTPUT_DIR / "precision_recall_curve.png"}')

# Test 8: Generate confidence histogram
print('\n[Test 8] Generating confidence histogram...')
y_pred = (y_pred_proba >= 0.5).astype(int)
correct_mask = (y_pred == y_true)
incorrect_mask = ~correct_mask
correct_confidence = y_pred_proba[correct_mask]
incorrect_confidence = y_pred_proba[incorrect_mask]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Overall confidence distribution
ax1 = axes[0]
ax1.hist(y_pred_proba, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
ax1.axvline(x=0.5, color='red', linestyle='--', linewidth=2, label='Decision Threshold')
ax1.set_xlabel('Predicted Probability', fontsize=11)
ax1.set_ylabel('Frequency', fontsize=11)
ax1.set_title('Overall Confidence Distribution', fontsize=12, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Confidence distribution by correctness
ax2 = axes[1]
ax2.hist(correct_confidence, bins=50, color='green', edgecolor='black', alpha=0.6, label=f'Correct (n={len(correct_confidence)})')
ax2.hist(incorrect_confidence, bins=50, color='red', edgecolor='black', alpha=0.6, label=f'Incorrect (n={len(incorrect_confidence)})')
ax2.axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold')
ax2.set_xlabel('Predicted Probability', fontsize=11)
ax2.set_ylabel('Frequency', fontsize=11)
ax2.set_title('Confidence Distribution by Prediction Correctness', fontsize=12, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'confidence_histogram.png', dpi=300, bbox_inches='tight')
plt.close()
print(f'✓ Saved confidence histogram to {OUTPUT_DIR / "confidence_histogram.png"}')

# Summary
print('\n' + '='*60)
print('VISUALIZATION TEST SUMMARY')
print('='*60)
saved_plots = sorted(OUTPUT_DIR.glob('*.png'))
print(f'\nTotal plots generated: {len(saved_plots)}')
print(f'Output directory: {OUTPUT_DIR}')
print('\nGenerated plots:')
for plot in saved_plots:
    print(f'  ✓ {plot.name}')

print(f'\nEvaluation Metrics:')
print(f'  - Total predictions: {len(y_true)}')
print(f'  - Accuracy: {100 * np.sum(y_pred == y_true) / len(y_true):.2f}%')
print(f'  - AUC: {roc_auc:.4f}')
print(f'  - Average Precision: {avg_precision:.4f}')

print('\n✓ All visualization tests passed!')
