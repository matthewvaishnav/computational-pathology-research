"""
Threshold Optimization for PatchCamelyon Results

Analyzes ROC curve to find optimal decision threshold that balances
precision and recall based on clinical requirements.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import (
    roc_curve, 
    precision_recall_curve, 
    f1_score,
    confusion_matrix,
    classification_report
)
import seaborn as sns

def load_results(results_path):
    """Load evaluation results with predictions, labels, and probabilities."""
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results

def analyze_roc_curve(labels, probabilities, output_dir):
    """Analyze ROC curve and find optimal thresholds."""
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(labels, probabilities)
    
    # Calculate Youden's J statistic (TPR - FPR)
    j_scores = tpr - fpr
    optimal_idx_youden = np.argmax(j_scores)
    optimal_threshold_youden = thresholds[optimal_idx_youden]
    
    # Find threshold for 90% sensitivity (recall)
    target_sensitivity = 0.90
    idx_90_sensitivity = np.argmin(np.abs(tpr - target_sensitivity))
    threshold_90_sensitivity = thresholds[idx_90_sensitivity]
    
    # Find threshold for 95% sensitivity
    target_sensitivity_95 = 0.95
    idx_95_sensitivity = np.argmin(np.abs(tpr - target_sensitivity_95))
    threshold_95_sensitivity = thresholds[idx_95_sensitivity]
    
    # Plot ROC curve with optimal points
    plt.figure(figsize=(12, 8))
    plt.plot(fpr, tpr, 'b-', linewidth=2, label='ROC Curve')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
    
    # Mark optimal points
    plt.plot(fpr[optimal_idx_youden], tpr[optimal_idx_youden], 'ro', 
             markersize=10, label=f'Youden\'s J (threshold={optimal_threshold_youden:.3f})')
    plt.plot(fpr[idx_90_sensitivity], tpr[idx_90_sensitivity], 'go', 
             markersize=10, label=f'90% Sensitivity (threshold={threshold_90_sensitivity:.3f})')
    plt.plot(fpr[idx_95_sensitivity], tpr[idx_95_sensitivity], 'mo', 
             markersize=10, label=f'95% Sensitivity (threshold={threshold_95_sensitivity:.3f})')
    
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=12)
    plt.title('ROC Curve with Optimal Thresholds', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curve_optimal.png', dpi=300, bbox_inches='tight')
    print(f"Saved ROC curve to {output_dir / 'roc_curve_optimal.png'}")
    
    return {
        'youden': {
            'threshold': float(optimal_threshold_youden),
            'tpr': float(tpr[optimal_idx_youden]),
            'fpr': float(fpr[optimal_idx_youden]),
            'j_score': float(j_scores[optimal_idx_youden])
        },
        'sensitivity_90': {
            'threshold': float(threshold_90_sensitivity),
            'tpr': float(tpr[idx_90_sensitivity]),
            'fpr': float(fpr[idx_90_sensitivity])
        },
        'sensitivity_95': {
            'threshold': float(threshold_95_sensitivity),
            'tpr': float(tpr[idx_95_sensitivity]),
            'fpr': float(fpr[idx_95_sensitivity])
        }
    }

def analyze_precision_recall_curve(labels, probabilities, output_dir):
    """Analyze precision-recall curve."""
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    
    # Calculate F1 scores for each threshold
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    optimal_idx_f1 = np.argmax(f1_scores)
    optimal_threshold_f1 = thresholds[optimal_idx_f1]
    
    # Find threshold for 90% recall
    target_recall = 0.90
    idx_90_recall = np.argmin(np.abs(recall[:-1] - target_recall))
    threshold_90_recall = thresholds[idx_90_recall]
    
    # Plot precision-recall curve
    plt.figure(figsize=(12, 8))
    plt.plot(recall, precision, 'b-', linewidth=2, label='Precision-Recall Curve')
    
    # Mark optimal points
    plt.plot(recall[optimal_idx_f1], precision[optimal_idx_f1], 'ro', 
             markersize=10, label=f'Max F1 (threshold={optimal_threshold_f1:.3f})')
    plt.plot(recall[idx_90_recall], precision[idx_90_recall], 'go', 
             markersize=10, label=f'90% Recall (threshold={threshold_90_recall:.3f})')
    
    plt.xlabel('Recall (Sensitivity)', fontsize=12)
    plt.ylabel('Precision', fontsize=12)
    plt.title('Precision-Recall Curve with Optimal Thresholds', fontsize=14, fontweight='bold')
    plt.legend(loc='lower left', fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / 'precision_recall_curve.png', dpi=300, bbox_inches='tight')
    print(f"Saved precision-recall curve to {output_dir / 'precision_recall_curve.png'}")
    
    return {
        'max_f1': {
            'threshold': float(optimal_threshold_f1),
            'precision': float(precision[optimal_idx_f1]),
            'recall': float(recall[optimal_idx_f1]),
            'f1': float(f1_scores[optimal_idx_f1])
        },
        'recall_90': {
            'threshold': float(threshold_90_recall),
            'precision': float(precision[idx_90_recall]),
            'recall': float(recall[idx_90_recall])
        }
    }

def evaluate_threshold(labels, probabilities, threshold):
    """Evaluate performance at a specific threshold."""
    predictions = (probabilities >= threshold).astype(int)
    
    # Calculate metrics
    cm = confusion_matrix(labels, predictions)
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    f1 = f1_score(labels, predictions)
    
    # False positive and false negative rates
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    
    return {
        'threshold': float(threshold),
        'accuracy': float(accuracy),
        'sensitivity': float(sensitivity),
        'specificity': float(specificity),
        'precision': float(precision),
        'f1': float(f1),
        'fpr': float(fpr),
        'fnr': float(fnr),
        'confusion_matrix': cm.tolist(),
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }

def compare_thresholds(labels, probabilities, thresholds_dict, output_dir):
    """Compare performance across different thresholds."""
    results = {}
    
    # Evaluate each threshold
    for name, threshold in thresholds_dict.items():
        results[name] = evaluate_threshold(labels, probabilities, threshold)
    
    # Create comparison table
    print("\n" + "=" * 80)
    print("THRESHOLD COMPARISON")
    print("=" * 80)
    
    print(f"\n{'Threshold':<20} {'Value':<10} {'Accuracy':<10} {'Sensitivity':<12} {'Specificity':<12} {'F1':<10}")
    print("-" * 80)
    
    for name, metrics in results.items():
        print(f"{name:<20} {metrics['threshold']:<10.3f} {metrics['accuracy']:<10.3f} "
              f"{metrics['sensitivity']:<12.3f} {metrics['specificity']:<12.3f} {metrics['f1']:<10.3f}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    names = list(results.keys())
    metrics_to_plot = [
        ('accuracy', 'Accuracy', axes[0, 0]),
        ('sensitivity', 'Sensitivity (Recall)', axes[0, 1]),
        ('specificity', 'Specificity', axes[1, 0]),
        ('f1', 'F1 Score', axes[1, 1])
    ]
    
    for metric_key, metric_name, ax in metrics_to_plot:
        values = [results[name][metric_key] for name in names]
        bars = ax.bar(names, values, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6'][:len(names)])
        ax.set_ylabel(metric_name, fontsize=11)
        ax.set_title(f'{metric_name} by Threshold', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}',
                    ha='center', va='bottom', fontsize=9)
        
        # Rotate x labels
        ax.set_xticklabels(names, rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'threshold_comparison.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved threshold comparison to {output_dir / 'threshold_comparison.png'}")
    
    return results

def generate_recommendations(threshold_results, current_threshold=0.5):
    """Generate recommendations based on threshold analysis."""
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    current_metrics = threshold_results['default_0.5']
    youden_metrics = threshold_results['youden_optimal']
    sensitivity_90_metrics = threshold_results['sensitivity_90']
    
    print(f"\n1. CURRENT THRESHOLD (0.5):")
    print(f"   - Sensitivity: {current_metrics['sensitivity']:.1%} (missing {current_metrics['fnr']:.1%} of tumors)")
    print(f"   - Specificity: {current_metrics['specificity']:.1%}")
    print(f"   - Clinical concern: High false negative rate")
    
    print(f"\n2. YOUDEN'S J OPTIMAL (threshold={youden_metrics['threshold']:.3f}):")
    print(f"   - Sensitivity: {youden_metrics['sensitivity']:.1%} (↑ {(youden_metrics['sensitivity'] - current_metrics['sensitivity'])*100:.1f}%)")
    print(f"   - Specificity: {youden_metrics['specificity']:.1%} (↓ {(current_metrics['specificity'] - youden_metrics['specificity'])*100:.1f}%)")
    print(f"   - Balanced approach, maximizes overall performance")
    print(f"   - Recommendation: GOOD for research, may still miss some tumors")
    
    print(f"\n3. 90% SENSITIVITY TARGET (threshold={sensitivity_90_metrics['threshold']:.3f}):")
    print(f"   - Sensitivity: {sensitivity_90_metrics['sensitivity']:.1%} (↑ {(sensitivity_90_metrics['sensitivity'] - current_metrics['sensitivity'])*100:.1f}%)")
    print(f"   - Specificity: {sensitivity_90_metrics['specificity']:.1%} (↓ {(current_metrics['specificity'] - sensitivity_90_metrics['specificity'])*100:.1f}%)")
    print(f"   - Catches 90% of tumors, acceptable false positive rate")
    print(f"   - Recommendation: BEST for clinical screening")
    
    print(f"\n4. CLINICAL DEPLOYMENT RECOMMENDATION:")
    print(f"   - Use threshold = {sensitivity_90_metrics['threshold']:.3f} for screening")
    print(f"   - Reduces false negatives from {current_metrics['fn']} to {sensitivity_90_metrics['fn']}")
    print(f"   - Saves approximately {current_metrics['fn'] - sensitivity_90_metrics['fn']} missed tumor cases")
    print(f"   - Trade-off: {sensitivity_90_metrics['fp'] - current_metrics['fp']} additional false positives")
    print(f"   - Clinical impact: Better to flag for review than miss cancer")

def generate_threshold_report(results, output_dir):
    """Generate comprehensive threshold optimization report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 80)
    print("THRESHOLD OPTIMIZATION FOR PATCHCAMELYON")
    print("=" * 80)
    
    labels = np.array(results['labels'])
    probabilities = np.array(results['probabilities'])
    
    # Analyze ROC curve
    print("\n[1/4] Analyzing ROC curve...")
    roc_analysis = analyze_roc_curve(labels, probabilities, output_dir)
    
    # Analyze precision-recall curve
    print("\n[2/4] Analyzing precision-recall curve...")
    pr_analysis = analyze_precision_recall_curve(labels, probabilities, output_dir)
    
    # Compare thresholds
    print("\n[3/4] Comparing threshold options...")
    thresholds_to_compare = {
        'default_0.5': 0.5,
        'youden_optimal': roc_analysis['youden']['threshold'],
        'max_f1': pr_analysis['max_f1']['threshold'],
        'sensitivity_90': roc_analysis['sensitivity_90']['threshold'],
        'sensitivity_95': roc_analysis['sensitivity_95']['threshold']
    }
    
    threshold_comparison = compare_thresholds(labels, probabilities, thresholds_to_compare, output_dir)
    
    # Generate recommendations
    print("\n[4/4] Generating recommendations...")
    generate_recommendations(threshold_comparison)
    
    # Compile report
    report = {
        'roc_analysis': roc_analysis,
        'precision_recall_analysis': pr_analysis,
        'threshold_comparison': threshold_comparison,
        'recommended_threshold': roc_analysis['sensitivity_90']['threshold'],
        'recommendation_rationale': 'Achieves 90% sensitivity for tumor detection while maintaining acceptable specificity'
    }
    
    # Save report
    report_path = output_dir / 'threshold_optimization.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n{'=' * 80}")
    print(f"Threshold optimization complete!")
    print(f"Report saved to: {report_path}")
    print(f"{'=' * 80}")
    
    return report

def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Optimize decision threshold for PCam classification')
    parser.add_argument('--results', type=str, default='results/pcam_real/metrics.json',
                        help='Path to evaluation results JSON')
    parser.add_argument('--output-dir', type=str, default='results/pcam_real/threshold_optimization',
                        help='Output directory for optimization results')
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.results}...")
    results = load_results(args.results)
    
    # Generate report
    report = generate_threshold_report(results, args.output_dir)
    
    print(f"\n✅ Optimization complete!")
    print(f"\n🎯 RECOMMENDED THRESHOLD: {report['recommended_threshold']:.3f}")
    print(f"   (Current default: 0.5)")

if __name__ == '__main__':
    main()
