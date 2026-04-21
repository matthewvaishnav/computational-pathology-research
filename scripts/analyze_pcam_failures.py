"""
Failure Analysis for PatchCamelyon Results

Analyzes misclassified samples to identify patterns and model weaknesses.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from collections import defaultdict
import seaborn as sns

def load_results(results_path):
    """Load evaluation results with predictions and labels."""
    with open(results_path, 'r') as f:
        results = json.load(f)
    return results

def analyze_failures(results):
    """Analyze misclassified samples."""
    predictions = np.array(results['predictions'])
    labels = np.array(results['labels'])
    
    # Identify misclassifications
    false_positives = (predictions == 1) & (labels == 0)
    false_negatives = (predictions == 0) & (labels == 1)
    
    fp_indices = np.where(false_positives)[0]
    fn_indices = np.where(false_negatives)[0]
    
    print(f"\n=== Failure Analysis ===")
    print(f"Total samples: {len(labels)}")
    print(f"Correct predictions: {np.sum(predictions == labels)} ({100 * np.mean(predictions == labels):.2f}%)")
    print(f"\nFalse Positives: {len(fp_indices)} ({100 * len(fp_indices) / len(labels):.2f}%)")
    print(f"  - Normal tissue incorrectly classified as tumor")
    print(f"False Negatives: {len(fn_indices)} ({100 * len(fn_indices) / len(labels):.2f}%)")
    print(f"  - Tumor tissue incorrectly classified as normal")
    
    return {
        'false_positives': fp_indices.tolist(),
        'false_negatives': fn_indices.tolist(),
        'fp_count': len(fp_indices),
        'fn_count': len(fn_indices),
        'total': len(labels)
    }

def analyze_confidence_distribution(results):
    """Analyze prediction confidence for correct vs incorrect predictions."""
    predictions = np.array(results['predictions'])
    labels = np.array(results['labels'])
    
    # Get probabilities if available
    if 'probabilities' in results:
        probs = np.array(results['probabilities'])
        
        # Handle 1D probability array (probability of class 1)
        if probs.ndim == 1:
            # Convert to confidence: max of (1-p, p)
            confidence = np.maximum(1 - probs, probs)
        else:
            # 2D array: confidence = max probability
            confidence = np.max(probs, axis=1)
        
        correct = predictions == labels
        incorrect = predictions != labels
        
        print(f"\n=== Confidence Analysis ===")
        print(f"Correct predictions - Mean confidence: {np.mean(confidence[correct]):.4f}")
        print(f"Incorrect predictions - Mean confidence: {np.mean(confidence[incorrect]):.4f}")
        
        # Plot confidence distributions
        plt.figure(figsize=(10, 6))
        plt.hist(confidence[correct], bins=50, alpha=0.5, label='Correct', color='green')
        plt.hist(confidence[incorrect], bins=50, alpha=0.5, label='Incorrect', color='red')
        plt.xlabel('Prediction Confidence')
        plt.ylabel('Count')
        plt.title('Confidence Distribution: Correct vs Incorrect Predictions')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return confidence
    else:
        print("\n=== Confidence Analysis ===")
        print("Probabilities not available in results")
        return None

def analyze_class_specific_errors(results):
    """Analyze errors specific to each class."""
    predictions = np.array(results['predictions'])
    labels = np.array(results['labels'])
    
    print(f"\n=== Class-Specific Error Analysis ===")
    
    # Class 0 (Normal)
    class_0_mask = labels == 0
    class_0_correct = np.sum((predictions == 0) & class_0_mask)
    class_0_total = np.sum(class_0_mask)
    class_0_accuracy = class_0_correct / class_0_total
    
    print(f"\nClass 0 (Normal Tissue):")
    print(f"  Total samples: {class_0_total}")
    print(f"  Correctly classified: {class_0_correct} ({100 * class_0_accuracy:.2f}%)")
    print(f"  Misclassified as tumor: {class_0_total - class_0_correct} ({100 * (1 - class_0_accuracy):.2f}%)")
    
    # Class 1 (Tumor)
    class_1_mask = labels == 1
    class_1_correct = np.sum((predictions == 1) & class_1_mask)
    class_1_total = np.sum(class_1_mask)
    class_1_accuracy = class_1_correct / class_1_total
    
    print(f"\nClass 1 (Tumor Tissue):")
    print(f"  Total samples: {class_1_total}")
    print(f"  Correctly classified: {class_1_correct} ({100 * class_1_accuracy:.2f}%)")
    print(f"  Misclassified as normal: {class_1_total - class_1_correct} ({100 * (1 - class_1_accuracy):.2f}%)")
    
    return {
        'class_0': {
            'total': int(class_0_total),
            'correct': int(class_0_correct),
            'accuracy': float(class_0_accuracy)
        },
        'class_1': {
            'total': int(class_1_total),
            'correct': int(class_1_correct),
            'accuracy': float(class_1_accuracy)
        }
    }

def create_error_rate_plot(results, output_dir):
    """Create visualization of error rates by class."""
    predictions = np.array(results['predictions'])
    labels = np.array(results['labels'])
    
    # Calculate metrics
    cm = results['confusion_matrix']
    tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
    
    # Error rates
    fp_rate = fp / (tn + fp)  # False positive rate
    fn_rate = fn / (fn + tp)  # False negative rate (miss rate)
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    categories = ['False Positive Rate\n(Normal → Tumor)', 'False Negative Rate\n(Tumor → Normal)']
    rates = [fp_rate, fn_rate]
    colors = ['#ff6b6b', '#ffd93d']
    
    bars = ax.bar(categories, rates, color=colors, alpha=0.7, edgecolor='black')
    
    # Add value labels on bars
    for bar, rate in zip(bars, rates):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{rate*100:.2f}%',
                ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Error Rate', fontsize=12)
    ax.set_title('Classification Error Rates by Type', fontsize=14, fontweight='bold')
    ax.set_ylim(0, max(rates) * 1.2)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'error_rates.png', dpi=300, bbox_inches='tight')
    print(f"\nSaved error rate plot to {output_dir / 'error_rates.png'}")
    
    return {'fp_rate': float(fp_rate), 'fn_rate': float(fn_rate)}

def generate_failure_report(results, output_dir):
    """Generate comprehensive failure analysis report."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("PatchCamelyon Failure Analysis")
    print("=" * 60)
    
    # Analyze failures
    failure_stats = analyze_failures(results)
    
    # Analyze confidence
    confidence = analyze_confidence_distribution(results)
    if confidence is not None:
        plt.savefig(output_dir / 'confidence_distribution.png', dpi=300, bbox_inches='tight')
        print(f"Saved confidence distribution to {output_dir / 'confidence_distribution.png'}")
    
    # Class-specific analysis
    class_stats = analyze_class_specific_errors(results)
    
    # Error rate visualization
    error_rates = create_error_rate_plot(results, output_dir)
    
    # Compile report
    report = {
        'failure_statistics': failure_stats,
        'class_statistics': class_stats,
        'error_rates': error_rates,
        'summary': {
            'total_samples': failure_stats['total'],
            'total_errors': failure_stats['fp_count'] + failure_stats['fn_count'],
            'error_rate': (failure_stats['fp_count'] + failure_stats['fn_count']) / failure_stats['total'],
            'fp_rate': error_rates['fp_rate'],
            'fn_rate': error_rates['fn_rate']
        }
    }
    
    # Save report
    report_path = output_dir / 'failure_analysis.json'
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    print(f"\n{'=' * 60}")
    print(f"Failure analysis complete!")
    print(f"Report saved to: {report_path}")
    print(f"{'=' * 60}")
    
    # Print key insights
    print(f"\n=== Key Insights ===")
    print(f"1. Model has higher false negative rate ({error_rates['fn_rate']*100:.2f}%) than false positive rate ({error_rates['fp_rate']*100:.2f}%)")
    print(f"   → Model tends to miss tumors more than it falsely identifies them")
    print(f"2. Class 0 (Normal) accuracy: {class_stats['class_0']['accuracy']*100:.2f}%")
    print(f"3. Class 1 (Tumor) accuracy: {class_stats['class_1']['accuracy']*100:.2f}%")
    
    if error_rates['fn_rate'] > error_rates['fp_rate']:
        print(f"\n⚠️  Clinical Concern: High false negative rate means tumors are being missed")
        print(f"   Recommendation: Consider adjusting decision threshold or using ensemble methods")
    
    return report

def main():
    """Main execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze PatchCamelyon failure cases')
    parser.add_argument('--results', type=str, default='results/pcam_real/metrics.json',
                        help='Path to evaluation results JSON')
    parser.add_argument('--output-dir', type=str, default='results/pcam_real/failure_analysis',
                        help='Output directory for analysis results')
    
    args = parser.parse_args()
    
    # Load results
    print(f"Loading results from {args.results}...")
    results = load_results(args.results)
    
    # Generate report
    report = generate_failure_report(results, args.output_dir)
    
    print(f"\n✅ Analysis complete!")

if __name__ == '__main__':
    main()
