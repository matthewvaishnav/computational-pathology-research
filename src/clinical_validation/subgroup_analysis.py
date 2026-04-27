"""
Subgroup Analysis for Clinical Validation

Implements comprehensive subgroup analysis for medical AI validation,
including demographic stratification, clinical subgroups, and bias detection.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any
from dataclasses import dataclass
from enum import Enum
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix
)
from scipy import stats
from itertools import combinations
import warnings
import logging

logger = logging.getLogger(__name__)

class SubgroupType(Enum):
    """Types of subgroup analyses"""
    DEMOGRAPHIC = "demographic"
    CLINICAL = "clinical"
    TECHNICAL = "technical"
    TEMPORAL = "temporal"
    GEOGRAPHIC = "geographic"
    INTERSECTIONAL = "intersectional"

@dataclass
class SubgroupMetrics:
    """Metrics for a specific subgroup"""
    subgroup_name: str
    subgroup_value: str
    n_samples: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    auc_roc: Optional[float] = None
    auc_pr: Optional[float] = None
    specificity: Optional[float] = None
    npv: Optional[float] = None  # Negative predictive value
    prevalence: Optional[float] = None
    confidence_intervals: Optional[Dict[str, Tuple[float, float]]] = None

@dataclass
class SubgroupComparison:
    """Comparison between subgroups"""
    subgroup1: str
    subgroup2: str
    metric: str
    difference: float
    p_value: float
    effect_size: float
    significant: bool
    clinical_significance: Optional[bool] = None

class ClinicalSubgroupAnalyzer:
    """Comprehensive subgroup analysis for clinical AI validation"""
    
    def __init__(self, alpha: float = 0.05, min_subgroup_size: int = 30):
        """Initialize subgroup analyzer"""
        self.alpha = alpha
        self.min_subgroup_size = min_subgroup_size
        
    def analyze_demographic_subgroups(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        demographic_columns: List[str]
    ) -> Dict[str, List[SubgroupMetrics]]:
        """Analyze performance across demographic subgroups"""
        results = {}
        
        for demo_col in demographic_columns:
            if demo_col not in data.columns:
                logger.warning(f"Column {demo_col} not found in data")
                continue
            
            subgroup_metrics = []
            unique_values = data[demo_col].unique()
            
            for value in unique_values:
                mask = data[demo_col] == value
                if np.sum(mask) < self.min_subgroup_size:
                    logger.warning(f"Subgroup {demo_col}={value} has only {np.sum(mask)} samples")
                    continue
                
                subgroup_preds = predictions[mask]
                subgroup_labels = true_labels[mask]
                
                metrics = self._calculate_subgroup_metrics(
                    subgroup_preds, subgroup_labels, f"{demo_col}={value}"
                )
                subgroup_metrics.append(metrics)
            
            results[demo_col] = subgroup_metrics
        
        return results
    
    def analyze_clinical_subgroups(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        clinical_columns: List[str]
    ) -> Dict[str, List[SubgroupMetrics]]:
        """Analyze performance across clinical subgroups"""
        results = {}
        
        for clinical_col in clinical_columns:
            if clinical_col not in data.columns:
                logger.warning(f"Column {clinical_col} not found in data")
                continue
            
            subgroup_metrics = []
            
            # Handle continuous variables by binning
            if data[clinical_col].dtype in ['float64', 'int64'] and len(data[clinical_col].unique()) > 10:
                # Create quartiles for continuous variables
                quartiles = pd.qcut(data[clinical_col], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
                unique_values = quartiles.unique()
                grouping_var = quartiles
            else:
                unique_values = data[clinical_col].unique()
                grouping_var = data[clinical_col]
            
            for value in unique_values:
                mask = grouping_var == value
                if np.sum(mask) < self.min_subgroup_size:
                    continue
                
                subgroup_preds = predictions[mask]
                subgroup_labels = true_labels[mask]
                
                metrics = self._calculate_subgroup_metrics(
                    subgroup_preds, subgroup_labels, f"{clinical_col}={value}"
                )
                subgroup_metrics.append(metrics)
            
            results[clinical_col] = subgroup_metrics
        
        return results
    
    def analyze_intersectional_subgroups(
        self,
        data: pd.DataFrame,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        intersect_columns: List[str],
        max_combinations: int = 3
    ) -> Dict[str, List[SubgroupMetrics]]:
        """Analyze intersectional subgroups (e.g., age + gender + race)"""
        results = {}
        
        # Generate combinations of columns
        for r in range(2, min(len(intersect_columns) + 1, max_combinations + 1)):
            for combo in combinations(intersect_columns, r):
                combo_name = " × ".join(combo)
                subgroup_metrics = []
                
                # Create combined grouping variable
                data['_temp_combo'] = data[list(combo)].apply(
                    lambda x: ' & '.join([f"{col}={val}" for col, val in zip(combo, x)]), 
                    axis=1
                )
                
                unique_combos = data['_temp_combo'].unique()
                
                for combo_value in unique_combos:
                    mask = data['_temp_combo'] == combo_value
                    if np.sum(mask) < self.min_subgroup_size:
                        continue
                    
                    subgroup_preds = predictions[mask]
                    subgroup_labels = true_labels[mask]
                    
                    metrics = self._calculate_subgroup_metrics(
                        subgroup_preds, subgroup_labels, combo_value
                    )
                    subgroup_metrics.append(metrics)
                
                if subgroup_metrics:  # Only add if we have valid subgroups
                    results[combo_name] = subgroup_metrics
                
                # Clean up temporary column
                data.drop('_temp_combo', axis=1, inplace=True)
        
        return results
    
    def _calculate_subgroup_metrics(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        subgroup_name: str
    ) -> SubgroupMetrics:
        """Calculate comprehensive metrics for a subgroup"""
        
        # Handle probabilistic predictions
        if predictions.ndim > 1 or (predictions.max() <= 1.0 and predictions.min() >= 0.0):
            # Probabilistic predictions
            if predictions.ndim > 1:
                pred_probs = predictions[:, 1] if predictions.shape[1] == 2 else predictions
            else:
                pred_probs = predictions
            pred_binary = (pred_probs > 0.5).astype(int)
        else:
            # Binary predictions
            pred_binary = predictions.astype(int)
            pred_probs = None
        
        # Basic metrics
        accuracy = accuracy_score(true_labels, pred_binary)
        precision = precision_score(true_labels, pred_binary, zero_division=0)
        recall = recall_score(true_labels, pred_binary, zero_division=0)
        f1 = f1_score(true_labels, pred_binary, zero_division=0)
        
        # Confusion matrix for additional metrics
        tn, fp, fn, tp = confusion_matrix(true_labels, pred_binary).ravel()
        
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        npv = tn / (tn + fn) if (tn + fn) > 0 else 0
        prevalence = np.mean(true_labels)
        
        # AUC metrics (if probabilistic predictions available)
        auc_roc = None
        auc_pr = None
        if pred_probs is not None and len(np.unique(true_labels)) > 1:
            try:
                auc_roc = roc_auc_score(true_labels, pred_probs)
                auc_pr = average_precision_score(true_labels, pred_probs)
            except ValueError:
                pass  # Handle edge cases
        
        # Calculate confidence intervals
        n = len(true_labels)
        confidence_intervals = self._calculate_confidence_intervals(
            accuracy, precision, recall, f1, specificity, npv, n
        )
        
        return SubgroupMetrics(
            subgroup_name=subgroup_name.split('=')[0] if '=' in subgroup_name else 'subgroup',
            subgroup_value=subgroup_name.split('=')[1] if '=' in subgroup_name else subgroup_name,
            n_samples=n,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1,
            auc_roc=auc_roc,
            auc_pr=auc_pr,
            specificity=specificity,
            npv=npv,
            prevalence=prevalence,
            confidence_intervals=confidence_intervals
        )
    
    def _calculate_confidence_intervals(
        self,
        accuracy: float,
        precision: float,
        recall: float,
        f1: float,
        specificity: float,
        npv: float,
        n: int,
        confidence_level: float = 0.95
    ) -> Dict[str, Tuple[float, float]]:
        """Calculate confidence intervals for metrics"""
        z_score = stats.norm.ppf((1 + confidence_level) / 2)
        
        def proportion_ci(p: float, n: int) -> Tuple[float, float]:
            """Wilson score interval for proportions"""
            if n == 0:
                return (0.0, 0.0)
            
            p_adj = (p + z_score**2 / (2*n)) / (1 + z_score**2 / n)
            margin = z_score * np.sqrt(p*(1-p)/n + z_score**2/(4*n**2)) / (1 + z_score**2/n)
            
            return (max(0, p_adj - margin), min(1, p_adj + margin))
        
        return {
            'accuracy': proportion_ci(accuracy, n),
            'precision': proportion_ci(precision, n),
            'recall': proportion_ci(recall, n),
            'specificity': proportion_ci(specificity, n),
            'npv': proportion_ci(npv, n)
        }
    
    def compare_subgroups(
        self,
        subgroup_metrics: List[SubgroupMetrics],
        metric_name: str = 'accuracy'
    ) -> List[SubgroupComparison]:
        """Compare performance between all pairs of subgroups"""
        comparisons = []
        
        for i, subgroup1 in enumerate(subgroup_metrics):
            for j, subgroup2 in enumerate(subgroup_metrics[i+1:], i+1):
                
                # Get metric values
                metric1 = getattr(subgroup1, metric_name)
                metric2 = getattr(subgroup2, metric_name)
                
                if metric1 is None or metric2 is None:
                    continue
                
                # Calculate difference
                difference = metric1 - metric2
                
                # Statistical test (using normal approximation)
                n1, n2 = subgroup1.n_samples, subgroup2.n_samples
                se1 = np.sqrt(metric1 * (1 - metric1) / n1)
                se2 = np.sqrt(metric2 * (1 - metric2) / n2)
                se_diff = np.sqrt(se1**2 + se2**2)
                
                if se_diff > 0:
                    z_stat = difference / se_diff
                    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
                else:
                    z_stat = 0
                    p_value = 1.0
                
                # Effect size (Cohen's h for proportions)
                h = 2 * (np.arcsin(np.sqrt(metric1)) - np.arcsin(np.sqrt(metric2)))
                
                comparison = SubgroupComparison(
                    subgroup1=f"{subgroup1.subgroup_name}={subgroup1.subgroup_value}",
                    subgroup2=f"{subgroup2.subgroup_name}={subgroup2.subgroup_value}",
                    metric=metric_name,
                    difference=difference,
                    p_value=p_value,
                    effect_size=h,
                    significant=p_value < self.alpha
                )
                
                comparisons.append(comparison)
        
        return comparisons
    
    def detect_bias(
        self,
        subgroup_results: Dict[str, List[SubgroupMetrics]],
        reference_metric: str = 'accuracy',
        bias_threshold: float = 0.05
    ) -> Dict[str, Any]:
        """Detect potential bias across subgroups"""
        bias_report = {
            'overall_bias_detected': False,
            'biased_variables': [],
            'bias_details': {}
        }
        
        for variable, metrics_list in subgroup_results.items():
            if len(metrics_list) < 2:
                continue
            
            # Get metric values
            metric_values = [getattr(m, reference_metric) for m in metrics_list if getattr(m, reference_metric) is not None]
            subgroup_names = [f"{m.subgroup_name}={m.subgroup_value}" for m in metrics_list if getattr(m, reference_metric) is not None]
            
            if len(metric_values) < 2:
                continue
            
            # Calculate range and standard deviation
            metric_range = max(metric_values) - min(metric_values)
            metric_std = np.std(metric_values)
            
            # Detect bias based on threshold
            bias_detected = metric_range > bias_threshold
            
            if bias_detected:
                bias_report['overall_bias_detected'] = True
                bias_report['biased_variables'].append(variable)
                
                # Find worst and best performing subgroups
                min_idx = np.argmin(metric_values)
                max_idx = np.argmax(metric_values)
                
                bias_report['bias_details'][variable] = {
                    'metric_range': metric_range,
                    'metric_std': metric_std,
                    'worst_subgroup': subgroup_names[min_idx],
                    'worst_performance': metric_values[min_idx],
                    'best_subgroup': subgroup_names[max_idx],
                    'best_performance': metric_values[max_idx],
                    'all_performances': dict(zip(subgroup_names, metric_values))
                }
        
        return bias_report
    
    def generate_subgroup_report(
        self,
        subgroup_results: Dict[str, List[SubgroupMetrics]],
        comparisons: Optional[Dict[str, List[SubgroupComparison]]] = None,
        bias_report: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate comprehensive subgroup analysis report"""
        
        report = {
            'summary': {
                'total_variables_analyzed': len(subgroup_results),
                'total_subgroups': sum(len(metrics) for metrics in subgroup_results.values()),
                'min_subgroup_size': self.min_subgroup_size
            },
            'subgroup_performance': {},
            'bias_analysis': bias_report or {},
            'statistical_comparisons': comparisons or {},
            'recommendations': []
        }
        
        # Summarize performance by variable
        for variable, metrics_list in subgroup_results.items():
            performances = {
                'accuracy': [m.accuracy for m in metrics_list],
                'precision': [m.precision for m in metrics_list],
                'recall': [m.recall for m in metrics_list],
                'f1_score': [m.f1_score for m in metrics_list]
            }
            
            report['subgroup_performance'][variable] = {
                'n_subgroups': len(metrics_list),
                'performance_summary': {
                    metric: {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'range': np.max(values) - np.min(values)
                    }
                    for metric, values in performances.items()
                },
                'subgroup_details': [
                    {
                        'name': f"{m.subgroup_name}={m.subgroup_value}",
                        'n_samples': m.n_samples,
                        'accuracy': m.accuracy,
                        'precision': m.precision,
                        'recall': m.recall,
                        'f1_score': m.f1_score
                    }
                    for m in metrics_list
                ]
            }
        
        # Generate recommendations
        if bias_report and bias_report.get('overall_bias_detected'):
            report['recommendations'].append(
                "Bias detected across subgroups. Consider additional training data or bias mitigation techniques."
            )
        
        # Check for small subgroups
        small_subgroups = []
        for variable, metrics_list in subgroup_results.items():
            for m in metrics_list:
                if m.n_samples < 50:  # Flag subgroups with < 50 samples
                    small_subgroups.append(f"{variable}={m.subgroup_value} (n={m.n_samples})")
        
        if small_subgroups:
            report['recommendations'].append(
                f"Small subgroups detected: {', '.join(small_subgroups[:3])}{'...' if len(small_subgroups) > 3 else ''}. "
                "Consider collecting more data for robust analysis."
            )
        
        return report
    
    def visualize_subgroup_performance(
        self,
        subgroup_results: Dict[str, List[SubgroupMetrics]],
        metric: str = 'accuracy',
        figsize: Tuple[int, int] = (12, 8)
    ) -> plt.Figure:
        """Create visualization of subgroup performance"""
        
        fig, axes = plt.subplots(len(subgroup_results), 1, figsize=figsize, squeeze=False)
        axes = axes.flatten()
        
        for idx, (variable, metrics_list) in enumerate(subgroup_results.items()):
            ax = axes[idx]
            
            # Extract data for plotting
            subgroup_names = [m.subgroup_value for m in metrics_list]
            metric_values = [getattr(m, metric) for m in metrics_list]
            sample_sizes = [m.n_samples for m in metrics_list]
            
            # Create bar plot
            bars = ax.bar(subgroup_names, metric_values)
            
            # Color bars by performance (red for low, green for high)
            norm = plt.Normalize(min(metric_values), max(metric_values))
            colors = plt.cm.RdYlGn(norm(metric_values))
            for bar, color in zip(bars, colors):
                bar.set_color(color)
            
            # Add sample size annotations
            for i, (bar, n) in enumerate(zip(bars, sample_sizes)):
                ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                       f'n={n}', ha='center', va='bottom', fontsize=8)
            
            ax.set_title(f'{variable.title()} - {metric.title()}')
            ax.set_ylabel(metric.title())
            ax.set_ylim(0, 1)
            
            # Rotate x-axis labels if needed
            if len(subgroup_names) > 3:
                ax.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        return fig

# Example usage and testing
if __name__ == "__main__":
    # Initialize subgroup analyzer
    analyzer = ClinicalSubgroupAnalyzer(alpha=0.05, min_subgroup_size=30)
    
    # Generate sample data
    np.random.seed(42)
    n_samples = 1000
    
    # Create sample dataset with demographic and clinical variables
    sample_data = pd.DataFrame({
        'age_group': np.random.choice(['18-30', '31-50', '51-70', '70+'], n_samples),
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'race': np.random.choice(['White', 'Black', 'Hispanic', 'Asian'], n_samples),
        'hospital_type': np.random.choice(['Academic', 'Community', 'Rural'], n_samples),
        'disease_stage': np.random.choice(['Early', 'Advanced'], n_samples, p=[0.7, 0.3])
    })
    
    # Generate predictions and labels with some bias
    true_labels = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    
    # Introduce bias: lower accuracy for certain subgroups
    predictions = true_labels.copy().astype(float)
    
    # Add noise with bias
    for i in range(n_samples):
        if sample_data.iloc[i]['race'] == 'Black':
            # Higher error rate for Black patients (simulating bias)
            if np.random.random() < 0.15:
                predictions[i] = 1 - predictions[i]
        elif sample_data.iloc[i]['hospital_type'] == 'Rural':
            # Higher error rate for rural hospitals
            if np.random.random() < 0.12:
                predictions[i] = 1 - predictions[i]
        else:
            # Lower error rate for other groups
            if np.random.random() < 0.08:
                predictions[i] = 1 - predictions[i]
    
    print("Clinical AI Subgroup Analysis")
    print("=" * 50)
    
    # Analyze demographic subgroups
    demo_results = analyzer.analyze_demographic_subgroups(
        sample_data, predictions, true_labels, ['age_group', 'gender', 'race']
    )
    
    # Analyze clinical subgroups
    clinical_results = analyzer.analyze_clinical_subgroups(
        sample_data, predictions, true_labels, ['hospital_type', 'disease_stage']
    )
    
    # Combine results
    all_results = {**demo_results, **clinical_results}
    
    # Detect bias
    bias_report = analyzer.detect_bias(all_results, 'accuracy', bias_threshold=0.05)
    
    # Generate comprehensive report
    report = analyzer.generate_subgroup_report(all_results, bias_report=bias_report)
    
    print(f"\nSubgroup Analysis Summary:")
    print(f"  Variables analyzed: {report['summary']['total_variables_analyzed']}")
    print(f"  Total subgroups: {report['summary']['total_subgroups']}")
    print(f"  Bias detected: {bias_report['overall_bias_detected']}")
    
    if bias_report['overall_bias_detected']:
        print(f"  Biased variables: {', '.join(bias_report['biased_variables'])}")
        
        for var in bias_report['biased_variables']:
            details = bias_report['bias_details'][var]
            print(f"\n  {var} bias details:")
            print(f"    Performance range: {details['metric_range']:.3f}")
            print(f"    Worst: {details['worst_subgroup']} ({details['worst_performance']:.3f})")
            print(f"    Best: {details['best_subgroup']} ({details['best_performance']:.3f})")
    
    # Show performance by race (example of bias detection)
    if 'race' in demo_results:
        print(f"\nPerformance by Race:")
        for metrics in demo_results['race']:
            print(f"  {metrics.subgroup_value}: {metrics.accuracy:.3f} (n={metrics.n_samples})")
    
    print(f"\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  - {rec}")