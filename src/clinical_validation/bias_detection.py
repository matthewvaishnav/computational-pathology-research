"""
Bias Detection and Fairness Analysis for Clinical AI

Comprehensive bias detection framework for identifying and quantifying
fairness issues in medical AI systems across protected attributes.
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
    roc_auc_score, confusion_matrix, equalized_odds_difference
)
from scipy import stats
import warnings
import logging

logger = logging.getLogger(__name__)

class BiasMetric(Enum):
    """Types of bias metrics"""
    DEMOGRAPHIC_PARITY = "demographic_parity"
    EQUALIZED_ODDS = "equalized_odds"
    PREDICTIVE_PARITY = "predictive_parity"
    CALIBRATION = "calibration"
    DISPARATE_IMPACT = "disparate_impact"
    THEIL_INDEX = "theil_index"
    EQUAL_OPPORTUNITY = "equal_opportunity"

@dataclass
class BiasResult:
    """Result of bias detection analysis"""
    metric: BiasMetric
    protected_attribute: str
    reference_group: str
    comparison_group: str
    reference_value: float
    comparison_value: float
    difference: float
    ratio: float
    p_value: Optional[float] = None
    significant: bool = False
    bias_detected: bool = False
    severity: str = "none"  # none, low, medium, high, critical

@dataclass
class FairnessReport:
    """Comprehensive fairness report"""
    overall_bias_score: float  # 0-1, higher = more bias
    bias_results: List[BiasResult]
    protected_attributes: List[str]
    recommendations: List[str]
    mitigation_strategies: List[str]

class BiasDetector:
    """Comprehensive bias detection for clinical AI"""
    
    def __init__(self, alpha: float = 0.05, bias_threshold: float = 0.1):
        """Initialize bias detector"""
        self.alpha = alpha
        self.bias_threshold = bias_threshold
        
    def detect_demographic_parity_bias(
        self,
        predictions: np.ndarray,
        protected_attribute: np.ndarray,
        reference_group: Any,
        comparison_group: Any
    ) -> BiasResult:
        """
        Detect demographic parity bias.
        
        Demographic parity: P(Y_pred=1|A=a) = P(Y_pred=1|A=b)
        """
        ref_mask = protected_attribute == reference_group
        comp_mask = protected_attribute == comparison_group
        
        ref_positive_rate = np.mean(predictions[ref_mask])
        comp_positive_rate = np.mean(predictions[comp_mask])
        
        difference = ref_positive_rate - comp_positive_rate
        ratio = comp_positive_rate / (ref_positive_rate + 1e-10)
        
        # Statistical test
        n_ref = np.sum(ref_mask)
        n_comp = np.sum(comp_mask)
        
        se = np.sqrt(
            (ref_positive_rate * (1 - ref_positive_rate) / n_ref) +
            (comp_positive_rate * (1 - comp_positive_rate) / n_comp)
        )
        
        z_stat = difference / (se + 1e-10)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
        
        bias_detected = abs(difference) > self.bias_threshold
        severity = self._assess_severity(abs(difference))
        
        return BiasResult(
            metric=BiasMetric.DEMOGRAPHIC_PARITY,
            protected_attribute=str(protected_attribute),
            reference_group=str(reference_group),
            comparison_group=str(comparison_group),
            reference_value=ref_positive_rate,
            comparison_value=comp_positive_rate,
            difference=difference,
            ratio=ratio,
            p_value=p_value,
            significant=p_value < self.alpha,
            bias_detected=bias_detected,
            severity=severity
        )
    
    def detect_equalized_odds_bias(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        protected_attribute: np.ndarray,
        reference_group: Any,
        comparison_group: Any
    ) -> BiasResult:
        """
        Detect equalized odds bias.
        
        Equalized odds: TPR and FPR equal across groups
        """
        ref_mask = protected_attribute == reference_group
        comp_mask = protected_attribute == comparison_group
        
        # Calculate TPR and FPR for reference group
        ref_tp = np.sum((predictions[ref_mask] == 1) & (true_labels[ref_mask] == 1))
        ref_fp = np.sum((predictions[ref_mask] == 1) & (true_labels[ref_mask] == 0))
        ref_fn = np.sum((predictions[ref_mask] == 0) & (true_labels[ref_mask] == 1))
        ref_tn = np.sum((predictions[ref_mask] == 0) & (true_labels[ref_mask] == 0))
        
        ref_tpr = ref_tp / (ref_tp + ref_fn + 1e-10)
        ref_fpr = ref_fp / (ref_fp + ref_tn + 1e-10)
        
        # Calculate TPR and FPR for comparison group
        comp_tp = np.sum((predictions[comp_mask] == 1) & (true_labels[comp_mask] == 1))
        comp_fp = np.sum((predictions[comp_mask] == 1) & (true_labels[comp_mask] == 0))
        comp_fn = np.sum((predictions[comp_mask] == 0) & (true_labels[comp_mask] == 1))
        comp_tn = np.sum((predictions[comp_mask] == 0) & (true_labels[comp_mask] == 0))
        
        comp_tpr = comp_tp / (comp_tp + comp_fn + 1e-10)
        comp_fpr = comp_fp / (comp_fp + comp_tn + 1e-10)
        
        # Equalized odds difference (max of TPR and FPR differences)
        tpr_diff = abs(ref_tpr - comp_tpr)
        fpr_diff = abs(ref_fpr - comp_fpr)
        difference = max(tpr_diff, fpr_diff)
        
        bias_detected = difference > self.bias_threshold
        severity = self._assess_severity(difference)
        
        return BiasResult(
            metric=BiasMetric.EQUALIZED_ODDS,
            protected_attribute=str(protected_attribute),
            reference_group=str(reference_group),
            comparison_group=str(comparison_group),
            reference_value=ref_tpr,
            comparison_value=comp_tpr,
            difference=difference,
            ratio=comp_tpr / (ref_tpr + 1e-10),
            bias_detected=bias_detected,
            severity=severity
        )
    
    def detect_predictive_parity_bias(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        protected_attribute: np.ndarray,
        reference_group: Any,
        comparison_group: Any
    ) -> BiasResult:
        """
        Detect predictive parity bias.
        
        Predictive parity: PPV (precision) equal across groups
        """
        ref_mask = protected_attribute == reference_group
        comp_mask = protected_attribute == comparison_group
        
        # Calculate precision for reference group
        ref_tp = np.sum((predictions[ref_mask] == 1) & (true_labels[ref_mask] == 1))
        ref_fp = np.sum((predictions[ref_mask] == 1) & (true_labels[ref_mask] == 0))
        ref_precision = ref_tp / (ref_tp + ref_fp + 1e-10)
        
        # Calculate precision for comparison group
        comp_tp = np.sum((predictions[comp_mask] == 1) & (true_labels[comp_mask] == 1))
        comp_fp = np.sum((predictions[comp_mask] == 1) & (true_labels[comp_mask] == 0))
        comp_precision = comp_tp / (comp_tp + comp_fp + 1e-10)
        
        difference = abs(ref_precision - comp_precision)
        ratio = comp_precision / (ref_precision + 1e-10)
        
        bias_detected = difference > self.bias_threshold
        severity = self._assess_severity(difference)
        
        return BiasResult(
            metric=BiasMetric.PREDICTIVE_PARITY,
            protected_attribute=str(protected_attribute),
            reference_group=str(reference_group),
            comparison_group=str(comparison_group),
            reference_value=ref_precision,
            comparison_value=comp_precision,
            difference=difference,
            ratio=ratio,
            bias_detected=bias_detected,
            severity=severity
        )
    
    def detect_calibration_bias(
        self,
        predictions_proba: np.ndarray,
        true_labels: np.ndarray,
        protected_attribute: np.ndarray,
        reference_group: Any,
        comparison_group: Any,
        n_bins: int = 10
    ) -> BiasResult:
        """
        Detect calibration bias.
        
        Calibration: P(Y=1|score=s) should be equal across groups
        """
        ref_mask = protected_attribute == reference_group
        comp_mask = protected_attribute == comparison_group
        
        # Bin predictions
        bins = np.linspace(0, 1, n_bins + 1)
        
        # Calculate calibration error for reference group
        ref_calibration_error = self._calculate_calibration_error(
            predictions_proba[ref_mask], true_labels[ref_mask], bins
        )
        
        # Calculate calibration error for comparison group
        comp_calibration_error = self._calculate_calibration_error(
            predictions_proba[comp_mask], true_labels[comp_mask], bins
        )
        
        difference = abs(ref_calibration_error - comp_calibration_error)
        
        bias_detected = difference > self.bias_threshold
        severity = self._assess_severity(difference)
        
        return BiasResult(
            metric=BiasMetric.CALIBRATION,
            protected_attribute=str(protected_attribute),
            reference_group=str(reference_group),
            comparison_group=str(comparison_group),
            reference_value=ref_calibration_error,
            comparison_value=comp_calibration_error,
            difference=difference,
            ratio=comp_calibration_error / (ref_calibration_error + 1e-10),
            bias_detected=bias_detected,
            severity=severity
        )
    
    def detect_disparate_impact(
        self,
        predictions: np.ndarray,
        protected_attribute: np.ndarray,
        reference_group: Any,
        comparison_group: Any,
        threshold: float = 0.8
    ) -> BiasResult:
        """
        Detect disparate impact (4/5 rule).
        
        Disparate impact: Selection rate ratio < 0.8 indicates bias
        """
        ref_mask = protected_attribute == reference_group
        comp_mask = protected_attribute == comparison_group
        
        ref_selection_rate = np.mean(predictions[ref_mask])
        comp_selection_rate = np.mean(predictions[comp_mask])
        
        # Disparate impact ratio
        di_ratio = comp_selection_rate / (ref_selection_rate + 1e-10)
        
        bias_detected = di_ratio < threshold
        severity = self._assess_severity(1 - di_ratio)
        
        return BiasResult(
            metric=BiasMetric.DISPARATE_IMPACT,
            protected_attribute=str(protected_attribute),
            reference_group=str(reference_group),
            comparison_group=str(comparison_group),
            reference_value=ref_selection_rate,
            comparison_value=comp_selection_rate,
            difference=ref_selection_rate - comp_selection_rate,
            ratio=di_ratio,
            bias_detected=bias_detected,
            severity=severity
        )
    
    def detect_theil_index_bias(
        self,
        predictions: np.ndarray,
        protected_attribute: np.ndarray
    ) -> BiasResult:
        """
        Detect bias using Theil index (entropy-based measure).
        
        Theil index: Measures inequality in prediction distribution
        """
        unique_groups = np.unique(protected_attribute)
        
        if len(unique_groups) < 2:
            raise ValueError("Need at least 2 groups for Theil index")
        
        # Calculate Theil index
        theil_index = 0
        total_samples = len(predictions)
        
        for group in unique_groups:
            mask = protected_attribute == group
            group_size = np.sum(mask)
            group_positive_rate = np.mean(predictions[mask])
            overall_positive_rate = np.mean(predictions)
            
            if group_positive_rate > 0 and overall_positive_rate > 0:
                theil_index += (group_size / total_samples) * np.log(
                    group_positive_rate / overall_positive_rate
                )
        
        bias_detected = abs(theil_index) > self.bias_threshold
        severity = self._assess_severity(abs(theil_index))
        
        return BiasResult(
            metric=BiasMetric.THEIL_INDEX,
            protected_attribute=str(protected_attribute),
            reference_group=unique_groups[0],
            comparison_group=unique_groups[1],
            reference_value=theil_index,
            comparison_value=0,  # Reference value
            difference=abs(theil_index),
            ratio=1.0,
            bias_detected=bias_detected,
            severity=severity
        )
    
    def _calculate_calibration_error(
        self,
        predictions_proba: np.ndarray,
        true_labels: np.ndarray,
        bins: np.ndarray
    ) -> float:
        """Calculate calibration error (ECE - Expected Calibration Error)"""
        calibration_error = 0
        
        for i in range(len(bins) - 1):
            mask = (predictions_proba >= bins[i]) & (predictions_proba < bins[i+1])
            
            if np.sum(mask) > 0:
                bin_accuracy = np.mean(true_labels[mask])
                bin_confidence = np.mean(predictions_proba[mask])
                calibration_error += np.sum(mask) / len(predictions_proba) * abs(bin_accuracy - bin_confidence)
        
        return calibration_error
    
    def _assess_severity(self, difference: float) -> str:
        """Assess severity of bias"""
        if difference < 0.05:
            return "none"
        elif difference < 0.1:
            return "low"
        elif difference < 0.2:
            return "medium"
        elif difference < 0.3:
            return "high"
        else:
            return "critical"
    
    def comprehensive_bias_analysis(
        self,
        predictions: np.ndarray,
        true_labels: np.ndarray,
        data: pd.DataFrame,
        protected_attributes: List[str],
        predictions_proba: Optional[np.ndarray] = None
    ) -> FairnessReport:
        """Perform comprehensive bias analysis across all protected attributes"""
        
        all_bias_results = []
        
        for attr in protected_attributes:
            if attr not in data.columns:
                logger.warning(f"Protected attribute {attr} not found in data")
                continue
            
            unique_groups = data[attr].unique()
            
            if len(unique_groups) < 2:
                logger.warning(f"Protected attribute {attr} has less than 2 groups")
                continue
            
            # Compare each pair of groups
            for i, ref_group in enumerate(unique_groups):
                for comp_group in unique_groups[i+1:]:
                    
                    # Demographic parity
                    dp_result = self.detect_demographic_parity_bias(
                        predictions, data[attr].values, ref_group, comp_group
                    )
                    all_bias_results.append(dp_result)
                    
                    # Equalized odds
                    eo_result = self.detect_equalized_odds_bias(
                        predictions, true_labels, data[attr].values, ref_group, comp_group
                    )
                    all_bias_results.append(eo_result)
                    
                    # Predictive parity
                    pp_result = self.detect_predictive_parity_bias(
                        predictions, true_labels, data[attr].values, ref_group, comp_group
                    )
                    all_bias_results.append(pp_result)
                    
                    # Calibration (if probabilistic predictions available)
                    if predictions_proba is not None:
                        cal_result = self.detect_calibration_bias(
                            predictions_proba, true_labels, data[attr].values, ref_group, comp_group
                        )
                        all_bias_results.append(cal_result)
                    
                    # Disparate impact
                    di_result = self.detect_disparate_impact(
                        predictions, data[attr].values, ref_group, comp_group
                    )
                    all_bias_results.append(di_result)
        
        # Calculate overall bias score
        bias_scores = [1.0 if r.bias_detected else 0.0 for r in all_bias_results]
        overall_bias_score = np.mean(bias_scores) if bias_scores else 0.0
        
        # Generate recommendations
        recommendations = self._generate_recommendations(all_bias_results)
        mitigation_strategies = self._generate_mitigation_strategies(all_bias_results)
        
        return FairnessReport(
            overall_bias_score=overall_bias_score,
            bias_results=all_bias_results,
            protected_attributes=protected_attributes,
            recommendations=recommendations,
            mitigation_strategies=mitigation_strategies
        )
    
    def _generate_recommendations(self, bias_results: List[BiasResult]) -> List[str]:
        """Generate recommendations based on bias results"""
        recommendations = []
        
        critical_biases = [r for r in bias_results if r.severity == "critical"]
        high_biases = [r for r in bias_results if r.severity == "high"]
        
        if critical_biases:
            recommendations.append(
                f"CRITICAL: {len(critical_biases)} critical bias issues detected. "
                "Immediate action required before deployment."
            )
        
        if high_biases:
            recommendations.append(
                f"HIGH: {len(high_biases)} high-severity bias issues detected. "
                "Recommend bias mitigation before production use."
            )
        
        # Specific recommendations by metric
        dp_biases = [r for r in bias_results if r.metric == BiasMetric.DEMOGRAPHIC_PARITY and r.bias_detected]
        if dp_biases:
            recommendations.append(
                "Demographic parity violations detected. Consider rebalancing training data "
                "or adjusting decision thresholds per group."
            )
        
        eo_biases = [r for r in bias_results if r.metric == BiasMetric.EQUALIZED_ODDS and r.bias_detected]
        if eo_biases:
            recommendations.append(
                "Equalized odds violations detected. Model performance differs significantly "
                "across groups. Consider group-specific model tuning."
            )
        
        return recommendations
    
    def _generate_mitigation_strategies(self, bias_results: List[BiasResult]) -> List[str]:
        """Generate mitigation strategies"""
        strategies = []
        
        strategies.append("1. Data augmentation: Collect more balanced training data for underrepresented groups")
        strategies.append("2. Reweighting: Apply sample weights to balance group representation during training")
        strategies.append("3. Threshold adjustment: Use group-specific decision thresholds")
        strategies.append("4. Fairness constraints: Add fairness regularization to loss function")
        strategies.append("5. Adversarial debiasing: Train model with adversarial fairness objectives")
        strategies.append("6. Post-processing: Apply calibration or threshold optimization per group")
        strategies.append("7. Feature engineering: Remove or transform features correlated with protected attributes")
        strategies.append("8. Ensemble methods: Combine models trained on different subgroups")
        
        return strategies
    
    def generate_fairness_report(self, fairness_report: FairnessReport) -> Dict[str, Any]:
        """Generate detailed fairness report"""
        
        report = {
            'summary': {
                'overall_bias_score': fairness_report.overall_bias_score,
                'total_bias_tests': len(fairness_report.bias_results),
                'critical_issues': sum(1 for r in fairness_report.bias_results if r.severity == "critical"),
                'high_issues': sum(1 for r in fairness_report.bias_results if r.severity == "high"),
                'medium_issues': sum(1 for r in fairness_report.bias_results if r.severity == "medium"),
                'low_issues': sum(1 for r in fairness_report.bias_results if r.severity == "low")
            },
            'bias_results_by_metric': {},
            'bias_results_by_attribute': {},
            'recommendations': fairness_report.recommendations,
            'mitigation_strategies': fairness_report.mitigation_strategies
        }
        
        # Group by metric
        for metric in BiasMetric:
            metric_results = [r for r in fairness_report.bias_results if r.metric == metric]
            if metric_results:
                report['bias_results_by_metric'][metric.value] = [
                    {
                        'reference_group': r.reference_group,
                        'comparison_group': r.comparison_group,
                        'difference': r.difference,
                        'ratio': r.ratio,
                        'p_value': r.p_value,
                        'bias_detected': r.bias_detected,
                        'severity': r.severity
                    }
                    for r in metric_results
                ]
        
        # Group by attribute
        for attr in fairness_report.protected_attributes:
            attr_results = [r for r in fairness_report.bias_results if r.protected_attribute == attr]
            if attr_results:
                report['bias_results_by_attribute'][attr] = [
                    {
                        'metric': r.metric.value,
                        'difference': r.difference,
                        'bias_detected': r.bias_detected,
                        'severity': r.severity
                    }
                    for r in attr_results
                ]
        
        return report

# Example usage and testing
if __name__ == "__main__":
    # Initialize bias detector
    detector = BiasDetector(alpha=0.05, bias_threshold=0.1)
    
    # Generate sample data with bias
    np.random.seed(42)
    n_samples = 1000
    
    # Create biased predictions
    true_labels = np.random.choice([0, 1], n_samples, p=[0.7, 0.3])
    predictions = true_labels.copy()
    
    # Introduce bias: lower accuracy for certain groups
    protected_attr = np.random.choice(['Group_A', 'Group_B', 'Group_C'], n_samples)
    
    for i in range(n_samples):
        if protected_attr[i] == 'Group_B':
            # Higher error rate for Group B
            if np.random.random() < 0.2:
                predictions[i] = 1 - predictions[i]
        elif protected_attr[i] == 'Group_C':
            # Even higher error rate for Group C
            if np.random.random() < 0.25:
                predictions[i] = 1 - predictions[i]
        else:
            # Lower error rate for Group A
            if np.random.random() < 0.1:
                predictions[i] = 1 - predictions[i]
    
    # Create DataFrame
    data = pd.DataFrame({
        'protected_attribute': protected_attr,
        'age_group': np.random.choice(['Young', 'Middle', 'Old'], n_samples),
        'gender': np.random.choice(['M', 'F'], n_samples)
    })
    
    print("Bias Detection Analysis")
    print("=" * 50)
    
    # Perform comprehensive analysis
    fairness_report = detector.comprehensive_bias_analysis(
        predictions, true_labels, data, ['protected_attribute', 'gender']
    )
    
    # Generate report
    report = detector.generate_fairness_report(fairness_report)
    
    print(f"\nOverall Bias Score: {report['summary']['overall_bias_score']:.3f}")
    print(f"Critical Issues: {report['summary']['critical_issues']}")
    print(f"High Issues: {report['summary']['high_issues']}")
    print(f"Medium Issues: {report['summary']['medium_issues']}")
    
    print(f"\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  - {rec}")
    
    print(f"\nMitigation Strategies:")
    for strategy in report['mitigation_strategies'][:3]:
        print(f"  - {strategy}")