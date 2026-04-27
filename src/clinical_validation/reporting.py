"""
Publication-Ready Reporting for Clinical AI Validation

Generates publication-quality tables, figures, and reports for medical AI validation studies.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class PublicationReporter:
    """Generate publication-ready reports and tables"""
    
    def __init__(self, study_name: str = "Clinical AI Validation Study"):
        """Initialize reporter"""
        self.study_name = study_name
        self.timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
    def create_demographics_table(
        self,
        data: pd.DataFrame,
        demographic_columns: List[str],
        stratify_by: Optional[str] = None
    ) -> pd.DataFrame:
        """Create publication-ready demographics table"""
        
        table_data = []
        
        for col in demographic_columns:
            if col not in data.columns:
                continue
            
            if stratify_by and stratify_by in data.columns:
                # Stratified demographics
                for group in data[stratify_by].unique():
                    group_data = data[data[stratify_by] == group]
                    
                    if data[col].dtype in ['object', 'category']:
                        # Categorical variable
                        value_counts = group_data[col].value_counts()
                        for value, count in value_counts.items():
                            pct = 100 * count / len(group_data)
                            table_data.append({
                                'Variable': col,
                                'Category': value,
                                'Group': group,
                                'N': count,
                                'Percentage': f"{pct:.1f}%"
                            })
                    else:
                        # Continuous variable
                        mean = group_data[col].mean()
                        std = group_data[col].std()
                        table_data.append({
                            'Variable': col,
                            'Category': 'Mean ± SD',
                            'Group': group,
                            'N': len(group_data),
                            'Percentage': f"{mean:.2f} ± {std:.2f}"
                        })
            else:
                # Overall demographics
                if data[col].dtype in ['object', 'category']:
                    value_counts = data[col].value_counts()
                    for value, count in value_counts.items():
                        pct = 100 * count / len(data)
                        table_data.append({
                            'Variable': col,
                            'Category': value,
                            'N': count,
                            'Percentage': f"{pct:.1f}%"
                        })
                else:
                    mean = data[col].mean()
                    std = data[col].std()
                    table_data.append({
                        'Variable': col,
                        'Category': 'Mean ± SD',
                        'N': len(data),
                        'Percentage': f"{mean:.2f} ± {std:.2f}"
                    })
        
        return pd.DataFrame(table_data)
    
    def create_performance_table(
        self,
        metrics_dict: Dict[str, Dict[str, float]],
        include_ci: bool = True
    ) -> pd.DataFrame:
        """Create publication-ready performance metrics table"""
        
        table_data = []
        
        for model_name, metrics in metrics_dict.items():
            row = {'Model': model_name}
            
            # Add metrics
            for metric_name, value in metrics.items():
                if isinstance(value, tuple):
                    # Confidence interval
                    if include_ci:
                        row[metric_name] = f"{value[0]:.3f}-{value[1]:.3f}"
                elif isinstance(value, float):
                    row[metric_name] = f"{value:.3f}"
                else:
                    row[metric_name] = str(value)
            
            table_data.append(row)
        
        return pd.DataFrame(table_data)
    
    def create_comparison_table(
        self,
        comparisons: List[Dict[str, Any]],
        metrics: List[str] = ['accuracy', 'sensitivity', 'specificity', 'auc_roc']
    ) -> pd.DataFrame:
        """Create model comparison table"""
        
        table_data = []
        
        for comp in comparisons:
            row = {
                'Model 1': comp.get('model1', ''),
                'Model 2': comp.get('model2', ''),
                'Metric': comp.get('metric', ''),
                'Difference': f"{comp.get('difference', 0):.4f}",
                'P-value': f"{comp.get('p_value', 1.0):.4f}",
                'Significant': 'Yes' if comp.get('p_value', 1.0) < 0.05 else 'No'
            }
            table_data.append(row)
        
        return pd.DataFrame(table_data)
    
    def create_subgroup_table(
        self,
        subgroup_results: Dict[str, List[Dict[str, Any]]]
    ) -> pd.DataFrame:
        """Create subgroup analysis table"""
        
        table_data = []
        
        for variable, results in subgroup_results.items():
            for result in results:
                row = {
                    'Variable': variable,
                    'Subgroup': result.get('subgroup', ''),
                    'N': result.get('n_samples', 0),
                    'Accuracy': f"{result.get('accuracy', 0):.3f}",
                    'Sensitivity': f"{result.get('sensitivity', 0):.3f}",
                    'Specificity': f"{result.get('specificity', 0):.3f}",
                    'AUC': f"{result.get('auc_roc', 0):.3f}"
                }
                table_data.append(row)
        
        return pd.DataFrame(table_data)
    
    def create_bias_table(
        self,
        bias_results: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Create bias analysis table"""
        
        table_data = []
        
        for result in bias_results:
            row = {
                'Metric': result.get('metric', ''),
                'Group 1': result.get('group1', ''),
                'Group 2': result.get('group2', ''),
                'Value 1': f"{result.get('value1', 0):.3f}",
                'Value 2': f"{result.get('value2', 0):.3f}",
                'Difference': f"{result.get('difference', 0):.3f}",
                'P-value': f"{result.get('p_value', 1.0):.4f}",
                'Bias Detected': 'Yes' if result.get('bias_detected', False) else 'No',
                'Severity': result.get('severity', 'None')
            }
            table_data.append(row)
        
        return pd.DataFrame(table_data)
    
    def export_table_to_latex(
        self,
        df: pd.DataFrame,
        caption: str = "",
        label: str = ""
    ) -> str:
        """Export table to LaTeX format"""
        
        latex_str = df.to_latex(index=False)
        
        if caption or label:
            # Wrap in table environment
            latex_str = f"""\\begin{{table}}[h]
\\centering
\\caption{{{caption}}}
\\label{{{label}}}
{latex_str}
\\end{{table}}"""
        
        return latex_str
    
    def export_table_to_html(
        self,
        df: pd.DataFrame,
        title: str = ""
    ) -> str:
        """Export table to HTML format"""
        
        html_str = df.to_html(index=False)
        
        if title:
            html_str = f"<h3>{title}</h3>\n{html_str}"
        
        return html_str
    
    def create_summary_report(
        self,
        study_data: Dict[str, Any],
        output_format: str = "text"
    ) -> str:
        """Create comprehensive summary report"""
        
        report_lines = []
        
        # Header
        report_lines.append("=" * 80)
        report_lines.append(f"CLINICAL AI VALIDATION STUDY REPORT")
        report_lines.append(f"Study: {self.study_name}")
        report_lines.append(f"Generated: {self.timestamp}")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Study Overview
        if 'overview' in study_data:
            report_lines.append("STUDY OVERVIEW")
            report_lines.append("-" * 40)
            for key, value in study_data['overview'].items():
                report_lines.append(f"  {key}: {value}")
            report_lines.append("")
        
        # Demographics
        if 'demographics' in study_data:
            report_lines.append("STUDY POPULATION")
            report_lines.append("-" * 40)
            for key, value in study_data['demographics'].items():
                report_lines.append(f"  {key}: {value}")
            report_lines.append("")
        
        # Primary Results
        if 'primary_results' in study_data:
            report_lines.append("PRIMARY RESULTS")
            report_lines.append("-" * 40)
            for key, value in study_data['primary_results'].items():
                if isinstance(value, float):
                    report_lines.append(f"  {key}: {value:.4f}")
                else:
                    report_lines.append(f"  {key}: {value}")
            report_lines.append("")
        
        # Secondary Results
        if 'secondary_results' in study_data:
            report_lines.append("SECONDARY RESULTS")
            report_lines.append("-" * 40)
            for key, value in study_data['secondary_results'].items():
                if isinstance(value, float):
                    report_lines.append(f"  {key}: {value:.4f}")
                else:
                    report_lines.append(f"  {key}: {value}")
            report_lines.append("")
        
        # Conclusions
        if 'conclusions' in study_data:
            report_lines.append("CONCLUSIONS")
            report_lines.append("-" * 40)
            for conclusion in study_data['conclusions']:
                report_lines.append(f"  • {conclusion}")
            report_lines.append("")
        
        # Recommendations
        if 'recommendations' in study_data:
            report_lines.append("RECOMMENDATIONS")
            report_lines.append("-" * 40)
            for rec in study_data['recommendations']:
                report_lines.append(f"  • {rec}")
            report_lines.append("")
        
        report_lines.append("=" * 80)
        
        return "\n".join(report_lines)
    
    def create_figure_grid(
        self,
        figures: List[plt.Figure],
        titles: List[str],
        figsize: Tuple[int, int] = (16, 12),
        n_cols: int = 2
    ) -> plt.Figure:
        """Create grid of figures for publication"""
        
        n_figs = len(figures)
        n_rows = (n_figs + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_figs > 1 else [axes]
        
        for idx, (figure, title) in enumerate(zip(figures, titles)):
            # Copy figure content to subplot
            ax = axes[idx]
            ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Hide unused subplots
        for idx in range(n_figs, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        return fig
    
    def generate_statistical_summary(
        self,
        test_results: List[Dict[str, Any]]
    ) -> pd.DataFrame:
        """Generate statistical test summary table"""
        
        table_data = []
        
        for result in test_results:
            row = {
                'Test': result.get('test_name', ''),
                'Statistic': f"{result.get('statistic', 0):.4f}",
                'P-value': f"{result.get('p_value', 1.0):.6f}",
                'Effect Size': f"{result.get('effect_size', 0):.4f}" if result.get('effect_size') else "N/A",
                'Significant': 'Yes' if result.get('p_value', 1.0) < 0.05 else 'No',
                'Interpretation': result.get('interpretation', '')
            }
            table_data.append(row)
        
        return pd.DataFrame(table_data)
    
    def create_validation_summary(
        self,
        validation_results: Dict[str, Any]
    ) -> str:
        """Create validation study summary"""
        
        summary_lines = []
        
        summary_lines.append("VALIDATION STUDY SUMMARY")
        summary_lines.append("=" * 60)
        
        # Study Design
        summary_lines.append("\nSTUDY DESIGN:")
        summary_lines.append(f"  Strategy: {validation_results.get('strategy', 'N/A')}")
        summary_lines.append(f"  Number of Folds: {validation_results.get('n_folds', 'N/A')}")
        summary_lines.append(f"  Total Samples: {validation_results.get('total_samples', 'N/A')}")
        
        # Performance Summary
        summary_lines.append("\nPERFORMANCE SUMMARY:")
        perf = validation_results.get('performance', {})
        for metric, value in perf.items():
            if isinstance(value, float):
                summary_lines.append(f"  {metric}: {value:.4f}")
            else:
                summary_lines.append(f"  {metric}: {value}")
        
        # Validation Results
        summary_lines.append("\nVALIDATION RESULTS:")
        val = validation_results.get('validation', {})
        for key, value in val.items():
            summary_lines.append(f"  {key}: {value}")
        
        # Recommendations
        summary_lines.append("\nRECOMMENDATIONS:")
        for rec in validation_results.get('recommendations', []):
            summary_lines.append(f"  • {rec}")
        
        summary_lines.append("\n" + "=" * 60)
        
        return "\n".join(summary_lines)

# Example usage
if __name__ == "__main__":
    reporter = PublicationReporter("Medical AI Validation Study")
    
    # Create sample data
    sample_metrics = {
        'Model A': {
            'Accuracy': 0.92,
            'Sensitivity': 0.88,
            'Specificity': 0.94,
            'AUC-ROC': 0.95
        },
        'Model B': {
            'Accuracy': 0.89,
            'Sensitivity': 0.85,
            'Specificity': 0.91,
            'AUC-ROC': 0.92
        }
    }
    
    # Create performance table
    perf_table = reporter.create_performance_table(sample_metrics)
    print("Performance Table:")
    print(perf_table)
    print()
    
    # Export to LaTeX
    latex_table = reporter.export_table_to_latex(
        perf_table,
        caption="Model Performance Comparison",
        label="tab:performance"
    )
    print("LaTeX Export:")
    print(latex_table)
    print()
    
    # Create summary report
    study_data = {
        'overview': {
            'Total Patients': 1000,
            'Study Duration': '12 months',
            'Sites': 5
        },
        'demographics': {
            'Mean Age': '55 ± 12 years',
            'Female': '45%',
            'Disease Prevalence': '30%'
        },
        'primary_results': {
            'Sensitivity': 0.92,
            'Specificity': 0.94,
            'AUC-ROC': 0.95
        },
        'conclusions': [
            'Model demonstrates high diagnostic accuracy',
            'Performance consistent across subgroups',
            'Ready for clinical deployment'
        ]
    }
    
    report = reporter.create_summary_report(study_data)
    print(report)