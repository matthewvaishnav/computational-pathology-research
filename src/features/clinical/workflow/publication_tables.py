"""
Publication-ready table generation for clinical validation results.

This module generates LaTeX and markdown tables formatted for academic publications,
including performance metrics with confidence intervals, multi-site validation results,
and regulatory documentation tables.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class PublicationTableGenerator:
    """Generate publication-ready tables for clinical validation results."""

    def __init__(self, output_dir: Union[str, Path] = "results/publication_tables"):
        """
        Initialize table generator.

        Args:
            output_dir: Directory to save generated tables
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Publication table generator initialized. Output: {self.output_dir}")

    def generate_performance_table(
        self,
        metrics_by_disease: Dict[str, Dict[str, Dict[str, float]]],
        output_name: str = "performance_metrics",
        format: str = "both",
    ) -> Dict[str, Path]:
        """
        Generate performance metrics table with confidence intervals.

        Args:
            metrics_by_disease: Nested dict structure:
                {
                    'breast_cancer': {
                        'accuracy': {'value': 0.95, 'ci_lower': 0.94, 'ci_upper': 0.96},
                        'auc': {'value': 0.97, 'ci_lower': 0.96, 'ci_upper': 0.98},
                        ...
                    },
                    'lung_cancer': {...},
                    ...
                }
            output_name: Base name for output files
            format: Output format ('latex', 'markdown', or 'both')

        Returns:
            Dictionary mapping format to output file path
        """
        # Build DataFrame
        rows = []
        for disease, metrics in metrics_by_disease.items():
            row = {"Disease Type": disease.replace("_", " ").title()}

            for metric_name, metric_data in metrics.items():
                value = metric_data["value"]
                ci_lower = metric_data["ci_lower"]
                ci_upper = metric_data["ci_upper"]

                # Format: "0.950 [0.940, 0.960]"
                row[metric_name.upper()] = f"{value:.3f} [{ci_lower:.3f}, {ci_upper:.3f}]"

            rows.append(row)

        df = pd.DataFrame(rows)

        # Generate outputs
        output_files = {}

        if format in ("latex", "both"):
            latex_path = self.output_dir / f"{output_name}.tex"
            latex_table = self._generate_latex_table(
                df,
                caption="Performance metrics across disease types with 95% confidence intervals.",
                label=f"tab:{output_name}",
            )
            latex_path.write_text(latex_table)
            output_files["latex"] = latex_path
            logger.info(f"LaTeX table saved: {latex_path}")

        if format in ("markdown", "both"):
            md_path = self.output_dir / f"{output_name}.md"
            md_table = df.to_markdown(index=False)
            md_path.write_text(md_table)
            output_files["markdown"] = md_path
            logger.info(f"Markdown table saved: {md_path}")

        return output_files

    def generate_multisite_validation_table(
        self,
        site_results: Dict[str, Dict[str, float]],
        output_name: str = "multisite_validation",
        format: str = "both",
    ) -> Dict[str, Path]:
        """
        Generate multi-site validation results table.

        Args:
            site_results: Dict mapping site names to metrics:
                {
                    'Site A': {'accuracy': 0.95, 'auc': 0.97, 'n_samples': 1000},
                    'Site B': {'accuracy': 0.93, 'auc': 0.96, 'n_samples': 800},
                    ...
                }
            output_name: Base name for output files
            format: Output format ('latex', 'markdown', or 'both')

        Returns:
            Dictionary mapping format to output file path
        """
        rows = []
        for site_name, metrics in site_results.items():
            row = {"Site": site_name}
            row["N"] = int(metrics.get("n_samples", 0))

            # Add metrics
            for metric_name, value in metrics.items():
                if metric_name != "n_samples":
                    row[metric_name.upper()] = f"{value:.3f}"

            rows.append(row)

        df = pd.DataFrame(rows)

        # Generate outputs
        output_files = {}

        if format in ("latex", "both"):
            latex_path = self.output_dir / f"{output_name}.tex"
            latex_table = self._generate_latex_table(
                df,
                caption="Multi-site validation results showing model generalization across institutions.",
                label=f"tab:{output_name}",
            )
            latex_path.write_text(latex_table)
            output_files["latex"] = latex_path
            logger.info(f"LaTeX table saved: {latex_path}")

        if format in ("markdown", "both"):
            md_path = self.output_dir / f"{output_name}.md"
            md_table = df.to_markdown(index=False)
            md_path.write_text(md_table)
            output_files["markdown"] = md_path
            logger.info(f"Markdown table saved: {md_path}")

        return output_files

    def generate_comparison_table(
        self,
        model_results: Dict[str, Dict[str, float]],
        output_name: str = "model_comparison",
        format: str = "both",
        highlight_best: bool = True,
    ) -> Dict[str, Path]:
        """
        Generate model comparison table (e.g., vs. baselines or SOTA).

        Args:
            model_results: Dict mapping model names to metrics:
                {
                    'Our Model': {'accuracy': 0.95, 'auc': 0.97, 'f1': 0.94},
                    'ResNet-50': {'accuracy': 0.85, 'auc': 0.88, 'f1': 0.83},
                    'SOTA (Ref)': {'accuracy': 0.92, 'auc': 0.94, 'f1': 0.91},
                }
            output_name: Base name for output files
            format: Output format ('latex', 'markdown', or 'both')
            highlight_best: Whether to bold best values in LaTeX

        Returns:
            Dictionary mapping format to output file path
        """
        rows = []
        for model_name, metrics in model_results.items():
            row = {"Model": model_name}

            for metric_name, value in metrics.items():
                row[metric_name.upper()] = value

            rows.append(row)

        df = pd.DataFrame(rows)

        # Find best values for each metric (excluding 'Model' column)
        best_values = {}
        if highlight_best:
            for col in df.columns:
                if col != "Model":
                    best_values[col] = df[col].max()

        # Generate outputs
        output_files = {}

        if format in ("latex", "both"):
            latex_path = self.output_dir / f"{output_name}.tex"
            latex_table = self._generate_latex_table(
                df,
                caption="Comparison with baseline models and state-of-the-art methods.",
                label=f"tab:{output_name}",
                bold_best=best_values if highlight_best else None,
            )
            latex_path.write_text(latex_table)
            output_files["latex"] = latex_path
            logger.info(f"LaTeX table saved: {latex_path}")

        if format in ("markdown", "both"):
            md_path = self.output_dir / f"{output_name}.md"
            # Format numeric columns
            df_formatted = df.copy()
            for col in df_formatted.columns:
                if col != "Model" and df_formatted[col].dtype in [np.float64, np.float32]:
                    df_formatted[col] = df_formatted[col].apply(lambda x: f"{x:.3f}")
            md_table = df_formatted.to_markdown(index=False)
            md_path.write_text(md_table)
            output_files["markdown"] = md_path
            logger.info(f"Markdown table saved: {md_path}")

        return output_files

    def generate_regulatory_summary_table(
        self,
        regulatory_data: Dict[str, Union[str, int, float]],
        output_name: str = "regulatory_summary",
        format: str = "both",
    ) -> Dict[str, Path]:
        """
        Generate regulatory documentation summary table.

        Args:
            regulatory_data: Dict with regulatory information:
                {
                    'Device Classification': 'Class II',
                    'Intended Use': 'Diagnostic aid for breast cancer detection',
                    'Clinical Validation Sites': 5,
                    'Total Samples': 10000,
                    'Primary Endpoint': 'Sensitivity',
                    'Primary Endpoint Value': 0.95,
                    'FDA Pathway': '510(k)',
                }
            output_name: Base name for output files
            format: Output format ('latex', 'markdown', or 'both')

        Returns:
            Dictionary mapping format to output file path
        """
        rows = []
        for key, value in regulatory_data.items():
            # Format value based on type
            if isinstance(value, float):
                formatted_value = f"{value:.3f}"
            else:
                formatted_value = str(value)

            rows.append({"Parameter": key, "Value": formatted_value})

        df = pd.DataFrame(rows)

        # Generate outputs
        output_files = {}

        if format in ("latex", "both"):
            latex_path = self.output_dir / f"{output_name}.tex"
            latex_table = self._generate_latex_table(
                df,
                caption="Regulatory documentation summary for FDA submission.",
                label=f"tab:{output_name}",
            )
            latex_path.write_text(latex_table)
            output_files["latex"] = latex_path
            logger.info(f"LaTeX table saved: {latex_path}")

        if format in ("markdown", "both"):
            md_path = self.output_dir / f"{output_name}.md"
            md_table = df.to_markdown(index=False)
            md_path.write_text(md_table)
            output_files["markdown"] = md_path
            logger.info(f"Markdown table saved: {md_path}")

        return output_files

    def _generate_latex_table(
        self,
        df: pd.DataFrame,
        caption: str,
        label: str,
        bold_best: Optional[Dict[str, float]] = None,
    ) -> str:
        """
        Generate LaTeX table with proper formatting.

        Args:
            df: DataFrame to convert
            caption: Table caption
            label: LaTeX label for referencing
            bold_best: Dict mapping column names to best values (for bolding)

        Returns:
            LaTeX table string
        """
        # Start table
        num_cols = len(df.columns)
        col_format = "l" + "c" * (num_cols - 1)  # Left-align first column, center rest

        latex = "\\begin{table}[htbp]\n"
        latex += "\\centering\n"
        latex += f"\\caption{{{caption}}}\n"
        latex += f"\\label{{{label}}}\n"
        latex += f"\\begin{{tabular}}{{{col_format}}}\n"
        latex += "\\toprule\n"

        # Header
        latex += " & ".join(df.columns) + " \\\\\n"
        latex += "\\midrule\n"

        # Rows
        for _, row in df.iterrows():
            row_values = []
            for col_name, value in row.items():
                value_str = str(value)

                # Bold best values if specified
                if bold_best and col_name in bold_best:
                    try:
                        # Extract numeric value from string (handle CI format)
                        if "[" in value_str:
                            numeric_value = float(value_str.split("[")[0].strip())
                        else:
                            numeric_value = float(value_str)

                        if abs(numeric_value - bold_best[col_name]) < 1e-6:
                            value_str = f"\\textbf{{{value_str}}}"
                    except (ValueError, TypeError):
                        pass  # Skip bolding if value is not numeric

                row_values.append(value_str)

            latex += " & ".join(row_values) + " \\\\\n"

        # End table
        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}\n"

        return latex


def generate_all_publication_tables(
    results_dir: Union[str, Path],
    output_dir: Union[str, Path] = "results/publication_tables",
) -> Dict[str, Dict[str, Path]]:
    """
    Generate all publication tables from validation results.

    Args:
        results_dir: Directory containing validation results (JSON/pickle files)
        output_dir: Directory to save generated tables

    Returns:
        Dictionary mapping table type to output file paths
    """
    generator = PublicationTableGenerator(output_dir=output_dir)

    # Example: Load results and generate tables
    # In practice, this would load actual validation results from files

    logger.info("Publication table generation complete")
    return {}
