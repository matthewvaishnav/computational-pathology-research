"""
Tests for publication table generation.
"""

import shutil
import tempfile
from pathlib import Path

import pytest

from src.clinical.publication_tables import PublicationTableGenerator


class TestPublicationTableGenerator:
    """Test publication table generation."""

    @pytest.fixture
    def temp_output_dir(self):
        """Create temporary output directory."""
        temp_dir = tempfile.mkdtemp()
        yield Path(temp_dir)
        shutil.rmtree(temp_dir)

    @pytest.fixture
    def generator(self, temp_output_dir):
        """Create table generator with temp directory."""
        return PublicationTableGenerator(output_dir=temp_output_dir)

    @pytest.fixture
    def sample_metrics(self):
        """Sample metrics data."""
        return {
            "breast_cancer": {
                "accuracy": {"value": 0.950, "ci_lower": 0.940, "ci_upper": 0.960},
                "auc": {"value": 0.970, "ci_lower": 0.960, "ci_upper": 0.980},
                "f1": {"value": 0.945, "ci_lower": 0.935, "ci_upper": 0.955},
            },
            "lung_cancer": {
                "accuracy": {"value": 0.920, "ci_lower": 0.910, "ci_upper": 0.930},
                "auc": {"value": 0.940, "ci_lower": 0.930, "ci_upper": 0.950},
                "f1": {"value": 0.915, "ci_lower": 0.905, "ci_upper": 0.925},
            },
        }

    def test_generator_initialization(self, temp_output_dir):
        """Test generator creates output directory."""
        generator = PublicationTableGenerator(output_dir=temp_output_dir)
        assert generator.output_dir.exists()
        assert generator.output_dir.is_dir()

    def test_performance_table_latex(self, generator, sample_metrics, temp_output_dir):
        """Test LaTeX performance table generation."""
        output_files = generator.generate_performance_table(
            sample_metrics, output_name="test_performance", format="latex"
        )

        assert "latex" in output_files
        latex_file = output_files["latex"]
        assert latex_file.exists()
        assert latex_file.suffix == ".tex"

        # Check content
        content = latex_file.read_text()
        assert "\\begin{table}" in content
        assert "\\caption" in content
        assert "\\label{tab:test_performance}" in content
        assert "0.950 [0.940, 0.960]" in content  # Check CI format
        assert "Breast Cancer" in content

    def test_performance_table_markdown(self, generator, sample_metrics, temp_output_dir):
        """Test markdown performance table generation."""
        output_files = generator.generate_performance_table(
            sample_metrics, output_name="test_performance", format="markdown"
        )

        assert "markdown" in output_files
        md_file = output_files["markdown"]
        assert md_file.exists()
        assert md_file.suffix == ".md"

        # Check content
        content = md_file.read_text()
        assert "|" in content  # Markdown table format
        assert "0.950 [0.940, 0.960]" in content

    def test_performance_table_both_formats(self, generator, sample_metrics, temp_output_dir):
        """Test generating both formats."""
        output_files = generator.generate_performance_table(
            sample_metrics, output_name="test_performance", format="both"
        )

        assert "latex" in output_files
        assert "markdown" in output_files
        assert output_files["latex"].exists()
        assert output_files["markdown"].exists()

    def test_multisite_validation_table(self, generator, temp_output_dir):
        """Test multi-site validation table."""
        site_results = {
            "Site A": {"accuracy": 0.950, "auc": 0.970, "f1": 0.945, "n_samples": 1000},
            "Site B": {"accuracy": 0.930, "auc": 0.950, "f1": 0.925, "n_samples": 800},
            "Site C": {"accuracy": 0.940, "auc": 0.960, "f1": 0.935, "n_samples": 1200},
        }

        output_files = generator.generate_multisite_validation_table(
            site_results, output_name="test_multisite", format="both"
        )

        assert "latex" in output_files
        assert "markdown" in output_files

        # Check LaTeX content
        latex_content = output_files["latex"].read_text()
        assert "Site A" in latex_content
        assert "1000" in latex_content  # Sample count
        assert "0.950" in latex_content

    def test_comparison_table(self, generator, temp_output_dir):
        """Test model comparison table."""
        model_results = {
            "Our Model": {"accuracy": 0.950, "auc": 0.970, "f1": 0.945},
            "ResNet-50": {"accuracy": 0.850, "auc": 0.880, "f1": 0.830},
            "SOTA (Ref)": {"accuracy": 0.920, "auc": 0.940, "f1": 0.910},
        }

        output_files = generator.generate_comparison_table(
            model_results, output_name="test_comparison", format="both", highlight_best=True
        )

        assert "latex" in output_files
        assert "markdown" in output_files

        # Check LaTeX content for bolding
        latex_content = output_files["latex"].read_text()
        assert "\\textbf{0.95}" in latex_content  # Best accuracy should be bolded

    def test_comparison_table_no_highlight(self, generator, temp_output_dir):
        """Test comparison table without highlighting."""
        model_results = {
            "Model A": {"accuracy": 0.950, "auc": 0.970},
            "Model B": {"accuracy": 0.920, "auc": 0.940},
        }

        output_files = generator.generate_comparison_table(
            model_results,
            output_name="test_comparison_no_bold",
            format="latex",
            highlight_best=False,
        )

        latex_content = output_files["latex"].read_text()
        assert "\\textbf" not in latex_content  # No bolding

    def test_regulatory_summary_table(self, generator, temp_output_dir):
        """Test regulatory summary table."""
        regulatory_data = {
            "Device Classification": "Class II",
            "Intended Use": "Diagnostic aid for breast cancer detection",
            "Clinical Validation Sites": 5,
            "Total Samples": 10000,
            "Primary Endpoint": "Sensitivity",
            "Primary Endpoint Value": 0.950,
            "FDA Pathway": "510(k)",
        }

        output_files = generator.generate_regulatory_summary_table(
            regulatory_data, output_name="test_regulatory", format="both"
        )

        assert "latex" in output_files
        assert "markdown" in output_files

        # Check content
        latex_content = output_files["latex"].read_text()
        assert "Class II" in latex_content
        assert "510(k)" in latex_content
        assert "0.950" in latex_content

    def test_latex_table_structure(self, generator, sample_metrics):
        """Test LaTeX table has proper structure."""
        output_files = generator.generate_performance_table(
            sample_metrics, output_name="test_structure", format="latex"
        )

        content = output_files["latex"].read_text()

        # Check required LaTeX elements
        assert "\\begin{table}[htbp]" in content
        assert "\\centering" in content
        assert "\\begin{tabular}" in content
        assert "\\toprule" in content
        assert "\\midrule" in content
        assert "\\bottomrule" in content
        assert "\\end{tabular}" in content
        assert "\\end{table}" in content

    def test_empty_metrics(self, generator):
        """Test handling of empty metrics."""
        empty_metrics = {}

        output_files = generator.generate_performance_table(
            empty_metrics, output_name="test_empty", format="both"
        )

        # Should still create files
        assert "latex" in output_files
        assert "markdown" in output_files
        assert output_files["latex"].exists()
        assert output_files["markdown"].exists()
