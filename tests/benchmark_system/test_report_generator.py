"""
Unit tests for ReportGenerator component.

Tests cover:
- Comparison table generation
- Statistical significance computation
- Visualization creation
- PERFORMANCE_COMPARISON.md update
- QA flags inclusion
- CSV/JSON export

Requirements: 7.1-7.10, 10.7
"""

import json
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for tests

import numpy as np
import pandas as pd
import pytest

from experiments.benchmark_system.models import (
    BenchmarkConfig,
    SignificanceTest,
    TaskSpecification,
    TrainingResult,
)
from experiments.benchmark_system.report_generator import ReportGenerator
from experiments.benchmark_system.result_validator import ResultValidator

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def sample_task_spec():
    """Create sample task specification."""
    return TaskSpecification(
        dataset_name="PatchCamelyon",
        data_root=Path("data/pcam"),
        model_architecture="resnet18_transformer",
        num_epochs=10,
        batch_size=32,
        learning_rate=1e-4,
        random_seed=42,
    )


@pytest.fixture
def sample_training_results(sample_task_spec):
    """Create sample training results for multiple frameworks."""
    results = []

    # HistoCore result
    results.append(
        TrainingResult(
            framework_name="HistoCore",
            task_spec=sample_task_spec,
            training_time_seconds=3600.0,
            epochs_completed=10,
            final_train_loss=0.15,
            final_val_loss=0.18,
            test_accuracy=0.8526,
            test_auc=0.9537,
            test_f1=0.8234,
            test_precision=0.8456,
            test_recall=0.8023,
            accuracy_ci=(0.8450, 0.8602),
            auc_ci=(0.9500, 0.9574),
            f1_ci=(0.8150, 0.8318),
            peak_gpu_memory_mb=8500.0,
            avg_gpu_utilization=85.0,
            peak_gpu_temperature=75.0,
            samples_per_second=450.0,
            inference_time_ms=2.2,
            model_parameters=25000000,
            checkpoint_path=Path("checkpoints/histocore.pth"),
            metrics_path=Path("metrics/histocore.json"),
            log_path=Path("logs/histocore.log"),
            status="success",
        )
    )

    # PathML result
    results.append(
        TrainingResult(
            framework_name="PathML",
            task_spec=sample_task_spec,
            training_time_seconds=4200.0,
            epochs_completed=10,
            final_train_loss=0.18,
            final_val_loss=0.21,
            test_accuracy=0.8400,
            test_auc=0.9400,
            test_f1=0.8100,
            test_precision=0.8300,
            test_recall=0.7900,
            accuracy_ci=(0.8320, 0.8480),
            auc_ci=(0.9360, 0.9440),
            f1_ci=(0.8020, 0.8180),
            peak_gpu_memory_mb=9200.0,
            avg_gpu_utilization=80.0,
            peak_gpu_temperature=78.0,
            samples_per_second=380.0,
            inference_time_ms=2.6,
            model_parameters=28000000,
            checkpoint_path=Path("checkpoints/pathml.pth"),
            metrics_path=Path("metrics/pathml.json"),
            log_path=Path("logs/pathml.log"),
            status="success",
        )
    )

    # CLAM result
    results.append(
        TrainingResult(
            framework_name="CLAM",
            task_spec=sample_task_spec,
            training_time_seconds=3900.0,
            epochs_completed=10,
            final_train_loss=0.17,
            final_val_loss=0.20,
            test_accuracy=0.8350,
            test_auc=0.9350,
            test_f1=0.8050,
            test_precision=0.8250,
            test_recall=0.7850,
            accuracy_ci=(0.8270, 0.8430),
            auc_ci=(0.9310, 0.9390),
            f1_ci=(0.7970, 0.8130),
            peak_gpu_memory_mb=8800.0,
            avg_gpu_utilization=82.0,
            peak_gpu_temperature=76.0,
            samples_per_second=410.0,
            inference_time_ms=2.4,
            model_parameters=26000000,
            checkpoint_path=Path("checkpoints/clam.pth"),
            metrics_path=Path("metrics/clam.json"),
            log_path=Path("logs/clam.log"),
            status="success",
        )
    )

    return results


@pytest.fixture
def report_generator():
    """Create ReportGenerator instance."""
    return ReportGenerator()


# ============================================================================
# COMPARISON TABLE TESTS
# ============================================================================


def test_generate_comparison_table_basic(report_generator, sample_training_results):
    """Test basic comparison table generation."""
    df = report_generator.generate_comparison_table(sample_training_results)

    # Verify DataFrame structure
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3  # Three frameworks
    assert "Framework" in df.columns
    assert "Accuracy" in df.columns
    assert "AUC" in df.columns
    assert "Training Time (s)" in df.columns

    # Verify data
    assert "HistoCore" in df["Framework"].values
    assert "PathML" in df["Framework"].values
    assert "CLAM" in df["Framework"].values

    # Verify sorted by accuracy (descending)
    assert df.iloc[0]["Framework"] == "HistoCore"  # Highest accuracy


def test_generate_comparison_table_empty_results(report_generator):
    """Test comparison table with empty results raises error."""
    with pytest.raises(ValueError, match="empty results"):
        report_generator.generate_comparison_table([])


def test_generate_comparison_table_includes_confidence_intervals(
    report_generator, sample_training_results
):
    """Test comparison table includes confidence intervals."""
    df = report_generator.generate_comparison_table(sample_training_results)

    assert "Accuracy CI Lower" in df.columns
    assert "Accuracy CI Upper" in df.columns
    assert "AUC CI Lower" in df.columns
    assert "AUC CI Upper" in df.columns

    # Verify CI values
    histocore_row = df[df["Framework"] == "HistoCore"].iloc[0]
    assert histocore_row["Accuracy CI Lower"] == 0.8450
    assert histocore_row["Accuracy CI Upper"] == 0.8602


# ============================================================================
# STATISTICAL SIGNIFICANCE TESTS
# ============================================================================


def test_compute_statistical_significance_accuracy(report_generator, sample_training_results):
    """Test statistical significance computation for accuracy."""
    histocore = sample_training_results[0]
    pathml = sample_training_results[1]

    test = report_generator.compute_statistical_significance(
        histocore, pathml, metric_name="accuracy"
    )

    # Verify SignificanceTest structure
    assert isinstance(test, SignificanceTest)
    assert test.histocore_metric == 0.8526
    assert test.competitor_metric == 0.8400
    assert test.competitor_name == "PathML"
    assert test.metric_name == "accuracy"

    # Verify improvement calculation
    assert test.improvement == pytest.approx(0.0126, abs=0.0001)
    assert test.improvement_pct > 0

    # Verify statistical measures
    assert isinstance(test.cohens_d, float)
    assert isinstance(test.p_value, float)
    assert isinstance(test.ci_overlap, bool)
    assert test.significance_level in [
        "Large Effect",
        "Medium Effect",
        "Small Effect",
        "No Effect",
    ]


def test_compute_statistical_significance_auc(report_generator, sample_training_results):
    """Test statistical significance computation for AUC."""
    histocore = sample_training_results[0]
    pathml = sample_training_results[1]

    test = report_generator.compute_statistical_significance(histocore, pathml, metric_name="auc")

    assert test.metric_name == "auc"
    assert test.histocore_metric == 0.9537
    assert test.competitor_metric == 0.9400


def test_compute_statistical_significance_f1(report_generator, sample_training_results):
    """Test statistical significance computation for F1."""
    histocore = sample_training_results[0]
    pathml = sample_training_results[1]

    test = report_generator.compute_statistical_significance(histocore, pathml, metric_name="f1")

    assert test.metric_name == "f1"
    assert test.histocore_metric == 0.8234
    assert test.competitor_metric == 0.8100


def test_compute_statistical_significance_invalid_metric(report_generator, sample_training_results):
    """Test statistical significance with invalid metric raises error."""
    histocore = sample_training_results[0]
    pathml = sample_training_results[1]

    with pytest.raises(ValueError, match="Unknown metric"):
        report_generator.compute_statistical_significance(
            histocore, pathml, metric_name="invalid_metric"
        )


def test_compute_statistical_significance_effect_size_interpretation(
    report_generator, sample_training_results
):
    """Test effect size interpretation (Cohen's d)."""
    histocore = sample_training_results[0]
    pathml = sample_training_results[1]

    test = report_generator.compute_statistical_significance(
        histocore, pathml, metric_name="accuracy"
    )

    # Verify effect size interpretation
    abs_d = abs(test.cohens_d)
    if abs_d >= 0.8:
        assert test.significance_level == "Large Effect"
    elif abs_d >= 0.5:
        assert test.significance_level == "Medium Effect"
    elif abs_d >= 0.2:
        assert test.significance_level == "Small Effect"
    else:
        assert test.significance_level == "No Effect"


# ============================================================================
# VISUALIZATION TESTS
# ============================================================================


def test_generate_visualizations_basic(report_generator, sample_training_results, tmp_path):
    """Test basic visualization generation."""
    output_dir = tmp_path / "visualizations"

    generated_files = report_generator.generate_visualizations(sample_training_results, output_dir)

    # Verify files were generated
    assert len(generated_files) > 0
    assert all(isinstance(f, Path) for f in generated_files)
    assert all(f.exists() for f in generated_files)

    # Verify expected plots
    file_names = [f.name for f in generated_files]
    assert any("accuracy_vs_parameters" in name for name in file_names)
    assert any("accuracy_vs_time" in name for name in file_names)
    assert any("memory_comparison" in name for name in file_names)
    assert any("throughput_comparison" in name for name in file_names)


def test_generate_visualizations_empty_results(report_generator, tmp_path):
    """Test visualization generation with empty results raises error."""
    with pytest.raises(ValueError, match="empty results"):
        report_generator.generate_visualizations([], tmp_path)


def test_generate_visualizations_creates_output_dir(
    report_generator, sample_training_results, tmp_path
):
    """Test visualization generation creates output directory."""
    output_dir = tmp_path / "nested" / "viz"

    report_generator.generate_visualizations(sample_training_results, output_dir)

    assert output_dir.exists()


def test_generate_visualizations_file_format(report_generator, sample_training_results, tmp_path):
    """Test visualization files use correct format."""
    output_dir = tmp_path / "visualizations"

    generated_files = report_generator.generate_visualizations(sample_training_results, output_dir)

    # Verify file format
    for file_path in generated_files:
        assert file_path.suffix == f".{report_generator.figure_format}"


# ============================================================================
# PERFORMANCE_COMPARISON.MD UPDATE TESTS
# ============================================================================


def test_update_performance_comparison_md_basic(
    report_generator, sample_training_results, tmp_path
):
    """Test basic PERFORMANCE_COMPARISON.md update."""
    output_path = tmp_path / "PERFORMANCE_COMPARISON.md"

    report_generator.update_performance_comparison_md(sample_training_results, output_path)

    # Verify file created
    assert output_path.exists()

    content = output_path.read_text(encoding="utf-8")

    # Verify sections
    assert "# Performance Comparison: HistoCore vs Competitors" in content
    assert "## Performance Summary" in content
    assert "## Statistical Significance" in content
    assert "## Detailed Metrics" in content

    # Verify data
    assert "HistoCore" in content
    assert "PathML" in content
    assert "CLAM" in content


def test_update_performance_comparison_md_with_reproducibility(
    report_generator, sample_training_results, tmp_path
):
    """Test PERFORMANCE_COMPARISON.md includes reproducibility section."""
    output_path = tmp_path / "PERFORMANCE_COMPARISON.md"

    report_generator.update_performance_comparison_md(
        sample_training_results, output_path, include_reproducibility=True
    )

    content = output_path.read_text(encoding="utf-8")

    # Verify reproducibility section
    assert "## Reproducibility" in content
    assert "### Environment Details" in content
    assert "### System Information" in content
    assert "### GPU Information" in content


def test_update_performance_comparison_md_with_qa_flags(
    report_generator, sample_training_results, tmp_path
):
    """Test PERFORMANCE_COMPARISON.md includes QA flags."""
    output_path = tmp_path / "PERFORMANCE_COMPARISON.md"

    # Create a result with issues to trigger QA flags
    problematic_result = TrainingResult(
        framework_name="ProblematicFramework",
        task_spec=sample_training_results[0].task_spec,
        training_time_seconds=1000.0,
        epochs_completed=10,
        final_train_loss=0.5,
        final_val_loss=0.5,
        test_accuracy=0.45,  # Below random chance for binary classification
        test_auc=0.5,
        test_f1=0.4,
        test_precision=0.4,
        test_recall=0.4,
        accuracy_ci=(0.43, 0.47),
        auc_ci=(0.48, 0.52),
        f1_ci=(0.38, 0.42),
        peak_gpu_memory_mb=5000.0,
        avg_gpu_utilization=50.0,
        peak_gpu_temperature=60.0,
        samples_per_second=200.0,
        inference_time_ms=5.0,
        model_parameters=10000000,
        checkpoint_path=Path("checkpoints/problematic.pth"),
        metrics_path=Path("metrics/problematic.json"),
        log_path=Path("logs/problematic.log"),
        status="success",
    )

    results_with_issues = sample_training_results + [problematic_result]

    report_generator.update_performance_comparison_md(
        results_with_issues, output_path, include_qa_flags=True
    )

    content = output_path.read_text(encoding="utf-8")

    # Verify QA flags section exists
    assert "## Quality Assurance Flags" in content


def test_update_performance_comparison_md_empty_results(report_generator, tmp_path):
    """Test PERFORMANCE_COMPARISON.md update with empty results raises error."""
    output_path = tmp_path / "PERFORMANCE_COMPARISON.md"

    with pytest.raises(ValueError, match="empty results"):
        report_generator.update_performance_comparison_md([], output_path)


def test_update_performance_comparison_md_creates_parent_dirs(
    report_generator, sample_training_results, tmp_path
):
    """Test PERFORMANCE_COMPARISON.md update creates parent directories."""
    output_path = tmp_path / "nested" / "dir" / "PERFORMANCE_COMPARISON.md"

    report_generator.update_performance_comparison_md(sample_training_results, output_path)

    assert output_path.exists()
    assert output_path.parent.exists()


def test_update_performance_comparison_md_statistical_significance(
    report_generator, sample_training_results, tmp_path
):
    """Test PERFORMANCE_COMPARISON.md includes statistical significance tests."""
    output_path = tmp_path / "PERFORMANCE_COMPARISON.md"

    report_generator.update_performance_comparison_md(sample_training_results, output_path)

    content = output_path.read_text(encoding="utf-8")

    # Verify statistical significance section
    assert "## Statistical Significance" in content
    assert "Cohen's d" in content
    assert "p-value" in content
    assert "Statistically Significant" in content


# ============================================================================
# CSV/JSON EXPORT TESTS
# ============================================================================


def test_export_to_csv(report_generator, sample_training_results, tmp_path):
    """Test CSV export."""
    output_path = tmp_path / "comparison.csv"

    report_generator.export_to_csv(sample_training_results, output_path)

    # Verify file created
    assert output_path.exists()

    # Verify CSV format
    df = pd.read_csv(output_path)
    assert len(df) == 3
    assert "Framework" in df.columns
    assert "Accuracy" in df.columns


def test_export_to_json(report_generator, sample_training_results, tmp_path):
    """Test JSON export."""
    output_path = tmp_path / "comparison.json"

    report_generator.export_to_json(sample_training_results, output_path)

    # Verify file created
    assert output_path.exists()

    # Verify JSON format
    with open(output_path) as f:
        data = json.load(f)

    assert isinstance(data, list)
    assert len(data) == 3
    assert data[0]["Framework"] == "HistoCore"


def test_export_to_csv_creates_parent_dirs(report_generator, sample_training_results, tmp_path):
    """Test CSV export creates parent directories."""
    output_path = tmp_path / "nested" / "comparison.csv"

    report_generator.export_to_csv(sample_training_results, output_path)

    assert output_path.exists()


def test_export_to_json_creates_parent_dirs(report_generator, sample_training_results, tmp_path):
    """Test JSON export creates parent directories."""
    output_path = tmp_path / "nested" / "comparison.json"

    report_generator.export_to_json(sample_training_results, output_path)

    assert output_path.exists()


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


def test_full_report_generation_workflow(report_generator, sample_training_results, tmp_path):
    """Test complete report generation workflow."""
    # Generate comparison table
    df = report_generator.generate_comparison_table(sample_training_results)
    assert len(df) == 3

    # Compute statistical significance
    histocore = sample_training_results[0]
    pathml = sample_training_results[1]
    test = report_generator.compute_statistical_significance(
        histocore, pathml, metric_name="accuracy"
    )
    assert test.statistically_significant is not None

    # Generate visualizations
    viz_dir = tmp_path / "visualizations"
    viz_files = report_generator.generate_visualizations(sample_training_results, viz_dir)
    assert len(viz_files) > 0

    # Update PERFORMANCE_COMPARISON.md
    md_path = tmp_path / "PERFORMANCE_COMPARISON.md"
    report_generator.update_performance_comparison_md(sample_training_results, md_path)
    assert md_path.exists()

    # Export to CSV and JSON
    csv_path = tmp_path / "comparison.csv"
    json_path = tmp_path / "comparison.json"
    report_generator.export_to_csv(sample_training_results, csv_path)
    report_generator.export_to_json(sample_training_results, json_path)
    assert csv_path.exists()
    assert json_path.exists()
