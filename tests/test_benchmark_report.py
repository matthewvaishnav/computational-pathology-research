"""
Comprehensive tests for benchmark_report.py

Tests cover:
- Main report generation function
- Executive summary generation
- Dataset description
- Model architecture
- Training configuration
- Test results (with/without CIs)
- Baseline comparison
- Hardware info
- Reproduction commands
- File I/O and encoding
"""

from pathlib import Path

import pytest

from src.utils.benchmark_report import (
    _add_baseline_comparison,
    _add_dataset_description,
    _add_executive_summary,
    _add_hardware_info,
    _add_model_architecture,
    _add_reproduction_commands,
    _add_test_results,
    _add_training_configuration,
    generate_benchmark_report,
)


# ============================================================================
# EXECUTIVE SUMMARY TESTS
# ============================================================================


def test_executive_summary_with_ci():
    """Test executive summary with confidence intervals."""
    lines = []
    test_metrics = {
        "accuracy": 0.8526,
        "accuracy_ci_lower": 0.8450,
        "accuracy_ci_upper": 0.8602,
        "auc": 0.9537,
        "auc_ci_lower": 0.9500,
        "auc_ci_upper": 0.9574,
    }
    dataset_info = {"train_samples": 262144, "test_samples": 32768}
    training_config = {"num_epochs": 10}
    
    _add_executive_summary(lines, test_metrics, dataset_info, training_config)
    
    content = "\n".join(lines)
    assert "85.26%" in content
    assert "0.9537" in content
    assert "95% CI" in content
    assert "262,144" in content
    assert "32,768" in content


def test_executive_summary_without_ci():
    """Test executive summary without confidence intervals."""
    lines = []
    test_metrics = {"accuracy": 0.8526, "auc": 0.9537}
    dataset_info = {"train_samples": 1000, "test_samples": 200}
    training_config = {"num_epochs": 5}
    
    _add_executive_summary(lines, test_metrics, dataset_info, training_config)
    
    content = "\n".join(lines)
    assert "85.26%" in content
    assert "0.9537" in content
    assert "95% CI" not in content


# ============================================================================
# DATASET DESCRIPTION TESTS
# ============================================================================


def test_dataset_description_basic():
    """Test dataset description generation."""
    lines = []
    dataset_info = {
        "name": "PatchCamelyon",
        "train_samples": 262144,
        "val_samples": 32768,
        "test_samples": 32768,
        "image_size": "96×96",
        "num_classes": 2,
        "task": "Metastatic tissue detection",
    }
    
    _add_dataset_description(lines, dataset_info)
    
    content = "\n".join(lines)
    assert "PatchCamelyon" in content
    assert "262,144" in content
    assert "32,768" in content
    assert "96×96" in content
    assert "binary" in content


def test_dataset_description_defaults():
    """Test dataset description with missing fields."""
    lines = []
    # Provide numeric defaults to avoid format error
    dataset_info = {
        "train_samples": 0,
        "val_samples": 0,
        "test_samples": 0,
    }
    
    _add_dataset_description(lines, dataset_info)
    
    content = "\n".join(lines)
    assert "PatchCamelyon" in content  # Default name
    assert "0" in content


# ============================================================================
# MODEL ARCHITECTURE TESTS
# ============================================================================


def test_model_architecture_basic():
    """Test model architecture generation."""
    lines = []
    model_info = {
        "feature_extractor": "ResNet50",
        "feature_dim": 2048,
        "encoder": "TransformerEncoder",
        "hidden_dim": 512,
        "total_params": 25000000,
        "pretrained": "Yes",
    }
    
    _add_model_architecture(lines, model_info)
    
    content = "\n".join(lines)
    assert "ResNet50" in content
    assert "2048" in content
    assert "TransformerEncoder" in content
    assert "512" in content
    assert "25,000,000" in content
    assert "Yes" in content


def test_model_architecture_defaults():
    """Test model architecture with missing fields."""
    lines = []
    # Provide numeric default for total_params
    model_info = {"total_params": 0}
    
    _add_model_architecture(lines, model_info)
    
    content = "\n".join(lines)
    assert "N/A" in content
    assert "0" in content


# ============================================================================
# TRAINING CONFIGURATION TESTS
# ============================================================================


def test_training_configuration_basic():
    """Test training configuration generation."""
    lines = []
    training_config = {
        "num_epochs": 10,
        "batch_size": 32,
        "learning_rate": 0.001,
        "weight_decay": 0.0001,
        "optimizer": "AdamW",
        "use_amp": True,
        "early_stopping_patience": 5,
    }
    
    _add_training_configuration(lines, training_config)
    
    content = "\n".join(lines)
    assert "num_epochs: 10" in content
    assert "batch_size: 32" in content
    assert "learning_rate: 0.001" in content
    assert "optimizer: AdamW" in content
    assert "use_amp: True" in content


def test_training_configuration_defaults():
    """Test training configuration with missing fields."""
    lines = []
    training_config = {}
    
    _add_training_configuration(lines, training_config)
    
    content = "\n".join(lines)
    assert "N/A" in content
    assert "AdamW" in content  # Default optimizer


# ============================================================================
# TEST RESULTS TESTS
# ============================================================================


def test_test_results_with_ci():
    """Test results with confidence intervals."""
    lines = []
    test_metrics = {
        "accuracy": 0.8526,
        "accuracy_ci_lower": 0.8450,
        "accuracy_ci_upper": 0.8602,
        "auc": 0.9537,
        "auc_ci_lower": 0.9500,
        "auc_ci_upper": 0.9574,
        "f1": 0.8234,
        "f1_ci_lower": 0.8150,
        "f1_ci_upper": 0.8318,
        "precision": 0.8456,
        "precision_ci_lower": 0.8370,
        "precision_ci_upper": 0.8542,
        "recall": 0.8023,
        "recall_ci_lower": 0.7940,
        "recall_ci_upper": 0.8106,
    }
    
    _add_test_results(lines, test_metrics)
    
    content = "\n".join(lines)
    assert "85.26%" in content
    assert "[84.50%, 86.02%]" in content
    assert "0.9537" in content
    assert "[0.9500, 0.9574]" in content


def test_test_results_without_ci():
    """Test results without confidence intervals."""
    lines = []
    test_metrics = {
        "accuracy": 0.8526,
        "auc": 0.9537,
        "f1": 0.8234,
        "precision": 0.8456,
        "recall": 0.8023,
    }
    
    _add_test_results(lines, test_metrics)
    
    content = "\n".join(lines)
    assert "85.26%" in content
    assert "0.9537" in content
    assert "[" not in content  # No CI brackets


def test_test_results_with_confusion_matrix():
    """Test results with confusion matrix."""
    lines = []
    test_metrics = {
        "accuracy": 0.85,
        "auc": 0.95,
        "f1": 0.82,
        "precision": 0.84,
        "recall": 0.80,
        "confusion_matrix": [[15000, 1500], [2000, 14268]],
    }
    
    _add_test_results(lines, test_metrics)
    
    content = "\n".join(lines)
    assert "Confusion Matrix" in content
    assert "15000" in content
    assert "14268" in content


# ============================================================================
# BASELINE COMPARISON TESTS
# ============================================================================


def test_baseline_comparison_with_ci():
    """Test baseline comparison with confidence intervals."""
    lines = []
    comparison_results = {
        "baselines": [
            {
                "model_name": "ResNet50",
                "test_metrics": {
                    "accuracy": 0.8400,
                    "accuracy_ci_lower": 0.8320,
                    "accuracy_ci_upper": 0.8480,
                    "auc": 0.9400,
                    "auc_ci_lower": 0.9360,
                    "auc_ci_upper": 0.9440,
                    "f1": 0.8100,
                    "f1_ci_lower": 0.8020,
                    "f1_ci_upper": 0.8180,
                },
                "model_parameters": 25000000,
            },
            {
                "model_name": "VGG16",
                "test_metrics": {
                    "accuracy": 0.8200,
                    "accuracy_ci_lower": 0.8120,
                    "accuracy_ci_upper": 0.8280,
                    "auc": 0.9200,
                    "auc_ci_lower": 0.9160,
                    "auc_ci_upper": 0.9240,
                    "f1": 0.7900,
                    "f1_ci_lower": 0.7820,
                    "f1_ci_upper": 0.7980,
                },
                "model_parameters": 138000000,
            },
        ]
    }
    
    _add_baseline_comparison(lines, comparison_results)
    
    content = "\n".join(lines)
    assert "ResNet50" in content
    assert "VGG16" in content
    assert "84.00%" in content
    assert "25,000,000" in content
    assert "138,000,000" in content


def test_baseline_comparison_without_ci():
    """Test baseline comparison without confidence intervals."""
    lines = []
    comparison_results = {
        "baselines": [
            {
                "model_name": "SimpleNet",
                "test_metrics": {"accuracy": 0.75, "auc": 0.85, "f1": 0.72},
                "model_parameters": 1000000,
            }
        ]
    }
    
    _add_baseline_comparison(lines, comparison_results)
    
    content = "\n".join(lines)
    assert "SimpleNet" in content
    assert "75.00%" in content
    assert "0.8500" in content
    assert "1,000,000" in content


def test_baseline_comparison_empty():
    """Test baseline comparison with no baselines."""
    lines = []
    comparison_results = {"baselines": []}
    
    _add_baseline_comparison(lines, comparison_results)
    
    content = "\n".join(lines)
    # Should have header but no data rows
    assert "Model" in content
    assert "Accuracy" in content


# ============================================================================
# HARDWARE INFO TESTS
# ============================================================================


def test_hardware_info_basic():
    """Test hardware info generation."""
    lines = []
    hardware_info = {
        "gpu": "NVIDIA RTX 4070",
        "gpu_memory": "12GB",
        "cpu": "AMD Ryzen 9 5900X",
        "ram": "32GB DDR4",
        "training_time": "2h 15m",
        "throughput": "450 samples/sec",
    }
    
    _add_hardware_info(lines, hardware_info)
    
    content = "\n".join(lines)
    assert "NVIDIA RTX 4070" in content
    assert "12GB" in content
    assert "AMD Ryzen 9 5900X" in content
    assert "32GB DDR4" in content
    assert "2h 15m" in content
    assert "450 samples/sec" in content


def test_hardware_info_defaults():
    """Test hardware info with missing fields."""
    lines = []
    hardware_info = {}
    
    _add_hardware_info(lines, hardware_info)
    
    content = "\n".join(lines)
    assert "N/A" in content


# ============================================================================
# REPRODUCTION COMMANDS TESTS
# ============================================================================


def test_reproduction_commands_basic():
    """Test reproduction commands generation."""
    lines = []
    training_config = {
        "config_file": "configs/test.yaml",
        "checkpoint_path": "checkpoints/model.pth",
    }
    dataset_info = {"data_root": "./data/test"}
    
    _add_reproduction_commands(lines, training_config, dataset_info)
    
    content = "\n".join(lines)
    assert "python scripts/download_pcam.py" in content
    assert "./data/test" in content
    assert "python experiments/train_pcam.py" in content
    assert "configs/test.yaml" in content
    assert "python experiments/evaluate_pcam.py" in content
    assert "checkpoints/model.pth" in content


def test_reproduction_commands_defaults():
    """Test reproduction commands with default paths."""
    lines = []
    training_config = {}
    dataset_info = {}
    
    _add_reproduction_commands(lines, training_config, dataset_info)
    
    content = "\n".join(lines)
    assert "python scripts/download_pcam.py" in content
    assert "python experiments/train_pcam.py" in content
    assert "python experiments/evaluate_pcam.py" in content


# ============================================================================
# FULL REPORT GENERATION TESTS
# ============================================================================


def test_generate_benchmark_report_basic(tmp_path):
    """Test full report generation with basic inputs."""
    output_path = tmp_path / "report.md"
    
    generate_benchmark_report(
        experiment_name="Test Experiment",
        dataset_info={"train_samples": 1000, "val_samples": 200, "test_samples": 200},
        model_info={"feature_extractor": "ResNet50", "total_params": 25000000},
        training_config={"num_epochs": 10, "batch_size": 32},
        test_metrics={"accuracy": 0.85, "auc": 0.95},
        output_path=str(output_path),
    )
    
    # Verify file created
    assert output_path.exists()
    
    content = output_path.read_text(encoding="utf-8")
    
    # Verify sections present
    assert "# Test Experiment" in content
    assert "## Executive Summary" in content
    assert "## Dataset" in content
    assert "## Model Architecture" in content
    assert "## Training Configuration" in content
    assert "## Test Results" in content
    assert "## Reproduction Commands" in content
    
    # Verify data
    assert "85.00%" in content
    assert "0.9500" in content
    assert "ResNet50" in content


def test_generate_benchmark_report_with_all_sections(tmp_path):
    """Test full report with all optional sections."""
    output_path = tmp_path / "full_report.md"
    
    generate_benchmark_report(
        experiment_name="Full Test",
        dataset_info={"train_samples": 10000, "val_samples": 2000, "test_samples": 2000},
        model_info={"feature_extractor": "VGG16", "total_params": 138000000},
        training_config={"num_epochs": 20, "batch_size": 64},
        test_metrics={
            "accuracy": 0.88,
            "accuracy_ci_lower": 0.87,
            "accuracy_ci_upper": 0.89,
            "auc": 0.96,
            "auc_ci_lower": 0.95,
            "auc_ci_upper": 0.97,
            "f1": 0.85,
            "f1_ci_lower": 0.84,
            "f1_ci_upper": 0.86,
            "precision": 0.87,
            "precision_ci_lower": 0.86,
            "precision_ci_upper": 0.88,
            "recall": 0.83,
            "recall_ci_lower": 0.82,
            "recall_ci_upper": 0.84,
            "confusion_matrix": [[900, 100], [150, 850]],
        },
        comparison_results={
            "baselines": [
                {
                    "model_name": "Baseline1",
                    "test_metrics": {"accuracy": 0.80, "auc": 0.90, "f1": 0.78},
                    "model_parameters": 10000000,
                }
            ]
        },
        hardware_info={
            "gpu": "NVIDIA A100",
            "gpu_memory": "40GB",
            "cpu": "Intel Xeon",
            "ram": "128GB",
            "training_time": "1h 30m",
            "throughput": "1000 samples/sec",
        },
        output_path=str(output_path),
    )
    
    assert output_path.exists()
    
    content = output_path.read_text(encoding="utf-8")
    
    # Verify all sections
    assert "## Baseline Comparison" in content
    assert "## Hardware Specifications" in content
    assert "Baseline1" in content
    assert "NVIDIA A100" in content
    assert "Confusion Matrix" in content


def test_generate_benchmark_report_utf8_encoding(tmp_path):
    """Test report handles UTF-8 encoding correctly."""
    output_path = tmp_path / "utf8_report.md"
    
    generate_benchmark_report(
        experiment_name="Test with UTF-8: ±×÷",
        dataset_info={"train_samples": 1000, "val_samples": 200, "test_samples": 200, "image_size": "96×96"},
        model_info={"feature_extractor": "ResNet50", "total_params": 25000000},
        training_config={"num_epochs": 10},
        test_metrics={"accuracy": 0.85, "auc": 0.95},
        output_path=str(output_path),
    )
    
    assert output_path.exists()
    
    # Read with UTF-8 encoding
    content = output_path.read_text(encoding="utf-8")
    
    assert "±" in content or "×" in content or "96×96" in content


def test_generate_benchmark_report_default_output_path(tmp_path, monkeypatch):
    """Test report uses default output path."""
    # Change to temp directory
    monkeypatch.chdir(tmp_path)
    
    generate_benchmark_report(
        experiment_name="Default Path Test",
        dataset_info={"train_samples": 1000, "val_samples": 200, "test_samples": 200},
        model_info={"feature_extractor": "ResNet50", "total_params": 25000000},
        training_config={"num_epochs": 10},
        test_metrics={"accuracy": 0.85, "auc": 0.95},
    )
    
    # Check default path
    default_path = tmp_path / "PCAM_BENCHMARK_RESULTS.md"
    assert default_path.exists()


def test_generate_benchmark_report_creates_parent_dirs(tmp_path):
    """Test report creates parent directories if needed."""
    output_path = tmp_path / "nested" / "dir" / "report.md"
    
    generate_benchmark_report(
        experiment_name="Nested Path Test",
        dataset_info={"train_samples": 1000, "val_samples": 200, "test_samples": 200},
        model_info={"feature_extractor": "ResNet50", "total_params": 25000000},
        training_config={"num_epochs": 10},
        test_metrics={"accuracy": 0.85, "auc": 0.95},
        output_path=str(output_path),
    )
    
    assert output_path.exists()
    assert output_path.parent.exists()


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


def test_report_with_zero_metrics(tmp_path):
    """Test report handles zero metrics gracefully."""
    output_path = tmp_path / "zero_metrics.md"
    
    generate_benchmark_report(
        experiment_name="Zero Metrics",
        dataset_info={"train_samples": 0, "val_samples": 0, "test_samples": 0},
        model_info={"total_params": 0},
        training_config={"num_epochs": 0},
        test_metrics={"accuracy": 0.0, "auc": 0.0},
        output_path=str(output_path),
    )
    
    assert output_path.exists()
    content = output_path.read_text()
    assert "0.00%" in content


def test_report_with_very_large_numbers(tmp_path):
    """Test report formats very large numbers correctly."""
    output_path = tmp_path / "large_numbers.md"
    
    generate_benchmark_report(
        experiment_name="Large Numbers",
        dataset_info={"train_samples": 1000000000, "val_samples": 100000000, "test_samples": 100000000},
        model_info={"total_params": 500000000},
        training_config={"num_epochs": 100},
        test_metrics={"accuracy": 0.999, "auc": 0.9999},
        output_path=str(output_path),
    )
    
    assert output_path.exists()
    content = output_path.read_text()
    # Check comma formatting
    assert "1,000,000,000" in content
    assert "500,000,000" in content
