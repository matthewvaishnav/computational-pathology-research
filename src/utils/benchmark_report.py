"""Benchmark report generation utilities."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def generate_benchmark_report(
    experiment_name: str,
    dataset_info: Dict[str, Any],
    model_info: Dict[str, Any],
    training_config: Dict[str, Any],
    test_metrics: Dict[str, Any],
    comparison_results: Optional[Dict[str, Any]] = None,
    hardware_info: Optional[Dict[str, Any]] = None,
    output_path: str = "PCAM_BENCHMARK_RESULTS.md",
) -> None:
    """
    Generate comprehensive benchmark report in markdown format.

    Args:
        experiment_name: Name of the experiment
        dataset_info: Dataset statistics and details
        model_info: Model architecture and parameters
        training_config: Training hyperparameters
        test_metrics: Test set metrics with CIs
        comparison_results: Optional baseline comparison results
        hardware_info: Optional hardware specifications
        output_path: Path to save the report

    Generates:
        - Executive summary
        - Dataset description
        - Model architecture
        - Training configuration
        - Results with confidence intervals
        - Baseline comparison table
        - Reproduction commands
        - Hardware specifications
    """
    report_lines = []

    # Header
    report_lines.append(f"# {experiment_name}")
    report_lines.append("")
    report_lines.append(f"**Date**: {datetime.now().strftime('%Y-%m-%d')}")
    report_lines.append("**Status**: ✅ COMPLETE")
    report_lines.append("")

    # Executive Summary
    report_lines.append("## Executive Summary")
    report_lines.append("")
    _add_executive_summary(report_lines, test_metrics, dataset_info, training_config)
    report_lines.append("")

    # Dataset Description
    report_lines.append("## Dataset")
    report_lines.append("")
    _add_dataset_description(report_lines, dataset_info)
    report_lines.append("")

    # Model Architecture
    report_lines.append("## Model Architecture")
    report_lines.append("")
    _add_model_architecture(report_lines, model_info)
    report_lines.append("")

    # Training Configuration
    report_lines.append("## Training Configuration")
    report_lines.append("")
    _add_training_configuration(report_lines, training_config)
    report_lines.append("")

    # Test Results
    report_lines.append("## Test Results")
    report_lines.append("")
    _add_test_results(report_lines, test_metrics)
    report_lines.append("")

    # Baseline Comparison (if provided)
    if comparison_results:
        report_lines.append("## Baseline Comparison")
        report_lines.append("")
        _add_baseline_comparison(report_lines, comparison_results)
        report_lines.append("")

    # Hardware Specifications (if provided)
    if hardware_info:
        report_lines.append("## Hardware Specifications")
        report_lines.append("")
        _add_hardware_info(report_lines, hardware_info)
        report_lines.append("")

    # Reproduction Commands
    report_lines.append("## Reproduction Commands")
    report_lines.append("")
    _add_reproduction_commands(report_lines, training_config, dataset_info)
    report_lines.append("")

    # Write report with UTF-8 encoding
    output_file = Path(output_path)
    output_file.write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Benchmark report saved to: {output_path}")


def _add_executive_summary(
    lines: list,
    test_metrics: Dict[str, Any],
    dataset_info: Dict[str, Any],
    training_config: Dict[str, Any],
) -> None:
    """Add executive summary section."""
    accuracy = test_metrics.get("accuracy", 0.0)
    auc = test_metrics.get("auc", 0.0)

    # Check if confidence intervals are available
    has_ci = "accuracy_ci_lower" in test_metrics

    if has_ci:
        acc_ci_lower = test_metrics.get("accuracy_ci_lower", 0.0)
        acc_ci_upper = test_metrics.get("accuracy_ci_upper", 0.0)
        auc_ci_lower = test_metrics.get("auc_ci_lower", 0.0)
        auc_ci_upper = test_metrics.get("auc_ci_upper", 0.0)

        lines.append(
            f"Successfully trained and evaluated on the full PatchCamelyon dataset, "
            f"achieving **{accuracy*100:.2f}% ± {(acc_ci_upper - acc_ci_lower)*50:.2f}% test accuracy** "
            f"and **{auc:.4f} ± {(auc_ci_upper - auc_ci_lower)/2:.4f} AUC** (95% CI)."
        )
    else:
        lines.append(
            f"Successfully trained and evaluated on the full PatchCamelyon dataset, "
            f"achieving **{accuracy*100:.2f}% test accuracy** and **{auc:.4f} AUC**."
        )

    lines.append("")
    lines.append("**Key Results:**")
    lines.append(f"- Training samples: {dataset_info.get('train_samples', 'N/A'):,}")
    lines.append(f"- Test samples: {dataset_info.get('test_samples', 'N/A'):,}")
    lines.append(f"- Training epochs: {training_config.get('num_epochs', 'N/A')}")
    lines.append(f"- Test accuracy: {accuracy*100:.2f}%")
    lines.append(f"- Test AUC: {auc:.4f}")


def _add_dataset_description(lines: list, dataset_info: Dict[str, Any]) -> None:
    """Add dataset description section."""
    lines.append("| Property | Value |")
    lines.append("|----------|-------|")
    lines.append(f"| **Dataset** | {dataset_info.get('name', 'PatchCamelyon')} |")
    lines.append(f"| **Train samples** | {dataset_info.get('train_samples', 'N/A'):,} |")
    lines.append(f"| **Validation samples** | {dataset_info.get('val_samples', 'N/A'):,} |")
    lines.append(f"| **Test samples** | {dataset_info.get('test_samples', 'N/A'):,} |")
    lines.append(f"| **Image size** | {dataset_info.get('image_size', '96×96')} |")
    lines.append(f"| **Classes** | {dataset_info.get('num_classes', 2)} (binary) |")
    lines.append(f"| **Task** | {dataset_info.get('task', 'Metastatic tissue detection')} |")


def _add_model_architecture(lines: list, model_info: Dict[str, Any]) -> None:
    """Add model architecture section."""
    lines.append("| Component | Details |")
    lines.append("|-----------|---------|")
    lines.append(f"| **Feature Extractor** | {model_info.get('feature_extractor', 'N/A')} |")
    lines.append(f"| **Feature Dimension** | {model_info.get('feature_dim', 'N/A')} |")
    lines.append(f"| **Encoder** | {model_info.get('encoder', 'N/A')} |")
    lines.append(f"| **Encoder Hidden Dim** | {model_info.get('hidden_dim', 'N/A')} |")
    lines.append(f"| **Total Parameters** | {model_info.get('total_params', 'N/A'):,} |")
    lines.append(f"| **Pretrained** | {model_info.get('pretrained', 'Yes')} |")


def _add_training_configuration(lines: list, training_config: Dict[str, Any]) -> None:
    """Add training configuration section."""
    lines.append("```yaml")
    lines.append(f"num_epochs: {training_config.get('num_epochs', 'N/A')}")
    lines.append(f"batch_size: {training_config.get('batch_size', 'N/A')}")
    lines.append(f"learning_rate: {training_config.get('learning_rate', 'N/A')}")
    lines.append(f"weight_decay: {training_config.get('weight_decay', 'N/A')}")
    lines.append(f"optimizer: {training_config.get('optimizer', 'AdamW')}")
    lines.append(f"use_amp: {training_config.get('use_amp', 'true')}")
    lines.append(
        f"early_stopping_patience: {training_config.get('early_stopping_patience', 'N/A')}"
    )
    lines.append("```")


def _add_test_results(lines: list, test_metrics: Dict[str, Any]) -> None:
    """Add test results section with confidence intervals if available."""
    has_ci = "accuracy_ci_lower" in test_metrics

    lines.append("### Overall Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("|--------|-------|")

    # Format metrics with CI if available
    if has_ci:
        accuracy = test_metrics.get("accuracy", 0.0)
        acc_ci_lower = test_metrics.get("accuracy_ci_lower", 0.0)
        acc_ci_upper = test_metrics.get("accuracy_ci_upper", 0.0)
        lines.append(
            f"| **Accuracy** | {accuracy*100:.2f}% [{acc_ci_lower*100:.2f}%, {acc_ci_upper*100:.2f}%] |"
        )

        auc = test_metrics.get("auc", 0.0)
        auc_ci_lower = test_metrics.get("auc_ci_lower", 0.0)
        auc_ci_upper = test_metrics.get("auc_ci_upper", 0.0)
        lines.append(f"| **AUC** | {auc:.4f} [{auc_ci_lower:.4f}, {auc_ci_upper:.4f}] |")

        f1 = test_metrics.get("f1", 0.0)
        f1_ci_lower = test_metrics.get("f1_ci_lower", 0.0)
        f1_ci_upper = test_metrics.get("f1_ci_upper", 0.0)
        lines.append(f"| **F1 Score** | {f1:.4f} [{f1_ci_lower:.4f}, {f1_ci_upper:.4f}] |")

        precision = test_metrics.get("precision", 0.0)
        prec_ci_lower = test_metrics.get("precision_ci_lower", 0.0)
        prec_ci_upper = test_metrics.get("precision_ci_upper", 0.0)
        lines.append(
            f"| **Precision** | {precision:.4f} [{prec_ci_lower:.4f}, {prec_ci_upper:.4f}] |"
        )

        recall = test_metrics.get("recall", 0.0)
        rec_ci_lower = test_metrics.get("recall_ci_lower", 0.0)
        rec_ci_upper = test_metrics.get("recall_ci_upper", 0.0)
        lines.append(f"| **Recall** | {recall:.4f} [{rec_ci_lower:.4f}, {rec_ci_upper:.4f}] |")
    else:
        lines.append(f"| **Accuracy** | {test_metrics.get('accuracy', 0.0)*100:.2f}% |")
        lines.append(f"| **AUC** | {test_metrics.get('auc', 0.0):.4f} |")
        lines.append(f"| **F1 Score** | {test_metrics.get('f1', 0.0):.4f} |")
        lines.append(f"| **Precision** | {test_metrics.get('precision', 0.0):.4f} |")
        lines.append(f"| **Recall** | {test_metrics.get('recall', 0.0):.4f} |")

    # Confusion matrix if available
    if "confusion_matrix" in test_metrics:
        lines.append("")
        lines.append("### Confusion Matrix")
        lines.append("")
        cm = test_metrics["confusion_matrix"]
        lines.append("```")
        lines.append("           Predicted")
        lines.append("           0    1")
        lines.append(f"Actual 0  [{cm[0][0]:5d} {cm[0][1]:5d}]")
        lines.append(f"       1  [{cm[1][0]:5d} {cm[1][1]:5d}]")
        lines.append("```")


def _add_baseline_comparison(lines: list, comparison_results: Dict[str, Any]) -> None:
    """Add baseline comparison table."""
    lines.append("| Model | Accuracy | AUC | F1 | Parameters |")
    lines.append("|-------|----------|-----|----|-----------:|")

    for baseline in comparison_results.get("baselines", []):
        model_name = baseline.get("model_name", "Unknown")
        metrics = baseline.get("test_metrics", {})
        params = baseline.get("model_parameters", 0)

        accuracy = metrics.get("accuracy", 0.0)
        auc = metrics.get("auc", 0.0)
        f1 = metrics.get("f1", 0.0)

        # Check if CI available
        if "accuracy_ci_lower" in metrics:
            acc_ci_lower = metrics.get("accuracy_ci_lower", 0.0)
            acc_ci_upper = metrics.get("accuracy_ci_upper", 0.0)
            auc_ci_lower = metrics.get("auc_ci_lower", 0.0)
            auc_ci_upper = metrics.get("auc_ci_upper", 0.0)
            f1_ci_lower = metrics.get("f1_ci_lower", 0.0)
            f1_ci_upper = metrics.get("f1_ci_upper", 0.0)

            lines.append(
                f"| {model_name} | "
                f"{accuracy*100:.2f}% ± {(acc_ci_upper - acc_ci_lower)*50:.2f}% | "
                f"{auc:.4f} ± {(auc_ci_upper - auc_ci_lower)/2:.4f} | "
                f"{f1:.4f} ± {(f1_ci_upper - f1_ci_lower)/2:.4f} | "
                f"{params:,} |"
            )
        else:
            lines.append(
                f"| {model_name} | "
                f"{accuracy*100:.2f}% | "
                f"{auc:.4f} | "
                f"{f1:.4f} | "
                f"{params:,} |"
            )


def _add_hardware_info(lines: list, hardware_info: Dict[str, Any]) -> None:
    """Add hardware specifications section."""
    lines.append("| Component | Specification |")
    lines.append("|-----------|---------------|")
    lines.append(f"| **GPU** | {hardware_info.get('gpu', 'N/A')} |")
    lines.append(f"| **GPU Memory** | {hardware_info.get('gpu_memory', 'N/A')} |")
    lines.append(f"| **CPU** | {hardware_info.get('cpu', 'N/A')} |")
    lines.append(f"| **RAM** | {hardware_info.get('ram', 'N/A')} |")
    lines.append(f"| **Training Time** | {hardware_info.get('training_time', 'N/A')} |")
    lines.append(f"| **Throughput** | {hardware_info.get('throughput', 'N/A')} |")


def _add_reproduction_commands(
    lines: list, training_config: Dict[str, Any], dataset_info: Dict[str, Any]
) -> None:
    """Add reproduction commands section."""
    config_file = training_config.get(
        "config_file", "experiments/configs/pcam_fullscale/gpu_16gb.yaml"
    )
    data_root = dataset_info.get("data_root", "./data/pcam")
    checkpoint_path = training_config.get(
        "checkpoint_path", "checkpoints/pcam_fullscale/best_model.pth"
    )

    lines.append("### Training")
    lines.append("")
    lines.append("```bash")
    lines.append("# Download dataset")
    lines.append(f"python scripts/download_pcam.py --output-dir {data_root}")
    lines.append("")
    lines.append("# Train model")
    lines.append(f"python experiments/train_pcam.py --config {config_file}")
    lines.append("```")
    lines.append("")

    lines.append("### Evaluation")
    lines.append("")
    lines.append("```bash")
    lines.append("# Evaluate with bootstrap confidence intervals")
    lines.append("python experiments/evaluate_pcam.py \\")
    lines.append(f"  --checkpoint {checkpoint_path} \\")
    lines.append(f"  --data-root {data_root} \\")
    lines.append("  --output-dir results/pcam_fullscale \\")
    lines.append("  --compute-bootstrap-ci \\")
    lines.append("  --bootstrap-samples 1000")
    lines.append("```")
