"""
Baseline model comparison script for PatchCamelyon.

Trains and evaluates multiple baseline models on PCam dataset:
- ResNet-18
- ResNet-50
- DenseNet-121
- EfficientNet-B0
- ViT-Base

Generates comparison tables and plots.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# Model configurations for RTX 4070 Laptop (8GB VRAM)
MODEL_CONFIGS = {
    "resnet18": {
        "batch_size": 64,
        "learning_rate": 0.001,
        "description": "ResNet-18 baseline",
    },
    "resnet50": {
        "batch_size": 48,
        "learning_rate": 0.001,
        "description": "ResNet-50 deeper baseline",
    },
    "densenet121": {
        "batch_size": 48,
        "learning_rate": 0.001,
        "description": "DenseNet-121 efficient baseline",
    },
    "efficientnet_b0": {
        "batch_size": 56,
        "learning_rate": 0.001,
        "description": "EfficientNet-B0 efficient baseline",
    },
    "vit_base_patch16_224": {
        "batch_size": 24,
        "learning_rate": 0.0005,
        "description": "Vision Transformer baseline",
    },
}


def load_results(results_dir: Path) -> Dict[str, Dict]:
    """
    Load results from all baseline experiments.

    Args:
        results_dir: Directory containing baseline results

    Returns:
        Dictionary mapping model names to their results
    """
    results = {}

    for model_name in MODEL_CONFIGS.keys():
        model_dir = results_dir / model_name
        metrics_file = model_dir / "metrics.json"

        if metrics_file.exists():
            with open(metrics_file, "r") as f:
                results[model_name] = json.load(f)
            logger.info(f"Loaded results for {model_name}")
        else:
            logger.warning(f"No results found for {model_name} at {metrics_file}")

    return results


def create_comparison_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """
    Create comparison table from results.

    Args:
        results: Dictionary of model results

    Returns:
        Pandas DataFrame with comparison metrics
    """
    data = []

    for model_name, metrics in results.items():
        config = MODEL_CONFIGS[model_name]

        row = {
            "Model": model_name,
            "Description": config["description"],
            "Batch Size": config["batch_size"],
            "Accuracy": metrics.get("test_accuracy", 0.0),
            "F1 Score": metrics.get("test_f1", 0.0),
            "AUC": metrics.get("test_auc", 0.0),
            "Training Time (h)": metrics.get("training_time_hours", 0.0),
            "Parameters (M)": metrics.get("num_parameters", 0) / 1e6,
        }
        data.append(row)

    df = pd.DataFrame(data)

    # Sort by AUC descending
    df = df.sort_values("AUC", ascending=False)

    return df


def plot_comparison(results: Dict[str, Dict], output_dir: Path):
    """
    Create comparison plots.

    Args:
        results: Dictionary of model results
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")
    plt.rcParams["figure.figsize"] = (12, 8)

    # 1. Accuracy comparison bar plot
    fig, ax = plt.subplots(figsize=(10, 6))

    models = list(results.keys())
    accuracies = [results[m].get("test_accuracy", 0) * 100 for m in models]
    f1_scores = [results[m].get("test_f1", 0) * 100 for m in models]
    aucs = [results[m].get("test_auc", 0) * 100 for m in models]

    x = np.arange(len(models))
    width = 0.25

    ax.bar(x - width, accuracies, width, label="Accuracy", alpha=0.8)
    ax.bar(x, f1_scores, width, label="F1 Score", alpha=0.8)
    ax.bar(x + width, aucs, width, label="AUC", alpha=0.8)

    ax.set_xlabel("Model", fontsize=12)
    ax.set_ylabel("Score (%)", fontsize=12)
    ax.set_title("Baseline Model Comparison - PCam Dataset", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "baseline_comparison.png", dpi=300, bbox_inches="tight")
    logger.info(f"Saved comparison plot to {output_dir / 'baseline_comparison.png'}")
    plt.close()

    # 2. Efficiency plot (Accuracy vs Parameters)
    fig, ax = plt.subplots(figsize=(10, 6))

    params = [results[m].get("num_parameters", 0) / 1e6 for m in models]
    accuracies = [results[m].get("test_accuracy", 0) * 100 for m in models]

    scatter = ax.scatter(params, accuracies, s=200, alpha=0.6, c=range(len(models)), cmap="viridis")

    for i, model in enumerate(models):
        ax.annotate(
            model,
            (params[i], accuracies[i]),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xlabel("Parameters (Millions)", fontsize=12)
    ax.set_ylabel("Test Accuracy (%)", fontsize=12)
    ax.set_title("Model Efficiency: Accuracy vs Parameters", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "efficiency_plot.png", dpi=300, bbox_inches="tight")
    logger.info(f"Saved efficiency plot to {output_dir / 'efficiency_plot.png'}")
    plt.close()

    # 3. Training time comparison
    fig, ax = plt.subplots(figsize=(10, 6))

    training_times = [results[m].get("training_time_hours", 0) for m in models]

    bars = ax.barh(models, training_times, alpha=0.7, color="steelblue")

    ax.set_xlabel("Training Time (hours)", fontsize=12)
    ax.set_ylabel("Model", fontsize=12)
    ax.set_title("Training Time Comparison (RTX 4070 Laptop)", fontsize=14, fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    # Add value labels
    for i, (model, time) in enumerate(zip(models, training_times)):
        ax.text(time + 0.1, i, f"{time:.1f}h", va="center", fontsize=10)

    plt.tight_layout()
    plt.savefig(output_dir / "training_time_comparison.png", dpi=300, bbox_inches="tight")
    logger.info(f"Saved training time plot to {output_dir / 'training_time_comparison.png'}")
    plt.close()


def generate_report(results: Dict[str, Dict], output_dir: Path):
    """
    Generate markdown report with comparison results.

    Args:
        results: Dictionary of model results
        output_dir: Directory to save report
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    df = create_comparison_table(results)

    report = f"""# PatchCamelyon Baseline Comparison Report

**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}
**Hardware:** RTX 4070 Laptop GPU (8GB VRAM)
**Dataset:** PatchCamelyon (262,144 train, 32,768 val, 32,768 test)

## Summary

Trained and evaluated {len(results)} baseline models on the PCam dataset.

## Results Table

{df.to_markdown(index=False, floatfmt='.4f')}

## Key Findings

### Best Overall Performance
- **Model:** {df.iloc[0]['Model']}
- **AUC:** {df.iloc[0]['AUC']:.4f}
- **Accuracy:** {df.iloc[0]['Accuracy']:.4f}
- **F1 Score:** {df.iloc[0]['F1 Score']:.4f}

### Most Efficient (Accuracy/Parameters)
"""

    # Calculate efficiency metric
    df["Efficiency"] = df["Accuracy"] / df["Parameters (M)"]
    most_efficient = df.loc[df["Efficiency"].idxmax()]

    report += f"""- **Model:** {most_efficient['Model']}
- **Accuracy:** {most_efficient['Accuracy']:.4f}
- **Parameters:** {most_efficient['Parameters (M)']:.2f}M
- **Efficiency Score:** {most_efficient['Efficiency']:.4f}

### Fastest Training
"""

    fastest = df.loc[df["Training Time (h)"].idxmin()]

    report += f"""- **Model:** {fastest['Model']}
- **Training Time:** {fastest['Training Time (h)']:.2f} hours
- **Accuracy:** {fastest['Accuracy']:.4f}

## Visualizations

![Baseline Comparison](baseline_comparison.png)

![Efficiency Plot](efficiency_plot.png)

![Training Time Comparison](training_time_comparison.png)

## Hardware Configuration

- **GPU:** RTX 4070 Laptop (8GB VRAM)
- **CPU:** Intel i7-14650HX (16 cores)
- **RAM:** 32GB DDR5
- **Mixed Precision:** Enabled (AMP)
- **cuDNN Benchmark:** Enabled

## Training Configuration

All models trained with:
- **Epochs:** 20
- **Optimizer:** AdamW
- **Scheduler:** Cosine Annealing
- **Weight Decay:** 0.0001
- **Mixed Precision:** Enabled
- **Early Stopping:** Patience 10

Batch sizes optimized per model for 8GB VRAM.

## Conclusion

{df.iloc[0]['Model']} achieved the best overall performance with an AUC of {df.iloc[0]['AUC']:.4f}.
All models successfully trained on the RTX 4070 Laptop GPU with appropriate batch size tuning.
"""

    report_path = output_dir / "baseline_comparison_report.md"
    with open(report_path, "w") as f:
        f.write(report)

    logger.info(f"Saved report to {report_path}")


def main():
    """Main comparison script."""
    parser = argparse.ArgumentParser(description="Compare baseline models on PCam")
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/baselines",
        help="Directory containing baseline results",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/baseline_comparison",
        help="Directory to save comparison outputs",
    )

    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)

    logger.info("=" * 60)
    logger.info("Baseline Model Comparison")
    logger.info("=" * 60)

    # Load results
    results = load_results(results_dir)

    if not results:
        logger.error("No results found. Please run baseline experiments first.")
        return

    logger.info(f"Loaded results for {len(results)} models")

    # Create comparison table
    df = create_comparison_table(results)
    print("\n" + "=" * 60)
    print("Comparison Table")
    print("=" * 60)
    print(df.to_string(index=False))

    # Save table
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / "baseline_comparison.csv", index=False)
    logger.info(f"Saved comparison table to {output_dir / 'baseline_comparison.csv'}")

    # Create plots
    plot_comparison(results, output_dir)

    # Generate report
    generate_report(results, output_dir)

    logger.info("\n" + "=" * 60)
    logger.info("Comparison complete!")
    logger.info(f"Results saved to {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
