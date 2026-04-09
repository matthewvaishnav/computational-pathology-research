"""
Comprehensive metrics analysis for trained models.

Analyzes training logs, generates plots, and creates detailed reports.
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_training_log(log_dir: Path) -> Optional[pd.DataFrame]:
    """
    Load training log from TensorBoard events or JSON.

    Args:
        log_dir: Directory containing training logs

    Returns:
        DataFrame with training metrics or None
    """
    # Try to load from events file (TensorBoard)
    try:
        from tensorboard.backend.event_processing import event_accumulator

        ea = event_accumulator.EventAccumulator(str(log_dir))
        ea.Reload()

        # Extract scalars
        data = []
        for tag in ea.Tags()["scalars"]:
            events = ea.Scalars(tag)
            for event in events:
                data.append(
                    {"step": event.step, "metric": tag, "value": event.value, "wall_time": event.wall_time}
                )

        if data:
            df = pd.DataFrame(data)
            logger.info(f"Loaded {len(df)} training log entries from TensorBoard")
            return df

    except Exception as e:
        logger.warning(f"Could not load TensorBoard logs: {e}")

    # Try to load from JSON
    json_log = log_dir / "training_log.json"
    if json_log.exists():
        with open(json_log, "r") as f:
            data = json.load(f)
        df = pd.DataFrame(data)
        logger.info(f"Loaded {len(df)} training log entries from JSON")
        return df

    logger.warning("No training logs found")
    return None


def plot_training_curves(log_df: pd.DataFrame, output_dir: Path):
    """
    Plot training curves (loss, accuracy, etc.).

    Args:
        log_df: DataFrame with training metrics
        output_dir: Directory to save plots
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")

    # Plot loss curves
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Training loss
    train_loss = log_df[log_df["metric"] == "train/loss"]
    if not train_loss.empty:
        axes[0, 0].plot(train_loss["step"], train_loss["value"], label="Train Loss", alpha=0.7)
        axes[0, 0].set_xlabel("Step")
        axes[0, 0].set_ylabel("Loss")
        axes[0, 0].set_title("Training Loss")
        axes[0, 0].legend()
        axes[0, 0].grid(alpha=0.3)

    # Validation metrics
    val_loss = log_df[log_df["metric"] == "epoch/val_loss"]
    val_acc = log_df[log_df["metric"] == "epoch/val_accuracy"]
    val_auc = log_df[log_df["metric"] == "epoch/val_auc"]

    if not val_loss.empty:
        axes[0, 1].plot(val_loss["step"], val_loss["value"], label="Val Loss", color="orange", alpha=0.7)
        axes[0, 1].set_xlabel("Epoch")
        axes[0, 1].set_ylabel("Loss")
        axes[0, 1].set_title("Validation Loss")
        axes[0, 1].legend()
        axes[0, 1].grid(alpha=0.3)

    if not val_acc.empty:
        axes[1, 0].plot(val_acc["step"], val_acc["value"], label="Val Accuracy", color="green", alpha=0.7)
        axes[1, 0].set_xlabel("Epoch")
        axes[1, 0].set_ylabel("Accuracy")
        axes[1, 0].set_title("Validation Accuracy")
        axes[1, 0].legend()
        axes[1, 0].grid(alpha=0.3)

    if not val_auc.empty:
        axes[1, 1].plot(val_auc["step"], val_auc["value"], label="Val AUC", color="purple", alpha=0.7)
        axes[1, 1].set_xlabel("Epoch")
        axes[1, 1].set_ylabel("AUC")
        axes[1, 1].set_title("Validation AUC")
        axes[1, 1].legend()
        axes[1, 1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "training_curves.png", dpi=300, bbox_inches="tight")
    logger.info(f"Saved training curves to {output_dir / 'training_curves.png'}")
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, output_dir: Path):
    """
    Plot confusion matrix.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        output_dir: Directory to save plot
    """
    cm = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Normal", "Tumor"],
        yticklabels=["Normal", "Tumor"],
        ax=ax,
    )
    ax.set_xlabel("Predicted Label", fontsize=12)
    ax.set_ylabel("True Label", fontsize=12)
    ax.set_title("Confusion Matrix", fontsize=14, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_dir / "confusion_matrix.png", dpi=300, bbox_inches="tight")
    logger.info(f"Saved confusion matrix to {output_dir / 'confusion_matrix.png'}")
    plt.close()


def plot_roc_curve(y_true: np.ndarray, y_scores: np.ndarray, output_dir: Path):
    """
    Plot ROC curve.

    Args:
        y_true: True labels
        y_scores: Predicted probabilities
        output_dir: Directory to save plot
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    auc = roc_auc_score(y_true, y_scores)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc:.4f})", linewidth=2)
    ax.plot([0, 1], [0, 1], "k--", label="Random Classifier", alpha=0.5)
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve", fontsize=14, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "roc_curve.png", dpi=300, bbox_inches="tight")
    logger.info(f"Saved ROC curve to {output_dir / 'roc_curve.png'}")
    plt.close()


def plot_precision_recall_curve(y_true: np.ndarray, y_scores: np.ndarray, output_dir: Path):
    """
    Plot precision-recall curve.

    Args:
        y_true: True labels
        y_scores: Predicted probabilities
        output_dir: Directory to save plot
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_scores)

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, linewidth=2)
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Curve", fontsize=14, fontweight="bold")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / "precision_recall_curve.png", dpi=300, bbox_inches="tight")
    logger.info(f"Saved precision-recall curve to {output_dir / 'precision_recall_curve.png'}")
    plt.close()


def generate_metrics_report(
    metrics: Dict,
    y_true: Optional[np.ndarray] = None,
    y_pred: Optional[np.ndarray] = None,
    output_dir: Optional[Path] = None,
) -> str:
    """
    Generate comprehensive metrics report.

    Args:
        metrics: Dictionary of metrics
        y_true: True labels (optional)
        y_pred: Predicted labels (optional)
        output_dir: Directory to save report

    Returns:
        Markdown report string
    """
    report = f"""# Model Metrics Analysis Report

**Date:** {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary Metrics

| Metric | Value |
|--------|-------|
| Accuracy | {metrics.get('accuracy', 0):.4f} |
| F1 Score | {metrics.get('f1', 0):.4f} |
| AUC | {metrics.get('auc', 0):.4f} |
| Precision | {metrics.get('precision', 0):.4f} |
| Recall | {metrics.get('recall', 0):.4f} |

"""

    if y_true is not None and y_pred is not None:
        report += f"""## Classification Report

```
{classification_report(y_true, y_pred, target_names=['Normal', 'Tumor'])}
```

## Confusion Matrix

![Confusion Matrix](confusion_matrix.png)

"""

    report += """## Training Curves

![Training Curves](training_curves.png)

## ROC Curve

![ROC Curve](roc_curve.png)

## Precision-Recall Curve

![Precision-Recall Curve](precision_recall_curve.png)

"""

    if output_dir:
        report_path = output_dir / "metrics_report.md"
        with open(report_path, "w") as f:
            f.write(report)
        logger.info(f"Saved metrics report to {report_path}")

    return report


def main():
    """Main metrics analysis script."""
    parser = argparse.ArgumentParser(description="Analyze model metrics")
    parser.add_argument(
        "--log-dir",
        type=str,
        required=True,
        help="Directory containing training logs",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint (optional)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/metrics_analysis",
        help="Directory to save analysis outputs",
    )

    args = parser.parse_args()

    log_dir = Path(args.log_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("Metrics Analysis")
    logger.info("=" * 60)

    # Load training log
    log_df = load_training_log(log_dir)

    if log_df is not None:
        # Plot training curves
        plot_training_curves(log_df, output_dir)

    # Load checkpoint if provided
    if args.checkpoint:
        checkpoint_path = Path(args.checkpoint)
        if checkpoint_path.exists():
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            metrics = checkpoint.get("metrics", {})

            logger.info("Checkpoint metrics:")
            for key, value in metrics.items():
                logger.info(f"  {key}: {value:.4f}")

            # Save metrics to JSON
            with open(output_dir / "metrics.json", "w") as f:
                json.dump(metrics, f, indent=2)

            # Generate report
            generate_metrics_report(metrics, output_dir=output_dir)

    logger.info("\n" + "=" * 60)
    logger.info("Analysis complete!")
    logger.info(f"Results saved to {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
