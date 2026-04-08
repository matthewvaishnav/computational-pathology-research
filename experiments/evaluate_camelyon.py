"""
Evaluation script for trained CAMELYON model.

This script loads a checkpoint from train_camelyon.py and evaluates the model
on the test set, computing slide-level metrics including accuracy, AUC,
precision, recall, F1, and generating confusion matrix and ROC curve plots.

Example usage:
    python evaluate_camelyon.py --checkpoint checkpoints/camelyon/best_model.pth
    python evaluate_camelyon.py --checkpoint checkpoints/camelyon/best_model.pth --batch-size 32
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
    roc_curve,
)

try:
    import seaborn as sns
    import matplotlib.pyplot as plt

    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False

from src.data.camelyon_dataset import CAMELYONSlideIndex, CAMELYONPatchDataset

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_checkpoint(
    checkpoint_path: str,
    device: str,
) -> Tuple[nn.Module, Dict, Dict, int]:
    """Load checkpoint and return (model, config, metrics, epoch).

    Args:
        checkpoint_path: Path to the checkpoint file.
        device: Device to map checkpoint tensors to.

    Returns:
        Tuple of (model, config, metrics, epoch).

    Raises:
        FileNotFoundError: If checkpoint file does not exist.
        RuntimeError: If checkpoint is corrupted or incompatible.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found at: {checkpoint_path}\n"
            "Please provide a valid checkpoint path using --checkpoint argument."
        )

    try:
        logger.info(f"Loading checkpoint from: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)

        # Import SimpleSlideClassifier from training script
        sys.path.insert(0, str(Path(__file__).parent))
        from train_camelyon import SimpleSlideClassifier

        # Get config
        config = checkpoint.get("config", {})

        # Reconstruct model from config
        feature_dim = checkpoint["model_state_dict"]["classifier.0.weight"].shape[1]
        hidden_dim = config["model"]["wsi"]["hidden_dim"]
        num_classes = config["task"]["num_classes"]
        dropout = config["task"]["classification"]["dropout"]

        model = SimpleSlideClassifier(
            feature_dim=feature_dim,
            hidden_dim=hidden_dim,
            num_classes=num_classes,
            pooling="mean",
            dropout=dropout,
        )

        # Load state dict
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(device)

        metrics = checkpoint.get("metrics", {})
        epoch = checkpoint.get("epoch", 0)

        logger.info(f"Successfully loaded checkpoint from epoch {epoch}")
        if metrics:
            logger.info(f"Checkpoint metrics: {metrics}")

        return model, config, metrics, epoch

    except KeyError as e:
        raise RuntimeError(
            f"Checkpoint is missing expected key: {e}\n"
            "The checkpoint may be corrupted or incompatible with this evaluation script."
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")


def evaluate_slide_level(
    model: nn.Module,
    slide_index: CAMELYONSlideIndex,
    features_dir: Path,
    split: str,
    device: str,
    aggregation: str = "mean",
) -> Dict[str, Any]:
    """Run slide-level inference and compute metrics.

    Args:
        model: Trained slide classifier model.
        slide_index: Slide index with metadata.
        features_dir: Directory containing HDF5 feature files.
        split: Which split to evaluate ('test', 'val', 'train').
        device: Device to run inference on.
        aggregation: Aggregation method ('mean' or 'max').

    Returns:
        Dictionary containing:
        - 'accuracy': float
        - 'auc': float
        - 'precision': float (macro)
        - 'recall': float (macro)
        - 'f1': float (macro)
        - 'confusion_matrix': np.ndarray [[TN, FP], [FN, TP]]
        - 'per_class_metrics': dict with precision/recall/f1 for each class
        - 'slide_predictions': dict mapping slide_id to prediction
        - 'slide_probabilities': dict mapping slide_id to probability
        - 'slide_labels': dict mapping slide_id to true label
    """
    model.eval()

    slides = slide_index.get_slides_by_split(split)

    slide_predictions = {}
    slide_probabilities = {}
    slide_labels = {}

    logger.info(f"Running slide-level inference on {len(slides)} slides...")

    with torch.no_grad():
        for slide in tqdm(slides, desc=f"Evaluating {split} slides"):
            slide_id = slide.slide_id
            label = slide.label

            # Load slide features
            feature_file = features_dir / f"{slide_id}.h5"
            if not feature_file.exists():
                logger.warning(f"Feature file not found: {feature_file}, skipping slide")
                continue

            import h5py

            with h5py.File(feature_file, "r") as f:
                features = torch.tensor(f["features"][:], dtype=torch.float32)

            # Add batch dimension: [num_patches, feature_dim] -> [1, num_patches, feature_dim]
            features = features.unsqueeze(0).to(device)

            # Forward pass
            logits = model(features).squeeze()  # [1]

            # Get prediction
            prob = torch.sigmoid(logits).item()
            pred = 1 if prob > 0.5 else 0

            slide_predictions[slide_id] = pred
            slide_probabilities[slide_id] = prob
            slide_labels[slide_id] = label

    # Convert to arrays for metrics computation
    slide_ids = list(slide_labels.keys())
    predictions = np.array([slide_predictions[sid] for sid in slide_ids])
    probabilities = np.array([slide_probabilities[sid] for sid in slide_ids])
    labels = np.array([slide_labels[sid] for sid in slide_ids])

    # Compute metrics
    metrics = compute_metrics(predictions, probabilities, labels)

    # Add slide-level data
    metrics["slide_predictions"] = slide_predictions
    metrics["slide_probabilities"] = slide_probabilities
    metrics["slide_labels"] = slide_labels
    metrics["num_slides"] = len(slide_ids)

    return metrics


def compute_metrics(
    predictions: np.ndarray,
    probabilities: np.ndarray,
    labels: np.ndarray,
) -> Dict[str, Any]:
    """Compute all classification metrics using sklearn.

    Args:
        predictions: Binary predictions (0 or 1).
        probabilities: Predicted probabilities for positive class.
        labels: Ground truth labels (0 or 1).

    Returns:
        Dictionary containing accuracy, AUC, precision, recall, F1,
        confusion matrix, and per-class metrics.
    """
    class_labels = [0, 1]

    # Basic metrics
    accuracy = accuracy_score(labels, predictions)

    if len(np.unique(labels)) < 2:
        auc = None
        logger.warning(
            "ROC AUC is undefined because only one class is present in the evaluation labels."
        )
    else:
        try:
            auc = float(roc_auc_score(labels, probabilities))
        except ValueError:
            auc = None
            logger.warning("ROC AUC could not be computed for this evaluation split.")

    # Per-class metrics
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        labels, predictions, labels=class_labels, average=None, zero_division=0
    )

    # Macro metrics
    precision_macro = float(np.mean(precision_per_class))
    recall_macro = float(np.mean(recall_per_class))
    f1_macro = float(np.mean(f1_per_class))

    # Binary metrics for the positive class (class 1)
    precision_binary = float(precision_per_class[1])
    recall_binary = float(recall_per_class[1])
    f1_binary = float(f1_per_class[1])

    # Confusion matrix: [[TN, FP], [FN, TP]]
    cm = confusion_matrix(labels, predictions, labels=class_labels)

    # Build results
    metrics = {
        "accuracy": float(accuracy),
        "auc": auc,
        "precision": precision_macro,
        "recall": recall_macro,
        "f1": f1_macro,
        "precision_binary": precision_binary,
        "recall_binary": recall_binary,
        "f1_binary": f1_binary,
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": {
            "class_0": {
                "precision": float(precision_per_class[0]),
                "recall": float(recall_per_class[0]),
                "f1": float(f1_per_class[0]),
            },
            "class_1": {
                "precision": float(precision_per_class[1]),
                "recall": float(recall_per_class[1]),
                "f1": float(f1_per_class[1]),
            },
        },
    }

    return metrics


def save_metrics(metrics: Dict, output_path: str) -> None:
    """Save metrics to JSON file with pretty printing.

    Args:
        metrics: Dictionary of metrics to save.
        output_path: Path to save the JSON file.
    """
    # Create output directory if needed
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python native types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32, np.float16)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj

    metrics_to_save = convert_to_serializable(metrics)

    with open(output_path, "w") as f:
        json.dump(metrics_to_save, f, indent=2)

    logger.info(f"Metrics saved to: {output_path}")


def plot_confusion_matrix(cm: np.ndarray, output_path: str) -> None:
    """Save confusion matrix heatmap using seaborn.

    Args:
        cm: Confusion matrix [[TN, FP], [FN, TP]].
        output_path: Path to save the plot.
    """
    if not PLOT_AVAILABLE:
        logger.warning("seaborn/matplotlib not available, skipping confusion matrix plot")
        return

    # Create output directory if needed
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Predicted Normal", "Predicted Tumor"],
        yticklabels=["Actual Normal", "Actual Tumor"],
        cbar_kws={"label": "Count"},
    )
    plt.title("Confusion Matrix - Slide-Level Classification")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info(f"Confusion matrix plot saved to: {output_path}")


def plot_roc_curve(fpr: np.ndarray, tpr: np.ndarray, auc: float, output_path: str) -> None:
    """Save ROC curve plot.

    Args:
        fpr: False positive rates.
        tpr: True positive rates.
        auc: Area under the ROC curve.
        output_path: Path to save the plot.
    """
    if not PLOT_AVAILABLE:
        logger.warning("seaborn/matplotlib not available, skipping ROC curve plot")
        return

    # Create output directory if needed
    output_dir = Path(output_path).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color="darkorange", lw=2, label=f"ROC curve (AUC = {auc:.4f})")
    plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label="Random classifier")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve - Slide-Level Classification")
    plt.legend(loc="lower right")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info(f"ROC curve plot saved to: {output_path}")


def count_model_parameters(model: nn.Module) -> int:
    """Count total number of parameters in a model.

    Args:
        model: PyTorch model.

    Returns:
        Total parameter count.
    """
    return sum(p.numel() for p in model.parameters())


def get_hardware_info() -> Dict[str, Any]:
    """Get hardware information including GPU details if available.

    Returns:
        Dictionary with hardware information.
    """
    info = {
        "cuda_available": torch.cuda.is_available(),
        "pytorch_version": torch.__version__,
    }

    if torch.cuda.is_available():
        info["gpu_name"] = torch.cuda.get_device_name(0)
        info["gpu_memory_total_gb"] = torch.cuda.get_device_properties(0).total_memory / 1e9
        info["cuda_version"] = torch.version.cuda

    return info


def log_evaluation_summary(
    checkpoint_path: str,
    epoch: int,
    num_slides: int,
    inference_time: float,
    test_metrics: Dict[str, Any],
    output_dir: Path,
    confusion_matrix_generated: bool,
    roc_curve_generated: bool,
) -> None:
    """Log the final evaluation summary."""
    cm = np.array(test_metrics["confusion_matrix"])

    logger.info("\n" + "=" * 60)
    logger.info("CAMELYON EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Checkpoint epoch: {epoch}")
    logger.info(f"Test slides: {num_slides}")
    logger.info(f"Inference time: {inference_time:.2f} seconds")
    logger.info("-" * 60)
    logger.info("Slide-Level Test Metrics:")
    logger.info(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    if test_metrics["auc"] is None:
        logger.info("  AUC:       undefined (single-class labels)")
    else:
        logger.info(f"  AUC:       {test_metrics['auc']:.4f}")
    logger.info(f"  Precision: {test_metrics['precision']:.4f}")
    logger.info(f"  Recall:    {test_metrics['recall']:.4f}")
    logger.info(f"  F1:        {test_metrics['f1']:.4f}")
    logger.info("-" * 60)
    logger.info("Per-class metrics (Normal / Tumor):")
    for cls in ["class_0", "class_1"]:
        cls_name = "Normal" if cls == "class_0" else "Tumor"
        cls_metrics = test_metrics["per_class_metrics"][cls]
        logger.info(f"  {cls_name}:")
        logger.info(f"    Precision: {cls_metrics['precision']:.4f}")
        logger.info(f"    Recall:    {cls_metrics['recall']:.4f}")
        logger.info(f"    F1:        {cls_metrics['f1']:.4f}")
    logger.info("-" * 60)
    logger.info("Confusion Matrix:")
    logger.info(f"  [[TN={cm[0,0]}, FP={cm[0,1]}],")
    logger.info(f"   [FN={cm[1,0]}, TP={cm[1,1]}]]")
    logger.info("-" * 60)
    logger.info("Output files:")
    logger.info(f"  Metrics: {output_dir / 'metrics.json'}")
    if confusion_matrix_generated:
        logger.info(f"  Confusion matrix: {output_dir / 'confusion_matrix.png'}")
    if roc_curve_generated:
        logger.info(f"  ROC curve: {output_dir / 'roc_curve.png'}")
    logger.info("=" * 60)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate CAMELYON model on test set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiments/evaluate_camelyon.py --checkpoint checkpoints/camelyon/best_model.pth
  python experiments/evaluate_camelyon.py --checkpoint checkpoints/camelyon_quick_test/best_model.pth --split test
  python experiments/evaluate_camelyon.py --checkpoint checkpoints/camelyon/best_model.pth --output-dir results/camelyon
        """,
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the checkpoint file (required)"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/camelyon",
        help="Root directory of CAMELYON dataset (default: data/camelyon)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Which split to evaluate (default: test)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/camelyon",
        help="Output directory for results (default: results/camelyon)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for evaluation (default: cuda if available)",
    )
    parser.add_argument(
        "--aggregation",
        type=str,
        default="mean",
        choices=["mean", "max"],
        help="Patch aggregation method (default: mean)",
    )

    args = parser.parse_args()

    # Determine device
    if args.device == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    # Log hardware info
    hw_info = get_hardware_info()
    logger.info("=" * 60)
    logger.info("CAMELYON Model Evaluation")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Device: {device}")
    logger.info(f"PyTorch version: {hw_info['pytorch_version']}")
    if hw_info["cuda_available"]:
        logger.info(f"GPU: {hw_info['gpu_name']}")
        logger.info(f"GPU memory: {hw_info['gpu_memory_total_gb']:.2f} GB")
        logger.info(f"CUDA version: {hw_info['cuda_version']}")
    logger.info("=" * 60)

    # Load checkpoint
    try:
        model, config, checkpoint_metrics, epoch = load_checkpoint(
            args.checkpoint,
            str(device),
        )
    except (FileNotFoundError, RuntimeError) as e:
        logger.error(f"Failed to load checkpoint: {e}")
        sys.exit(1)

    # Load slide index
    data_root = Path(args.data_root)
    index_path = data_root / "slide_index.json"
    features_dir = data_root / "features"

    if not index_path.exists():
        logger.error(f"Slide index not found: {index_path}")
        logger.error("Please generate synthetic data first:")
        logger.error("  python scripts/generate_synthetic_camelyon.py")
        sys.exit(1)

    if not features_dir.exists():
        logger.error(f"Features directory not found: {features_dir}")
        logger.error("Please generate synthetic data first:")
        logger.error("  python scripts/generate_synthetic_camelyon.py")
        sys.exit(1)

    logger.info(f"Loading slide index from: {index_path}")
    slide_index = CAMELYONSlideIndex.load(index_path)
    logger.info(f"Loaded {len(slide_index)} slides")

    # Count model parameters
    total_params = count_model_parameters(model)
    logger.info(f"Model parameters: {total_params:,}")

    # Run evaluation
    logger.info(f"Starting evaluation on {args.split} split...")
    start_time = time.time()

    test_metrics = evaluate_slide_level(
        model=model,
        slide_index=slide_index,
        features_dir=features_dir,
        split=args.split,
        device=str(device),
        aggregation=args.aggregation,
    )

    inference_time = time.time() - start_time
    slides_per_second = test_metrics["num_slides"] / inference_time

    logger.info(f"Evaluation completed in {inference_time:.2f} seconds")
    logger.info(f"Throughput: {slides_per_second:.2f} slides/second")

    # Add metadata to metrics
    test_metrics["hardware_info"] = hw_info
    test_metrics["inference_time_seconds"] = inference_time
    test_metrics["slides_per_second"] = slides_per_second
    test_metrics["model_parameters"] = total_params
    test_metrics["checkpoint_epoch"] = epoch
    test_metrics["checkpoint_path"] = args.checkpoint
    test_metrics["checkpoint_metrics"] = checkpoint_metrics
    test_metrics["aggregation_method"] = args.aggregation
    test_metrics["split"] = args.split

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics JSON
    metrics_path = output_dir / "metrics.json"
    save_metrics(test_metrics, str(metrics_path))

    cm = np.array(test_metrics["confusion_matrix"])

    # Generate plots
    confusion_matrix_generated = False
    roc_curve_generated = False

    if PLOT_AVAILABLE:
        plot_confusion_matrix(cm, str(output_dir / "confusion_matrix.png"))
        confusion_matrix_generated = True

        # ROC curve
        if test_metrics["auc"] is not None:
            slide_ids = list(test_metrics["slide_labels"].keys())
            labels = np.array([test_metrics["slide_labels"][sid] for sid in slide_ids])
            probabilities = np.array(
                [test_metrics["slide_probabilities"][sid] for sid in slide_ids]
            )

            if len(np.unique(labels)) >= 2:
                fpr, tpr, _ = roc_curve(labels, probabilities)
                auc = test_metrics["auc"]
                plot_roc_curve(fpr, tpr, auc, str(output_dir / "roc_curve.png"))
                roc_curve_generated = True
    else:
        logger.warning("matplotlib/seaborn not available - skipping plot generation")

    log_evaluation_summary(
        checkpoint_path=args.checkpoint,
        epoch=epoch,
        num_slides=test_metrics["num_slides"],
        inference_time=inference_time,
        test_metrics=test_metrics,
        output_dir=output_dir,
        confusion_matrix_generated=confusion_matrix_generated,
        roc_curve_generated=roc_curve_generated,
    )


if __name__ == "__main__":
    main()
