"""
Evaluation script for trained PCam model.

This script loads a checkpoint from train_pcam.py and evaluates the model
on the test set, computing comprehensive metrics including accuracy, AUC,
precision, recall, F1, and generating confusion matrix and ROC curve plots.

Example usage:
    python evaluate_pcam.py --checkpoint checkpoints/pcam/best_model.pth
    python evaluate_pcam.py --checkpoint checkpoints/pcam/checkpoint_epoch_10.pth --batch-size 64
"""

import argparse
import json
import logging
import os
import sys
import time
from collections import OrderedDict
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

from experiments.generate_pcam_interpretability import (
    generate_pcam_interpretability_artifacts as build_pcam_interpretability_artifacts,
)

from typing import Optional, Dict, Any

try:
    import seaborn as sns
    import matplotlib.pyplot as plt

    PLOT_AVAILABLE = True
except ImportError:
    PLOT_AVAILABLE = False

from src.data.pcam_dataset import PCamDataset, get_pcam_transforms
from src.models.feature_extractors import ResNetFeatureExtractor
from src.models.encoders import WSIEncoder
from src.models.heads import ClassificationHead

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_checkpoint(
    checkpoint_path: str,
    feature_extractor: nn.Module,
    encoder: nn.Module,
    head: nn.Module,
    device: str,
) -> Tuple[Dict, Dict, int]:
    """Load checkpoint, return (config, metrics, epoch).

    Args:
        checkpoint_path: Path to the checkpoint file.
        feature_extractor: Feature extractor model to load state into.
        encoder: Encoder model to load state into.
        head: Classification head model to load state into.
        device: Device to map checkpoint tensors to.

    Returns:
        Tuple of (config, metrics, epoch).

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

        # Load state dicts
        feature_extractor.load_state_dict(checkpoint["feature_extractor_state_dict"])
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
        head.load_state_dict(checkpoint["head_state_dict"])

        config = checkpoint.get("config", {})
        metrics = checkpoint.get("metrics", {})
        epoch = checkpoint.get("epoch", 0)

        logger.info(f"Successfully loaded checkpoint from epoch {epoch}")
        logger.info(f"Checkpoint metrics: {metrics}")

        return config, metrics, epoch

    except KeyError as e:
        raise RuntimeError(
            f"Checkpoint is missing expected key: {e}\n"
            "The checkpoint may be corrupted or incompatible with this evaluation script."
        )
    except Exception as e:
        raise RuntimeError(f"Failed to load checkpoint: {e}")


def evaluate_model(
    feature_extractor: nn.Module,
    encoder: nn.Module,
    head: nn.Module,
    dataloader: DataLoader,
    device: str,
    return_predictions: bool = False,
) -> Dict[str, Any]:
    """Run inference and compute metrics.

    Args:
        feature_extractor: ResNet feature extractor model.
        encoder: WSI encoder model.
        head: Classification head model.
        dataloader: Test data loader.
        device: Device to run inference on.
        return_predictions: If True, return predictions and probabilities.

    Returns:
        Dictionary containing:
        - 'accuracy': float
        - 'auc': float
        - 'precision': float (macro)
        - 'recall': float (macro)
        - 'f1': float (macro)
        - 'confusion_matrix': np.ndarray [[TN, FP], [FN, TP]]
        - 'per_class_metrics': dict with precision/recall/f1 for each class
        - 'predictions': List[int] (if return_predictions=True)
        - 'probabilities': List[float] (if return_predictions=True)
        - 'labels': List[int] (if return_predictions=True)
    """
    feature_extractor.eval()
    encoder.eval()
    head.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    logger.info("Running inference on test set...")

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch["image"].to(device)
            labels = batch["label"].cpu().numpy()

            # Extract features
            features = feature_extractor(images)

            # Add sequence dimension: [batch, feature_dim] -> [batch, 1, feature_dim]
            features = features.unsqueeze(1)

            # Encode via WSI encoder
            encoded = encoder(features)

            # Classify
            logits = head(encoded)

            # Get predictions
            probs = torch.sigmoid(logits).cpu().numpy()
            preds = (probs > 0.5).astype(int)

            all_probs.extend(probs.flatten())
            all_preds.extend(preds.flatten())
            all_labels.extend(labels)

    # Convert to numpy arrays
    predictions = np.array(all_preds)
    probabilities = np.array(all_probs)
    labels = np.array(all_labels)

    # Compute metrics
    metrics = compute_metrics(predictions, probabilities, labels)

    if return_predictions:
        metrics["predictions"] = all_preds
        metrics["probabilities"] = all_probs
        metrics["labels"] = all_labels

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
    if np.unique(labels).size < 2:
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

    # Per-class metrics with fixed label order so tiny or degenerate splits
    # still produce a stable binary report shape.
    precision_per_class, recall_per_class, f1_per_class, _ = precision_recall_fscore_support(
        labels, predictions, labels=class_labels, average=None, zero_division=0
    )

    # Macro metrics over the fixed binary class set.
    precision_macro = float(np.mean(precision_per_class))
    recall_macro = float(np.mean(recall_per_class))
    f1_macro = float(np.mean(f1_per_class))

    # Binary metrics for the positive class (class 1).
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
        xticklabels=["Predicted 0", "Predicted 1"],
        yticklabels=["Actual 0", "Actual 1"],
        cbar_kws={"label": "Count"},
    )
    plt.title("Confusion Matrix")
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
    plt.title("Receiver Operating Characteristic (ROC) Curve")
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


def build_interpretability_metadata(
    args: argparse.Namespace, config: Dict[str, Any], output_dir: Path
) -> Dict[str, Any]:
    """Build reproducibility metadata for optional interpretability artifacts."""
    return {
        "checkpoint_path": args.checkpoint,
        "data_root": args.data_root,
        "split": "test",
        "batch_size": args.batch_size,
        "max_samples": args.interpretability_max_samples,
        "top_k": args.interpretability_top_k,
        "experiment_name": config.get("experiment", {}).get("name"),
        "evaluation_output_dir": str(output_dir),
    }


def maybe_generate_interpretability_artifacts(
    args: argparse.Namespace,
    config: Dict[str, Any],
    feature_extractor: nn.Module,
    encoder: nn.Module,
    head: nn.Module,
    dataloader: DataLoader,
    output_dir: Path,
    device: torch.device,
) -> Optional[Dict[str, Any]]:
    """Optionally generate interpretability artifacts alongside evaluation outputs."""
    if not args.generate_interpretability:
        return None

    metadata = build_interpretability_metadata(args, config, output_dir)
    interpretability_output_dir = (
        Path(args.interpretability_output_dir)
        if args.interpretability_output_dir is not None
        else output_dir / "interpretability"
    )

    logger.info("Generating interpretability artifacts...")
    try:
        summary = build_pcam_interpretability_artifacts(
            feature_extractor=feature_extractor,
            encoder=encoder,
            head=head,
            dataloader=dataloader,
            output_dir=str(interpretability_output_dir),
            device=device,
            max_samples=args.interpretability_max_samples,
            top_k=args.interpretability_top_k,
            metadata=metadata,
        )
        summary["status"] = "success"
        logger.info("Interpretability summary saved to: %s", summary["summary_path"])
        return summary
    except Exception as exc:
        logger.warning("Interpretability artifact generation failed: %s", exc)
        return {
            "status": "failed",
            "error": str(exc),
            "output_dir": str(interpretability_output_dir),
            "metadata": metadata,
        }


def log_evaluation_summary(
    checkpoint_path: str,
    epoch: int,
    test_dataset_size: int,
    inference_time: float,
    test_metrics: Dict[str, Any],
    output_dir: Path,
    confusion_matrix_generated: bool,
    roc_curve_generated: bool,
    interpretability_summary: Optional[Dict[str, Any]],
) -> None:
    """Log the final evaluation summary in a plotting-independent way."""
    cm = np.array(test_metrics["confusion_matrix"])

    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 60)
    logger.info(f"Checkpoint: {checkpoint_path}")
    logger.info(f"Checkpoint epoch: {epoch}")
    logger.info(f"Test samples: {test_dataset_size}")
    logger.info(f"Inference time: {inference_time:.2f} seconds")
    logger.info("-" * 60)
    logger.info("Test Metrics:")
    logger.info(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
    if test_metrics["auc"] is None:
        logger.info("  AUC:       undefined (single-class labels)")
    else:
        logger.info(f"  AUC:       {test_metrics['auc']:.4f}")
    logger.info(f"  Precision: {test_metrics['precision']:.4f}")
    logger.info(f"  Recall:    {test_metrics['recall']:.4f}")
    logger.info(f"  F1:        {test_metrics['f1']:.4f}")
    logger.info("-" * 60)
    logger.info("Per-class metrics (class 0 / class 1):")
    for cls in ["class_0", "class_1"]:
        cls_metrics = test_metrics["per_class_metrics"][cls]
        logger.info(f"  {cls}:")
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
    if interpretability_summary is not None:
        if interpretability_summary["status"] == "success":
            logger.info(f"  Interpretability summary: {interpretability_summary['summary_path']}")
            logger.info(f"  Interpretability report: {interpretability_summary['report_path']}")
        else:
            logger.info("  Interpretability: failed (%s)", interpretability_summary["error"])
    logger.info("=" * 60)


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate PCam model on test set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python experiments/evaluate_pcam.py --checkpoint checkpoints/pcam/best_model.pth
  python experiments/evaluate_pcam.py --checkpoint checkpoints/pcam/best_model.pth --batch-size 128
  python experiments/evaluate_pcam.py --checkpoint checkpoints/pcam/best_model.pth --output-dir results/pcam
  python experiments/evaluate_pcam.py --checkpoint checkpoints/pcam/best_model.pth --generate-interpretability
        """,
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to the checkpoint file (required)"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="data/pcam",
        help="Root directory of PCam dataset (default: data/pcam)",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="Batch size for evaluation (default: 64)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/pcam",
        help="Output directory for results (default: results/pcam)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of data loading workers (default: 0 for Windows/HDF5 compatibility)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use for evaluation (default: cuda if available)",
    )
    parser.add_argument(
        "--generate-interpretability",
        action="store_true",
        help="Generate PCam interpretability artifacts alongside evaluation results",
    )
    parser.add_argument(
        "--interpretability-output-dir",
        type=str,
        default=None,
        help="Optional output directory for interpretability artifacts (default: <output-dir>/interpretability)",
    )
    parser.add_argument(
        "--interpretability-max-samples",
        type=int,
        default=128,
        help="Maximum number of samples to use for interpretability artifacts (default: 128)",
    )
    parser.add_argument(
        "--interpretability-top-k",
        type=int,
        default=20,
        help="Number of top feature saliency entries to save (default: 20)",
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
    logger.info("PCam Model Evaluation")
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
        # First, load checkpoint to get config
        logger.info(f"Loading checkpoint from: {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=str(device))

        if "config" not in checkpoint:
            raise RuntimeError(
                "Checkpoint does not contain config. Cannot determine model architecture."
            )

        config = checkpoint["config"]
        checkpoint_metrics = checkpoint.get("metrics", {})
        epoch = checkpoint.get("epoch", 0)

        logger.info(f"Successfully loaded checkpoint from epoch {epoch}")
        logger.info(f"Checkpoint metrics: {checkpoint_metrics}")

        # Now create models with correct dimensions from config
        feature_extractor_config = config["model"]["feature_extractor"]
        feature_extractor = ResNetFeatureExtractor(
            model_name=feature_extractor_config["model"],
            pretrained=False,
            feature_dim=feature_extractor_config.get("feature_dim", 512),
        )

        wsi_config = config["model"]["wsi"]
        encoder = WSIEncoder(
            input_dim=wsi_config["input_dim"],
            hidden_dim=wsi_config["hidden_dim"],
            output_dim=config["model"]["embed_dim"],
            num_heads=wsi_config["num_heads"],
            num_layers=wsi_config["num_layers"],
            pooling=wsi_config.get("pooling", "mean"),
            dropout=config["training"].get("dropout", 0.1),
        )

        classification_config = config["task"]["classification"]
        hidden_dims = classification_config.get("hidden_dims", [128])
        use_hidden_layer = len(hidden_dims) > 0
        hidden_dim = hidden_dims[0] if use_hidden_layer else 128

        head = ClassificationHead(
            input_dim=config["model"]["embed_dim"],
            hidden_dim=hidden_dim,
            num_classes=1,
            dropout=classification_config["dropout"],
            use_hidden_layer=use_hidden_layer,
        )

        # Load state dicts
        feature_extractor.load_state_dict(checkpoint["feature_extractor_state_dict"])
        encoder.load_state_dict(checkpoint["encoder_state_dict"])
        head.load_state_dict(checkpoint["head_state_dict"])

        # Move to device
        feature_extractor = feature_extractor.to(device)
        encoder = encoder.to(device)
        head = head.to(device)

    except (FileNotFoundError, RuntimeError, KeyError) as e:
        logger.error(f"Failed to load checkpoint: {e}")
        sys.exit(1)

    # Create test dataloader
    logger.info("Creating test dataloader...")
    test_transform = get_pcam_transforms(split="test", augmentation=False)

    try:
        test_dataset = PCamDataset(
            root_dir=args.data_root,
            split="test",
            transform=test_transform,
            download=False,
        )
    except RuntimeError as e:
        logger.error(f"Failed to load test dataset: {e}")
        logger.error(f"Please ensure the dataset exists at: {args.data_root}")
        sys.exit(1)

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    logger.info(f"Test dataset size: {len(test_dataset)}")

    # Count model parameters
    total_params_fe = count_model_parameters(feature_extractor)
    total_params_enc = count_model_parameters(encoder)
    total_params_head = count_model_parameters(head)
    total_params = total_params_fe + total_params_enc + total_params_head

    logger.info(f"Feature extractor parameters: {total_params_fe:,}")
    logger.info(f"Encoder parameters: {total_params_enc:,}")
    logger.info(f"Classification head parameters: {total_params_head:,}")
    logger.info(f"Total model parameters: {total_params:,}")

    # Run evaluation
    logger.info("Starting evaluation...")
    start_time = time.time()

    test_metrics = evaluate_model(
        feature_extractor,
        encoder,
        head,
        test_loader,
        device,
        return_predictions=True,
    )

    inference_time = time.time() - start_time
    samples_per_second = len(test_dataset) / inference_time

    logger.info(f"Evaluation completed in {inference_time:.2f} seconds")
    logger.info(f"Throughput: {samples_per_second:.2f} samples/second")

    # Add metadata to metrics
    test_metrics["hardware_info"] = hw_info
    test_metrics["inference_time_seconds"] = inference_time
    test_metrics["samples_per_second"] = samples_per_second
    test_metrics["model_parameters"] = {
        "feature_extractor": total_params_fe,
        "encoder": total_params_enc,
        "head": total_params_head,
        "total": total_params,
    }
    test_metrics["checkpoint_epoch"] = epoch
    test_metrics["checkpoint_path"] = args.checkpoint
    test_metrics["checkpoint_metrics"] = checkpoint_metrics

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    interpretability_summary = maybe_generate_interpretability_artifacts(
        args=args,
        config=config,
        feature_extractor=feature_extractor,
        encoder=encoder,
        head=head,
        dataloader=test_loader,
        output_dir=output_dir,
        device=device,
    )
    if interpretability_summary is not None:
        test_metrics["interpretability"] = {
            "status": interpretability_summary["status"],
            "metadata": interpretability_summary.get("metadata", {}),
        }
        if interpretability_summary["status"] == "success":
            test_metrics["interpretability"].update(
                {
                    "summary_path": interpretability_summary["summary_path"],
                    "report_path": interpretability_summary["report_path"],
                    "artifacts": interpretability_summary["artifacts"],
                }
            )
        else:
            test_metrics["interpretability"].update(
                {
                    "error": interpretability_summary["error"],
                    "output_dir": interpretability_summary["output_dir"],
                }
            )

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
        labels = np.array(test_metrics["labels"])
        probabilities = np.array(test_metrics["probabilities"])
        if test_metrics["auc"] is None or len(np.unique(labels)) < 2:
            logger.warning("Skipping ROC curve plot because ROC AUC is undefined for this split")
        else:
            fpr, tpr, _ = roc_curve(labels, probabilities)
            auc = test_metrics["auc"]
            plot_roc_curve(fpr, tpr, auc, str(output_dir / "roc_curve.png"))
            roc_curve_generated = True
    else:
        logger.warning("matplotlib/seaborn not available - skipping plot generation")

    log_evaluation_summary(
        checkpoint_path=args.checkpoint,
        epoch=epoch,
        test_dataset_size=len(test_dataset),
        inference_time=inference_time,
        test_metrics=test_metrics,
        output_dir=output_dir,
        confusion_matrix_generated=confusion_matrix_generated,
        roc_curve_generated=roc_curve_generated,
        interpretability_summary=interpretability_summary,
    )


if __name__ == "__main__":
    main()
