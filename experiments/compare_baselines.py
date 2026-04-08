"""
Baseline comparison script for evaluating multimodal fusion models.

This script compares the performance of MultimodalFusionModel against
baseline architectures on synthetic data to understand the contribution
of different model components.

Baselines tested:
1. SingleModalityModel - Individual modality performance (WSI, genomic, clinical)
2. LateFusionModel - Simple concatenation without cross-attention
3. AttentionBaseline - Self-attention only, no cross-modal interaction

Usage:
    python experiments/compare_baselines.py
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from tqdm import tqdm

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

from src.models import (
    AttentionBaseline,
    ClassificationHead,
    LateFusionModel,
    MultimodalFusionModel,
    SingleModalityModel,
)
from src.models.baselines import get_baseline_model

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def set_seed(seed: int = 42):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_synthetic_data(
    batch_size: int = 32,
    num_samples: int = 500,
    num_classes: int = 4,
    embed_dim: int = 256,
    device: str = "cuda",
) -> List[Dict]:
    """
    Generate synthetic multimodal data for baseline comparison.

    Creates realistic synthetic inputs matching the format expected
    by the models with appropriate shapes and value ranges.

    Args:
        batch_size: Number of samples per batch
        num_samples: Total number of samples to generate
        num_classes: Number of classes for classification
        embed_dim: Embedding dimension
        device: Device to create tensors on

    Returns:
        List of data batches
    """
    num_batches = (num_samples + batch_size - 1) // batch_size
    batches = []

    for _ in range(num_batches):
        batch = {
            "wsi_features": torch.randn(batch_size, 100, 1024, device=device),
            "wsi_mask": torch.ones(batch_size, 100, dtype=torch.bool, device=device),
            "genomic": torch.randn(batch_size, 2000, device=device),
            "clinical_text": torch.randint(0, 30000, (batch_size, 128), device=device),
            "clinical_mask": torch.ones(batch_size, 128, dtype=torch.bool, device=device),
            "label": torch.randint(0, num_classes, (batch_size,), device=device),
        }
        batches.append(batch)

    return batches


def evaluate_model(
    model: nn.Module, task_head: nn.Module, batches: List[Dict], device: str = "cuda"
) -> Dict[str, float]:
    """
    Evaluate a model on given batches and compute metrics.

    Args:
        model: The model to evaluate
        task_head: The classification head
        batches: List of data batches
        device: Device to run on

    Returns:
        Dictionary of computed metrics
    """
    model.eval()
    task_head.eval()

    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in batches:
            # Move batch to device
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            }
            labels = batch.pop("label")

            # Forward pass
            embeddings = model(batch)
            logits = task_head(embeddings)

            # Get predictions
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Compute metrics
    metrics = {
        "accuracy": accuracy_score(all_labels, all_preds),
        "precision": precision_score(all_labels, all_preds, average="weighted", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="weighted", zero_division=0),
        "f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        "macro_f1": f1_score(all_labels, all_preds, average="macro", zero_division=0),
    }

    # Per-class metrics
    num_classes = len(np.unique(all_labels))
    for i in range(num_classes):
        class_mask = all_labels == i
        if class_mask.sum() > 0:
            class_acc = accuracy_score(all_labels[class_mask], all_preds[class_mask])
            class_f1 = f1_score(
                all_labels[class_mask], all_preds[class_mask], average="binary", zero_division=0
            )
            metrics[f"class_{i}_accuracy"] = class_acc
            metrics[f"class_{i}_f1"] = class_f1

    return metrics


def print_metrics_table(results: Dict[str, Dict[str, float]], metrics_to_show: List[str] = None):
    """Print a formatted table of metrics for all models."""
    if metrics_to_show is None:
        # Collect all metrics
        all_metrics = set()
        for model_results in results.values():
            all_metrics.update(model_results.keys())
        all_metrics = sorted([m for m in all_metrics if m.startswith("class_")])
        metrics_to_show = ["accuracy", "precision", "recall", "f1", "macro_f1"] + all_metrics

    # Header
    print("\n" + "=" * 100)
    print(
        f"{'Model':<30} | {'Accuracy':>10} | {'Precision':>10} | {'Recall':>10} | {'F1':>10} | {'Macro F1':>10}"
    )
    print("-" * 100)

    # Rows
    for model_name, metrics in results.items():
        row = f"{model_name:<30} | "
        row += f"{metrics.get('accuracy', 0):>10.4f} | "
        row += f"{metrics.get('precision', 0):>10.4f} | "
        row += f"{metrics.get('recall', 0):>10.4f} | "
        row += f"{metrics.get('f1', 0):>10.4f} | "
        row += f"{metrics.get('macro_f1', 0):>10.4f}"
        print(row)

    print("=" * 100 + "\n")


def print_detailed_results(results: Dict[str, Dict[str, float]]):
    """Print detailed results for each model including per-class metrics."""
    for model_name, metrics in results.items():
        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(f"{'='*60}")

        # Overall metrics
        print("\nOverall Metrics:")
        for key in ["accuracy", "precision", "recall", "f1", "macro_f1"]:
            if key in metrics:
                print(f"  {key}: {metrics[key]:.4f}")

        # Per-class metrics
        print("\nPer-Class Metrics:")
        class_keys = [k for k in metrics.keys() if k.startswith("class_")]
        if class_keys:
            for key in sorted(class_keys):
                print(f"  {key}: {metrics[key]:.4f}")


def save_results(results: Dict[str, Dict[str, float]], output_dir: Path):
    """Save comparison results to JSON file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full results
    results_path = output_dir / "baseline_comparison.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Saved results to {results_path}")

    # Save summary table as CSV
    csv_path = output_dir / "baseline_summary.csv"
    with open(csv_path, "w") as f:
        f.write("model,accuracy,precision,recall,f1,macro_f1\n")
        for model_name, metrics in results.items():
            f.write(
                f"{model_name},{metrics.get('accuracy', 0):.4f},"
                f"{metrics.get('precision', 0):.4f},"
                f"{metrics.get('recall', 0):.4f},"
                f"{metrics.get('f1', 0):.4f},"
                f"{metrics.get('macro_f1', 0):.4f}\n"
            )
    logger.info(f"Saved CSV summary to {csv_path}")


def run_baseline_comparison(
    config: Optional[Dict] = None,
    output_dir: str = "results/baseline_comparison",
    num_samples: int = 500,
    batch_size: int = 32,
    embed_dim: int = 256,
    num_classes: int = 4,
    device: str = "cuda",
) -> Dict[str, Dict[str, float]]:
    """
    Run complete baseline comparison experiment.

    Args:
        config: Model configuration dictionary
        output_dir: Directory to save results
        num_samples: Number of synthetic samples to evaluate
        batch_size: Batch size for evaluation
        embed_dim: Embedding dimension
        num_classes: Number of classes
        device: Device to run on

    Returns:
        Dictionary mapping model names to their metrics
    """
    logger.info("=" * 60)
    logger.info("Starting Baseline Comparison Experiment")
    logger.info("=" * 60)
    logger.info(
        f"Configuration: {num_samples} samples, batch_size={batch_size}, "
        f"embed_dim={embed_dim}, num_classes={num_classes}"
    )

    # Generate synthetic data
    logger.info("Generating synthetic data...")
    batches = generate_synthetic_data(
        batch_size=batch_size,
        num_samples=num_samples,
        num_classes=num_classes,
        embed_dim=embed_dim,
        device=device,
    )
    logger.info(f"Generated {len(batches)} batches")

    # Initialize results dictionary
    results = {}

    # =========================================================================
    # 1. MultimodalFusionModel (full model)
    # =========================================================================
    logger.info("\n" + "-" * 60)
    logger.info("Evaluating: MultimodalFusionModel (Full)")
    logger.info("-" * 60)

    model = MultimodalFusionModel(embed_dim=embed_dim, **(config or {})).to(device)
    task_head = ClassificationHead(input_dim=embed_dim, num_classes=num_classes).to(device)

    metrics = evaluate_model(model, task_head, batches, device)
    results["MultimodalFusionModel"] = metrics
    logger.info(f"Results: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")

    # =========================================================================
    # 2. SingleModalityModel baselines
    # =========================================================================
    for modality in ["wsi", "genomic", "clinical"]:
        logger.info("\n" + "-" * 60)
        logger.info(f"Evaluating: SingleModalityModel ({modality})")
        logger.info("-" * 60)

        model = SingleModalityModel(modality=modality, config=config, embed_dim=embed_dim).to(
            device
        )
        task_head = ClassificationHead(input_dim=embed_dim, num_classes=num_classes).to(device)

        metrics = evaluate_model(model, task_head, batches, device)
        results[f"SingleModality_{modality}"] = metrics
        logger.info(f"Results: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")

    # =========================================================================
    # 3. LateFusionModel
    # =========================================================================
    logger.info("\n" + "-" * 60)
    logger.info("Evaluating: LateFusionModel")
    logger.info("-" * 60)

    model = LateFusionModel(config=config, embed_dim=embed_dim).to(device)
    task_head = ClassificationHead(input_dim=embed_dim, num_classes=num_classes).to(device)

    metrics = evaluate_model(model, task_head, batches, device)
    results["LateFusionModel"] = metrics
    logger.info(f"Results: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")

    # =========================================================================
    # 4. AttentionBaseline
    # =========================================================================
    logger.info("\n" + "-" * 60)
    logger.info("Evaluating: AttentionBaseline")
    logger.info("-" * 60)

    model = AttentionBaseline(config=config, embed_dim=embed_dim).to(device)
    task_head = ClassificationHead(input_dim=embed_dim, num_classes=num_classes).to(device)

    metrics = evaluate_model(model, task_head, batches, device)
    results["AttentionBaseline"] = metrics
    logger.info(f"Results: Accuracy={metrics['accuracy']:.4f}, F1={metrics['f1']:.4f}")

    # =========================================================================
    # Save and summarize results
    # =========================================================================
    logger.info("\n" + "=" * 60)
    logger.info("Baseline Comparison Complete")
    logger.info("=" * 60)

    # Print summary table
    print_metrics_table(results)

    # Print detailed results
    print_detailed_results(results)

    # Save results to file
    output_path = Path(output_dir)
    save_results(results, output_path)

    # Calculate and print improvement summary
    full_model_acc = results["MultimodalFusionModel"]["accuracy"]
    full_model_f1 = results["MultimodalFusionModel"]["f1"]

    print("\n" + "=" * 60)
    print("IMPROVEMENT SUMMARY (vs MultimodalFusionModel)")
    print("=" * 60)
    print(f"\nFull Model: Accuracy={full_model_acc:.4f}, F1={full_model_f1:.4f}")
    print("\nImprovement over baselines:")

    for model_name, metrics in results.items():
        if model_name != "MultimodalFusionModel":
            acc_diff = metrics["accuracy"] - full_model_acc
            f1_diff = metrics["f1"] - full_model_f1
            print(f"  {model_name}: Accuracy={acc_diff:+.4f}, F1={f1_diff:+.4f}")

    print("\n" + "=" * 60)
    print(f"Results saved to: {output_dir}")
    print("=" * 60)

    return results


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare baseline models with MultimodalFusionModel"
    )

    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/baseline_comparison",
        help="Directory to save results",
    )
    parser.add_argument(
        "--num-samples", type=int, default=500, help="Number of synthetic samples to evaluate"
    )
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--embed-dim", type=int, default=256, help="Embedding dimension")
    parser.add_argument(
        "--num-classes", type=int, default=4, help="Number of classes for classification"
    )
    parser.add_argument("--device", type=str, default="cuda", help="Device to run on")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Set random seed
    set_seed(args.seed)

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Run comparison
    results = run_baseline_comparison(
        output_dir=args.output_dir,
        num_samples=args.num_samples,
        batch_size=args.batch_size,
        embed_dim=args.embed_dim,
        num_classes=args.num_classes,
        device=device,
    )
