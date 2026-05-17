"""
Evaluate trained PANDA model and generate comprehensive results.

Usage:
    python experiments/evaluate_panda.py --checkpoint checkpoints/panda/best_model.pth --data_dir data/panda --features_dir data/panda/features --output_dir results/panda
"""

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    cohen_kappa_score,
    confusion_matrix,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data.panda_dataset import (
    PANDASlideDataset,
    PANDASlideIndex,
    collate_panda_bags,
    compute_quadratic_weighted_kappa,
)
from src.models.nnmil import nnMIL

logger = logging.getLogger(__name__)


def load_model(checkpoint_path: Path, device: torch.device):
    """Load trained model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    config = checkpoint['config']
    model = nnMIL(
        feature_dim=config['input_dim'],
        hidden_dim=config['hidden_dim'],
        num_classes=config['num_classes'],
        dropout=config.get('dropout', 0.2),
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info(f"Loaded model from {checkpoint_path}")
    logger.info(f"Epoch: {checkpoint['epoch']}, Val Kappa: {checkpoint.get('val_kappa', 'N/A'):.4f}")
    
    return model, config


def evaluate_model(model, dataloader, device, ordinal=False):
    """Evaluate model on dataset."""
    all_preds = []
    all_labels = []
    all_slide_ids = []
    all_probs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            features = batch["features"].to(device)
            labels = batch["labels"]
            num_patches = batch["num_patches"].to(device)
            
            logits = model(features, num_patches)
            
            if ordinal:
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).sum(dim=1)
                true_labels = labels.sum(dim=1).long()
                all_probs.append(probs.cpu().numpy())
            else:
                probs = torch.softmax(logits, dim=1)
                preds = logits.argmax(dim=1)
                true_labels = labels
                all_probs.append(probs.cpu().numpy())
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(true_labels.cpu().numpy())
            all_slide_ids.extend(batch["slide_ids"])
    
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.concatenate(all_probs, axis=0)
    
    return all_preds, all_labels, all_slide_ids, all_probs


def plot_confusion_matrix(y_true, y_pred, output_path, title="Confusion Matrix"):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm,
        annot=True,
        fmt='d',
        cmap='Blues',
        xticklabels=range(6),
        yticklabels=range(6),
    )
    plt.title(title)
    plt.ylabel('True ISUP Grade')
    plt.xlabel('Predicted ISUP Grade')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved confusion matrix to {output_path}")


def plot_grade_distribution(y_true, y_pred, output_path):
    """Plot grade distribution comparison."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    grades = range(6)
    true_counts = [np.sum(y_true == g) for g in grades]
    pred_counts = [np.sum(y_pred == g) for g in grades]
    
    x = np.arange(len(grades))
    width = 0.35
    
    ax1.bar(x - width/2, true_counts, width, label='True', alpha=0.8)
    ax1.bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.8)
    ax1.set_xlabel('ISUP Grade')
    ax1.set_ylabel('Count')
    ax1.set_title('Grade Distribution Comparison')
    ax1.set_xticks(x)
    ax1.set_xticklabels(grades)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Per-grade accuracy
    accuracies = []
    for g in grades:
        mask = y_true == g
        if mask.sum() > 0:
            acc = (y_pred[mask] == y_true[mask]).mean()
            accuracies.append(acc)
        else:
            accuracies.append(0)
    
    ax2.bar(grades, accuracies, alpha=0.8, color='green')
    ax2.set_xlabel('ISUP Grade')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Per-Grade Accuracy')
    ax2.set_ylim([0, 1])
    ax2.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Saved grade distribution to {output_path}")


def save_results(results, output_path):
    """Save evaluation results to JSON."""
    # Convert numpy types to Python types
    results_serializable = {}
    for key, value in results.items():
        if isinstance(value, np.ndarray):
            results_serializable[key] = value.tolist()
        elif isinstance(value, (np.integer, np.floating)):
            results_serializable[key] = value.item()
        else:
            results_serializable[key] = value
    
    with open(output_path, 'w') as f:
        json.dump(results_serializable, f, indent=2)
    
    logger.info(f"Saved results to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate PANDA model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--data_dir", type=str, default="data/panda", help="PANDA data directory")
    parser.add_argument("--features_dir", type=str, required=True, help="Features directory")
    parser.add_argument("--index_path", type=str, default="data/panda/slide_index.json", help="Slide index")
    parser.add_argument("--output_dir", type=str, default="results/panda", help="Output directory")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Split to evaluate")
    parser.add_argument("--batch_size", type=str, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    # Setup logging
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(output_dir / "evaluation.log"),
            logging.StreamHandler(),
        ],
    )
    
    device = torch.device(args.device)
    logger.info(f"Using device: {device}")
    
    # Load model
    checkpoint_path = Path(args.checkpoint)
    model, config = load_model(checkpoint_path, device)
    ordinal = config.get('ordinal', False)
    
    # Load dataset
    slide_index = PANDASlideIndex.load(args.index_path)
    dataset = PANDASlideDataset(
        slide_index=slide_index,
        features_dir=args.features_dir,
        split=args.split,
        ordinal=ordinal,
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_panda_bags,
    )
    
    logger.info(f"Evaluating on {len(dataset)} slides from {args.split} split")
    
    # Evaluate
    preds, labels, slide_ids, probs = evaluate_model(model, dataloader, device, ordinal)
    
    # Compute metrics
    accuracy = accuracy_score(labels, preds)
    kappa = compute_quadratic_weighted_kappa(labels, preds)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Evaluation Results ({args.split} split)")
    logger.info(f"{'='*70}")
    logger.info(f"Accuracy: {accuracy:.4f}")
    logger.info(f"Quadratic Weighted Kappa: {kappa:.4f}")
    
    # Per-grade metrics
    logger.info(f"\nPer-Grade Accuracy:")
    for grade in range(6):
        mask = labels == grade
        if mask.sum() > 0:
            grade_acc = (preds[mask] == labels[mask]).mean()
            logger.info(f"  Grade {grade}: {grade_acc:.4f} ({mask.sum()} samples)")
    
    # Classification report
    logger.info(f"\nClassification Report:")
    report = classification_report(labels, preds, target_names=[f"Grade {i}" for i in range(6)])
    logger.info(f"\n{report}")
    
    # Save results
    results = {
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "num_samples": len(dataset),
        "accuracy": float(accuracy),
        "quadratic_weighted_kappa": float(kappa),
        "predictions": preds.tolist(),
        "labels": labels.tolist(),
        "slide_ids": slide_ids,
        "probabilities": probs.tolist(),
    }
    
    save_results(results, output_dir / f"results_{args.split}.json")
    
    # Generate plots
    plot_confusion_matrix(
        labels,
        preds,
        output_dir / f"confusion_matrix_{args.split}.png",
        title=f"Confusion Matrix ({args.split} split)"
    )
    
    plot_grade_distribution(
        labels,
        preds,
        output_dir / f"grade_distribution_{args.split}.png"
    )
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Evaluation complete! Results saved to {output_dir}")
    logger.info(f"{'='*70}")


if __name__ == "__main__":
    main()
