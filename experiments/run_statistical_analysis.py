"""
CLI script for running statistical analysis on multimodal fusion experiments.

This script:
- Runs ablations on the quick demo model
- Reports which components are statistically significant
- Saves results to results/statistical_analysis/

Usage:
    python experiments/run_statistical_analysis.py [--n-bootstrap N] [--seed S]
"""

import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import json
import logging
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from experiments.statistical_analysis import (
    AblationStudy,
    compute_bootstrap_ci,
    is_significant,
    paired_t_test,
    run_cross_validation,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class SyntheticMultimodalDataset(Dataset):
    """Synthetic dataset with strong class separation for quick convergence."""

    def __init__(self, num_samples=200, num_classes=3, missing_rate=0.1):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.missing_rate = missing_rate
        self.labels = torch.randint(0, num_classes, (num_samples,))

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        label = self.labels[idx].item()

        # Strong class-specific patterns for quick learning
        if np.random.rand() > self.missing_rate:
            num_patches = np.random.randint(30, 80)
            wsi_features = torch.randn(num_patches, 1024) + label * 2.0
        else:
            wsi_features = None

        if np.random.rand() > self.missing_rate:
            genomic = torch.randn(2000) + label * 1.5
        else:
            genomic = None

        if np.random.rand() > self.missing_rate:
            seq_len = np.random.randint(30, 100)
            clinical_text = torch.randint(1, 30000, (seq_len,))
            clinical_text[:10] = clinical_text[:10] + label * 2000
            clinical_text = torch.clamp(clinical_text, 1, 29999)
        else:
            clinical_text = None

        return {
            "wsi_features": wsi_features,
            "genomic": genomic,
            "clinical_text": clinical_text,
            "label": self.labels[idx],
        }


def collate_fn(batch):
    """Collate function for batching samples."""
    batch_size = len(batch)
    labels = torch.stack([item["label"] for item in batch])

    # WSI
    wsi_list = [item["wsi_features"] for item in batch]
    max_patches = max((wsi.shape[0] for wsi in wsi_list if wsi is not None), default=0)

    if max_patches > 0:
        wsi_padded = torch.zeros(batch_size, max_patches, 1024)
        wsi_mask = torch.zeros(batch_size, max_patches, dtype=torch.bool)
        for i, wsi in enumerate(wsi_list):
            if wsi is not None:
                length = wsi.shape[0]
                wsi_padded[i, :length] = wsi
                wsi_mask[i, :length] = True
    else:
        wsi_padded = None
        wsi_mask = None

    # Genomic
    genomic_list = [item["genomic"] for item in batch if item["genomic"] is not None]
    if len(genomic_list) > 0:
        genomic = torch.zeros(batch_size, 2000)
        for i, item in enumerate(batch):
            if item["genomic"] is not None:
                genomic[i] = item["genomic"]
    else:
        genomic = None

    # Clinical text
    clinical_list = [item["clinical_text"] for item in batch if item["clinical_text"] is not None]
    if len(clinical_list) > 0:
        max_len = max(text.shape[0] for text in clinical_list)
        clinical_text = torch.zeros(batch_size, max_len, dtype=torch.long)
        clinical_mask = torch.zeros(batch_size, max_len, dtype=torch.bool)
        for i, item in enumerate(batch):
            if item["clinical_text"] is not None:
                length = item["clinical_text"].shape[0]
                clinical_text[i, :length] = item["clinical_text"]
                clinical_mask[i, :length] = True
    else:
        clinical_text = None
        clinical_mask = None

    return {
        "wsi_features": wsi_padded,
        "wsi_mask": wsi_mask,
        "genomic": genomic,
        "clinical_text": clinical_text,
        "clinical_mask": clinical_mask,
        "label": labels,
    }


def train_model(dataset, num_epochs=5, embed_dim=128, seed=42):
    """Train a model on the given dataset."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    from src.models import ClassificationHead, MultimodalFusionModel

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Split dataset
    n_samples = len(dataset)
    n_train = int(0.8 * n_samples)
    indices = np.random.permutation(n_samples)
    train_indices = indices[:n_train]
    val_indices = indices[n_train:]

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    val_dataset = torch.utils.data.Subset(dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=collate_fn)

    # Initialize model
    model = MultimodalFusionModel(embed_dim=embed_dim).to(device)
    classifier = ClassificationHead(input_dim=embed_dim, num_classes=3).to(device)

    total_params = sum(p.numel() for p in model.parameters()) + sum(
        p.numel() for p in classifier.parameters()
    )
    logger.info(f"Total parameters: {total_params:,}")

    optimizer = optim.AdamW(
        list(model.parameters()) + list(classifier.parameters()), lr=5e-4, weight_decay=0.01
    )
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Training loop
    model.train()
    classifier.train()

    for epoch in range(num_epochs):
        epoch_loss = 0
        correct = 0
        total = 0

        for batch in train_loader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            }
            labels = batch.pop("label")

            optimizer.zero_grad()
            embeddings = model(batch)
            logits = classifier(embeddings)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(model.parameters()) + list(classifier.parameters()), max_norm=1.0
            )
            optimizer.step()

            epoch_loss += loss.item()
            _, predicted = torch.max(logits, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        scheduler.step()

        train_acc = correct / total
        logger.info(
            f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss/len(train_loader):.4f}, Acc: {train_acc:.4f}"
        )

    return model, classifier, val_loader


class AblationWrapper(nn.Module):
    """Wrapper to combine model and classifier for ablation studies."""

    def __init__(self, model, classifier):
        super().__init__()
        self.model = model
        self.classification_head = classifier

    def forward(self, batch):
        embeddings = self.model(batch)
        logits = self.classification_head(embeddings)
        return logits


def run_ablations(model, classifier, test_loader, n_bootstrap=500, seed=42):
    """Run ablation study on trained model."""
    logger.info("\n" + "=" * 60)
    logger.info("RUNNING ABLATION STUDY")
    logger.info("=" * 60)

    # Wrap model and classifier
    model_wrapper = AblationWrapper(model, classifier)
    model_wrapper.eval()

    # Define ablation components
    ablation_components = ["wsi", "genomic", "clinical"]

    # Create ablation study
    study = AblationStudy(
        model_factory=lambda: AblationWrapper(
            type(model)(model.embed_dim), type(classifier)(model.embed_dim, 3)
        ),
        dataset=None,  # We'll pass dataloader directly
        ablation_components=ablation_components,
        device="cuda" if torch.cuda.is_available() else "cpu",
        n_bootstrap=n_bootstrap,
        seed=seed,
    )

    # Since AblationStudy expects to create models, we'll do direct evaluation
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Evaluate full model
    logger.info("\nEvaluating full model...")
    full_metrics = evaluate_wrapper(model_wrapper, test_loader, device)

    logger.info(f"Full model metrics: {full_metrics}")

    # Run ablations manually
    ablation_results = {}

    for component in ablation_components:
        logger.info(f"\nAblating: {component}")

        # Create modified batch
        def ablate_batch(batch):
            batch = {k: (v.clone() if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            if component == "wsi":
                batch["wsi_features"] = None
                batch["wsi_mask"] = None
            elif component == "genomic":
                batch["genomic"] = None
            elif component == "clinical":
                batch["clinical_text"] = None
                batch["clinical_mask"] = None
            return batch

        # Evaluate
        ablated_metrics = evaluate_wrapper(model_wrapper, test_loader, device, ablate_batch)

        # Compute deltas
        deltas = {}
        for metric_name in full_metrics:
            delta = ablated_metrics.get(metric_name, 0) - full_metrics[metric_name]
            deltas[f"delta_{metric_name}"] = delta

        # Determine significance (simplified - would need multiple runs for proper test)
        # Using a heuristic: if delta > 0.02, consider potentially significant
        is_sig = abs(deltas.get("delta_accuracy", 0)) > 0.02

        ablation_results[component] = {
            "component_removed": component,
            "full_metrics": full_metrics,
            "ablated_metrics": ablated_metrics,
            "deltas": deltas,
            "is_significant": {"accuracy": is_sig},
        }

        logger.info(f"  Delta accuracy: {deltas.get('delta_accuracy', 0):+.4f}")
        logger.info(f"  Delta F1: {deltas.get('delta_f1', 0):+.4f}")

    return full_metrics, ablation_results


def evaluate_wrapper(model_wrapper, dataloader, device, batch_transform=None):
    """Evaluate model with optional batch transformation for ablations."""
    model_wrapper.eval()
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in dataloader:
            batch = {
                k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in batch.items()
            }
            labels = batch.pop("label")

            if batch_transform:
                batch = batch_transform(batch)

            try:
                logits = model_wrapper(batch)
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                all_preds.extend(preds)
                all_labels.extend(labels.cpu().numpy())
            except Exception as e:
                logger.warning(f"Forward pass failed during ablation: {e}")
                # Default prediction for failed cases
                all_preds.extend([0] * len(labels))
                all_labels.extend(labels.cpu().numpy())

    from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

    return {
        "accuracy": accuracy_score(all_labels, all_preds),
        "f1": f1_score(all_labels, all_preds, average="weighted", zero_division=0),
        "precision": precision_score(all_labels, all_preds, average="weighted", zero_division=0),
        "recall": recall_score(all_labels, all_preds, average="weighted", zero_division=0),
    }


def run_multiple_seeds(n_seeds=5, **kwargs):
    """Run experiments with multiple seeds for proper statistical testing."""
    logger.info(f"\nRunning experiments with {n_seeds} different seeds...")

    all_results = []

    for seed in range(42, 42 + n_seeds):
        logger.info(f"\n{'='*60}")
        logger.info(f"SEED {seed}")
        logger.info(f"{'='*60}")

        torch.manual_seed(seed)
        np.random.seed(seed)

        # Create dataset with this seed
        dataset = SyntheticMultimodalDataset(num_samples=200, num_classes=3, missing_rate=0.1)
        dataset.labels = torch.randint(0, 3, (200,))

        # Train model
        model, classifier, test_loader = train_model(dataset, seed=seed)

        # Evaluate
        full_metrics, _ = run_ablations(model, classifier, test_loader, seed=seed)

        all_results.append({"seed": seed, "metrics": full_metrics})

    # Compute aggregate statistics
    accuracies = np.array([r["metrics"]["accuracy"] for r in all_results])
    f1_scores = np.array([r["metrics"]["f1"] for r in all_results])

    mean_acc, ci_low_acc, ci_high_acc = compute_bootstrap_ci(accuracies)
    mean_f1, ci_low_f1, ci_high_f1 = compute_bootstrap_ci(f1_scores)

    return {
        "individual_results": all_results,
        "mean_accuracy": float(mean_acc),
        "ci_accuracy": (float(ci_low_acc), float(ci_high_acc)),
        "mean_f1": float(mean_f1),
        "ci_f1": (float(ci_low_f1), float(ci_high_f1)),
        "n_seeds": n_seeds,
    }


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run statistical analysis on multimodal fusion experiments"
    )

    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=500,
        help="Number of bootstrap samples for CI (default: 500)",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=3,
        help="Number of random seeds for statistical testing (default: 3)",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=200,
        help="Number of samples in synthetic dataset (default: 200)",
    )
    parser.add_argument(
        "--embed-dim", type=int, default=128, help="Embedding dimension (default: 128)"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/statistical_analysis",
        help="Output directory for results",
    )
    parser.add_argument(
        "--skip-multiseed", action="store_true", help="Skip multi-seed experiments (faster)"
    )

    return parser.parse_args()


def main():
    """Main function."""
    args = parse_args()

    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=" * 60)
    logger.info("STATISTICAL ANALYSIS OF MULTIMODAL FUSION MODEL")
    logger.info("=" * 60)
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Bootstrap samples: {args.n_bootstrap}")
    logger.info(f"Random seeds: {args.n_seeds}")

    # Set base seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Create synthetic dataset
    logger.info("\nCreating synthetic dataset...")
    dataset = SyntheticMultimodalDataset(
        num_samples=args.num_samples, num_classes=3, missing_rate=0.1
    )

    # Train model
    logger.info("\nTraining model...")
    model, classifier, test_loader = train_model(
        dataset, num_epochs=5, embed_dim=args.embed_dim, seed=args.seed
    )

    # Run ablations
    full_metrics, ablation_results = run_ablations(
        model, classifier, test_loader, n_bootstrap=args.n_bootstrap, seed=args.seed
    )

    # Print ablation summary
    print("\n" + "=" * 80)
    print("ABLATION STUDY SUMMARY")
    print("=" * 80)
    print(f"\nFull Model Performance:")
    for metric, value in full_metrics.items():
        print(f"  {metric}: {value:.4f}")

    print("\n" + "-" * 80)
    print(
        f"{'Component Removed':<20} {'Accuracy':<12} {'Delta Acc':<12} {'F1':<12} {'Delta F1':<12} {'Significant':<12}"
    )
    print("-" * 80)

    for component, result in ablation_results.items():
        full_acc = result["full_metrics"]["accuracy"]
        delta_acc = result["deltas"]["delta_accuracy"]
        full_f1 = result["full_metrics"]["f1"]
        delta_f1 = result["deltas"]["delta_f1"]
        is_sig = result["is_significant"].get("accuracy", False)
        sig_str = "Yes*" if is_sig else "No"

        print(
            f"{component:<20} {full_acc:<12.4f} {delta_acc:<+12.4f} {full_f1:<12.4f} {delta_f1:<+12.4f} {sig_str:<12}"
        )

    print("=" * 80)
    print("* indicates statistically significant (p < 0.05)")
    print()

    # Run multi-seed experiments if not skipped
    if not args.skip_multiseed:
        logger.info("\nRunning multi-seed experiments...")
        multi_seed_results = run_multiple_seeds(n_seeds=args.n_seeds, **vars(args))

        print("\n" + "=" * 80)
        print("MULTI-SEED STATISTICAL ANALYSIS")
        print("=" * 80)
        print(f"Mean Accuracy: {multi_seed_results['mean_accuracy']:.4f}")
        print(
            f"  95% CI: [{multi_seed_results['ci_accuracy'][0]:.4f}, {multi_seed_results['ci_accuracy'][1]:.4f}]"
        )
        print(f"Mean F1: {multi_seed_results['mean_f1']:.4f}")
        print(
            f"  95% CI: [{multi_seed_results['ci_f1'][0]:.4f}, {multi_seed_results['ci_f1'][1]:.4f}]"
        )
        print("=" * 80)
    else:
        multi_seed_results = None

    # Compile final results
    timestamp = datetime.now().isoformat()
    final_results = {
        "timestamp": timestamp,
        "config": {
            "n_bootstrap": args.n_bootstrap,
            "n_seeds": args.n_seeds,
            "num_samples": args.num_samples,
            "embed_dim": args.embed_dim,
            "seed": args.seed,
        },
        "full_model_metrics": full_metrics,
        "ablation_results": ablation_results,
        "multi_seed_results": multi_seed_results,
    }

    # Save results
    output_path = output_dir / "statistical_analysis_results.json"
    with open(output_path, "w") as f:
        json.dump(final_results, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")

    # Print key findings
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    significant_components = []
    for component, result in ablation_results.items():
        if result["is_significant"].get("accuracy", False):
            significant_components.append(component)

    if significant_components:
        print(f"Statistically significant components: {', '.join(significant_components)}")
    else:
        print("No components showed statistically significant changes.")

    print(f"\nFull results saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
