#!/usr/bin/env python3
"""
PCam Heterogeneous Federated Benchmark

Tests weighting strategies under institutional heterogeneity:
- Class imbalance across sites
- Variable site sizes
- Label noise
- Domain shift simulation

Key question: Does FAIR-WEIGHTS-H maintain worst-site performance
while achieving competitive global accuracy?

Usage:
    python scripts/federated/run_pcam_heterogeneous_benchmark.py \\
        --weighting fair_weights_h --rounds 30 --seed 42
"""

import argparse
import json
import logging
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.nn as nn

from src.features.federated.pathology_fl.aggregator.fedavg import FedAvgAggregator
from src.features.federated.pathology_fl.client.trainer import LocalTrainer
from src.features.federated.pathology_fl.common.data_models import ClientUpdate
from src.features.federated.pathology_fl.coordinator.orchestrator import (
    TrainingOrchestrator,
)

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


class SimplePCamCNN(nn.Module):
    """Simple CNN for PCam patch classification."""

    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 24 * 24, 128)
        self.fc2 = nn.Linear(128, 2)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.pool(x)
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.reshape(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x


def load_pcam_data(data_dir: Path, split: str = "train", max_samples: int = None):
    """Load PCam data from .npy files."""
    images_path = data_dir / split / "images.npy"
    labels_path = data_dir / split / "labels.npy"

    if not images_path.exists() or not labels_path.exists():
        raise FileNotFoundError(
            f"PCam data not found at {data_dir}/{split}. "
            f"Expected images.npy and labels.npy"
        )

    logger.info(f"Loading PCam {split} data from {data_dir}/{split}")
    images = np.load(images_path, mmap_mode="r")
    labels = np.load(labels_path, mmap_mode="r")

    if max_samples is not None and len(images) > max_samples:
        logger.info(f"Using subset of {max_samples} samples")
        indices = np.random.choice(len(images), max_samples, replace=False)
        images = images[indices]
        labels = labels[indices]

    images = np.array(images)
    labels = np.array(labels)

    logger.info(f"Loaded {len(images)} images, shape {images.shape}")
    return images, labels


def split_heterogeneous_sites(
    images: np.ndarray, labels: np.ndarray, seed: int
) -> Dict:
    """
    Split data into 5 heterogeneous sites:
    
    Site 0: Balanced (50% pos), 1000 samples, clean
    Site 1: Pos-heavy (70% pos), 1000 samples
    Site 2: Neg-heavy (30% pos), 1000 samples  
    Site 3: Small volume (500 samples), balanced
    Site 4: Noisy labels (10% flipped), 1000 samples
    """
    rng = np.random.RandomState(seed)
    
    # Separate pos/neg samples
    pos_idx = np.where(labels == 1)[0]
    neg_idx = np.where(labels == 0)[0]
    
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    
    sites = {}
    pos_ptr = 0
    neg_ptr = 0
    
    # Site 0: Balanced, 1000 samples
    n_pos = 500
    n_neg = 500
    site_pos_idx = pos_idx[pos_ptr:pos_ptr + n_pos]
    site_neg_idx = neg_idx[neg_ptr:neg_ptr + n_neg]
    pos_ptr += n_pos
    neg_ptr += n_neg
    
    site_idx = np.concatenate([site_pos_idx, site_neg_idx])
    rng.shuffle(site_idx)
    
    sites[0] = {
        "images": torch.from_numpy(images[site_idx]).float() / 255.0,
        "labels": torch.from_numpy(labels[site_idx]).long().squeeze(),
        "description": "Balanced, clean"
    }
    
    # Site 1: Pos-heavy (70%), 1000 samples
    n_pos = 700
    n_neg = 300
    site_pos_idx = pos_idx[pos_ptr:pos_ptr + n_pos]
    site_neg_idx = neg_idx[neg_ptr:neg_ptr + n_neg]
    pos_ptr += n_pos
    neg_ptr += n_neg
    
    site_idx = np.concatenate([site_pos_idx, site_neg_idx])
    rng.shuffle(site_idx)
    
    sites[1] = {
        "images": torch.from_numpy(images[site_idx]).float() / 255.0,
        "labels": torch.from_numpy(labels[site_idx]).long().squeeze(),
        "description": "Pos-heavy (70%)"
    }
    
    # Site 2: Neg-heavy (30%), 1000 samples
    n_pos = 300
    n_neg = 700
    site_pos_idx = pos_idx[pos_ptr:pos_ptr + n_pos]
    site_neg_idx = neg_idx[neg_ptr:neg_ptr + n_neg]
    pos_ptr += n_pos
    neg_ptr += n_neg
    
    site_idx = np.concatenate([site_pos_idx, site_neg_idx])
    rng.shuffle(site_idx)
    
    sites[2] = {
        "images": torch.from_numpy(images[site_idx]).float() / 255.0,
        "labels": torch.from_numpy(labels[site_idx]).long().squeeze(),
        "description": "Neg-heavy (30%)"
    }
    
    # Site 3: Small volume, balanced, 500 samples
    n_pos = 250
    n_neg = 250
    site_pos_idx = pos_idx[pos_ptr:pos_ptr + n_pos]
    site_neg_idx = neg_idx[neg_ptr:neg_ptr + n_neg]
    pos_ptr += n_pos
    neg_ptr += n_neg
    
    site_idx = np.concatenate([site_pos_idx, site_neg_idx])
    rng.shuffle(site_idx)
    
    sites[3] = {
        "images": torch.from_numpy(images[site_idx]).float() / 255.0,
        "labels": torch.from_numpy(labels[site_idx]).long().squeeze(),
        "description": "Small volume (500)"
    }
    
    # Site 4: Noisy labels (10% flipped), 1000 samples
    n_pos = 500
    n_neg = 500
    site_pos_idx = pos_idx[pos_ptr:pos_ptr + n_pos]
    site_neg_idx = neg_idx[neg_ptr:neg_ptr + n_neg]
    pos_ptr += n_pos
    neg_ptr += n_neg
    
    site_idx = np.concatenate([site_pos_idx, site_neg_idx])
    rng.shuffle(site_idx)
    
    site_labels = labels[site_idx].copy()
    
    # Flip 10% of labels
    n_flip = int(0.1 * len(site_labels))
    flip_idx = rng.choice(len(site_labels), n_flip, replace=False)
    site_labels[flip_idx] = 1 - site_labels[flip_idx]
    
    sites[4] = {
        "images": torch.from_numpy(images[site_idx]).float() / 255.0,
        "labels": torch.from_numpy(site_labels).long().squeeze(),
        "description": "Noisy labels (10% flipped)"
    }
    
    # Log site stats
    for site_id, site_data in sites.items():
        pos_rate = site_data["labels"].float().mean().item()
        logger.info(
            f"Site {site_id}: {len(site_data['labels'])} samples, "
            f"pos_rate={pos_rate:.3f}, {site_data['description']}"
        )
    
    return sites


def compute_weight_entropy(weights: Dict[int, float]) -> float:
    """Normalized weight entropy."""
    K = len(weights)
    if K <= 1:
        return 0.0
    w = np.array(list(weights.values()))
    w = w / w.sum()
    entropy = -np.sum(w * np.log(w + 1e-10))
    return float(entropy / np.log(K))


def compute_n_eff(weights: Dict[int, float]) -> float:
    """Effective number of sites."""
    w = np.array(list(weights.values()))
    w = w / w.sum()
    return float(1.0 / np.sum(w**2))


def evaluate_site(model: nn.Module, images: torch.Tensor, labels: torch.Tensor):
    """Evaluate model accuracy on site."""
    model.eval()
    correct = 0
    total = 0
    batch_size = 32

    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch_images = images[i : i + batch_size]
            batch_labels = labels[i : i + batch_size]
            logits = model(batch_images)
            preds = logits.argmax(dim=1)
            correct += (preds == batch_labels).sum().item()
            total += len(batch_labels)

    return correct / total if total > 0 else 0.0


@dataclass
class BenchmarkResult:
    """Benchmark results."""
    strategy: str
    rounds: int
    num_sites: int
    seed: int
    final_weights: Dict[int, float]
    weight_entropy: float
    n_eff: float
    global_accuracy: float
    site_accuracies: Dict[int, float]
    worst_site_accuracy: float
    weight_trajectories: List[Dict[int, float]]
    accuracy_trajectories: List[float]


def run_benchmark(
    weighting_strategy: str,
    num_rounds: int,
    data_dir: Path,
    output_dir: Path,
    seed: int,
) -> BenchmarkResult:
    """Run heterogeneous benchmark."""
    logger.info(
        f"=== PCam Heterogeneous Benchmark: {weighting_strategy} "
        f"({num_rounds} rounds, seed={seed}) ==="
    )

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Load data
    logger.info("Loading PCam data...")
    images, labels = load_pcam_data(data_dir, split="train", max_samples=5000)
    
    # Split into heterogeneous sites
    logger.info("Creating heterogeneous sites...")
    sites = split_heterogeneous_sites(images, labels, seed)
    num_sites = len(sites)
    
    # Initialize
    global_model = SimplePCamCNN()
    checkpoint_dir = output_dir / f"checkpoints_{weighting_strategy}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    orchestrator = TrainingOrchestrator(
        global_model,
        checkpoint_dir=str(checkpoint_dir),
        min_clients_per_round=num_sites,
    )

    aggregator = FedAvgAggregator()

    clients = {}
    for site_id in range(num_sites):
        clients[site_id] = LocalTrainer(model=SimplePCamCNN(), device="cpu")

    logger.info(f"✓ Initialized {num_sites} clients")

    # Track trajectories
    weight_trajectories = []
    accuracy_trajectories = []

    # Training loop
    for round_num in range(1, num_rounds + 1):
        logger.info(f"\n--- Round {round_num}/{num_rounds} ---")

        client_ids = [f"site_{i}" for i in range(num_sites)]
        orchestrator.start_round(client_ids)
        global_state = orchestrator.get_global_model()

        client_updates = []
        site_metrics = {}

        for site_id, client in clients.items():
            client.load_global_model(global_state)

            site_data = sites[site_id]
            X = site_data["images"].permute(0, 3, 1, 2)
            y = site_data["labels"]

            client.set_data(X, y)
            _ = client.train_local_epochs(num_epochs=1)

            serialized = client.serialize_update()

            update = ClientUpdate(
                client_id=f"site_{site_id}",
                round_id=round_num,
                model_version=orchestrator.current_version,
                gradients=serialized["model_update"],
                dataset_size=serialized["metadata"]["dataset_size"],
                training_time_seconds=serialized["metadata"]["training_time"],
            )
            client_updates.append(update)

            site_acc = evaluate_site(client.model, X, y)
            site_metrics[site_id] = {"accuracy": site_acc}

        # Compute weights
        if weighting_strategy == "equal":
            weights = {i: 1.0 / num_sites for i in range(num_sites)}
        elif weighting_strategy == "volume":
            total_size = sum(u.dataset_size for u in client_updates)
            weights = {
                i: u.dataset_size / total_size for i, u in enumerate(client_updates)
            }
        elif weighting_strategy == "prestige":
            errors = [1.0 - site_metrics[i]["accuracy"] for i in range(num_sites)]
            inv_errors = [1.0 / (e + 0.01) for e in errors]
            total = sum(inv_errors)
            weights = {i: inv_errors[i] / total for i in range(num_sites)}
        elif weighting_strategy == "fair_weights_h":
            accs = [site_metrics[i]["accuracy"] for i in range(num_sites)]
            sizes = [client_updates[i].dataset_size for i in range(num_sites)]

            quality_scores = [a for a in accs]
            volume_scores = [np.log1p(s) for s in sizes]
            fairness_scores = [1.0 / (a + 0.1) for a in accs]

            combined_scores = [
                0.4 * q + 0.3 * v + 0.3 * f
                for q, v, f in zip(quality_scores, volume_scores, fairness_scores)
            ]

            exp_scores = [np.exp(s) for s in combined_scores]
            total = sum(exp_scores)
            weights = {i: exp_scores[i] / total for i in range(num_sites)}
        else:
            raise ValueError(f"Unknown strategy: {weighting_strategy}")

        weight_trajectories.append(weights.copy())

        # Aggregate
        aggregated_update = aggregator.aggregate(client_updates)
        orchestrator.update_global_model(aggregated_update)
        orchestrator.save_checkpoint()

        # Eval global accuracy
        all_images = torch.cat([sites[i]["images"] for i in range(num_sites)])
        all_labels = torch.cat([sites[i]["labels"] for i in range(num_sites)])
        all_images = all_images.permute(0, 3, 1, 2)
        global_acc = evaluate_site(global_model, all_images, all_labels)
        accuracy_trajectories.append(global_acc)

        orchestrator.complete_round({"global_accuracy": global_acc})

        if round_num % 5 == 0:
            logger.info(f"Round {round_num}: global_acc={global_acc:.3f}, weights={weights}")

    # Final eval
    final_site_accs = {}
    for site_id in range(num_sites):
        site_images = sites[site_id]["images"].permute(0, 3, 1, 2)
        final_site_accs[site_id] = evaluate_site(
            global_model, site_images, sites[site_id]["labels"]
        )

    worst_site_acc = min(final_site_accs.values())

    result = BenchmarkResult(
        strategy=weighting_strategy,
        rounds=num_rounds,
        num_sites=num_sites,
        seed=seed,
        final_weights=weight_trajectories[-1],
        weight_entropy=compute_weight_entropy(weight_trajectories[-1]),
        n_eff=compute_n_eff(weight_trajectories[-1]),
        global_accuracy=accuracy_trajectories[-1],
        site_accuracies=final_site_accs,
        worst_site_accuracy=worst_site_acc,
        weight_trajectories=weight_trajectories,
        accuracy_trajectories=accuracy_trajectories,
    )

    logger.info("\n✓ Benchmark complete")
    logger.info(f"  Global accuracy: {result.global_accuracy:.3f}")
    logger.info(f"  Worst-site accuracy: {result.worst_site_accuracy:.3f}")
    logger.info(f"  Weight entropy: {result.weight_entropy:.3f}")
    logger.info(f"  N_eff: {result.n_eff:.2f}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="PCam Heterogeneous Federated Benchmark"
    )
    parser.add_argument(
        "--weighting",
        type=str,
        required=True,
        choices=["equal", "volume", "prestige", "fair_weights_h"],
        help="Weighting strategy",
    )
    parser.add_argument("--rounds", type=int, default=30, help="Number of rounds")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/pcam_real"),
        help="PCam data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/pcam_heterogeneous_benchmark"),
        help="Output directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    result = run_benchmark(
        weighting_strategy=args.weighting,
        num_rounds=args.rounds,
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        seed=args.seed,
    )

    # Save
    result_dict = {
        "strategy": result.strategy,
        "rounds": result.rounds,
        "num_sites": result.num_sites,
        "seed": result.seed,
        "metrics": {
            "final_weights": result.final_weights,
            "weight_entropy": result.weight_entropy,
            "n_eff": result.n_eff,
            "global_accuracy": result.global_accuracy,
            "site_accuracies": result.site_accuracies,
            "worst_site_accuracy": result.worst_site_accuracy,
        },
        "trajectories": {
            "weights": result.weight_trajectories,
            "accuracy": result.accuracy_trajectories,
        },
    }

    output_file = args.output_dir / f"heterogeneous_{args.weighting}_seed{args.seed}.json"
    with open(output_file, "w") as f:
        json.dump(result_dict, f, indent=2)

    logger.info(f"\nResults saved to {output_file}")

    print("\n" + "=" * 60)
    print(f"HETEROGENEOUS BENCHMARK: {args.weighting}")
    print("=" * 60)
    print(f"Global accuracy:      {result.global_accuracy:.3f}")
    print(f"Worst-site accuracy:  {result.worst_site_accuracy:.3f}")
    print(f"Weight entropy:       {result.weight_entropy:.3f}")
    print(f"N_eff:                {result.n_eff:.2f}")
    print(f"\nSite accuracies:")
    for site_id, acc in result.site_accuracies.items():
        print(f"  Site {site_id}: {acc:.3f}")


if __name__ == "__main__":
    main()
