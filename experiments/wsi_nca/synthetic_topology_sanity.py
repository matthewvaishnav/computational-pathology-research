#!/usr/bin/env python3
"""Topology-only mechanistic sanity check for WSI-NCA.

Every synthetic slide contains exactly eight type-A and eight type-B cells, so a
coordinate-blind permutation-invariant bag model receives the same multiset for
both classes. The target depends only on spatial organization:

- class 0: checkerboard / low local homophily;
- class 1: spatial split / high local homophily.

This is not pathology evidence. It asks whether the implementation can exploit
feature-topology correspondence at all, and whether that signal disappears when
coordinates are reassigned to cells.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import Tensor

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.models.wsi_nca import WSINCA


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WSI-NCA synthetic topology sanity check")
    parser.add_argument("--train-samples", type=int, default=512)
    parser.add_argument("--val-samples", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--hidden-dim", type=int, default=24)
    parser.add_argument("--k-neighbors", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-3)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--out", default=None)
    return parser.parse_args()


def make_dataset(num_samples: int, seed: int) -> Tuple[Tensor, Tensor, Tensor]:
    if num_samples % 2 != 0:
        raise ValueError("num_samples must be even for an exactly balanced dataset")

    rng = np.random.default_rng(seed)
    grid = torch.tensor([[x, y] for y in range(4) for x in range(4)], dtype=torch.float32)
    features = []
    coordinates = []
    labels = []

    for sample_index in range(num_samples):
        label = sample_index % 2
        if label == 1:
            # Two compact tissue compartments; orientation is randomized.
            if int(rng.integers(0, 2)) == 0:
                cell_types = torch.tensor(
                    [0 if x < 2 else 1 for y in range(4) for x in range(4)]
                )
            else:
                cell_types = torch.tensor(
                    [0 if y < 2 else 1 for y in range(4) for x in range(4)]
                )
        else:
            phase = int(rng.integers(0, 2))
            cell_types = torch.tensor(
                [(x + y + phase) % 2 for y in range(4) for x in range(4)]
            )

        # Make type identity itself non-predictive.
        if rng.random() < 0.5:
            cell_types = 1 - cell_types

        sample_coordinates = grid.clone()
        sample_coordinates += torch.tensor(
            [int(rng.integers(-10, 11)), int(rng.integers(-10, 11))],
            dtype=torch.float32,
        )

        features.append(torch.nn.functional.one_hot(cell_types, num_classes=2).float())
        coordinates.append(sample_coordinates)
        labels.append(label)

    permutation = torch.tensor(rng.permutation(num_samples), dtype=torch.long)
    return (
        torch.stack(features)[permutation],
        torch.stack(coordinates)[permutation],
        torch.tensor(labels, dtype=torch.long)[permutation],
    )


def reassign_coordinates(coordinates: Tensor, seed: int) -> Tensor:
    """Break cell-feature/coordinate correspondence within every synthetic slide."""
    rng = np.random.default_rng(seed)
    reassigned = coordinates.clone()
    for sample_index in range(coordinates.shape[0]):
        permutation = torch.tensor(rng.permutation(coordinates.shape[1]), dtype=torch.long)
        reassigned[sample_index] = reassigned[sample_index, permutation]
    return reassigned


@torch.no_grad()
def accuracy(model: WSINCA, features: Tensor, coordinates: Tensor, labels: Tensor) -> float:
    model.eval()
    logits = model(features, coordinates).logits
    return float((logits.argmax(dim=1) == labels).float().mean().item())


def train_model(
    train_features: Tensor,
    train_coordinates: Tensor,
    train_labels: Tensor,
    val_features: Tensor,
    val_coordinates: Tensor,
    val_labels: Tensor,
    *,
    num_steps: int,
    hidden_dim: int,
    k_neighbors: int,
    lr: float,
    epochs: int,
    seed: int,
    device: torch.device,
) -> Tuple[WSINCA, float]:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    model = WSINCA(
        input_dim=2,
        hidden_dim=hidden_dim,
        num_classes=2,
        num_steps=num_steps,
        k_neighbors=k_neighbors,
        dynamics_mode="tied",
        dropout=0.0,
    ).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    train_features = train_features.to(device)
    train_coordinates = train_coordinates.to(device)
    train_labels = train_labels.to(device)
    val_features = val_features.to(device)
    val_coordinates = val_coordinates.to(device)
    val_labels = val_labels.to(device)

    batch_size = 64
    for _ in range(epochs):
        model.train()
        order = torch.randperm(train_labels.shape[0], device=device)
        for start in range(0, order.numel(), batch_size):
            batch_index = order[start : start + batch_size]
            logits = model(
                train_features[batch_index],
                train_coordinates[batch_index],
            ).logits
            loss = torch.nn.functional.cross_entropy(logits, train_labels[batch_index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    return model, accuracy(model, val_features, val_coordinates, val_labels)


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    train_features, train_coordinates, train_labels = make_dataset(args.train_samples, 123)
    val_features, val_coordinates, val_labels = make_dataset(args.val_samples, 456)

    results: Dict[str, float] = {}
    trained_models: Dict[int, WSINCA] = {}
    for steps in (0, 1, 2, 4):
        model, val_accuracy = train_model(
            train_features,
            train_coordinates,
            train_labels,
            val_features,
            val_coordinates,
            val_labels,
            num_steps=steps,
            hidden_dim=args.hidden_dim,
            k_neighbors=args.k_neighbors,
            lr=args.lr,
            epochs=args.epochs,
            seed=args.seed,
            device=device,
        )
        trained_models[steps] = model
        results[f"t{steps}_true_coordinates"] = val_accuracy

    shuffled_val_coordinates = reassign_coordinates(val_coordinates, 200).to(device)
    results["t4_true_train_shuffled_eval"] = accuracy(
        trained_models[4],
        val_features.to(device),
        shuffled_val_coordinates,
        val_labels.to(device),
    )

    shuffled_train_coordinates = reassign_coordinates(train_coordinates, 100)
    shuffled_model, shuffled_accuracy = train_model(
        train_features,
        shuffled_train_coordinates,
        train_labels,
        val_features,
        reassign_coordinates(val_coordinates, 200),
        val_labels,
        num_steps=4,
        hidden_dim=args.hidden_dim,
        k_neighbors=args.k_neighbors,
        lr=args.lr,
        epochs=args.epochs,
        seed=args.seed,
        device=device,
    )
    del shuffled_model
    results["t4_shuffled_train_and_eval"] = shuffled_accuracy

    payload = {
        "status": "synthetic mechanism sanity only; not pathology evidence",
        "results": results,
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.out is not None:
        output_path = Path(args.out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
