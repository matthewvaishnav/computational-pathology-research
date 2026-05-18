"""
Training script for TransnnMIL v2.0.

Supports:
- Three-branch architecture (TransMIL + Hierarchical + Topology)
- Two-branch ablations (AB, AC, BC)
- Adaptive pruning
- Multi-dataset training (TCGA, PANDA)

Usage:
    # Three-branch (full model)
    python scripts/train_v2_0.py --config configs/transnnmil_v2.yaml
    
    # Two-branch ablation
    python scripts/train_v2_0.py --config configs/transnnmil_v2.yaml --branches AB
    
    # With pruning
    python scripts/train_v2_0.py --config configs/transnnmil_v2.yaml --use-pruning --keep-ratio 0.5
"""

import argparse
import os
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.models.transnnmil_v2 import TransnnMILv2, TransnnMILv2TwoBranch


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> float:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for batch in tqdm(dataloader, desc="Training"):
        features = batch["features"].to(device)
        coords = batch["coords"].to(device)
        labels = batch["label"].to(device)
        mask = batch.get("mask", None)
        if mask is not None:
            mask = mask.to(device)

        # Forward
        logits = model(features, coords, mask)
        loss = criterion(logits, labels)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    return avg_loss


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """Validate model."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validation"):
            features = batch["features"].to(device)
            coords = batch["coords"].to(device)
            labels = batch["label"].to(device)
            mask = batch.get("mask", None)
            if mask is not None:
                mask = mask.to(device)

            # Forward
            logits = model(features, coords, mask)
            loss = criterion(logits, labels)

            # Metrics
            total_loss += loss.item()
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total

    return avg_loss, accuracy, all_preds, all_labels


def main():
    parser = argparse.ArgumentParser(description="Train TransnnMIL v2.0")

    # Model args
    parser.add_argument("--feature-dim", type=int, default=1024, help="Feature dimension")
    parser.add_argument("--num-classes", type=int, default=2, help="Number of classes")
    parser.add_argument("--num-regions", type=int, default=16, help="Number of regions")
    parser.add_argument("--k-neighbors", type=int, default=8, help="Number of neighbors")
    parser.add_argument(
        "--gnn-type", type=str, default="gat", choices=["gat", "sage", "gin"], help="GNN type"
    )
    parser.add_argument("--use-pruning", action="store_true", help="Enable pruning")
    parser.add_argument("--keep-ratio", type=float, default=0.5, help="Pruning keep ratio")
    parser.add_argument(
        "--branches",
        type=str,
        default="ABC",
        choices=["ABC", "AB", "AC", "BC"],
        help="Branches to use",
    )

    # Training args
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name")
    parser.add_argument("--data-dir", type=str, required=True, help="Data directory")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-5, help="Weight decay")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")

    # Checkpoint args
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints", help="Checkpoint dir")
    parser.add_argument("--save-freq", type=int, default=5, help="Save frequency (epochs)")

    args = parser.parse_args()

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Model
    if args.branches == "ABC":
        model = TransnnMILv2(
            feature_dim=args.feature_dim,
            num_classes=args.num_classes,
            num_regions=args.num_regions,
            k_neighbors=args.k_neighbors,
            gnn_type=args.gnn_type,
            use_pruning=args.use_pruning,
            keep_ratio=args.keep_ratio,
        )
    else:
        model = TransnnMILv2TwoBranch(
            feature_dim=args.feature_dim,
            num_classes=args.num_classes,
            branches=args.branches,
            num_regions=args.num_regions,
            k_neighbors=args.k_neighbors,
            gnn_type=args.gnn_type,
        )

    model = model.to(device)
    print(f"Model: TransnnMIL v2.0 ({args.branches} branches)")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # Loss
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Data loaders (placeholder - replace with actual dataset)
    print(f"Loading dataset: {args.dataset}")
    print(f"Data directory: {args.data_dir}")
    print("Note: Dataset loading not implemented. Add your dataset here.")

    # Checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    best_acc = 0.0

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        # Train
        # train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        # print(f"Train Loss: {train_loss:.4f}")

        # Validate
        # val_loss, val_acc, preds, labels = validate(model, val_loader, criterion, device)
        # print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

        # Scheduler step
        scheduler.step()

        # Save checkpoint
        # if (epoch + 1) % args.save_freq == 0:
        #     checkpoint_path = checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pth"
        #     torch.save(
        #         {
        #             "epoch": epoch + 1,
        #             "model_state_dict": model.state_dict(),
        #             "optimizer_state_dict": optimizer.state_dict(),
        #             "scheduler_state_dict": scheduler.state_dict(),
        #             "val_acc": val_acc,
        #         },
        #         checkpoint_path,
        #     )
        #     print(f"Saved checkpoint: {checkpoint_path}")

        # Save best model
        # if val_acc > best_acc:
        #     best_acc = val_acc
        #     best_path = checkpoint_dir / "best_model.pth"
        #     torch.save(model.state_dict(), best_path)
        #     print(f"Saved best model: {best_path} (acc={best_acc:.4f})")

    print("\n✓ Training complete")
    print(f"Best accuracy: {best_acc:.4f}")


if __name__ == "__main__":
    main()
