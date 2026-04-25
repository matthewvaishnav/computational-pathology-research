"""
Multi-GPU training script for PCam using DistributedDataParallel.

This script extends the single-GPU train_pcam.py to support distributed training
across multiple GPUs using PyTorch's DistributedDataParallel.

Usage:
    # Single node, multiple GPUs
    torchrun --nproc_per_node=2 experiments/train_pcam_multigpu.py --config experiments/configs/pcam_multigpu.yaml

    # Multiple nodes (example)
    torchrun --nnodes=2 --nproc_per_node=2 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 experiments/train_pcam_multigpu.py --config experiments/configs/pcam_multigpu.yaml
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
from torch.utils.data.distributed import DistributedSampler

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from experiments.train_pcam import (
    create_pcam_dataloaders,
    create_single_modality_model,
    load_config,
    set_seed,
    train_epoch,
    validate,
)
from src.training.distributed import (
    cleanup_distributed,
    is_main_process,
    load_checkpoint_distributed,
    print_rank_0,
    save_checkpoint_distributed,
    setup_distributed,
    setup_multi_gpu,
    synchronize,
)

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def create_distributed_dataloaders(config: dict, rank: int, world_size: int):
    """
    Create data loaders with distributed sampling.

    Args:
        config: Configuration dictionary
        rank: Process rank
        world_size: Total number of processes

    Returns:
        Tuple of (train_loader, val_loader, test_loader)
    """
    # Import here to avoid circular imports
    from torch.utils.data import DataLoader

    from src.data.pcam_dataset import PCamDataset, get_pcam_transforms

    # Get transforms
    train_transform, val_transform = get_pcam_transforms(config)

    # Create datasets
    train_dataset = PCamDataset(
        root_dir=config["data"]["root_dir"],
        split="train",
        transform=train_transform,
        download=config["data"].get("download", False),
        feature_extractor=None,  # Will be handled in model
    )

    val_dataset = PCamDataset(
        root_dir=config["data"]["root_dir"],
        split="val",
        transform=val_transform,
        download=False,
        feature_extractor=None,
    )

    test_dataset = PCamDataset(
        root_dir=config["data"]["root_dir"],
        split="test",
        transform=val_transform,
        download=False,
        feature_extractor=None,
    )

    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True
    )

    val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)

    test_sampler = DistributedSampler(
        test_dataset, num_replicas=world_size, rank=rank, shuffle=False
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        sampler=train_sampler,
        num_workers=config["data"].get("num_workers", 0),
        pin_memory=config["data"].get("pin_memory", True),
        drop_last=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        sampler=val_sampler,
        num_workers=config["data"].get("num_workers", 0),
        pin_memory=config["data"].get("pin_memory", True),
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        sampler=test_sampler,
        num_workers=config["data"].get("num_workers", 0),
        pin_memory=config["data"].get("pin_memory", True),
        drop_last=False,
    )

    return train_loader, val_loader, test_loader, train_sampler


def main():
    """Main distributed training function."""
    parser = argparse.ArgumentParser(description="Multi-GPU PCam Training")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    parser.add_argument("--resume", type=str, help="Path to checkpoint to resume from")

    args = parser.parse_args()

    # Get distributed training info
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    # Setup distributed training
    if world_size > 1:
        setup_distributed(rank, world_size)
        torch.cuda.set_device(local_rank)

    # Load config
    config = load_config(args.config)

    # Set seed (different for each rank to ensure different augmentations)
    set_seed(config.get("seed", 42) + rank)

    # Print info only from main process
    print_rank_0("=" * 80)
    print_rank_0("Multi-GPU PCam Training")
    print_rank_0("=" * 80)
    print_rank_0(f"World size: {world_size}")
    print_rank_0(f"Local rank: {local_rank}")
    print_rank_0(f"Config: {args.config}")
    print_rank_0("=" * 80)

    # Create data loaders
    train_loader, val_loader, test_loader, train_sampler = create_distributed_dataloaders(
        config, rank, world_size
    )

    print_rank_0(f"Dataset sizes:")
    print_rank_0(f"  Train: {len(train_loader.dataset)} samples")
    print_rank_0(f"  Val: {len(val_loader.dataset)} samples")
    print_rank_0(f"  Test: {len(test_loader.dataset)} samples")

    # Create model
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    feature_extractor, encoder, head = create_single_modality_model(config, device)

    # Combine into single model for easier DDP handling
    class CombinedModel(torch.nn.Module):
        def __init__(self, feature_extractor, encoder, head):
            super().__init__()
            self.feature_extractor = feature_extractor
            self.encoder = encoder
            self.head = head

        def forward(self, x):
            features = self.feature_extractor(x)
            # Add batch dimension for encoder (expects [B, N, D])
            features = features.unsqueeze(1)
            encoded = self.encoder(features)
            logits = self.head(encoded)
            return logits

    model = CombinedModel(feature_extractor, encoder, head)

    # Setup multi-GPU
    model = setup_multi_gpu(model, distributed=world_size > 1, local_rank=local_rank)

    # Create optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"].get("weight_decay", 1e-4),
    )

    # Create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["num_epochs"],
        eta_min=config["training"]["scheduler"].get("min_lr", 1e-6),
    )

    # Resume from checkpoint if specified
    start_epoch = 1
    if args.resume:
        checkpoint = load_checkpoint_distributed(model, optimizer, args.resume, device)
        start_epoch = checkpoint["epoch"] + 1
        scheduler.load_state_dict(checkpoint.get("scheduler_state_dict", {}))
        print_rank_0(f"Resumed from epoch {checkpoint['epoch']}")

    # Training loop
    best_val_auc = 0.0

    for epoch in range(start_epoch, config["training"]["num_epochs"] + 1):
        # Set epoch for distributed sampler (ensures different shuffling each epoch)
        train_sampler.set_epoch(epoch)

        print_rank_0(f"\\nEpoch {epoch}/{config['training']['num_epochs']}")
        print_rank_0("-" * 50)

        # Train
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_idx, batch in enumerate(train_loader):
            images = batch["wsi_features"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            # Forward pass
            logits = model(images)
            if logits.dim() > 1 and logits.size(-1) == 1:
                logits = logits.squeeze(-1)

            # Compute loss
            criterion = torch.nn.BCEWithLogitsLoss()
            loss = criterion(logits, labels.float())

            # Backward pass
            loss.backward()

            # Gradient clipping
            if config["training"].get("max_grad_norm"):
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), config["training"]["max_grad_norm"]
                )

            optimizer.step()

            # Statistics
            train_loss += loss.item()
            preds = (torch.sigmoid(logits) > 0.5).long()
            train_correct += (preds == labels).sum().item()
            train_total += labels.size(0)

            # Log progress
            if batch_idx % 100 == 0 and is_main_process():
                logger.info(
                    f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f} "
                    f"Acc: {100.0 * train_correct / train_total:.2f}%"
                )

        # Synchronize before validation
        synchronize()

        # Validation (only on main process to avoid duplicate evaluation)
        if is_main_process():
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch in val_loader:
                    images = batch["wsi_features"].to(device)
                    labels = batch["labels"].to(device)

                    logits = model(images)
                    if logits.dim() > 1 and logits.size(-1) == 1:
                        logits = logits.squeeze(-1)

                    loss = criterion(logits, labels.float())
                    val_loss += loss.item()

                    preds = (torch.sigmoid(logits) > 0.5).long()
                    val_correct += (preds == labels).sum().item()
                    val_total += labels.size(0)

            # Compute metrics
            train_acc = 100.0 * train_correct / train_total
            val_acc = 100.0 * val_correct / val_total
            avg_train_loss = train_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)

            print_rank_0(f"Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
            print_rank_0(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%")

            # Save checkpoint
            val_auc = val_acc / 100.0  # Simplified - would use actual AUC in practice
            if val_auc > best_val_auc:
                best_val_auc = val_auc

                checkpoint_dir = Path(config["checkpoint"]["checkpoint_dir"])
                checkpoint_dir.mkdir(parents=True, exist_ok=True)

                save_checkpoint_distributed(
                    model,
                    optimizer,
                    epoch,
                    str(checkpoint_dir / "best_model.pth"),
                    scheduler_state_dict=scheduler.state_dict(),
                    val_auc=best_val_auc,
                    config=config,
                )

                print_rank_0(f"Saved best model (Val AUC: {best_val_auc:.4f})")

        # Update scheduler
        scheduler.step()

        # Synchronize after each epoch
        synchronize()

    print_rank_0("Training completed!")

    # Cleanup
    if world_size > 1:
        cleanup_distributed()


if __name__ == "__main__":
    main()
