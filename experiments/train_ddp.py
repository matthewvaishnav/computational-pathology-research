"""
Distributed training with DistributedDataParallel (DDP).

Multi-GPU training for faster experiments:
- Linear scaling with num GPUs
- Gradient synchronization across GPUs
- Efficient all-reduce operations

Usage:
    # Single node, 2 GPUs
    torchrun --nproc_per_node=2 experiments/train_ddp.py --batch-size 32
    
    # Multi-node (2 nodes, 4 GPUs each)
    # Node 0:
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
        --master_addr=192.168.1.1 --master_port=29500 \
        experiments/train_ddp.py --batch-size 32
    # Node 1:
    torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
        --master_addr=192.168.1.1 --master_port=29500 \
        experiments/train_ddp.py --batch-size 32
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

sys.path.append(str(Path(__file__).parent.parent))

from src.data import MultimodalDataset
from src.data.loaders import collate_multimodal
from src.models import ClassificationHead, MultimodalFusionModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def setup_ddp():
    """Init DDP process group."""
    dist.init_process_group(backend="nccl")
    
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    
    torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank


def cleanup_ddp():
    """Cleanup DDP."""
    dist.destroy_process_group()


def train_epoch(
    model: DDP,
    task_head: DDP,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    rank: int,
    epoch: int,
    use_amp: bool = False,
):
    """Train one epoch."""
    model.train()
    task_head.train()
    
    scaler = torch.cuda.amp.GradScaler() if use_amp else None
    
    total_loss = 0.0
    num_batches = 0
    
    for batch_idx, batch in enumerate(train_loader):
        # Move to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        optimizer.zero_grad()
        
        # Forward
        if use_amp:
            with torch.cuda.amp.autocast():
                embeddings = model(batch)
                logits = task_head(embeddings)
                loss = criterion(logits, batch["label"])
        else:
            embeddings = model(batch)
            logits = task_head(embeddings)
            loss = criterion(logits, batch["label"])
        
        # Backward
        if use_amp:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        if rank == 0 and batch_idx % 10 == 0:
            logger.info(f"Epoch {epoch} [{batch_idx}/{len(train_loader)}] Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / num_batches
    
    # Sync loss across GPUs
    loss_tensor = torch.tensor([avg_loss], device=device)
    dist.all_reduce(loss_tensor, op=dist.ReduceOp.AVG)
    
    return loss_tensor.item()


@torch.no_grad()
def validate(
    model: DDP,
    task_head: DDP,
    val_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    rank: int,
):
    """Validate."""
    model.eval()
    task_head.eval()
    
    total_loss = 0.0
    correct = 0
    total = 0
    
    for batch in val_loader:
        # Move to device
        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device)
        
        embeddings = model(batch)
        logits = task_head(embeddings)
        loss = criterion(logits, batch["label"])
        
        total_loss += loss.item()
        
        preds = torch.argmax(logits, dim=1)
        correct += (preds == batch["label"]).sum().item()
        total += batch["label"].size(0)
    
    avg_loss = total_loss / len(val_loader)
    accuracy = correct / total
    
    # Sync metrics across GPUs
    metrics = torch.tensor([avg_loss, correct, total], device=device)
    dist.all_reduce(metrics, op=dist.ReduceOp.SUM)
    
    world_size = dist.get_world_size()
    avg_loss = metrics[0].item() / world_size
    accuracy = metrics[1].item() / metrics[2].item()
    
    return avg_loss, accuracy


def main():
    parser = argparse.ArgumentParser(description="DDP training")
    parser.add_argument("--data-dir", type=str, default="./data")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--num-classes", type=int, default=2)
    parser.add_argument("--num-epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--use-gradient-checkpointing", action="store_true")
    parser.add_argument("--checkpoint-dir", type=str, default="./checkpoints_ddp")
    
    args = parser.parse_args()
    
    # Setup DDP
    rank, world_size, local_rank = setup_ddp()
    device = torch.device(f"cuda:{local_rank}")
    
    if rank == 0:
        logger.info(f"DDP training on {world_size} GPUs")
        logger.info(f"Effective batch size: {args.batch_size * world_size}")
    
    # Model
    model = MultimodalFusionModel(embed_dim=args.embed_dim).to(device)
    task_head = ClassificationHead(
        input_dim=args.embed_dim,
        num_classes=args.num_classes
    ).to(device)
    
    if args.use_gradient_checkpointing:
        model.enable_gradient_checkpointing()
    
    # Wrap with DDP
    model = DDP(model, device_ids=[local_rank])
    task_head = DDP(task_head, device_ids=[local_rank])
    
    # Data
    train_dataset = MultimodalDataset(
        data_dir=args.data_dir,
        split="train",
        config=vars(args)
    )
    val_dataset = MultimodalDataset(
        data_dir=args.data_dir,
        split="val",
        config=vars(args)
    )
    
    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True
    )
    val_sampler = DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_multimodal,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_multimodal,
    )
    
    # Optimizer
    params = list(model.parameters()) + list(task_head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.learning_rate)
    
    criterion = nn.CrossEntropyLoss()
    
    # Training loop
    best_acc = 0.0
    
    for epoch in range(args.num_epochs):
        train_sampler.set_epoch(epoch)
        
        train_loss = train_epoch(
            model, task_head, train_loader, optimizer, criterion,
            device, rank, epoch, args.use_amp
        )
        
        val_loss, val_acc = validate(
            model, task_head, val_loader, criterion, device, rank
        )
        
        if rank == 0:
            logger.info(
                f"Epoch {epoch}: "
                f"Train Loss={train_loss:.4f}, "
                f"Val Loss={val_loss:.4f}, "
                f"Val Acc={val_acc:.4f}"
            )
            
            # Save checkpoint
            if val_acc > best_acc:
                best_acc = val_acc
                checkpoint_dir = Path(args.checkpoint_dir)
                checkpoint_dir.mkdir(parents=True, exist_ok=True)
                
                torch.save({
                    "epoch": epoch,
                    "model_state_dict": model.module.state_dict(),
                    "task_head_state_dict": task_head.module.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_acc": val_acc,
                }, checkpoint_dir / "best_model.pth")
    
    cleanup_ddp()


if __name__ == "__main__":
    main()
