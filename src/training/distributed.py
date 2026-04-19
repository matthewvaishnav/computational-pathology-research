"""
Distributed and multi-GPU training utilities.

This module provides utilities for distributed training across multiple GPUs
using PyTorch's DistributedDataParallel (DDP) and DataParallel.
"""

import logging
import os
from typing import Optional

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

logger = logging.getLogger(__name__)


def setup_distributed(rank: int, world_size: int, backend: str = "nccl") -> None:
    """
    Initialize distributed training process group.
    
    Args:
        rank: Rank of current process
        world_size: Total number of processes
        backend: Backend to use ('nccl' for GPU, 'gloo' for CPU)
    """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    
    dist.init_process_group(backend, rank=rank, world_size=world_size)
    logger.info(f"Initialized distributed training: rank {rank}/{world_size}")


def cleanup_distributed() -> None:
    """Clean up distributed training process group."""
    if dist.is_initialized():
        dist.destroy_process_group()
        logger.info("Cleaned up distributed training")


def setup_multi_gpu(
    model: nn.Module,
    device_ids: Optional[list] = None,
    distributed: bool = False,
    local_rank: Optional[int] = None
) -> nn.Module:
    """
    Setup model for multi-GPU training.
    
    Args:
        model: PyTorch model to parallelize
        device_ids: List of GPU device IDs to use (None for all available)
        distributed: Whether to use DistributedDataParallel (True) or DataParallel (False)
        local_rank: Local rank for distributed training
        
    Returns:
        Parallelized model
    """
    if not torch.cuda.is_available():
        logger.warning("CUDA not available, using CPU")
        return model
    
    num_gpus = torch.cuda.device_count()
    if num_gpus <= 1:
        logger.info(f"Only {num_gpus} GPU available, using single GPU")
        return model.cuda()
    
    if device_ids is None:
        device_ids = list(range(num_gpus))
    
    logger.info(f"Setting up multi-GPU training on devices: {device_ids}")
    
    if distributed:
        if local_rank is None:
            local_rank = int(os.environ.get('LOCAL_RANK', 0))
        
        torch.cuda.set_device(local_rank)
        model = model.cuda(local_rank)
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=True
        )
        logger.info(f"Using DistributedDataParallel on device {local_rank}")
    else:
        model = model.cuda()
        model = DataParallel(model, device_ids=device_ids)
        logger.info(f"Using DataParallel on devices: {device_ids}")
    
    return model


def get_world_size() -> int:
    """Get world size for distributed training."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank() -> int:
    """Get rank for distributed training."""
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def is_main_process() -> bool:
    """Check if current process is the main process."""
    return get_rank() == 0


def reduce_tensor(tensor: torch.Tensor, world_size: int) -> torch.Tensor:
    """
    Reduce tensor across all processes in distributed training.
    
    Args:
        tensor: Tensor to reduce
        world_size: Number of processes
        
    Returns:
        Reduced tensor
    """
    if world_size == 1:
        return tensor
    
    if dist.is_available() and dist.is_initialized():
        tensor = tensor.clone()
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        tensor /= world_size
    
    return tensor


class DistributedSampler:
    """
    Distributed sampler for ensuring each process gets different data.
    
    This is a simplified version - in practice, you'd use
    torch.utils.data.distributed.DistributedSampler
    """
    
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True):
        if num_replicas is None:
            num_replicas = get_world_size()
        if rank is None:
            rank = get_rank()
            
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        
        self.num_samples = len(dataset) // num_replicas
        self.total_size = self.num_samples * num_replicas
        
    def __iter__(self):
        indices = list(range(len(self.dataset)))
        
        if self.shuffle:
            import random
            random.shuffle(indices)
        
        # Subsample for this rank
        indices = indices[self.rank:self.total_size:self.num_replicas]
        
        return iter(indices)
    
    def __len__(self):
        return self.num_samples


def save_checkpoint_distributed(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    filepath: str,
    **kwargs
) -> None:
    """
    Save checkpoint in distributed training (only from main process).
    
    Args:
        model: Model to save
        optimizer: Optimizer to save
        epoch: Current epoch
        filepath: Path to save checkpoint
        **kwargs: Additional items to save
    """
    if not is_main_process():
        return
    
    # Extract model state dict (handle DDP wrapper)
    if isinstance(model, (DataParallel, DistributedDataParallel)):
        model_state_dict = model.module.state_dict()
    else:
        model_state_dict = model.state_dict()
    
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model_state_dict,
        'optimizer_state_dict': optimizer.state_dict(),
        **kwargs
    }
    
    torch.save(checkpoint, filepath)
    logger.info(f"Saved checkpoint to {filepath}")


def load_checkpoint_distributed(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    filepath: str,
    device: torch.device
) -> dict:
    """
    Load checkpoint in distributed training.
    
    Args:
        model: Model to load state into
        optimizer: Optimizer to load state into
        filepath: Path to checkpoint
        device: Device to load checkpoint on
        
    Returns:
        Checkpoint dictionary
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    # Load model state dict (handle DDP wrapper)
    if isinstance(model, (DataParallel, DistributedDataParallel)):
        model.module.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint['model_state_dict'])
    
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    logger.info(f"Loaded checkpoint from {filepath}")
    return checkpoint


def synchronize():
    """Synchronize all processes in distributed training."""
    if dist.is_available() and dist.is_initialized():
        dist.barrier()


def print_rank_0(message: str):
    """Print message only from rank 0 process."""
    if is_main_process():
        print(message)


# Example usage functions
def example_distributed_training():
    """
    Example of how to use distributed training utilities.
    
    This would typically be called from a training script with proper
    argument parsing and configuration.
    """
    # Setup (this would be in your main training script)
    local_rank = int(os.environ.get('LOCAL_RANK', 0))
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    
    # Initialize distributed training
    if world_size > 1:
        setup_distributed(local_rank, world_size)
    
    # Create model
    model = torch.nn.Linear(10, 1)  # Example model
    
    # Setup multi-GPU
    model = setup_multi_gpu(
        model,
        distributed=world_size > 1,
        local_rank=local_rank
    )
    
    # Create optimizer
    optimizer = torch.optim.Adam(model.parameters())
    
    # Training loop would go here...
    
    # Cleanup
    if world_size > 1:
        cleanup_distributed()


if __name__ == "__main__":
    # Example usage
    print("Distributed training utilities")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    if torch.cuda.is_available():
        print("GPU devices:")
        for i in range(torch.cuda.device_count()):
            print(f"  {i}: {torch.cuda.get_device_name(i)}")