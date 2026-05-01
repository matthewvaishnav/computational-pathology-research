"""
Training Pipeline Performance Profiler

Profile training loop to identify bottlenecks:
- Data loading time
- Forward pass time
- Backward pass time
- GPU utilization
- Memory usage

Usage:
    python scripts/profile_training.py --config configs/train_config.yaml
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict

import torch
import torch.nn as nn
from torch.profiler import ProfilerActivity, profile, record_function
from torch.utils.data import DataLoader

sys.path.append(str(Path(__file__).parent.parent))

from src.data import MultimodalDataset
from src.data.loaders import collate_multimodal
from src.models import ClassificationHead, MultimodalFusionModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TrainingProfiler:
    """Profile training pipeline performance."""

    def __init__(
        self,
        model: nn.Module,
        task_head: nn.Module,
        train_loader: DataLoader,
        device: str = "cuda",
    ):
        self.model = model.to(device)
        self.task_head = task_head.to(device)
        self.train_loader = train_loader
        self.device = device
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(
            list(model.parameters()) + list(task_head.parameters()), lr=1e-4
        )

    def profile_single_batch(self, num_batches: int = 10) -> Dict[str, float]:
        """Profile single batch operations."""
        self.model.train()
        self.task_head.train()

        timings = {
            "data_loading": [],
            "to_device": [],
            "forward": [],
            "loss": [],
            "backward": [],
            "optimizer": [],
            "total": [],
        }

        iterator = iter(self.train_loader)

        for i in range(num_batches):
            # Data loading
            t0 = time.perf_counter()
            batch = next(iterator)
            t1 = time.perf_counter()
            timings["data_loading"].append(t1 - t0)

            # Move to device
            batch = {
                k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                for k, v in batch.items()
            }
            t2 = time.perf_counter()
            timings["to_device"].append(t2 - t1)

            labels = batch.pop("label")

            # Forward pass
            self.optimizer.zero_grad()
            embeddings = self.model(batch)
            logits = self.task_head(embeddings)
            t3 = time.perf_counter()
            timings["forward"].append(t3 - t2)

            # Loss computation
            loss = self.criterion(logits, labels)
            t4 = time.perf_counter()
            timings["loss"].append(t4 - t3)

            # Backward pass
            loss.backward()
            t5 = time.perf_counter()
            timings["backward"].append(t5 - t4)

            # Optimizer step
            self.optimizer.step()
            t6 = time.perf_counter()
            timings["optimizer"].append(t6 - t5)

            timings["total"].append(t6 - t0)

        # Compute averages
        avg_timings = {k: sum(v) / len(v) * 1000 for k, v in timings.items()}  # Convert to ms

        return avg_timings

    def profile_with_pytorch_profiler(self, num_batches: int = 10) -> None:
        """Profile using PyTorch profiler."""
        self.model.train()
        self.task_head.train()

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,
        ) as prof:
            iterator = iter(self.train_loader)

            for i in range(num_batches):
                with record_function("data_loading"):
                    batch = next(iterator)
                    batch = {
                        k: v.to(self.device) if isinstance(v, torch.Tensor) else v
                        for k, v in batch.items()
                    }
                    labels = batch.pop("label")

                with record_function("forward"):
                    self.optimizer.zero_grad()
                    embeddings = self.model(batch)
                    logits = self.task_head(embeddings)

                with record_function("loss"):
                    loss = self.criterion(logits, labels)

                with record_function("backward"):
                    loss.backward()

                with record_function("optimizer"):
                    self.optimizer.step()

        # Print profiler results
        print("\n" + "=" * 80)
        print("PyTorch Profiler Results (CPU Time)")
        print("=" * 80)
        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=20))

        print("\n" + "=" * 80)
        print("PyTorch Profiler Results (CUDA Time)")
        print("=" * 80)
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

        print("\n" + "=" * 80)
        print("PyTorch Profiler Results (Memory)")
        print("=" * 80)
        print(
            prof.key_averages().table(
                sort_by="self_cuda_memory_usage", row_limit=20
            )
        )

    def profile_data_loader(self, num_batches: int = 100) -> Dict[str, float]:
        """Profile data loader performance."""
        logger.info(f"Profiling data loader for {num_batches} batches...")

        times = []
        for i, batch in enumerate(self.train_loader):
            if i >= num_batches:
                break

            t0 = time.perf_counter()
            # Simulate minimal processing
            _ = {k: v.shape if isinstance(v, torch.Tensor) else v for k, v in batch.items()}
            t1 = time.perf_counter()
            times.append(t1 - t0)

        avg_time = sum(times) / len(times) * 1000  # ms
        throughput = 1000 / avg_time  # batches/sec

        return {
            "avg_batch_time_ms": avg_time,
            "throughput_batches_per_sec": throughput,
        }


def main():
    parser = argparse.ArgumentParser(description="Profile training pipeline")
    parser.add_argument("--data_dir", type=str, default="data/processed", help="Data directory")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--num_workers", type=int, default=4, help="DataLoader workers")
    parser.add_argument("--num_batches", type=int, default=10, help="Number of batches to profile")
    parser.add_argument("--device", type=str, default="cuda", help="Device")
    args = parser.parse_args()

    # Create dummy config
    config = {
        "wsi_enabled": True,
        "genomic_enabled": True,
        "clinical_text_enabled": True,
        "wsi_feature_dim": 1024,
        "genomic_feature_dim": 2000,
        "max_text_length": 512,
    }

    # Create dataset and loader
    logger.info("Creating dataset...")
    dataset = MultimodalDataset(Path(args.data_dir), "train", config)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_multimodal,
        pin_memory=True if args.device == "cuda" else False,
    )

    # Create model
    logger.info("Creating model...")
    model = MultimodalFusionModel(
        wsi_feature_dim=1024,
        genomic_feature_dim=2000,
        clinical_vocab_size=30000,
        fusion_dim=512,
    )
    task_head = ClassificationHead(input_dim=512, num_classes=2)

    # Create profiler
    profiler = TrainingProfiler(model, task_head, loader, device=args.device)

    # Profile single batch operations
    logger.info("\n" + "=" * 80)
    logger.info("Profiling Single Batch Operations")
    logger.info("=" * 80)
    timings = profiler.profile_single_batch(num_batches=args.num_batches)

    print("\nAverage Timings (ms):")
    print("-" * 40)
    for key, value in timings.items():
        print(f"{key:20s}: {value:8.2f} ms")

    # Calculate percentages
    total_time = timings["total"]
    print("\nTime Distribution:")
    print("-" * 40)
    for key in ["data_loading", "to_device", "forward", "backward", "optimizer"]:
        pct = (timings[key] / total_time) * 100
        print(f"{key:20s}: {pct:6.1f}%")

    # Profile data loader
    logger.info("\n" + "=" * 80)
    logger.info("Profiling Data Loader")
    logger.info("=" * 80)
    loader_stats = profiler.profile_data_loader(num_batches=100)
    print(f"\nData Loader Performance:")
    print(f"  Avg batch time: {loader_stats['avg_batch_time_ms']:.2f} ms")
    print(f"  Throughput: {loader_stats['throughput_batches_per_sec']:.2f} batches/sec")

    # PyTorch profiler
    if args.device == "cuda":
        logger.info("\n" + "=" * 80)
        logger.info("Running PyTorch Profiler")
        logger.info("=" * 80)
        profiler.profile_with_pytorch_profiler(num_batches=args.num_batches)


if __name__ == "__main__":
    main()
