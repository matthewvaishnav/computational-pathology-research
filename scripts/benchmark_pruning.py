"""
Benchmark adaptive token pruning speedup.

Measures:
- Forward pass time (with/without pruning)
- Memory usage
- Accuracy vs speedup tradeoff

Usage:
    python scripts/benchmark_pruning.py --keep-ratios 0.25 0.5 0.75 1.0
    python scripts/benchmark_pruning.py --num-patches 50 100 200 500
"""

import argparse
import time
from typing import List

import torch
import torch.nn as nn

from src.models.adaptive_pruning import PrunedTransMIL


def benchmark_forward_time(
    model: nn.Module,
    features: torch.Tensor,
    num_runs: int = 100,
    warmup: int = 10,
) -> float:
    """
    Benchmark forward pass time.

    Args:
        model: Model to benchmark
        features: Input features
        num_runs: Number of runs
        warmup: Number of warmup runs

    Returns:
        avg_time: Average time per forward pass (ms)
    """
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(warmup):
            _ = model(features)

    # Benchmark
    torch.cuda.synchronize() if torch.cuda.is_available() else None

    start = time.time()
    with torch.no_grad():
        for _ in range(num_runs):
            _ = model(features)

    torch.cuda.synchronize() if torch.cuda.is_available() else None
    end = time.time()

    avg_time = (end - start) / num_runs * 1000  # ms
    return avg_time


def benchmark_memory(model: nn.Module, features: torch.Tensor) -> float:
    """
    Benchmark memory usage.

    Args:
        model: Model to benchmark
        features: Input features

    Returns:
        memory_mb: Memory usage (MB)
    """
    if not torch.cuda.is_available():
        return 0.0

    torch.cuda.reset_peak_memory_stats()

    model.eval()
    with torch.no_grad():
        _ = model(features)

    memory_mb = torch.cuda.max_memory_allocated() / 1024 / 1024
    return memory_mb


def main():
    parser = argparse.ArgumentParser(description="Benchmark adaptive pruning")

    parser.add_argument(
        "--keep-ratios",
        type=float,
        nargs="+",
        default=[0.25, 0.5, 0.75, 1.0],
        help="Keep ratios to benchmark",
    )
    parser.add_argument(
        "--num-patches",
        type=int,
        nargs="+",
        default=[50, 100, 200, 500],
        help="Number of patches to benchmark",
    )
    parser.add_argument(
        "--feature-dim", type=int, default=1024, help="Feature dimension (default: 1024)"
    )
    parser.add_argument(
        "--num-classes", type=int, default=2, help="Number of classes (default: 2)"
    )
    parser.add_argument(
        "--batch-size", type=int, default=4, help="Batch size (default: 4)"
    )
    parser.add_argument(
        "--num-runs", type=int, default=100, help="Number of benchmark runs (default: 100)"
    )
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")

    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Device: {device}")
    print(f"Feature dim: {args.feature_dim}")
    print(f"Batch size: {args.batch_size}")
    print()

    # Benchmark 1: Keep ratio vs speedup
    print("=" * 80)
    print("Benchmark 1: Keep Ratio vs Speedup")
    print("=" * 80)
    print(f"{'Keep Ratio':<15} {'Time (ms)':<15} {'Speedup':<15} {'Memory (MB)':<15}")
    print("-" * 80)

    num_patches = 200
    features = torch.randn(args.batch_size, num_patches, args.feature_dim).to(device)

    baseline_time = None
    baseline_memory = None

    for keep_ratio in args.keep_ratios:
        model = PrunedTransMIL(
            feature_dim=args.feature_dim,
            num_classes=args.num_classes,
            keep_ratio=keep_ratio,
        ).to(device)

        # Benchmark time
        avg_time = benchmark_forward_time(model, features, num_runs=args.num_runs)

        # Benchmark memory
        memory_mb = benchmark_memory(model, features)

        # Compute speedup
        if baseline_time is None:
            baseline_time = avg_time
            baseline_memory = memory_mb
            speedup = 1.0
        else:
            speedup = baseline_time / avg_time

        print(
            f"{keep_ratio:<15.2f} {avg_time:<15.2f} {speedup:<15.2f} {memory_mb:<15.2f}"
        )

    print()

    # Benchmark 2: Number of patches vs speedup
    print("=" * 80)
    print("Benchmark 2: Number of Patches vs Speedup (keep_ratio=0.5)")
    print("=" * 80)
    print(
        f"{'Num Patches':<15} {'Time (ms)':<15} {'Speedup':<15} {'Theoretical':<15}"
    )
    print("-" * 80)

    keep_ratio = 0.5

    for num_patches in args.num_patches:
        features = torch.randn(args.batch_size, num_patches, args.feature_dim).to(device)

        # Full model (keep_ratio=1.0)
        model_full = PrunedTransMIL(
            feature_dim=args.feature_dim,
            num_classes=args.num_classes,
            keep_ratio=1.0,
        ).to(device)

        time_full = benchmark_forward_time(model_full, features, num_runs=args.num_runs)

        # Pruned model (keep_ratio=0.5)
        model_pruned = PrunedTransMIL(
            feature_dim=args.feature_dim,
            num_classes=args.num_classes,
            keep_ratio=keep_ratio,
        ).to(device)

        time_pruned = benchmark_forward_time(
            model_pruned, features, num_runs=args.num_runs
        )

        # Compute speedup
        speedup = time_full / time_pruned

        # Theoretical speedup (quadratic complexity)
        theoretical_speedup = (1.0 / keep_ratio) ** 2

        print(
            f"{num_patches:<15} {time_pruned:<15.2f} {speedup:<15.2f} {theoretical_speedup:<15.2f}"
        )

    print()

    # Benchmark 3: Scoring method comparison
    print("=" * 80)
    print("Benchmark 3: Scoring Method Comparison (keep_ratio=0.5)")
    print("=" * 80)
    print(f"{'Method':<15} {'Time (ms)':<15} {'Memory (MB)':<15}")
    print("-" * 80)

    num_patches = 200
    features = torch.randn(args.batch_size, num_patches, args.feature_dim).to(device)

    for scoring_method in ["learned", "attention", "confidence"]:
        model = PrunedTransMIL(
            feature_dim=args.feature_dim,
            num_classes=args.num_classes,
            keep_ratio=0.5,
            scoring_method=scoring_method,
        ).to(device)

        avg_time = benchmark_forward_time(model, features, num_runs=args.num_runs)
        memory_mb = benchmark_memory(model, features)

        print(f"{scoring_method:<15} {avg_time:<15.2f} {memory_mb:<15.2f}")

    print()
    print("✓ Benchmark complete")


if __name__ == "__main__":
    main()
