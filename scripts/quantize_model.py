#!/usr/bin/env python3
"""
Model Quantization Script

Quantize trained models for faster inference and reduced memory usage.
"""

import argparse
import logging
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add src to path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.inference.quantization import ModelQuantizer, quantize_attention_mil
from src.models import AttentionMIL
from src.data.pcam_dataset import PCamDataset

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_model(checkpoint_path: Path, device: str = "cpu") -> nn.Module:
    """Load trained model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint
        device: Device to load model on

    Returns:
        Loaded model
    """
    logger.info(f"Loading model from {checkpoint_path}")

    # Create model
    model = AttentionMIL(
        feature_dim=2048,
        hidden_dim=256,
        num_classes=2,
        dropout=0.25,
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load state dict
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()
    logger.info("Model loaded successfully")

    return model


def create_calibration_dataloader(
    data_dir: Path, batch_size: int = 32, num_samples: int = 1000
) -> DataLoader:
    """Create calibration dataloader.

    Args:
        data_dir: Data directory
        batch_size: Batch size
        num_samples: Number of calibration samples

    Returns:
        Calibration dataloader
    """
    logger.info(f"Creating calibration dataloader with {num_samples} samples")

    # Create dataset
    dataset = PCamDataset(
        data_dir=data_dir,
        split="valid",
        transform=None,  # Use default transforms
    )

    # Limit to num_samples
    if len(dataset) > num_samples:
        indices = torch.randperm(len(dataset))[:num_samples]
        dataset = torch.utils.data.Subset(dataset, indices)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    logger.info(f"Calibration dataloader created with {len(dataset)} samples")

    return dataloader


def quantize_model(
    model: nn.Module,
    method: str,
    calibration_data: DataLoader = None,
    output_path: Path = None,
) -> nn.Module:
    """Quantize model.

    Args:
        model: Model to quantize
        method: Quantization method
        calibration_data: Calibration data (for static)
        output_path: Output path

    Returns:
        Quantized model
    """
    logger.info(f"Quantizing model with method: {method}")

    # Create quantizer
    quantizer = ModelQuantizer()

    # Quantize
    if method == "dynamic":
        quantized_model = quantizer.quantize_dynamic(model, dtype=torch.qint8)
    elif method == "static":
        if calibration_data is None:
            raise ValueError("Calibration data required for static quantization")
        quantized_model = quantizer.quantize_static(model, calibration_data)
    elif method == "fp16":
        quantized_model = quantizer.quantize_to_fp16(model)
    else:
        raise ValueError(f"Unknown quantization method: {method}")

    # Save if output path provided
    if output_path:
        quantizer.save_quantized_model(
            quantized_model,
            output_path,
            metadata={
                "method": method,
                "original_model": "AttentionMIL",
            },
        )

    return quantized_model


def benchmark_quantized_model(
    original_model: nn.Module,
    quantized_model: nn.Module,
    test_input: torch.Tensor,
    num_runs: int = 100,
):
    """Benchmark quantized model.

    Args:
        original_model: Original model
        quantized_model: Quantized model
        test_input: Test input
        num_runs: Number of runs
    """
    logger.info("Benchmarking quantized model...")

    quantizer = ModelQuantizer()

    results = quantizer.compare_models(
        original_model, quantized_model, test_input, num_runs
    )

    # Print results
    print("\n" + "=" * 60)
    print("QUANTIZATION BENCHMARK RESULTS")
    print("=" * 60)

    print("\nOriginal Model:")
    print(f"  Mean inference time: {results['original']['mean_time']*1000:.2f} ms")
    print(f"  P95 inference time: {results['original']['p95_time']*1000:.2f} ms")
    print(f"  Model size: {results['original']['model_size']/1024/1024:.2f} MB")

    print("\nQuantized Model:")
    print(f"  Mean inference time: {results['quantized']['mean_time']*1000:.2f} ms")
    print(f"  P95 inference time: {results['quantized']['p95_time']*1000:.2f} ms")
    print(f"  Model size: {results['quantized']['model_size']/1024/1024:.2f} MB")

    print("\nImprovements:")
    print(f"  Speedup: {results['improvements']['speedup']:.2f}x")
    print(f"  Memory reduction: {results['improvements']['memory_reduction']:.2f}x")
    print(
        f"  Latency reduction: {results['improvements']['latency_reduction_ms']:.2f} ms"
    )

    print("\n" + "=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Quantize trained models")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--method",
        type=str,
        choices=["dynamic", "static", "fp16"],
        default="dynamic",
        help="Quantization method",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data/pcam",
        help="Data directory (for calibration)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output path for quantized model",
    )
    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark quantized model",
    )
    parser.add_argument(
        "--num-runs",
        type=int,
        default=100,
        help="Number of benchmark runs",
    )
    parser.add_argument(
        "--calibration-samples",
        type=int,
        default=1000,
        help="Number of calibration samples (for static quantization)",
    )

    args = parser.parse_args()

    # Load model
    model = load_model(Path(args.checkpoint))

    # Create calibration data if needed
    calibration_data = None
    if args.method == "static":
        calibration_data = create_calibration_dataloader(
            Path(args.data_dir),
            batch_size=32,
            num_samples=args.calibration_samples,
        )

    # Quantize model
    output_path = Path(args.output) if args.output else None
    quantized_model = quantize_model(
        model,
        args.method,
        calibration_data,
        output_path,
    )

    # Benchmark if requested
    if args.benchmark:
        # Create test input (batch of 8 instances with 100 features each)
        test_input = torch.randn(8, 100, 2048)

        benchmark_quantized_model(
            model,
            quantized_model,
            test_input,
            args.num_runs,
        )

    logger.info("Quantization complete!")


if __name__ == "__main__":
    main()
