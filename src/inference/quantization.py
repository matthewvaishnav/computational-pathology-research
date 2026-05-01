#!/usr/bin/env python3
"""
Model Quantization for Faster Inference

Implements dynamic, static, and QAT (Quantization-Aware Training) quantization
for reduced model size and faster inference.
"""

import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.quantization as quant
from torch.quantization import (
    get_default_qconfig,
    prepare,
    convert,
    prepare_qat,
    quantize_dynamic,
)

logger = logging.getLogger(__name__)


class ModelQuantizer:
    """Quantize models for faster inference and reduced memory."""

    def __init__(self, backend: str = "qnnpack"):
        """Initialize quantizer.

        Args:
            backend: Quantization backend ('fbgemm' for x86, 'qnnpack' for ARM/Windows)
        """
        # Auto-select backend based on platform
        if backend == "fbgemm":
            try:
                torch.backends.quantized.engine = backend
                self.backend = backend
            except RuntimeError:
                # Fallback to qnnpack if fbgemm not supported
                logger.warning("FBGEMM not supported, falling back to qnnpack")
                backend = "qnnpack"
                torch.backends.quantized.engine = backend
                self.backend = backend
        else:
            self.backend = backend
            torch.backends.quantized.engine = backend

        logger.info(f"ModelQuantizer initialized with backend: {self.backend}")

    def quantize_dynamic(
        self,
        model: nn.Module,
        dtype: torch.dtype = torch.qint8,
        modules_to_quantize: Optional[set] = None,
    ) -> nn.Module:
        """Apply dynamic quantization (weights quantized, activations computed in fp32).

        Best for: Models with dynamic input sizes, LSTM/GRU, Transformers
        Speedup: 2-4x on CPU
        Memory: 4x reduction

        Args:
            model: Model to quantize
            dtype: Quantization dtype (qint8 or float16)
            modules_to_quantize: Module types to quantize (default: Linear, LSTM, GRU)

        Returns:
            Quantized model
        """
        if modules_to_quantize is None:
            modules_to_quantize = {nn.Linear, nn.LSTM, nn.GRU}

        logger.info(f"Applying dynamic quantization with dtype={dtype}")

        # Apply dynamic quantization
        quantized_model = quantize_dynamic(
            model,
            qconfig_spec=modules_to_quantize,
            dtype=dtype,
        )

        logger.info("Dynamic quantization complete")
        return quantized_model

    def quantize_static(
        self,
        model: nn.Module,
        calibration_data: torch.utils.data.DataLoader,
        qconfig: Optional[Any] = None,
    ) -> nn.Module:
        """Apply static quantization (weights and activations quantized).

        Best for: CNNs, fixed input sizes
        Speedup: 2-4x on CPU, potential GPU acceleration
        Memory: 4x reduction
        Accuracy: May need calibration

        Args:
            model: Model to quantize
            calibration_data: DataLoader for calibration
            qconfig: Quantization config (default: fbgemm default)

        Returns:
            Quantized model
        """
        logger.info("Applying static quantization")

        # Set model to eval mode
        model.eval()

        # Set quantization config
        if qconfig is None:
            qconfig = get_default_qconfig(self.backend)

        model.qconfig = qconfig

        # Fuse modules (Conv+BN+ReLU, etc.)
        model = self._fuse_modules(model)

        # Prepare for quantization (insert observers)
        model_prepared = prepare(model, inplace=False)

        # Calibrate with representative data
        logger.info("Calibrating model...")
        self._calibrate(model_prepared, calibration_data)

        # Convert to quantized model
        quantized_model = convert(model_prepared, inplace=False)

        logger.info("Static quantization complete")
        return quantized_model

    def prepare_qat(
        self,
        model: nn.Module,
        qconfig: Optional[Any] = None,
    ) -> nn.Module:
        """Prepare model for Quantization-Aware Training.

        Best for: Maximum accuracy with quantization
        Speedup: 2-4x after training
        Memory: 4x reduction after training
        Requires: Retraining/fine-tuning

        Args:
            model: Model to prepare
            qconfig: Quantization config

        Returns:
            Model prepared for QAT
        """
        logger.info("Preparing model for QAT")

        # Set model to train mode
        model.train()

        # Set quantization config
        if qconfig is None:
            qconfig = get_default_qconfig(self.backend)

        model.qconfig = qconfig

        # Fuse modules
        model = self._fuse_modules(model)

        # Prepare for QAT (insert fake quantization modules)
        model_prepared = prepare_qat(model, inplace=False)

        logger.info("Model prepared for QAT")
        return model_prepared

    def convert_qat(self, model: nn.Module) -> nn.Module:
        """Convert QAT model to quantized model.

        Args:
            model: QAT-trained model

        Returns:
            Quantized model
        """
        logger.info("Converting QAT model to quantized")

        # Set to eval mode
        model.eval()

        # Convert to quantized
        quantized_model = convert(model, inplace=False)

        logger.info("QAT conversion complete")
        return quantized_model

    def quantize_to_int8(
        self,
        model: nn.Module,
        calibration_data: Optional[torch.utils.data.DataLoader] = None,
        method: str = "dynamic",
    ) -> nn.Module:
        """Convenience method to quantize model to INT8.

        Args:
            model: Model to quantize
            calibration_data: Calibration data (required for static)
            method: Quantization method ('dynamic' or 'static')

        Returns:
            INT8 quantized model
        """
        if method == "dynamic":
            return self.quantize_dynamic(model, dtype=torch.qint8)
        elif method == "static":
            if calibration_data is None:
                raise ValueError("Calibration data required for static quantization")
            return self.quantize_static(model, calibration_data)
        else:
            raise ValueError(f"Unknown quantization method: {method}")

    def quantize_to_fp16(self, model: nn.Module) -> nn.Module:
        """Quantize model to FP16 (half precision).

        Best for: GPU inference
        Speedup: 1.5-2x on GPU
        Memory: 2x reduction
        Accuracy: Minimal loss

        Args:
            model: Model to quantize

        Returns:
            FP16 model
        """
        logger.info("Converting model to FP16")

        # Convert to half precision
        model_fp16 = model.half()

        logger.info("FP16 conversion complete")
        return model_fp16

    def _fuse_modules(self, model: nn.Module) -> nn.Module:
        """Fuse consecutive modules for better quantization.

        Fuses:
        - Conv2d + BatchNorm2d + ReLU
        - Conv2d + BatchNorm2d
        - Conv2d + ReLU
        - Linear + ReLU
        """
        # Try to fuse common patterns
        try:
            # This is a simplified version - real implementation would
            # need to traverse the model graph and identify fusion patterns
            if hasattr(model, "fuse_model"):
                model.fuse_model()
                logger.info("Modules fused using model.fuse_model()")
            else:
                logger.info("No automatic fusion available")
        except Exception as e:
            logger.warning(f"Module fusion failed: {e}")

        return model

    def _calibrate(
        self,
        model: nn.Module,
        calibration_data: torch.utils.data.DataLoader,
        num_batches: int = 100,
    ):
        """Calibrate model with representative data.

        Args:
            model: Model with observers
            calibration_data: Calibration dataset
            num_batches: Number of batches to use
        """
        model.eval()

        with torch.no_grad():
            for i, (images, _) in enumerate(calibration_data):
                if i >= num_batches:
                    break

                # Forward pass to collect statistics
                _ = model(images)

                if (i + 1) % 10 == 0:
                    logger.info(f"Calibrated {i + 1}/{num_batches} batches")

        logger.info(f"Calibration complete with {min(i + 1, num_batches)} batches")

    def compare_models(
        self,
        original_model: nn.Module,
        quantized_model: nn.Module,
        test_input: torch.Tensor,
        num_runs: int = 100,
    ) -> Dict[str, Any]:
        """Compare original and quantized models.

        Args:
            original_model: Original model
            quantized_model: Quantized model
            test_input: Test input tensor
            num_runs: Number of benchmark runs

        Returns:
            Comparison results
        """
        logger.info("Comparing original and quantized models...")

        # Benchmark original model
        original_stats = self._benchmark_model(original_model, test_input, num_runs)

        # Benchmark quantized model
        quantized_stats = self._benchmark_model(quantized_model, test_input, num_runs)

        # Calculate improvements
        speedup = original_stats["mean_time"] / quantized_stats["mean_time"]
        memory_reduction = original_stats["model_size"] / quantized_stats["model_size"]

        results = {
            "original": original_stats,
            "quantized": quantized_stats,
            "improvements": {
                "speedup": speedup,
                "memory_reduction": memory_reduction,
                "latency_reduction_ms": (
                    original_stats["mean_time"] - quantized_stats["mean_time"]
                )
                * 1000,
            },
        }

        logger.info(f"Speedup: {speedup:.2f}x")
        logger.info(f"Memory reduction: {memory_reduction:.2f}x")
        logger.info(
            f"Latency reduction: {results['improvements']['latency_reduction_ms']:.2f}ms"
        )

        return results

    def _benchmark_model(
        self, model: nn.Module, test_input: torch.Tensor, num_runs: int
    ) -> Dict[str, Any]:
        """Benchmark model performance.

        Args:
            model: Model to benchmark
            test_input: Test input
            num_runs: Number of runs

        Returns:
            Performance statistics
        """
        model.eval()

        # Warm up
        with torch.no_grad():
            for _ in range(10):
                _ = model(test_input)

        # Benchmark
        times = []
        with torch.no_grad():
            for _ in range(num_runs):
                start = time.perf_counter()
                _ = model(test_input)
                times.append(time.perf_counter() - start)

        import numpy as np

        # Get model size
        model_size = self._get_model_size(model)

        return {
            "mean_time": np.mean(times),
            "std_time": np.std(times),
            "min_time": np.min(times),
            "max_time": np.max(times),
            "p95_time": np.percentile(times, 95),
            "p99_time": np.percentile(times, 99),
            "model_size": model_size,
        }

    def _get_model_size(self, model: nn.Module) -> int:
        """Get model size in bytes.

        Args:
            model: Model

        Returns:
            Size in bytes
        """
        # Save to temporary buffer
        import io

        buffer = io.BytesIO()
        torch.save(model.state_dict(), buffer)
        size = buffer.tell()
        buffer.close()

        return size

    def save_quantized_model(
        self, model: nn.Module, save_path: Path, metadata: Optional[Dict] = None
    ):
        """Save quantized model.

        Args:
            model: Quantized model
            save_path: Path to save
            metadata: Optional metadata
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Prepare checkpoint
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "quantization_backend": self.backend,
            "metadata": metadata or {},
        }

        # Save
        torch.save(checkpoint, save_path)
        logger.info(f"Quantized model saved to {save_path}")

        # Log size
        size_mb = save_path.stat().st_size / (1024 * 1024)
        logger.info(f"Model size: {size_mb:.2f} MB")

    def load_quantized_model(
        self, model: nn.Module, load_path: Path
    ) -> Tuple[nn.Module, Dict]:
        """Load quantized model.

        Args:
            model: Model architecture (will be loaded with quantized weights)
            load_path: Path to load from

        Returns:
            Loaded model and metadata
        """
        load_path = Path(load_path)

        if not load_path.exists():
            raise FileNotFoundError(f"Model not found: {load_path}")

        # Load checkpoint
        checkpoint = torch.load(load_path, map_location="cpu")

        # Load state dict
        model.load_state_dict(checkpoint["model_state_dict"])

        # Get metadata
        metadata = checkpoint.get("metadata", {})

        logger.info(f"Quantized model loaded from {load_path}")

        return model, metadata


def quantize_attention_mil(
    model: nn.Module,
    calibration_data: Optional[torch.utils.data.DataLoader] = None,
    method: str = "dynamic",
) -> nn.Module:
    """Quantize AttentionMIL model.

    Args:
        model: AttentionMIL model
        calibration_data: Calibration data (for static quantization)
        method: Quantization method ('dynamic', 'static', or 'fp16')

    Returns:
        Quantized model
    """
    quantizer = ModelQuantizer()

    if method == "dynamic":
        # Dynamic quantization for attention layers
        quantized_model = quantizer.quantize_dynamic(
            model, dtype=torch.qint8, modules_to_quantize={nn.Linear}
        )
    elif method == "static":
        if calibration_data is None:
            raise ValueError("Calibration data required for static quantization")
        quantized_model = quantizer.quantize_static(model, calibration_data)
    elif method == "fp16":
        quantized_model = quantizer.quantize_to_fp16(model)
    else:
        raise ValueError(f"Unknown quantization method: {method}")

    return quantized_model


if __name__ == "__main__":
    # Quick test
    logging.basicConfig(level=logging.INFO)

    # Create simple test model
    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc1 = nn.Linear(100, 50)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(50, 10)

        def forward(self, x):
            x = self.fc1(x)
            x = self.relu(x)
            x = self.fc2(x)
            return x

    # Create model
    model = SimpleModel()
    model.eval()

    # Create quantizer
    quantizer = ModelQuantizer()

    # Test dynamic quantization
    quantized_model = quantizer.quantize_dynamic(model)

    # Test input
    test_input = torch.randn(1, 100)

    # Compare models
    results = quantizer.compare_models(model, quantized_model, test_input, num_runs=50)

    print(f"\nQuantization Results:")
    print(f"  Speedup: {results['improvements']['speedup']:.2f}x")
    print(f"  Memory reduction: {results['improvements']['memory_reduction']:.2f}x")
    print(
        f"  Original size: {results['original']['model_size'] / 1024:.2f} KB"
    )
    print(
        f"  Quantized size: {results['quantized']['model_size'] / 1024:.2f} KB"
    )
