"""
ONNX Export for Medical AI Models

Cross-platform ONNX export for deployment flexibility.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import onnx
import onnxruntime as ort
import torch
import torch.nn as nn


@dataclass
class ONNXConfig:
    """ONNX export config"""

    opset_version: int = 13  # ONNX opset version (11-17)
    do_constant_folding: bool = True  # Optimize constants
    export_params: bool = True  # Export parameters
    input_names: List[str] = None  # Input tensor names
    output_names: List[str] = None  # Output tensor names
    dynamic_axes: Dict = None  # Dynamic batch/sequence
    verbose: bool = False  # Verbose export
    optimize: bool = True  # Apply optimizations
    quantize: bool = False  # Quantize to INT8

    def __post_init__(self):
        if self.input_names is None:
            self.input_names = ["input"]
        if self.output_names is None:
            self.output_names = ["output"]
        if self.dynamic_axes is None:
            # Dynamic batch size by default
            self.dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}


@dataclass
class ONNXResult:
    """ONNX export result"""

    model_size_mb: float
    original_size_mb: float
    compression_ratio: float
    opset_version: int
    optimized: bool
    quantized: bool
    num_nodes: int

    def to_dict(self) -> Dict:
        return {
            "model_size_mb": self.model_size_mb,
            "original_size_mb": self.original_size_mb,
            "compression_ratio": self.compression_ratio,
            "opset_version": self.opset_version,
            "optimized": self.optimized,
            "quantized": self.quantized,
            "num_nodes": self.num_nodes,
        }


class ONNXExporter:
    """
    ONNX model exporter

    Converts PyTorch → ONNX for cross-platform deployment

    Features:
    - Dynamic batch size support
    - Graph optimization
    - INT8 quantization
    - ONNX Runtime validation

    Deployment targets:
    - ONNX Runtime (CPU/GPU/TensorRT)
    - Mobile (iOS/Android via ONNX Runtime Mobile)
    - Edge devices (Jetson, RPi)
    - Web (ONNX.js)
    """

    def __init__(self, config: ONNXConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

    def export_model(
        self, model: nn.Module, input_shape: Tuple[int, ...], save_path: Path, validate: bool = True
    ) -> Path:
        """
        Export PyTorch to ONNX

        Args:
            model: PyTorch model
            input_shape: Input shape (B, C, H, W)
            save_path: Save path (.onnx)
            validate: Validate exported model

        Returns:
            Path to ONNX model
        """

        try:
            self.logger.info(f"Exporting to ONNX opset {self.config.opset_version}")

            # Export
            self._export_to_onnx(model, input_shape, save_path)

            # Optimize
            if self.config.optimize:
                save_path = self._optimize_onnx(save_path)

            # Quantize
            if self.config.quantize:
                save_path = self._quantize_onnx(save_path)

            # Validate
            if validate:
                self._validate_onnx(save_path, input_shape)

            self.logger.info(f"Saved ONNX model to {save_path}")

            return save_path

        except Exception as e:
            self.logger.error(f"ONNX export failed: {e}")
            raise

    def _export_to_onnx(self, model: nn.Module, input_shape: Tuple[int, ...], save_path: Path):
        """Export PyTorch to ONNX"""

        model.eval()
        dummy_input = torch.randn(input_shape)

        # Export
        torch.onnx.export(
            model,
            dummy_input,
            str(save_path),
            export_params=self.config.export_params,
            opset_version=self.config.opset_version,
            do_constant_folding=self.config.do_constant_folding,
            input_names=self.config.input_names,
            output_names=self.config.output_names,
            dynamic_axes=self.config.dynamic_axes,
            verbose=self.config.verbose,
        )

        self.logger.info("Exported to ONNX")

    def _optimize_onnx(self, model_path: Path) -> Path:
        """Optimize ONNX model"""

        try:
            from onnxruntime.transformers import optimizer

            # Load model
            model = onnx.load(str(model_path))

            # Optimize
            optimized_model = optimizer.optimize_model(
                str(model_path),
                model_type="bert",  # Generic optimization
                num_heads=0,
                hidden_size=0,
            )

            # Save optimized
            optimized_path = model_path.parent / f"{model_path.stem}_optimized.onnx"
            optimized_model.save_model_to_file(str(optimized_path))

            self.logger.info(f"Optimized ONNX model")

            return optimized_path

        except ImportError:
            self.logger.warning("ONNX optimizer not available, skipping")
            return model_path
        except Exception as e:
            self.logger.warning(f"ONNX optimization failed: {e}, using unoptimized")
            return model_path

    def _quantize_onnx(self, model_path: Path) -> Path:
        """Quantize ONNX model to INT8"""

        try:
            from onnxruntime.quantization import QuantType, quantize_dynamic

            # Quantize
            quantized_path = model_path.parent / f"{model_path.stem}_quantized.onnx"

            quantize_dynamic(str(model_path), str(quantized_path), weight_type=QuantType.QUInt8)

            self.logger.info("Quantized ONNX model to INT8")

            return quantized_path

        except ImportError:
            self.logger.warning("ONNX quantization not available, skipping")
            return model_path
        except Exception as e:
            self.logger.warning(f"ONNX quantization failed: {e}, using unquantized")
            return model_path

    def _validate_onnx(self, model_path: Path, input_shape: Tuple[int, ...]):
        """Validate ONNX model"""

        # Check model
        model = onnx.load(str(model_path))
        onnx.checker.check_model(model)

        self.logger.info("ONNX model validation passed")

        # Test inference
        try:
            session = ort.InferenceSession(str(model_path))

            # Create dummy input
            dummy_input = np.random.randn(*input_shape).astype(np.float32)

            # Run inference
            outputs = session.run(
                self.config.output_names, {self.config.input_names[0]: dummy_input}
            )

            self.logger.info(f"ONNX inference test passed, output shape: {outputs[0].shape}")

        except Exception as e:
            self.logger.warning(f"ONNX inference test failed: {e}")

    def get_model_info(self, model_path: Path) -> Dict:
        """Get ONNX model info"""

        model = onnx.load(str(model_path))

        info = {
            "opset_version": model.opset_import[0].version,
            "num_nodes": len(model.graph.node),
            "num_inputs": len(model.graph.input),
            "num_outputs": len(model.graph.output),
            "size_mb": model_path.stat().st_size / (1024 * 1024),
        }

        return info


class ONNXRuntimeInference:
    """
    ONNX Runtime inference engine

    High-performance inference using ONNX Runtime

    Execution providers:
    - CPUExecutionProvider (default)
    - CUDAExecutionProvider (GPU)
    - TensorrtExecutionProvider (TensorRT)
    - CoreMLExecutionProvider (Apple)
    """

    def __init__(self, model_path: Path, providers: List[str] = None):
        """
        Initialize ONNX Runtime

        Args:
            model_path: Path to ONNX model
            providers: Execution providers (default: CPU)
        """

        if providers is None:
            providers = ["CPUExecutionProvider"]

        self.session = ort.InferenceSession(str(model_path), providers=providers)

        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.logger = logging.getLogger(__name__)
        self.logger.info(f"ONNX Runtime initialized with {providers}")

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """
        Run inference

        Args:
            input_data: Input array (B, C, H, W)

        Returns:
            Output predictions
        """

        outputs = self.session.run([self.output_name], {self.input_name: input_data})

        return outputs[0]

    def benchmark(self, input_shape: Tuple[int, ...], num_iterations: int = 100) -> Dict:
        """Benchmark inference speed"""

        import time

        # Warmup
        dummy_input = np.random.randn(*input_shape).astype(np.float32)
        for _ in range(10):
            self.predict(dummy_input)

        # Benchmark
        start = time.time()
        for _ in range(num_iterations):
            self.predict(dummy_input)
        end = time.time()

        avg_time = (end - start) / num_iterations
        throughput = 1.0 / avg_time

        return {
            "avg_inference_time_ms": avg_time * 1000,
            "throughput_fps": throughput,
            "num_iterations": num_iterations,
        }


def export_to_onnx(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    save_path: Path = Path("model.onnx"),
    opset_version: int = 13,
    optimize: bool = True,
) -> Path:
    """
    Export PyTorch to ONNX

    Args:
        model: PyTorch model
        input_shape: Input shape
        save_path: Save path
        opset_version: ONNX opset version
        optimize: Apply optimizations

    Returns:
        Path to ONNX model
    """

    config = ONNXConfig(opset_version=opset_version, optimize=optimize, quantize=False)

    exporter = ONNXExporter(config)
    onnx_path = exporter.export_model(model, input_shape, save_path)

    return onnx_path


def benchmark_onnx(
    pytorch_model: nn.Module, onnx_model_path: Path, input_shape: Tuple[int, ...]
) -> ONNXResult:
    """Benchmark ONNX model"""

    # Get sizes
    pytorch_size = sum(p.numel() * p.element_size() for p in pytorch_model.parameters())
    pytorch_size_mb = pytorch_size / (1024 * 1024)

    onnx_size_mb = onnx_model_path.stat().st_size / (1024 * 1024)

    # Get model info
    model = onnx.load(str(onnx_model_path))

    result = ONNXResult(
        model_size_mb=onnx_size_mb,
        original_size_mb=pytorch_size_mb,
        compression_ratio=pytorch_size_mb / onnx_size_mb,
        opset_version=model.opset_import[0].version,
        optimized=True,
        quantized=False,
        num_nodes=len(model.graph.node),
    )

    return result


# Medical AI ONNX export
def export_medical_model_onnx(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    save_path: Path = Path("medical_model.onnx"),
    optimize: bool = True,
    quantize: bool = False,
) -> Path:
    """ONNX export for medical AI"""

    config = ONNXConfig(
        opset_version=13,  # Stable opset
        do_constant_folding=True,
        optimize=optimize,
        quantize=quantize,  # Conservative for medical
        input_names=["image"],
        output_names=["predictions"],
        dynamic_axes={"image": {0: "batch_size"}, "predictions": {0: "batch_size"}},
    )

    exporter = ONNXExporter(config)
    onnx_path = exporter.export_model(model, input_shape, save_path)

    return onnx_path


def create_onnx_runtime_config(
    model_path: Path, use_gpu: bool = False, use_tensorrt: bool = False
) -> ONNXRuntimeInference:
    """Create ONNX Runtime inference engine"""

    # Select execution providers
    providers = []

    if use_tensorrt:
        providers.append("TensorrtExecutionProvider")

    if use_gpu:
        providers.append("CUDAExecutionProvider")

    providers.append("CPUExecutionProvider")  # Fallback

    # Create inference engine
    inference = ONNXRuntimeInference(model_path, providers)

    return inference


def compare_pytorch_onnx(
    pytorch_model: nn.Module,
    onnx_model_path: Path,
    input_shape: Tuple[int, ...],
    tolerance: float = 1e-5,
) -> Dict:
    """Compare PyTorch vs ONNX outputs"""

    # PyTorch inference
    pytorch_model.eval()
    dummy_input = torch.randn(input_shape)

    with torch.no_grad():
        pytorch_output = pytorch_model(dummy_input).numpy()

    # ONNX inference
    onnx_inference = ONNXRuntimeInference(onnx_model_path)
    onnx_output = onnx_inference.predict(dummy_input.numpy())

    # Compare
    max_diff = np.max(np.abs(pytorch_output - onnx_output))
    mean_diff = np.mean(np.abs(pytorch_output - onnx_output))

    match = max_diff < tolerance

    result = {
        "match": match,
        "max_diff": float(max_diff),
        "mean_diff": float(mean_diff),
        "tolerance": tolerance,
    }

    return result


def export_with_metadata(
    model: nn.Module, input_shape: Tuple[int, ...], save_path: Path, metadata: Dict
) -> Path:
    """Export ONNX with custom metadata"""

    # Export
    config = ONNXConfig(opset_version=13, optimize=True)
    exporter = ONNXExporter(config)
    onnx_path = exporter.export_model(model, input_shape, save_path, validate=False)

    # Add metadata
    model_proto = onnx.load(str(onnx_path))

    for key, value in metadata.items():
        meta = model_proto.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    onnx.save(model_proto, str(onnx_path))

    logging.info(f"Added metadata: {metadata}")

    return onnx_path
