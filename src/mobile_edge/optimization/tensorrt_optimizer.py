"""
TensorRT Optimization for Medical AI Models

NVIDIA TensorRT optimization for GPU inference acceleration.
"""

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

try:
    import pycuda.autoinit
    import pycuda.driver as cuda
    import tensorrt as trt

    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False
    logging.warning("TensorRT not available")


@dataclass
class TensorRTConfig:
    """TensorRT optimization config"""

    precision: str = "fp16"  # fp32/fp16/int8
    max_batch_size: int = 32  # Max batch size
    max_workspace_size: int = 1 << 30  # 1GB workspace
    strict_type_constraints: bool = False  # Strict type checking
    enable_dla: bool = False  # Deep learning accelerator
    dla_core: int = 0  # DLA core ID
    calibration_cache: str = None  # INT8 calibration cache
    min_timing_iterations: int = 2  # Timing iterations
    avg_timing_iterations: int = 2  # Average timing

    def __post_init__(self):
        if not (self.precision in ["fp32", "fp16"):
            raise ValueError("int8")


@dataclass
class TensorRTResult:
    """TensorRT optimization result"""

    original_latency_ms: float
    optimized_latency_ms: float
    speedup: float
    original_throughput: float
    optimized_throughput: float
    precision: str
    engine_size_mb: float

    def to_dict(self) -> Dict:
        return {
            "original_latency_ms": self.original_latency_ms,
            "optimized_latency_ms": self.optimized_latency_ms,
            "speedup": self.speedup,
            "original_throughput": self.original_throughput,
            "optimized_throughput": self.optimized_throughput,
            "precision": self.precision,
            "engine_size_mb": self.engine_size_mb,
        }


class TensorRTOptimizer:
    """
    TensorRT model optimizer

    Optimizations:
    - Layer fusion (conv+bn+relu)
    - Kernel auto-tuning
    - Precision calibration (FP16/INT8)
    - Memory optimization
    - Dynamic tensor memory

    Typical speedup: 2-5x over PyTorch
    """

    def __init__(self, config: TensorRTConfig):
        if not TRT_AVAILABLE:
            raise RuntimeError("TensorRT not available")

        self.config = config
        self.logger = logging.getLogger(__name__)

        # TensorRT components
        self.trt_logger = trt.Logger(trt.Logger.WARNING)
        self.builder = None
        self.network = None
        self.engine = None
        self.context = None

    def optimize_model(
        self,
        model: nn.Module,
        input_shape: Tuple[int, ...],
        save_path: Path = None,
        calibration_dataloader=None,
    ) -> "TensorRTEngine":
        """
        Optimize PyTorch model with TensorRT

        Args:
            model: PyTorch model
            input_shape: Input tensor shape (B, C, H, W)
            save_path: Path to save engine
            calibration_dataloader: DataLoader for INT8 calibration (required for INT8)

        Returns:
            TensorRTEngine for inference
        """

        try:
            self.logger.info(f"Optimizing model with TensorRT {self.config.precision}")

            # Create calibration dataset for INT8
            if self.config.precision == "int8":
                if calibration_dataloader is None:
                    raise ValueError("INT8 precision requires calibration_dataloader")
                self._calibration_data = self.create_calibration_dataset(calibration_dataloader)
                self.config.calibration_cache = "calibration.cache"

            # Export to ONNX first
            onnx_path = Path("temp_model.onnx")
            self._export_onnx(model, input_shape, onnx_path)

            # Build TensorRT engine
            engine = self._build_engine(onnx_path, input_shape)

            # Save engine
            if save_path:
                self._save_engine(engine, save_path)

            # Create inference wrapper
            trt_engine = TensorRTEngine(engine, input_shape)

            self.logger.info("TensorRT optimization complete")

            return trt_engine

        except Exception as e:
            self.logger.error(f"TensorRT optimization failed: {e}")
            raise

    def _export_onnx(self, model: nn.Module, input_shape: Tuple[int, ...], onnx_path: Path):
        """Export PyTorch to ONNX"""

        model.eval()
        dummy_input = torch.randn(input_shape)

        torch.onnx.export(
            model,
            dummy_input,
            str(onnx_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        )

        self.logger.info(f"Exported ONNX to {onnx_path}")

    def _build_engine(self, onnx_path: Path, input_shape: Tuple[int, ...]) -> trt.ICudaEngine:
        """Build TensorRT engine from ONNX"""

        # Create builder
        builder = trt.Builder(self.trt_logger)
        network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        parser = trt.OnnxParser(network, self.trt_logger)

        # Parse ONNX
        with open(onnx_path, "rb") as f:
            if not parser.parse(f.read()):
                for error in range(parser.num_errors):
                    self.logger.error(parser.get_error(error))
                raise RuntimeError("ONNX parsing failed")

        # Builder config
        config = builder.create_builder_config()
        config.max_workspace_size = self.config.max_workspace_size

        # Precision
        if self.config.precision == "fp16":
            config.set_flag(trt.BuilderFlag.FP16)
        elif self.config.precision == "int8":
            config.set_flag(trt.BuilderFlag.INT8)
            # INT8 calibration needed
            if self.config.calibration_cache:
                # Load calibration data if available
                calibration_data = getattr(self, "_calibration_data", None)
                if calibration_data:
                    config.int8_calibrator = self._create_calibrator(calibration_data)
                else:
                    raise RuntimeError(
                        "INT8 calibration requires calibration dataset. Use create_calibration_dataset() first."
                    )

        # Timing cache
        (
            config.set_flag(trt.BuilderFlag.STRICT_TYPES)
            if self.config.strict_type_constraints
            else None
        )

        # Build engine
        self.logger.info("Building TensorRT engine (may take minutes)...")
        engine = builder.build_engine(network, config)

        if engine is None:
            raise RuntimeError("Engine build failed")

        self.logger.info("Engine built successfully")

        return engine

    def _create_calibrator(self, calibration_data: List[torch.Tensor]):
        """Create INT8 calibrator with real calibration dataset."""
        return self._create_int8_calibrator(calibration_data)

    def _create_int8_calibrator(self, calibration_data: List[torch.Tensor]) -> trt.IInt8Calibrator:
        """Create INT8 calibrator with calibration dataset."""
        import os

        class PythonEntropyCalibrator(trt.IInt8EntropyCalibrator2):
            def __init__(self, calibration_data: List[torch.Tensor], cache_file: str):
                trt.IInt8EntropyCalibrator2.__init__(self)
                self.calibration_data = calibration_data
                self.cache_file = cache_file
                self.current_index = 0
                self.device_input = None

            def get_batch_size(self):
                return 1

            def get_batch(self, names):
                if self.current_index >= len(self.calibration_data):
                    return None

                batch = self.calibration_data[self.current_index]
                self.current_index += 1

                # Allocate device memory if needed
                if self.device_input is None:
                    self.device_input = cuda.mem_alloc(batch.nbytes)

                # Copy to device
                cuda.memcpy_htod(self.device_input, batch.numpy())
                return [self.device_input]

            def read_calibration_cache(self):
                if os.path.exists(self.cache_file):
                    with open(self.cache_file, "rb") as f:
                        return f.read()
                return None

            def write_calibration_cache(self, cache):
                with open(self.cache_file, "wb") as f:
                    f.write(cache)

        cache_file = f"calibration_cache_{hash(str(calibration_data))}.cache"
        return PythonEntropyCalibrator(calibration_data, cache_file)

    def create_calibration_dataset(self, dataloader, num_samples: int = 500) -> List[torch.Tensor]:
        """
        Create calibration dataset from dataloader.

        Args:
            dataloader: PyTorch DataLoader
            num_samples: Number of calibration samples

        Returns:
            List of calibration tensors
        """
        calibration_data = []

        self.logger.info(f"Creating calibration dataset with {num_samples} samples")

        with torch.no_grad():
            for i, (data, _) in enumerate(dataloader):
                if i >= num_samples:
                    break

                # Take first sample from batch
                if isinstance(data, torch.Tensor):
                    sample = data[0:1]  # Keep batch dimension
                else:
                    sample = data[0][0:1]  # Handle tuple inputs

                calibration_data.append(sample.cpu())

                if (i + 1) % 100 == 0:
                    self.logger.info(f"Collected {i + 1}/{num_samples} calibration samples")

        self.logger.info(f"Calibration dataset created with {len(calibration_data)} samples")
        return calibration_data

    def _save_engine(self, engine: trt.ICudaEngine, save_path: Path):
        """Save TensorRT engine"""

        with open(save_path, "wb") as f:
            f.write(engine.serialize())

        self.logger.info(f"Saved engine to {save_path}")

    def load_engine(self, engine_path: Path) -> trt.ICudaEngine:
        """Load TensorRT engine"""

        runtime = trt.Runtime(self.trt_logger)

        with open(engine_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())

        if engine is None:
            raise RuntimeError("Engine load failed")

        self.logger.info(f"Loaded engine from {engine_path}")

        return engine


class TensorRTEngine:
    """TensorRT inference engine wrapper"""

    def __init__(self, engine: trt.ICudaEngine, input_shape: Tuple[int, ...]):
        self.engine = engine
        self.context = engine.create_execution_context()
        self.input_shape = input_shape
        self.logger = logging.getLogger(__name__)

        # Allocate buffers
        self.inputs, self.outputs, self.bindings, self.stream = self._allocate_buffers()

    def _allocate_buffers(self):
        """Allocate GPU buffers"""

        inputs = []
        outputs = []
        bindings = []
        stream = cuda.Stream()

        for binding in self.engine:
            size = trt.volume(self.engine.get_binding_shape(binding)) * self.engine.max_batch_size
            dtype = trt.nptype(self.engine.get_binding_dtype(binding))

            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)

            bindings.append(int(device_mem))

            if self.engine.binding_is_input(binding):
                inputs.append({"host": host_mem, "device": device_mem})
            else:
                outputs.append({"host": host_mem, "device": device_mem})

        return inputs, outputs, bindings, stream

    def infer(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference"""

        # Copy input to device
        np.copyto(self.inputs[0]["host"], input_data.ravel())
        cuda.memcpy_htod_async(self.inputs[0]["device"], self.inputs[0]["host"], self.stream)

        # Run inference
        self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)

        # Copy output to host
        cuda.memcpy_dtoh_async(self.outputs[0]["host"], self.outputs[0]["device"], self.stream)
        self.stream.synchronize()

        return self.outputs[0]["host"]

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """PyTorch-style inference"""

        # Convert to numpy
        x_np = x.cpu().numpy()

        # Infer
        output_np = self.infer(x_np)

        # Convert back to torch
        output = torch.from_numpy(output_np).to(x.device)

        return output


def optimize_with_tensorrt(
    model: nn.Module,
    input_shape: Tuple[int, ...] = (1, 3, 224, 224),
    precision: str = "fp16",
    save_path: Path = None,
) -> TensorRTEngine:
    """
    Optimize model with TensorRT

    Args:
        model: PyTorch model
        input_shape: Input shape
        precision: fp32/fp16/int8
        save_path: Save path

    Returns:
        TensorRT engine
    """

    config = TensorRTConfig(
        precision=precision, max_batch_size=input_shape[0], max_workspace_size=1 << 30
    )

    optimizer = TensorRTOptimizer(config)
    engine = optimizer.optimize_model(model, input_shape, save_path)

    return engine


def benchmark_tensorrt(
    pytorch_model: nn.Module,
    trt_engine: TensorRTEngine,
    input_shape: Tuple[int, ...],
    num_iterations: int = 100,
) -> TensorRTResult:
    """Benchmark TensorRT vs PyTorch"""

    import time

    device = torch.device("cuda")
    dummy_input = torch.randn(input_shape).to(device)

    # PyTorch benchmark
    pytorch_model.eval()
    pytorch_model = pytorch_model.to(device)

    # Warmup
    for _ in range(10):
        _ = pytorch_model(dummy_input)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        _ = pytorch_model(dummy_input)
    torch.cuda.synchronize()
    pytorch_time = (time.time() - start) / num_iterations

    # TensorRT benchmark
    # Warmup
    for _ in range(10):
        _ = trt_engine(dummy_input)

    torch.cuda.synchronize()
    start = time.time()
    for _ in range(num_iterations):
        _ = trt_engine(dummy_input)
    torch.cuda.synchronize()
    trt_time = (time.time() - start) / num_iterations

    # Results
    result = TensorRTResult(
        original_latency_ms=pytorch_time * 1000,
        optimized_latency_ms=trt_time * 1000,
        speedup=pytorch_time / trt_time,
        original_throughput=1.0 / pytorch_time,
        optimized_throughput=1.0 / trt_time,
        precision="fp16",
        engine_size_mb=0.0,  # Placeholder
    )

    return result


# Medical AI TensorRT optimization
def optimize_medical_model_tensorrt(
    model: nn.Module, input_shape: Tuple[int, ...] = (1, 3, 224, 224), save_path: Path = None
) -> TensorRTEngine:
    """TensorRT optimization for medical AI"""

    config = TensorRTConfig(
        precision="fp16",  # FP16 for medical precision
        max_batch_size=input_shape[0],
        max_workspace_size=2 << 30,  # 2GB workspace
        strict_type_constraints=True,  # Strict types for medical
        min_timing_iterations=4,  # More timing for stability
        avg_timing_iterations=4,
    )

    optimizer = TensorRTOptimizer(config)
    engine = optimizer.optimize_model(model, input_shape, save_path)

    return engine
