"""
Mobile Inference Engines for Medical AI

Cross-platform mobile inference with TFLite, ONNX Runtime Mobile, PyTorch Mobile.
"""

import gc
import logging
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import psutil
import torch
import torch.nn as nn


class InferenceBackend(Enum):
    """Mobile inference backends"""

    TFLITE = "tflite"  # TensorFlow Lite
    ONNX_MOBILE = "onnx_mobile"  # ONNX Runtime Mobile
    PYTORCH_MOBILE = "pytorch_mobile"  # PyTorch Mobile
    COREML = "coreml"  # Apple CoreML
    NCNN = "ncnn"  # Tencent NCNN


@dataclass
class MobileInferenceConfig:
    """Mobile inference config"""

    backend: InferenceBackend
    num_threads: int = 4  # CPU threads
    use_gpu: bool = False  # GPU acceleration
    use_npu: bool = False  # NPU acceleration (mobile)
    batch_size: int = 1  # Batch size
    warmup_runs: int = 5  # Warmup iterations
    benchmark_runs: int = 100  # Benchmark iterations

    def __post_init__(self):
        if not (self.num_threads > 0):
            raise AssertionError("self.num_threads > 0")
        if not (self.batch_size > 0):
            raise AssertionError("self.batch_size > 0")


@dataclass
class InferenceResult:
    """Inference result"""

    predictions: np.ndarray
    inference_time_ms: float
    preprocessing_time_ms: float
    postprocessing_time_ms: float
    total_time_ms: float

    def to_dict(self) -> Dict:
        return {
            "predictions": self.predictions.tolist(),
            "inference_time_ms": self.inference_time_ms,
            "preprocessing_time_ms": self.preprocessing_time_ms,
            "postprocessing_time_ms": self.postprocessing_time_ms,
            "total_time_ms": self.total_time_ms,
        }


@dataclass
class BenchmarkResult:
    """Benchmark result"""

    backend: str
    avg_inference_time_ms: float
    min_inference_time_ms: float
    max_inference_time_ms: float
    std_inference_time_ms: float
    throughput_fps: float
    memory_mb: float

    def to_dict(self) -> Dict:
        return {
            "backend": self.backend,
            "avg_inference_time_ms": self.avg_inference_time_ms,
            "min_inference_time_ms": self.min_inference_time_ms,
            "max_inference_time_ms": self.max_inference_time_ms,
            "std_inference_time_ms": self.std_inference_time_ms,
            "throughput_fps": self.throughput_fps,
            "memory_mb": self.memory_mb,
        }


class TFLiteInference:
    """
    TensorFlow Lite inference

    Optimized for mobile/edge devices

    Features:
    - INT8/FP16 quantization
    - GPU delegate (Android/iOS)
    - NNAPI delegate (Android)
    - CoreML delegate (iOS)
    - Hexagon delegate (Qualcomm)
    """

    def __init__(self, model_path: Path, config: MobileInferenceConfig):
        try:
            import tensorflow as tf

            self.tf = tf
            TFLITE_AVAILABLE = True
        except ImportError:
            raise RuntimeError("TensorFlow Lite not available")

        self.config = config
        self.logger = logging.getLogger(__name__)

        # Load model
        self.interpreter = self._load_model(model_path)

        # Get input/output details
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        self.logger.info(f"TFLite model loaded: {model_path}")

    def _load_model(self, model_path: Path):
        """Load TFLite model"""

        interpreter = self.tf.lite.Interpreter(
            model_path=str(model_path), num_threads=self.config.num_threads
        )

        interpreter.allocate_tensors()

        return interpreter

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference"""

        # Set input
        self.interpreter.set_tensor(self.input_details[0]["index"], input_data.astype(np.float32))

        # Invoke
        self.interpreter.invoke()

        # Get output
        output = self.interpreter.get_tensor(self.output_details[0]["index"])

        return output

    def benchmark(self, input_shape: Tuple[int, ...]) -> BenchmarkResult:
        """Benchmark inference with memory monitoring"""

        import time

        dummy_input = np.random.randn(*input_shape).astype(np.float32)

        # Memory monitoring
        process = psutil.Process()
        memory_before = process.memory_info().rss / 1024 / 1024  # MB

        # Warmup
        for _ in range(self.config.warmup_runs):
            self.predict(dummy_input)

        # Benchmark
        times = []
        for _ in range(self.config.benchmark_runs):
            start = time.perf_counter()
            self.predict(dummy_input)
            end = time.perf_counter()
            times.append((end - start) * 1000)

        # Memory after inference
        gc.collect()  # Force garbage collection
        memory_after = process.memory_info().rss / 1024 / 1024  # MB
        memory_used = max(0.0, memory_after - memory_before)  # Ensure non-negative

        times = np.array(times)

        result = BenchmarkResult(
            backend="tflite",
            avg_inference_time_ms=float(np.mean(times)),
            min_inference_time_ms=float(np.min(times)),
            max_inference_time_ms=float(np.max(times)),
            std_inference_time_ms=float(np.std(times)),
            throughput_fps=1000.0 / np.mean(times),
            memory_mb=memory_used,
        )

        return result


class ONNXMobileInference:
    """
    ONNX Runtime Mobile inference

    Cross-platform mobile inference

    Features:
    - CPU/GPU/NPU execution
    - INT8/FP16 quantization
    - Dynamic shape support
    - Small binary size (~1-5MB)
    """

    def __init__(self, model_path: Path, config: MobileInferenceConfig):
        try:
            import onnxruntime as ort

            self.ort = ort
        except ImportError:
            raise RuntimeError("ONNX Runtime not available")

        self.config = config
        self.logger = logging.getLogger(__name__)

        # Create session
        self.session = self._create_session(model_path)

        # Get input/output names
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        self.logger.info(f"ONNX Runtime Mobile loaded: {model_path}")

    def _create_session(self, model_path: Path):
        """Create ONNX Runtime session"""

        # Session options
        sess_options = self.ort.SessionOptions()
        sess_options.intra_op_num_threads = self.config.num_threads
        sess_options.graph_optimization_level = self.ort.GraphOptimizationLevel.ORT_ENABLE_ALL

        # Execution providers
        providers = []
        if self.config.use_gpu:
            providers.append("CUDAExecutionProvider")
        providers.append("CPUExecutionProvider")

        # Create session
        session = self.ort.InferenceSession(
            str(model_path), sess_options=sess_options, providers=providers
        )

        return session

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference"""

        outputs = self.session.run(
            [self.output_name], {self.input_name: input_data.astype(np.float32)}
        )

        return outputs[0]

    def benchmark(self, input_shape: Tuple[int, ...]) -> BenchmarkResult:
        """Benchmark inference"""

        import time

        dummy_input = np.random.randn(*input_shape).astype(np.float32)

        # Warmup
        for _ in range(self.config.warmup_runs):
            self.predict(dummy_input)

        # Benchmark
        times = []
        for _ in range(self.config.benchmark_runs):
            start = time.perf_counter()
            self.predict(dummy_input)
            end = time.perf_counter()
            times.append((end - start) * 1000)

        times = np.array(times)

        result = BenchmarkResult(
            backend="onnx_mobile",
            avg_inference_time_ms=float(np.mean(times)),
            min_inference_time_ms=float(np.min(times)),
            max_inference_time_ms=float(np.max(times)),
            std_inference_time_ms=float(np.std(times)),
            throughput_fps=1000.0 / np.mean(times),
            memory_mb=0.0,
        )

        return result


class PyTorchMobileInference:
    """
    PyTorch Mobile inference

    Native PyTorch mobile deployment

    Features:
    - TorchScript optimization
    - Quantization support
    - Metal backend (iOS)
    - Vulkan backend (Android)
    """

    def __init__(self, model_path: Path, config: MobileInferenceConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Load model
        self.model = torch.jit.load(str(model_path))
        self.model.eval()

        self.logger.info(f"PyTorch Mobile loaded: {model_path}")

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference"""

        # Convert to tensor
        input_tensor = torch.from_numpy(input_data).float()

        # Inference
        with torch.no_grad():
            output = self.model(input_tensor)

        return output.numpy()

    def benchmark(self, input_shape: Tuple[int, ...]) -> BenchmarkResult:
        """Benchmark inference"""

        import time

        dummy_input = np.random.randn(*input_shape).astype(np.float32)

        # Warmup
        for _ in range(self.config.warmup_runs):
            self.predict(dummy_input)

        # Benchmark
        times = []
        for _ in range(self.config.benchmark_runs):
            start = time.perf_counter()
            self.predict(dummy_input)
            end = time.perf_counter()
            times.append((end - start) * 1000)

        times = np.array(times)

        result = BenchmarkResult(
            backend="pytorch_mobile",
            avg_inference_time_ms=float(np.mean(times)),
            min_inference_time_ms=float(np.min(times)),
            max_inference_time_ms=float(np.max(times)),
            std_inference_time_ms=float(np.std(times)),
            throughput_fps=1000.0 / np.mean(times),
            memory_mb=0.0,
        )

        return result


class MobileInferenceEngine:
    """
    Unified mobile inference engine

    Supports multiple backends with automatic fallback
    """

    def __init__(self, model_path: Path, config: MobileInferenceConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Create backend
        self.backend = self._create_backend(model_path)

    def _create_backend(self, model_path: Path):
        """Create inference backend"""

        backend_map = {
            InferenceBackend.TFLITE: TFLiteInference,
            InferenceBackend.ONNX_MOBILE: ONNXMobileInference,
            InferenceBackend.PYTORCH_MOBILE: PyTorchMobileInference,
        }

        backend_class = backend_map.get(self.config.backend)

        if backend_class is None:
            raise ValueError(f"Unsupported backend: {self.config.backend}")

        try:
            return backend_class(model_path, self.config)
        except Exception as e:
            self.logger.error(f"Failed to create {self.config.backend}: {e}")
            raise

    def predict(self, input_data: np.ndarray) -> np.ndarray:
        """Run inference"""
        return self.backend.predict(input_data)

    def benchmark(self, input_shape: Tuple[int, ...]) -> BenchmarkResult:
        """Benchmark inference"""
        return self.backend.benchmark(input_shape)


def convert_pytorch_to_tflite(
    model: nn.Module, input_shape: Tuple[int, ...], save_path: Path, quantize: bool = False
) -> Path:
    """Convert PyTorch to TFLite"""

    try:
        import onnx
        import tensorflow as tf
        from onnx_tf.backend import prepare
    except ImportError:
        raise RuntimeError("TensorFlow/ONNX-TF not available")

    # Export to ONNX
    onnx_path = save_path.parent / "temp.onnx"

    model.eval()
    dummy_input = torch.randn(input_shape)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        opset_version=13,
        input_names=["input"],
        output_names=["output"],
    )

    # Convert ONNX to TensorFlow
    onnx_model = onnx.load(str(onnx_path))
    tf_rep = prepare(onnx_model)

    # Export to SavedModel
    saved_model_path = save_path.parent / "saved_model"
    tf_rep.export_graph(str(saved_model_path))

    # Convert to TFLite
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model_path))

    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    # Save
    with open(save_path, "wb") as f:
        f.write(tflite_model)

    # Cleanup
    onnx_path.unlink()

    logging.info(f"Converted to TFLite: {save_path}")

    return save_path


def convert_pytorch_to_mobile(
    model: nn.Module, input_shape: Tuple[int, ...], save_path: Path
) -> Path:
    """Convert PyTorch to PyTorch Mobile"""

    model.eval()

    # Trace model
    dummy_input = torch.randn(input_shape)
    traced_model = torch.jit.trace(model, dummy_input)

    # Optimize for mobile
    optimized_model = torch.jit.optimize_for_inference(traced_model)

    # Save
    optimized_model._save_for_lite_interpreter(str(save_path))

    logging.info(f"Converted to PyTorch Mobile: {save_path}")

    return save_path


def benchmark_all_backends(
    model: nn.Module, input_shape: Tuple[int, ...], model_dir: Path
) -> List[BenchmarkResult]:
    """Benchmark all mobile backends"""

    results = []

    # Convert to all formats
    formats = {
        "onnx": model_dir / "model.onnx",
        "pytorch_mobile": model_dir / "model.ptl",
        "tflite": model_dir / "model.tflite",
    }

    # Export ONNX
    from .onnx_exporter import export_to_onnx

    export_to_onnx(model, input_shape, formats["onnx"])

    # Export PyTorch Mobile
    convert_pytorch_to_mobile(model, input_shape, formats["pytorch_mobile"])

    # Export TFLite (optional)
    try:
        convert_pytorch_to_tflite(model, input_shape, formats["tflite"])
    except Exception as e:
        logging.warning(f"TFLite conversion failed: {e}")

    # Benchmark each backend
    backends = [
        (InferenceBackend.ONNX_MOBILE, formats["onnx"]),
        (InferenceBackend.PYTORCH_MOBILE, formats["pytorch_mobile"]),
    ]

    if formats["tflite"].exists():
        backends.append((InferenceBackend.TFLITE, formats["tflite"]))

    for backend, model_path in backends:
        try:
            config = MobileInferenceConfig(
                backend=backend, num_threads=4, warmup_runs=5, benchmark_runs=100
            )

            engine = MobileInferenceEngine(model_path, config)
            result = engine.benchmark(input_shape)
            results.append(result)

        except Exception as e:
            logging.warning(f"Benchmark failed for {backend}: {e}")

    return results


def compare_backends(results: List[BenchmarkResult]) -> Dict:
    """Compare backend performance"""

    comparison = {
        "backends": [r.backend for r in results],
        "avg_times_ms": [r.avg_inference_time_ms for r in results],
        "throughputs_fps": [r.throughput_fps for r in results],
    }

    # Find fastest
    fastest_idx = np.argmin([r.avg_inference_time_ms for r in results])
    comparison["fastest_backend"] = results[fastest_idx].backend
    comparison["fastest_time_ms"] = results[fastest_idx].avg_inference_time_ms

    return comparison


# Medical AI mobile inference
def create_medical_mobile_engine(
    model_path: Path, backend: str = "onnx_mobile", num_threads: int = 4
) -> MobileInferenceEngine:
    """Create mobile inference engine for medical AI"""

    backend_enum = InferenceBackend(backend)

    config = MobileInferenceConfig(
        backend=backend_enum,
        num_threads=num_threads,
        use_gpu=False,  # CPU for medical reliability
        batch_size=1,  # Single image inference
        warmup_runs=5,
        benchmark_runs=100,
    )

    engine = MobileInferenceEngine(model_path, config)

    return engine
