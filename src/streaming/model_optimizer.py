"""
Model optimization for HistoCore Real-Time WSI Streaming.

Provides TensorRT integration, model quantization (INT8, FP16), ONNX export,
and multi-GPU data/pipeline parallelism for <30s processing acceleration.
"""

import logging
import os
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from torch.nn.parallel import DataParallel, DistributedDataParallel

from .metrics import (
    record_processing_time,
    record_throughput_measurement,
    timed_operation,
    track_gpu_memory,
)

logger = logging.getLogger(__name__)

# Optional TensorRT imports
try:
    import tensorrt as trt
    import pycuda.driver as cuda
    import pycuda.autoinit
    TENSORRT_AVAILABLE = True
    logger.info("TensorRT available: version %s", trt.__version__)
except ImportError:
    TENSORRT_AVAILABLE = False
    logger.warning("TensorRT not available. Install with: pip install tensorrt pycuda")

# Optional ONNX imports
try:
    import onnx
    import onnxruntime as ort
    ONNX_AVAILABLE = True
    logger.info("ONNX available: onnx=%s onnxruntime=%s", onnx.__version__, ort.__version__)
except ImportError:
    ONNX_AVAILABLE = False
    logger.warning("ONNX not available. Install with: pip install onnx onnxruntime-gpu")


@dataclass
class OptimizationConfig:
    """Configuration for model optimization."""
    
    # TensorRT optimization
    enable_tensorrt: bool = True
    tensorrt_precision: str = "fp16"  # fp32, fp16, int8
    tensorrt_workspace_gb: float = 4.0
    tensorrt_max_batch_size: int = 64
    tensorrt_cache_dir: str = "./tensorrt_cache"
    
    # Quantization
    enable_quantization: bool = True
    quantization_mode: str = "dynamic"  # dynamic, static, qat
    calibration_dataset_size: int = 1000
    
    # ONNX export
    enable_onnx: bool = True
    onnx_opset_version: int = 17
    onnx_dynamic_axes: bool = True
    
    # Multi-GPU
    enable_data_parallel: bool = True
    enable_pipeline_parallel: bool = False
    gpu_ids: Optional[List[int]] = None
    
    # Performance
    enable_torch_compile: bool = True  # PyTorch 2.0+
    torch_compile_mode: str = "default"  # default, reduce-overhead, max-autotune
    enable_channels_last: bool = True
    enable_mixed_precision: bool = True


class TensorRTOptimizer:
    """TensorRT model optimization and inference."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize TensorRT optimizer."""
        self.config = config
        self.engine = None
        self.context = None
        self.stream = None
        self.bindings = []
        self.input_shapes = {}
        self.output_shapes = {}
        
        if not TENSORRT_AVAILABLE:
            raise RuntimeError("TensorRT not available. Install tensorrt and pycuda.")
        
        # Create cache directory
        Path(config.tensorrt_cache_dir).mkdir(parents=True, exist_ok=True)
        
        logger.info("TensorRT optimizer initialized: precision=%s workspace=%.1fGB", 
                   config.tensorrt_precision, config.tensorrt_workspace_gb)
    
    def build_engine(
        self,
        onnx_path: str,
        input_shapes: Dict[str, Tuple[int, ...]],
        cache_key: Optional[str] = None
    ) -> bool:
        """Build TensorRT engine from ONNX model."""
        try:
            # Check for cached engine
            if cache_key:
                cache_path = Path(self.config.tensorrt_cache_dir) / f"{cache_key}.trt"
                if cache_path.exists():
                    logger.info("Loading cached TensorRT engine: %s", cache_path)
                    return self._load_engine(str(cache_path))
            
            logger.info("Building TensorRT engine from ONNX: %s", onnx_path)
            
            # Create builder and network
            builder = trt.Builder(trt.Logger(trt.Logger.WARNING))
            network = builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network, trt.Logger(trt.Logger.WARNING))
            
            # Parse ONNX model
            with open(onnx_path, 'rb') as model:
                if not parser.parse(model.read()):
                    logger.error("Failed to parse ONNX model")
                    for error in range(parser.num_errors):
                        logger.error("Parser error: %s", parser.get_error(error))
                    return False
            
            # Configure builder
            config = builder.create_builder_config()
            config.max_workspace_size = int(self.config.tensorrt_workspace_gb * (1 << 30))
            
            # Set precision
            if self.config.tensorrt_precision == "fp16":
                config.set_flag(trt.BuilderFlag.FP16)
                logger.info("Enabled FP16 precision")
            elif self.config.tensorrt_precision == "int8":
                config.set_flag(trt.BuilderFlag.INT8)
                
                # Create calibration dataset from validation data
                if hasattr(self, 'calibration_data') and self.calibration_data:
                    calibrator = self._create_int8_calibrator(self.calibration_data)
                    config.int8_calibrator = calibrator
                    logger.info("Enabled INT8 precision with calibration")
                else:
                    logger.warning("INT8 precision requested but no calibration data provided")
                    config.set_flag(trt.BuilderFlag.FP16)  # Fallback to FP16
                    logger.info("Falling back to FP16 precision")
            
            # Set optimization profiles for dynamic shapes
            profile = builder.create_optimization_profile()
            
            for input_name, shape in input_shapes.items():
                # Find input in network
                for i in range(network.num_inputs):
                    if network.get_input(i).name == input_name:
                        # Set min, opt, max shapes (dynamic batch size)
                        min_shape = (1,) + shape[1:]
                        opt_shape = (self.config.tensorrt_max_batch_size // 2,) + shape[1:]
                        max_shape = (self.config.tensorrt_max_batch_size,) + shape[1:]
                        
                        profile.set_shape(input_name, min_shape, opt_shape, max_shape)
                        logger.info("Set dynamic shape for %s: min=%s opt=%s max=%s",
                                   input_name, min_shape, opt_shape, max_shape)
                        break
            
            config.add_optimization_profile(profile)
            
            # Build engine
            start_time = time.time()
            engine = builder.build_engine(network, config)
            build_time = time.time() - start_time
            
            if engine is None:
                logger.error("Failed to build TensorRT engine")
                return False
            
            logger.info("TensorRT engine built in %.2fs", build_time)
            
            # Save engine to cache
            if cache_key:
                cache_path = Path(self.config.tensorrt_cache_dir) / f"{cache_key}.trt"
                with open(cache_path, 'wb') as f:
                    f.write(engine.serialize())
                logger.info("Cached TensorRT engine: %s", cache_path)
            
            self.engine = engine
            self._setup_inference()
            
            return True
            
        except Exception as e:
            logger.error("TensorRT engine build failed: %s", e)
            return False
    
    def _load_engine(self, engine_path: str) -> bool:
        """Load TensorRT engine from file."""
        try:
            runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
            
            with open(engine_path, 'rb') as f:
                engine_data = f.read()
            
            self.engine = runtime.deserialize_cuda_engine(engine_data)
            
            if self.engine is None:
                logger.error("Failed to load TensorRT engine")
                return False
            
            self._setup_inference()
            logger.info("TensorRT engine loaded: %s", engine_path)
            
            return True
            
        except Exception as e:
            logger.error("Failed to load TensorRT engine: %s", e)
            return False
    
    def _setup_inference(self):
        """Setup inference context and bindings."""
        self.context = self.engine.create_execution_context()
        self.stream = cuda.Stream()
        
        # Setup bindings
        self.bindings = []
        self.input_shapes = {}
        self.output_shapes = {}
        
        for i in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(i)
            shape = self.engine.get_binding_shape(i)
            dtype = trt.nptype(self.engine.get_binding_dtype(i))
            
            if self.engine.binding_is_input(i):
                self.input_shapes[name] = shape
            else:
                self.output_shapes[name] = shape
            
            # Allocate device memory
            size = trt.volume(shape) * np.dtype(dtype).itemsize
            device_mem = cuda.mem_alloc(size)
            self.bindings.append(int(device_mem))
            
            logger.debug("Binding %d: %s shape=%s dtype=%s", i, name, shape, dtype)
    
    @track_gpu_memory("0")
    def infer(self, inputs: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Run TensorRT inference."""
        if self.engine is None or self.context is None:
            raise RuntimeError("TensorRT engine not initialized")
        
        # Set dynamic shapes
        for input_name, input_data in inputs.items():
            if input_name in self.input_shapes:
                self.context.set_binding_shape(
                    self.engine.get_binding_index(input_name),
                    input_data.shape
                )
        
        # Copy inputs to device
        input_bindings = {}
        for i, (name, data) in enumerate(inputs.items()):
            binding_idx = self.engine.get_binding_index(name)
            cuda.memcpy_htod_async(self.bindings[binding_idx], data, self.stream)
            input_bindings[name] = binding_idx
        
        # Run inference
        start_time = time.time()
        success = self.context.execute_async_v2(bindings=self.bindings, stream_handle=self.stream.handle)
        
        if not success:
            raise RuntimeError("TensorRT inference failed")
        
        # Copy outputs from device
        outputs = {}
        for i in range(self.engine.num_bindings):
            if not self.engine.binding_is_input(i):
                name = self.engine.get_binding_name(i)
                shape = self.context.get_binding_shape(i)
                dtype = trt.nptype(self.engine.get_binding_dtype(i))
                
                # Allocate host memory
                output_data = np.empty(shape, dtype=dtype)
                
                # Copy from device
                cuda.memcpy_dtoh_async(output_data, self.bindings[i], self.stream)
                outputs[name] = output_data
        
        # Synchronize
        self.stream.synchronize()
        
        inference_time = time.time() - start_time
        record_processing_time(inference_time, "tensorrt_inference")
        
        return outputs
    
    def cleanup(self):
        """Clean up TensorRT resources."""
        if self.stream:
            self.stream.synchronize()
        
        # Free device memory
        for binding in self.bindings:
            cuda.mem_free(binding)
        
        self.bindings.clear()
        self.context = None
        self.engine = None
        self.stream = None


class ONNXOptimizer:
    """ONNX model export and optimization."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize ONNX optimizer."""
        self.config = config
        
        if not ONNX_AVAILABLE:
            raise RuntimeError("ONNX not available. Install onnx and onnxruntime-gpu.")
        
        logger.info("ONNX optimizer initialized: opset=%d dynamic_axes=%s",
                   config.onnx_opset_version, config.onnx_dynamic_axes)
    
    def export_model(
        self,
        model: nn.Module,
        dummy_input: torch.Tensor,
        output_path: str,
        input_names: List[str] = ["input"],
        output_names: List[str] = ["output"]
    ) -> bool:
        """Export PyTorch model to ONNX."""
        try:
            model.eval()
            
            # Dynamic axes for variable batch size and sequence length
            dynamic_axes = {}
            if self.config.onnx_dynamic_axes:
                for name in input_names:
                    dynamic_axes[name] = {0: "batch_size", 1: "num_patches"}
                for name in output_names:
                    dynamic_axes[name] = {0: "batch_size"}
            
            logger.info("Exporting model to ONNX: %s", output_path)
            
            torch.onnx.export(
                model,
                dummy_input,
                output_path,
                export_params=True,
                opset_version=self.config.onnx_opset_version,
                do_constant_folding=True,
                input_names=input_names,
                output_names=output_names,
                dynamic_axes=dynamic_axes if self.config.onnx_dynamic_axes else None,
                verbose=False
            )
            
            # Verify exported model
            onnx_model = onnx.load(output_path)
            onnx.checker.check_model(onnx_model)
            
            logger.info("ONNX export successful: %s", output_path)
            return True
            
        except Exception as e:
            logger.error("ONNX export failed: %s", e)
            return False
    
    def optimize_model(self, onnx_path: str, optimized_path: str) -> bool:
        """Optimize ONNX model."""
        try:
            # Load model
            model = onnx.load(onnx_path)
            
            # Basic optimizations
            from onnxruntime.tools import optimizer
            
            opt_model = optimizer.optimize_model(
                onnx_path,
                model_type='bert',  # Use BERT optimizations for transformer models
                num_heads=8,  # Adjust based on model
                hidden_size=256,  # Adjust based on model
                optimization_options=None
            )
            
            # Save optimized model
            opt_model.save_model_to_file(optimized_path)
            
            logger.info("ONNX model optimized: %s -> %s", onnx_path, optimized_path)
            return True
            
        except Exception as e:
            logger.error("ONNX optimization failed: %s", e)
            return False


class QuantizationOptimizer:
    """Model quantization for INT8/FP16 acceleration."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize quantization optimizer."""
        self.config = config
        
        logger.info("Quantization optimizer initialized: mode=%s", config.quantization_mode)
    
    def quantize_dynamic(self, model: nn.Module) -> nn.Module:
        """Apply dynamic quantization."""
        try:
            logger.info("Applying dynamic quantization")
            
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                {nn.Linear, nn.Conv2d},
                dtype=torch.qint8
            )
            
            logger.info("Dynamic quantization applied")
            return quantized_model
            
        except Exception as e:
            logger.error("Dynamic quantization failed: %s", e)
            return model
    
    def prepare_qat(self, model: nn.Module) -> nn.Module:
        """Prepare model for Quantization Aware Training."""
        try:
            logger.info("Preparing model for QAT")
            
            # Set quantization config
            model.qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
            
            # Prepare model
            prepared_model = torch.quantization.prepare_qat(model)
            
            logger.info("Model prepared for QAT")
            return prepared_model
            
        except Exception as e:
            logger.error("QAT preparation failed: %s", e)
            return model
    
    def convert_qat(self, model: nn.Module) -> nn.Module:
        """Convert QAT model to quantized model."""
        try:
            logger.info("Converting QAT model to quantized")
            
            model.eval()
            quantized_model = torch.quantization.convert(model)
            
            logger.info("QAT model converted to quantized")
            return quantized_model
            
        except Exception as e:
            logger.error("QAT conversion failed: %s", e)
            return model


class MultiGPUOptimizer:
    """Multi-GPU parallelization optimizer."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize multi-GPU optimizer."""
        self.config = config
        self.available_gpus = self._get_available_gpus()
        
        logger.info("Multi-GPU optimizer initialized: available_gpus=%d target_gpus=%s",
                   len(self.available_gpus), config.gpu_ids)
    
    def _get_available_gpus(self) -> List[int]:
        """Get list of available GPU IDs."""
        if not torch.cuda.is_available():
            return []
        
        gpu_count = torch.cuda.device_count()
        return list(range(gpu_count))
    
    def setup_data_parallel(self, model: nn.Module) -> nn.Module:
        """Setup data parallelism across multiple GPUs."""
        if not self.config.enable_data_parallel:
            return model
        
        if len(self.available_gpus) < 2:
            logger.warning("Data parallel requires multiple GPUs")
            return model
        
        try:
            # Use specified GPU IDs or all available
            gpu_ids = self.config.gpu_ids or self.available_gpus
            gpu_ids = [gpu_id for gpu_id in gpu_ids if gpu_id in self.available_gpus]
            
            if len(gpu_ids) < 2:
                logger.warning("Insufficient GPUs for data parallel: %s", gpu_ids)
                return model
            
            # Move model to first GPU
            model = model.to(f'cuda:{gpu_ids[0]}')
            
            # Wrap with DataParallel
            model = DataParallel(model, device_ids=gpu_ids)
            
            logger.info("Data parallel enabled on GPUs: %s", gpu_ids)
            return model
            
        except Exception as e:
            logger.error("Data parallel setup failed: %s", e)
            return model
    
    def setup_distributed_parallel(self, model: nn.Module, rank: int, world_size: int) -> nn.Module:
        """Setup distributed data parallelism."""
        try:
            # Initialize process group
            torch.distributed.init_process_group(
                backend='nccl',
                rank=rank,
                world_size=world_size
            )
            
            # Move model to GPU
            device = torch.device(f'cuda:{rank}')
            model = model.to(device)
            
            # Wrap with DistributedDataParallel
            model = DistributedDataParallel(model, device_ids=[rank])
            
            logger.info("Distributed parallel enabled: rank=%d world_size=%d", rank, world_size)
            return model
            
        except Exception as e:
            logger.error("Distributed parallel setup failed: %s", e)
            return model


class ModelOptimizer:
    """Comprehensive model optimization pipeline."""
    
    def __init__(self, config: OptimizationConfig):
        """Initialize model optimizer."""
        self.config = config
        
        # Initialize sub-optimizers
        self.tensorrt_optimizer = None
        self.onnx_optimizer = None
        self.quantization_optimizer = None
        self.multigpu_optimizer = None
        
        if config.enable_tensorrt and TENSORRT_AVAILABLE:
            self.tensorrt_optimizer = TensorRTOptimizer(config)
        
        if config.enable_onnx and ONNX_AVAILABLE:
            self.onnx_optimizer = ONNXOptimizer(config)
        
        if config.enable_quantization:
            self.quantization_optimizer = QuantizationOptimizer(config)
        
        self.multigpu_optimizer = MultiGPUOptimizer(config)
        
        logger.info("Model optimizer initialized: tensorrt=%s onnx=%s quantization=%s multigpu=%s",
                   self.tensorrt_optimizer is not None,
                   self.onnx_optimizer is not None,
                   self.quantization_optimizer is not None,
                   True)
    
    @timed_operation("model_optimization")
    def optimize_model(
        self,
        model: nn.Module,
        dummy_input: torch.Tensor,
        model_name: str = "model"
    ) -> Tuple[nn.Module, Optional[Dict[str, Any]]]:
        """Apply comprehensive model optimization."""
        optimized_model = model
        optimization_info = {
            "original_model": type(model).__name__,
            "optimizations_applied": [],
            "tensorrt_engine": None,
            "onnx_path": None
        }
        
        try:
            # 1. PyTorch optimizations
            optimized_model = self._apply_pytorch_optimizations(optimized_model)
            
            # 2. Multi-GPU setup
            optimized_model = self.multigpu_optimizer.setup_data_parallel(optimized_model)
            
            # 3. Quantization
            if self.quantization_optimizer:
                if self.config.quantization_mode == "dynamic":
                    optimized_model = self.quantization_optimizer.quantize_dynamic(optimized_model)
                    optimization_info["optimizations_applied"].append("dynamic_quantization")
            
            # 4. ONNX export
            onnx_path = None
            if self.onnx_optimizer:
                onnx_path = f"/tmp/{model_name}.onnx"
                
                # Use original model for ONNX export (before DataParallel wrapping)
                export_model = model
                if hasattr(optimized_model, 'module'):
                    export_model = optimized_model.module
                
                if self.onnx_optimizer.export_model(export_model, dummy_input, onnx_path):
                    optimization_info["onnx_path"] = onnx_path
                    optimization_info["optimizations_applied"].append("onnx_export")
            
            # 5. TensorRT optimization
            if self.tensorrt_optimizer and onnx_path:
                input_shapes = {"input": dummy_input.shape}
                cache_key = f"{model_name}_{self.config.tensorrt_precision}"
                
                if self.tensorrt_optimizer.build_engine(onnx_path, input_shapes, cache_key):
                    optimization_info["tensorrt_engine"] = self.tensorrt_optimizer
                    optimization_info["optimizations_applied"].append("tensorrt")
            
            logger.info("Model optimization complete: %s", optimization_info["optimizations_applied"])
            
            return optimized_model, optimization_info
            
        except Exception as e:
            logger.error("Model optimization failed: %s", e)
            return model, optimization_info
    
    def _apply_pytorch_optimizations(self, model: nn.Module) -> nn.Module:
        """Apply PyTorch-specific optimizations."""
        try:
            # Enable channels last memory format for better performance
            if self.config.enable_channels_last:
                model = model.to(memory_format=torch.channels_last)
                logger.info("Enabled channels last memory format")
            
            # PyTorch 2.0 compile
            if self.config.enable_torch_compile and hasattr(torch, 'compile'):
                model = torch.compile(model, mode=self.config.torch_compile_mode)
                logger.info("Applied torch.compile with mode: %s", self.config.torch_compile_mode)
            
            return model
            
        except Exception as e:
            logger.error("PyTorch optimizations failed: %s", e)
            return model
    
    def create_optimized_inference_fn(
        self,
        model: nn.Module,
        optimization_info: Dict[str, Any]
    ) -> callable:
        """Create optimized inference function."""
        
        # Use TensorRT if available
        if optimization_info.get("tensorrt_engine"):
            tensorrt_engine = optimization_info["tensorrt_engine"]
            
            def tensorrt_inference(features: torch.Tensor) -> torch.Tensor:
                # Convert to numpy
                input_np = features.cpu().numpy()
                
                # Run TensorRT inference
                outputs = tensorrt_engine.infer({"input": input_np})
                
                # Convert back to torch
                output_tensor = torch.from_numpy(outputs["output"]).to(features.device)
                return output_tensor
            
            return tensorrt_inference
        
        # Use optimized PyTorch model
        else:
            def pytorch_inference(features: torch.Tensor) -> torch.Tensor:
                with torch.no_grad():
                    if self.config.enable_mixed_precision:
                        with torch.cuda.amp.autocast():
                            return model(features)
                    else:
                        return model(features)
            
            return pytorch_inference
    
    def benchmark_model(
        self,
        inference_fn: callable,
        dummy_input: torch.Tensor,
        num_iterations: int = 100,
        warmup_iterations: int = 10
    ) -> Dict[str, float]:
        """Benchmark optimized model performance."""
        device = dummy_input.device
        
        # Warmup
        for _ in range(warmup_iterations):
            _ = inference_fn(dummy_input)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        start_time = time.time()
        
        for _ in range(num_iterations):
            _ = inference_fn(dummy_input)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        total_time = time.time() - start_time
        avg_time = total_time / num_iterations
        throughput = dummy_input.shape[0] / avg_time  # samples per second
        
        # Record metrics
        record_throughput_measurement(throughput, str(device.index) if device.type == 'cuda' else 'cpu')
        
        return {
            "avg_inference_time_ms": avg_time * 1000,
            "throughput_samples_per_sec": throughput,
            "total_benchmark_time_s": total_time,
            "iterations": num_iterations
        }
    
    def cleanup(self):
        """Clean up optimization resources."""
        if self.tensorrt_optimizer:
            self.tensorrt_optimizer.cleanup()


# Convenience functions
def optimize_attention_model(
    model: nn.Module,
    config: OptimizationConfig,
    dummy_input: torch.Tensor,
    model_name: str = "attention_mil"
) -> Tuple[nn.Module, callable, Dict[str, Any]]:
    """Optimize attention-based MIL model for inference."""
    optimizer = ModelOptimizer(config)
    
    # Optimize model
    optimized_model, optimization_info = optimizer.optimize_model(model, dummy_input, model_name)
    
    # Create inference function
    inference_fn = optimizer.create_optimized_inference_fn(optimized_model, optimization_info)
    
    # Benchmark performance
    benchmark_results = optimizer.benchmark_model(inference_fn, dummy_input)
    optimization_info["benchmark"] = benchmark_results
    
    logger.info("Model optimization complete: avg_time=%.2fms throughput=%.1f samples/sec",
               benchmark_results["avg_inference_time_ms"],
               benchmark_results["throughput_samples_per_sec"])
    
    return optimized_model, inference_fn, optimization_info


def get_optimization_config(
    precision: str = "fp16",
    enable_tensorrt: bool = True,
    enable_quantization: bool = True,
    gpu_ids: Optional[List[int]] = None
) -> OptimizationConfig:
    """Get default optimization configuration."""
    return OptimizationConfig(
        enable_tensorrt=enable_tensorrt and TENSORRT_AVAILABLE,
        tensorrt_precision=precision,
        enable_quantization=enable_quantization,
        enable_data_parallel=gpu_ids is None or len(gpu_ids or []) > 1,
        gpu_ids=gpu_ids
    )