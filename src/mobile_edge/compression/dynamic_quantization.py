"""
Dynamic Quantization for Medical AI Models

Implements dynamic quantization where weights are quantized ahead of time
but activations are quantized dynamically at runtime.
"""

import torch
import torch.nn as nn
import torch.quantization as quant
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass
from pathlib import Path
import json
import copy
import time

from ..utils.model_utils import get_model_size, count_parameters


@dataclass
class DynamicQuantizationConfig:
    """Configuration for dynamic quantization"""
    dtype: torch.dtype = torch.qint8        # Quantization dtype (qint8 or float16)
    quantize_linear: bool = True            # Quantize Linear layers
    quantize_lstm: bool = True              # Quantize LSTM layers
    quantize_gru: bool = False              # Quantize GRU layers
    quantize_embedding: bool = False        # Quantize Embedding layers
    reduce_range: bool = False              # Reduce quantization range
    skip_layers: List[str] = None           # Layer patterns to skip
    
    def __post_init__(self):
        if self.skip_layers is None:
            self.skip_layers = ['classifier', 'fc', 'head']


@dataclass
class DynamicQuantizationResult:
    """Results from dynamic quantization"""
    original_size_mb: float
    quantized_size_mb: float
    compression_ratio: float
    original_accuracy: float
    quantized_accuracy: float
    accuracy_drop: float
    inference_speedup: float
    quantized_layers: int
    dtype: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'original_size_mb': self.original_size_mb,
            'quantized_size_mb': self.quantized_size_mb,
            'compression_ratio': self.compression_ratio,
            'original_accuracy': self.original_accuracy,
            'quantized_accuracy': self.quantized_accuracy,
            'accuracy_drop': self.accuracy_drop,
            'inference_speedup': self.inference_speedup,
            'quantized_layers': self.quantized_layers,
            'dtype': self.dtype
        }


class DynamicQuantizer:
    """
    Dynamic quantization implementation
    
    Dynamic quantization quantizes:
    - Weights: Quantized ahead of time (static)
    - Activations: Quantized dynamically at runtime
    
    Benefits:
    - No calibration data needed
    - No accuracy loss in many cases
    - Faster inference for compute-bound models
    - Smaller model size
    
    Best for:
    - Models with Linear/LSTM layers
    - CPU inference
    - When calibration data is unavailable
    """
    
    def __init__(self, config: DynamicQuantizationConfig):
        """Initialize dynamic quantizer"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Quantization state
        self.original_model = None
        self.quantized_model = None
        self.quantized_layer_names = []
        
    def quantize_model(self, model: nn.Module,
                      val_loader: torch.utils.data.DataLoader = None,
                      device: torch.device = None,
                      criterion: nn.Module = None) -> DynamicQuantizationResult:
        """
        Apply dynamic quantization to model
        
        Args:
            model: PyTorch model to quantize
            val_loader: Validation dataloader (for accuracy evaluation)
            device: Device for computation
            criterion: Loss function
            
        Returns:
            DynamicQuantizationResult with compression statistics
        """
        try:
            # Store original model
            self.original_model = copy.deepcopy(model)
            
            # Evaluate original model
            original_accuracy = 0.0
            if val_loader is not None and device is not None and criterion is not None:
                original_accuracy = self._evaluate_model(model, val_loader, device, criterion)
            
            original_size = self._get_model_size_mb(model)
            
            self.logger.info(f"Original model - Size: {original_size:.2f}MB, "
                           f"Accuracy: {original_accuracy:.4f}")
            
            # Apply dynamic quantization
            quantized_model = self._apply_dynamic_quantization(model)
            self.quantized_model = quantized_model
            
            # Evaluate quantized model
            quantized_accuracy = 0.0
            if val_loader is not None and device is not None and criterion is not None:
                quantized_accuracy = self._evaluate_quantized_model(
                    quantized_model, val_loader, device, criterion
                )
            
            quantized_size = self._get_model_size_mb(quantized_model)
            
            # Measure inference speedup
            inference_speedup = self._measure_inference_speedup(
                model, quantized_model, device
            )
            
            # Create result
            result = DynamicQuantizationResult(
                original_size_mb=original_size,
                quantized_size_mb=quantized_size,
                compression_ratio=original_size / quantized_size if quantized_size > 0 else 1.0,
                original_accuracy=original_accuracy,
                quantized_accuracy=quantized_accuracy,
                accuracy_drop=original_accuracy - quantized_accuracy,
                inference_speedup=inference_speedup,
                quantized_layers=len(self.quantized_layer_names),
                dtype=str(self.config.dtype)
            )
            
            self.logger.info(f"Dynamic quantization complete - "
                           f"Compression: {result.compression_ratio:.2f}x, "
                           f"Speedup: {result.inference_speedup:.2f}x, "
                           f"Accuracy drop: {result.accuracy_drop:.4f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error during dynamic quantization: {e}")
            raise
    
    def _apply_dynamic_quantization(self, model: nn.Module) -> nn.Module:
        """Apply dynamic quantization to model"""
        try:
            # Determine which layer types to quantize
            qconfig_spec = {}
            
            # Build list of layer types to quantize
            layer_types = set()
            if self.config.quantize_linear:
                layer_types.add(nn.Linear)
            if self.config.quantize_lstm:
                layer_types.add(nn.LSTM)
            if self.config.quantize_gru:
                layer_types.add(nn.GRU)
            if self.config.quantize_embedding:
                layer_types.add(nn.Embedding)
            
            if not layer_types:
                self.logger.warning("No layer types selected for quantization")
                return model
            
            # Identify layers to quantize
            self.quantized_layer_names = []
            for name, module in model.named_modules():
                if type(module) in layer_types:
                    # Check if layer should be skipped
                    skip = False
                    for skip_pattern in self.config.skip_layers:
                        if skip_pattern in name:
                            skip = True
                            break
                    
                    if not skip:
                        self.quantized_layer_names.append(name)
            
            self.logger.info(f"Quantizing {len(self.quantized_layer_names)} layers")
            
            # Apply dynamic quantization
            quantized_model = torch.quantization.quantize_dynamic(
                model,
                qconfig_spec=layer_types,
                dtype=self.config.dtype,
                inplace=False
            )
            
            return quantized_model
            
        except Exception as e:
            self.logger.error(f"Error applying dynamic quantization: {e}")
            raise
    
    def _evaluate_model(self, model: nn.Module,
                       dataloader: torch.utils.data.DataLoader,
                       device: torch.device,
                       criterion: nn.Module) -> float:
        """Evaluate floating-point model"""
        try:
            model.eval()
            model = model.to(device)
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(dataloader):
                    data, target = data.to(device), target.to(device)
                    
                    output = model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
                    
                    # Limit evaluation batches
                    if batch_idx >= 100:
                        break
            
            accuracy = correct / total if total > 0 else 0.0
            return accuracy
            
        except Exception as e:
            self.logger.error(f"Error evaluating model: {e}")
            return 0.0
    
    def _evaluate_quantized_model(self, model: nn.Module,
                                 dataloader: torch.utils.data.DataLoader,
                                 device: torch.device,
                                 criterion: nn.Module) -> float:
        """Evaluate quantized model"""
        try:
            model.eval()
            
            # Dynamic quantized models run on CPU
            eval_device = torch.device('cpu')
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(dataloader):
                    data = data.to(eval_device)
                    target = target.to(eval_device)
                    
                    output = model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
                    
                    # Limit evaluation batches
                    if batch_idx >= 100:
                        break
            
            accuracy = correct / total if total > 0 else 0.0
            return accuracy
            
        except Exception as e:
            self.logger.error(f"Error evaluating quantized model: {e}")
            return 0.0
    
    def _get_model_size_mb(self, model: nn.Module) -> float:
        """Get model size in MB"""
        try:
            param_size = 0
            buffer_size = 0
            
            for param in model.parameters():
                param_size += param.nelement() * param.element_size()
            
            for buffer in model.buffers():
                buffer_size += buffer.nelement() * buffer.element_size()
            
            size_mb = (param_size + buffer_size) / (1024 * 1024)
            return size_mb
            
        except Exception as e:
            self.logger.error(f"Error calculating model size: {e}")
            return 0.0
    
    def _measure_inference_speedup(self, original_model: nn.Module,
                                  quantized_model: nn.Module,
                                  device: torch.device) -> float:
        """Measure inference speedup of quantized model"""
        try:
            # Create dummy input
            dummy_input = torch.randn(1, 3, 224, 224)
            
            # Warm up original model
            original_model.eval()
            original_model = original_model.to(device)
            for _ in range(10):
                with torch.no_grad():
                    _ = original_model(dummy_input.to(device))
            
            # Measure original model
            start_time = time.time()
            for _ in range(100):
                with torch.no_grad():
                    _ = original_model(dummy_input.to(device))
            original_time = time.time() - start_time
            
            # Warm up quantized model (CPU)
            quantized_model.eval()
            eval_device = torch.device('cpu')
            for _ in range(10):
                with torch.no_grad():
                    _ = quantized_model(dummy_input.to(eval_device))
            
            # Measure quantized model
            start_time = time.time()
            for _ in range(100):
                with torch.no_grad():
                    _ = quantized_model(dummy_input.to(eval_device))
            quantized_time = time.time() - start_time
            
            speedup = original_time / quantized_time if quantized_time > 0 else 1.0
            
            self.logger.info(f"Inference timing - Original: {original_time*10:.2f}ms, "
                           f"Quantized: {quantized_time*10:.2f}ms, "
                           f"Speedup: {speedup:.2f}x")
            
            return speedup
            
        except Exception as e:
            self.logger.warning(f"Error measuring inference speedup: {e}")
            return 1.0
    
    def save_quantized_model(self, save_path: Path):
        """Save quantized model"""
        try:
            if self.quantized_model is None:
                raise ValueError("No quantized model to save")
            
            # Save as TorchScript for deployment
            scripted_model = torch.jit.script(self.quantized_model)
            torch.jit.save(scripted_model, save_path)
            
            self.logger.info(f"Saved quantized model to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving quantized model: {e}")
            raise
    
    def load_quantized_model(self, load_path: Path) -> nn.Module:
        """Load quantized model"""
        try:
            quantized_model = torch.jit.load(load_path)
            self.quantized_model = quantized_model
            
            self.logger.info(f"Loaded quantized model from {load_path}")
            return quantized_model
            
        except Exception as e:
            self.logger.error(f"Error loading quantized model: {e}")
            raise


def create_dynamic_quantization_config(dtype: torch.dtype = torch.qint8,
                                     quantize_all: bool = True) -> DynamicQuantizationConfig:
    """Create dynamic quantization configuration"""
    return DynamicQuantizationConfig(
        dtype=dtype,
        quantize_linear=quantize_all,
        quantize_lstm=quantize_all,
        quantize_gru=quantize_all,
        quantize_embedding=False,  # Usually keep embeddings in FP32
        reduce_range=False
    )


def quantize_model_dynamic(model: nn.Module,
                          val_loader: torch.utils.data.DataLoader = None,
                          dtype: torch.dtype = torch.qint8) -> Tuple[nn.Module, DynamicQuantizationResult]:
    """
    Apply dynamic quantization to model
    
    Args:
        model: Model to quantize
        val_loader: Validation loader for accuracy evaluation
        dtype: Quantization dtype (qint8 or float16)
        
    Returns:
        Tuple of (quantized_model, result)
    """
    config = DynamicQuantizationConfig(
        dtype=dtype,
        quantize_linear=True,
        quantize_lstm=True,
        quantize_gru=False,
        quantize_embedding=False
    )
    
    quantizer = DynamicQuantizer(config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    
    result = quantizer.quantize_model(model, val_loader, device, criterion)
    
    return quantizer.quantized_model, result


# Example usage for medical AI
def quantize_pathology_model_dynamic(model: nn.Module,
                                   val_loader: torch.utils.data.DataLoader) -> DynamicQuantizationResult:
    """Dynamic quantization optimized for pathology models"""
    config = DynamicQuantizationConfig(
        dtype=torch.qint8,              # INT8 for maximum compression
        quantize_linear=True,           # Quantize all Linear layers
        quantize_lstm=True,             # Quantize LSTM if present
        quantize_gru=False,             # Skip GRU
        quantize_embedding=False,       # Keep embeddings in FP32
        reduce_range=False,             # Full range for medical precision
        skip_layers=['classifier', 'fc', 'head', 'final']  # Preserve final layers
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    
    quantizer = DynamicQuantizer(config)
    return quantizer.quantize_model(model, val_loader, device, criterion)


def compare_quantization_dtypes(model: nn.Module,
                               val_loader: torch.utils.data.DataLoader) -> Dict[str, DynamicQuantizationResult]:
    """Compare different quantization dtypes"""
    results = {}
    
    # INT8 quantization
    int8_config = DynamicQuantizationConfig(dtype=torch.qint8)
    int8_quantizer = DynamicQuantizer(int8_config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    results['qint8'] = int8_quantizer.quantize_model(
        copy.deepcopy(model), val_loader, device, criterion
    )
    
    # FP16 quantization (if supported)
    try:
        fp16_config = DynamicQuantizationConfig(dtype=torch.float16)
        fp16_quantizer = DynamicQuantizer(fp16_config)
        results['float16'] = fp16_quantizer.quantize_model(
            copy.deepcopy(model), val_loader, device, criterion
        )
    except Exception as e:
        logging.warning(f"FP16 dynamic quantization not supported: {e}")
    
    return results
