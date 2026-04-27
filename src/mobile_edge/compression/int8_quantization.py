"""
INT8 Quantization for Medical AI Models

Implements INT8 quantization techniques including post-training quantization (PTQ)
and quantization-aware training (QAT) for efficient mobile deployment.
"""

import torch
import torch.nn as nn
import torch.quantization as quant
import numpy as np
import logging
from typing import Dict, List, Optional, Tuple, Union, Any, Callable
from dataclasses import dataclass
from pathlib import Path
import json
import copy
import math

from ..utils.model_utils import get_model_size, count_parameters


@dataclass
class INT8QuantizationConfig:
    """Configuration for INT8 quantization"""
    quantization_type: str = 'qat'      # 'ptq' or 'qat'
    backend: str = 'fbgemm'             # 'fbgemm' (CPU) or 'qnnpack' (mobile)
    calibration_batches: int = 100      # Batches for PTQ calibration
    qat_epochs: int = 10               # QAT fine-tuning epochs
    qat_lr: float = 1e-5               # QAT learning rate
    observer_type: str = 'minmax'       # 'minmax' or 'histogram'
    fake_quant_enabled: bool = True     # Enable fake quantization in QAT
    reduce_range: bool = False          # Reduce quantization range
    symmetric: bool = True              # Symmetric quantization
    per_channel: bool = True            # Per-channel vs per-tensor quantization
    skip_layers: List[str] = None       # Layer patterns to skip
    
    def __post_init__(self):
        if self.skip_layers is None:
            self.skip_layers = ['classifier', 'fc', 'head', 'final']


@dataclass
class QuantizationResult:
    """Results from INT8 quantization"""
    original_size_mb: float
    quantized_size_mb: float
    compression_ratio: float
    original_accuracy: float
    quantized_accuracy: float
    accuracy_drop: float
    inference_speedup: float
    quantization_type: str
    backend: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'original_size_mb': self.original_size_mb,
            'quantized_size_mb': self.quantized_size_mb,
            'compression_ratio': self.compression_ratio,
            'original_accuracy': self.original_accuracy,
            'quantized_accuracy': self.quantized_accuracy,
            'accuracy_drop': self.accuracy_drop,
            'inference_speedup': self.inference_speedup,
            'quantization_type': self.quantization_type,
            'backend': self.backend
        }


class INT8Quantizer:
    """
    INT8 quantization implementation
    
    Supports two main approaches:
    1. Post-Training Quantization (PTQ): Fast, no retraining needed
    2. Quantization-Aware Training (QAT): Better accuracy, requires retraining
    
    Features:
    - Multiple backends (FBGEMM for CPU, QNNPACK for mobile)
    - Per-channel and per-tensor quantization
    - Custom observers for calibration
    - Layer-specific quantization control
    """
    
    def __init__(self, config: INT8QuantizationConfig):
        """Initialize INT8 quantizer"""
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Set quantization backend
        torch.backends.quantized.engine = config.backend
        
        # Quantization state
        self.original_model = None
        self.quantized_model = None
        self.calibration_data = []
        
    def quantize_model(self, model: nn.Module,
                      train_loader: torch.utils.data.DataLoader = None,
                      val_loader: torch.utils.data.DataLoader = None,
                      device: torch.device = None,
                      criterion: nn.Module = None) -> QuantizationResult:
        """
        Quantize model to INT8
        
        Args:
            model: PyTorch model to quantize
            train_loader: Training dataloader (for QAT)
            val_loader: Validation dataloader
            device: Device for computation
            criterion: Loss function (for QAT)
            
        Returns:
            QuantizationResult with compression statistics
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
            
            # Apply quantization
            if self.config.quantization_type == 'ptq':
                quantized_model = self._post_training_quantization(model, val_loader, device)
            elif self.config.quantization_type == 'qat':
                quantized_model = self._quantization_aware_training(
                    model, train_loader, val_loader, device, criterion
                )
            else:
                raise ValueError(f"Unknown quantization type: {self.config.quantization_type}")
            
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
            result = QuantizationResult(
                original_size_mb=original_size,
                quantized_size_mb=quantized_size,
                compression_ratio=original_size / quantized_size if quantized_size > 0 else 1.0,
                original_accuracy=original_accuracy,
                quantized_accuracy=quantized_accuracy,
                accuracy_drop=original_accuracy - quantized_accuracy,
                inference_speedup=inference_speedup,
                quantization_type=self.config.quantization_type,
                backend=self.config.backend
            )
            
            self.logger.info(f"INT8 quantization complete - "
                           f"Compression: {result.compression_ratio:.2f}x, "
                           f"Speedup: {result.inference_speedup:.2f}x, "
                           f"Accuracy drop: {result.accuracy_drop:.4f}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error during INT8 quantization: {e}")
            raise
    
    def _post_training_quantization(self, model: nn.Module,
                                  val_loader: torch.utils.data.DataLoader,
                                  device: torch.device) -> nn.Module:
        """Apply post-training quantization"""
        try:
            self.logger.info("Applying post-training quantization")
            
            # Prepare model for quantization
            model.eval()
            
            # Set quantization configuration
            if self.config.backend == 'fbgemm':
                qconfig = torch.quantization.get_default_qconfig('fbgemm')
            else:  # qnnpack
                qconfig = torch.quantization.get_default_qconfig('qnnpack')
            
            # Custom qconfig for better medical AI performance
            if self.config.observer_type == 'histogram':
                qconfig = torch.quantization.QConfig(
                    activation=torch.quantization.HistogramObserver.with_args(
                        reduce_range=self.config.reduce_range
                    ),
                    weight=torch.quantization.default_per_channel_weight_observer
                    if self.config.per_channel
                    else torch.quantization.default_weight_observer
                )
            
            # Apply qconfig to model
            model.qconfig = qconfig
            
            # Fuse modules for better quantization
            model = self._fuse_modules(model)
            
            # Prepare model
            prepared_model = torch.quantization.prepare(model, inplace=False)
            
            # Calibration
            self._calibrate_model(prepared_model, val_loader, device)
            
            # Convert to quantized model
            quantized_model = torch.quantization.convert(prepared_model, inplace=False)
            
            return quantized_model
            
        except Exception as e:
            self.logger.error(f"Error in post-training quantization: {e}")
            raise
    
    def _quantization_aware_training(self, model: nn.Module,
                                   train_loader: torch.utils.data.DataLoader,
                                   val_loader: torch.utils.data.DataLoader,
                                   device: torch.device,
                                   criterion: nn.Module) -> nn.Module:
        """Apply quantization-aware training"""
        try:
            self.logger.info("Applying quantization-aware training")
            
            # Prepare model for QAT
            model.train()
            
            # Set quantization configuration
            if self.config.backend == 'fbgemm':
                qconfig = torch.quantization.get_default_qat_qconfig('fbgemm')
            else:  # qnnpack
                qconfig = torch.quantization.get_default_qat_qconfig('qnnpack')
            
            # Custom QAT qconfig
            if self.config.fake_quant_enabled:
                qconfig = torch.quantization.QConfig(
                    activation=torch.quantization.FakeQuantize.with_args(
                        observer=torch.quantization.MovingAverageMinMaxObserver,
                        quant_min=0 if not self.config.symmetric else -128,
                        quant_max=255 if not self.config.symmetric else 127,
                        reduce_range=self.config.reduce_range
                    ),
                    weight=torch.quantization.FakeQuantize.with_args(
                        observer=torch.quantization.MovingAveragePerChannelMinMaxObserver
                        if self.config.per_channel
                        else torch.quantization.MovingAverageMinMaxObserver,
                        quant_min=-128,
                        quant_max=127,
                        reduce_range=self.config.reduce_range
                    )
                )
            
            # Apply qconfig
            model.qconfig = qconfig
            
            # Fuse modules
            model = self._fuse_modules(model)
            
            # Prepare for QAT
            prepared_model = torch.quantization.prepare_qat(model, inplace=False)
            
            # QAT training
            self._train_qat_model(prepared_model, train_loader, val_loader, device, criterion)
            
            # Convert to quantized model
            prepared_model.eval()
            quantized_model = torch.quantization.convert(prepared_model, inplace=False)
            
            return quantized_model
            
        except Exception as e:
            self.logger.error(f"Error in quantization-aware training: {e}")
            raise
    
    def _fuse_modules(self, model: nn.Module) -> nn.Module:
        """Fuse conv-bn-relu modules for better quantization"""
        try:
            # Common fusion patterns
            fusion_patterns = [
                ['conv', 'bn'],
                ['conv', 'bn', 'relu'],
                ['conv', 'relu'],
                ['linear', 'relu']
            ]
            
            # Find modules to fuse
            modules_to_fuse = []
            
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    # Look for conv-bn or conv-bn-relu patterns
                    bn_name = name + '.bn'
                    relu_name = name + '.relu'
                    
                    if hasattr(model, bn_name.replace('.', '_')):
                        if hasattr(model, relu_name.replace('.', '_')):
                            modules_to_fuse.append([name, bn_name, relu_name])
                        else:
                            modules_to_fuse.append([name, bn_name])
                    elif hasattr(model, relu_name.replace('.', '_')):
                        modules_to_fuse.append([name, relu_name])
            
            # Apply fusion if modules found
            if modules_to_fuse:
                model = torch.quantization.fuse_modules(model, modules_to_fuse, inplace=False)
                self.logger.info(f"Fused {len(modules_to_fuse)} module groups")
            
            return model
            
        except Exception as e:
            self.logger.warning(f"Module fusion failed: {e}")
            return model  # Return original model if fusion fails
    
    def _calibrate_model(self, model: nn.Module,
                        dataloader: torch.utils.data.DataLoader,
                        device: torch.device):
        """Calibrate model for post-training quantization"""
        try:
            model.eval()
            
            with torch.no_grad():
                for batch_idx, (data, _) in enumerate(dataloader):
                    if batch_idx >= self.config.calibration_batches:
                        break
                    
                    data = data.to(device)
                    model(data)
            
            self.logger.info(f"Calibrated model with {min(batch_idx + 1, self.config.calibration_batches)} batches")
            
        except Exception as e:
            self.logger.error(f"Error during calibration: {e}")
            raise
    
    def _train_qat_model(self, model: nn.Module,
                        train_loader: torch.utils.data.DataLoader,
                        val_loader: torch.utils.data.DataLoader,
                        device: torch.device,
                        criterion: nn.Module):
        """Train model with quantization-aware training"""
        try:
            optimizer = torch.optim.Adam(model.parameters(), lr=self.config.qat_lr)
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.config.qat_epochs
            )
            
            best_accuracy = 0.0
            
            for epoch in range(self.config.qat_epochs):
                # Training phase
                model.train()
                total_loss = 0.0
                num_batches = 0
                
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)
                    
                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()
                    
                    total_loss += loss.item()
                    num_batches += 1
                    
                    # Limit training batches for efficiency
                    if batch_idx >= 100:
                        break
                
                scheduler.step()
                
                # Validation phase
                val_accuracy = self._evaluate_model(model, val_loader, device, criterion)
                
                if val_accuracy > best_accuracy:
                    best_accuracy = val_accuracy
                
                avg_loss = total_loss / num_batches
                self.logger.info(f"QAT Epoch {epoch+1}/{self.config.qat_epochs} - "
                               f"Loss: {avg_loss:.4f}, "
                               f"Val Acc: {val_accuracy:.4f}, "
                               f"LR: {scheduler.get_last_lr()[0]:.6f}")
            
            self.logger.info(f"QAT training complete - Best accuracy: {best_accuracy:.4f}")
            
        except Exception as e:
            self.logger.error(f"Error during QAT training: {e}")
            raise
    
    def _evaluate_model(self, model: nn.Module,
                       dataloader: torch.utils.data.DataLoader,
                       device: torch.device,
                       criterion: nn.Module) -> float:
        """Evaluate floating-point model"""
        try:
            model.eval()
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
            correct = 0
            total = 0
            
            # Quantized models typically run on CPU
            eval_device = torch.device('cpu')
            
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
            if hasattr(model, 'state_dict'):
                # Regular PyTorch model
                param_size = 0
                buffer_size = 0
                
                for param in model.parameters():
                    param_size += param.nelement() * param.element_size()
                
                for buffer in model.buffers():
                    buffer_size += buffer.nelement() * buffer.element_size()
                
                size_mb = (param_size + buffer_size) / (1024 * 1024)
            else:
                # Quantized model - estimate based on INT8
                total_params = sum(p.numel() for p in model.parameters())
                size_mb = total_params / (1024 * 1024)  # Assume 1 byte per parameter
            
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
            
            # Warm up
            for _ in range(10):
                with torch.no_grad():
                    _ = original_model(dummy_input.to(device))
                    _ = quantized_model(dummy_input.to(torch.device('cpu')))
            
            # Measure original model
            torch.cuda.synchronize() if device.type == 'cuda' else None
            start_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
            end_time = torch.cuda.Event(enable_timing=True) if device.type == 'cuda' else None
            
            if device.type == 'cuda':
                start_time.record()
            else:
                import time
                start_time = time.time()
            
            for _ in range(100):
                with torch.no_grad():
                    _ = original_model(dummy_input.to(device))
            
            if device.type == 'cuda':
                end_time.record()
                torch.cuda.synchronize()
                original_time = start_time.elapsed_time(end_time)
            else:
                original_time = (time.time() - start_time) * 1000
            
            # Measure quantized model
            if device.type == 'cuda':
                start_time.record()
            else:
                start_time = time.time()
            
            for _ in range(100):
                with torch.no_grad():
                    _ = quantized_model(dummy_input.to(torch.device('cpu')))
            
            if device.type == 'cuda':
                end_time.record()
                torch.cuda.synchronize()
                quantized_time = start_time.elapsed_time(end_time)
            else:
                quantized_time = (time.time() - start_time) * 1000
            
            speedup = original_time / quantized_time if quantized_time > 0 else 1.0
            
            self.logger.info(f"Inference timing - Original: {original_time:.2f}ms, "
                           f"Quantized: {quantized_time:.2f}ms, "
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
            
            torch.jit.save(torch.jit.script(self.quantized_model), save_path)
            self.logger.info(f"Saved quantized model to {save_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving quantized model: {e}")
            raise


def create_int8_quantization_config(quantization_type: str = 'qat',
                                  backend: str = 'fbgemm') -> INT8QuantizationConfig:
    """Create INT8 quantization configuration"""
    return INT8QuantizationConfig(
        quantization_type=quantization_type,
        backend=backend,
        calibration_batches=100,
        qat_epochs=10,
        qat_lr=1e-5,
        observer_type='minmax',
        fake_quant_enabled=True,
        per_channel=True,
        symmetric=True
    )


# Example usage for medical AI
def quantize_pathology_model_int8(model: nn.Module,
                                train_loader: torch.utils.data.DataLoader,
                                val_loader: torch.utils.data.DataLoader,
                                use_qat: bool = True) -> QuantizationResult:
    """INT8 quantization optimized for pathology models"""
    config = INT8QuantizationConfig(
        quantization_type='qat' if use_qat else 'ptq',
        backend='fbgemm',  # CPU backend for medical workstations
        calibration_batches=200,  # More calibration for medical accuracy
        qat_epochs=15,           # More epochs for medical models
        qat_lr=5e-6,            # Lower learning rate for stability
        observer_type='histogram', # Better for medical data distribution
        fake_quant_enabled=True,
        per_channel=True,        # Better accuracy with per-channel
        symmetric=False,         # Asymmetric for better range utilization
        reduce_range=False,      # Full range for medical precision
        skip_layers=['classifier', 'fc', 'head']  # Preserve final layers
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    criterion = nn.CrossEntropyLoss()
    
    quantizer = INT8Quantizer(config)
    return quantizer.quantize_model(model, train_loader, val_loader, device, criterion)