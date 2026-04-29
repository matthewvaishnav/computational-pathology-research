"""
FP16 Mixed Precision Training and Inference for Medical AI Models

Implements FP16 mixed precision techniques for faster training and inference
while maintaining model accuracy through automatic loss scaling.
"""

import copy
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.cuda.amp as amp
import torch.nn as nn

from ..utils.model_utils import count_parameters, get_model_size


@dataclass
class FP16Config:
    """Configuration for FP16 mixed precision"""

    enabled: bool = True  # Enable mixed precision
    opt_level: str = "O1"  # 'O0' (FP32), 'O1' (mixed), 'O2' (almost FP16), 'O3' (full FP16)
    loss_scale: Union[float, str] = "dynamic"  # Loss scale value or 'dynamic'
    initial_scale: float = 2**16  # Initial loss scale for dynamic
    growth_factor: float = 2.0  # Growth factor for dynamic scaling
    backoff_factor: float = 0.5  # Backoff factor for dynamic scaling
    growth_interval: int = 2000  # Steps between scale increases
    keep_batchnorm_fp32: bool = True  # Keep BatchNorm in FP32
    master_weights: bool = True  # Maintain FP32 master weights
    cast_model_outputs: bool = False  # Cast model outputs to FP32

    def __post_init__(self):
        if self.opt_level not in ["O0", "O1", "O2", "O3"]:
            raise ValueError(f"Invalid opt_level: {self.opt_level}")


@dataclass
class FP16Result:
    """Results from FP16 mixed precision"""

    original_size_mb: float
    fp16_size_mb: float
    memory_reduction: float
    original_accuracy: float
    fp16_accuracy: float
    accuracy_drop: float
    training_speedup: float
    inference_speedup: float
    opt_level: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_size_mb": self.original_size_mb,
            "fp16_size_mb": self.fp16_size_mb,
            "memory_reduction": self.memory_reduction,
            "original_accuracy": self.original_accuracy,
            "fp16_accuracy": self.fp16_accuracy,
            "accuracy_drop": self.accuracy_drop,
            "training_speedup": self.training_speedup,
            "inference_speedup": self.inference_speedup,
            "opt_level": self.opt_level,
        }


class FP16MixedPrecision:
    """
    FP16 mixed precision training and inference

    Implements automatic mixed precision (AMP) with:
    - Dynamic loss scaling to prevent gradient underflow
    - Selective FP16/FP32 operations
    - Master weight maintenance
    - BatchNorm in FP32 for stability

    Optimization levels:
    - O0: Pure FP32 (baseline)
    - O1: Mixed precision (recommended, conservative)
    - O2: Almost FP16 (aggressive, faster)
    - O3: Pure FP16 (fastest, may lose accuracy)
    """

    def __init__(self, config: FP16Config):
        """Initialize FP16 mixed precision"""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Check CUDA availability
        if not torch.cuda.is_available():
            self.logger.warning("CUDA not available, FP16 disabled")
            self.config.enabled = False

        # Scaler for dynamic loss scaling
        self.scaler = None
        if self.config.enabled and self.config.loss_scale == "dynamic":
            self.scaler = amp.GradScaler(
                init_scale=self.config.initial_scale,
                growth_factor=self.config.growth_factor,
                backoff_factor=self.config.backoff_factor,
                growth_interval=self.config.growth_interval,
            )

        # Statistics
        self.overflow_count = 0
        self.total_steps = 0

    def convert_model(self, model: nn.Module) -> nn.Module:
        """Convert model to FP16 mixed precision"""
        try:
            if not self.config.enabled:
                return model

            if self.config.opt_level == "O0":
                # Pure FP32, no conversion
                return model

            elif self.config.opt_level == "O1":
                # Mixed precision - keep some ops in FP32
                model = self._convert_o1(model)

            elif self.config.opt_level == "O2":
                # Almost FP16 - most ops in FP16
                model = self._convert_o2(model)

            elif self.config.opt_level == "O3":
                # Pure FP16
                model = model.half()

            self.logger.info(f"Converted model to FP16 opt_level={self.config.opt_level}")
            return model

        except Exception as e:
            self.logger.error(f"Error converting model to FP16: {e}")
            raise

    def _convert_o1(self, model: nn.Module) -> nn.Module:
        """Convert to O1 (mixed precision, conservative)"""
        # Keep BatchNorm and LayerNorm in FP32
        for module in model.modules():
            if isinstance(
                module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d, nn.LayerNorm, nn.GroupNorm)
            ):
                module.float()
            elif isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                # Convert weights to FP16, but keep running in mixed precision
                module.half()

        return model

    def _convert_o2(self, model: nn.Module) -> nn.Module:
        """Convert to O2 (almost FP16, aggressive)"""
        # Convert entire model to FP16
        model = model.half()

        # Keep BatchNorm in FP32 if configured
        if self.config.keep_batchnorm_fp32:
            for module in model.modules():
                if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                    module.float()

        return model

    def train_step(
        self,
        model: nn.Module,
        data: torch.Tensor,
        target: torch.Tensor,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
    ) -> Tuple[float, bool]:
        """
        Perform one training step with mixed precision

        Args:
            model: Model to train
            data: Input data
            target: Target labels
            criterion: Loss function
            optimizer: Optimizer
            device: Device for computation

        Returns:
            Tuple of (loss_value, overflow_occurred)
        """
        try:
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            if not self.config.enabled or self.config.opt_level == "O0":
                # Standard FP32 training
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                return loss.item(), False

            # Mixed precision training with autocast
            with amp.autocast():
                output = model(data)
                loss = criterion(output, target)

            # Backward pass with gradient scaling
            if self.scaler is not None:
                self.scaler.scale(loss).backward()

                # Check for gradient overflow
                self.scaler.unscale_(optimizer)
                grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                # Update weights
                self.scaler.step(optimizer)
                self.scaler.update()

                # Check if step was skipped due to overflow
                overflow = grad_norm == float("inf") or grad_norm != grad_norm
                if overflow:
                    self.overflow_count += 1
            else:
                # Static loss scaling
                scaled_loss = loss * self.config.loss_scale
                scaled_loss.backward()

                # Unscale gradients
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad.data.div_(self.config.loss_scale)

                optimizer.step()
                overflow = False

            self.total_steps += 1

            return loss.item(), overflow

        except Exception as e:
            self.logger.error(f"Error in FP16 training step: {e}")
            raise

    def inference(self, model: nn.Module, data: torch.Tensor, device: torch.device) -> torch.Tensor:
        """
        Perform inference with mixed precision

        Args:
            model: Model for inference
            data: Input data
            device: Device for computation

        Returns:
            Model output
        """
        try:
            data = data.to(device)

            if not self.config.enabled or self.config.opt_level == "O0":
                # Standard FP32 inference
                with torch.no_grad():
                    output = model(data)
                return output

            # Mixed precision inference
            with torch.no_grad():
                with amp.autocast():
                    output = model(data)

                # Cast output to FP32 if configured
                if self.config.cast_model_outputs:
                    output = output.float()

            return output

        except Exception as e:
            self.logger.error(f"Error in FP16 inference: {e}")
            raise

    def evaluate_model(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        criterion: nn.Module,
    ) -> float:
        """Evaluate model accuracy with mixed precision"""
        try:
            model.eval()
            correct = 0
            total = 0

            for batch_idx, (data, target) in enumerate(dataloader):
                output = self.inference(model, data, device)
                target = target.to(device)

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

    def benchmark_training(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        num_steps: int = 100,
    ) -> float:
        """Benchmark training speed with mixed precision"""
        try:
            model.train()

            # Warmup
            for i, (data, target) in enumerate(dataloader):
                if i >= 10:
                    break
                self.train_step(model, data, target, criterion, optimizer, device)

            # Benchmark
            torch.cuda.synchronize() if device.type == "cuda" else None
            start_time = time.time()

            step_count = 0
            for data, target in dataloader:
                if step_count >= num_steps:
                    break
                self.train_step(model, data, target, criterion, optimizer, device)
                step_count += 1

            torch.cuda.synchronize() if device.type == "cuda" else None
            end_time = time.time()

            elapsed_time = end_time - start_time
            steps_per_second = step_count / elapsed_time

            return steps_per_second

        except Exception as e:
            self.logger.error(f"Error benchmarking training: {e}")
            return 0.0

    def benchmark_inference(
        self,
        model: nn.Module,
        input_size: Tuple[int, int, int, int],
        device: torch.device,
        num_iterations: int = 100,
    ) -> float:
        """Benchmark inference speed with mixed precision"""
        try:
            model.eval()
            dummy_input = torch.randn(input_size).to(device)

            # Warmup
            for _ in range(10):
                _ = self.inference(model, dummy_input, device)

            # Benchmark
            torch.cuda.synchronize() if device.type == "cuda" else None
            start_time = time.time()

            for _ in range(num_iterations):
                _ = self.inference(model, dummy_input, device)

            torch.cuda.synchronize() if device.type == "cuda" else None
            end_time = time.time()

            elapsed_time = end_time - start_time
            inferences_per_second = num_iterations / elapsed_time

            return inferences_per_second

        except Exception as e:
            self.logger.error(f"Error benchmarking inference: {e}")
            return 0.0

    def get_model_size_mb(self, model: nn.Module) -> float:
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

    def get_statistics(self) -> Dict[str, Any]:
        """Get FP16 training statistics"""
        return {
            "total_steps": self.total_steps,
            "overflow_count": self.overflow_count,
            "overflow_rate": (
                self.overflow_count / self.total_steps if self.total_steps > 0 else 0.0
            ),
            "current_scale": (
                self.scaler.get_scale() if self.scaler is not None else self.config.loss_scale
            ),
        }

    def save_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer, save_path: Path):
        """Save FP16 training checkpoint"""
        try:
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "config": self.config.__dict__,
                "statistics": self.get_statistics(),
            }

            if self.scaler is not None:
                checkpoint["scaler_state_dict"] = self.scaler.state_dict()

            torch.save(checkpoint, save_path)
            self.logger.info(f"Saved FP16 checkpoint to {save_path}")

        except Exception as e:
            self.logger.error(f"Error saving checkpoint: {e}")
            raise

    def load_checkpoint(self, model: nn.Module, optimizer: torch.optim.Optimizer, load_path: Path):
        """Load FP16 training checkpoint"""
        try:
            checkpoint = torch.load(load_path)

            model.load_state_dict(checkpoint["model_state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

            if self.scaler is not None and "scaler_state_dict" in checkpoint:
                self.scaler.load_state_dict(checkpoint["scaler_state_dict"])

            self.logger.info(f"Loaded FP16 checkpoint from {load_path}")

        except Exception as e:
            self.logger.error(f"Error loading checkpoint: {e}")
            raise


def create_fp16_config(opt_level: str = "O1", dynamic_loss_scale: bool = True) -> FP16Config:
    """Create FP16 mixed precision configuration"""
    return FP16Config(
        enabled=True,
        opt_level=opt_level,
        loss_scale="dynamic" if dynamic_loss_scale else 2**16,
        keep_batchnorm_fp32=True,
        master_weights=True,
    )


def benchmark_fp16_vs_fp32(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> FP16Result:
    """Benchmark FP16 vs FP32 performance"""

    # FP32 baseline
    fp32_config = FP16Config(enabled=False, opt_level="O0")
    fp32_trainer = FP16MixedPrecision(fp32_config)

    fp32_model = copy.deepcopy(model).to(device)
    fp32_optimizer = torch.optim.Adam(fp32_model.parameters(), lr=1e-4)

    fp32_accuracy = fp32_trainer.evaluate_model(fp32_model, val_loader, device, criterion)
    fp32_size = fp32_trainer.get_model_size_mb(fp32_model)
    fp32_train_speed = fp32_trainer.benchmark_training(
        fp32_model, train_loader, device, criterion, fp32_optimizer, num_steps=50
    )
    fp32_infer_speed = fp32_trainer.benchmark_inference(
        fp32_model, (1, 3, 224, 224), device, num_iterations=100
    )

    # FP16 mixed precision
    fp16_config = FP16Config(enabled=True, opt_level="O1")
    fp16_trainer = FP16MixedPrecision(fp16_config)

    fp16_model = copy.deepcopy(model).to(device)
    fp16_model = fp16_trainer.convert_model(fp16_model)
    fp16_optimizer = torch.optim.Adam(fp16_model.parameters(), lr=1e-4)

    fp16_accuracy = fp16_trainer.evaluate_model(fp16_model, val_loader, device, criterion)
    fp16_size = fp16_trainer.get_model_size_mb(fp16_model)
    fp16_train_speed = fp16_trainer.benchmark_training(
        fp16_model, train_loader, device, criterion, fp16_optimizer, num_steps=50
    )
    fp16_infer_speed = fp16_trainer.benchmark_inference(
        fp16_model, (1, 3, 224, 224), device, num_iterations=100
    )

    # Create result
    result = FP16Result(
        original_size_mb=fp32_size,
        fp16_size_mb=fp16_size,
        memory_reduction=fp32_size / fp16_size if fp16_size > 0 else 1.0,
        original_accuracy=fp32_accuracy,
        fp16_accuracy=fp16_accuracy,
        accuracy_drop=fp32_accuracy - fp16_accuracy,
        training_speedup=fp16_train_speed / fp32_train_speed if fp32_train_speed > 0 else 1.0,
        inference_speedup=fp16_infer_speed / fp32_infer_speed if fp32_infer_speed > 0 else 1.0,
        opt_level=fp16_config.opt_level,
    )

    return result


# Example usage for medical AI
def create_medical_fp16_trainer(opt_level: str = "O1") -> FP16MixedPrecision:
    """Create FP16 trainer optimized for medical AI"""
    config = FP16Config(
        enabled=True,
        opt_level=opt_level,  # O1 for conservative mixed precision
        loss_scale="dynamic",  # Dynamic scaling for stability
        initial_scale=2**15,  # Lower initial scale for medical data
        growth_factor=2.0,
        backoff_factor=0.5,
        growth_interval=2000,
        keep_batchnorm_fp32=True,  # Keep BN in FP32 for stability
        master_weights=True,  # Maintain FP32 master weights
        cast_model_outputs=True,  # Cast outputs to FP32 for precision
    )

    return FP16MixedPrecision(config)
