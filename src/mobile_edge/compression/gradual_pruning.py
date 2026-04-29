"""
Gradual Pruning Scheduler for Medical AI Models

Implements various gradual pruning schedules to achieve target sparsity
while maintaining model performance through controlled compression.
"""

import json
import logging
import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from ..utils.model_utils import count_parameters, get_model_size


@dataclass
class GradualPruningConfig:
    """Configuration for gradual pruning"""

    initial_sparsity: float = 0.0  # Starting sparsity
    target_sparsity: float = 0.9  # Target final sparsity
    pruning_frequency: int = 100  # Prune every N steps
    schedule_type: str = "polynomial"  # 'polynomial', 'exponential', 'cosine', 'linear'
    schedule_power: float = 3.0  # Power for polynomial schedule
    warmup_steps: int = 1000  # Steps before pruning starts
    total_steps: int = 10000  # Total training steps
    magnitude_based: bool = True  # Use magnitude-based pruning
    global_pruning: bool = True  # Global vs layer-wise pruning
    recovery_steps: int = 500  # Steps for recovery after pruning
    min_sparsity_per_layer: float = 0.1  # Minimum sparsity per layer
    max_sparsity_per_layer: float = 0.95  # Maximum sparsity per layer
    skip_layers: List[str] = None  # Layer patterns to skip

    def __post_init__(self):
        if self.skip_layers is None:
            self.skip_layers = ["classifier", "fc", "head", "final", "output"]


@dataclass
class PruningStep:
    """Information about a single pruning step"""

    step: int
    target_sparsity: float
    actual_sparsity: float
    accuracy_before: float
    accuracy_after: float
    pruned_weights: int
    total_weights: int


class PruningSchedule(ABC):
    """Abstract base class for pruning schedules"""

    @abstractmethod
    def get_sparsity_at_step(self, step: int, config: GradualPruningConfig) -> float:
        """Get target sparsity at given step"""
        pass

    @abstractmethod
    def should_prune_at_step(self, step: int, config: GradualPruningConfig) -> bool:
        """Check if pruning should occur at given step"""
        pass


class PolynomialSchedule(PruningSchedule):
    """Polynomial pruning schedule: s_t = s_f + (s_i - s_f) * (1 - t/T)^p"""

    def get_sparsity_at_step(self, step: int, config: GradualPruningConfig) -> float:
        if step < config.warmup_steps:
            return config.initial_sparsity

        if step >= config.total_steps:
            return config.target_sparsity

        # Normalize step to [0, 1]
        t = (step - config.warmup_steps) / (config.total_steps - config.warmup_steps)

        # Polynomial schedule
        sparsity = config.target_sparsity + (config.initial_sparsity - config.target_sparsity) * (
            (1 - t) ** config.schedule_power
        )

        return sparsity

    def should_prune_at_step(self, step: int, config: GradualPruningConfig) -> bool:
        return (
            step >= config.warmup_steps
            and step < config.total_steps
            and step % config.pruning_frequency == 0
        )


class ExponentialSchedule(PruningSchedule):
    """Exponential pruning schedule: s_t = s_f - (s_f - s_i) * exp(-λt)"""

    def get_sparsity_at_step(self, step: int, config: GradualPruningConfig) -> float:
        if step < config.warmup_steps:
            return config.initial_sparsity

        if step >= config.total_steps:
            return config.target_sparsity

        # Normalize step to [0, 1]
        t = (step - config.warmup_steps) / (config.total_steps - config.warmup_steps)

        # Exponential schedule (λ = 5 for reasonable decay)
        lambda_val = 5.0
        sparsity = config.target_sparsity - (
            config.target_sparsity - config.initial_sparsity
        ) * math.exp(-lambda_val * t)

        return sparsity

    def should_prune_at_step(self, step: int, config: GradualPruningConfig) -> bool:
        return (
            step >= config.warmup_steps
            and step < config.total_steps
            and step % config.pruning_frequency == 0
        )


class CosineSchedule(PruningSchedule):
    """Cosine annealing pruning schedule"""

    def get_sparsity_at_step(self, step: int, config: GradualPruningConfig) -> float:
        if step < config.warmup_steps:
            return config.initial_sparsity

        if step >= config.total_steps:
            return config.target_sparsity

        # Normalize step to [0, 1]
        t = (step - config.warmup_steps) / (config.total_steps - config.warmup_steps)

        # Cosine schedule
        sparsity = (
            config.initial_sparsity
            + (config.target_sparsity - config.initial_sparsity) * (1 - math.cos(math.pi * t)) / 2
        )

        return sparsity

    def should_prune_at_step(self, step: int, config: GradualPruningConfig) -> bool:
        return (
            step >= config.warmup_steps
            and step < config.total_steps
            and step % config.pruning_frequency == 0
        )


class LinearSchedule(PruningSchedule):
    """Linear pruning schedule"""

    def get_sparsity_at_step(self, step: int, config: GradualPruningConfig) -> float:
        if step < config.warmup_steps:
            return config.initial_sparsity

        if step >= config.total_steps:
            return config.target_sparsity

        # Linear interpolation
        t = (step - config.warmup_steps) / (config.total_steps - config.warmup_steps)
        sparsity = config.initial_sparsity + (config.target_sparsity - config.initial_sparsity) * t

        return sparsity

    def should_prune_at_step(self, step: int, config: GradualPruningConfig) -> bool:
        return (
            step >= config.warmup_steps
            and step < config.total_steps
            and step % config.pruning_frequency == 0
        )


class GradualPruner:
    """
    Gradual pruning implementation with various schedules

    Supports multiple pruning schedules:
    - Polynomial (default, good for most cases)
    - Exponential (aggressive early pruning)
    - Cosine (smooth transitions)
    - Linear (constant rate)

    Features:
    - Magnitude-based or gradient-based pruning
    - Global or layer-wise pruning strategies
    - Recovery periods after pruning steps
    - Layer-specific sparsity constraints
    """

    def __init__(self, config: GradualPruningConfig):
        """Initialize gradual pruner"""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Initialize schedule
        self.schedule = self._create_schedule()

        # Pruning state
        self.masks = {}
        self.current_step = 0
        self.pruning_history = []
        self.last_prune_step = -1

        # Statistics
        self.total_weights = 0
        self.current_sparsity = 0.0

    def _create_schedule(self) -> PruningSchedule:
        """Create pruning schedule based on config"""
        if self.config.schedule_type == "polynomial":
            return PolynomialSchedule()
        elif self.config.schedule_type == "exponential":
            return ExponentialSchedule()
        elif self.config.schedule_type == "cosine":
            return CosineSchedule()
        elif self.config.schedule_type == "linear":
            return LinearSchedule()
        else:
            raise ValueError(f"Unknown schedule type: {self.config.schedule_type}")

    def initialize_masks(self, model: nn.Module):
        """Initialize pruning masks for all prunable layers"""
        try:
            self.masks = {}
            self.total_weights = 0

            for name, param in model.named_parameters():
                if self._is_prunable_layer(name):
                    # Initialize with all ones (no pruning initially)
                    self.masks[name] = torch.ones_like(param.data)
                    self.total_weights += param.numel()

            self.logger.info(
                f"Initialized masks for {len(self.masks)} layers, "
                f"total weights: {self.total_weights}"
            )

        except Exception as e:
            self.logger.error(f"Error initializing masks: {e}")
            raise

    def step(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer = None,
        val_loader: torch.utils.data.DataLoader = None,
        device: torch.device = None,
        criterion: nn.Module = None,
    ) -> bool:
        """
        Perform one step of gradual pruning

        Args:
            model: Model to prune
            optimizer: Optimizer (for gradient-based pruning)
            val_loader: Validation loader (for accuracy tracking)
            device: Device for evaluation
            criterion: Loss criterion

        Returns:
            True if pruning occurred at this step
        """
        try:
            self.current_step += 1

            # Check if we should prune at this step
            if not self.schedule.should_prune_at_step(self.current_step, self.config):
                return False

            # Get target sparsity for this step
            target_sparsity = self.schedule.get_sparsity_at_step(self.current_step, self.config)

            # Skip if target sparsity hasn't increased significantly
            if target_sparsity <= self.current_sparsity + 0.001:
                return False

            # Evaluate accuracy before pruning
            accuracy_before = 0.0
            if val_loader is not None and device is not None and criterion is not None:
                accuracy_before = self._evaluate_model(model, val_loader, device, criterion)

            # Apply pruning to reach target sparsity
            pruned_weights = self._prune_to_sparsity(model, target_sparsity)

            # Update current sparsity
            self.current_sparsity = self._calculate_current_sparsity()

            # Evaluate accuracy after pruning
            accuracy_after = 0.0
            if val_loader is not None and device is not None and criterion is not None:
                accuracy_after = self._evaluate_model(model, val_loader, device, criterion)

            # Record pruning step
            step_info = PruningStep(
                step=self.current_step,
                target_sparsity=target_sparsity,
                actual_sparsity=self.current_sparsity,
                accuracy_before=accuracy_before,
                accuracy_after=accuracy_after,
                pruned_weights=pruned_weights,
                total_weights=self.total_weights,
            )
            self.pruning_history.append(step_info)
            self.last_prune_step = self.current_step

            self.logger.info(
                f"Pruning step {self.current_step} - "
                f"Target: {target_sparsity:.3f}, "
                f"Actual: {self.current_sparsity:.3f}, "
                f"Pruned: {pruned_weights} weights"
            )

            if accuracy_before > 0 and accuracy_after > 0:
                accuracy_drop = accuracy_before - accuracy_after
                self.logger.info(
                    f"Accuracy: {accuracy_before:.4f} → {accuracy_after:.4f} "
                    f"(drop: {accuracy_drop:.4f})"
                )

            return True

        except Exception as e:
            self.logger.error(f"Error in pruning step: {e}")
            return False

    def _is_prunable_layer(self, layer_name: str) -> bool:
        """Check if layer should be pruned"""
        # Skip bias terms
        if "bias" in layer_name:
            return False

        # Skip specified layer patterns
        for skip_pattern in self.config.skip_layers:
            if skip_pattern in layer_name:
                return False

        return True

    def _prune_to_sparsity(self, model: nn.Module, target_sparsity: float) -> int:
        """Prune model to achieve target sparsity"""
        try:
            if self.config.global_pruning:
                return self._global_magnitude_pruning(model, target_sparsity)
            else:
                return self._layerwise_magnitude_pruning(model, target_sparsity)

        except Exception as e:
            self.logger.error(f"Error pruning to sparsity: {e}")
            return 0

    def _global_magnitude_pruning(self, model: nn.Module, target_sparsity: float) -> int:
        """Global magnitude-based pruning to target sparsity"""
        # Collect all weights with their magnitudes
        all_weights = []
        weight_locations = []

        for name, param in model.named_parameters():
            if name in self.masks:
                # Only consider currently active weights
                active_mask = self.masks[name] == 1
                if not active_mask.any():
                    continue

                active_weights = param.data[active_mask]
                magnitudes = torch.abs(active_weights)

                # Store weights and their locations
                flat_magnitudes = magnitudes.flatten()
                all_weights.extend(flat_magnitudes.tolist())

                # Store location info for updating masks
                active_indices = torch.nonzero(active_mask, as_tuple=False)
                for i, mag in enumerate(flat_magnitudes):
                    weight_locations.append((name, active_indices[i], mag.item()))

        if len(all_weights) == 0:
            return 0

        # Calculate number of weights to keep
        num_weights = len(all_weights)
        num_to_keep = int(num_weights * (1 - target_sparsity))

        if num_to_keep >= num_weights:
            return 0  # No pruning needed

        # Find threshold for keeping top weights
        sorted_weights = sorted(all_weights, reverse=True)
        threshold = sorted_weights[num_to_keep - 1] if num_to_keep > 0 else float("inf")

        # Update masks based on threshold
        pruned_count = 0
        for name, param in model.named_parameters():
            if name in self.masks:
                # Create new mask based on magnitude threshold
                magnitude_mask = torch.abs(param.data) >= threshold

                # Apply layer-specific constraints
                layer_sparsity = 1.0 - magnitude_mask.float().mean().item()
                if layer_sparsity < self.config.min_sparsity_per_layer:
                    # Force minimum sparsity
                    num_weights_layer = param.numel()
                    num_to_keep_layer = int(
                        num_weights_layer * (1 - self.config.min_sparsity_per_layer)
                    )
                    flat_weights = torch.abs(param.data).flatten()
                    threshold_layer = torch.kthvalue(
                        flat_weights, num_weights_layer - num_to_keep_layer + 1
                    )[0]
                    magnitude_mask = torch.abs(param.data) >= threshold_layer
                elif layer_sparsity > self.config.max_sparsity_per_layer:
                    # Limit maximum sparsity
                    num_weights_layer = param.numel()
                    num_to_keep_layer = int(
                        num_weights_layer * (1 - self.config.max_sparsity_per_layer)
                    )
                    flat_weights = torch.abs(param.data).flatten()
                    threshold_layer = torch.kthvalue(
                        flat_weights, num_weights_layer - num_to_keep_layer + 1
                    )[0]
                    magnitude_mask = torch.abs(param.data) >= threshold_layer

                # Count pruned weights
                old_active = self.masks[name].sum().item()
                self.masks[name] = magnitude_mask.float()
                new_active = self.masks[name].sum().item()
                pruned_count += int(old_active - new_active)

        return pruned_count

    def _layerwise_magnitude_pruning(self, model: nn.Module, target_sparsity: float) -> int:
        """Layer-wise magnitude-based pruning"""
        total_pruned = 0

        for name, param in model.named_parameters():
            if name in self.masks:
                # Calculate target sparsity for this layer
                current_layer_sparsity = 1.0 - self.masks[name].mean().item()

                # Interpolate between current and target sparsity
                layer_target_sparsity = min(
                    max(target_sparsity, self.config.min_sparsity_per_layer),
                    self.config.max_sparsity_per_layer,
                )

                if layer_target_sparsity <= current_layer_sparsity:
                    continue

                # Determine number of weights to keep
                num_weights = param.numel()
                num_to_keep = int(num_weights * (1 - layer_target_sparsity))

                if num_to_keep <= 0:
                    continue

                # Find threshold for this layer
                flat_weights = torch.abs(param.data).flatten()
                threshold = torch.kthvalue(flat_weights, num_weights - num_to_keep + 1)[0]

                # Update mask
                old_active = self.masks[name].sum().item()
                self.masks[name] = (torch.abs(param.data) >= threshold).float()
                new_active = self.masks[name].sum().item()
                total_pruned += int(old_active - new_active)

        return total_pruned

    def _calculate_current_sparsity(self) -> float:
        """Calculate current overall sparsity"""
        if self.total_weights == 0:
            return 0.0

        active_weights = sum(mask.sum().item() for mask in self.masks.values())
        return 1.0 - (active_weights / self.total_weights)

    def apply_masks(self, model: nn.Module):
        """Apply current pruning masks to model parameters"""
        for name, param in model.named_parameters():
            if name in self.masks:
                param.data *= self.masks[name]

    def zero_pruned_gradients(self, model: nn.Module):
        """Zero gradients of pruned weights"""
        for name, param in model.named_parameters():
            if name in self.masks and param.grad is not None:
                param.grad *= self.masks[name]

    def is_in_recovery_period(self) -> bool:
        """Check if we're in recovery period after recent pruning"""
        if self.last_prune_step < 0:
            return False

        steps_since_prune = self.current_step - self.last_prune_step
        return steps_since_prune < self.config.recovery_steps

    def get_current_sparsity(self) -> float:
        """Get current sparsity level"""
        return self.current_sparsity

    def get_target_sparsity_at_step(self, step: int) -> float:
        """Get target sparsity at given step"""
        return self.schedule.get_sparsity_at_step(step, self.config)

    def _evaluate_model(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        criterion: nn.Module,
    ) -> float:
        """Evaluate model accuracy"""
        try:
            model.eval()
            correct = 0
            total = 0

            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(dataloader):
                    data, target = data.to(device), target.to(device)

                    # Apply masks before evaluation
                    self.apply_masks(model)

                    output = model(data)
                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)

                    # Limit evaluation for efficiency
                    if batch_idx >= 50:
                        break

            accuracy = correct / total if total > 0 else 0.0
            return accuracy

        except Exception as e:
            self.logger.error(f"Error evaluating model: {e}")
            return 0.0

    def save_pruning_state(self, save_path: Path):
        """Save current pruning state"""
        try:
            state = {
                "config": self.config.__dict__,
                "current_step": self.current_step,
                "current_sparsity": self.current_sparsity,
                "total_weights": self.total_weights,
                "last_prune_step": self.last_prune_step,
                "masks": {name: mask.cpu().numpy().tolist() for name, mask in self.masks.items()},
                "pruning_history": [
                    {
                        "step": step.step,
                        "target_sparsity": step.target_sparsity,
                        "actual_sparsity": step.actual_sparsity,
                        "accuracy_before": step.accuracy_before,
                        "accuracy_after": step.accuracy_after,
                        "pruned_weights": step.pruned_weights,
                        "total_weights": step.total_weights,
                    }
                    for step in self.pruning_history
                ],
            }

            with open(save_path, "w") as f:
                json.dump(state, f, indent=2)

            self.logger.info(f"Saved pruning state to {save_path}")

        except Exception as e:
            self.logger.error(f"Error saving pruning state: {e}")
            raise


def create_gradual_pruning_config(
    target_sparsity: float = 0.9, total_steps: int = 10000, schedule_type: str = "polynomial"
) -> GradualPruningConfig:
    """Create gradual pruning configuration"""
    return GradualPruningConfig(
        initial_sparsity=0.0,
        target_sparsity=target_sparsity,
        pruning_frequency=100,
        schedule_type=schedule_type,
        schedule_power=3.0,
        warmup_steps=1000,
        total_steps=total_steps,
        magnitude_based=True,
        global_pruning=True,
        recovery_steps=500,
        min_sparsity_per_layer=0.1,
        max_sparsity_per_layer=0.95,
    )


# Example usage for medical AI training loop
def create_medical_gradual_pruner(
    target_sparsity: float = 0.8, total_epochs: int = 100, steps_per_epoch: int = 1000
) -> GradualPruner:
    """Create gradual pruner optimized for medical AI training"""
    total_steps = total_epochs * steps_per_epoch

    config = GradualPruningConfig(
        initial_sparsity=0.0,
        target_sparsity=target_sparsity,
        pruning_frequency=200,  # Prune every 200 steps
        schedule_type="polynomial",  # Smooth polynomial schedule
        schedule_power=3.0,  # Cubic schedule for gradual increase
        warmup_steps=total_steps // 10,  # 10% warmup
        total_steps=int(total_steps * 0.8),  # Complete pruning by 80% of training
        magnitude_based=True,
        global_pruning=True,
        recovery_steps=1000,  # Longer recovery for medical models
        min_sparsity_per_layer=0.05,  # Conservative minimum
        max_sparsity_per_layer=0.9,  # Conservative maximum
        skip_layers=["classifier", "fc", "head", "final", "output"],
    )

    return GradualPruner(config)
