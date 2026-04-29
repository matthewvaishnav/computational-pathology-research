"""
Structured Pruning for Medical AI Models

Implements structured pruning (channels, filters, blocks) for neural networks
to achieve hardware-friendly compression with actual speedup benefits.
"""

import copy
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
import torch.nn as nn

from ..utils.model_utils import count_parameters, get_model_size


@dataclass
class StructuredPruningConfig:
    """Configuration for structured pruning"""

    pruning_ratio: float = 0.5  # Ratio of structures to prune
    pruning_granularity: str = "channel"  # 'channel', 'filter', 'block'
    importance_metric: str = "l1_norm"  # 'l1_norm', 'l2_norm', 'gradient', 'taylor'
    global_ranking: bool = True  # Global vs layer-wise ranking
    min_channels: int = 8  # Minimum channels to keep per layer
    skip_layers: List[str] = None  # Layer names to skip
    gradual_pruning: bool = True  # Gradual vs one-shot pruning
    pruning_steps: int = 5  # Number of gradual steps
    recovery_epochs: int = 10  # Fine-tuning epochs per step

    def __post_init__(self):
        if self.skip_layers is None:
            self.skip_layers = []


@dataclass
class StructuredPruningResult:
    """Results from structured pruning"""

    original_params: int
    pruned_params: int
    original_flops: int
    pruned_flops: int
    compression_ratio: float
    speedup_ratio: float
    accuracy_before: float
    accuracy_after: float
    accuracy_drop: float
    layers_pruned: int

    def to_dict(self) -> Dict[str, Any]:
        return {
            "original_params": self.original_params,
            "pruned_params": self.pruned_params,
            "original_flops": self.original_flops,
            "pruned_flops": self.pruned_flops,
            "compression_ratio": self.compression_ratio,
            "speedup_ratio": self.speedup_ratio,
            "accuracy_before": self.accuracy_before,
            "accuracy_after": self.accuracy_after,
            "accuracy_drop": self.accuracy_drop,
            "layers_pruned": self.layers_pruned,
        }


class StructuredPruner:
    """
    Structured neural network pruning

    Implements structured pruning strategies:
    - Channel pruning (remove entire channels)
    - Filter pruning (remove entire filters)
    - Block pruning (remove attention heads, etc.)
    - Various importance metrics
    - Hardware-aware optimization
    """

    def __init__(self, config: StructuredPruningConfig):
        """Initialize structured pruner"""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Pruning state
        self.original_model = None
        self.layer_info = {}
        self.importance_scores = {}
        self.pruning_plan = {}

    def prune_model(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer = None,
    ) -> StructuredPruningResult:
        """
        Apply structured pruning to model

        Args:
            model: PyTorch model to prune
            dataloader: Validation dataloader
            device: Device for computation
            criterion: Loss function
            optimizer: Optimizer for fine-tuning

        Returns:
            StructuredPruningResult with compression statistics
        """
        try:
            # Store original model
            self.original_model = copy.deepcopy(model)

            # Analyze model structure
            self._analyze_model_structure(model)

            # Evaluate original model
            original_accuracy = self._evaluate_model(model, dataloader, device, criterion)
            original_params = count_parameters(model)
            original_flops = self._estimate_flops(model)

            self.logger.info(
                f"Original model - Params: {original_params/1e6:.2f}M, "
                f"FLOPs: {original_flops/1e9:.2f}G, Accuracy: {original_accuracy:.4f}"
            )

            # Calculate importance scores
            self._calculate_importance_scores(model, dataloader, device, criterion)

            # Create pruning plan
            self._create_pruning_plan(model)

            # Apply pruning
            if self.config.gradual_pruning:
                pruned_model = self._gradual_structured_pruning(
                    model, dataloader, device, criterion, optimizer
                )
            else:
                pruned_model = self._one_shot_structured_pruning(model)

            # Evaluate pruned model
            pruned_accuracy = self._evaluate_model(pruned_model, dataloader, device, criterion)
            pruned_params = count_parameters(pruned_model)
            pruned_flops = self._estimate_flops(pruned_model)

            # Create result
            result = StructuredPruningResult(
                original_params=original_params,
                pruned_params=pruned_params,
                original_flops=original_flops,
                pruned_flops=pruned_flops,
                compression_ratio=original_params / pruned_params,
                speedup_ratio=original_flops / pruned_flops,
                accuracy_before=original_accuracy,
                accuracy_after=pruned_accuracy,
                accuracy_drop=original_accuracy - pruned_accuracy,
                layers_pruned=len(self.pruning_plan),
            )

            self.logger.info(
                f"Structured pruning complete - "
                f"Compression: {result.compression_ratio:.2f}x, "
                f"Speedup: {result.speedup_ratio:.2f}x, "
                f"Accuracy drop: {result.accuracy_drop:.4f}"
            )

            return result

        except Exception as e:
            self.logger.error(f"Error during structured pruning: {e}")
            raise

    def _analyze_model_structure(self, model: nn.Module):
        """Analyze model structure for pruning"""
        try:
            self.layer_info = {}

            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    self.layer_info[name] = {
                        "type": "conv2d",
                        "in_channels": module.in_channels,
                        "out_channels": module.out_channels,
                        "kernel_size": module.kernel_size,
                        "stride": module.stride,
                        "padding": module.padding,
                        "groups": module.groups,
                        "prunable": name not in self.config.skip_layers,
                    }
                elif isinstance(module, nn.Linear):
                    self.layer_info[name] = {
                        "type": "linear",
                        "in_features": module.in_features,
                        "out_features": module.out_features,
                        "prunable": name not in self.config.skip_layers,
                    }
                elif isinstance(module, nn.BatchNorm2d):
                    self.layer_info[name] = {
                        "type": "batchnorm2d",
                        "num_features": module.num_features,
                        "prunable": False,  # BN layers follow conv layers
                    }

            prunable_layers = sum(1 for info in self.layer_info.values() if info["prunable"])
            self.logger.info(f"Found {prunable_layers} prunable layers")

        except Exception as e:
            self.logger.error(f"Error analyzing model structure: {e}")
            raise

    def _calculate_importance_scores(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        criterion: nn.Module,
    ):
        """Calculate importance scores for structured elements"""
        try:
            self.importance_scores = {}

            if self.config.importance_metric == "l1_norm":
                self._calculate_l1_importance(model)
            elif self.config.importance_metric == "l2_norm":
                self._calculate_l2_importance(model)
            elif self.config.importance_metric == "gradient":
                self._calculate_gradient_importance(model, dataloader, device, criterion)
            elif self.config.importance_metric == "taylor":
                self._calculate_taylor_importance(model, dataloader, device, criterion)
            else:
                raise ValueError(f"Unknown importance metric: {self.config.importance_metric}")

            self.logger.info(f"Calculated {self.config.importance_metric} importance scores")

        except Exception as e:
            self.logger.error(f"Error calculating importance scores: {e}")
            raise

    def _calculate_l1_importance(self, model: nn.Module):
        """Calculate L1 norm based importance"""
        for name, module in model.named_modules():
            if name in self.layer_info and self.layer_info[name]["prunable"]:
                if isinstance(module, nn.Conv2d):
                    if self.config.pruning_granularity == "channel":
                        # Importance of output channels
                        weights = module.weight.data  # [out_ch, in_ch, h, w]
                        importance = torch.sum(torch.abs(weights), dim=(1, 2, 3))
                    elif self.config.pruning_granularity == "filter":
                        # Importance of input channels (filters)
                        weights = module.weight.data  # [out_ch, in_ch, h, w]
                        importance = torch.sum(torch.abs(weights), dim=(0, 2, 3))
                    else:
                        continue

                elif isinstance(module, nn.Linear):
                    if self.config.pruning_granularity == "channel":
                        # Importance of output features
                        weights = module.weight.data  # [out_feat, in_feat]
                        importance = torch.sum(torch.abs(weights), dim=1)
                    else:
                        continue
                else:
                    continue

                self.importance_scores[name] = importance.cpu().numpy()

    def _calculate_l2_importance(self, model: nn.Module):
        """Calculate L2 norm based importance"""
        for name, module in model.named_modules():
            if name in self.layer_info and self.layer_info[name]["prunable"]:
                if isinstance(module, nn.Conv2d):
                    if self.config.pruning_granularity == "channel":
                        weights = module.weight.data
                        importance = torch.sqrt(torch.sum(weights**2, dim=(1, 2, 3)))
                    elif self.config.pruning_granularity == "filter":
                        weights = module.weight.data
                        importance = torch.sqrt(torch.sum(weights**2, dim=(0, 2, 3)))
                    else:
                        continue

                elif isinstance(module, nn.Linear):
                    if self.config.pruning_granularity == "channel":
                        weights = module.weight.data
                        importance = torch.sqrt(torch.sum(weights**2, dim=1))
                    else:
                        continue
                else:
                    continue

                self.importance_scores[name] = importance.cpu().numpy()

    def _calculate_gradient_importance(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        criterion: nn.Module,
    ):
        """Calculate gradient-based importance"""
        model.train()

        # Accumulate gradients
        gradient_accumulator = {}

        for batch_idx, (data, target) in enumerate(dataloader):
            if batch_idx >= 10:  # Limit to 10 batches for efficiency
                break

            data, target = data.to(device), target.to(device)

            model.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # Accumulate gradients
            for name, module in model.named_modules():
                if name in self.layer_info and self.layer_info[name]["prunable"]:
                    if hasattr(module, "weight") and module.weight.grad is not None:
                        if name not in gradient_accumulator:
                            gradient_accumulator[name] = torch.zeros_like(module.weight.grad)
                        gradient_accumulator[name] += torch.abs(module.weight.grad)

        # Calculate importance from accumulated gradients
        for name, grad in gradient_accumulator.items():
            module = dict(model.named_modules())[name]

            if isinstance(module, nn.Conv2d):
                if self.config.pruning_granularity == "channel":
                    importance = torch.sum(grad, dim=(1, 2, 3))
                elif self.config.pruning_granularity == "filter":
                    importance = torch.sum(grad, dim=(0, 2, 3))
                else:
                    continue
            elif isinstance(module, nn.Linear):
                if self.config.pruning_granularity == "channel":
                    importance = torch.sum(grad, dim=1)
                else:
                    continue
            else:
                continue

            self.importance_scores[name] = importance.cpu().numpy()

    def _calculate_taylor_importance(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        criterion: nn.Module,
    ):
        """Calculate Taylor expansion based importance"""
        # Taylor importance = |weight * gradient|
        model.train()

        taylor_accumulator = {}

        for batch_idx, (data, target) in enumerate(dataloader):
            if batch_idx >= 10:  # Limit batches
                break

            data, target = data.to(device), target.to(device)

            model.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()

            # Calculate Taylor scores
            for name, module in model.named_modules():
                if name in self.layer_info and self.layer_info[name]["prunable"]:
                    if hasattr(module, "weight") and module.weight.grad is not None:
                        taylor_score = torch.abs(module.weight.data * module.weight.grad)

                        if name not in taylor_accumulator:
                            taylor_accumulator[name] = torch.zeros_like(taylor_score)
                        taylor_accumulator[name] += taylor_score

        # Calculate importance from Taylor scores
        for name, taylor in taylor_accumulator.items():
            module = dict(model.named_modules())[name]

            if isinstance(module, nn.Conv2d):
                if self.config.pruning_granularity == "channel":
                    importance = torch.sum(taylor, dim=(1, 2, 3))
                elif self.config.pruning_granularity == "filter":
                    importance = torch.sum(taylor, dim=(0, 2, 3))
                else:
                    continue
            elif isinstance(module, nn.Linear):
                if self.config.pruning_granularity == "channel":
                    importance = torch.sum(taylor, dim=1)
                else:
                    continue
            else:
                continue

            self.importance_scores[name] = importance.cpu().numpy()

    def _create_pruning_plan(self, model: nn.Module):
        """Create structured pruning plan"""
        try:
            self.pruning_plan = {}

            if self.config.global_ranking:
                # Global ranking across all layers
                all_scores = []
                layer_indices = {}

                for name, scores in self.importance_scores.items():
                    start_idx = len(all_scores)
                    all_scores.extend(scores.tolist())
                    layer_indices[name] = (start_idx, len(all_scores))

                # Sort globally and determine pruning threshold
                sorted_indices = np.argsort(all_scores)
                num_to_prune = int(len(all_scores) * self.config.pruning_ratio)
                prune_indices = set(sorted_indices[:num_to_prune])

                # Map back to layers
                for name, (start_idx, end_idx) in layer_indices.items():
                    layer_prune_indices = []
                    for i in range(start_idx, end_idx):
                        if i in prune_indices:
                            layer_prune_indices.append(i - start_idx)

                    if layer_prune_indices:
                        # Ensure minimum channels
                        total_channels = len(self.importance_scores[name])
                        max_prune = max(0, total_channels - self.config.min_channels)
                        layer_prune_indices = layer_prune_indices[:max_prune]

                        if layer_prune_indices:
                            self.pruning_plan[name] = layer_prune_indices

            else:
                # Layer-wise ranking
                for name, scores in self.importance_scores.items():
                    num_channels = len(scores)
                    num_to_prune = int(num_channels * self.config.pruning_ratio)

                    # Ensure minimum channels
                    max_prune = max(0, num_channels - self.config.min_channels)
                    num_to_prune = min(num_to_prune, max_prune)

                    if num_to_prune > 0:
                        prune_indices = np.argsort(scores)[:num_to_prune].tolist()
                        self.pruning_plan[name] = prune_indices

            total_pruned = sum(len(indices) for indices in self.pruning_plan.values())
            self.logger.info(
                f"Created pruning plan: {len(self.pruning_plan)} layers, "
                f"{total_pruned} structures to prune"
            )

        except Exception as e:
            self.logger.error(f"Error creating pruning plan: {e}")
            raise

    def _one_shot_structured_pruning(self, model: nn.Module) -> nn.Module:
        """Apply one-shot structured pruning"""
        try:
            # Apply pruning according to plan
            for layer_name, prune_indices in self.pruning_plan.items():
                self._prune_layer_structures(model, layer_name, prune_indices)

            return model

        except Exception as e:
            self.logger.error(f"Error in one-shot structured pruning: {e}")
            raise

    def _gradual_structured_pruning(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> nn.Module:
        """Apply gradual structured pruning"""
        try:
            if optimizer is None:
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

            # Create gradual pruning schedule
            pruning_schedule = self._create_gradual_schedule()

            for step, step_plan in enumerate(pruning_schedule):
                self.logger.info(f"Gradual pruning step {step+1}/{len(pruning_schedule)}")

                # Apply pruning for this step
                for layer_name, prune_indices in step_plan.items():
                    self._prune_layer_structures(model, layer_name, prune_indices)

                # Fine-tune model
                if self.config.recovery_epochs > 0:
                    self._fine_tune_model(model, dataloader, device, criterion, optimizer)

                # Evaluate intermediate result
                accuracy = self._evaluate_model(model, dataloader, device, criterion)
                params = count_parameters(model)

                self.logger.info(
                    f"Step {step+1} - Params: {params/1e6:.2f}M, Accuracy: {accuracy:.4f}"
                )

            return model

        except Exception as e:
            self.logger.error(f"Error in gradual structured pruning: {e}")
            raise

    def _create_gradual_schedule(self) -> List[Dict[str, List[int]]]:
        """Create gradual pruning schedule"""
        schedule = []

        for step in range(self.config.pruning_steps):
            step_plan = {}

            for layer_name, total_prune_indices in self.pruning_plan.items():
                # Determine how many to prune in this step
                total_to_prune = len(total_prune_indices)
                step_progress = (step + 1) / self.config.pruning_steps

                # Cubic schedule for gradual pruning
                num_in_step = int(total_to_prune * (step_progress**3))

                # Get indices for this step (not already pruned)
                already_pruned = 0
                for prev_step in schedule:
                    if layer_name in prev_step:
                        already_pruned += len(prev_step[layer_name])

                step_indices = total_prune_indices[already_pruned:num_in_step]

                if step_indices:
                    step_plan[layer_name] = step_indices

            if step_plan:
                schedule.append(step_plan)

        return schedule

    def _prune_layer_structures(self, model: nn.Module, layer_name: str, prune_indices: List[int]):
        """Prune structures from a specific layer"""
        try:
            module = dict(model.named_modules())[layer_name]

            if isinstance(module, nn.Conv2d):
                self._prune_conv2d_structures(module, prune_indices)
            elif isinstance(module, nn.Linear):
                self._prune_linear_structures(module, prune_indices)

            # Update corresponding BatchNorm layers
            self._update_batchnorm_layers(model, layer_name, prune_indices)

        except Exception as e:
            self.logger.error(f"Error pruning layer {layer_name}: {e}")
            raise

    def _prune_conv2d_structures(self, module: nn.Conv2d, prune_indices: List[int]):
        """Prune Conv2d structures (channels or filters)"""
        if self.config.pruning_granularity == "channel":
            # Prune output channels
            keep_indices = [i for i in range(module.out_channels) if i not in prune_indices]

            # Update weight
            module.weight.data = module.weight.data[keep_indices]
            module.out_channels = len(keep_indices)

            # Update bias if exists
            if module.bias is not None:
                module.bias.data = module.bias.data[keep_indices]

        elif self.config.pruning_granularity == "filter":
            # Prune input channels (filters)
            keep_indices = [i for i in range(module.in_channels) if i not in prune_indices]

            # Update weight
            module.weight.data = module.weight.data[:, keep_indices]
            module.in_channels = len(keep_indices)

    def _prune_linear_structures(self, module: nn.Linear, prune_indices: List[int]):
        """Prune Linear layer structures"""
        if self.config.pruning_granularity == "channel":
            # Prune output features
            keep_indices = [i for i in range(module.out_features) if i not in prune_indices]

            # Update weight
            module.weight.data = module.weight.data[keep_indices]
            module.out_features = len(keep_indices)

            # Update bias if exists
            if module.bias is not None:
                module.bias.data = module.bias.data[keep_indices]

    def _update_batchnorm_layers(
        self, model: nn.Module, conv_layer_name: str, prune_indices: List[int]
    ):
        """Update BatchNorm layers that follow pruned Conv layers"""
        try:
            # Find corresponding BatchNorm layer
            bn_layer_name = None
            for name, module in model.named_modules():
                if isinstance(module, nn.BatchNorm2d) and name.startswith(
                    conv_layer_name.rsplit(".", 1)[0]
                ):
                    bn_layer_name = name
                    break

            if bn_layer_name and self.config.pruning_granularity == "channel":
                bn_module = dict(model.named_modules())[bn_layer_name]
                keep_indices = [i for i in range(bn_module.num_features) if i not in prune_indices]

                # Update BatchNorm parameters
                bn_module.weight.data = bn_module.weight.data[keep_indices]
                bn_module.bias.data = bn_module.bias.data[keep_indices]
                bn_module.running_mean.data = bn_module.running_mean.data[keep_indices]
                bn_module.running_var.data = bn_module.running_var.data[keep_indices]
                bn_module.num_features = len(keep_indices)

        except Exception as e:
            self.logger.error(f"Error updating BatchNorm layers: {e}")

    def _fine_tune_model(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ):
        """Fine-tune model after pruning step"""
        try:
            model.train()

            for epoch in range(self.config.recovery_epochs):
                total_loss = 0.0
                num_batches = 0

                for batch_idx, (data, target) in enumerate(dataloader):
                    data, target = data.to(device), target.to(device)

                    optimizer.zero_grad()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                    optimizer.step()

                    total_loss += loss.item()
                    num_batches += 1

                    # Limit fine-tuning batches
                    if batch_idx >= 100:
                        break

                avg_loss = total_loss / num_batches
                self.logger.debug(f"Fine-tuning epoch {epoch+1}, loss: {avg_loss:.4f}")

        except Exception as e:
            self.logger.error(f"Error during fine-tuning: {e}")
            raise

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

    def _estimate_flops(
        self, model: nn.Module, input_size: Tuple[int, int, int, int] = (1, 3, 224, 224)
    ) -> int:
        """Estimate FLOPs for the model"""
        try:
            total_flops = 0

            # Simple FLOP estimation for Conv2d and Linear layers
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    # FLOPs = output_elements * (kernel_flops + bias_flops)
                    kernel_flops = (
                        module.kernel_size[0] * module.kernel_size[1] * module.in_channels
                    )
                    output_elements = (
                        (input_size[2] // module.stride[0])
                        * (input_size[3] // module.stride[1])
                        * module.out_channels
                    )
                    flops = output_elements * kernel_flops
                    total_flops += flops

                elif isinstance(module, nn.Linear):
                    flops = module.in_features * module.out_features
                    total_flops += flops

            return total_flops

        except Exception as e:
            self.logger.error(f"Error estimating FLOPs: {e}")
            return 0


def create_structured_pruning_config(
    pruning_ratio: float = 0.5,
    granularity: str = "channel",
    importance_metric: str = "l1_norm",
    gradual: bool = True,
) -> StructuredPruningConfig:
    """Create structured pruning configuration"""
    return StructuredPruningConfig(
        pruning_ratio=pruning_ratio,
        pruning_granularity=granularity,
        importance_metric=importance_metric,
        global_ranking=True,
        gradual_pruning=gradual,
        pruning_steps=5 if gradual else 1,
        recovery_epochs=10 if gradual else 0,
    )


# Example usage for medical AI
def prune_pathology_model_structured(
    model: nn.Module, val_loader: torch.utils.data.DataLoader, pruning_ratio: float = 0.3
) -> StructuredPruningResult:
    """Structured pruning optimized for pathology models"""
    config = StructuredPruningConfig(
        pruning_ratio=pruning_ratio,
        pruning_granularity="channel",  # Channel pruning for better hardware support
        importance_metric="taylor",  # Taylor for better accuracy preservation
        global_ranking=True,  # Global for optimal distribution
        min_channels=16,  # Higher minimum for medical accuracy
        gradual_pruning=True,  # Gradual for stability
        pruning_steps=8,  # More steps for medical applications
        recovery_epochs=15,  # More recovery epochs
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)  # Lower LR for medical models

    pruner = StructuredPruner(config)
    return pruner.prune_model(model, val_loader, device, criterion, optimizer)
