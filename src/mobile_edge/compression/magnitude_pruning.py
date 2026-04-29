"""
Magnitude-Based Pruning for Medical AI Models

Implements magnitude-based weight pruning to compress neural networks
for mobile and edge deployment while maintaining diagnostic accuracy.
"""

import copy
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune

from ..utils.model_utils import count_parameters, get_model_size


@dataclass
class PruningConfig:
    """Configuration for magnitude-based pruning"""

    sparsity_ratio: float = 0.5  # Target sparsity (0.5 = 50% weights pruned)
    structured: bool = False  # Structured vs unstructured pruning
    global_pruning: bool = True  # Global vs layer-wise pruning
    exclude_layers: List[str] = None  # Layer types to exclude from pruning
    gradual_pruning: bool = True  # Gradual vs one-shot pruning
    pruning_steps: int = 10  # Number of gradual pruning steps
    recovery_epochs: int = 5  # Fine-tuning epochs after each step

    def __post_init__(self):
        if self.exclude_layers is None:
            self.exclude_layers = ["BatchNorm2d", "LayerNorm", "Embedding"]


@dataclass
class PruningResult:
    """Results from pruning operation"""

    original_size: int
    pruned_size: int
    compression_ratio: float
    sparsity_achieved: float
    accuracy_before: float
    accuracy_after: float
    accuracy_drop: float

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            "original_size": self.original_size,
            "pruned_size": self.pruned_size,
            "compression_ratio": self.compression_ratio,
            "sparsity_achieved": self.sparsity_achieved,
            "accuracy_before": self.accuracy_before,
            "accuracy_after": self.accuracy_after,
            "accuracy_drop": self.accuracy_drop,
        }


class MagnitudePruner:
    """
    Magnitude-based neural network pruning

    Implements various magnitude-based pruning strategies:
    - Unstructured pruning (individual weights)
    - Structured pruning (channels, filters)
    - Global vs layer-wise pruning
    - Gradual pruning with recovery
    """

    def __init__(self, config: PruningConfig):
        """Initialize magnitude pruner"""
        self.config = config
        self.logger = logging.getLogger(__name__)

        # Pruning state
        self.original_model = None
        self.pruned_modules = []
        self.pruning_masks = {}

    def prune_model(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer = None,
    ) -> PruningResult:
        """
        Prune model using magnitude-based pruning

        Args:
            model: PyTorch model to prune
            dataloader: Validation dataloader for accuracy evaluation
            device: Device to run evaluation on
            criterion: Loss function for evaluation
            optimizer: Optimizer for fine-tuning (optional)

        Returns:
            PruningResult with compression statistics
        """
        try:
            # Store original model
            self.original_model = copy.deepcopy(model)

            # Evaluate original model
            original_accuracy = self._evaluate_model(model, dataloader, device, criterion)
            original_size = get_model_size(model)

            self.logger.info(
                f"Original model - Size: {original_size/1e6:.2f}MB, Accuracy: {original_accuracy:.4f}"
            )

            # Apply pruning
            if self.config.gradual_pruning:
                pruned_model = self._gradual_pruning(
                    model, dataloader, device, criterion, optimizer
                )
            else:
                pruned_model = self._one_shot_pruning(model)

            # Evaluate pruned model
            pruned_accuracy = self._evaluate_model(pruned_model, dataloader, device, criterion)
            pruned_size = get_model_size(pruned_model)

            # Calculate sparsity
            sparsity = self._calculate_sparsity(pruned_model)

            # Create result
            result = PruningResult(
                original_size=original_size,
                pruned_size=pruned_size,
                compression_ratio=original_size / pruned_size,
                sparsity_achieved=sparsity,
                accuracy_before=original_accuracy,
                accuracy_after=pruned_accuracy,
                accuracy_drop=original_accuracy - pruned_accuracy,
            )

            self.logger.info(
                f"Pruning complete - Compression: {result.compression_ratio:.2f}x, "
                f"Sparsity: {result.sparsity_achieved:.2f}, "
                f"Accuracy drop: {result.accuracy_drop:.4f}"
            )

            return result

        except Exception as e:
            self.logger.error(f"Error during pruning: {e}")
            raise

    def _one_shot_pruning(self, model: nn.Module) -> nn.Module:
        """Apply one-shot magnitude-based pruning"""
        try:
            # Get modules to prune
            modules_to_prune = self._get_prunable_modules(model)

            if self.config.global_pruning:
                # Global magnitude-based pruning
                self._apply_global_pruning(modules_to_prune)
            else:
                # Layer-wise pruning
                self._apply_layerwise_pruning(modules_to_prune)

            # Make pruning permanent
            self._make_pruning_permanent(model)

            return model

        except Exception as e:
            self.logger.error(f"Error in one-shot pruning: {e}")
            raise

    def _gradual_pruning(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
    ) -> nn.Module:
        """Apply gradual magnitude-based pruning with recovery"""
        try:
            if optimizer is None:
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

            # Calculate sparsity schedule
            sparsity_schedule = self._create_sparsity_schedule()

            for step, target_sparsity in enumerate(sparsity_schedule):
                self.logger.info(
                    f"Pruning step {step+1}/{len(sparsity_schedule)}, "
                    f"target sparsity: {target_sparsity:.3f}"
                )

                # Apply pruning for this step
                self._apply_pruning_step(model, target_sparsity)

                # Fine-tune model
                if self.config.recovery_epochs > 0:
                    self._fine_tune_model(model, dataloader, device, criterion, optimizer)

                # Evaluate intermediate result
                accuracy = self._evaluate_model(model, dataloader, device, criterion)
                sparsity = self._calculate_sparsity(model)

                self.logger.info(
                    f"Step {step+1} - Sparsity: {sparsity:.3f}, Accuracy: {accuracy:.4f}"
                )

            # Make final pruning permanent
            self._make_pruning_permanent(model)

            return model

        except Exception as e:
            self.logger.error(f"Error in gradual pruning: {e}")
            raise

    def _get_prunable_modules(self, model: nn.Module) -> List[Tuple[nn.Module, str]]:
        """Get list of modules that can be pruned"""
        modules_to_prune = []

        for name, module in model.named_modules():
            # Check if module type should be excluded
            module_type = type(module).__name__
            if module_type in self.config.exclude_layers:
                continue

            # Include Conv2d and Linear layers
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                modules_to_prune.append((module, "weight"))
                self.pruned_modules.append((module, "weight"))

        self.logger.info(f"Found {len(modules_to_prune)} prunable modules")
        return modules_to_prune

    def _apply_global_pruning(self, modules_to_prune: List[Tuple[nn.Module, str]]):
        """Apply global magnitude-based pruning"""
        try:
            if self.config.structured:
                # Global structured pruning
                prune.global_unstructured(
                    modules_to_prune,
                    pruning_method=prune.L1Unstructured,
                    amount=self.config.sparsity_ratio,
                )
            else:
                # Global unstructured pruning
                prune.global_unstructured(
                    modules_to_prune,
                    pruning_method=prune.L1Unstructured,
                    amount=self.config.sparsity_ratio,
                )

        except Exception as e:
            self.logger.error(f"Error in global pruning: {e}")
            raise

    def _apply_layerwise_pruning(self, modules_to_prune: List[Tuple[nn.Module, str]]):
        """Apply layer-wise magnitude-based pruning"""
        try:
            for module, parameter_name in modules_to_prune:
                if self.config.structured:
                    # Structured pruning (remove entire channels/filters)
                    if isinstance(module, nn.Conv2d):
                        prune.ln_structured(
                            module,
                            name=parameter_name,
                            amount=self.config.sparsity_ratio,
                            n=1,  # L1 norm
                            dim=0,  # Prune output channels
                        )
                    else:
                        # For Linear layers, use unstructured pruning
                        prune.l1_unstructured(
                            module, name=parameter_name, amount=self.config.sparsity_ratio
                        )
                else:
                    # Unstructured pruning
                    prune.l1_unstructured(
                        module, name=parameter_name, amount=self.config.sparsity_ratio
                    )

        except Exception as e:
            self.logger.error(f"Error in layer-wise pruning: {e}")
            raise

    def _create_sparsity_schedule(self) -> List[float]:
        """Create sparsity schedule for gradual pruning"""
        # Polynomial decay schedule
        sparsities = []
        for step in range(self.config.pruning_steps):
            progress = (step + 1) / self.config.pruning_steps
            sparsity = self.config.sparsity_ratio * (progress**3)
            sparsities.append(sparsity)

        return sparsities

    def _apply_pruning_step(self, model: nn.Module, target_sparsity: float):
        """Apply single pruning step"""
        try:
            # Remove existing pruning masks
            for module, parameter_name in self.pruned_modules:
                if hasattr(module, f"{parameter_name}_mask"):
                    prune.remove(module, parameter_name)

            # Apply new pruning with target sparsity
            modules_to_prune = [(module, param) for module, param in self.pruned_modules]

            if self.config.global_pruning:
                prune.global_unstructured(
                    modules_to_prune, pruning_method=prune.L1Unstructured, amount=target_sparsity
                )
            else:
                for module, parameter_name in modules_to_prune:
                    prune.l1_unstructured(module, name=parameter_name, amount=target_sparsity)

        except Exception as e:
            self.logger.error(f"Error applying pruning step: {e}")
            raise

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

                    # Limit fine-tuning batches for efficiency
                    if batch_idx >= 50:  # Process max 50 batches per epoch
                        break

                avg_loss = total_loss / num_batches
                self.logger.debug(f"Fine-tuning epoch {epoch+1}, loss: {avg_loss:.4f}")

        except Exception as e:
            self.logger.error(f"Error during fine-tuning: {e}")
            raise

    def _make_pruning_permanent(self, model: nn.Module):
        """Make pruning permanent by removing masks"""
        try:
            for module, parameter_name in self.pruned_modules:
                if hasattr(module, f"{parameter_name}_mask"):
                    prune.remove(module, parameter_name)

        except Exception as e:
            self.logger.error(f"Error making pruning permanent: {e}")
            raise

    def _calculate_sparsity(self, model: nn.Module) -> float:
        """Calculate overall sparsity of the model"""
        try:
            total_params = 0
            zero_params = 0

            for module in model.modules():
                if isinstance(module, (nn.Conv2d, nn.Linear)):
                    weight = module.weight.data
                    total_params += weight.numel()
                    zero_params += (weight == 0).sum().item()

            sparsity = zero_params / total_params if total_params > 0 else 0.0
            return sparsity

        except Exception as e:
            self.logger.error(f"Error calculating sparsity: {e}")
            return 0.0

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
            total_loss = 0.0

            with torch.no_grad():
                for batch_idx, (data, target) in enumerate(dataloader):
                    data, target = data.to(device), target.to(device)

                    output = model(data)
                    loss = criterion(output, target)

                    pred = output.argmax(dim=1, keepdim=True)
                    correct += pred.eq(target.view_as(pred)).sum().item()
                    total += target.size(0)
                    total_loss += loss.item()

                    # Limit evaluation for efficiency
                    if batch_idx >= 100:  # Process max 100 batches
                        break

            accuracy = correct / total if total > 0 else 0.0
            return accuracy

        except Exception as e:
            self.logger.error(f"Error evaluating model: {e}")
            return 0.0

    def save_pruned_model(self, model: nn.Module, save_path: str, result: PruningResult):
        """Save pruned model and metadata"""
        try:
            save_path = Path(save_path)
            save_path.parent.mkdir(parents=True, exist_ok=True)

            # Save model state dict
            torch.save(model.state_dict(), save_path)

            # Save pruning metadata
            metadata = {
                "pruning_config": {
                    "sparsity_ratio": self.config.sparsity_ratio,
                    "structured": self.config.structured,
                    "global_pruning": self.config.global_pruning,
                    "gradual_pruning": self.config.gradual_pruning,
                    "pruning_steps": self.config.pruning_steps,
                },
                "results": result.to_dict(),
            }

            metadata_path = save_path.with_suffix(".json")
            with open(metadata_path, "w") as f:
                json.dump(metadata, f, indent=2)

            self.logger.info(f"Saved pruned model to {save_path}")

        except Exception as e:
            self.logger.error(f"Error saving pruned model: {e}")
            raise

    def load_pruned_model(self, model: nn.Module, load_path: str) -> Dict[str, Any]:
        """Load pruned model and metadata"""
        try:
            load_path = Path(load_path)

            # Load model state dict
            state_dict = torch.load(load_path, map_location="cpu")
            model.load_state_dict(state_dict)

            # Load metadata
            metadata_path = load_path.with_suffix(".json")
            if metadata_path.exists():
                with open(metadata_path, "r") as f:
                    metadata = json.load(f)
            else:
                metadata = {}

            self.logger.info(f"Loaded pruned model from {load_path}")
            return metadata

        except Exception as e:
            self.logger.error(f"Error loading pruned model: {e}")
            raise

    def analyze_pruning_sensitivity(
        self,
        model: nn.Module,
        dataloader: torch.utils.data.DataLoader,
        device: torch.device,
        criterion: nn.Module,
        sparsity_levels: List[float] = None,
    ) -> Dict[str, List[float]]:
        """Analyze sensitivity to different sparsity levels"""
        try:
            if sparsity_levels is None:
                sparsity_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

            original_accuracy = self._evaluate_model(model, dataloader, device, criterion)

            results = {"sparsity_levels": sparsity_levels, "accuracies": [], "accuracy_drops": []}

            for sparsity in sparsity_levels:
                # Create temporary model copy
                temp_model = copy.deepcopy(model)

                # Apply pruning
                temp_config = copy.deepcopy(self.config)
                temp_config.sparsity_ratio = sparsity
                temp_config.gradual_pruning = False

                temp_pruner = MagnitudePruner(temp_config)
                temp_pruner._one_shot_pruning(temp_model)

                # Evaluate
                accuracy = self._evaluate_model(temp_model, dataloader, device, criterion)
                accuracy_drop = original_accuracy - accuracy

                results["accuracies"].append(accuracy)
                results["accuracy_drops"].append(accuracy_drop)

                self.logger.info(
                    f"Sparsity {sparsity:.1f}: Accuracy {accuracy:.4f}, "
                    f"Drop {accuracy_drop:.4f}"
                )

            return results

        except Exception as e:
            self.logger.error(f"Error in sensitivity analysis: {e}")
            raise


def create_pruning_config(
    sparsity_ratio: float = 0.5,
    structured: bool = False,
    global_pruning: bool = True,
    gradual: bool = True,
) -> PruningConfig:
    """Create pruning configuration with common settings"""
    return PruningConfig(
        sparsity_ratio=sparsity_ratio,
        structured=structured,
        global_pruning=global_pruning,
        gradual_pruning=gradual,
        pruning_steps=10 if gradual else 1,
        recovery_epochs=5 if gradual else 0,
    )


# Example usage functions
def prune_pathology_model(
    model: nn.Module, val_loader: torch.utils.data.DataLoader, sparsity: float = 0.5
) -> PruningResult:
    """Prune pathology model with medical AI optimized settings"""
    config = PruningConfig(
        sparsity_ratio=sparsity,
        structured=False,  # Unstructured for better accuracy preservation
        global_pruning=True,  # Global for optimal sparsity distribution
        gradual_pruning=True,  # Gradual for better recovery
        pruning_steps=15,  # More steps for medical accuracy requirements
        recovery_epochs=10,  # More recovery for critical applications
        exclude_layers=["BatchNorm2d", "LayerNorm", "Dropout"],
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    pruner = MagnitudePruner(config)
    return pruner.prune_model(model, val_loader, device, criterion, optimizer)
